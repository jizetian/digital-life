"""Batch L1 distillation with a task checkpoint per segment.

``task_id`` is ``sha256(f"{segment_id}|{prompt_version}|{model_version}")``.
Completed and skipped tasks are not sent to the model again. Pending,
retrying, and interrupted running tasks are resumed on the next invocation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

from common.clock import utc_now
from common.config import distillation_config
from common.database import init_db
from common.logger import get_logger
from common.statuses import COMPLETED, FAILED, PENDING, RETRYING, RUNNING, SKIPPED
from distillation.llm import LLMClient, build_llm_client
from distillation.parser import Observation, parse_observations
from distillation.prompting import PROMPT_VERSION, load_system_prompt, render_user_prompt
from segmentation.service import read_checkpoint

log = get_logger(__name__)


class DistillationError(Exception):
    """The distillation run cannot start or a segment failed closed."""


def make_task_id(segment_id: str, prompt_version: str, model_version: str) -> str:
    raw = f"{segment_id}|{prompt_version}|{model_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def run_distillation(
    db_path: str | Path,
    *,
    batch_size: int | None = None,
    client: LLMClient | None = None,
    prompt_version: str | None = None,
    model: str | None = None,
    model_version: str | None = None,
    base_url: str | None = None,
    max_tasks: int | None = None,
    max_attempts: int | None = None,
    retry_failed: bool = False,
) -> dict:
    """Distill active segments in batches.

    ``max_tasks`` stops after that many attempts and leaves the rest pending
    so a later run can resume. Each attempt commits before the next one.
    The returned status is ``completed`` when nothing claimable remains, and
    ``paused`` when this process stopped with tasks left.
    """
    database = Path(db_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"database not found: {database}")
    if max_tasks is not None and max_tasks < 1:
        raise ValueError("max_tasks must be positive")
    settings = _settings(
        model=model,
        model_version=model_version,
        prompt_version=prompt_version,
        batch_size=batch_size,
        max_attempts=max_attempts,
        base_url=base_url,
        client=client,
    )
    conn = init_db(database)
    try:
        checkpoint = read_checkpoint(conn)
        if checkpoint is None or not checkpoint.get("params_hash"):
            raise DistillationError("no segmentation checkpoint; run segmentation first")
        params_hash = str(checkpoint["params_hash"])
        _sync_tasks(
            conn,
            params_hash=params_hash,
            prompt_version=settings["prompt_version"],
            model_version=settings["model_version"],
            retry_failed=retry_failed,
        )
        conn.commit()
        if client is None:
            client = build_llm_client(
                base_url=settings["base_url"],
                model=settings["model"],
                timeout=settings["timeout_seconds"],
                temperature=settings["temperature"],
            )
        # Claim one task at a time under BEGIN IMMEDIATE so parallel workers
        # do not process the same segment. Fresh RUNNING rows are left alone;
        # only stale RUNNING (crash leftovers) are reclaimable.
        attempts_completed = 0
        attempts_failed = 0
        seen = 0
        while max_tasks is None or seen < max_tasks:
            task_id = _claim_one(
                conn,
                params_hash=params_hash,
                prompt_version=settings["prompt_version"],
                model_version=settings["model_version"],
                max_attempts=settings["max_attempts"],
            )
            if task_id is None:
                break
            outcome = _execute_task(
                conn,
                task_id,
                client=client,
                max_attempts=settings["max_attempts"],
                max_chars=settings["max_chars_per_message"],
                already_claimed=True,
            )
            seen += 1
            if outcome == COMPLETED:
                attempts_completed += 1
            elif outcome == FAILED or outcome == RETRYING:
                attempts_failed += 1
            if seen % settings["batch_size"] == 0:
                log.info(
                    "distillation batch done size=%s completed=%s failed=%s",
                    settings["batch_size"],
                    attempts_completed,
                    attempts_failed,
                )
        tasks = _status_counts(conn, settings["prompt_version"], settings["model_version"])
        remaining = _remaining(
            conn,
            params_hash=params_hash,
            prompt_version=settings["prompt_version"],
            model_version=settings["model_version"],
            max_attempts=settings["max_attempts"],
        )
        observations = int(conn.execute("SELECT COUNT(*) AS n FROM l1_observations").fetchone()["n"])
        # ``paused`` means this process stopped and left claimable tasks.
        # It is not the per-task ``running`` status, which is only used while a call is in flight.
        status = "paused" if remaining else COMPLETED
        return {
            "status": status,
            "prompt_version": settings["prompt_version"],
            "model_version": settings["model_version"],
            "model": settings["model"],
            "batch_size": settings["batch_size"],
            "attempts_completed": attempts_completed,
            "attempts_failed": attempts_failed,
            "tasks_remaining": remaining,
            "tasks": tasks,
            "observations": observations,
        }
    finally:
        conn.close()


def _settings(
    *,
    model: str | None,
    model_version: str | None,
    prompt_version: str | None,
    batch_size: int | None,
    max_attempts: int | None,
    base_url: str | None,
    client: LLMClient | None,
) -> dict:
    cfg = distillation_config()
    chosen_model = str(cfg.get("model") or "") if model is None else str(model)
    version = str(cfg.get("model_version") or "") if model_version is None else str(model_version)
    if not version:
        version = chosen_model
    if prompt_version is not None:
        prompt = prompt_version
    else:
        prompt = os.environ.get("CHATTWIN_PROMPT_VERSION") or PROMPT_VERSION
    size = int(cfg.get("batch_size", 8) if batch_size is None else batch_size)
    attempts = int(cfg.get("max_attempts", 3) if max_attempts is None else max_attempts)
    if size < 1:
        raise ValueError("batch_size must be positive")
    if attempts < 1:
        raise ValueError("max_attempts must be positive")
    if not version:
        raise DistillationError("model_version is empty")
    if client is None and not chosen_model:
        raise DistillationError("LLM model is empty")
    return {
        "model": chosen_model,
        "model_version": version,
        "prompt_version": prompt,
        "batch_size": size,
        "max_attempts": attempts,
        "base_url": None if base_url is None else base_url,
        "max_chars_per_message": int(cfg.get("max_chars_per_message", 4000)),
        "timeout_seconds": float(cfg.get("timeout_seconds", 120)),
        "temperature": float(cfg.get("temperature", 0)),
    }


def _sync_tasks(
    conn: sqlite3.Connection,
    *,
    params_hash: str,
    prompt_version: str,
    model_version: str,
    retry_failed: bool,
) -> None:
    rows = conn.execute(
        """
        SELECT s.segment_id,
               COALESCE(SUM(
                   CASE
                       WHEN m.content IS NOT NULL AND TRIM(m.content) != '' THEN 1
                       ELSE 0
                   END
               ), 0) AS nonempty
        FROM conversation_segments s
        LEFT JOIN conversation_segment_messages sm ON sm.segment_id = s.segment_id
        LEFT JOIN messages m ON m.message_id = sm.message_id
        WHERE s.params_hash = ?
        GROUP BY s.segment_id
        """,
        (params_hash,),
    ).fetchall()
    if not rows:
        raise DistillationError("no conversation segments; run segmentation first")
    now = utc_now()
    for row in rows:
        status = SKIPPED if int(row["nonempty"]) == 0 else PENDING
        task_id = make_task_id(row["segment_id"], prompt_version, model_version)
        conn.execute(
            """
            INSERT INTO distillation_tasks (
                task_id, segment_id, prompt_version, model_version, status,
                attempt_count, observations_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 0, 0, ?, ?)
            ON CONFLICT(task_id) DO NOTHING
            """,
            (task_id, row["segment_id"], prompt_version, model_version, status, now, now),
        )
        if status == SKIPPED:
            conn.execute(
                """
                UPDATE conversation_segments
                SET processing_status = ?, updated_at = ?
                WHERE segment_id = ? AND processing_status = ?
                """,
                (SKIPPED, now, row["segment_id"], PENDING),
            )
            conn.execute(
                """
                UPDATE distillation_tasks
                SET status = ?, updated_at = ?, finished_at = COALESCE(finished_at, ?)
                WHERE task_id = ? AND status = ?
                """,
                (SKIPPED, now, now, task_id, PENDING),
            )
    if retry_failed:
        conn.execute(
            """
            UPDATE distillation_tasks
            SET status = ?, attempt_count = 0, error_message = NULL,
                finished_at = NULL, updated_at = ?
            WHERE status = ?
              AND prompt_version = ?
              AND model_version = ?
              AND segment_id IN (
                  SELECT segment_id FROM conversation_segments WHERE params_hash = ?
              )
            """,
            (RETRYING, now, FAILED, prompt_version, model_version, params_hash),
        )
        conn.execute(
            """
            UPDATE conversation_segments
            SET processing_status = ?, updated_at = ?
            WHERE params_hash = ?
              AND segment_id IN (
                  SELECT segment_id FROM distillation_tasks
                  WHERE status = ? AND prompt_version = ? AND model_version = ?
              )
            """,
            (RETRYING, now, params_hash, RETRYING, prompt_version, model_version),
        )


def _running_stale_before() -> str:
    """RUNNING rows newer than this ISO timestamp are owned by a live worker."""
    from datetime import datetime, timedelta, timezone

    raw = os.environ.get("CHATTWIN_RUNNING_STALE_SECONDS", "120")
    try:
        seconds = max(0, int(raw))
    except ValueError:
        seconds = 120
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def _claimable(
    conn: sqlite3.Connection,
    *,
    params_hash: str,
    prompt_version: str,
    model_version: str,
    max_attempts: int,
) -> list[sqlite3.Row]:
    stale_before = _running_stale_before()
    return conn.execute(
        """
        SELECT t.task_id
        FROM distillation_tasks t
        JOIN conversation_segments s ON s.segment_id = t.segment_id
        WHERE s.params_hash = ?
          AND t.prompt_version = ?
          AND t.model_version = ?
          AND (
                (t.status IN (?, ?) AND t.attempt_count < ?)
                OR (t.status = ? AND COALESCE(t.updated_at, '') < ?)
              )
        ORDER BY s.start_time IS NULL, s.start_time, s.segment_id
        """,
        (
            params_hash,
            prompt_version,
            model_version,
            PENDING,
            RETRYING,
            max_attempts,
            RUNNING,
            stale_before,
        ),
    ).fetchall()


def _claim_one(
    conn: sqlite3.Connection,
    *,
    params_hash: str,
    prompt_version: str,
    model_version: str,
    max_attempts: int,
) -> str | None:
    """Atomically claim the next task. Safe for a couple of parallel workers."""
    stale_before = _running_stale_before()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            """
            SELECT t.task_id, t.segment_id, t.status, t.attempt_count
            FROM distillation_tasks t
            JOIN conversation_segments s ON s.segment_id = t.segment_id
            WHERE s.params_hash = ?
              AND t.prompt_version = ?
              AND t.model_version = ?
              AND (
                    (t.status IN (?, ?) AND t.attempt_count < ?)
                    OR (t.status = ? AND COALESCE(t.updated_at, '') < ?)
                  )
            ORDER BY s.start_time IS NULL, s.start_time, s.segment_id
            LIMIT 1
            """,
            (
                params_hash,
                prompt_version,
                model_version,
                PENDING,
                RETRYING,
                max_attempts,
                RUNNING,
                stale_before,
            ),
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        task_id = row["task_id"]
        segment_id = row["segment_id"]
        attempt = int(row["attempt_count"]) + 1
        now = utc_now()
        cur = conn.execute(
            """
            UPDATE distillation_tasks
            SET status = ?, attempt_count = ?, started_at = COALESCE(started_at, ?),
                updated_at = ?, error_message = NULL
            WHERE task_id = ? AND status = ?
            """,
            (RUNNING, attempt, now, now, task_id, row["status"]),
        )
        if cur.rowcount != 1:
            conn.commit()
            return None
        conn.execute(
            "UPDATE conversation_segments SET processing_status = ?, updated_at = ? WHERE segment_id = ?",
            (RUNNING, now, segment_id),
        )
        conn.commit()
        return task_id
    except Exception:
        conn.rollback()
        raise


def _remaining(conn: sqlite3.Connection, **kwargs) -> int:
    return len(_claimable(conn, **kwargs))


def _execute_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    client: LLMClient,
    max_attempts: int,
    max_chars: int,
    already_claimed: bool = False,
) -> str:
    row = conn.execute("SELECT * FROM distillation_tasks WHERE task_id = ?", (task_id,)).fetchone()
    if row is None or row["status"] in {COMPLETED, SKIPPED, FAILED}:
        return SKIPPED
    segment_id = row["segment_id"]
    if already_claimed:
        # Claim path already moved PENDING/RETRYING/stale-RUNNING -> RUNNING
        # and incremented attempt_count under BEGIN IMMEDIATE.
        attempt = int(row["attempt_count"])
        if row["status"] != RUNNING or attempt < 1:
            return SKIPPED
    else:
        attempt = int(row["attempt_count"]) + 1
        now = utc_now()
        cur = conn.execute(
            """
            UPDATE distillation_tasks
            SET status = ?, attempt_count = ?, started_at = COALESCE(started_at, ?),
                updated_at = ?, error_message = NULL
            WHERE task_id = ? AND status IN (?, ?, ?)
            """,
            (RUNNING, attempt, now, now, task_id, PENDING, RETRYING, RUNNING),
        )
        if cur.rowcount != 1:
            conn.rollback()
            return SKIPPED
        conn.execute(
            "UPDATE conversation_segments SET processing_status = ?, updated_at = ? WHERE segment_id = ?",
            (RUNNING, now, segment_id),
        )
        conn.commit()
    try:
        segment = conn.execute(
            "SELECT * FROM conversation_segments WHERE segment_id = ?",
            (segment_id,),
        ).fetchone()
        if segment is None:
            raise DistillationError(f"segment {segment_id} disappeared")
        messages = _load_messages(conn, segment_id)
        expected = json.loads(segment["message_ids_json"])
        got = [item["message_id"] for item in messages]
        if got != expected:
            raise DistillationError("segment message references do not match stored message ids")
        raw = client.complete(
            system=load_system_prompt(),
            user=render_user_prompt(
                segment_id=segment_id,
                conversation_id=segment["conversation_id"],
                messages=messages,
                max_chars=max_chars,
            ),
        )
        if not isinstance(raw, str):
            raise DistillationError("LLM client returned a non-text response")
        observations, dropped = parse_observations(raw, allowed_message_ids=set(expected))
        _fill_time_ranges(observations, messages)
        _replace_observations(conn, task_id, segment_id, observations)
        finished = utc_now()
        conn.execute(
            """
            UPDATE distillation_tasks
            SET status = ?, observations_count = ?, finished_at = ?, updated_at = ?, error_message = NULL
            WHERE task_id = ?
            """,
            (COMPLETED, len(observations), finished, finished, task_id),
        )
        conn.execute(
            "UPDATE conversation_segments SET processing_status = ?, updated_at = ? WHERE segment_id = ?",
            (COMPLETED, finished, segment_id),
        )
        conn.commit()
        if dropped:
            log.info("task %s completed observations=%s dropped=%s", task_id, len(observations), dropped)
        return COMPLETED
    except Exception as exc:
        _mark_attempt_failed(conn, task_id, segment_id, attempt, max_attempts, exc)
        return RETRYING if attempt < max_attempts else FAILED


def _mark_attempt_failed(
    conn: sqlite3.Connection,
    task_id: str,
    segment_id: str,
    attempt: int,
    max_attempts: int,
    exc: BaseException,
) -> None:
    conn.rollback()
    status = RETRYING if attempt < max_attempts else FAILED
    now = utc_now()
    message = f"{type(exc).__name__}: {exc}"
    message = " ".join(message.split())[:500]
    conn.execute(
        """
        UPDATE distillation_tasks
        SET status = ?, error_message = ?, finished_at = ?, updated_at = ?
        WHERE task_id = ?
        """,
        (status, message, now, now, task_id),
    )
    conn.execute(
        "UPDATE conversation_segments SET processing_status = ?, updated_at = ? WHERE segment_id = ?",
        (status, now, segment_id),
    )
    conn.commit()
    log.info("task %s %s attempt=%s", task_id, status, attempt)


def _load_messages(conn: sqlite3.Connection, segment_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT sm.message_id, sm.position, sm.is_overlap,
               m.sender_role, m.timestamp, m.message_type, m.content, m.conversation_id
        FROM conversation_segment_messages sm
        JOIN messages m ON m.message_id = sm.message_id
        WHERE sm.segment_id = ?
        ORDER BY sm.position
        """,
        (segment_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _fill_time_ranges(observations: list[Observation], messages: list[dict]) -> None:
    stamps = {
        item["message_id"]: item["timestamp"]
        for item in messages
        if item["timestamp"] is not None
    }
    for observation in observations:
        if observation.time_range_start is not None and observation.time_range_end is not None:
            continue
        cited = [int(stamps[mid]) for mid in observation.evidence_message_ids if mid in stamps]
        if not cited:
            continue
        if observation.time_range_start is None:
            observation.time_range_start = min(cited)
        if observation.time_range_end is None:
            observation.time_range_end = max(cited)


def _replace_observations(
    conn: sqlite3.Connection,
    task_id: str,
    segment_id: str,
    observations: list[Observation],
) -> None:
    conn.execute("DELETE FROM l1_observations WHERE task_id = ?", (task_id,))
    now = utc_now()
    for index, observation in enumerate(observations):
        payload = {
            "category": observation.category,
            "summary": observation.summary,
            "evidence_message_ids": observation.evidence_message_ids,
            "confidence": observation.confidence,
            "time_range": {
                "start": observation.time_range_start,
                "end": observation.time_range_end,
            },
        }
        observation_id = hashlib.sha256(
            f"{task_id}|{index}|{observation.category}|{observation.summary}".encode("utf-8")
        ).hexdigest()
        conn.execute(
            """
            INSERT INTO l1_observations (
                observation_id, task_id, segment_id, category, summary,
                evidence_message_ids_json, confidence, time_range_start, time_range_end,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation_id,
                task_id,
                segment_id,
                observation.category,
                observation.summary,
                json.dumps(observation.evidence_message_ids, ensure_ascii=False),
                observation.confidence,
                observation.time_range_start,
                observation.time_range_end,
                json.dumps(payload, ensure_ascii=False),
                now,
            ),
        )


def _status_counts(conn: sqlite3.Connection, prompt_version: str, model_version: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS n
        FROM distillation_tasks
        WHERE prompt_version = ? AND model_version = ?
        GROUP BY status
        """,
        (prompt_version, model_version),
    ).fetchall()
    return {row["status"]: int(row["n"]) for row in rows}
