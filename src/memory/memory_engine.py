"""Extract long-term memories from segments and L1 observations, then merge them.

``task_id`` is ``sha256(f"{segment_id}|{kind}|{prompt_version}|{model_version}")``.
``kind`` is ``extract`` or ``merge``. Completed and skipped tasks are not sent
to the model again. Pending, retrying, and interrupted running tasks are
resumed on the next invocation. Each task is attempted at most once per run.

The merge step never deletes a memory. An older row stays, with status
``historical``, ``superseded``, or ``uncertain``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from common.clock import utc_now
from common.config import distillation_config, memory_config
from common.database import init_db
from common.logger import get_logger
from common.statuses import COMPLETED, FAILED, PENDING, PIPELINE_STATUSES, RETRYING, RUNNING, SKIPPED
from distillation.llm import LLMClient, build_llm_client
from memory.memory_merger import apply_merge, resolve_actions
from memory.models import CandidateMemory, memory_identity, normalize_text, time_precision_for
from memory.parser import parse_candidates, parse_merge_actions
from memory.prompting import (
    PROMPT_VERSION,
    load_extract_prompt,
    load_merge_prompt,
    render_extract_prompt,
    render_merge_prompt,
)
from memory.status import memory_counts_for
from segmentation.service import read_checkpoint

log = get_logger(__name__)


class MemoryEngineError(Exception):
    """The memory run cannot start, or one segment failed closed."""


def make_task_id(segment_id: str, kind: str, prompt_version: str, model_version: str) -> str:
    raw = f"{segment_id}|{kind}|{prompt_version}|{model_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def run_memory(
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
    """Build memories for the active segmentation.

    ``max_tasks`` stops after that many attempts and leaves the rest pending,
    retrying, or still unmerged so a later run can resume. The returned
    status is ``paused`` when claimable tasks remain, ``failed`` when every
    claimable task has been tried and at least one is failed, and
    ``completed`` otherwise.
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
            raise MemoryEngineError("no segmentation checkpoint; run segmentation first")
        params_hash = str(checkpoint["params_hash"])
        _sync_extracts(
            conn,
            params_hash=params_hash,
            prompt_version=settings["prompt_version"],
            model_version=settings["model_version"],
        )
        if retry_failed:
            _requeue_failed(conn, settings["prompt_version"], settings["model_version"])
        _sync_merges(conn, params_hash, settings["prompt_version"], settings["model_version"])
        _write_checkpoint(conn, settings, RUNNING)
        conn.commit()
        if client is None:
            client = build_llm_client(
                base_url=settings["base_url"],
                model=settings["model"],
                timeout=settings["timeout_seconds"],
                temperature=settings["temperature"],
            )
        attempts_completed = 0
        attempts_failed = 0
        seen = 0
        touched: set[str] = set()
        while max_tasks is None or seen < max_tasks:
            task = _claim_one(
                conn,
                params_hash=params_hash,
                prompt_version=settings["prompt_version"],
                model_version=settings["model_version"],
                max_attempts=settings["max_attempts"],
                exclude=touched,
            )
            if task is None:
                _sync_merges(conn, params_hash, settings["prompt_version"], settings["model_version"])
                conn.commit()
                task = _claim_one(
                    conn,
                    params_hash=params_hash,
                    prompt_version=settings["prompt_version"],
                    model_version=settings["model_version"],
                    max_attempts=settings["max_attempts"],
                    exclude=touched,
                )
                if task is None:
                    break
            touched.add(task["task_id"])
            outcome = _execute_task(
                conn,
                task["task_id"],
                client=client,
                max_attempts=settings["max_attempts"],
                max_chars=settings["max_chars_per_message"],
                max_existing=settings["max_existing"],
            )
            seen += 1
            if outcome == COMPLETED:
                attempts_completed += 1
            elif outcome in {FAILED, RETRYING}:
                attempts_failed += 1
            if seen % settings["batch_size"] == 0:
                log.info(
                    "memory batch done size=%s completed=%s failed=%s",
                    settings["batch_size"],
                    attempts_completed,
                    attempts_failed,
                )
        _sync_merges(conn, params_hash, settings["prompt_version"], settings["model_version"])
        conn.commit()
        tasks = _status_counts(conn, settings["prompt_version"], settings["model_version"])
        remaining = _remaining(
            conn,
            params_hash=params_hash,
            prompt_version=settings["prompt_version"],
            model_version=settings["model_version"],
            max_attempts=settings["max_attempts"],
        )
        if remaining:
            status = "paused"
        elif tasks.get(FAILED, 0):
            status = FAILED
        else:
            status = COMPLETED
        _write_checkpoint(conn, settings, status)
        conn.commit()
        memories = memory_counts_for(
            conn,
            prompt_version=settings["prompt_version"],
            model_version=settings["model_version"],
        )
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
            "memories": memories,
            "entities": int(conn.execute("SELECT COUNT(*) AS n FROM entities").fetchone()["n"]),
            "relations": int(conn.execute("SELECT COUNT(*) AS n FROM memory_relations").fetchone()["n"]),
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
    distilled = distillation_config()
    cfg = memory_config()
    chosen_model = str(distilled.get("model") or "") if model is None else str(model)
    version = str(distilled.get("model_version") or "") if model_version is None else str(model_version)
    if not version:
        version = chosen_model
    if prompt_version is not None:
        prompt = prompt_version
    else:
        prompt = str(cfg.get("prompt_version") or PROMPT_VERSION)
    size = int(cfg.get("batch_size", 4) if batch_size is None else batch_size)
    attempts = int(cfg.get("max_attempts", 3) if max_attempts is None else max_attempts)
    max_existing = int(cfg.get("max_existing", 40))
    if size < 1:
        raise ValueError("batch_size must be positive")
    if attempts < 1:
        raise ValueError("max_attempts must be positive")
    if max_existing < 1:
        raise ValueError("max_existing must be positive")
    if not version:
        raise MemoryEngineError("model_version is empty")
    if client is None and not chosen_model:
        raise MemoryEngineError("LLM model is empty")
    return {
        "model": chosen_model,
        "model_version": version,
        "prompt_version": prompt,
        "batch_size": size,
        "max_attempts": attempts,
        "max_existing": max_existing,
        "base_url": None if base_url is None else base_url,
        "max_chars_per_message": int(cfg.get("max_chars_per_message", 2000)),
        "timeout_seconds": float(distilled.get("timeout_seconds", 120)),
        "temperature": float(distilled.get("temperature", 0)),
    }


def _sync_extracts(
    conn: sqlite3.Connection,
    *,
    params_hash: str,
    prompt_version: str,
    model_version: str,
) -> None:
    rows = conn.execute(
        """
        SELECT s.segment_id,
               COALESCE(SUM(
                   CASE
                       WHEN m.content IS NOT NULL AND TRIM(m.content) != '' THEN 1
                       ELSE 0
                   END
               ), 0) AS nonempty,
               (
                   SELECT COUNT(*) FROM l1_observations o WHERE o.segment_id = s.segment_id
               ) AS observations
        FROM conversation_segments s
        LEFT JOIN conversation_segment_messages sm ON sm.segment_id = s.segment_id
        LEFT JOIN messages m ON m.message_id = sm.message_id
        WHERE s.params_hash = ?
        GROUP BY s.segment_id
        """,
        (params_hash,),
    ).fetchall()
    if not rows:
        raise MemoryEngineError("no conversation segments; run segmentation first")
    now = utc_now()
    for row in rows:
        should_skip = int(row["nonempty"]) == 0 and int(row["observations"]) == 0
        status = SKIPPED if should_skip else PENDING
        task_id = make_task_id(row["segment_id"], "extract", prompt_version, model_version)
        conn.execute(
            """
            INSERT INTO memory_tasks (
                task_id, segment_id, task_kind, prompt_version, model_version, status,
                attempt_count, result_count, created_at, updated_at, finished_at
            ) VALUES (?, ?, 'extract', ?, ?, ?, 0, 0, ?, ?, ?)
            ON CONFLICT(task_id) DO NOTHING
            """,
            (
                task_id,
                row["segment_id"],
                prompt_version,
                model_version,
                status,
                now,
                now,
                now if should_skip else None,
            ),
        )
        if should_skip:
            conn.execute(
                """
                UPDATE memory_tasks
                SET status = ?, updated_at = ?, finished_at = COALESCE(finished_at, ?)
                WHERE task_id = ? AND status = ?
                """,
                (SKIPPED, now, now, task_id, PENDING),
            )
        else:
            conn.execute(
                """
                UPDATE memory_tasks
                SET status = ?, updated_at = ?, finished_at = NULL
                WHERE task_id = ? AND status = ?
                """,
                (PENDING, now, task_id, SKIPPED),
            )


def _requeue_failed(conn: sqlite3.Connection, prompt_version: str, model_version: str) -> None:
    now = utc_now()
    conn.execute(
        """
        UPDATE memory_tasks
        SET status = ?, attempt_count = 0, error_message = NULL, finished_at = NULL, updated_at = ?
        WHERE status = ? AND prompt_version = ? AND model_version = ?
        """,
        (RETRYING, now, FAILED, prompt_version, model_version),
    )


def _sync_merges(
    conn: sqlite3.Connection,
    params_hash: str,
    prompt_version: str,
    model_version: str,
) -> None:
    rows = conn.execute(
        """
        SELECT t.*
        FROM memory_tasks t
        JOIN conversation_segments s ON s.segment_id = t.segment_id
        WHERE s.params_hash = ?
          AND t.task_kind = 'extract'
          AND t.prompt_version = ?
          AND t.model_version = ?
          AND t.status = ?
        """,
        (params_hash, prompt_version, model_version, COMPLETED),
    ).fetchall()
    now = utc_now()
    for row in rows:
        _sync_one_merge(conn, row, now)


def _sync_one_merge(conn: sqlite3.Connection, extract_row: sqlite3.Row, now: str) -> None:
    result_json = extract_row["result_json"] or ""
    digest = hashlib.sha256(result_json.encode("utf-8")).hexdigest()
    try:
        payload = json.loads(result_json) if result_json else {}
    except json.JSONDecodeError:
        payload = {}
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    has_candidates = isinstance(candidates, list) and len(candidates) > 0
    desired = PENDING if has_candidates else SKIPPED
    task_id = make_task_id(
        extract_row["segment_id"],
        "merge",
        extract_row["prompt_version"],
        extract_row["model_version"],
    )
    existing = conn.execute("SELECT * FROM memory_tasks WHERE task_id = ?", (task_id,)).fetchone()
    if existing is None:
        conn.execute(
            """
            INSERT INTO memory_tasks (
                task_id, segment_id, task_kind, prompt_version, model_version, status,
                attempt_count, input_digest, result_count, created_at, updated_at, finished_at
            ) VALUES (?, ?, 'merge', ?, ?, ?, 0, ?, 0, ?, ?, ?)
            """,
            (
                task_id,
                extract_row["segment_id"],
                extract_row["prompt_version"],
                extract_row["model_version"],
                desired,
                digest,
                now,
                now,
                now if desired == SKIPPED else None,
            ),
        )
        return
    if existing["status"] == RUNNING:
        return
    if existing["input_digest"] == digest and existing["status"] in {COMPLETED, SKIPPED}:
        return
    if existing["input_digest"] == digest and existing["status"] in {PENDING, RETRYING, FAILED}:
        if desired == SKIPPED:
            conn.execute(
                """
                UPDATE memory_tasks
                SET status = ?, input_digest = ?, finished_at = ?, updated_at = ?, error_message = NULL
                WHERE task_id = ?
                """,
                (SKIPPED, digest, now, now, task_id),
            )
        return
    if existing["status"] == COMPLETED and not has_candidates:
        return
    conn.execute(
        """
        UPDATE memory_tasks
        SET status = ?, attempt_count = 0, error_message = NULL, input_digest = ?,
            result_json = NULL, result_count = 0, finished_at = ?, updated_at = ?
        WHERE task_id = ?
        """,
        (desired, digest, now if desired == SKIPPED else None, now, task_id),
    )


def _claim_one(
    conn: sqlite3.Connection,
    *,
    params_hash: str,
    prompt_version: str,
    model_version: str,
    max_attempts: int,
    exclude: set[str],
) -> sqlite3.Row | None:
    rows = conn.execute(
        """
        SELECT t.task_id, t.task_kind
        FROM memory_tasks t
        JOIN conversation_segments s ON s.segment_id = t.segment_id
        WHERE s.params_hash = ?
          AND t.prompt_version = ?
          AND t.model_version = ?
          AND (
                (t.status IN (?, ?) AND t.attempt_count < ?)
                OR t.status = ?
              )
        ORDER BY CASE t.task_kind WHEN 'merge' THEN 0 ELSE 1 END,
                 s.start_time IS NULL, s.start_time, s.segment_id
        """,
        (
            params_hash,
            prompt_version,
            model_version,
            PENDING,
            RETRYING,
            max_attempts,
            RUNNING,
        ),
    ).fetchall()
    for row in rows:
        if row["task_id"] not in exclude:
            return row
    return None


def _remaining(conn: sqlite3.Connection, **kwargs) -> int:
    rows = conn.execute(
        """
        SELECT t.task_id
        FROM memory_tasks t
        JOIN conversation_segments s ON s.segment_id = t.segment_id
        WHERE s.params_hash = ?
          AND t.prompt_version = ?
          AND t.model_version = ?
          AND (
                (t.status IN (?, ?) AND t.attempt_count < ?)
                OR t.status = ?
              )
        """,
        (
            kwargs["params_hash"],
            kwargs["prompt_version"],
            kwargs["model_version"],
            PENDING,
            RETRYING,
            kwargs["max_attempts"],
            RUNNING,
        ),
    ).fetchall()
    count = len(rows)
    return count


def _execute_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    client: LLMClient,
    max_attempts: int,
    max_chars: int,
    max_existing: int,
) -> str:
    row = conn.execute("SELECT * FROM memory_tasks WHERE task_id = ?", (task_id,)).fetchone()
    if row is None or row["status"] in {COMPLETED, SKIPPED, FAILED}:
        return SKIPPED
    attempt = int(row["attempt_count"]) + 1
    now = utc_now()
    conn.execute(
        """
        UPDATE memory_tasks
        SET status = ?, attempt_count = ?, started_at = COALESCE(started_at, ?),
            updated_at = ?, error_message = NULL
        WHERE task_id = ?
        """,
        (RUNNING, attempt, now, now, task_id),
    )
    conn.commit()
    try:
        if row["task_kind"] == "extract":
            outcome_count = _run_extract(conn, row, client=client, max_chars=max_chars)
        elif row["task_kind"] == "merge":
            outcome_count = _run_merge(
                conn,
                row,
                client=client,
                max_chars=max_chars,
                max_existing=max_existing,
            )
        else:
            raise MemoryEngineError(f"unknown memory task kind: {row['task_kind']}")
        finished = utc_now()
        conn.execute(
            """
            UPDATE memory_tasks
            SET status = ?, result_count = ?, finished_at = ?, updated_at = ?, error_message = NULL
            WHERE task_id = ?
            """,
            (COMPLETED, outcome_count, finished, finished, task_id),
        )
        if row["task_kind"] == "extract":
            updated = conn.execute("SELECT * FROM memory_tasks WHERE task_id = ?", (task_id,)).fetchone()
            _sync_one_merge(conn, updated, finished)
        conn.commit()
        log.info("memory task %s %s count=%s", row["task_kind"], COMPLETED, outcome_count)
        return COMPLETED
    except Exception as exc:
        _mark_attempt_failed(conn, task_id, attempt, max_attempts, exc)
        return RETRYING if attempt < max_attempts else FAILED


def _run_extract(conn: sqlite3.Connection, task: sqlite3.Row, *, client: LLMClient, max_chars: int) -> int:
    segment_id = task["segment_id"]
    segment = conn.execute(
        "SELECT * FROM conversation_segments WHERE segment_id = ?",
        (segment_id,),
    ).fetchone()
    if segment is None:
        raise MemoryEngineError("segment disappeared")
    messages = _load_messages(conn, segment_id)
    expected = json.loads(segment["message_ids_json"])
    got = [item["message_id"] for item in messages]
    if got != expected:
        raise MemoryEngineError("segment message references do not match stored message ids")
    observations = _load_observations(conn, segment_id)
    raw = client.complete(
        system=load_extract_prompt(),
        user=render_extract_prompt(
            segment_id=segment_id,
            conversation_id=segment["conversation_id"],
            messages=messages,
            observations=observations,
            max_chars=max_chars,
        ),
    )
    if not isinstance(raw, str):
        raise MemoryEngineError("LLM client returned a non-text response")
    allowed_messages = set(expected)
    allowed_observations = {item["observation_id"] for item in observations}
    candidates, dropped = parse_candidates(
        raw,
        allowed_message_ids=allowed_messages,
        allowed_observation_ids=allowed_observations,
    )
    candidates, unbound = _bind_observation_evidence(candidates, observations, allowed_messages)
    dropped += unbound
    _fill_time_ranges(candidates, messages)
    body = json.dumps(
        {"candidates": [item.to_dict() for item in candidates], "dropped": dropped},
        ensure_ascii=False,
        sort_keys=True,
    )
    conn.execute(
        "UPDATE memory_tasks SET result_json = ? WHERE task_id = ?",
        (body, task["task_id"]),
    )
    if dropped:
        log.info("extract dropped=%s kept=%s", dropped, len(candidates))
    return len(candidates)


def _run_merge(
    conn: sqlite3.Connection,
    task: sqlite3.Row,
    *,
    client: LLMClient,
    max_chars: int,
    max_existing: int,
) -> int:
    extract_id = make_task_id(task["segment_id"], "extract", task["prompt_version"], task["model_version"])
    extract = conn.execute("SELECT result_json, status FROM memory_tasks WHERE task_id = ?", (extract_id,)).fetchone()
    if extract is None or extract["status"] != COMPLETED:
        raise MemoryEngineError("merge requires a completed extract task")
    try:
        payload = json.loads(extract["result_json"] or "{}")
    except json.JSONDecodeError as exc:
        raise MemoryEngineError("extract result was not JSON") from exc
    raw_candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(raw_candidates, list) or not raw_candidates:
        return 0
    candidates = [CandidateMemory.from_dict(item) for item in raw_candidates]
    existing = _existing_for_prompt(conn, candidates, max_existing, task["prompt_version"], task["model_version"])
    allowed = {item["memory_id"] for item in existing}
    raw = client.complete(
        system=load_merge_prompt(),
        user=render_merge_prompt(
            segment_id=task["segment_id"],
            existing=existing,
            candidates=candidates,
            max_chars=max_chars,
        ),
    )
    if not isinstance(raw, str):
        raise MemoryEngineError("LLM client returned a non-text response")
    parsed = parse_merge_actions(raw)
    actions = resolve_actions(candidates, parsed, allowed_target_ids=allowed)
    stats = apply_merge(
        conn,
        segment_id=task["segment_id"],
        candidates=candidates,
        actions=actions,
        prompt_version=task["prompt_version"],
        model_version=task["model_version"],
        allowed_target_ids=allowed,
    )
    summary = stats.to_dict()
    summary["input_digest"] = task["input_digest"]
    conn.execute(
        "UPDATE memory_tasks SET result_json = ?, input_digest = ? WHERE task_id = ?",
        (json.dumps(summary, ensure_ascii=False, sort_keys=True), task["input_digest"], task["task_id"]),
    )
    log.info(
        "merge inserted=%s duplicated=%s superseded=%s historical=%s conflicts=%s dropped=%s",
        stats.inserted,
        stats.duplicated,
        stats.superseded,
        stats.historical,
        stats.conflicts,
        stats.dropped,
    )
    return stats.inserted + stats.duplicated + stats.superseded + stats.historical + stats.conflicts + stats.related


def _mark_attempt_failed(
    conn: sqlite3.Connection,
    task_id: str,
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
        UPDATE memory_tasks
        SET status = ?, error_message = ?, finished_at = ?, updated_at = ?
        WHERE task_id = ?
        """,
        (status, message, now, now, task_id),
    )
    conn.commit()
    log.info("memory task %s %s attempt=%s", task_id, status, attempt)


def _bind_observation_evidence(
    candidates: list[CandidateMemory],
    observations: list[dict],
    allowed_message_ids: set[str],
) -> tuple[list[CandidateMemory], int]:
    by_id = {item["observation_id"]: item for item in observations}
    kept: list[CandidateMemory] = []
    dropped = 0
    for candidate in candidates:
        ids = list(candidate.evidence_message_ids)
        seen = set(ids)
        for observation_id in candidate.observation_ids:
            source = by_id.get(observation_id) or {}
            for message_id in source.get("evidence_message_ids") or []:
                if message_id in allowed_message_ids and message_id not in seen:
                    seen.add(message_id)
                    ids.append(message_id)
        candidate.evidence_message_ids = ids
        if not ids:
            dropped += 1
            continue
        kept.append(candidate)
    return kept, dropped


def _fill_time_ranges(candidates: list[CandidateMemory], messages: list[dict]) -> None:
    stamps = {
        item["message_id"]: int(item["timestamp"])
        for item in messages
        if item.get("timestamp") is not None
    }
    for candidate in candidates:
        if candidate.time_start is None or candidate.time_end is None:
            cited = [stamps[mid] for mid in candidate.evidence_message_ids if mid in stamps]
            if cited:
                if candidate.time_start is None:
                    candidate.time_start = min(cited)
                if candidate.time_end is None:
                    candidate.time_end = max(cited)
        candidate.time_precision = time_precision_for(
            candidate.time_start,
            candidate.time_end,
            candidate.time_precision,
        )


def _existing_for_prompt(
    conn: sqlite3.Connection,
    candidates: list[CandidateMemory],
    limit: int,
    prompt_version: str,
    model_version: str,
) -> list[dict]:
    total = int(
        conn.execute(
            """
            SELECT COUNT(*) AS n FROM memories
            WHERE prompt_version = ? AND model_version = ?
            """,
            (prompt_version, model_version),
        ).fetchone()["n"]
    )
    if total <= limit:
        rows = conn.execute(
            """
            SELECT * FROM memories
            WHERE prompt_version = ? AND model_version = ?
            ORDER BY time_start IS NULL, time_start, memory_id
            """,
            (prompt_version, model_version),
        ).fetchall()
        return [_compact_memory(conn, row) for row in rows]

    ordered: list[str] = []
    seen: set[str] = set()

    def add(memory_id: str) -> None:
        if memory_id not in seen:
            seen.add(memory_id)
            ordered.append(memory_id)

    for candidate in candidates:
        identity = memory_identity(prompt_version, model_version, candidate.memory_type, candidate.statement)
        found = conn.execute("SELECT memory_id FROM memories WHERE memory_id = ?", (identity,)).fetchone()
        if found is not None:
            add(found["memory_id"])
    keys = [(mention.entity_type, normalize_text(mention.name)) for candidate in candidates for mention in candidate.entities]
    for offset in range(0, len(keys), 80):
        chunk = keys[offset : offset + 80]
        placeholders = ",".join("(?, ?)" for _ in chunk)
        flat = [part for pair in chunk for part in pair]
        rows = conn.execute(
            f"""
            SELECT DISTINCT m.memory_id
            FROM memories m
            JOIN memory_entities me ON me.memory_id = m.memory_id
            JOIN entities e ON e.entity_id = me.entity_id
            WHERE m.prompt_version = ? AND m.model_version = ?
              AND (e.entity_type, e.name_key) IN ({placeholders})
            ORDER BY m.time_start IS NULL, m.time_start, m.memory_id
            """,
            [prompt_version, model_version, *flat],
        ).fetchall()
        for row in rows:
            if len(ordered) >= limit:
                break
            add(row["memory_id"])
    if len(ordered) < limit:
        types = sorted({candidate.memory_type for candidate in candidates})
        if types:
            placeholders = ",".join("?" for _ in types)
            rows = conn.execute(
                f"""
                SELECT memory_id FROM memories
                WHERE prompt_version = ? AND model_version = ? AND memory_type IN ({placeholders})
                ORDER BY time_start IS NULL, time_start, memory_id
                LIMIT ?
                """,
                [prompt_version, model_version, *types, limit],
            ).fetchall()
            for row in rows:
                if len(ordered) >= limit:
                    break
                add(row["memory_id"])
    compacted = []
    for memory_id in ordered[:limit]:
        row = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
        if row is not None:
            compacted.append(_compact_memory(conn, row))
    return compacted


def _compact_memory(conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
    entities = conn.execute(
        """
        SELECT e.canonical_name, e.entity_type, e.status, me.role
        FROM memory_entities me
        JOIN entities e ON e.entity_id = me.entity_id
        WHERE me.memory_id = ?
        ORDER BY me.role, e.canonical_name
        """,
        (row["memory_id"],),
    ).fetchall()
    rendered = "|".join(
        f"{_prompt_name(item['canonical_name'])}:{item['role']}:{item['status']}:{item['entity_type']}"
        for item in entities
    )
    return {
        "memory_id": row["memory_id"],
        "memory_type": row["memory_type"],
        "status": row["status"],
        "statement": row["statement"],
        "situation": row["situation"],
        "importance": row["importance"],
        "confidence": row["confidence"],
        "time_start": row["time_start"],
        "time_end": row["time_end"],
        "entities": rendered,
    }


def _prompt_name(name: str) -> str:
    return " ".join(str(name).split()).replace("|", "/").replace(":", "/")


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


def _load_observations(conn: sqlite3.Connection, segment_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT observation_id, category, summary, evidence_message_ids_json,
               confidence, time_range_start, time_range_end
        FROM l1_observations
        WHERE segment_id = ?
        ORDER BY observation_id
        """,
        (segment_id,),
    ).fetchall()
    items = []
    for row in rows:
        try:
            evidence = json.loads(row["evidence_message_ids_json"])
        except json.JSONDecodeError:
            evidence = []
        if not isinstance(evidence, list):
            evidence = []
        items.append(
            {
                "observation_id": row["observation_id"],
                "category": row["category"],
                "summary": row["summary"],
                "evidence_message_ids": [str(item) for item in evidence if item is not None],
                "confidence": row["confidence"],
                "time_start": row["time_range_start"],
                "time_end": row["time_range_end"],
            }
        )
    return items


def _write_checkpoint(conn: sqlite3.Connection, settings: dict, status: str) -> None:
    prompt = settings["prompt_version"]
    model = settings["model_version"]
    checkpoint_id = f"{prompt}|{model}"
    extracts = int(
        conn.execute(
            """
            SELECT COUNT(*) AS n FROM memory_tasks
            WHERE task_kind = 'extract' AND status = ? AND prompt_version = ? AND model_version = ?
            """,
            (COMPLETED, prompt, model),
        ).fetchone()["n"]
    )
    merges = int(
        conn.execute(
            """
            SELECT COUNT(*) AS n FROM memory_tasks
            WHERE task_kind = 'merge' AND status = ? AND prompt_version = ? AND model_version = ?
            """,
            (COMPLETED, prompt, model),
        ).fetchone()["n"]
    )
    memories = int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE prompt_version = ? AND model_version = ?",
            (prompt, model),
        ).fetchone()["n"]
    )
    now = utc_now()
    conn.execute(
        """
        INSERT INTO memory_checkpoint (
            checkpoint_id, prompt_version, model_version, status,
            extracts_completed, merges_completed, memories_written, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(checkpoint_id) DO UPDATE SET
            status = excluded.status,
            extracts_completed = excluded.extracts_completed,
            merges_completed = excluded.merges_completed,
            memories_written = excluded.memories_written,
            updated_at = excluded.updated_at
        """,
        (checkpoint_id, prompt, model, status, extracts, merges, memories, now),
    )


def _status_counts(conn: sqlite3.Connection, prompt_version: str, model_version: str) -> dict[str, int]:
    counts = {status: 0 for status in PIPELINE_STATUSES}
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS n
        FROM memory_tasks
        WHERE prompt_version = ? AND model_version = ?
        GROUP BY status
        """,
        (prompt_version, model_version),
    ).fetchall()
    for row in rows:
        counts[row["status"]] = int(row["n"])
    return counts
