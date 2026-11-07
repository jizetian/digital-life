"""L2 merge and L3 profile build, with a checkpoint per model call.

``task_id`` is ``sha256(f"{kind}|{dimension}|{batch_key}|{prompt_version}|{model_version}")``.
Completed and skipped tasks are not sent to the model again. Pending,
retrying, and interrupted running tasks are resumed on the next invocation.

The model sees L1 summaries and observation ids. It does not see raw messages.
Message ids stay on the observation rows and are copied into the evidence file.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

from common.clock import utc_now
from common.config import ROOT, distillation_config, personality_config
from common.database import init_db
from common.logger import get_logger
from common.statuses import COMPLETED, FAILED, PENDING, RETRYING, RUNNING, SKIPPED
from distillation.llm import LLMClient, build_llm_client
from personality.contradiction_detector import finalize_traits, parse_judgments
from personality.dimensions import CONTEXT_CATEGORIES, DIMENSIONS, dimension_for_category
from personality.models import ObsView, TraitRecord
from personality.profile_builder import build_profile
from personality.prompting import (
    PROMPT_VERSION,
    load_contradict_prompt,
    load_merge_prompt,
    load_reduce_prompt,
    render_contradict_prompt,
    render_merge_prompt,
    render_reduce_prompt,
)
from personality.trait_merger import (
    assign_trait_keys,
    batch_key_for,
    local_traits_from_results,
    observation_fingerprint,
    parse_merge_response,
    parse_reduce_response,
    plan_batches,
    reduce_batch_key,
    trait_fingerprint,
)

log = get_logger(__name__)

_TERMINAL = {COMPLETED, SKIPPED}


class PersonalityError(Exception):
    """The personality run cannot start or a task failed closed."""


def make_task_id(
    kind: str,
    dimension: str,
    batch_key: str,
    prompt_version: str,
    model_version: str,
) -> str:
    raw = f"{kind}|{dimension}|{batch_key}|{prompt_version}|{model_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def run_personality(
    db_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    batch_size: int | None = None,
    contradict_batch_size: int | None = None,
    client: LLMClient | None = None,
    prompt_version: str | None = None,
    model: str | None = None,
    model_version: str | None = None,
    base_url: str | None = None,
    max_tasks: int | None = None,
    max_attempts: int | None = None,
    min_independent: int | None = None,
    retry_failed: bool = False,
) -> dict:
    """Merge L1 observations and, when every task is finished, write the profile.

    ``max_tasks`` stops after that many model calls and leaves the rest pending.
    The returned status is ``completed`` when the profile was written,
    ``paused`` when claimable tasks remain, and ``failed`` when nothing
    claimable remains but a task is failed.
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
        contradict_batch_size=contradict_batch_size,
        max_attempts=max_attempts,
        min_independent=min_independent,
        base_url=base_url,
        output_dir=output_dir,
    )
    conn = init_db(database)
    try:
        by_dimension, context_counts, index, fingerprint = _load_observations(conn)
        _drop_stale_traits(
            conn,
            prompt_version=settings["prompt_version"],
            model_version=settings["model_version"],
            fingerprint=fingerprint,
        )
        attempts_completed = 0
        attempts_failed = 0
        # One attempt per task per invocation. A retry waits for the next run,
        # matching L1 distillation, so a crash does not burn every attempt.
        attempted_now: set[str] = set()
        while max_tasks is None or (attempts_completed + attempts_failed) < max_tasks:
            _sync(
                conn,
                by_dimension=by_dimension,
                prompt_version=settings["prompt_version"],
                model_version=settings["model_version"],
                batch_size=settings["batch_size"],
                contradict_batch_size=settings["contradict_batch_size"],
                retry_failed=retry_failed,
            )
            retry_failed = False
            claimable = [
                row
                for row in _claimable(
                    conn,
                    prompt_version=settings["prompt_version"],
                    model_version=settings["model_version"],
                    max_attempts=settings["max_attempts"],
                )
                if row["task_id"] not in attempted_now
            ]
            if not claimable:
                break
            if client is None:
                if not settings["model"]:
                    raise PersonalityError("LLM model is empty")
                client = build_llm_client(
                    base_url=settings["base_url"],
                    model=settings["model"],
                    timeout=settings["timeout_seconds"],
                    temperature=settings["temperature"],
                )
            task_id = claimable[0]["task_id"]
            attempted_now.add(task_id)
            outcome = _execute_task(
                conn,
                task_id,
                client=client,
                index=index,
                max_attempts=settings["max_attempts"],
                max_chars=settings["max_chars_per_summary"],
            )
            if outcome == COMPLETED:
                attempts_completed += 1
            elif outcome in {FAILED, RETRYING}:
                attempts_failed += 1
        _sync(
            conn,
            by_dimension=by_dimension,
            prompt_version=settings["prompt_version"],
            model_version=settings["model_version"],
            batch_size=settings["batch_size"],
            contradict_batch_size=settings["contradict_batch_size"],
            retry_failed=False,
        )
        remaining = len(
            _claimable(
                conn,
                prompt_version=settings["prompt_version"],
                model_version=settings["model_version"],
                max_attempts=settings["max_attempts"],
            )
        )
        failed = _failed_count(
            conn,
            prompt_version=settings["prompt_version"],
            model_version=settings["model_version"],
        )
        published = None
        if remaining:
            status = "paused"
        elif failed:
            status = FAILED
        else:
            status = COMPLETED
            records = _materialize(
                conn,
                by_dimension=by_dimension,
                prompt_version=settings["prompt_version"],
                model_version=settings["model_version"],
                min_independent=settings["min_independent"],
            )
            _replace_traits(
                conn,
                records,
                prompt_version=settings["prompt_version"],
                model_version=settings["model_version"],
            )
            published = build_profile(
                output_dir=settings["output_dir"],
                traits=records,
                observations=index,
                context_counts=context_counts,
                prompt_version=settings["prompt_version"],
                model_version=settings["model_version"],
                observation_fingerprint=fingerprint,
                min_independent=settings["min_independent"],
                generated_at=utc_now(),
            )
            conn.commit()
        _write_checkpoint(
            conn,
            settings=settings,
            fingerprint=fingerprint,
            status=status,
            published=published,
        )
        conn.commit()
        tasks = _status_counts(
            conn,
            prompt_version=settings["prompt_version"],
            model_version=settings["model_version"],
        )
        trait_counts = _trait_status_counts(
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
            "min_independent": settings["min_independent"],
            "attempts_completed": attempts_completed,
            "attempts_failed": attempts_failed,
            "tasks_remaining": remaining,
            "tasks": tasks,
            "traits": trait_counts,
            "observations": {
                "by_dimension": {name: len(by_dimension[name]) for name in DIMENSIONS},
                "context": context_counts,
            },
            "output_dir": None if published is None else published["output_dir"],
            "files": [] if published is None else published["files"],
            "observation_fingerprint": fingerprint,
        }
    finally:
        conn.close()


def _settings(
    *,
    model: str | None,
    model_version: str | None,
    prompt_version: str | None,
    batch_size: int | None,
    contradict_batch_size: int | None,
    max_attempts: int | None,
    min_independent: int | None,
    base_url: str | None,
    output_dir: str | Path | None,
) -> dict:
    cfg = personality_config()
    distill = distillation_config()
    chosen_model = str(distill.get("model") or "") if model is None else str(model)
    version = str(distill.get("model_version") or "") if model_version is None else str(model_version)
    if not version:
        version = chosen_model
    if prompt_version is not None:
        prompt = prompt_version
    else:
        prompt = str(cfg.get("prompt_version") or os.environ.get("CHATTWIN_L2_PROMPT_VERSION") or PROMPT_VERSION)
    size = int(cfg.get("batch_size", 20) if batch_size is None else batch_size)
    contradict_size = int(
        cfg.get("contradict_batch_size", 30) if contradict_batch_size is None else contradict_batch_size
    )
    attempts = int(cfg.get("max_attempts", 3) if max_attempts is None else max_attempts)
    independent = int(cfg.get("min_independent", 2) if min_independent is None else min_independent)
    if size < 1:
        raise ValueError("batch_size must be positive")
    if contradict_size < 1:
        raise ValueError("contradict_batch_size must be positive")
    if attempts < 1:
        raise ValueError("max_attempts must be positive")
    if independent < 1:
        raise ValueError("min_independent must be positive")
    if not version:
        raise PersonalityError("model_version is empty")
    return {
        "model": chosen_model,
        "model_version": version,
        "prompt_version": prompt,
        "batch_size": size,
        "contradict_batch_size": contradict_size,
        "max_attempts": attempts,
        "min_independent": independent,
        "base_url": None if base_url is None else base_url,
        "max_chars_per_summary": int(cfg.get("max_chars_per_summary", 500)),
        "timeout_seconds": float(distill.get("timeout_seconds", 120)),
        "temperature": float(distill.get("temperature", 0)),
        "output_dir": _resolve_output_dir(output_dir, cfg.get("output_dir")),
    }


def _resolve_output_dir(explicit, configured) -> Path:
    if explicit is not None and str(explicit).strip():
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path
    path = Path(str(configured or "personality")).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def _load_observations(
    conn: sqlite3.Connection,
) -> tuple[dict[str, list[ObsView]], dict[str, int], dict[str, ObsView], str]:
    rows = conn.execute(
        """
        SELECT observation_id, segment_id, category, summary,
               evidence_message_ids_json, confidence, time_range_start, time_range_end
        FROM l1_observations
        """
    ).fetchall()
    by_dimension = {name: [] for name in DIMENSIONS}
    context = {name: 0 for name in CONTEXT_CATEGORIES}
    index: dict[str, ObsView] = {}
    fingerprint_ids: list[str] = []
    for row in rows:
        summary = " ".join((row["summary"] or "").split())
        if not summary:
            continue
        category = row["category"]
        dimension = dimension_for_category(category)
        fingerprint_ids.append(row["observation_id"])
        if dimension is None:
            if category in context:
                context[category] += 1
            continue
        message_ids = _message_ids(row["evidence_message_ids_json"])
        start = None if row["time_range_start"] is None else int(row["time_range_start"])
        end = None if row["time_range_end"] is None else int(row["time_range_end"])
        if start is not None and end is not None and start > end:
            start, end = end, start
        obs = ObsView(
            observation_id=row["observation_id"],
            segment_id=row["segment_id"] or "",
            category=category,
            dimension=dimension,
            summary=summary,
            evidence_message_ids=message_ids,
            confidence=None if row["confidence"] is None else float(row["confidence"]),
            time_start=start,
            time_end=end,
        )
        by_dimension[dimension].append(obs)
        index[obs.observation_id] = obs
    return by_dimension, context, index, observation_fingerprint(fingerprint_ids)


def _message_ids(raw: str) -> list[str]:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if item is not None]


def _drop_stale_traits(conn, *, prompt_version: str, model_version: str, fingerprint: str) -> None:
    row = conn.execute(
        "SELECT observation_fingerprint FROM personality_checkpoint WHERE checkpoint_id = ?",
        (_checkpoint_id(prompt_version, model_version),),
    ).fetchone()
    if row is not None and row["observation_fingerprint"] != fingerprint:
        conn.execute(
            "DELETE FROM l2_traits WHERE prompt_version = ? AND model_version = ?",
            (prompt_version, model_version),
        )
        conn.commit()


def _sync(
    conn,
    *,
    by_dimension: dict[str, list[ObsView]],
    prompt_version: str,
    model_version: str,
    batch_size: int,
    contradict_batch_size: int,
    retry_failed: bool,
) -> None:
    now = utc_now()
    planned: list[str] = []
    for dimension in DIMENSIONS:
        batches = plan_batches(by_dimension[dimension], batch_size)
        for batch in batches:
            ids = [item.observation_id for item in batch]
            key = batch_key_for(ids)
            task_id = make_task_id("merge", dimension, key, prompt_version, model_version)
            planned.append(task_id)
            _insert_task(
                conn,
                task_id=task_id,
                kind="merge",
                dimension=dimension,
                batch_key=key,
                prompt_version=prompt_version,
                model_version=model_version,
                status=PENDING,
                source_ids=ids,
                result=None,
                now=now,
            )
        merge_rows = _tasks_for(
            conn,
            kind="merge",
            dimension=dimension,
            prompt_version=prompt_version,
            model_version=model_version,
            task_ids=planned,
        )
        if not merge_rows or any(row["status"] not in _TERMINAL for row in merge_rows):
            continue
        local = _local_from_rows(merge_rows)
        productive = [
            row for row in merge_rows if _trait_count(row) > 0
        ]
        if len(productive) >= 2:
            key = reduce_batch_key(local)
            task_id = make_task_id("reduce", dimension, key, prompt_version, model_version)
            planned.append(task_id)
            _insert_task(
                conn,
                task_id=task_id,
                kind="reduce",
                dimension=dimension,
                batch_key=key,
                prompt_version=prompt_version,
                model_version=model_version,
                status=PENDING,
                source_ids=sorted({oid for trait in local.values() for oid in trait["observation_ids"]}),
                result=None,
                now=now,
            )
        else:
            traits = list(local.values())
            key = reduce_batch_key(local)
            task_id = make_task_id("reduce", dimension, key, prompt_version, model_version)
            planned.append(task_id)
            _insert_task(
                conn,
                task_id=task_id,
                kind="reduce",
                dimension=dimension,
                batch_key=key,
                prompt_version=prompt_version,
                model_version=model_version,
                status=SKIPPED,
                source_ids=sorted({oid for trait in traits for oid in trait["observation_ids"]}),
                result={"traits": traits, "passthrough": True},
                now=now,
            )
        reduce_rows = _tasks_for(
            conn,
            kind="reduce",
            dimension=dimension,
            prompt_version=prompt_version,
            model_version=model_version,
            task_ids=planned,
        )
        if not reduce_rows or any(row["status"] not in _TERMINAL for row in reduce_rows):
            continue
        traits = _traits_from_result(reduce_rows[0]["result_json"])
        observation_ids = [item.observation_id for item in by_dimension[dimension]]
        fingerprint = trait_fingerprint(traits)
        if not traits or not _needs_contradiction(traits, set(observation_ids)):
            key = _contradict_key(fingerprint, [])
            task_id = make_task_id("contradict", dimension, key, prompt_version, model_version)
            planned.append(task_id)
            _insert_task(
                conn,
                task_id=task_id,
                kind="contradict",
                dimension=dimension,
                batch_key=key,
                prompt_version=prompt_version,
                model_version=model_version,
                status=SKIPPED,
                source_ids=[],
                result={"judgments": []},
                now=now,
            )
            continue
        ordered = sorted(
            by_dimension[dimension],
            key=lambda item: (item.time_start is None, item.time_start or 0, item.observation_id),
        )
        for offset in range(0, len(ordered), contradict_batch_size):
            chunk = ordered[offset : offset + contradict_batch_size]
            ids = [item.observation_id for item in chunk]
            key = _contradict_key(fingerprint, ids)
            task_id = make_task_id("contradict", dimension, key, prompt_version, model_version)
            planned.append(task_id)
            _insert_task(
                conn,
                task_id=task_id,
                kind="contradict",
                dimension=dimension,
                batch_key=key,
                prompt_version=prompt_version,
                model_version=model_version,
                status=PENDING,
                source_ids=ids,
                result=None,
                now=now,
            )
    _deactivate_unplanned(conn, planned, prompt_version, model_version, now)
    if retry_failed:
        conn.execute(
            """
            UPDATE personality_tasks
            SET status = ?, attempt_count = 0, error_message = NULL,
                finished_at = NULL, updated_at = ?
            WHERE active = 1 AND status = ? AND prompt_version = ? AND model_version = ?
            """,
            (RETRYING, now, FAILED, prompt_version, model_version),
        )
    conn.commit()


def _insert_task(
    conn,
    *,
    task_id: str,
    kind: str,
    dimension: str,
    batch_key: str,
    prompt_version: str,
    model_version: str,
    status: str,
    source_ids: list[str],
    result: dict | None,
    now: str,
) -> None:
    result_json = None if result is None else json.dumps(result, ensure_ascii=False)
    result_count = 0
    if result is not None:
        result_count = len(result.get("traits") or result.get("judgments") or [])
    finished = now if status in _TERMINAL else None
    conn.execute(
        """
        INSERT INTO personality_tasks (
            task_id, task_kind, dimension, batch_key, prompt_version, model_version,
            status, active, attempt_count, source_ids_json, result_json, result_count,
            created_at, updated_at, finished_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(task_id) DO UPDATE SET active = 1
        """,
        (
            task_id,
            kind,
            dimension,
            batch_key,
            prompt_version,
            model_version,
            status,
            json.dumps(source_ids, ensure_ascii=False),
            result_json,
            result_count,
            now,
            now,
            finished,
        ),
    )


def _tasks_for(conn, *, kind: str, dimension: str, prompt_version: str, model_version: str, task_ids: list[str]):
    """Active tasks for this plan. Older rows stay active until deactivate runs."""
    wanted = set(task_ids)
    if not wanted:
        return []
    rows = conn.execute(
        """
        SELECT * FROM personality_tasks
        WHERE active = 1 AND task_kind = ? AND dimension = ?
          AND prompt_version = ? AND model_version = ?
        ORDER BY batch_key
        """,
        (kind, dimension, prompt_version, model_version),
    ).fetchall()
    return [row for row in rows if row["task_id"] in wanted]


def _deactivate_unplanned(conn, planned: list[str], prompt_version: str, model_version: str, now: str) -> None:
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS personality_plan (task_id TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM personality_plan")
    if planned:
        conn.executemany("INSERT INTO personality_plan (task_id) VALUES (?)", [(task_id,) for task_id in planned])
    conn.execute(
        """
        UPDATE personality_tasks
        SET active = 0, updated_at = ?
        WHERE prompt_version = ? AND model_version = ? AND active = 1
          AND task_id NOT IN (SELECT task_id FROM personality_plan)
        """,
        (now, prompt_version, model_version),
    )


def _claimable(conn, *, prompt_version: str, model_version: str, max_attempts: int):
    return conn.execute(
        """
        SELECT task_id
        FROM personality_tasks
        WHERE active = 1 AND prompt_version = ? AND model_version = ?
          AND (
                (status IN (?, ?) AND attempt_count < ?)
                OR status = ?
              )
        ORDER BY CASE task_kind WHEN 'merge' THEN 0 WHEN 'reduce' THEN 1 ELSE 2 END,
                 dimension, batch_key
        """,
        (prompt_version, model_version, PENDING, RETRYING, max_attempts, RUNNING),
    ).fetchall()


def _failed_count(conn, *, prompt_version: str, model_version: str) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS n FROM personality_tasks
        WHERE active = 1 AND status = ? AND prompt_version = ? AND model_version = ?
        """,
        (FAILED, prompt_version, model_version),
    ).fetchone()
    return int(row["n"])


def _execute_task(conn, task_id: str, *, client: LLMClient, index: dict[str, ObsView], max_attempts: int, max_chars: int) -> str:
    row = conn.execute("SELECT * FROM personality_tasks WHERE task_id = ?", (task_id,)).fetchone()
    if row is None or row["status"] in {COMPLETED, SKIPPED, FAILED}:
        return SKIPPED
    attempt = int(row["attempt_count"]) + 1
    now = utc_now()
    conn.execute(
        """
        UPDATE personality_tasks
        SET status = ?, attempt_count = ?, started_at = COALESCE(started_at, ?),
            updated_at = ?, error_message = NULL
        WHERE task_id = ?
        """,
        (RUNNING, attempt, now, now, task_id),
    )
    conn.commit()
    try:
        result, count = _run_kind(conn, row, client=client, index=index, max_chars=max_chars)
        finished = utc_now()
        conn.execute(
            """
            UPDATE personality_tasks
            SET status = ?, result_json = ?, result_count = ?, finished_at = ?,
                updated_at = ?, error_message = NULL
            WHERE task_id = ?
            """,
            (
                COMPLETED,
                json.dumps(result, ensure_ascii=False),
                count,
                finished,
                finished,
                task_id,
            ),
        )
        conn.commit()
        log.info("personality task %s completed kind=%s count=%s", task_id[:12], row["task_kind"], count)
        return COMPLETED
    except Exception as exc:
        _mark_failed(conn, task_id, attempt, max_attempts, exc)
        return RETRYING if attempt < max_attempts else FAILED


def _run_kind(conn, row, *, client: LLMClient, index: dict[str, ObsView], max_chars: int) -> tuple[dict, int]:
    kind = row["task_kind"]
    source_ids = json.loads(row["source_ids_json"])
    if kind == "merge":
        observations = [index[oid] for oid in source_ids if oid in index]
        raw = client.complete(
            system=load_merge_prompt(),
            user=render_merge_prompt(
                dimension=row["dimension"],
                observations=observations,
                max_chars=max_chars,
            ),
        )
        _require_text(raw)
        traits = parse_merge_response(raw, allowed_ids=set(source_ids))
        return {"traits": traits}, len(traits)
    if kind == "reduce":
        merge_rows = conn.execute(
            """
            SELECT batch_key, result_json FROM personality_tasks
            WHERE active = 1 AND task_kind = 'merge' AND dimension = ?
              AND prompt_version = ? AND model_version = ? AND status = ?
            ORDER BY batch_key
            """,
            (row["dimension"], row["prompt_version"], row["model_version"], COMPLETED),
        ).fetchall()
        local = local_traits_from_results([(item["batch_key"], _loads(item["result_json"])) for item in merge_rows])
        raw = client.complete(
            system=load_reduce_prompt(),
            user=render_reduce_prompt(dimension=row["dimension"], local_traits=list(local.items())),
        )
        _require_text(raw)
        traits = parse_reduce_response(raw, local_traits=local)
        return {"traits": traits}, len(traits)
    if kind == "contradict":
        reduce_rows = conn.execute(
            """
            SELECT result_json FROM personality_tasks
            WHERE active = 1 AND task_kind = 'reduce' AND dimension = ?
              AND prompt_version = ? AND model_version = ?
              AND status IN (?, ?)
            """,
            (row["dimension"], row["prompt_version"], row["model_version"], COMPLETED, SKIPPED),
        ).fetchall()
        if len(reduce_rows) != 1:
            raise PersonalityError("contradict task expected one reduced trait set")
        reduce_row = reduce_rows[0]
        traits = _traits_from_result(reduce_row["result_json"])
        keyed = assign_trait_keys(traits)
        observations = [index[oid] for oid in source_ids if oid in index]
        raw = client.complete(
            system=load_contradict_prompt(),
            user=render_contradict_prompt(
                dimension=row["dimension"],
                traits=keyed,
                observations=observations,
                max_chars=max_chars,
            ),
        )
        _require_text(raw)
        judgments = parse_judgments(
            raw,
            trait_keys={key for key, _ in keyed},
            allowed_observation_ids=set(source_ids),
        )
        return {"judgments": judgments}, len(judgments)
    raise PersonalityError(f"unknown personality task kind {kind}")


def _require_text(raw) -> None:
    if not isinstance(raw, str):
        raise PersonalityError("LLM client returned a non-text response")


def _mark_failed(conn, task_id: str, attempt: int, max_attempts: int, exc: BaseException) -> None:
    conn.rollback()
    status = RETRYING if attempt < max_attempts else FAILED
    now = utc_now()
    message = " ".join(f"{type(exc).__name__}: {exc}".split())[:500]
    conn.execute(
        """
        UPDATE personality_tasks
        SET status = ?, error_message = ?, finished_at = ?, updated_at = ?
        WHERE task_id = ?
        """,
        (status, message, now, now, task_id),
    )
    conn.commit()
    log.info("personality task %s %s attempt=%s", task_id[:12], status, attempt)


def _materialize(
    conn,
    *,
    by_dimension: dict[str, list[ObsView]],
    prompt_version: str,
    model_version: str,
    min_independent: int,
) -> list[TraitRecord]:
    records: list[TraitRecord] = []
    for dimension in DIMENSIONS:
        if not by_dimension[dimension]:
            continue
        reduce_rows = conn.execute(
            """
            SELECT result_json FROM personality_tasks
            WHERE active = 1 AND task_kind = 'reduce' AND dimension = ?
              AND prompt_version = ? AND model_version = ?
              AND status IN (?, ?)
            """,
            (dimension, prompt_version, model_version, COMPLETED, SKIPPED),
        ).fetchall()
        if len(reduce_rows) != 1:
            raise PersonalityError(f"dimension {dimension} expected one reduced trait set")
        reduce_row = reduce_rows[0]
        traits = _traits_from_result(reduce_row["result_json"])
        keyed = assign_trait_keys(traits)
        judgments: list[dict] = []
        rows = conn.execute(
            """
            SELECT batch_key, result_json FROM personality_tasks
            WHERE active = 1 AND task_kind = 'contradict' AND dimension = ?
              AND prompt_version = ? AND model_version = ?
              AND status IN (?, ?)
            ORDER BY batch_key
            """,
            (dimension, prompt_version, model_version, COMPLETED, SKIPPED),
        ).fetchall()
        for row in rows:
            payload = _loads(row["result_json"])
            items = payload.get("judgments") or []
            if isinstance(items, list):
                judgments.extend(item for item in items if isinstance(item, dict))
        dimension_index = {item.observation_id: item for item in by_dimension[dimension]}
        records.extend(
            finalize_traits(
                keyed,
                judgments,
                dimension_index,
                dimension=dimension,
                min_independent=min_independent,
                prompt_version=prompt_version,
                model_version=model_version,
            )
        )
    return records


def _replace_traits(conn, records: list[TraitRecord], *, prompt_version: str, model_version: str) -> None:
    now = utc_now()
    conn.execute(
        "DELETE FROM l2_traits WHERE prompt_version = ? AND model_version = ?",
        (prompt_version, model_version),
    )
    for record in records:
        payload = {
            "variants": record.variants,
            "counter_details": record.counter_details,
            "stages": record.stages,
        }
        conn.execute(
            """
            INSERT INTO l2_traits (
                trait_id, prompt_version, model_version, dimension, statement,
                situational_scope, confidence, evidence_count, independent_evidence_count,
                counter_evidence_count, evidence_ids_json, counter_evidence_ids_json,
                valid_from, valid_to, status, stages_json, variants_json, payload_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.trait_id,
                prompt_version,
                model_version,
                record.dimension,
                record.statement,
                record.situational_scope,
                record.confidence,
                record.evidence_count,
                record.independent_evidence_count,
                record.counter_evidence_count,
                json.dumps(record.evidence_ids, ensure_ascii=False),
                json.dumps(record.counter_evidence_ids, ensure_ascii=False),
                record.valid_from,
                record.valid_to,
                record.status,
                json.dumps(record.stages, ensure_ascii=False),
                json.dumps(record.variants, ensure_ascii=False),
                json.dumps(payload, ensure_ascii=False),
                now,
                now,
            ),
        )


def _write_checkpoint(conn, *, settings: dict, fingerprint: str, status: str, published: dict | None) -> None:
    now = utc_now()
    traits = {"stable": 0, "provisional": 0, "contested": 0}
    if published is not None:
        traits = {
            "stable": published["traits_stable"],
            "provisional": published["traits_provisional"],
            "contested": published["traits_contested"],
        }
    written_at = now if status == COMPLETED else None
    payload = {
        "files": [] if published is None else published["files"],
        "observation_fingerprint": fingerprint,
    }
    conn.execute(
        """
        INSERT INTO personality_checkpoint (
            checkpoint_id, prompt_version, model_version, observation_fingerprint,
            status, output_dir, profile_written_at, min_independent,
            traits_stable, traits_provisional, traits_contested, payload_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(checkpoint_id) DO UPDATE SET
            observation_fingerprint = excluded.observation_fingerprint,
            status = excluded.status,
            output_dir = excluded.output_dir,
            profile_written_at = excluded.profile_written_at,
            min_independent = excluded.min_independent,
            traits_stable = excluded.traits_stable,
            traits_provisional = excluded.traits_provisional,
            traits_contested = excluded.traits_contested,
            payload_json = excluded.payload_json,
            updated_at = excluded.updated_at
        """,
        (
            _checkpoint_id(settings["prompt_version"], settings["model_version"]),
            settings["prompt_version"],
            settings["model_version"],
            fingerprint,
            status,
            str(settings["output_dir"]),
            written_at,
            settings["min_independent"],
            traits["stable"],
            traits["provisional"],
            traits["contested"],
            json.dumps(payload, ensure_ascii=False),
            now,
        ),
    )


def _checkpoint_id(prompt_version: str, model_version: str) -> str:
    return f"personality|{prompt_version}|{model_version}"


def _status_counts(conn, *, prompt_version: str, model_version: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS n
        FROM personality_tasks
        WHERE active = 1 AND prompt_version = ? AND model_version = ?
        GROUP BY status
        """,
        (prompt_version, model_version),
    ).fetchall()
    return {row["status"]: int(row["n"]) for row in rows}


def _trait_status_counts(conn, *, prompt_version: str, model_version: str) -> dict[str, int]:
    counts = {"stable": 0, "provisional": 0, "contested": 0}
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS n
        FROM l2_traits
        WHERE prompt_version = ? AND model_version = ?
        GROUP BY status
        """,
        (prompt_version, model_version),
    ).fetchall()
    for row in rows:
        if row["status"] in counts:
            counts[row["status"]] = int(row["n"])
    return counts


def _local_from_rows(rows) -> dict[str, dict]:
    pairs = []
    for row in rows:
        if row["status"] != COMPLETED:
            continue
        pairs.append((row["batch_key"], _loads(row["result_json"])))
    return local_traits_from_results(pairs)


def _trait_count(row) -> int:
    if row["status"] != COMPLETED or not row["result_json"]:
        return 0
    return len(_traits_from_result(row["result_json"]))


def _traits_from_result(raw: str | None) -> list[dict]:
    payload = _loads(raw)
    traits = payload.get("traits") or []
    if not isinstance(traits, list):
        return []
    return [item for item in traits if isinstance(item, dict)]


def _loads(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _needs_contradiction(traits: list[dict], observation_ids: set[str]) -> bool:
    for trait in traits:
        outside = observation_ids - set(trait.get("observation_ids") or [])
        if outside:
            return True
    return False


def _contradict_key(fingerprint: str, observation_ids: list[str]) -> str:
    raw = fingerprint + "|" + ",".join(sorted(observation_ids))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
