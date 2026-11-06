"""Import a Moect export into SQLite with checkpoints and a quality report."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cleaner.deduplicator import Deduplicator
from cleaner.normalize import MessageNormalizeError, normalize_message
from common.checkpoint import ResumeConflictError, find_resumable_job
from common.config import default_batch_size
from common.database import init_db, upsert_sender_mapping
from common.logger import get_logger
from common.schemas import REPORT_FIELDS, ROLE_TARGET, NormalizedMessage, RunCounters
from importer.json_adapter import iter_moect_events
from importer.schema_detector import SchemaError, detect_schema

log = get_logger(__name__)

INSERT_MESSAGE = """
INSERT INTO messages (
    message_id, conversation_id, local_id, server_id, shard,
    sender_id, sender_name, sender_role, is_sender, timestamp, time_str,
    message_type, type_code, raw_type, type_name, content, raw_content,
    is_forwarded, resource_status, resource_notice, source_file, source_position,
    job_id, created_at
) VALUES (
    ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?,
    ?, ?
)
ON CONFLICT(message_id) DO NOTHING
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_import(
    input_path: str | Path,
    db_path: str | Path,
    *,
    resume: bool = False,
    batch_size: int | None = None,
    job_id: str | None = None,
    max_messages: int | None = None,
    chunk_size: int | None = None,
) -> dict:
    """Import ``input_path`` into ``db_path``.

    Re-importing the same messages is idempotent: ``message_id`` is the primary
    key (``server_id``, or ``session:local_id`` when the server id is empty).
    ``--resume`` continues the newest unfinished job for this absolute path.
    ``max_messages`` pauses after that many records so a run can be continued.
    """
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    database = Path(db_path).expanduser().resolve()
    size = batch_size if batch_size is not None else default_batch_size()
    if size < 1:
        raise ValueError("batch_size must be positive")
    file_size = path.stat().st_size
    conn = init_db(database)
    try:
        job_id, start_after = _open_job(
            conn,
            path=path,
            file_size=file_size,
            resume=resume,
            batch_size=size,
            job_id=job_id,
        )
        try:
            report = _execute(
                conn,
                path=path,
                job_id=job_id,
                start_after=start_after,
                batch_size=size,
                max_messages=max_messages,
                chunk_size=chunk_size,
            )
        except Exception as exc:
            conn.rollback()
            text = str(exc) if isinstance(exc, (SchemaError, ResumeConflictError, OSError, UnicodeError, json.JSONDecodeError)) else f"{type(exc).__name__}: {exc}"
            _mark_failed(conn, job_id, text)
            raise
        return report
    finally:
        conn.close()


def begin_import(
    input_path: str | Path,
    db_path: str | Path,
    *,
    resume: bool = False,
    batch_size: int | None = None,
) -> dict:
    """Create or reopen a job row and return it without reading the export."""
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    database = Path(db_path).expanduser().resolve()
    size = batch_size if batch_size is not None else default_batch_size()
    conn = init_db(database)
    try:
        job_id, _start_after = _open_job(
            conn,
            path=path,
            file_size=path.stat().st_size,
            resume=resume,
            batch_size=size,
            job_id=None,
            status="pending",
        )
        row = conn.execute("SELECT * FROM import_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return job_to_dict(row)
    finally:
        conn.close()


def get_job(db_path: str | Path, job_id: str) -> dict | None:
    database = Path(db_path).expanduser().resolve()
    if not database.is_file():
        return None
    conn = init_db(database)
    try:
        row = conn.execute("SELECT * FROM import_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return job_to_dict(row)
    finally:
        conn.close()


def job_to_dict(row: sqlite3.Row) -> dict:
    stats = json.loads(row["stats_json"]) if row["stats_json"] else None
    return {
        "job_id": row["job_id"],
        "source_file": row["source_file"],
        "source_path": row["source_path"],
        "file_size": row["file_size"],
        "status": row["status"],
        "processed_count": row["processed_count"],
        "last_position": row["last_position"],
        "total_messages": row["total_messages"],
        "success_count": row["success_count"],
        "failed_count": row["failed_count"],
        "duplicate_count": row["duplicate_count"],
        "stats": stats,
        "error_message": row["error_message"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "finished_at": row["finished_at"],
    }


def format_report(job: dict) -> str:
    stats = job.get("stats") or {}
    lines = [
        "=== ChatTwin 导入报告 / import report ===",
        f"job_id: {job.get('job_id')}",
        f"status: {job.get('status')}",
        f"source_file: {job.get('source_file')}",
        f"总消息数 total_messages: {stats.get('total_messages')}",
        f"成功 success_count: {stats.get('success_count')}",
        f"失败 failed_count: {stats.get('failed_count')}",
        f"重复 duplicate_count: {stats.get('duplicate_count')}",
        f"时间缺失 missing_timestamp_count: {stats.get('missing_timestamp_count')}",
        f"发送者缺失 missing_sender_count: {stats.get('missing_sender_count')}",
        f"目标人物消息比例 target_person_ratio: {stats.get('target_person_ratio')}",
        f"文本消息比例 text_ratio: {stats.get('text_ratio')}",
        "kind:",
    ]
    for key, value in (stats.get("kind_counts") or {}).items():
        lines.append(f"  {key}: {value}")
    lines.append("type_name:")
    for key, value in (stats.get("type_name_counts") or {}).items():
        lines.append(f"  {key}: {value}")
    for field in REPORT_FIELDS:
        if field not in stats:
            lines.append(f"MISSING_FIELD {field}")
    return "\n".join(lines)


def _open_job(
    conn: sqlite3.Connection,
    *,
    path: Path,
    file_size: int,
    resume: bool,
    batch_size: int,
    job_id: str | None,
    status: str = "running",
) -> tuple[str, int]:
    source_path = str(path)
    now = utc_now()
    if resume:
        if job_id:
            row = conn.execute("SELECT * FROM import_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise ResumeConflictError(f"unknown job_id {job_id}")
            if int(row["file_size"] or -1) != file_size:
                raise ResumeConflictError("export file size changed; start a new import without --resume")
        else:
            row = find_resumable_job(conn, source_path, file_size)
        if row is not None and row["status"] != "completed":
            last = row["last_position"]
            start_after = -1 if last is None else int(last)
            conn.execute(
                """
                UPDATE import_jobs
                SET status = ?, error_message = NULL, finished_at = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                (status, now, row["job_id"]),
            )
            conn.execute("DELETE FROM import_errors WHERE job_id = ?", (row["job_id"],))
            conn.commit()
            log.info("resume job %s after position %s", row["job_id"], start_after)
            return row["job_id"], start_after
        log.info("no unfinished job for %s; starting a new import", path.name)

    new_id = job_id or uuid.uuid4().hex
    conn.execute(
        """
        INSERT INTO import_jobs (
            job_id, source_file, source_path, file_size, status, batch_size,
            processed_count, last_position, total_messages, success_count,
            failed_count, duplicate_count, stats_json, metadata_json, error_message,
            created_at, updated_at, finished_at
        ) VALUES (?, ?, ?, ?, ?, ?, 0, NULL, NULL, 0, 0, 0, NULL, NULL, NULL, ?, ?, NULL)
        """,
        (new_id, path.name, source_path, file_size, status, batch_size, now, now),
    )
    conn.commit()
    return new_id, -1


def _execute(
    conn: sqlite3.Connection,
    *,
    path: Path,
    job_id: str,
    start_after: int,
    batch_size: int,
    max_messages: int | None,
    chunk_size: int | None,
) -> dict:
    counters = RunCounters()
    deduper = Deduplicator()
    pending: list[NormalizedMessage] = []
    errors: list[tuple] = []
    last_flushed = start_after
    source_file = path.name
    created_at = utc_now()
    stream_kwargs = {"chunk_size": chunk_size} if chunk_size else {}

    def flush(through_index: int, status: str = "running") -> None:
        nonlocal last_flushed, pending, errors
        _write_batch(conn, job_id, pending, errors, created_at)
        pending = []
        errors = []
        last_flushed = through_index
        _store_progress(conn, job_id, counters, through_index, status=status, finished=False)
        conn.commit()

    for event in iter_moect_events(path, **stream_kwargs):
        if event[0] == "metadata":
            metadata = event[1]
            detect_schema(metadata)
            _remember_metadata(conn, job_id, metadata)
            conn.commit()
            continue
        index = event[1]
        raw = event[2]
        counters.total_messages += 1
        try:
            message = normalize_message(raw, source_file=source_file, source_position=index)
        except MessageNormalizeError as exc:
            counters.failed_count += 1
            errors.append(_error_row(job_id, source_file, index, None, exc, raw))
        else:
            counters.observe(message)
            first = not deduper.seen_in_file(message.message_id)
            if index > start_after and first:
                pending.append(message)
            if message.sender_id and message.conversation_id and index > start_after:
                _map_sender(conn, message)
        reached_limit = max_messages is not None and counters.total_messages >= max_messages
        should_flush = (
            reached_limit
            or counters.total_messages % batch_size == 0
            or len(pending) >= batch_size
        )
        if should_flush and index > last_flushed:
            flush(index, status="paused" if reached_limit else "running")
            log.info("imported through position %s (%s messages)", index, counters.total_messages)
        if reached_limit:
            return _finalize(conn, job_id, counters, status="paused")

    if counters.total_messages - 1 > last_flushed or pending or errors:
        last_index = counters.total_messages - 1
        flush(last_index if last_index >= 0 else -1)
    return _finalize(conn, job_id, counters, status="completed")


def _write_batch(conn, job_id, pending: list[NormalizedMessage], errors: list[tuple], created_at: str) -> None:
    if pending:
        conn.executemany(
            INSERT_MESSAGE,
            [(_message_params(message, job_id, created_at)) for message in pending],
        )
    if errors:
        conn.executemany(
            """
            INSERT INTO import_errors (
                job_id, source_file, source_position, message_id, error_type,
                error_message, raw_snippet, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            errors,
        )


def _message_params(message: NormalizedMessage, job_id: str, created_at: str) -> tuple:
    return (
        message.message_id,
        message.conversation_id,
        message.local_id,
        message.server_id,
        message.shard,
        message.sender_id,
        message.sender_name,
        message.sender_role,
        None if message.is_sender is None else int(message.is_sender),
        message.timestamp,
        message.time_str,
        message.message_type,
        message.type_code,
        message.raw_type,
        message.type_name,
        message.content,
        json.dumps(message.raw_content, ensure_ascii=False, separators=(",", ":")),
        int(message.is_forwarded),
        message.resource_status,
        message.resource_notice,
        message.source_file,
        message.source_position,
        job_id,
        created_at,
    )


def _error_row(job_id, source_file, index, message_id, exc: Exception, raw) -> tuple:
    return (
        job_id,
        source_file,
        index,
        message_id,
        type(exc).__name__,
        str(exc),
        _snippet(raw),
        utc_now(),
    )


def _snippet(raw) -> str:
    try:
        from cleaner.message_cleaner import slim_raw

        payload = slim_raw(raw) if isinstance(raw, dict) else raw
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(raw)
    if len(text) > 2000:
        return text[:2000] + "…"
    return text


def _remember_metadata(conn, job_id: str, metadata: dict) -> None:
    conn.execute(
        "UPDATE import_jobs SET metadata_json = ?, updated_at = ? WHERE job_id = ?",
        (json.dumps(metadata, ensure_ascii=False), utc_now(), job_id),
    )
    session = metadata.get("session")
    if not isinstance(session, str) or not session.strip():
        return
    name = metadata.get("name")
    sender_name = name if isinstance(name, str) and name.strip() else None
    upsert_sender_mapping(
        conn,
        conversation_id=session.strip(),
        sender_id=session.strip(),
        sender_name=sender_name,
        role=ROLE_TARGET,
        is_sender=None,
        from_metadata=1,
        from_message=0,
        now=utc_now(),
    )


def _map_sender(conn, message: NormalizedMessage) -> None:
    upsert_sender_mapping(
        conn,
        conversation_id=message.conversation_id,
        sender_id=message.sender_id,
        sender_name=message.sender_name,
        role=message.sender_role,
        is_sender=None if message.is_sender is None else int(message.is_sender),
        from_metadata=0,
        from_message=1,
        now=utc_now(),
    )


def _store_progress(
    conn,
    job_id: str,
    counters: RunCounters,
    through_index: int,
    *,
    status: str,
    finished: bool,
) -> None:
    success = _success_count(conn, job_id)
    duplicate = counters.total_messages - counters.failed_count - success
    if duplicate < 0:
        duplicate = 0
    report = counters.to_report(success, duplicate)
    now = utc_now()
    finished_at = now if finished else None
    total = counters.total_messages if finished or status in {"completed", "paused"} else None
    conn.execute(
        """
        UPDATE import_jobs SET
            status = ?,
            processed_count = ?,
            last_position = ?,
            total_messages = ?,
            success_count = ?,
            failed_count = ?,
            duplicate_count = ?,
            stats_json = ?,
            error_message = NULL,
            updated_at = ?,
            finished_at = CASE WHEN ? THEN ? ELSE finished_at END
        WHERE job_id = ?
        """,
        (
            status,
            counters.total_messages,
            through_index if through_index >= 0 else None,
            total if total is not None else counters.total_messages,
            success,
            counters.failed_count,
            duplicate,
            json.dumps(report, ensure_ascii=False),
            now,
            1 if finished else 0,
            finished_at,
            job_id,
        ),
    )


def _finalize(conn, job_id: str, counters: RunCounters, *, status: str) -> dict:
    through = counters.total_messages - 1
    _store_progress(
        conn,
        job_id,
        counters,
        through,
        status=status,
        finished=status in {"completed", "failed"},
    )
    conn.commit()
    row = conn.execute("SELECT * FROM import_jobs WHERE job_id = ?", (job_id,)).fetchone()
    result = job_to_dict(row)
    stats = result.get("stats") or {}
    log.info(
        "job %s %s total=%s success=%s failed=%s duplicate=%s",
        job_id,
        status,
        stats.get("total_messages"),
        result["success_count"],
        result["failed_count"],
        result["duplicate_count"],
    )
    return result


def _success_count(conn, job_id: str) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM messages WHERE job_id = ?", (job_id,)).fetchone()
    return int(row["n"])


def _mark_failed(conn, job_id: str, message: str) -> None:
    now = utc_now()
    try:
        conn.execute(
            """
            UPDATE import_jobs
            SET status = 'failed', error_message = ?, updated_at = ?, finished_at = ?
            WHERE job_id = ?
            """,
            (message, now, now, job_id),
        )
        conn.commit()
    except sqlite3.Error:
        log.exception("could not mark job %s failed", job_id)
