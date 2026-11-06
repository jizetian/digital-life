"""Persist conversation segments and a resumable segmentation checkpoint.

Messages are paged with a keyset on ``(conversation_id, source_position,
message_id)`` so a few hundred thousand rows are not loaded at once. Each
page commits the closed segments and the open-segment checkpoint together.
A later run continues after ``last_key`` without calling an LLM.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from common.clock import utc_now
from common.config import segmentation_config
from common.database import init_db
from common.logger import get_logger
from common.statuses import COMPLETED, FAILED, PENDING, RUNNING
from segmentation.splitter import (
    ClosedSegment,
    MessageRef,
    SegmentParams,
    Segmenter,
    open_segment_from_payload,
    open_segment_payload,
    segment_id_for,
    time_bounds,
)

log = get_logger(__name__)

CHECKPOINT_ID = "segmentation_active"


class SegmentationConflictError(Exception):
    """The saved checkpoint does not match this segmentation request."""


def resolve_params(
    *,
    gap_minutes: float | None = None,
    max_messages: int | None = None,
    overlap_messages: int | None = None,
    overlap_on_time_gap: bool | None = None,
) -> SegmentParams:
    cfg = segmentation_config()
    minutes = float(cfg.get("gap_minutes", 30) if gap_minutes is None else gap_minutes)
    maximum = int(cfg.get("max_messages", 50) if max_messages is None else max_messages)
    overlap = int(cfg.get("overlap_messages", 4) if overlap_messages is None else overlap_messages)
    if overlap_on_time_gap is None:
        flag = bool(cfg.get("overlap_on_time_gap", False))
    else:
        flag = bool(overlap_on_time_gap)
    return SegmentParams(
        gap_seconds=int(round(minutes * 60)),
        max_messages=maximum,
        overlap_messages=overlap,
        overlap_on_time_gap=flag,
    )


def run_segmentation(
    db_path: str | Path,
    *,
    params: SegmentParams | None = None,
    gap_minutes: float | None = None,
    max_messages: int | None = None,
    overlap_messages: int | None = None,
    overlap_on_time_gap: bool | None = None,
    rebuild: bool = False,
    max_segments: int | None = None,
    page_size: int | None = None,
) -> dict:
    """Segment ``db_path``.

    A finished checkpoint whose message count still matches is left as-is.
    An unfinished checkpoint for the same parameters resumes after the last
    committed message. ``--rebuild`` deletes segments (and cascaded
    distillation rows) and starts over. ``max_segments`` stops early and
    leaves a running checkpoint.
    """
    database = Path(db_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"database not found: {database}")
    resolved = params if params is not None else resolve_params(
        gap_minutes=gap_minutes,
        max_messages=max_messages,
        overlap_messages=overlap_messages,
        overlap_on_time_gap=overlap_on_time_gap,
    )
    params_hash = resolved.digest()
    page = int(segmentation_config().get("page_size", 2000) if page_size is None else page_size)
    if page < 1:
        raise ValueError("page_size must be positive")
    if max_segments is not None and max_segments < 1:
        raise ValueError("max_segments must be positive")

    conn = init_db(database)
    try:
        total = _message_count(conn)
        checkpoint = None if rebuild else read_checkpoint(conn)
        if rebuild:
            _clear_segmentation(conn)
            resumed = False
        elif checkpoint is None:
            resumed = False
        else:
            _ensure_compatible(checkpoint, resolved, total)
            if checkpoint["status"] == COMPLETED:
                return _public_result(
                    status=COMPLETED,
                    unchanged=True,
                    resumed=False,
                    params=resolved,
                    params_hash=params_hash,
                    messages_consumed=int(checkpoint.get("messages_consumed") or 0),
                    message_total=total,
                    segments_written=0,
                    segment_count=_segment_count(conn, params_hash),
                )
            resumed = True

        segmenter = Segmenter(
            resolved,
            open_segment=_restore_open(checkpoint) if resumed and checkpoint is not None else None,
        )
        after = tuple(checkpoint["last_key"]) if resumed and checkpoint is not None else None
        messages_consumed = int(checkpoint["messages_consumed"]) if resumed and checkpoint is not None else 0
        closed_this_run = 0
        stopped_early = False

        while True:
            rows = _fetch_page(conn, after, page)
            if not rows:
                break
            pending: list[ClosedSegment] = []
            for row in rows:
                message = _ref(row)
                pending.extend(segmenter.push(message))
                after = message.sort_key()
                messages_consumed += 1
                if max_segments is not None and closed_this_run + len(pending) >= max_segments:
                    stopped_early = True
                    break
            _commit_progress(
                conn,
                pending,
                segmenter=segmenter,
                after=after,
                messages_consumed=messages_consumed,
                message_total=total,
                params=resolved,
                params_hash=params_hash,
                status=RUNNING,
            )
            closed_this_run += len(pending)
            if stopped_early or len(rows) < page:
                break

        if stopped_early:
            log.info(
                "segmentation paused messages=%s segments_written=%s",
                messages_consumed,
                closed_this_run,
            )
            return _public_result(
                status=RUNNING,
                unchanged=False,
                resumed=resumed,
                params=resolved,
                params_hash=params_hash,
                messages_consumed=messages_consumed,
                message_total=total,
                segments_written=closed_this_run,
                segment_count=_segment_count(conn, params_hash),
            )

        final = segmenter.finish()
        written = closed_this_run + (1 if final is not None else 0)
        _commit_progress(
            conn,
            [final] if final is not None else [],
            segmenter=segmenter,
            after=after,
            messages_consumed=messages_consumed,
            message_total=total,
            params=resolved,
            params_hash=params_hash,
            status=COMPLETED,
        )
        log.info(
            "segmentation completed messages=%s segments_written=%s",
            messages_consumed,
            written,
        )
        return _public_result(
            status=COMPLETED,
            unchanged=False,
            resumed=resumed,
            params=resolved,
            params_hash=params_hash,
            messages_consumed=messages_consumed,
            message_total=total,
            segments_written=written,
            segment_count=_segment_count(conn, params_hash),
        )
    finally:
        conn.close()


def read_checkpoint(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute(
        "SELECT status, payload_json FROM processing_jobs WHERE job_id = ?",
        (CHECKPOINT_ID,),
    ).fetchone()
    if row is None or not row["payload_json"]:
        return None
    payload = json.loads(row["payload_json"])
    if not isinstance(payload, dict):
        return None
    payload["status"] = row["status"]
    return payload


def _ensure_compatible(checkpoint: dict, params: SegmentParams, total: int) -> None:
    if checkpoint.get("params_hash") != params.digest():
        previous = checkpoint.get("params") or {}
        raise SegmentationConflictError(
            "segmentation checkpoint uses different parameters "
            f"({previous}); requested {params.as_dict()}. Re-run with --rebuild."
        )
    recorded = checkpoint.get("message_total")
    if recorded is not None and int(recorded) != int(total):
        raise SegmentationConflictError(
            f"messages changed since segmentation ({recorded} -> {total}). Re-run with --rebuild."
        )
    if checkpoint.get("status") == COMPLETED:
        return
    if checkpoint.get("status") not in {RUNNING, FAILED}:
        raise SegmentationConflictError(
            f"segmentation checkpoint status {checkpoint.get('status')!r} cannot be resumed. "
            "Re-run with --rebuild."
        )
    if not checkpoint.get("last_key") or not checkpoint.get("open"):
        raise SegmentationConflictError(
            "segmentation checkpoint is missing resume state. Re-run with --rebuild."
        )


def _restore_open(checkpoint: dict):
    segment = open_segment_from_payload(checkpoint["open"])
    if not segment.messages:
        raise SegmentationConflictError("segmentation checkpoint open segment is empty. Re-run with --rebuild.")
    last = segment.messages[-1].sort_key()
    if tuple(last) != tuple(checkpoint["last_key"]):
        raise SegmentationConflictError(
            "segmentation checkpoint open segment does not end at last_key. Re-run with --rebuild."
        )
    return segment


def _clear_segmentation(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM conversation_segments")
    conn.execute("DELETE FROM processing_jobs WHERE job_id = ?", (CHECKPOINT_ID,))
    conn.commit()


def _fetch_page(conn: sqlite3.Connection, after: tuple | None, limit: int) -> list[sqlite3.Row]:
    columns = "message_id, conversation_id, timestamp, source_position"
    order = "ORDER BY COALESCE(conversation_id, ''), source_position, message_id"
    if after is None:
        return conn.execute(
            f"SELECT {columns} FROM messages {order} LIMIT ?",
            (limit,),
        ).fetchall()
    conversation_id, source_position, message_id = after
    return conn.execute(
        f"""
        SELECT {columns} FROM messages
        WHERE (COALESCE(conversation_id, ''), source_position, message_id) > (?, ?, ?)
        {order}
        LIMIT ?
        """,
        (conversation_id, source_position, message_id, limit),
    ).fetchall()


def _ref(row: sqlite3.Row) -> MessageRef:
    timestamp = row["timestamp"]
    return MessageRef(
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        timestamp=None if timestamp is None else int(timestamp),
        source_position=int(row["source_position"]),
    )


def _commit_progress(
    conn: sqlite3.Connection,
    closed: list[ClosedSegment],
    *,
    segmenter: Segmenter,
    after: tuple | None,
    messages_consumed: int,
    message_total: int,
    params: SegmentParams,
    params_hash: str,
    status: str,
) -> None:
    now = utc_now()
    try:
        for segment in closed:
            _insert_segment(conn, segment, params_hash=params_hash, now=now)
        payload = {
            "params_hash": params_hash,
            "params": params.as_dict(),
            "last_key": list(after) if after is not None else None,
            "messages_consumed": messages_consumed,
            "message_total": message_total,
            "open": open_segment_payload(segmenter.open),
        }
        _write_checkpoint(conn, status=status, payload=payload, now=now)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    if closed:
        log.info(
            "segmentation checkpoint status=%s messages=%s closed_batch=%s",
            status,
            messages_consumed,
            len(closed),
        )


def _insert_segment(conn: sqlite3.Connection, segment: ClosedSegment, *, params_hash: str, now: str) -> str:
    primary = segment.primary_messages
    if not primary:
        raise RuntimeError("refusing to store a segment with no primary messages")
    all_ids = [item.message_id for item in segment.messages]
    overlap_ids = [item.message_id for item in segment.overlap_messages]
    segment_id = segment_id_for(params_hash, segment.conversation_id, [item.message_id for item in primary])
    start_time, end_time = time_bounds(segment.messages)
    conn.execute(
        """
        INSERT INTO conversation_segments (
            segment_id, conversation_id, params_hash, start_time, end_time,
            start_message_id, end_message_id, message_ids_json, overlap_message_ids_json,
            message_count, target_message_count, processing_status, opened_by, closed_by,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            segment_id,
            segment.conversation_id,
            params_hash,
            start_time,
            end_time,
            all_ids[0],
            all_ids[-1],
            json.dumps(all_ids, ensure_ascii=False),
            json.dumps(overlap_ids, ensure_ascii=False),
            len(all_ids),
            len(primary),
            PENDING,
            segment.opened_by,
            segment.closed_by,
            now,
            now,
        ),
    )
    conn.executemany(
        """
        INSERT INTO conversation_segment_messages (segment_id, message_id, position, is_overlap)
        VALUES (?, ?, ?, ?)
        """,
        [
            (segment_id, item.message_id, index, 1 if index < segment.overlap_count else 0)
            for index, item in enumerate(segment.messages)
        ],
    )
    return segment_id


def _write_checkpoint(conn: sqlite3.Connection, *, status: str, payload: dict, now: str) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    conn.execute(
        """
        INSERT INTO processing_jobs (job_id, job_type, status, payload_json, created_at, updated_at)
        VALUES (?, 'segmentation_checkpoint', ?, ?, ?, ?)
        ON CONFLICT(job_id) DO UPDATE SET
            job_type = excluded.job_type,
            status = excluded.status,
            payload_json = excluded.payload_json,
            updated_at = excluded.updated_at
        """,
        (CHECKPOINT_ID, status, body, now, now),
    )


def _message_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"])


def _segment_count(conn: sqlite3.Connection, params_hash: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM conversation_segments WHERE params_hash = ?",
        (params_hash,),
    ).fetchone()
    return int(row["n"])


def _public_result(
    *,
    status: str,
    unchanged: bool,
    resumed: bool,
    params: SegmentParams,
    params_hash: str,
    messages_consumed: int,
    message_total: int,
    segments_written: int,
    segment_count: int,
) -> dict:
    return {
        "status": status,
        "unchanged": unchanged,
        "resumed": resumed,
        "params_hash": params_hash,
        "params": params.as_dict(),
        "messages_consumed": messages_consumed,
        "message_total": message_total,
        "segments_written": segments_written,
        "segment_count": segment_count,
    }
