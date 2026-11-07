"""Counts for segments, distillation tasks, and L1 observations."""

from __future__ import annotations

from pathlib import Path

from common.database import init_db
from common.statuses import PIPELINE_STATUSES
from memory.status import format_memory_status, memory_snapshot_conn
from personality.status import format_personality_status, personality_snapshot_conn
from segmentation.service import read_checkpoint


def pipeline_status(db_path: str | Path) -> dict:
    database = Path(db_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"database not found: {database}")
    conn = init_db(database)
    try:
        segments = _grouped(conn, "conversation_segments", "processing_status")
        tasks = _grouped(conn, "distillation_tasks", "status")
        categories = {
            row["category"]: int(row["n"])
            for row in conn.execute(
                """
                SELECT category, COUNT(*) AS n
                FROM l1_observations
                GROUP BY category
                ORDER BY category
                """
            ).fetchall()
        }
        checkpoint = read_checkpoint(conn)
        checkpoint_view = None
        if checkpoint is not None:
            checkpoint_view = {
                "status": checkpoint.get("status"),
                "params_hash": checkpoint.get("params_hash"),
                "params": checkpoint.get("params"),
                "messages_consumed": checkpoint.get("messages_consumed"),
                "message_total": checkpoint.get("message_total"),
            }
        return {
            "segments": segments,
            "tasks": tasks,
            "observations": {
                "total": sum(categories.values()),
                "by_category": categories,
            },
            "segmentation_checkpoint": checkpoint_view,
            "personality": personality_snapshot_conn(conn),
            "memory": memory_snapshot_conn(conn),
        }
    finally:
        conn.close()


def format_pipeline_status(status: dict) -> str:
    lines = ["=== ChatTwin 流水线 / pipeline ==="]
    lines.append(_block("segments", status["segments"]))
    lines.append(_block("tasks", status["tasks"]))
    observations = status["observations"]
    lines.append(f"observations: {observations['total']}")
    for category, count in observations["by_category"].items():
        lines.append(f"  {category}: {count}")
    checkpoint = status.get("segmentation_checkpoint")
    if checkpoint is None:
        lines.append("segmentation: (none)")
    else:
        params = checkpoint.get("params") or {}
        lines.append(
            "segmentation: "
            f"{checkpoint.get('status')} "
            f"messages {checkpoint.get('messages_consumed')}/{checkpoint.get('message_total')} "
            f"gap_seconds={params.get('gap_seconds')} "
            f"max_messages={params.get('max_messages')} "
            f"overlap_messages={params.get('overlap_messages')}"
        )
        lines.append(f"params_hash: {checkpoint.get('params_hash')}")
    personality = status.get("personality")
    if personality is not None:
        lines.append(format_personality_status(personality))
    memory = status.get("memory")
    if memory is not None:
        lines.append(format_memory_status(memory))
    return "\n".join(lines)


def _grouped(conn, table: str, column: str) -> dict:
    if table not in {"conversation_segments", "distillation_tasks"}:
        raise ValueError(table)
    if column not in {"processing_status", "status"}:
        raise ValueError(column)
    rows = conn.execute(
        f"SELECT {column} AS k, COUNT(*) AS n FROM {table} GROUP BY {column}"
    ).fetchall()
    by_status = {status: 0 for status in PIPELINE_STATUSES}
    for row in rows:
        by_status[row["k"]] = int(row["n"])
    return {"total": sum(by_status.values()), "by_status": by_status}


def _block(title: str, section: dict) -> str:
    parts = [f"{title}: {section['total']}"]
    for status, count in section["by_status"].items():
        if count:
            parts.append(f"  {status}: {count}")
    return "\n".join(parts)
