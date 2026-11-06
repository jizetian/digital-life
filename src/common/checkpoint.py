"""Find an unfinished import job so a run can continue after a stop."""

from __future__ import annotations

import sqlite3

RESUMABLE_STATUSES = ("running", "paused", "failed", "pending")


class ResumeConflictError(Exception):
    """The export file changed since the unfinished job was created."""


def find_resumable_job(
    conn: sqlite3.Connection,
    source_path: str,
    file_size: int,
) -> sqlite3.Row | None:
    placeholders = ",".join("?" for _ in RESUMABLE_STATUSES)
    row = conn.execute(
        f"""
        SELECT * FROM import_jobs
        WHERE source_path = ? AND status IN ({placeholders})
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (source_path, *RESUMABLE_STATUSES),
    ).fetchone()
    if row is None:
        return None
    recorded = row["file_size"]
    if recorded is not None and int(recorded) != int(file_size):
        raise ResumeConflictError(
            "export file size changed "
            f"({recorded} -> {file_size}); start a new import without --resume"
        )
    return row
