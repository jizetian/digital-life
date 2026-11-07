"""Insert synthetic messages into a ChatTwin SQLite file."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from common.database import init_db


def seed_messages(db_path: Path, items: list[dict]) -> None:
    conn = init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO import_jobs (
            job_id, source_file, source_path, status, created_at, updated_at
        ) VALUES ('seed', 'seed.json', 'seed.json', 'completed', ?, ?)
        """,
        (now, now),
    )
    for index, item in enumerate(items):
        conn.execute(
            """
            INSERT INTO messages (
                message_id, conversation_id, sender_role, timestamp, message_type,
                content, raw_content, is_forwarded, source_file, source_position,
                job_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, '{}', 0, 'seed.json', ?, 'seed', ?)
            """,
            (
                item["message_id"],
                item.get("conversation_id", "conv"),
                item.get("sender_role", "owner"),
                item.get("timestamp"),
                item.get("message_type", "text"),
                item.get("content", "hello"),
                item.get("source_position", index),
                now,
            ),
        )
    conn.commit()
    conn.close()


def insert_message(db_path: Path, item: dict) -> None:
    conn = init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO messages (
            message_id, conversation_id, sender_role, timestamp, message_type,
            content, raw_content, is_forwarded, source_file, source_position,
            job_id, created_at
        ) VALUES (?, ?, 'owner', ?, 'text', ?, '{}', 0, 'seed.json', ?, 'seed', ?)
        """,
        (
            item["message_id"],
            item.get("conversation_id", "conv"),
            item.get("timestamp"),
            item.get("content", "hello"),
            item["source_position"],
            now,
        ),
    )
    conn.commit()
    conn.close()
