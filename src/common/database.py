"""SQLite schema for messages, import jobs, and sender mapping."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS import_jobs (
    job_id TEXT PRIMARY KEY,
    source_file TEXT NOT NULL,
    source_path TEXT NOT NULL,
    file_size INTEGER,
    status TEXT NOT NULL,
    batch_size INTEGER,
    processed_count INTEGER NOT NULL DEFAULT 0,
    last_position INTEGER,
    total_messages INTEGER,
    success_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    duplicate_count INTEGER NOT NULL DEFAULT 0,
    stats_json TEXT,
    metadata_json TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_import_jobs_source
    ON import_jobs(source_path, status);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    local_id INTEGER,
    server_id TEXT,
    shard INTEGER,
    sender_id TEXT,
    sender_name TEXT,
    sender_role TEXT NOT NULL,
    is_sender INTEGER,
    timestamp INTEGER,
    time_str TEXT,
    message_type TEXT,
    type_code INTEGER,
    raw_type INTEGER,
    type_name TEXT,
    content TEXT,
    raw_content TEXT NOT NULL,
    is_forwarded INTEGER NOT NULL DEFAULT 0,
    resource_status TEXT,
    resource_notice TEXT,
    source_file TEXT NOT NULL,
    source_position INTEGER NOT NULL,
    job_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES import_jobs(job_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_sender
    ON messages(sender_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp
    ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_type
    ON messages(message_type);
CREATE INDEX IF NOT EXISTS idx_messages_role
    ON messages(sender_role);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_time
    ON messages(conversation_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_source
    ON messages(source_file, source_position);
CREATE INDEX IF NOT EXISTS idx_messages_server
    ON messages(server_id);
CREATE INDEX IF NOT EXISTS idx_messages_job
    ON messages(job_id);

CREATE TABLE IF NOT EXISTS import_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    source_file TEXT,
    source_position INTEGER,
    message_id TEXT,
    error_type TEXT,
    error_message TEXT NOT NULL,
    raw_snippet TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES import_jobs(job_id)
);

CREATE INDEX IF NOT EXISTS idx_import_errors_job
    ON import_errors(job_id);

CREATE TABLE IF NOT EXISTS sender_mapping (
    conversation_id TEXT NOT NULL,
    sender_id TEXT NOT NULL,
    sender_name TEXT,
    role TEXT NOT NULL,
    is_sender INTEGER,
    from_metadata INTEGER NOT NULL DEFAULT 0,
    from_message INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (conversation_id, sender_id, role)
);

CREATE TABLE IF NOT EXISTS processing_jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT,
    status TEXT,
    payload_json TEXT,
    created_at TEXT,
    updated_at TEXT
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db(db_path: str | Path) -> sqlite3.Connection:
    conn = connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def upsert_sender_mapping(
    conn: sqlite3.Connection,
    *,
    conversation_id: str,
    sender_id: str,
    sender_name: str | None,
    role: str,
    is_sender: int | None,
    from_metadata: int,
    from_message: int,
    now: str,
) -> None:
    conn.execute(
        """
        INSERT INTO sender_mapping (
            conversation_id, sender_id, sender_name, role, is_sender,
            from_metadata, from_message, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id, sender_id, role) DO UPDATE SET
            sender_name = COALESCE(excluded.sender_name, sender_mapping.sender_name),
            is_sender = COALESCE(excluded.is_sender, sender_mapping.is_sender),
            from_metadata = MAX(sender_mapping.from_metadata, excluded.from_metadata),
            from_message = MAX(sender_mapping.from_message, excluded.from_message),
            updated_at = excluded.updated_at
        """,
        (
            conversation_id,
            sender_id,
            sender_name,
            role,
            is_sender,
            from_metadata,
            from_message,
            now,
            now,
        ),
    )
