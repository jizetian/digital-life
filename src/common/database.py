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
CREATE INDEX IF NOT EXISTS idx_messages_seg_order
    ON messages(COALESCE(conversation_id, ''), source_position, message_id);

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

CREATE TABLE IF NOT EXISTS conversation_segments (
    segment_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    params_hash TEXT NOT NULL,
    start_time INTEGER,
    end_time INTEGER,
    start_message_id TEXT NOT NULL,
    end_message_id TEXT NOT NULL,
    message_ids_json TEXT NOT NULL,
    overlap_message_ids_json TEXT NOT NULL,
    message_count INTEGER NOT NULL,
    target_message_count INTEGER NOT NULL,
    processing_status TEXT NOT NULL,
    opened_by TEXT NOT NULL,
    closed_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (processing_status IN ('pending', 'running', 'completed', 'failed', 'retrying', 'skipped')),
    CHECK (target_message_count >= 1),
    CHECK (message_count >= target_message_count)
);

CREATE INDEX IF NOT EXISTS idx_segments_params
    ON conversation_segments(params_hash, processing_status);
CREATE INDEX IF NOT EXISTS idx_segments_conversation
    ON conversation_segments(conversation_id, start_time);

CREATE TABLE IF NOT EXISTS conversation_segment_messages (
    segment_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    is_overlap INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (segment_id, message_id),
    FOREIGN KEY (segment_id) REFERENCES conversation_segments(segment_id) ON DELETE CASCADE,
    FOREIGN KEY (message_id) REFERENCES messages(message_id)
);

CREATE INDEX IF NOT EXISTS idx_segment_messages_message
    ON conversation_segment_messages(message_id);

CREATE TABLE IF NOT EXISTS distillation_tasks (
    task_id TEXT PRIMARY KEY,
    segment_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    observations_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'retrying', 'skipped')),
    FOREIGN KEY (segment_id) REFERENCES conversation_segments(segment_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_distill_identity
    ON distillation_tasks(segment_id, prompt_version, model_version);
CREATE INDEX IF NOT EXISTS idx_distill_status
    ON distillation_tasks(status, prompt_version, model_version);

CREATE TABLE IF NOT EXISTS l1_observations (
    observation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    category TEXT NOT NULL,
    summary TEXT NOT NULL,
    evidence_message_ids_json TEXT NOT NULL,
    confidence REAL,
    time_range_start INTEGER,
    time_range_end INTEGER,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK (category IN (
        'communication', 'emotion', 'opinion', 'value', 'thinking',
        'decision', 'social', 'interest', 'event', 'fact'
    )),
    FOREIGN KEY (task_id) REFERENCES distillation_tasks(task_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_l1_task ON l1_observations(task_id);
CREATE INDEX IF NOT EXISTS idx_l1_segment ON l1_observations(segment_id);
CREATE INDEX IF NOT EXISTS idx_l1_category ON l1_observations(category);
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
