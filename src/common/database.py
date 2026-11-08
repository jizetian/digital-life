"""SQLite schema for messages, segments, L1 observations, personality, long-term memory, and retrieval."""

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

CREATE TABLE IF NOT EXISTS personality_tasks (
    task_id TEXT PRIMARY KEY,
    task_kind TEXT NOT NULL,
    dimension TEXT NOT NULL,
    batch_key TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    source_ids_json TEXT NOT NULL,
    result_json TEXT,
    result_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'retrying', 'skipped')),
    CHECK (task_kind IN ('merge', 'reduce', 'contradict')),
    CHECK (active IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_personality_tasks_claim
    ON personality_tasks(prompt_version, model_version, active, status);

CREATE TABLE IF NOT EXISTS l2_traits (
    trait_id TEXT PRIMARY KEY,
    prompt_version TEXT NOT NULL,
    model_version TEXT NOT NULL,
    dimension TEXT NOT NULL,
    statement TEXT NOT NULL,
    situational_scope TEXT NOT NULL,
    confidence REAL,
    evidence_count INTEGER NOT NULL,
    independent_evidence_count INTEGER NOT NULL,
    counter_evidence_count INTEGER NOT NULL,
    evidence_ids_json TEXT NOT NULL,
    counter_evidence_ids_json TEXT NOT NULL,
    valid_from INTEGER,
    valid_to INTEGER,
    status TEXT NOT NULL,
    stages_json TEXT NOT NULL,
    variants_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (status IN ('stable', 'provisional', 'contested'))
);

CREATE INDEX IF NOT EXISTS idx_l2_traits_version
    ON l2_traits(prompt_version, model_version, dimension, status);

CREATE TABLE IF NOT EXISTS personality_checkpoint (
    checkpoint_id TEXT PRIMARY KEY,
    prompt_version TEXT NOT NULL,
    model_version TEXT NOT NULL,
    observation_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    output_dir TEXT,
    profile_written_at TEXT,
    min_independent INTEGER NOT NULL,
    traits_stable INTEGER NOT NULL DEFAULT 0,
    traits_provisional INTEGER NOT NULL DEFAULT 0,
    traits_contested INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    status TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    description TEXT,
    confidence REAL,
    first_seen INTEGER,
    last_seen INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (entity_type IN ('person', 'place', 'organization', 'object', 'concept', 'other')),
    CHECK (status IN ('confirmed', 'inferred', 'unknown')),
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_type_key
    ON entities(entity_type, name_key);

CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    status TEXT NOT NULL,
    statement TEXT NOT NULL,
    situation TEXT,
    importance REAL,
    confidence REAL,
    time_start INTEGER,
    time_end INTEGER,
    time_precision TEXT NOT NULL DEFAULT 'unknown',
    evidence_message_ids_json TEXT NOT NULL,
    source_observation_ids_json TEXT NOT NULL DEFAULT '[]',
    origin_segment_id TEXT,
    prompt_version TEXT NOT NULL,
    model_version TEXT NOT NULL,
    superseded_by TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (memory_type IN ('fact', 'event', 'relationship', 'opinion', 'situational')),
    CHECK (status IN ('active', 'historical', 'superseded', 'uncertain')),
    CHECK (time_precision IN ('instant', 'day', 'range', 'unknown')),
    CHECK (importance IS NULL OR (importance >= 0 AND importance <= 1)),
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    FOREIGN KEY (origin_segment_id) REFERENCES conversation_segments(segment_id) ON DELETE SET NULL,
    FOREIGN KEY (superseded_by) REFERENCES memories(memory_id)
);

CREATE INDEX IF NOT EXISTS idx_memories_time
    ON memories(time_start, status);
CREATE INDEX IF NOT EXISTS idx_memories_type_status
    ON memories(memory_type, status);
CREATE INDEX IF NOT EXISTS idx_memories_version
    ON memories(prompt_version, model_version, status);

CREATE TABLE IF NOT EXISTS memory_evidence (
    evidence_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    message_id TEXT,
    observation_id TEXT,
    segment_id TEXT,
    evidence_role TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK (evidence_role IN ('support', 'conflict', 'context')),
    FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
    FOREIGN KEY (message_id) REFERENCES messages(message_id),
    FOREIGN KEY (observation_id) REFERENCES l1_observations(observation_id) ON DELETE CASCADE,
    FOREIGN KEY (segment_id) REFERENCES conversation_segments(segment_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_evidence_memory
    ON memory_evidence(memory_id, evidence_role);
CREATE INDEX IF NOT EXISTS idx_memory_evidence_message
    ON memory_evidence(message_id);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY (memory_id, entity_id, role),
    CHECK (role IN ('subject', 'object', 'participant', 'location', 'mentioned')),
    FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_entities_entity
    ON memory_entities(entity_id);

CREATE TABLE IF NOT EXISTS memory_relations (
    relation_id TEXT PRIMARY KEY,
    relation_type TEXT NOT NULL,
    from_memory_id TEXT NOT NULL,
    to_memory_id TEXT NOT NULL,
    confidence REAL,
    note TEXT,
    created_at TEXT NOT NULL,
    CHECK (relation_type IN ('supersedes', 'historical_of', 'conflicts_with', 'supports', 'related_to')),
    CHECK (from_memory_id != to_memory_id),
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    FOREIGN KEY (from_memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
    FOREIGN KEY (to_memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_relations_unique
    ON memory_relations(relation_type, from_memory_id, to_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_relations_to
    ON memory_relations(to_memory_id, relation_type);

CREATE TABLE IF NOT EXISTS memory_tasks (
    task_id TEXT PRIMARY KEY,
    segment_id TEXT NOT NULL,
    task_kind TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    input_digest TEXT,
    result_json TEXT,
    result_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'retrying', 'skipped')),
    CHECK (task_kind IN ('extract', 'merge')),
    FOREIGN KEY (segment_id) REFERENCES conversation_segments(segment_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_task_identity
    ON memory_tasks(segment_id, task_kind, prompt_version, model_version);
CREATE INDEX IF NOT EXISTS idx_memory_tasks_claim
    ON memory_tasks(prompt_version, model_version, status, task_kind);

CREATE TABLE IF NOT EXISTS memory_checkpoint (
    checkpoint_id TEXT PRIMARY KEY,
    prompt_version TEXT NOT NULL,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL,
    extracts_completed INTEGER NOT NULL DEFAULT 0,
    merges_completed INTEGER NOT NULL DEFAULT 0,
    memories_written INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    statement,
    content='memories',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, statement) VALUES (new.rowid, new.statement);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, statement) VALUES ('delete', old.rowid, old.statement);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, statement) VALUES ('delete', old.rowid, old.statement);
    INSERT INTO memories_fts(rowid, statement) VALUES (new.rowid, new.statement);
END;

-- Rebuildable projection for hybrid search. The source rows stay in
-- conversation_segments, memories, and l2_traits. Qdrant can be deleted
-- and rebuilt from retrieval_documents plus retrieval_embeddings.
CREATE TABLE IF NOT EXISTS retrieval_documents (
    doc_id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    source_id TEXT NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    time_start INTEGER,
    time_end INTEGER,
    metadata_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (collection IN (
        'conversation_segments',
        'long_term_memories',
        'personality_observations'
    ))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_retrieval_source
    ON retrieval_documents(collection, source_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_collection
    ON retrieval_documents(collection);

CREATE TABLE IF NOT EXISTS retrieval_embeddings (
    doc_id TEXT PRIMARY KEY,
    text_hash TEXT NOT NULL,
    model_name TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    point_id TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES retrieval_documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_retrieval_embeddings_model
    ON retrieval_embeddings(model_name, dimension);

-- Trigram tokens match Chinese and English substrings of at least 3 characters.
CREATE VIRTUAL TABLE IF NOT EXISTS retrieval_fts USING fts5(
    text,
    content='retrieval_documents',
    content_rowid='rowid',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS retrieval_fts_ai AFTER INSERT ON retrieval_documents BEGIN
    INSERT INTO retrieval_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS retrieval_fts_ad AFTER DELETE ON retrieval_documents BEGIN
    INSERT INTO retrieval_fts(retrieval_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER IF NOT EXISTS retrieval_fts_au AFTER UPDATE OF text ON retrieval_documents BEGIN
    INSERT INTO retrieval_fts(retrieval_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO retrieval_fts(rowid, text) VALUES (new.rowid, new.text);
END;

-- Digital-twin chat. Sessions stay in this SQLite file and are not source data.
-- Evidence ids are copied from a retrieval hit; they are not a foreign key, so a
-- later segment rebuild does not delete the conversation.
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT,
    target_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    retrieved_memories_json TEXT,
    evidence_message_ids_json TEXT,
    context_hits_json TEXT,
    confidence REAL,
    uncertainty TEXT,
    retrieval_weak INTEGER,
    model TEXT,
    created_at TEXT NOT NULL,
    CHECK (role IN ('user', 'assistant')),
    CHECK (retrieval_weak IS NULL OR retrieval_weak IN (0, 1)),
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages(session_id, created_at);
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
