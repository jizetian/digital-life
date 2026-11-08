"""Chat sessions in SQLite. The rows are derived dialogue, not the Moect export."""

from __future__ import annotations

import json
import sqlite3
import uuid

from common.clock import utc_now


def new_id() -> str:
    return uuid.uuid4().hex


def ensure_session(
    conn: sqlite3.Connection,
    session_id: str | None,
    *,
    target_name: str,
) -> sqlite3.Row:
    if session_id:
        row = conn.execute(
            "SELECT * FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return row
    now = utc_now()
    created = new_id()
    conn.execute(
        """
        INSERT INTO chat_sessions (session_id, title, target_name, created_at, updated_at)
        VALUES (?, NULL, ?, ?, ?)
        """,
        (created, target_name, now, now),
    )
    conn.commit()
    return conn.execute("SELECT * FROM chat_sessions WHERE session_id = ?", (created,)).fetchone()


def recent_turns(conn: sqlite3.Connection, session_id: str, limit: int) -> list[dict]:
    if limit < 1:
        return []
    rows = conn.execute(
        """
        SELECT role, content
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY created_at DESC, rowid DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    turns = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
    return turns


def add_user_message(conn: sqlite3.Connection, session_id: str, content: str) -> str:
    message_id = new_id()
    now = utc_now()
    conn.execute(
        """
        INSERT INTO chat_messages (
            message_id, session_id, role, content, created_at
        ) VALUES (?, ?, 'user', ?, ?)
        """,
        (message_id, session_id, content, now),
    )
    title = conn.execute("SELECT title FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone()
    if title is not None and not (title["title"] or "").strip():
        conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
            (_title_from(content), now, session_id),
        )
    else:
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )
    conn.commit()
    return message_id


def add_assistant_message(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    content: str,
    retrieved_memories: list[dict],
    evidence_message_ids: list[str],
    context_hits: list[dict],
    confidence: float,
    uncertainty: str | None,
    retrieval_weak: bool,
    model: str,
) -> str:
    message_id = new_id()
    now = utc_now()
    conn.execute(
        """
        INSERT INTO chat_messages (
            message_id, session_id, role, content,
            retrieved_memories_json, evidence_message_ids_json, context_hits_json,
            confidence, uncertainty, retrieval_weak, model, created_at
        ) VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message_id,
            session_id,
            content,
            _dump(retrieved_memories),
            _dump(evidence_message_ids),
            _dump(context_hits),
            confidence,
            uncertainty,
            1 if retrieval_weak else 0,
            model or None,
            now,
        ),
    )
    conn.execute(
        "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
        (now, session_id),
    )
    conn.commit()
    return message_id


def list_sessions(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    rows = conn.execute(
        """
        SELECT s.session_id, s.title, s.target_name, s.created_at, s.updated_at,
               COUNT(m.message_id) AS message_count
        FROM chat_sessions AS s
        LEFT JOIN chat_messages AS m ON m.session_id = s.session_id
        GROUP BY s.session_id
        ORDER BY s.updated_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "session_id": row["session_id"],
            "title": row["title"] or "新对话",
            "target_name": row["target_name"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": int(row["message_count"]),
        }
        for row in rows
    ]


def get_session(conn: sqlite3.Connection, session_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM chat_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    messages = [
        _message_dict(item)
        for item in conn.execute(
            """
            SELECT * FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at, rowid
            """,
            (session_id,),
        )
    ]
    return {
        "session_id": row["session_id"],
        "title": row["title"] or "新对话",
        "target_name": row["target_name"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "messages": messages,
    }


def _message_dict(row) -> dict:
    return {
        "message_id": row["message_id"],
        "role": row["role"],
        "content": row["content"],
        "retrieved_memories": _load(row["retrieved_memories_json"], []),
        "evidence_message_ids": _load(row["evidence_message_ids_json"], []),
        "context_hits": _load(row["context_hits_json"], []),
        "confidence": row["confidence"],
        "uncertainty": row["uncertainty"],
        "retrieval_weak": bool(row["retrieval_weak"]) if row["retrieval_weak"] is not None else None,
        "model": row["model"],
        "created_at": row["created_at"],
    }


def _title_from(message: str) -> str:
    flat = " ".join(message.split())
    if not flat:
        return "新对话"
    if len(flat) <= 24:
        return flat
    return flat[:24] + "…"


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load(raw, default):
    if not raw:
        return default
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return default
    if default == [] and not isinstance(value, list):
        return []
    return value
