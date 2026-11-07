"""Keep memory evidence pointed at real messages and L1 observations.

The denormalized ``evidence_message_ids_json`` column lists support messages
only. Conflict notes live in ``memory_evidence`` with role ``conflict`` and
are not copied into that column.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

_CHUNK = 400


def filter_ids(values, allowed: set[str]) -> list[str]:
    """Keep unique ids that the caller already trusts, in first-seen order."""
    if not isinstance(values, list):
        return []
    kept: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        item = str(value).strip()
        if not item or item not in allowed or item in seen:
            continue
        seen.add(item)
        kept.append(item)
    return kept


def existing_message_ids(conn: sqlite3.Connection, message_ids: list[str]) -> list[str]:
    return _existing(conn, "messages", "message_id", message_ids)


def existing_observation_ids(conn: sqlite3.Connection, observation_ids: list[str]) -> list[str]:
    return _existing(conn, "l1_observations", "observation_id", observation_ids)


def add_evidence(
    conn: sqlite3.Connection,
    *,
    memory_id: str,
    segment_id: str | None,
    message_ids: list[str],
    observation_ids: list[str],
    role: str,
    created_at: str,
) -> None:
    """Insert evidence rows. Support rows refresh the denormalized id lists."""
    for message_id in message_ids:
        _insert_row(
            conn,
            memory_id=memory_id,
            message_id=message_id,
            observation_id=None,
            segment_id=segment_id,
            role=role,
            created_at=created_at,
        )
    for observation_id in observation_ids:
        _insert_row(
            conn,
            memory_id=memory_id,
            message_id=None,
            observation_id=observation_id,
            segment_id=segment_id,
            role=role,
            created_at=created_at,
        )
    if role == "support":
        sync_denormalized(conn, memory_id)


def sync_denormalized(conn: sqlite3.Connection, memory_id: str) -> None:
    messages = [
        row["message_id"]
        for row in conn.execute(
            """
            SELECT DISTINCT message_id FROM memory_evidence
            WHERE memory_id = ? AND evidence_role = 'support' AND message_id IS NOT NULL
            ORDER BY message_id
            """,
            (memory_id,),
        )
    ]
    observations = [
        row["observation_id"]
        for row in conn.execute(
            """
            SELECT DISTINCT observation_id FROM memory_evidence
            WHERE memory_id = ? AND evidence_role = 'support' AND observation_id IS NOT NULL
            ORDER BY observation_id
            """,
            (memory_id,),
        )
    ]
    conn.execute(
        """
        UPDATE memories
        SET evidence_message_ids_json = ?, source_observation_ids_json = ?
        WHERE memory_id = ?
        """,
        (
            json.dumps(messages, ensure_ascii=False),
            json.dumps(observations, ensure_ascii=False),
            memory_id,
        ),
    )


def _insert_row(
    conn: sqlite3.Connection,
    *,
    memory_id: str,
    message_id: str | None,
    observation_id: str | None,
    segment_id: str | None,
    role: str,
    created_at: str,
) -> None:
    if message_id is None and observation_id is None:
        return
    raw = f"{memory_id}|{role}|{message_id or ''}|{observation_id or ''}|{segment_id or ''}"
    evidence_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    conn.execute(
        """
        INSERT OR IGNORE INTO memory_evidence (
            evidence_id, memory_id, message_id, observation_id, segment_id,
            evidence_role, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (evidence_id, memory_id, message_id, observation_id, segment_id, role, created_at),
    )


def _existing(conn: sqlite3.Connection, table: str, column: str, ids: list[str]) -> list[str]:
    if table not in {"messages", "l1_observations"} or column not in {"message_id", "observation_id"}:
        raise ValueError("unsupported evidence lookup")
    if not ids:
        return []
    found: set[str] = set()
    for offset in range(0, len(ids), _CHUNK):
        chunk = ids[offset : offset + _CHUNK]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT {column} AS item FROM {table} WHERE {column} IN ({placeholders})",
            chunk,
        ).fetchall()
        found.update(row["item"] for row in rows)
    return [item for item in ids if item in found]
