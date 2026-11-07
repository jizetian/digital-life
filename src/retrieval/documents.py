"""Project segments, memories, and L2 traits into ``retrieval_documents``.

The projection is rebuildable. Source tables stay authoritative: a later
``sync_documents`` inserts new rows, updates changed text, and deletes rows
whose source disappeared.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field

from common.clock import utc_now
from retrieval.collections import COLLECTIONS

_PAGE = 200


@dataclass(slots=True)
class SyncStats:
    upserted: int = 0
    unchanged: int = 0
    deleted: int = 0
    by_collection: dict[str, int] = field(default_factory=dict)
    deleted_points: list[tuple[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class _Draft:
    doc_id: str
    collection: str
    source_id: str
    text: str
    text_hash: str
    content_hash: str
    time_start: int | None
    time_end: int | None
    metadata: dict


def point_id_for(doc_id: str) -> str:
    """Stable Qdrant point id. It can be recomputed after the SQLite row is gone."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "chattwin:" + doc_id))


def doc_id_for(collection: str, source_id: str) -> str:
    return f"{collection}:{source_id}"


def sync_documents(
    conn: sqlite3.Connection,
    collections: tuple[str, ...],
    *,
    max_chars: int,
) -> SyncStats:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    stats = SyncStats(by_collection={name: 0 for name in COLLECTIONS})
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS retrieval_seen (doc_id TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM retrieval_seen")
    for collection in collections:
        for draft in _drafts(conn, collection, max_chars=max_chars):
            conn.execute("INSERT INTO retrieval_seen (doc_id) VALUES (?)", (draft.doc_id,))
            _upsert_draft(conn, draft, stats)
    placeholders = ",".join("?" for _ in collections)
    doomed = conn.execute(
        f"""
        SELECT d.doc_id, d.collection, e.point_id
        FROM retrieval_documents AS d
        LEFT JOIN retrieval_embeddings AS e ON e.doc_id = d.doc_id
        WHERE d.collection IN ({placeholders})
          AND d.doc_id NOT IN (SELECT doc_id FROM retrieval_seen)
        """,
        collections,
    ).fetchall()
    for row in doomed:
        point_id = row["point_id"] or point_id_for(row["doc_id"])
        stats.deleted_points.append((row["collection"], point_id))
    if doomed:
        conn.execute(
            f"""
            DELETE FROM retrieval_documents
            WHERE collection IN ({placeholders})
              AND doc_id NOT IN (SELECT doc_id FROM retrieval_seen)
            """,
            collections,
        )
        stats.deleted = len(doomed)
    for row in conn.execute(
        f"""
        SELECT collection, COUNT(*) AS n
        FROM retrieval_documents
        WHERE collection IN ({placeholders})
        GROUP BY collection
        """,
        collections,
    ):
        stats.by_collection[row["collection"]] = int(row["n"])
    return stats


def clear_embedding_rows(conn: sqlite3.Connection, collections: tuple[str, ...]) -> None:
    placeholders = ",".join("?" for _ in collections)
    conn.execute(
        f"""
        DELETE FROM retrieval_embeddings
        WHERE doc_id IN (
            SELECT doc_id FROM retrieval_documents WHERE collection IN ({placeholders})
        )
        """,
        collections,
    )


def embedding_signature(conn: sqlite3.Connection) -> tuple[str, int] | None:
    row = conn.execute(
        """
        SELECT model_name, dimension
        FROM retrieval_embeddings
        ORDER BY model_name, dimension
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    return str(row["model_name"]), int(row["dimension"])


def pending_documents(
    conn: sqlite3.Connection,
    collections: tuple[str, ...],
    *,
    model_name: str,
    dimension: int,
    limit: int,
):
    if limit < 1:
        raise ValueError("limit must be positive")
    placeholders = ",".join("?" for _ in collections)
    return conn.execute(
        f"""
        SELECT d.doc_id, d.collection, d.source_id, d.text, d.text_hash,
               d.time_start, d.time_end, d.metadata_json
        FROM retrieval_documents AS d
        LEFT JOIN retrieval_embeddings AS e ON e.doc_id = d.doc_id
        WHERE d.collection IN ({placeholders})
          AND (
                e.doc_id IS NULL
             OR e.text_hash != d.text_hash
             OR e.model_name != ?
             OR e.dimension != ?
          )
        ORDER BY d.doc_id
        LIMIT ?
        """,
        (*collections, model_name, dimension, limit),
    ).fetchall()


def mark_embedded(
    conn: sqlite3.Connection,
    rows: list[tuple[str, str, str, int, str]],
    *,
    now: str,
) -> None:
    conn.executemany(
        """
        INSERT INTO retrieval_embeddings (
            doc_id, text_hash, model_name, dimension, point_id, embedded_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            text_hash = excluded.text_hash,
            model_name = excluded.model_name,
            dimension = excluded.dimension,
            point_id = excluded.point_id,
            embedded_at = excluded.embedded_at
        """,
        [(doc_id, text_hash, model_name, dimension, point_id, now) for doc_id, text_hash, model_name, dimension, point_id in rows],
    )


def payload_for(row) -> dict:
    """Filter fields stored in Qdrant. The searchable text stays in SQLite."""
    try:
        meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    except json.JSONDecodeError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    payload = {
        "doc_id": row["doc_id"],
        "collection": row["collection"],
        "source_id": row["source_id"],
    }
    for key in ("status", "memory_type", "dimension"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            payload[key] = value
    start = row["time_start"]
    end = row["time_end"]
    if start is None and end is not None:
        start = end
    if end is None and start is not None:
        end = start
    if start is not None:
        payload["time_start"] = int(start)
    if end is not None:
        payload["time_end"] = int(end)
    return payload


def _upsert_draft(conn: sqlite3.Connection, draft: _Draft, stats: SyncStats) -> None:
    existing = conn.execute(
        "SELECT content_hash FROM retrieval_documents WHERE doc_id = ?",
        (draft.doc_id,),
    ).fetchone()
    if existing is not None and existing["content_hash"] == draft.content_hash:
        stats.unchanged += 1
        return
    metadata = json.dumps(draft.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    now = utc_now()
    conn.execute(
        """
        INSERT INTO retrieval_documents (
            doc_id, collection, source_id, text, text_hash, content_hash,
            time_start, time_end, metadata_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            collection = excluded.collection,
            source_id = excluded.source_id,
            text = excluded.text,
            text_hash = excluded.text_hash,
            content_hash = excluded.content_hash,
            time_start = excluded.time_start,
            time_end = excluded.time_end,
            metadata_json = excluded.metadata_json,
            updated_at = excluded.updated_at
        """,
        (
            draft.doc_id,
            draft.collection,
            draft.source_id,
            draft.text,
            draft.text_hash,
            draft.content_hash,
            draft.time_start,
            draft.time_end,
            metadata,
            now,
        ),
    )
    stats.upserted += 1


def _drafts(conn: sqlite3.Connection, collection: str, *, max_chars: int):
    if collection == "conversation_segments":
        yield from _segment_drafts(conn, max_chars=max_chars)
    elif collection == "long_term_memories":
        yield from _memory_drafts(conn, max_chars=max_chars)
    elif collection == "personality_observations":
        yield from _trait_drafts(conn, max_chars=max_chars)
    else:
        raise ValueError(f"unknown collection: {collection}")


def _segment_drafts(conn: sqlite3.Connection, *, max_chars: int):
    last = ""
    while True:
        segments = conn.execute(
            """
            SELECT segment_id, conversation_id, start_time, end_time, message_count
            FROM conversation_segments
            WHERE segment_id > ?
            ORDER BY segment_id
            LIMIT ?
            """,
            (last, _PAGE),
        ).fetchall()
        if not segments:
            return
        ids = [row["segment_id"] for row in segments]
        messages: dict[str, list] = {segment_id: [] for segment_id in ids}
        placeholders = ",".join("?" for _ in ids)
        for row in conn.execute(
            f"""
            SELECT sm.segment_id, m.message_id, m.sender_role, m.timestamp, m.content
            FROM conversation_segment_messages AS sm
            JOIN messages AS m ON m.message_id = sm.message_id
            WHERE sm.segment_id IN ({placeholders})
            ORDER BY sm.segment_id, sm.position
            """,
            ids,
        ):
            messages[row["segment_id"]].append(row)
        for segment in segments:
            last = segment["segment_id"]
            lines = []
            message_ids = []
            for message in messages.get(segment["segment_id"], []):
                content = (message["content"] or "").strip()
                if not content:
                    continue
                message_ids.append(message["message_id"])
                role = message["sender_role"] or "unknown"
                stamp = message["timestamp"]
                prefix = f"[{role} {stamp}]" if stamp is not None else f"[{role}]"
                lines.append(f"{prefix} {content}")
            text = _cap("\n".join(lines), max_chars)
            if not text:
                continue
            metadata = {
                "conversation_id": segment["conversation_id"],
                "message_ids": message_ids,
                "message_count": int(segment["message_count"]),
            }
            yield _make_draft(
                "conversation_segments",
                segment["segment_id"],
                text,
                segment["start_time"],
                segment["end_time"],
                metadata,
            )


def _memory_drafts(conn: sqlite3.Connection, *, max_chars: int):
    last = ""
    while True:
        rows = conn.execute(
            """
            SELECT memory_id, memory_type, status, statement, situation,
                   importance, confidence, time_start, time_end, time_precision,
                   evidence_message_ids_json
            FROM memories
            WHERE memory_id > ?
            ORDER BY memory_id
            LIMIT ?
            """,
            (last, _PAGE),
        ).fetchall()
        if not rows:
            return
        ids = [row["memory_id"] for row in rows]
        names: dict[str, list[str]] = {memory_id: [] for memory_id in ids}
        placeholders = ",".join("?" for _ in ids)
        for row in conn.execute(
            f"""
            SELECT me.memory_id, e.canonical_name
            FROM memory_entities AS me
            JOIN entities AS e ON e.entity_id = me.entity_id
            WHERE me.memory_id IN ({placeholders})
            ORDER BY me.memory_id, e.canonical_name
            """,
            ids,
        ):
            names[row["memory_id"]].append(row["canonical_name"])
        for row in rows:
            last = row["memory_id"]
            statement = (row["statement"] or "").strip()
            if not statement:
                continue
            parts = [f"[{row['memory_type']}/{row['status']}] {statement}"]
            situation = (row["situation"] or "").strip()
            if situation:
                parts.append(f"situation: {situation}")
            entity_names = list(dict.fromkeys(names.get(row["memory_id"], [])))
            if entity_names:
                parts.append("entities: " + ", ".join(entity_names))
            text = _cap("\n".join(parts), max_chars)
            metadata = {
                "memory_type": row["memory_type"],
                "status": row["status"],
                "importance": row["importance"],
                "confidence": row["confidence"],
                "time_precision": row["time_precision"],
                "evidence_message_ids": _string_list(row["evidence_message_ids_json"]),
                "entity_names": entity_names,
            }
            if situation:
                metadata["situation"] = situation
            yield _make_draft(
                "long_term_memories",
                row["memory_id"],
                text,
                row["time_start"],
                row["time_end"],
                metadata,
            )


def _trait_drafts(conn: sqlite3.Connection, *, max_chars: int):
    last = ""
    while True:
        rows = conn.execute(
            """
            SELECT trait_id, dimension, status, statement, situational_scope,
                   confidence, evidence_count, counter_evidence_count,
                   evidence_ids_json, counter_evidence_ids_json,
                   valid_from, valid_to
            FROM l2_traits
            WHERE trait_id > ?
            ORDER BY trait_id
            LIMIT ?
            """,
            (last, _PAGE),
        ).fetchall()
        if not rows:
            return
        for row in rows:
            last = row["trait_id"]
            statement = (row["statement"] or "").strip()
            if not statement:
                continue
            parts = [f"[{row['dimension']}/{row['status']}] {statement}"]
            scope = (row["situational_scope"] or "").strip()
            if scope:
                parts.append(f"scope: {scope}")
            text = _cap("\n".join(parts), max_chars)
            metadata = {
                "dimension": row["dimension"],
                "status": row["status"],
                "confidence": row["confidence"],
                "evidence_count": int(row["evidence_count"]),
                "counter_evidence_count": int(row["counter_evidence_count"]),
                "evidence_ids": _string_list(row["evidence_ids_json"]),
                "counter_evidence_ids": _string_list(row["counter_evidence_ids_json"]),
            }
            if scope:
                metadata["situational_scope"] = scope
            yield _make_draft(
                "personality_observations",
                row["trait_id"],
                text,
                row["valid_from"],
                row["valid_to"],
                metadata,
            )


def _make_draft(collection, source_id, text, time_start, time_end, metadata: dict) -> _Draft:
    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    content_hash = hashlib.sha256(f"{text_hash}|{encoded}".encode("utf-8")).hexdigest()
    return _Draft(
        doc_id=doc_id_for(collection, source_id),
        collection=collection,
        source_id=source_id,
        text=text,
        text_hash=text_hash,
        content_hash=content_hash,
        time_start=time_start,
        time_end=time_end,
        metadata=metadata,
    )


def _cap(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    newline = clipped.rfind("\n")
    if newline >= max_chars // 2:
        return clipped[:newline]
    return clipped


def _string_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None and str(item) != ""]
