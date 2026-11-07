"""Order stored memories on a timeline and expose the FTS5 lexical hook.

``search_memory_ids`` matches statement text only. It is not hybrid search
and it does not rank by embeddings. Phase 5 owns retrieval.
"""

from __future__ import annotations

import sqlite3

from memory.types import MEMORY_STATUSES, MEMORY_TYPES


def build_timeline(
    conn: sqlite3.Connection,
    *,
    entity: str | None = None,
    memory_type: str | None = None,
    status: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Return memories ordered by start time. Rows without a time go last.

    Historical and superseded rows stay on the timeline. ``status`` filters
    the view; it does not delete anything.
    """
    if memory_type is not None and memory_type not in MEMORY_TYPES:
        raise ValueError(f"unknown memory_type: {memory_type}")
    if status is not None and status not in MEMORY_STATUSES:
        raise ValueError(f"unknown memory status: {status}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    clauses = ["1 = 1"]
    params: list = []
    if memory_type is not None:
        clauses.append("m.memory_type = ?")
        params.append(memory_type)
    if status is not None:
        clauses.append("m.status = ?")
        params.append(status)
    if entity:
        clauses.append(
            """
            m.memory_id IN (
                SELECT me.memory_id
                FROM memory_entities me
                JOIN entities e ON e.entity_id = me.entity_id
                WHERE e.entity_id = ? OR e.name_key = ? OR e.canonical_name = ?
            )
            """
        )
        key = " ".join(entity.strip().casefold().split())
        params.extend([entity, key, entity.strip()])
    sql = f"""
        SELECT m.*
        FROM memories m
        WHERE {' AND '.join(clauses)}
        ORDER BY m.time_start IS NULL, m.time_start, m.importance IS NULL, m.importance DESC, m.memory_id
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return []
    memory_ids = [row["memory_id"] for row in rows]
    entities = _entities_for(conn, memory_ids)
    relations = _relations_for(conn, memory_ids)
    items: list[dict] = []
    for row in rows:
        memory_id = row["memory_id"]
        items.append(
            {
                "memory_id": memory_id,
                "memory_type": row["memory_type"],
                "status": row["status"],
                "statement": row["statement"],
                "situation": row["situation"],
                "importance": row["importance"],
                "confidence": row["confidence"],
                "time_start": row["time_start"],
                "time_end": row["time_end"],
                "time_precision": row["time_precision"],
                "evidence_message_ids": _loads_ids(row["evidence_message_ids_json"]),
                "observation_ids": _loads_ids(row["source_observation_ids_json"]),
                "superseded_by": row["superseded_by"],
                "entities": entities.get(memory_id, []),
                "relations": relations.get(memory_id, []),
            }
        )
    return items


def render_timeline(items: list[dict]) -> str:
    lines = ["=== ChatTwin 时间线 / timeline ===", f"count: {len(items)}"]
    if not items:
        lines.append("(none)")
        return "\n".join(lines)
    for item in items:
        start = _stamp(item.get("time_start"))
        end = _stamp(item.get("time_end"))
        lines.append(
            f"{start}..{end}  [{item['memory_type']}/{item['status']}] {item['statement']}"
        )
        if item.get("situation"):
            lines.append(f"    situation: {item['situation']}")
        entity_bits = [
            f"{entity['name']} ({entity['entity_type']}, {entity['role']}, {entity['status']})"
            for entity in item.get("entities") or []
        ]
        if entity_bits:
            lines.append("    entities: " + ", ".join(entity_bits))
        evidence = item.get("evidence_message_ids") or []
        if evidence:
            lines.append("    evidence: " + ",".join(evidence))
        for relation in item.get("relations") or []:
            lines.append(
                f"    relation: {relation['relation_type']} {relation['direction']} {relation['other_memory_id']}"
            )
    return "\n".join(lines)


def search_memory_ids(conn: sqlite3.Connection, query: str, *, limit: int = 20) -> list[str]:
    """Lexical lookup over ``memories_fts``. An empty or invalid query matches nothing."""
    phrase = _fts_phrase(query)
    if not phrase:
        return []
    if limit < 1:
        raise ValueError("limit must be positive")
    try:
        rows = conn.execute(
            """
            SELECT m.memory_id
            FROM memories_fts
            JOIN memories m ON m.rowid = memories_fts.rowid
            WHERE memories_fts MATCH ?
            LIMIT ?
            """,
            (phrase, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [row["memory_id"] for row in rows]


def _entities_for(conn: sqlite3.Connection, memory_ids: list[str]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {memory_id: [] for memory_id in memory_ids}
    for chunk in _chunks(memory_ids):
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT me.memory_id, me.role, e.entity_id, e.canonical_name, e.entity_type, e.status
            FROM memory_entities me
            JOIN entities e ON e.entity_id = me.entity_id
            WHERE me.memory_id IN ({placeholders})
            ORDER BY me.memory_id, me.role, e.canonical_name
            """,
            chunk,
        ).fetchall()
        for row in rows:
            grouped[row["memory_id"]].append(
                {
                    "entity_id": row["entity_id"],
                    "name": row["canonical_name"],
                    "entity_type": row["entity_type"],
                    "role": row["role"],
                    "status": row["status"],
                }
            )
    return grouped


def _relations_for(conn: sqlite3.Connection, memory_ids: list[str]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {memory_id: [] for memory_id in memory_ids}
    wanted = set(memory_ids)
    for chunk in _chunks(memory_ids):
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT relation_type, from_memory_id, to_memory_id, note
            FROM memory_relations
            WHERE from_memory_id IN ({placeholders}) OR to_memory_id IN ({placeholders})
            ORDER BY relation_type, from_memory_id, to_memory_id
            """,
            [*chunk, *chunk],
        ).fetchall()
        for row in rows:
            if row["from_memory_id"] in wanted:
                grouped[row["from_memory_id"]].append(
                    {
                        "relation_type": row["relation_type"],
                        "direction": "to",
                        "other_memory_id": row["to_memory_id"],
                        "note": row["note"],
                    }
                )
            if row["to_memory_id"] in wanted:
                grouped[row["to_memory_id"]].append(
                    {
                        "relation_type": row["relation_type"],
                        "direction": "from",
                        "other_memory_id": row["from_memory_id"],
                        "note": row["note"],
                    }
                )
    return grouped


def _chunks(ids: list[str], size: int = 400):
    for offset in range(0, len(ids), size):
        yield ids[offset : offset + size]


def _loads_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    import json

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def _stamp(value) -> str:
    return "" if value is None else str(value)


def _fts_phrase(query: str) -> str:
    cleaned = " ".join(query.split())
    if not cleaned:
        return ""
    return '"' + cleaned.replace('"', '""') + '"'
