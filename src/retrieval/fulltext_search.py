"""SQLite FTS5 search over ``retrieval_documents``.

The index uses the trigram tokenizer so a Chinese or English substring of at
least three characters can match. Shorter queries fall back to a case-folded
substring scan. This module only reads the projection; it does not rank vectors.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass



@dataclass(slots=True)
class FtsHit:
    doc_id: str
    score: float


def search_fulltext(
    conn: sqlite3.Connection,
    query: str,
    collections: tuple[str, ...],
    limit: int,
    *,
    memory_status: str | None = None,
    memory_type: str | None = None,
    trait_status: str | None = None,
    dimension: str | None = None,
    time_start: int | None = None,
    time_end: int | None = None,
) -> list[FtsHit]:
    if limit < 1 or not collections:
        return []
    cleaned = " ".join(query.split())
    if not cleaned:
        return []
    where, params = _filters(
        collections,
        memory_status=memory_status,
        memory_type=memory_type,
        trait_status=trait_status,
        dimension=dimension,
        time_start=time_start,
        time_end=time_end,
    )
    match = _match_query(cleaned)
    if match is not None:
        try:
            rows = conn.execute(
                f"""
                SELECT d.doc_id AS doc_id, bm25(retrieval_fts) AS score
                FROM retrieval_fts
                JOIN retrieval_documents AS d ON d.rowid = retrieval_fts.rowid
                WHERE retrieval_fts MATCH ?
                  AND {where}
                ORDER BY score
                LIMIT ?
                """,
                (match, *params, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return _like_hits(_like_rows(conn, cleaned, where, params, limit))
        return [FtsHit(doc_id=row["doc_id"], score=-float(row["score"])) for row in rows]
    return _like_hits(_like_rows(conn, cleaned, where, params, limit))


def _like_hits(rows) -> list[FtsHit]:
    # Best row first. The score only breaks ties inside this list; RRF uses rank.
    total = len(rows)
    return [FtsHit(doc_id=row["doc_id"], score=float(total - index)) for index, row in enumerate(rows)]


def _like_rows(conn, query: str, where: str, params: list, limit: int):
    # ``instr`` plus ``lower`` covers ASCII. Chinese is unchanged by lower().
    return conn.execute(
        f"""
        SELECT doc_id
        FROM retrieval_documents AS d
        WHERE instr(lower(d.text), lower(?)) > 0
          AND {where}
        ORDER BY d.doc_id
        LIMIT ?
        """,
        (query, *params, limit),
    ).fetchall()


def _match_query(query: str) -> str | None:
    if len(query) < 3:
        return None
    return '"' + query.replace('"', '""') + '"'


def _filters(
    collections: tuple[str, ...],
    *,
    memory_status: str | None,
    memory_type: str | None,
    trait_status: str | None,
    dimension: str | None,
    time_start: int | None,
    time_end: int | None,
) -> tuple[str, list]:
    placeholders = ",".join("?" for _ in collections)
    clauses = [f"d.collection IN ({placeholders})"]
    params: list = list(collections)
    if memory_status:
        clauses.append(
            "(d.collection != 'long_term_memories' OR json_extract(d.metadata_json, '$.status') = ?)"
        )
        params.append(memory_status)
    if memory_type:
        clauses.append(
            "(d.collection != 'long_term_memories' OR json_extract(d.metadata_json, '$.memory_type') = ?)"
        )
        params.append(memory_type)
    if trait_status:
        clauses.append(
            "(d.collection != 'personality_observations' OR json_extract(d.metadata_json, '$.status') = ?)"
        )
        params.append(trait_status)
    if dimension:
        clauses.append(
            "(d.collection != 'personality_observations' OR json_extract(d.metadata_json, '$.dimension') = ?)"
        )
        params.append(dimension)
    if time_start is not None or time_end is not None:
        clauses.append("NOT (d.time_start IS NULL AND d.time_end IS NULL)")
    if time_start is not None:
        clauses.append("COALESCE(d.time_end, d.time_start) >= ?")
        params.append(time_start)
    if time_end is not None:
        clauses.append("COALESCE(d.time_start, d.time_end) <= ?")
        params.append(time_end)
    return " AND ".join(clauses), params
