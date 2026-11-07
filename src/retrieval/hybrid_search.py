"""Merge dense and full-text hits, then optionally rerank.

Reciprocal rank fusion scores a document by ``1 / (k + rank)`` in each list
and adds those contributions. The same ``doc_id`` is returned once. SQLite is
read again before the hit is shown, so a vector that no longer has a document
is dropped.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from retrieval.fulltext_search import search_fulltext
from retrieval.vector_store import VectorStore, time_overlaps


@dataclass(slots=True)
class SearchHit:
    doc_id: str
    collection: str
    source_id: str
    text: str
    score: float
    vector_score: float | None
    fts_score: float | None
    vector_rank: int | None
    fts_rank: int | None
    time_start: int | None
    time_end: int | None
    metadata: dict
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "collection": self.collection,
            "source_id": self.source_id,
            "text": self.text,
            "score": self.score,
            "vector_score": self.vector_score,
            "fts_score": self.fts_score,
            "vector_rank": self.vector_rank,
            "fts_rank": self.fts_rank,
            "rerank_score": self.rerank_score,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SearchFilters:
    memory_status: str | None = None
    memory_type: str | None = None
    trait_status: str | None = None
    dimension: str | None = None
    time_start: int | None = None
    time_end: int | None = None


def hybrid_search(
    conn: sqlite3.Connection,
    store: VectorStore,
    query: str,
    *,
    collections: tuple[str, ...],
    query_vector: list[float] | None,
    filters: SearchFilters,
    limit: int,
    candidate_limit: int,
    rrf_k: int,
    use_vector: bool,
    use_fts: bool,
    reranker=None,
) -> list[SearchHit]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    if limit < 1:
        raise ValueError("limit must be positive")
    vector_ranked = _vector_ranked(store, query_vector, collections, filters, candidate_limit) if use_vector else []
    fts_ranked = []
    if use_fts:
        for hit in search_fulltext(
            conn,
            query,
            collections,
            candidate_limit,
            memory_status=filters.memory_status,
            memory_type=filters.memory_type,
            trait_status=filters.trait_status,
            dimension=filters.dimension,
            time_start=filters.time_start,
            time_end=filters.time_end,
        ):
            fts_ranked.append((hit.doc_id, hit.score))
    fused = _fuse(vector_ranked, fts_ranked, rrf_k)
    window = fused[: max(limit, candidate_limit)]
    hits = _hydrate(conn, window, filters)
    if reranker is not None:
        hits = list(reranker.rerank(query, hits))
    return hits[:limit]


def _vector_ranked(store, query_vector, collections, filters: SearchFilters, limit: int) -> list[tuple[str, float]]:
    if query_vector is None:
        return []
    pooled: list[tuple[str, float]] = []
    for collection in collections:
        for hit in store.search(collection, query_vector, limit, _vector_filter(collection, filters)):
            doc_id = hit.payload.get("doc_id")
            if not doc_id:
                continue
            pooled.append((str(doc_id), float(hit.score)))
    pooled.sort(key=lambda item: (-item[1], item[0]))
    ranked: list[tuple[str, float]] = []
    seen = set()
    for doc_id, score in pooled:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ranked.append((doc_id, score))
    return ranked


def _vector_filter(collection: str, filters: SearchFilters) -> dict | None:
    payload: dict = {}
    if collection == "long_term_memories":
        if filters.memory_status:
            payload["status"] = filters.memory_status
        if filters.memory_type:
            payload["memory_type"] = filters.memory_type
    elif collection == "personality_observations":
        if filters.trait_status:
            payload["status"] = filters.trait_status
        if filters.dimension:
            payload["dimension"] = filters.dimension
    if filters.time_start is not None:
        payload["time_start"] = filters.time_start
    if filters.time_end is not None:
        payload["time_end"] = filters.time_end
    return payload or None


def _fuse(vector_ranked, fts_ranked, k: int) -> list[dict]:
    details: dict[str, dict] = {}

    def add(ranked, score_key: str, rank_key: str) -> None:
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            slot = details.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "score": 0.0,
                    "vector_score": None,
                    "fts_score": None,
                    "vector_rank": None,
                    "fts_rank": None,
                },
            )
            slot["score"] += 1.0 / (k + rank)
            slot[score_key] = score
            slot[rank_key] = rank

    add(vector_ranked, "vector_score", "vector_rank")
    add(fts_ranked, "fts_score", "fts_rank")
    ordered = sorted(details.values(), key=lambda slot: (-slot["score"], slot["doc_id"]))
    return ordered


def _hydrate(conn: sqlite3.Connection, fused: list[dict], filters: SearchFilters) -> list[SearchHit]:
    if not fused:
        return []
    ids = [slot["doc_id"] for slot in fused]
    placeholders = ",".join("?" for _ in ids)
    rows = {
        row["doc_id"]: row
        for row in conn.execute(
            f"SELECT * FROM retrieval_documents WHERE doc_id IN ({placeholders})",
            ids,
        )
    }
    hits: list[SearchHit] = []
    for slot in fused:
        row = rows.get(slot["doc_id"])
        if row is None or not _row_matches(row, filters):
            continue
        try:
            metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        except json.JSONDecodeError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        hits.append(
            SearchHit(
                doc_id=row["doc_id"],
                collection=row["collection"],
                source_id=row["source_id"],
                text=row["text"],
                score=float(slot["score"]),
                vector_score=slot["vector_score"],
                fts_score=slot["fts_score"],
                vector_rank=slot["vector_rank"],
                fts_rank=slot["fts_rank"],
                time_start=row["time_start"],
                time_end=row["time_end"],
                metadata=metadata,
            )
        )
    return hits


def _row_matches(row, filters: SearchFilters) -> bool:
    try:
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    except json.JSONDecodeError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    collection = row["collection"]
    if collection == "long_term_memories":
        if filters.memory_status and metadata.get("status") != filters.memory_status:
            return False
        if filters.memory_type and metadata.get("memory_type") != filters.memory_type:
            return False
    elif collection == "personality_observations":
        if filters.trait_status and metadata.get("status") != filters.trait_status:
            return False
        if filters.dimension and metadata.get("dimension") != filters.dimension:
            return False
    return time_overlaps(row["time_start"], row["time_end"], filters.time_start, filters.time_end)
