"""Build the retrieval projection and run a hybrid search.

SQLite stores the documents and the FTS5 index. The vector store stores
point ids and filter fields. Deleting the vector store and running the
index again reconstructs it from SQLite.
"""

from __future__ import annotations

from pathlib import Path

from common.clock import utc_now
from common.config import retrieval_config
from common.database import init_db
from common.logger import get_logger
from memory.types import MEMORY_STATUS_SET, MEMORY_TYPE_SET
from personality.dimensions import DIMENSIONS
from retrieval.collections import COLLECTIONS, parse_collections
from retrieval.documents import (
    clear_embedding_rows,
    embedding_signature,
    mark_embedded,
    payload_for,
    pending_documents,
    point_id_for,
    sync_documents,
)
from retrieval.embedding import EmbeddingClient, build_embedding_client
from retrieval.errors import RetrievalError
from retrieval.hybrid_search import SearchFilters, hybrid_search
from retrieval.reranker import build_reranker
from retrieval.vector_store import VectorPoint, VectorStore, build_vector_store

log = get_logger(__name__)

_TRAIT_STATUSES = frozenset({"stable", "provisional", "contested"})
_DIMENSIONS = frozenset(DIMENSIONS)


def run_index(
    db_path: str | Path,
    *,
    embedder: EmbeddingClient | None = None,
    store: VectorStore | None = None,
    collections: list[str] | tuple[str, ...] | None = None,
    rebuild: bool = False,
    batch_size: int | None = None,
    max_chars: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    dimension: int | None = None,
    qdrant_path: str | Path | None = None,
    qdrant_url: str | None = None,
    vector_backend: str | None = None,
) -> dict:
    database = _database(db_path)
    cfg = retrieval_config()
    selected = parse_collections(collections)
    batch = int(cfg.get("embedding_batch_size", 16) if batch_size is None else batch_size)
    cap = int(cfg.get("max_chars", 8000) if max_chars is None else max_chars)
    if batch < 1:
        raise ValueError("batch_size must be positive")
    if cap < 1:
        raise ValueError("max_chars must be positive")
    client = embedder if embedder is not None else build_embedding_client(
        provider=provider,
        model=model,
        base_url=base_url,
        dimension=dimension,
    )
    owned_store = store is None
    index = store if store is not None else build_vector_store(
        backend=vector_backend,
        path=qdrant_path,
        url=qdrant_url,
    )
    conn = init_db(database)
    try:
        width = _client_dimension(client)
        model_name = str(client.model_name)
        signature = embedding_signature(conn)
        mismatch = signature is not None and signature != (model_name, width)
        target = COLLECTIONS if mismatch else selected
        recreate = bool(rebuild or mismatch)
        synced = sync_documents(conn, target, max_chars=cap)
        conn.commit()
        vector_deleted = 0
        if recreate:
            for name in target:
                index.ensure_collection(name, width, recreate=True)
            clear_embedding_rows(conn, target)
            conn.commit()
        else:
            grouped: dict[str, list[str]] = {}
            for collection, point_id in synced.deleted_points:
                grouped.setdefault(collection, []).append(point_id)
            for collection, point_ids in grouped.items():
                index.delete(collection, point_ids)
                vector_deleted += len(point_ids)
            # A deleted Qdrant directory leaves the embedding rows in place.
            # Those rows are derived, so a count mismatch embeds the collection again.
            for name in target:
                index.ensure_collection(name, width, recreate=False)
                if index.count(name) != _embedding_count(conn, (name,)):
                    index.ensure_collection(name, width, recreate=True)
                    clear_embedding_rows(conn, (name,))
                    recreate = True
            conn.commit()
        embedded = _embed_pending(conn, client, index, target, model_name=model_name, dimension=width, batch_size=batch)
        counts = _collection_counts(conn)
        embedded_total = int(conn.execute("SELECT COUNT(*) AS n FROM retrieval_embeddings").fetchone()["n"])
    finally:
        conn.close()
        if owned_store:
            index.close()
    documents_total = sum(counts.values())
    log.info(
        "retrieval index model=%s dimension=%s upserted=%s deleted=%s embedded=%s",
        model_name,
        width,
        synced.upserted,
        synced.deleted,
        embedded,
    )
    return {
        "status": "completed",
        "provider": getattr(client, "provider", "injected"),
        "model": model_name,
        "dimension": width,
        "rebuilt_vectors": recreate,
        "documents": {
            "upserted": synced.upserted,
            "unchanged": synced.unchanged,
            "deleted": synced.deleted,
            "total": documents_total,
            "by_collection": counts,
        },
        "embedded": embedded,
        "embedded_total": embedded_total,
        "skipped": max(embedded_total - embedded, 0),
        "vector_deleted": vector_deleted,
    }


def run_search(
    db_path: str | Path,
    query: str,
    *,
    embedder: EmbeddingClient | None = None,
    store: VectorStore | None = None,
    reranker=None,
    collections: list[str] | tuple[str, ...] | None = None,
    limit: int | None = None,
    candidate_limit: int | None = None,
    use_vector: bool = True,
    use_fts: bool = True,
    rerank: bool = False,
    memory_status: str | None = None,
    memory_type: str | None = None,
    trait_status: str | None = None,
    dimension: str | None = None,
    time_start: int | None = None,
    time_end: int | None = None,
    rrf_k: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    qdrant_path: str | Path | None = None,
    qdrant_url: str | None = None,
    vector_backend: str | None = None,
) -> dict:
    cleaned = " ".join((query or "").split())
    if not cleaned:
        raise ValueError("query is empty")
    if not use_vector and not use_fts:
        raise ValueError("search needs vector, full text, or both")
    _check(memory_status, MEMORY_STATUS_SET, "memory_status")
    _check(memory_type, MEMORY_TYPE_SET, "memory_type")
    _check(trait_status, _TRAIT_STATUSES, "trait_status")
    _check(dimension, _DIMENSIONS, "dimension")
    database = _database(db_path)
    cfg = retrieval_config()
    selected = parse_collections(collections)
    resolved_limit = int(cfg.get("default_limit", 10) if limit is None else limit)
    resolved_candidates = int(cfg.get("candidate_limit", 40) if candidate_limit is None else candidate_limit)
    resolved_k = int(cfg.get("rrf_k", 60) if rrf_k is None else rrf_k)
    if resolved_limit < 1 or resolved_limit > 100:
        raise ValueError("limit must be between 1 and 100")
    if resolved_candidates < 1:
        raise ValueError("candidate_limit must be positive")
    if resolved_k < 1:
        raise ValueError("rrf_k must be positive")
    filters = SearchFilters(
        memory_status=memory_status,
        memory_type=memory_type,
        trait_status=trait_status,
        dimension=dimension,
        time_start=time_start,
        time_end=time_end,
    )
    conn = init_db(database)
    owned_store = store is None
    index = store if store is not None else build_vector_store(
        backend=vector_backend,
        path=qdrant_path,
        url=qdrant_url,
    )
    client = embedder
    try:
        indexed = int(conn.execute("SELECT COUNT(*) AS n FROM retrieval_documents").fetchone()["n"])
        embedded = _embedding_count(conn, selected)
        vector_note = None
        vector_active = bool(use_vector and embedded > 0)
        if use_vector and embedded == 0:
            vector_note = "no embeddings"
        query_vector = None
        model_name = ""
        if vector_active:
            if client is None:
                client = build_embedding_client(provider=provider, model=model, base_url=base_url)
            vectors = client.embed([cleaned])
            if len(vectors) != 1 or not vectors[0]:
                raise RetrievalError("embedding client returned no query vector")
            query_vector = vectors[0]
            model_name = str(client.model_name)
        elif client is not None:
            model_name = str(getattr(client, "model_name", ""))
        active_reranker = reranker
        reranker_name = "none"
        if active_reranker is None and rerank:
            active_reranker = build_reranker()
        if active_reranker is not None:
            reranker_name = str(getattr(active_reranker, "name", "custom"))
        hits = hybrid_search(
            conn,
            index,
            cleaned,
            collections=selected,
            query_vector=query_vector,
            filters=filters,
            limit=resolved_limit,
            candidate_limit=max(resolved_candidates, resolved_limit),
            rrf_k=resolved_k,
            use_vector=vector_active,
            use_fts=use_fts,
            reranker=active_reranker,
        )
    finally:
        conn.close()
        if owned_store:
            index.close()
    log.info("retrieval search query_chars=%s hits=%s vector=%s fts=%s", len(cleaned), len(hits), vector_active, use_fts)
    result = {
        "model": model_name,
        "query_chars": len(cleaned),
        "indexed_documents": indexed,
        "used_vector": vector_active,
        "used_fts": bool(use_fts),
        "reranker": reranker_name,
        "hits": [hit.to_dict() for hit in hits],
    }
    if vector_note:
        result["vector_note"] = vector_note
    return result


def _embed_pending(conn, client, store, collections, *, model_name: str, dimension: int, batch_size: int) -> int:
    embedded = 0
    while True:
        batch = list(
            pending_documents(
                conn,
                collections,
                model_name=model_name,
                dimension=dimension,
                limit=batch_size,
            )
        )
        if not batch:
            return embedded
        vectors = client.embed([row["text"] for row in batch])
        if len(vectors) != len(batch):
            raise RetrievalError("embedding client returned the wrong number of vectors")
        grouped: dict[str, list[VectorPoint]] = {}
        stored = []
        for row, vector in zip(batch, vectors):
            if len(vector) != dimension:
                raise RetrievalError("embedding dimension does not match the collection")
            point_id = point_id_for(row["doc_id"])
            grouped.setdefault(row["collection"], []).append(
                VectorPoint(point_id=point_id, vector=vector, payload=payload_for(row))
            )
            stored.append((row["doc_id"], row["text_hash"], model_name, dimension, point_id))
        for collection, points in grouped.items():
            store.upsert(collection, points)
        mark_embedded(conn, stored, now=utc_now())
        conn.commit()
        embedded += len(batch)


def _client_dimension(client) -> int:
    dimension = getattr(client, "dimension", 0) or 0
    if isinstance(dimension, int) and dimension > 0:
        return dimension
    vectors = client.embed(["."])
    if len(vectors) != 1 or not vectors[0]:
        raise RetrievalError("embedding client returned no dimension")
    return len(vectors[0])


def _collection_counts(conn) -> dict:
    counts = {name: 0 for name in COLLECTIONS}
    for row in conn.execute("SELECT collection, COUNT(*) AS n FROM retrieval_documents GROUP BY collection"):
        if row["collection"] in counts:
            counts[row["collection"]] = int(row["n"])
    return counts


def _embedding_count(conn, collections: tuple[str, ...]) -> int:
    placeholders = ",".join("?" for _ in collections)
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS n
        FROM retrieval_embeddings AS e
        JOIN retrieval_documents AS d ON d.doc_id = e.doc_id
        WHERE d.collection IN ({placeholders})
        """,
        collections,
    ).fetchone()
    return int(row["n"])


def _database(db_path: str | Path) -> Path:
    database = Path(db_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"database not found: {database}")
    return database


def _check(value: str | None, allowed: frozenset, label: str) -> None:
    if value is not None and value not in allowed:
        raise ValueError(f"unknown {label}: {value}")
