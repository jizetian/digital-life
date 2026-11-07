"""Vector indexes for the three retrieval collections.

``InMemoryVectorStore`` is the test double: cosine similarity over Python
lists, no Qdrant process and no GPU. ``QdrantVectorStore`` talks to a local
Qdrant path or a remote URL. Vectors are derived from SQLite and can be
deleted and rebuilt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from common.config import qdrant_path, retrieval_config
from retrieval.errors import VectorStoreError


@dataclass(slots=True)
class VectorPoint:
    point_id: str
    vector: list[float]
    payload: dict


@dataclass(slots=True)
class VectorHit:
    point_id: str
    score: float
    payload: dict


class VectorStore(Protocol):
    def ensure_collection(self, name: str, dimension: int, *, recreate: bool = False) -> None:
        """Create ``name`` when missing. ``recreate`` drops an existing collection first."""

    def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        """Insert or replace points by id."""

    def delete(self, collection: str, point_ids: list[str]) -> None:
        """Remove points. Missing ids are ignored."""

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int,
        payload_filter: dict | None = None,
    ) -> list[VectorHit]:
        """Nearest neighbours, highest score first. An unknown collection returns nothing."""

    def count(self, collection: str) -> int:
        """How many points the collection currently holds."""

    def close(self) -> None:
        """Release a local Qdrant lock. In-memory stores keep their points."""


@dataclass
class _MemoryCollection:
    dimension: int
    points: dict[str, tuple[list[float], dict]] = field(default_factory=dict)


class InMemoryVectorStore:
    """Process-local cosine index. Two instances do not share points."""

    def __init__(self) -> None:
        self._collections: dict[str, _MemoryCollection] = {}

    def ensure_collection(self, name: str, dimension: int, *, recreate: bool = False) -> None:
        if dimension < 1:
            raise VectorStoreError("vector dimension must be positive")
        current = self._collections.get(name)
        if current is not None and recreate:
            current = None
        if current is None:
            self._collections[name] = _MemoryCollection(dimension=dimension)
            return
        if current.dimension != dimension:
            raise VectorStoreError(
                f"collection {name} has dimension {current.dimension}, expected {dimension}. "
                "Re-run with --rebuild."
            )

    def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        target = self._require(collection)
        for point in points:
            if len(point.vector) != target.dimension:
                raise VectorStoreError(
                    f"point {point.point_id} has dimension {len(point.vector)}, expected {target.dimension}"
                )
            target.points[point.point_id] = (list(point.vector), dict(point.payload))

    def delete(self, collection: str, point_ids: list[str]) -> None:
        target = self._collections.get(collection)
        if target is None:
            return
        for point_id in point_ids:
            target.points.pop(point_id, None)

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int,
        payload_filter: dict | None = None,
    ) -> list[VectorHit]:
        target = self._collections.get(collection)
        if target is None or limit < 1:
            return []
        ranked: list[VectorHit] = []
        for point_id, (stored, payload) in target.points.items():
            if not _payload_matches(payload, payload_filter):
                continue
            score = _cosine(vector, stored)
            if score <= 0.0:
                continue
            ranked.append(VectorHit(point_id=point_id, score=score, payload=dict(payload)))
        ranked.sort(key=lambda hit: (-hit.score, hit.point_id))
        return ranked[:limit]

    def count(self, collection: str) -> int:
        target = self._collections.get(collection)
        if target is None:
            return 0
        return len(target.points)

    def close(self) -> None:
        return None

    def _require(self, collection: str) -> _MemoryCollection:
        target = self._collections.get(collection)
        if target is None:
            raise VectorStoreError(f"collection {collection} does not exist")
        return target


class QdrantVectorStore:
    """Local path or remote Qdrant. The client is imported only when this is constructed."""

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
        client=None,
    ) -> None:
        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise VectorStoreError(
                    "qdrant-client is not installed. pip install qdrant-client, "
                    "or pass --backend memory for a process-local index."
                ) from exc
            if url:
                self._client = QdrantClient(url=url, api_key=api_key or None)
            elif path is not None and str(path) != "":
                if str(path) != ":memory:":
                    Path(path).mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(path))
            else:
                raise VectorStoreError("Qdrant needs a path or a url")
        from qdrant_client.http import models as qmodels

        self._models = qmodels

    def ensure_collection(self, name: str, dimension: int, *, recreate: bool = False) -> None:
        if dimension < 1:
            raise VectorStoreError("vector dimension must be positive")
        exists = bool(self._client.collection_exists(name))
        if exists and recreate:
            self._client.delete_collection(name)
            exists = False
        if exists:
            size = _vector_size(self._client.get_collection(name))
            if size != dimension:
                raise VectorStoreError(
                    f"collection {name} has dimension {size}, expected {dimension}. "
                    "Re-run with --rebuild."
                )
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=self._models.VectorParams(size=dimension, distance=self._models.Distance.COSINE),
        )

    def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        if not points:
            return
        structs = [
            self._models.PointStruct(id=point.point_id, vector=point.vector, payload=point.payload)
            for point in points
        ]
        self._client.upsert(collection_name=collection, points=structs)

    def delete(self, collection: str, point_ids: list[str]) -> None:
        if not point_ids or not self._client.collection_exists(collection):
            return
        selector = self._models.PointIdsList(points=list(point_ids))
        self._client.delete(collection_name=collection, points_selector=selector)

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int,
        payload_filter: dict | None = None,
    ) -> list[VectorHit]:
        if limit < 1 or not self._client.collection_exists(collection):
            return []
        try:
            response = self._client.query_points(
                collection_name=collection,
                query=vector,
                limit=limit,
                query_filter=_qdrant_filter(self._models, payload_filter),
            )
        except ValueError:
            return []
        hits: list[VectorHit] = []
        for point in response.points:
            if point.score is None or float(point.score) <= 0.0:
                continue
            hits.append(
                VectorHit(
                    point_id=str(point.id),
                    score=float(point.score),
                    payload=dict(point.payload or {}),
                )
            )
        return hits

    def count(self, collection: str) -> int:
        if not self._client.collection_exists(collection):
            return 0
        return int(self._client.count(collection).count)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def build_vector_store(
    *,
    backend: str | None = None,
    path: str | Path | None = None,
    url: str | None = None,
    api_key: str | None = None,
) -> VectorStore:
    cfg = retrieval_config()
    chosen = (backend or cfg.get("vector_backend") or "qdrant").strip().lower()
    if chosen == "memory":
        return InMemoryVectorStore()
    if chosen != "qdrant":
        raise VectorStoreError(f"unknown vector backend: {chosen}")
    resolved_url = url if url is not None else str(cfg.get("qdrant_url") or "")
    resolved_key = api_key if api_key is not None else str(cfg.get("qdrant_api_key") or "")
    if resolved_url:
        return QdrantVectorStore(url=resolved_url, api_key=resolved_key or None)
    resolved_path = Path(path).expanduser() if path else qdrant_path()
    return QdrantVectorStore(path=resolved_path, api_key=resolved_key or None)


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def _payload_matches(payload: dict, payload_filter: dict | None) -> bool:
    if not payload_filter:
        return True
    start = payload_filter.get("time_start")
    end = payload_filter.get("time_end")
    if start is not None or end is not None:
        if not time_overlaps(payload.get("time_start"), payload.get("time_end"), start, end):
            return False
    for key, value in payload_filter.items():
        if key in {"time_start", "time_end"}:
            continue
        if payload.get(key) != value:
            return False
    return True


def time_overlaps(doc_start, doc_end, start, end) -> bool:
    """True when the document interval overlaps ``[start, end]``.

    A missing document bound is filled from the other bound. A document with
    no time at all does not match a time filter.
    """
    if start is None and end is None:
        return True
    if doc_start is None and doc_end is None:
        return False
    left = doc_start if doc_start is not None else doc_end
    right = doc_end if doc_end is not None else doc_start
    if start is not None and right < start:
        return False
    if end is not None and left > end:
        return False
    return True


def _qdrant_filter(models, payload_filter: dict | None):
    if not payload_filter:
        return None
    must = []
    for key, value in payload_filter.items():
        if key == "time_start":
            must.append(models.FieldCondition(key="time_end", range=models.Range(gte=value)))
        elif key == "time_end":
            must.append(models.FieldCondition(key="time_start", range=models.Range(lte=value)))
        else:
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    return models.Filter(must=must)


def _vector_size(info) -> int:
    vectors = info.config.params.vectors
    size = getattr(vectors, "size", None)
    if isinstance(size, int):
        return size
    if isinstance(vectors, dict) and vectors:
        first = next(iter(vectors.values()))
        return int(first.size)
    raise VectorStoreError("Qdrant collection is missing a vector size")
