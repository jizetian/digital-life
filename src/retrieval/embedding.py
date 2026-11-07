"""Pluggable dense embedding clients.

The default provider talks to an OpenAI-compatible ``/embeddings`` endpoint.
Point it at Ollama (``http://127.0.0.1:11434/v1``, model ``bge-m3``) or any
server that serves BGE-M3 dense vectors. ``hash`` is a deterministic local
stand-in for tests and smoke runs. It is not BGE-M3. ``sentence_transformers``
loads ``BAAI/bge-m3`` only when that package is installed; it is not a
dependency, so CI does not need a GPU.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol

import httpx

from common.config import distillation_config, retrieval_config
from retrieval.errors import EmbeddingError

_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_CJK = re.compile(r"[\u4e00-\u9fff]")


class EmbeddingClient(Protocol):
    model_name: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per input text, in the same order."""


class HashEmbeddingClient:
    """Feature-hashing bag of words. Shared tokens raise cosine similarity.

    English words and CJK characters each take a bucket. Adjacent CJK
    characters also contribute a bigram, so a Chinese query overlaps a longer
    sentence that contains it. Vectors are L2-normalized.
    """

    def __init__(self, *, dimension: int = 256, model_name: str = "hash-v1") -> None:
        if dimension < 8:
            raise EmbeddingError("hash embedding dimension must be at least 8")
        self.dimension = int(dimension)
        self.model_name = model_name
        self.provider = "hash"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vector(text, self.dimension) for text in texts]


class OpenAIEmbeddingClient:
    """POST ``{base_url}/embeddings``. BGE-M3 on Ollama uses this shape."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url or not str(base_url).strip():
            raise EmbeddingError("embedding base_url is empty")
        if not model or not str(model).strip():
            raise EmbeddingError("embedding model is empty")
        self.base_url = str(base_url).rstrip("/")
        self.model_name = str(model)
        self.provider = "openai"
        self.api_key = api_key or None
        self.timeout = float(timeout)
        self._transport = transport
        self._dimension = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self.base_url}/embeddings"
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model_name, "input": list(texts)}
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                response = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise EmbeddingError("embedding request timed out") from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embedding request failed: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise EmbeddingError(f"embedding HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise EmbeddingError("embedding response was not JSON") from exc
        vectors = _vectors_from_response(body, expected=len(texts))
        width = len(vectors[0])
        if self._dimension and width != self._dimension:
            raise EmbeddingError("embedding dimension changed within one client")
        self._dimension = width
        return vectors


class SentenceTransformerEmbeddingClient:
    """Local ``sentence-transformers`` loader. Optional; not imported otherwise."""

    def __init__(self, *, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. Use the OpenAI-compatible "
                "client with model bge-m3, or pip install sentence-transformers."
            ) from exc
        self.model_name = model_name
        self.provider = "sentence_transformers"
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self._model.encode(list(texts), normalize_embeddings=True)
        vectors = [[float(value) for value in row] for row in matrix]
        if any(len(vector) != self.dimension for vector in vectors):
            raise EmbeddingError("sentence-transformers returned an unexpected dimension")
        return vectors


def build_embedding_client(
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    dimension: int | None = None,
    timeout: float | None = None,
    transport: httpx.BaseTransport | None = None,
) -> EmbeddingClient:
    cfg = retrieval_config()
    chosen = (provider or cfg.get("embedding_provider") or "openai").strip().lower()
    model_name = model or str(cfg.get("embedding_model") or "bge-m3")
    if chosen == "hash":
        width = int(dimension if dimension is not None else cfg.get("embedding_dimension") or 256)
        label = f"hash:{model}" if model else "hash-v1"
        return HashEmbeddingClient(dimension=width, model_name=label)
    if chosen in {"sentence_transformers", "sentence-transformers", "bge-m3-local"}:
        local_model = model or "BAAI/bge-m3"
        return SentenceTransformerEmbeddingClient(model_name=local_model)
    if chosen not in {"openai", "openai-compatible", "ollama"}:
        raise EmbeddingError(f"unknown embedding provider: {chosen}")
    resolved_url = base_url or str(cfg.get("embedding_base_url") or "")
    if not resolved_url:
        resolved_url = str(distillation_config().get("base_url") or "")
    api_key = os.environ.get("CHATTWIN_EMBEDDING_API_KEY") or os.environ.get("CHATTWIN_LLM_API_KEY") or None
    if api_key == "":
        api_key = None
    wait = float(cfg.get("embedding_timeout_seconds", 120) if timeout is None else timeout)
    return OpenAIEmbeddingClient(
        base_url=resolved_url,
        model=model_name,
        api_key=api_key,
        timeout=wait,
        transport=transport,
    )


def _hash_vector(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    for token in _features(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in (0, 8):
            bucket = int.from_bytes(digest[offset:offset + 4], "little") % dimension
            sign = 1.0 if digest[offset + 4] % 2 == 0 else -1.0
            vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _features(text: str) -> list[str]:
    folded = text.casefold()
    parts = _TOKEN.findall(folded)
    cjk = _CJK.findall(folded)
    bigrams = [left + right for left, right in zip(cjk, cjk[1:])]
    return parts + bigrams


def _vectors_from_response(body: dict, *, expected: int) -> list[list[float]]:
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or len(data) != expected:
        raise EmbeddingError("embedding response did not contain one vector per input")
    ordered: list[list[float] | None] = [None] * expected
    for item in data:
        if not isinstance(item, dict):
            raise EmbeddingError("embedding response item is not an object")
        try:
            index = int(item["index"])
            raw = item["embedding"]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError("embedding response is missing index or embedding") from exc
        if index < 0 or index >= expected or ordered[index] is not None:
            raise EmbeddingError("embedding response indexes are invalid")
        if not isinstance(raw, list) or not raw:
            raise EmbeddingError("embedding vector is empty")
        try:
            ordered[index] = [float(value) for value in raw]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("embedding vector is not numeric") from exc
    vectors = [vector for vector in ordered if vector is not None]
    if len(vectors) != expected:
        raise EmbeddingError("embedding response left a vector missing")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise EmbeddingError("embedding vectors differ in dimension")
    return vectors
