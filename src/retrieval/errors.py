"""Errors raised while indexing or searching. They never include message text."""

from __future__ import annotations


class RetrievalError(Exception):
    """Indexing or search cannot continue."""


class EmbeddingError(RetrievalError):
    """The embedding client did not return vectors."""


class VectorStoreError(RetrievalError):
    """The vector index could not be read or updated."""
