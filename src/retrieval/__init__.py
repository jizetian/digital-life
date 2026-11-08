"""Hybrid retrieval over segments, long-term memories, and personality traits.

SQLite holds the documents and the FTS5 index. A vector store holds dense
embeddings and can be deleted and rebuilt.
"""

from retrieval.service import run_index, run_search

__all__ = ["run_index", "run_search"]
