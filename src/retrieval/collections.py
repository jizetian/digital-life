"""Qdrant collection names and the SQLite rows they are built from.

``personality_observations`` indexes L2 traits (``l2_traits``), the personality
statements. L1 rows stay the evidence layer and are cited from trait metadata.
"""

from __future__ import annotations

COLLECTIONS = (
    "conversation_segments",
    "long_term_memories",
    "personality_observations",
)

COLLECTION_SET = frozenset(COLLECTIONS)

SOURCE_TABLE = {
    "conversation_segments": "conversation_segments",
    "long_term_memories": "memories",
    "personality_observations": "l2_traits",
}


def parse_collections(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not values:
        return COLLECTIONS
    unknown = [name for name in values if name not in COLLECTION_SET]
    if unknown:
        joined = ", ".join(unknown)
        known = ", ".join(COLLECTIONS)
        raise ValueError(f"unknown collection: {joined}. Expected one of: {known}")
    return tuple(dict.fromkeys(values))
