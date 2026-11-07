"""Memory types, statuses, and the closed vocabularies the model must use."""

from __future__ import annotations

MEMORY_TYPES = ("fact", "event", "relationship", "opinion", "situational")
MEMORY_TYPE_SET = frozenset(MEMORY_TYPES)

MEMORY_STATUSES = ("active", "historical", "superseded", "uncertain")
MEMORY_STATUS_SET = frozenset(MEMORY_STATUSES)

ENTITY_TYPES = ("person", "place", "organization", "object", "concept", "other")
ENTITY_TYPE_SET = frozenset(ENTITY_TYPES)

ENTITY_STATUSES = ("confirmed", "inferred", "unknown")
ENTITY_STATUS_SET = frozenset(ENTITY_STATUSES)
ENTITY_STATUS_RANK = {"unknown": 0, "inferred": 1, "confirmed": 2}

ENTITY_ROLES = ("subject", "object", "participant", "location", "mentioned")
ENTITY_ROLE_SET = frozenset(ENTITY_ROLES)

TIME_PRECISIONS = ("instant", "day", "range", "unknown")
TIME_PRECISION_SET = frozenset(TIME_PRECISIONS)

MERGE_ACTIONS = ("insert", "duplicate", "supersede", "historical", "conflict", "related")
MERGE_ACTION_SET = frozenset(MERGE_ACTIONS)
MERGE_ACTIONS_NEEDING_TARGET = frozenset({"duplicate", "supersede", "historical", "conflict", "related"})

RELATION_FOR_ACTION = {
    "supersede": "supersedes",
    "historical": "historical_of",
    "conflict": "conflicts_with",
    "related": "related_to",
}

EVIDENCE_ROLES = ("support", "conflict", "context")

MAX_CANDIDATES_PER_SEGMENT = 30
MAX_STATEMENT_CHARS = 2000
MAX_NOTE_CHARS = 300
MAX_ALIAS_COUNT = 20
