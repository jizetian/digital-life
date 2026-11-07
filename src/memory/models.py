"""In-memory records passed between the parser, merger, and engine."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from memory.types import TIME_PRECISION_SET


def normalize_text(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def clamp_unit(value) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return number


def memory_identity(prompt_version: str, model_version: str, memory_type: str, statement: str) -> str:
    raw = f"{prompt_version}|{model_version}|{memory_type}|{normalize_text(statement)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def entity_id_for(entity_type: str, name: str) -> str:
    raw = f"{entity_type}|{normalize_text(name)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def time_precision_for(start: int | None, end: int | None, requested: str | None = None) -> str:
    if requested == "day" and (start is not None or end is not None):
        return "day"
    if start is None and end is None:
        return "unknown"
    if start is not None and end is not None and start == end:
        return "instant"
    if requested in TIME_PRECISION_SET and requested != "unknown":
        return requested
    return "range"


@dataclass(slots=True)
class EntityMention:
    name: str
    entity_type: str
    role: str
    status: str
    description: str | None = None
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "role": self.role,
            "status": self.status,
            "description": self.description,
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> EntityMention:
        return cls(
            name=str(payload["name"]),
            entity_type=str(payload["entity_type"]),
            role=str(payload["role"]),
            status=str(payload["status"]),
            description=payload.get("description"),
            aliases=list(payload.get("aliases") or []),
        )


@dataclass(slots=True)
class CandidateMemory:
    memory_type: str
    statement: str
    situation: str | None
    importance: float | None
    confidence: float | None
    time_start: int | None
    time_end: int | None
    time_precision: str
    evidence_message_ids: list[str]
    observation_ids: list[str]
    entities: list[EntityMention]

    def to_dict(self) -> dict:
        return {
            "memory_type": self.memory_type,
            "statement": self.statement,
            "situation": self.situation,
            "importance": self.importance,
            "confidence": self.confidence,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "time_precision": self.time_precision,
            "evidence_message_ids": list(self.evidence_message_ids),
            "observation_ids": list(self.observation_ids),
            "entities": [item.to_dict() for item in self.entities],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> CandidateMemory:
        return cls(
            memory_type=str(payload["memory_type"]),
            statement=str(payload["statement"]),
            situation=payload.get("situation"),
            importance=payload.get("importance"),
            confidence=payload.get("confidence"),
            time_start=payload.get("time_start"),
            time_end=payload.get("time_end"),
            time_precision=str(payload.get("time_precision") or "unknown"),
            evidence_message_ids=list(payload.get("evidence_message_ids") or []),
            observation_ids=list(payload.get("observation_ids") or []),
            entities=[EntityMention.from_dict(item) for item in payload.get("entities") or []],
        )


@dataclass(slots=True)
class MergeAction:
    candidate_index: int
    action: str
    target_memory_id: str | None
    note: str | None


@dataclass(slots=True)
class MergeStats:
    inserted: int = 0
    duplicated: int = 0
    superseded: int = 0
    historical: int = 0
    conflicts: int = 0
    related: int = 0
    dropped: int = 0

    def to_dict(self) -> dict:
        return {
            "inserted": self.inserted,
            "duplicated": self.duplicated,
            "superseded": self.superseded,
            "historical": self.historical,
            "conflicts": self.conflicts,
            "related": self.related,
            "dropped": self.dropped,
        }
