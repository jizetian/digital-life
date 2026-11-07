"""Apply merge decisions without deleting an older memory.

A conflicting or outdated row stays in ``memories``. Its status becomes
``superseded``, ``historical``, or ``uncertain``, and ``memory_relations``
records why the new row points at it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

from common.clock import utc_now
from memory.evidence_manager import add_evidence, existing_message_ids, existing_observation_ids
from memory.models import (
    CandidateMemory,
    EntityMention,
    MergeAction,
    MergeStats,
    clamp_unit,
    entity_id_for,
    memory_identity,
    normalize_text,
    time_precision_for,
)
from memory.types import (
    ENTITY_STATUS_RANK,
    MAX_ALIAS_COUNT,
    MERGE_ACTIONS_NEEDING_TARGET,
    RELATION_FOR_ACTION,
)


def resolve_actions(
    candidates: list[CandidateMemory],
    parsed: list[MergeAction],
    *,
    allowed_target_ids: set[str],
) -> list[MergeAction]:
    """One action per candidate. A missing or unusable target becomes an insert."""
    chosen: dict[int, MergeAction] = {}
    for action in parsed:
        index = action.candidate_index
        if index in chosen or index < 0 or index >= len(candidates):
            continue
        kind = action.action
        target = action.target_memory_id
        if kind in MERGE_ACTIONS_NEEDING_TARGET and (not target or target not in allowed_target_ids):
            kind = "insert"
            target = None
        if kind == "insert":
            target = None
        chosen[index] = MergeAction(index, kind, target, action.note)
    resolved: list[MergeAction] = []
    for index in range(len(candidates)):
        resolved.append(chosen.get(index) or MergeAction(index, "insert", None, None))
    return resolved


def apply_merge(
    conn: sqlite3.Connection,
    *,
    segment_id: str | None,
    candidates: list[CandidateMemory],
    actions: list[MergeAction],
    prompt_version: str,
    model_version: str,
    allowed_target_ids: set[str] | None = None,
) -> MergeStats:
    """Write candidates. Re-applying the same candidate updates evidence in place."""
    del allowed_target_ids  # targets were already checked by resolve_actions
    if len(candidates) != len(actions):
        raise ValueError("merge actions must align with candidates")
    stats = MergeStats()
    now = utc_now()
    for candidate, action in zip(candidates, actions):
        label = _apply_one(
            conn,
            candidate=candidate,
            action=action,
            segment_id=segment_id,
            prompt_version=prompt_version,
            model_version=model_version,
            now=now,
        )
        setattr(stats, _STAT_FIELD[label], getattr(stats, _STAT_FIELD[label]) + 1)
    return stats


_STAT_FIELD = {
    "inserted": "inserted",
    "duplicated": "duplicated",
    "superseded": "superseded",
    "historical": "historical",
    "conflicts": "conflicts",
    "related": "related",
    "dropped": "dropped",
}


def _apply_one(
    conn: sqlite3.Connection,
    *,
    candidate: CandidateMemory,
    action: MergeAction,
    segment_id: str | None,
    prompt_version: str,
    model_version: str,
    now: str,
) -> str:
    message_ids = existing_message_ids(conn, candidate.evidence_message_ids)
    observation_ids = existing_observation_ids(conn, candidate.observation_ids)
    if not message_ids:
        return "dropped"
    identity = memory_identity(prompt_version, model_version, candidate.memory_type, candidate.statement)
    exists = _load(conn, identity) is not None
    kind = action.action
    target_id = action.target_memory_id
    # The same statement is already stored. Keep that row, even if the model
    # pointed the action at itself.
    if exists and (target_id == identity or kind in {"insert", "duplicate"} and not target_id):
        _absorb(conn, identity, candidate, segment_id, message_ids, observation_ids, now)
        return "duplicated"
    if kind == "insert" and exists:
        _absorb(conn, identity, candidate, segment_id, message_ids, observation_ids, now)
        return "duplicated"
    if kind == "duplicate":
        if target_id is None or _load(conn, target_id) is None:
            if exists:
                _absorb(conn, identity, candidate, segment_id, message_ids, observation_ids, now)
                return "duplicated"
            _insert(
                conn,
                memory_id=identity,
                candidate=candidate,
                status="active",
                segment_id=segment_id,
                message_ids=message_ids,
                observation_ids=observation_ids,
                prompt_version=prompt_version,
                model_version=model_version,
                action_name="insert",
                now=now,
            )
            return "inserted"
        _absorb(conn, target_id, candidate, segment_id, message_ids, observation_ids, now)
        return "duplicated"

    if exists:
        memory_id = identity
        _absorb(conn, identity, candidate, segment_id, message_ids, observation_ids, now)
        created = False
    else:
        status = "uncertain" if kind == "conflict" else "active"
        memory_id = identity
        _insert(
            conn,
            memory_id=memory_id,
            candidate=candidate,
            status=status,
            segment_id=segment_id,
            message_ids=message_ids,
            observation_ids=observation_ids,
            prompt_version=prompt_version,
            model_version=model_version,
            action_name=kind,
            now=now,
        )
        created = True

    if target_id is None or target_id == memory_id or _load(conn, target_id) is None:
        return "inserted" if created else "duplicated"

    if kind == "supersede":
        _set_status(conn, target_id, "superseded", superseded_by=memory_id, now=now)
        _relate(conn, "supersedes", memory_id, target_id, candidate.confidence, action.note, now)
        return "superseded"
    if kind == "historical":
        _set_status(conn, target_id, "historical", superseded_by=None, now=now)
        _relate(conn, "historical_of", memory_id, target_id, candidate.confidence, action.note, now)
        return "historical"
    if kind == "conflict":
        _set_status(conn, target_id, "uncertain", superseded_by=None, now=now)
        _set_status(conn, memory_id, "uncertain", superseded_by=None, now=now)
        _relate(conn, "conflicts_with", memory_id, target_id, candidate.confidence, action.note, now)
        add_evidence(
            conn,
            memory_id=target_id,
            segment_id=segment_id,
            message_ids=message_ids,
            observation_ids=[],
            role="conflict",
            created_at=now,
        )
        return "conflicts"
    if kind == "related":
        relation = RELATION_FOR_ACTION["related"]
        _relate(conn, relation, memory_id, target_id, candidate.confidence, action.note, now)
        return "related"
    return "inserted" if created else "duplicated"


def _insert(
    conn: sqlite3.Connection,
    *,
    memory_id: str,
    candidate: CandidateMemory,
    status: str,
    segment_id: str | None,
    message_ids: list[str],
    observation_ids: list[str],
    prompt_version: str,
    model_version: str,
    action_name: str,
    now: str,
) -> None:
    precision = time_precision_for(candidate.time_start, candidate.time_end, candidate.time_precision)
    payload = _payload(candidate, action_name)
    conn.execute(
        """
        INSERT INTO memories (
            memory_id, memory_type, status, statement, situation, importance, confidence,
            time_start, time_end, time_precision, evidence_message_ids_json,
            source_observation_ids_json, origin_segment_id, prompt_version, model_version,
            superseded_by, payload_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', '[]', ?, ?, ?, NULL, ?, ?, ?)
        """,
        (
            memory_id,
            candidate.memory_type,
            status,
            candidate.statement,
            candidate.situation,
            clamp_unit(candidate.importance),
            clamp_unit(candidate.confidence),
            candidate.time_start,
            candidate.time_end,
            precision,
            segment_id,
            prompt_version,
            model_version,
            json.dumps(payload, ensure_ascii=False),
            now,
            now,
        ),
    )
    _link_entities(
        conn,
        memory_id,
        candidate.entities,
        candidate.time_start,
        candidate.time_end,
        candidate.confidence,
        now,
    )
    add_evidence(
        conn,
        memory_id=memory_id,
        segment_id=segment_id,
        message_ids=message_ids,
        observation_ids=observation_ids,
        role="support",
        created_at=now,
    )


def _absorb(
    conn: sqlite3.Connection,
    memory_id: str,
    candidate: CandidateMemory,
    segment_id: str | None,
    message_ids: list[str],
    observation_ids: list[str],
    now: str,
) -> None:
    row = _load(conn, memory_id)
    if row is None:
        return
    start = _min_time(row["time_start"], candidate.time_start)
    end = _max_time(row["time_end"], candidate.time_end)
    precision = time_precision_for(start, end, candidate.time_precision if candidate.time_precision == "day" else None)
    if row["time_precision"] == "day":
        precision = "day"
    situation = row["situation"] or candidate.situation
    conn.execute(
        """
        UPDATE memories
        SET situation = ?, importance = ?, confidence = ?, time_start = ?, time_end = ?,
            time_precision = ?, updated_at = ?
        WHERE memory_id = ?
        """,
        (
            situation,
            _max_unit(row["importance"], candidate.importance),
            _max_unit(row["confidence"], candidate.confidence),
            start,
            end,
            precision,
            now,
            memory_id,
        ),
    )
    _link_entities(conn, memory_id, candidate.entities, start, end, candidate.confidence, now)
    add_evidence(
        conn,
        memory_id=memory_id,
        segment_id=segment_id,
        message_ids=message_ids,
        observation_ids=observation_ids,
        role="support",
        created_at=now,
    )


def _set_status(
    conn: sqlite3.Connection,
    memory_id: str,
    status: str,
    *,
    superseded_by: str | None,
    now: str,
) -> None:
    row = _load(conn, memory_id)
    if row is None:
        return
    current = row["status"]
    new_status = current
    if status == "superseded":
        new_status = "superseded"
    elif status == "historical" and current in {"active", "uncertain"}:
        new_status = "historical"
    elif status == "uncertain" and current == "active":
        new_status = "uncertain"
    pointer = row["superseded_by"]
    if superseded_by and not pointer:
        pointer = superseded_by
    conn.execute(
        """
        UPDATE memories
        SET status = ?, superseded_by = ?, updated_at = ?
        WHERE memory_id = ?
        """,
        (new_status, pointer, now, memory_id),
    )


def _relate(
    conn: sqlite3.Connection,
    relation_type: str,
    from_memory_id: str,
    to_memory_id: str,
    confidence: float | None,
    note: str | None,
    now: str,
) -> None:
    if from_memory_id == to_memory_id:
        return
    relation_id = hashlib.sha256(f"{relation_type}|{from_memory_id}|{to_memory_id}".encode("utf-8")).hexdigest()
    conn.execute(
        """
        INSERT OR IGNORE INTO memory_relations (
            relation_id, relation_type, from_memory_id, to_memory_id, confidence, note, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            relation_id,
            relation_type,
            from_memory_id,
            to_memory_id,
            clamp_unit(confidence),
            note,
            now,
        ),
    )


def _link_entities(
    conn: sqlite3.Connection,
    memory_id: str,
    mentions: list[EntityMention],
    seen_start: int | None,
    seen_end: int | None,
    confidence: float | None,
    now: str,
) -> None:
    for mention in mentions:
        entity_id = _upsert_entity(
            conn,
            mention,
            seen_start=seen_start,
            seen_end=seen_end,
            confidence=confidence,
            now=now,
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO memory_entities (memory_id, entity_id, role)
            VALUES (?, ?, ?)
            """,
            (memory_id, entity_id, mention.role),
        )


def _upsert_entity(
    conn: sqlite3.Connection,
    mention: EntityMention,
    *,
    seen_start: int | None,
    seen_end: int | None,
    confidence: float | None,
    now: str,
) -> str:
    entity_id = entity_id_for(mention.entity_type, mention.name)
    row = conn.execute("SELECT * FROM entities WHERE entity_id = ?", (entity_id,)).fetchone()
    stored_confidence = clamp_unit(confidence)
    if row is None:
        conn.execute(
            """
            INSERT INTO entities (
                entity_id, canonical_name, name_key, entity_type, status, aliases_json,
                description, confidence, first_seen, last_seen, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity_id,
                mention.name,
                normalize_text(mention.name),
                mention.entity_type,
                mention.status,
                json.dumps(_alias_list(mention.name, mention.aliases), ensure_ascii=False),
                mention.description,
                stored_confidence,
                seen_start,
                seen_end if seen_end is not None else seen_start,
                now,
                now,
            ),
        )
        return entity_id
    status = mention.status
    if ENTITY_STATUS_RANK[mention.status] < ENTITY_STATUS_RANK[row["status"]]:
        status = row["status"]
    aliases = json.loads(row["aliases_json"]) if row["aliases_json"] else []
    aliases = _merge_aliases(row["canonical_name"], aliases, [mention.name, *mention.aliases])
    description = row["description"] or mention.description
    conn.execute(
        """
        UPDATE entities
        SET status = ?, aliases_json = ?, description = ?, confidence = ?,
            first_seen = ?, last_seen = ?, updated_at = ?
        WHERE entity_id = ?
        """,
        (
            status,
            json.dumps(aliases, ensure_ascii=False),
            description,
            _max_unit(row["confidence"], stored_confidence),
            _min_time(row["first_seen"], seen_start),
            _max_time(row["last_seen"], seen_end if seen_end is not None else seen_start),
            now,
            entity_id,
        ),
    )
    return entity_id


def _alias_list(canonical: str, aliases: list[str]) -> list[str]:
    return _merge_aliases(canonical, [], aliases)


def _merge_aliases(canonical: str, current: list[str], extra: list[str]) -> list[str]:
    merged: list[str] = []
    seen = {normalize_text(canonical)}
    for value in [*current, *extra]:
        if not isinstance(value, str):
            continue
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            continue
        key = normalize_text(cleaned)
        if key in seen:
            continue
        seen.add(key)
        merged.append(cleaned[:80])
        if len(merged) >= MAX_ALIAS_COUNT:
            break
    return merged


def _payload(candidate: CandidateMemory, action_name: str) -> dict:
    return {
        "memory_type": candidate.memory_type,
        "statement": candidate.statement,
        "situation": candidate.situation,
        "entities": [item.to_dict() for item in candidate.entities],
        "action": action_name,
        "time_precision": candidate.time_precision,
    }


def _load(conn: sqlite3.Connection, memory_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()


def _max_unit(left, right) -> float | None:
    values = [clamp_unit(left), clamp_unit(right)]
    present = [item for item in values if item is not None]
    if not present:
        return None
    return max(present)


def _min_time(left, right) -> int | None:
    values = [item for item in (left, right) if item is not None]
    if not values:
        return None
    return int(min(values))


def _max_time(left, right) -> int | None:
    values = [item for item in (left, right) if item is not None]
    if not values:
        return None
    return int(max(values))
