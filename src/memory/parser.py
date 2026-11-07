"""Parse extract and merge JSON. Unknown evidence ids are removed."""

from __future__ import annotations

import json
import re

from memory.evidence_manager import filter_ids
from memory.models import CandidateMemory, EntityMention, MergeAction, clamp_unit, time_precision_for
from memory.types import (
    ENTITY_ROLE_SET,
    ENTITY_STATUS_SET,
    ENTITY_TYPE_SET,
    MAX_CANDIDATES_PER_SEGMENT,
    MAX_NOTE_CHARS,
    MAX_STATEMENT_CHARS,
    MEMORY_TYPE_SET,
    MERGE_ACTION_SET,
    TIME_PRECISION_SET,
)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_ACTION_ALIASES = {
    "supersedes": "supersede",
    "historical_of": "historical",
    "conflicts_with": "conflict",
    "conflicts": "conflict",
    "related_to": "related",
    "supports": "related",
}


class ResponseParseError(Exception):
    """The model response was not the JSON object this stage expects."""


def parse_candidates(
    text: str,
    *,
    allowed_message_ids: set[str],
    allowed_observation_ids: set[str],
) -> tuple[list[CandidateMemory], int]:
    """Return kept candidates and how many items were dropped.

    A candidate survives when it has a known type, a statement, and at least
    one message id or observation id from this segment. Message ids the
    segment does not contain are removed before that check.
    """
    payload = _load_object(text)
    if "memories" not in payload and "candidates" not in payload:
        raise ResponseParseError("JSON object is missing memories")
    items = payload.get("memories")
    if items is None:
        items = payload.get("candidates")
    if items is None:
        items = []
    if not isinstance(items, list):
        raise ResponseParseError("memories must be a list")

    kept: list[CandidateMemory] = []
    dropped = 0
    for item in items:
        parsed = _one_candidate(
            item,
            allowed_message_ids=allowed_message_ids,
            allowed_observation_ids=allowed_observation_ids,
        )
        if parsed is None:
            dropped += 1
            continue
        kept.append(parsed)
        if len(kept) >= MAX_CANDIDATES_PER_SEGMENT:
            dropped += max(0, len(items) - len(kept) - dropped)
            break
    return kept, dropped


def parse_merge_actions(text: str) -> list[MergeAction]:
    """Return explicit actions. Missing candidates are filled in by the merger."""
    payload = _load_object(text)
    if "actions" not in payload:
        raise ResponseParseError("JSON object is missing actions")
    items = payload["actions"]
    if items is None:
        items = []
    if not isinstance(items, list):
        raise ResponseParseError("actions must be a list")
    actions: list[MergeAction] = []
    for item in items:
        parsed = _one_action(item)
        if parsed is not None:
            actions.append(parsed)
    return actions


def _one_candidate(
    item,
    *,
    allowed_message_ids: set[str],
    allowed_observation_ids: set[str],
) -> CandidateMemory | None:
    if not isinstance(item, dict):
        return None
    memory_type = str(item.get("memory_type", item.get("type", ""))).strip().lower()
    if memory_type not in MEMORY_TYPE_SET:
        return None
    statement = item.get("statement", item.get("summary"))
    if not isinstance(statement, str):
        return None
    statement = " ".join(statement.strip().split())
    if not statement:
        return None
    statement = statement[:MAX_STATEMENT_CHARS]
    message_ids = filter_ids(item.get("evidence_message_ids", item.get("evidence_ids")), allowed_message_ids)
    observation_ids = filter_ids(item.get("observation_ids"), allowed_observation_ids)
    if not message_ids and not observation_ids:
        return None
    start, end = _time_range(item.get("time_range"))
    requested = item.get("time_precision")
    precision = str(requested).strip().lower() if isinstance(requested, str) else ""
    if precision not in TIME_PRECISION_SET:
        precision = time_precision_for(start, end, None)
    else:
        precision = time_precision_for(start, end, precision)
    situation = _optional_line(item.get("situation"), limit=300)
    return CandidateMemory(
        memory_type=memory_type,
        statement=statement,
        situation=situation,
        importance=clamp_unit(item.get("importance")),
        confidence=clamp_unit(item.get("confidence")),
        time_start=start,
        time_end=end,
        time_precision=precision,
        evidence_message_ids=message_ids,
        observation_ids=observation_ids,
        entities=_entities(item.get("entities")),
    )


def _entities(value) -> list[EntityMention]:
    if not isinstance(value, list):
        return []
    mentions: list[EntityMention] = []
    seen: set[tuple[str, str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        name = " ".join(name.strip().split())
        if not name:
            continue
        name = name[:80]
        entity_type = str(item.get("type", item.get("entity_type", ""))).strip().lower()
        if entity_type not in ENTITY_TYPE_SET:
            entity_type = "other"
        role = str(item.get("role", "")).strip().lower()
        if role not in ENTITY_ROLE_SET:
            role = "mentioned"
        status = str(item.get("status", "")).strip().lower()
        if status not in ENTITY_STATUS_SET:
            status = "unknown"
        key = (entity_type, name.casefold(), role)
        if key in seen:
            continue
        seen.add(key)
        aliases: list[str] = []
        raw_aliases = item.get("aliases")
        if isinstance(raw_aliases, list):
            for alias in raw_aliases:
                if not isinstance(alias, str):
                    continue
                cleaned = " ".join(alias.strip().split())
                if cleaned and cleaned.casefold() != name.casefold():
                    aliases.append(cleaned[:80])
        mentions.append(
            EntityMention(
                name=name,
                entity_type=entity_type,
                role=role,
                status=status,
                description=_optional_line(item.get("description"), limit=300),
                aliases=aliases[:20],
            )
        )
    return mentions


def _one_action(item) -> MergeAction | None:
    if not isinstance(item, dict):
        return None
    try:
        index = int(item.get("candidate_index"))
    except (TypeError, ValueError):
        return None
    if index < 0:
        return None
    action = str(item.get("action", "")).strip().lower()
    action = _ACTION_ALIASES.get(action, action)
    if action not in MERGE_ACTION_SET:
        action = "insert"
    target = item.get("target_memory_id")
    target_id = None
    if target is not None and str(target).strip() and str(target).strip().lower() != "null":
        target_id = str(target).strip()
    return MergeAction(
        candidate_index=index,
        action=action,
        target_memory_id=target_id,
        note=_optional_line(item.get("note"), limit=MAX_NOTE_CHARS),
    )


def _optional_line(value, *, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    if not text:
        return None
    return text[:limit]


def _load_object(text: str) -> dict:
    if text is None or not str(text).strip():
        raise ResponseParseError("model returned an empty response")
    stripped = _FENCE.sub("", str(text).strip()).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ResponseParseError("model did not return JSON") from None
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ResponseParseError("model did not return JSON") from exc
    if not isinstance(payload, dict):
        raise ResponseParseError("JSON must be an object")
    return payload


def _time_range(value) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    start = end = None
    if isinstance(value, dict):
        start = _unix(value.get("start", value.get("start_time")))
        end = _unix(value.get("end", value.get("end_time")))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        start = _unix(value[0])
        end = _unix(value[1])
    if start is not None and end is not None and start > end:
        start, end = end, start
    return start, end


def _unix(value) -> int | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return None
    if abs(number) > 10_000_000_000:
        number //= 1000
    return number
