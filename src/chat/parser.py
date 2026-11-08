"""Parse the twin JSON and drop any evidence id the prompt did not list."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(slots=True)
class ParsedReply:
    reply: str
    cited_memory_ids: list[str]
    cited_message_ids: list[str]
    rejected_ids: list[str]
    confidence: float
    uncertainty: str
    structured: bool


@dataclass(slots=True)
class GroundedCitations:
    cited_memory_ids: list[str] = field(default_factory=list)
    cited_message_ids: list[str] = field(default_factory=list)
    evidence_message_ids: list[str] = field(default_factory=list)
    rejected_ids: list[str] = field(default_factory=list)


def parse_twin_response(text: str) -> ParsedReply:
    """Read the model JSON. Unstructured text becomes a reply with no citations."""
    payload = _load_object(text)
    if payload is None:
        reply = " ".join((text or "").strip().split())
        return ParsedReply(
            reply=reply,
            cited_memory_ids=[],
            cited_message_ids=[],
            rejected_ids=[],
            confidence=0.2,
            uncertainty="模型没有返回可解析的 JSON，因此没有采纳任何证据编号。",
            structured=False,
        )
    reply = _reply_text(payload)
    if not reply:
        reply = " ".join((text or "").strip().split())
        return ParsedReply(
            reply=reply,
            cited_memory_ids=[],
            cited_message_ids=[],
            rejected_ids=[],
            confidence=0.2,
            uncertainty="模型没有返回可解析的 JSON，因此没有采纳任何证据编号。",
            structured=False,
        )
    return ParsedReply(
        reply=reply,
        cited_memory_ids=_raw_ids(payload, ("cited_memory_ids", "memory_ids")),
        cited_message_ids=_raw_ids(payload, ("cited_message_ids", "evidence_message_ids", "message_ids")),
        rejected_ids=[],
        confidence=_confidence(payload.get("confidence")),
        uncertainty=_uncertainty(payload.get("uncertainty", payload.get("note"))),
        structured=True,
    )


def ground_citations(
    parsed: ParsedReply,
    *,
    allowed_memory_ids: set[str],
    allowed_message_ids: set[str],
    memory_evidence: dict[str, list[str]],
) -> GroundedCitations:
    """Keep ids that retrieval put in the prompt. Everything else is rejected.

    A cited memory also contributes its own evidence message ids. Those ids
    were already on the hit, so they are not a new claim from the model.
    """
    grounded = GroundedCitations()
    seen_rejected: set[str] = set()

    def take(raw: str, origin: str) -> None:
        if raw in allowed_memory_ids and (origin == "memory" or raw not in allowed_message_ids):
            if raw not in grounded.cited_memory_ids:
                grounded.cited_memory_ids.append(raw)
            return
        if raw in allowed_message_ids:
            if raw not in grounded.cited_message_ids:
                grounded.cited_message_ids.append(raw)
            return
        if raw not in seen_rejected:
            seen_rejected.add(raw)
            grounded.rejected_ids.append(raw)

    for raw in parsed.cited_memory_ids:
        take(raw, "memory")
    for raw in parsed.cited_message_ids:
        take(raw, "message")

    for memory_id in grounded.cited_memory_ids:
        for message_id in memory_evidence.get(memory_id, []):
            if message_id in allowed_message_ids and message_id not in grounded.evidence_message_ids:
                grounded.evidence_message_ids.append(message_id)
    for message_id in grounded.cited_message_ids:
        if message_id not in grounded.evidence_message_ids:
            grounded.evidence_message_ids.append(message_id)
    return grounded


def _load_object(text: str) -> dict | None:
    if text is None or not str(text).strip():
        return None
    stripped = _FENCE.sub("", str(text).strip()).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    return payload


def _reply_text(payload: dict) -> str:
    for key in ("reply", "content", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _raw_ids(payload: dict, keys: tuple[str, ...]) -> list[str]:
    value = None
    for key in keys:
        if key in payload:
            value = payload.get(key)
            break
    return _id_list(value)


def _id_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, list):
        return []
    found: list[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, dict):
            item = item.get("memory_id", item.get("message_id", item.get("id")))
        if item is None or isinstance(item, bool):
            continue
        text = str(item).strip()
        if not text or text.lower() == "null" or text in seen:
            continue
        seen.add(text)
        found.append(text)
    return found


def _confidence(value) -> float:
    if value is None or isinstance(value, bool) or value == "":
        return 0.4
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.4
    if number < 0:
        number = 0.0
    if number > 1:
        number = 1.0
    return number


def _uncertainty(value) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())
