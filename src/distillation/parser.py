"""Parse an L1 JSON response and keep only evidence that cites real message ids."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from distillation.categories import L1_CATEGORY_SET

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class ResponseParseError(Exception):
    """The model response was not the JSON object this stage expects."""


@dataclass(slots=True)
class Observation:
    category: str
    summary: str
    evidence_message_ids: list[str]
    confidence: float | None
    time_range_start: int | None
    time_range_end: int | None


def parse_observations(text: str, *, allowed_message_ids: set[str]) -> tuple[list[Observation], int]:
    """Return kept observations and how many items were dropped.

    Unknown message ids are removed. An item with no remaining evidence is
    dropped so every stored observation points at messages in the segment.
    """
    if text is None or not str(text).strip():
        raise ResponseParseError("model returned an empty response")
    payload = _load_payload(str(text))
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        if "observations" not in payload:
            raise ResponseParseError("JSON object is missing observations")
        items = payload["observations"]
        if items is None:
            items = []
    else:
        raise ResponseParseError("JSON must be an object or a list")
    if not isinstance(items, list):
        raise ResponseParseError("observations must be a list")

    kept: list[Observation] = []
    dropped = 0
    for item in items:
        parsed = _one(item, allowed_message_ids)
        if parsed is None:
            dropped += 1
        else:
            kept.append(parsed)
    return kept, dropped


def _load_payload(text: str):
    stripped = text.strip()
    stripped = _FENCE.sub("", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ResponseParseError("model did not return JSON") from exc
        raise ResponseParseError("model did not return JSON")


def _one(item, allowed_message_ids: set[str]) -> Observation | None:
    if not isinstance(item, dict):
        return None
    category = str(item.get("category", "")).strip().lower()
    if category not in L1_CATEGORY_SET:
        return None
    summary = item.get("summary", item.get("observation"))
    if not isinstance(summary, str) or not summary.strip():
        return None
    raw_ids = item.get("evidence_message_ids", item.get("evidence_ids"))
    if not isinstance(raw_ids, list):
        return None
    evidence: list[str] = []
    seen: set[str] = set()
    for value in raw_ids:
        if value is None:
            continue
        message_id = str(value)
        if message_id in allowed_message_ids and message_id not in seen:
            seen.add(message_id)
            evidence.append(message_id)
    if not evidence:
        return None
    start, end = _time_range(item.get("time_range"))
    return Observation(
        category=category,
        summary=summary.strip(),
        evidence_message_ids=evidence,
        confidence=_confidence(item.get("confidence")),
        time_range_start=start,
        time_range_end=end,
    )


def _confidence(value) -> float | None:
    if value is None or value == "":
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
