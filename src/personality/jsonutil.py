"""Pull one JSON object out of a model response without echoing the body."""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class ResponseParseError(Exception):
    """The model response was not the JSON object this stage expects."""


def load_json_payload(text: str) -> dict:
    if text is None or not str(text).strip():
        raise ResponseParseError("model returned an empty response")
    stripped = _FENCE.sub("", str(text).strip()).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(stripped[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ResponseParseError("model did not return JSON") from exc
        else:
            raise ResponseParseError("model did not return JSON") from None
    if not isinstance(payload, dict):
        raise ResponseParseError("JSON must be an object")
    return payload
