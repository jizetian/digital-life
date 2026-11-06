"""Trim storage-only fields. Original text is kept verbatim in content and raw_content."""

from __future__ import annotations

_DROP_KEYS = frozenset({"avatar_url"})


def slim_raw(value):
    """Return a JSON-ready copy with avatar URLs removed."""
    if isinstance(value, dict):
        return {key: slim_raw(item) for key, item in value.items() if key not in _DROP_KEYS}
    if isinstance(value, list):
        return [slim_raw(item) for item in value]
    return value


def as_int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def as_text(value) -> str | None:
    if not isinstance(value, str):
        return None
    if value.strip() == "":
        return None
    return value


def as_optional_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return None


def message_content(raw: dict) -> str | None:
    text = raw.get("text")
    if text is None:
        return None
    if isinstance(text, str):
        return text
    return None
