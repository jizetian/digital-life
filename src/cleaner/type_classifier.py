"""Classify kind and forwarded messages from fields the export actually has."""

from __future__ import annotations

# WeChat appmsg type 19 is a merged chat-history forward. Quote replies are 57
# and are not forwards. Only explicit structural signals flip is_forwarded.
_FORWARD_KINDS = frozenset({"forward", "forwarded", "chat_record", "chat_history"})
_FORWARD_TYPE_NAMES = frozenset({"聊天记录", "合并转发", "转发"})
_FORWARD_EXTRA_KEYS = ("is_forward", "is_forwarded", "forwarded")


def message_type_of(raw: dict) -> str | None:
    kind = raw.get("kind")
    if isinstance(kind, str) and kind.strip():
        return kind.strip()
    return None


def type_name_of(raw: dict) -> str | None:
    name = raw.get("type_name")
    if isinstance(name, str) and name.strip():
        return name
    return None


def is_forwarded(raw: dict) -> bool:
    if not isinstance(raw, dict):
        return False
    kind = raw.get("kind")
    if isinstance(kind, str) and kind.strip() in _FORWARD_KINDS:
        return True
    type_name = raw.get("type_name")
    if isinstance(type_name, str) and type_name.strip() in _FORWARD_TYPE_NAMES:
        return True
    if _truthy_forward(raw.get("forward")):
        return True
    extra = raw.get("extra")
    if isinstance(extra, dict):
        for key in _FORWARD_EXTRA_KEYS:
            if extra.get(key) is True:
                return True
        if _truthy_forward(extra.get("forward")):
            return True
        if extra.get("appmsg_type") == 19:
            return True
    return False


def _truthy_forward(value) -> bool:
    if value is True:
        return True
    if isinstance(value, dict) and any(item not in (None, "", [], {}) for item in value.values()):
        return True
    return False
