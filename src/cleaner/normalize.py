"""Build a NormalizedMessage from one raw Moect object."""

from __future__ import annotations

from cleaner.identity_resolver import resolve_identity, sender_is_missing
from cleaner.message_cleaner import as_int, as_text, message_content, slim_raw
from cleaner.type_classifier import is_forwarded, message_type_of, type_name_of
from common.schemas import NormalizedMessage


class MessageNormalizeError(Exception):
    """This record cannot be stored. The rest of the file can continue."""


def build_message_id(raw: dict) -> str:
    server_id = raw.get("server_id")
    if isinstance(server_id, str) and server_id.strip():
        return server_id.strip()
    if isinstance(server_id, int) and not isinstance(server_id, bool):
        return str(server_id)
    local_id = as_int(raw.get("local_id"))
    if local_id is None:
        raise MessageNormalizeError("missing server_id and local_id")
    session = raw.get("session")
    if isinstance(session, str) and session.strip():
        return f"{session.strip()}:{local_id}"
    return str(local_id)


def normalize_message(raw: dict, *, source_file: str, source_position: int) -> NormalizedMessage:
    if not isinstance(raw, dict):
        raise MessageNormalizeError(f"message at {source_position} is not an object")
    message_id = build_message_id(raw)
    sender_id = as_text(raw.get("sender_wxid"))
    sender_name = as_text(raw.get("sender"))
    role, is_sender_flag = resolve_identity(raw)
    timestamp = as_int(raw.get("create_time")) if "create_time" in raw else None
    if "create_time" in raw and as_int(raw.get("create_time")) is None and raw.get("create_time") is not None:
        timestamp = None
    missing_timestamp = timestamp is None
    time_str = raw.get("time_str") if isinstance(raw.get("time_str"), str) else None
    server_id = raw.get("server_id")
    if isinstance(server_id, str):
        stored_server_id = server_id.strip() or None
    elif isinstance(server_id, int) and not isinstance(server_id, bool):
        stored_server_id = str(server_id)
    else:
        stored_server_id = None
    session = as_text(raw.get("session"))
    resource_status = raw.get("resource_status")
    if not isinstance(resource_status, str) or resource_status.strip() == "":
        resource_status = None
    resource_notice = raw.get("resource_notice")
    if not isinstance(resource_notice, str) or resource_notice.strip() == "":
        resource_notice = None
    return NormalizedMessage(
        message_id=message_id,
        conversation_id=session,
        local_id=as_int(raw.get("local_id")),
        server_id=stored_server_id,
        shard=as_int(raw.get("shard")),
        sender_id=sender_id,
        sender_name=sender_name,
        sender_role=role,
        is_sender=is_sender_flag,
        timestamp=timestamp,
        time_str=time_str,
        message_type=message_type_of(raw),
        type_code=as_int(raw.get("type")),
        raw_type=as_int(raw.get("raw_type")),
        type_name=type_name_of(raw),
        content=message_content(raw),
        raw_content=slim_raw(raw),
        is_forwarded=is_forwarded(raw),
        resource_status=resource_status,
        resource_notice=resource_notice,
        source_file=source_file,
        source_position=source_position,
        missing_timestamp=missing_timestamp,
        missing_sender=sender_is_missing(sender_id, sender_name),
    )
