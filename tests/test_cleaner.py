"""Cleaner unit tests. These do not touch SQLite."""

from __future__ import annotations

import json

from cleaner.deduplicator import Deduplicator
from cleaner.identity_resolver import resolve_identity
from cleaner.message_cleaner import slim_raw
from cleaner.normalize import MessageNormalizeError, normalize_message
from cleaner.type_classifier import is_forwarded
from common.schemas import ROLE_OWNER, ROLE_TARGET, ROLE_UNKNOWN


def test_identity_follows_is_sender_only():
    assert resolve_identity({"is_sender": True}) == (ROLE_OWNER, True)
    assert resolve_identity({"is_sender": False}) == (ROLE_TARGET, False)
    # Same wxid as the session must not be guessed into a role.
    assert resolve_identity({"sender_wxid": "wxid_target", "session": "wxid_target"}) == (ROLE_UNKNOWN, None)


def test_message_id_server_id_then_session_local_id():
    with_server = normalize_message(
        {"server_id": " 99 ", "local_id": 1, "session": "s", "text": "a", "is_sender": True},
        source_file="a.json",
        source_position=0,
    )
    assert with_server.message_id == "99"
    fallback = normalize_message(
        {"local_id": 7, "session": "wxid_s", "text": "a", "is_sender": False},
        source_file="a.json",
        source_position=1,
    )
    assert fallback.message_id == "wxid_s:7"
    assert fallback.server_id is None


def test_missing_ids_are_a_record_error():
    try:
        normalize_message({"text": "orphan", "kind": "text"}, source_file="a.json", source_position=3)
    except MessageNormalizeError as exc:
        assert "server_id" in str(exc)
    else:
        raise AssertionError("expected MessageNormalizeError")


def test_missing_timestamp_and_sender_are_flags_not_inventions():
    message = normalize_message(
        {
            "server_id": "t1",
            "kind": "text",
            "text": "hi",
            "time_str": "2023-03-17 21:54:12",
            "type": 1,
            "type_name": "文本",
        },
        source_file="a.json",
        source_position=0,
    )
    assert message.timestamp is None
    assert message.missing_timestamp is True
    assert message.time_str == "2023-03-17 21:54:12"
    assert message.missing_sender is True
    assert message.sender_role == ROLE_UNKNOWN
    assert message.message_type == "text"
    # type code 1 is not used to invent a kind when kind is absent on another record
    untyped = normalize_message(
        {"server_id": "t2", "type": 1, "type_name": "文本", "text": "x", "is_sender": True, "create_time": 10},
        source_file="a.json",
        source_position=1,
    )
    assert untyped.message_type is None
    assert untyped.type_code == 1
    assert untyped.type_name == "文本"


def test_content_is_preserved_and_avatar_url_is_dropped():
    raw = {
        "server_id": "c1",
        "text": "  保留空格  ",
        "avatar_url": "https://example.invalid/very/long",
        "is_sender": False,
        "sender": "不吃香菜",
        "sender_wxid": "wxid_target",
        "quote": {"text": "quoted", "avatar_url": "https://example.invalid/avatar"},
        "create_time": 1,
        "kind": "quote",
    }
    message = normalize_message(raw, source_file="a.json", source_position=4)
    assert message.content == "  保留空格  "
    assert "avatar_url" not in json.dumps(message.raw_content)
    assert message.raw_content["quote"]["text"] == "quoted"
    assert message.sender_role == ROLE_TARGET


def test_forward_signals_are_structural():
    quote = {
        "kind": "quote",
        "type_name": "应用消息",
        "extra": {"appmsg_type": 57, "title": "回复"},
        "quote": {"forward": None, "text": "原文"},
        "text": "回复",
    }
    assert is_forwarded(quote) is False
    assert is_forwarded({"kind": "text", "text": "这是转发的一句话"}) is False
    assert is_forwarded({"kind": "forward", "text": "x"}) is True
    assert is_forwarded({"type_name": "聊天记录", "kind": "app"}) is True
    assert is_forwarded({"kind": "app", "extra": {"appmsg_type": 19}}) is True
    assert is_forwarded({"kind": "app", "extra": {"is_forwarded": True}}) is True
    assert is_forwarded({"kind": "app", "forward": {"title": "record"}}) is True


def test_slim_raw_keeps_non_avatar_fields():
    slimmed = slim_raw({"text": "a", "avatar_url": "drop", "resources": []})
    assert slimmed == {"text": "a", "resources": []}


def test_deduplicator_marks_only_the_second_sighting():
    deduper = Deduplicator()
    assert deduper.seen_in_file("a") is False
    assert deduper.seen_in_file("a") is True
    assert deduper.seen_in_file("b") is False
    assert len(deduper) == 2
