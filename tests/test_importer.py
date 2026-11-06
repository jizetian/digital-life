"""Importer tests against the 300-message Moect fixture and a few synthetic files."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from common.schemas import REPORT_FIELDS
from importer.import_service import run_import
from importer.json_adapter import iter_moect_events
from importer.schema_detector import SchemaError
from tests.conftest import FIXTURE, ROOT

TARGET_WXID = "wxid_k52skna9jsse22"
OWNER_WXID = "wxid_jx1tm6udiumo22"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def test_fixture_exists():
    assert FIXTURE.is_file()
    assert FIXTURE.stat().st_size > 1000


def test_stream_parser_reads_metadata_and_messages_without_json_load(monkeypatch):
    def forbid_load(*args, **kwargs):
        raise AssertionError("json.load must not read the export")

    monkeypatch.setattr(json, "load", forbid_load)
    events = iter_moect_events(FIXTURE, chunk_size=128)
    kind, metadata = next(events)
    assert kind == "metadata"
    assert metadata["application"] == "Moect"
    assert metadata["version"] == "1.2.0"
    assert metadata["session"] == TARGET_WXID
    assert metadata["name"] == "不吃香菜"
    assert metadata["time_zone"] == "Asia/Shanghai"
    messages = [(index, raw) for event, index, raw in events]
    assert len(messages) == 300
    assert messages[0][0] == 0
    assert messages[0][1]["kind"] == "system"
    assert messages[0][1]["text"].startswith("你已添加了")
    text = next(raw for _index, raw in messages if raw["kind"] == "text")
    assert text["text"]
    assert messages[-1][0] == 299


def test_stream_parser_handles_multibyte_chunk_boundaries(tmp_path: Path):
    payload = {
        "metadata": {"application": "Moect", "session": "wxid_目标", "name": "不吃香菜"},
        "messages": [
            {
                "server_id": "1",
                "local_id": 1,
                "text": "设计滴",
                "kind": "text",
                "is_sender": False,
                "sender": "不吃香菜",
                "sender_wxid": "wxid_目标",
                "create_time": 10,
            }
        ],
    }
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    events = list(iter_moect_events(path, chunk_size=1))
    assert events[0][0] == "metadata"
    assert events[1][2]["text"] == "设计滴"


def test_adapter_does_not_slurp_the_file(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("full-file read")

    monkeypatch.setattr(Path, "read_text", deny)
    monkeypatch.setattr(Path, "read_bytes", deny)
    metadata = next(iter_moect_events(FIXTURE, chunk_size=1024))
    assert metadata[1]["application"] == "Moect"


def test_import_fixture_mapping_stats_and_idempotency(tmp_path: Path, monkeypatch):
    def forbid_load(*args, **kwargs):
        raise AssertionError("json.load must not read the export")

    monkeypatch.setattr(json, "load", forbid_load)
    db_path = tmp_path / "cleaned_messages.db"
    first = run_import(FIXTURE, db_path, batch_size=50, chunk_size=256)
    stats = first["stats"]
    for field in REPORT_FIELDS:
        assert field in stats
    assert first["status"] == "completed"
    assert stats["total_messages"] == 300
    assert stats["success_count"] == 300
    assert stats["failed_count"] == 0
    assert stats["duplicate_count"] == 0
    assert stats["missing_timestamp_count"] == 0
    assert stats["missing_sender_count"] == 0
    assert stats["target_person_count"] == 156
    assert stats["owner_count"] == 144
    assert stats["target_person_ratio"] == pytest.approx(156 / 300)
    assert stats["text_ratio"] == pytest.approx(237 / 300)
    assert stats["kind_counts"] == {
        "emoji": 29,
        "image": 1,
        "quote": 27,
        "system": 4,
        "text": 237,
        "video": 2,
    }
    assert stats["type_name_counts"]["文本"] == 237
    assert stats["type_name_counts"]["动画表情"] == 29
    assert stats["type_name_counts"]["应用消息"] == 27
    assert stats["type_name_counts"]["系统消息"] == 4

    conn = _connect(db_path)
    counts = conn.execute("SELECT COUNT(*) AS n, COUNT(DISTINCT message_id) AS distinct_n FROM messages").fetchone()
    assert counts["n"] == 300
    assert counts["distinct_n"] == 300
    first_row = conn.execute(
        "SELECT * FROM messages WHERE source_position = 0"
    ).fetchone()
    assert first_row["message_id"] == "5487675668292982957"
    assert first_row["sender_role"] == "target_person"
    assert first_row["is_sender"] == 0
    assert first_row["conversation_id"] == TARGET_WXID
    assert first_row["source_file"] == FIXTURE.name
    owner = conn.execute(
        "SELECT * FROM messages WHERE content = ?",
        ("数媒专业是学啥的呀",),
    ).fetchone()
    assert owner["sender_role"] == "owner"
    assert owner["sender_id"] == OWNER_WXID
    assert owner["sender_name"] == "不进三分不改名"
    assert owner["message_type"] == "text"
    target = conn.execute("SELECT * FROM messages WHERE content = ?", ("设计滴",)).fetchone()
    assert target["sender_role"] == "target_person"
    assert target["sender_id"] == TARGET_WXID
    assert target["sender_name"] == "不吃香菜"
    forwarded = conn.execute("SELECT COUNT(*) AS n FROM messages WHERE is_forwarded = 1").fetchone()
    assert forwarded["n"] == 0
    raw = json.loads(owner["raw_content"])
    assert "avatar_url" not in raw
    assert raw["text"] == "数媒专业是学啥的呀"
    emoji = conn.execute(
        "SELECT resource_status FROM messages WHERE message_type = 'emoji' LIMIT 1"
    ).fetchone()
    assert emoji["resource_status"] == "skipped"

    mapping = {
        row["sender_id"]: row
        for row in conn.execute("SELECT * FROM sender_mapping")
    }
    assert set(mapping) == {TARGET_WXID, OWNER_WXID}
    assert mapping[TARGET_WXID]["role"] == "target_person"
    assert mapping[TARGET_WXID]["sender_name"] == "不吃香菜"
    assert mapping[TARGET_WXID]["from_metadata"] == 1
    assert mapping[TARGET_WXID]["from_message"] == 1
    assert mapping[TARGET_WXID]["is_sender"] == 0
    assert mapping[OWNER_WXID]["role"] == "owner"
    assert mapping[OWNER_WXID]["from_metadata"] == 0
    assert mapping[OWNER_WXID]["from_message"] == 1
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"messages", "import_jobs", "import_errors", "sender_mapping", "processing_jobs"} <= tables
    conn.close()

    second = run_import(FIXTURE, db_path, batch_size=80)
    assert second["job_id"] != first["job_id"]
    assert second["stats"]["success_count"] == 0
    assert second["stats"]["duplicate_count"] == 300
    assert second["stats"]["total_messages"] == 300
    assert second["stats"]["target_person_ratio"] == pytest.approx(156 / 300)
    assert second["stats"]["text_ratio"] == pytest.approx(237 / 300)
    conn = _connect(db_path)
    again = conn.execute("SELECT COUNT(*) AS n, COUNT(DISTINCT message_id) AS distinct_n FROM messages").fetchone()
    assert again["n"] == 300
    assert again["distinct_n"] == 300
    conn.execute("DELETE FROM messages WHERE message_id = ?", ("5487675668292982957",))
    conn.commit()
    conn.close()
    third = run_import(FIXTURE, db_path)
    assert third["stats"]["success_count"] == 1
    assert third["stats"]["duplicate_count"] == 299
    conn = _connect(db_path)
    restored = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()
    assert restored["n"] == 300
    conn.close()


def test_resume_continues_the_same_job(tmp_path: Path):
    db_path = tmp_path / "resume.db"
    paused = run_import(FIXTURE, db_path, batch_size=10, max_messages=100)
    assert paused["status"] == "paused"
    assert paused["last_position"] == 99
    assert paused["stats"]["total_messages"] == 100
    conn = _connect(db_path)
    partial = conn.execute("SELECT COUNT(*) AS n, MAX(source_position) AS max_pos FROM messages").fetchone()
    assert partial["n"] == 100
    assert partial["max_pos"] == 99
    conn.close()

    finished = run_import(FIXTURE, db_path, resume=True, batch_size=10)
    assert finished["job_id"] == paused["job_id"]
    assert finished["status"] == "completed"
    assert finished["stats"]["total_messages"] == 300
    assert finished["stats"]["success_count"] == 300
    assert finished["stats"]["duplicate_count"] == 0
    assert finished["stats"]["failed_count"] == 0
    conn = _connect(db_path)
    final = conn.execute(
        "SELECT COUNT(*) AS n, COUNT(DISTINCT message_id) AS distinct_n, MIN(source_position) AS min_pos, MAX(source_position) AS max_pos FROM messages"
    ).fetchone()
    assert final["n"] == 300
    assert final["distinct_n"] == 300
    assert final["min_pos"] == 0
    assert final["max_pos"] == 299
    conn.close()


def test_resume_refuses_a_changed_file(tmp_path: Path):
    source = tmp_path / "export.json"
    source.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    db_path = tmp_path / "changed.db"
    run_import(source, db_path, max_messages=20, batch_size=5)
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(Exception, match="file size changed"):
        run_import(source, db_path, resume=True)


def test_synthetic_failures_duplicates_and_metadata_default(tmp_path: Path):
    payload = {
        "metadata": {"application": "Moect", "session": "wxid_target", "name": "目标"},
        "messages": [
            {
                "local_id": 1,
                "server_id": "a",
                "session": "wxid_target",
                "text": "hi",
                "kind": "text",
                "type_name": "文本",
                "is_sender": False,
                "sender": "目标",
                "sender_wxid": "wxid_target",
                "create_time": 10,
            },
            {
                "local_id": 2,
                "server_id": "b",
                "session": "wxid_target",
                "text": "yo",
                "kind": "text",
                "type_name": "文本",
                "is_sender": True,
                "sender": "我",
                "sender_wxid": "wxid_me",
            },
            {
                "local_id": 3,
                "server_id": "a",
                "session": "wxid_target",
                "text": "dup",
                "kind": "text",
                "type_name": "文本",
                "is_sender": True,
                "sender": "我",
                "sender_wxid": "wxid_me",
                "create_time": 12,
            },
            {"kind": "text", "text": "no id"},
            {
                "local_id": 5,
                "server_id": "c",
                "session": "wxid_target",
                "text": "anon",
                "kind": "text",
                "type_name": "文本",
                "create_time": 13,
            },
        ],
    }
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    db_path = tmp_path / "synthetic.db"
    job = run_import(path, db_path, batch_size=2)
    stats = job["stats"]
    assert stats["total_messages"] == 5
    assert stats["failed_count"] == 1
    assert stats["success_count"] == 3
    assert stats["duplicate_count"] == 1
    assert stats["missing_timestamp_count"] == 1
    assert stats["missing_sender_count"] == 1
    conn = _connect(db_path)
    errors = conn.execute("SELECT COUNT(*) AS n FROM import_errors WHERE job_id = ?", (job["job_id"],)).fetchone()
    assert errors["n"] == 1
    unknown = conn.execute("SELECT sender_role FROM messages WHERE message_id = 'c'").fetchone()
    assert unknown["sender_role"] == "unknown"
    conn.close()

    empty = {"metadata": {"application": "Moect", "session": "wxid_only", "name": "仅元数据"}, "messages": []}
    empty_path = tmp_path / "empty.json"
    empty_path.write_text(json.dumps(empty), encoding="utf-8")
    empty_db = tmp_path / "empty.db"
    empty_job = run_import(empty_path, empty_db)
    assert empty_job["stats"]["total_messages"] == 0
    assert empty_job["stats"]["target_person_ratio"] is None
    assert empty_job["stats"]["text_ratio"] is None
    conn = _connect(empty_db)
    row = conn.execute("SELECT * FROM sender_mapping").fetchone()
    assert row["sender_id"] == "wxid_only"
    assert row["role"] == "target_person"
    assert row["sender_name"] == "仅元数据"
    assert row["from_metadata"] == 1
    assert row["from_message"] == 0
    assert row["is_sender"] is None
    conn.close()


def test_unsupported_application_fails_the_job(tmp_path: Path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"metadata": {"application": "SomethingElse"}, "messages": []}), encoding="utf-8")
    db_path = tmp_path / "bad.db"
    with pytest.raises(SchemaError):
        run_import(path, db_path)
    conn = _connect(db_path)
    job = conn.execute("SELECT status, error_message FROM import_jobs").fetchone()
    assert job["status"] == "failed"
    assert "SomethingElse" in job["error_message"]
    assert conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"] == 0
    conn.close()


def test_cli_import_report(tmp_path: Path):
    db_path = tmp_path / "cli.db"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.import_data",
            "--input",
            str(FIXTURE),
            "--db",
            str(db_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "总消息数 total_messages: 300" in result.stdout
    assert "成功 success_count: 300" in result.stdout
    assert "目标人物消息比例" in result.stdout
    assert "文本消息比例" in result.stdout
    assert "text: 237" in result.stdout
    assert db_path.is_file()
