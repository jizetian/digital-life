"""Chat HTTP routes. The model and the retriever are fakes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from common.database import init_db
from tests.test_chat_agent import HITS, FakeLLM


def test_chat_page_and_happy_path(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "api.db"
    init_db(db_path).close()
    fake = FakeLLM()
    monkeypatch.setattr("chat.agent.build_llm_client", lambda **kwargs: fake)
    monkeypatch.setattr("chat.agent.search_hits", lambda *args, **kwargs: HITS)
    client = TestClient(create_app())

    page = client.get("/")
    assert page.status_code == 200
    assert "zh-Hans-CN" in page.text
    assert "不吃香菜" in page.text
    assert "AI 模拟" in page.text

    created = client.post(
        "/api/v1/chat",
        json={"db_path": str(db_path), "message": "你后来住哪"},
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["simulation"] is True
    assert body["reply"] == "我后来搬到上海了。"
    assert body["evidence_message_ids"] == ["m-real"]
    assert "mem-fake" not in json.dumps(body["retrieved_memories"], ensure_ascii=False)
    assert "m-hallucinated" not in body["evidence_message_ids"]
    assert body["prompt_version"] == "twin-v1"

    listed = client.get("/api/v1/chat/sessions", params={"db": str(db_path)})
    assert listed.status_code == 200
    assert listed.json()["sessions"][0]["session_id"] == body["session_id"]
    assert listed.json()["sessions"][0]["title"] == "你后来住哪"

    history = client.get(f"/api/v1/chat/sessions/{body['session_id']}", params={"db": str(db_path)})
    assert history.status_code == 200, history.text
    messages = history.json()["messages"]
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[1]["evidence_message_ids"] == ["m-real"]
    assert messages[1]["confidence"] == body["confidence"]

    missing_session = client.get("/api/v1/chat/sessions/does-not-exist", params={"db": str(db_path)})
    assert missing_session.status_code == 404


def test_chat_routes_reject_a_missing_database_and_an_empty_message(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("chat.agent.search_hits", lambda *args, **kwargs: [])
    client = TestClient(create_app())
    missing = tmp_path / "missing.db"
    posted = client.post("/api/v1/chat", json={"db_path": str(missing), "message": "你好"})
    assert posted.status_code == 400
    listed = client.get("/api/v1/chat/sessions", params={"db": str(missing)})
    assert listed.status_code == 404
    db_path = tmp_path / "blank.db"
    init_db(db_path).close()
    blank = client.post("/api/v1/chat", json={"db_path": str(db_path), "message": "  "})
    assert blank.status_code == 400
    unknown = client.post(
        "/api/v1/chat",
        json={"db_path": str(db_path), "message": "你好", "session_id": "missing-session"},
    )
    assert unknown.status_code == 404
