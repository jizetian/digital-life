"""Twin replies with a fake model. Evidence ids must come from retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chat.agent import chat_turn
from chat.parser import ground_citations, parse_twin_response
from common.database import init_db
from distillation.llm import LLMError
from scripts.chat import main as chat_main
from scripts.serve import build_parser as serve_parser


HITS = [
    {
        "collection": "long_term_memories",
        "source_id": "mem-shanghai",
        "text": "[fact/active] 搬到了上海",
        "score": 0.2,
        "metadata": {
            "status": "active",
            "memory_type": "fact",
            "evidence_message_ids": ["m-real"],
        },
    }
]


class FakeLLM:
    model = "fake-twin"

    def __init__(self, payload: dict | None = None, plain: str | None = None) -> None:
        self.payload = payload
        self.plain = plain
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if self.plain is not None:
            return self.plain
        body = self.payload or {
            "reply": "我后来搬到上海了。",
            "cited_memory_ids": ["mem-shanghai", "mem-fake"],
            "cited_message_ids": ["m-real", "m-hallucinated"],
            "confidence": 0.91,
            "uncertainty": "",
        }
        return json.dumps(body, ensure_ascii=False)


def test_grounding_drops_ids_the_model_invented(tmp_path: Path):
    db_path = tmp_path / "twin.db"
    conn = init_db(db_path)
    conn.execute(
        """
        INSERT INTO l2_traits (
            trait_id, prompt_version, model_version, dimension, statement,
            situational_scope, confidence, evidence_count, independent_evidence_count,
            counter_evidence_count, evidence_ids_json, counter_evidence_ids_json,
            valid_from, valid_to, status, stages_json, variants_json, payload_json,
            created_at, updated_at
        ) VALUES ('t1', 'l2-v1', 'test-model', 'communication', '说话很短，先讲结论。',
                  '', 0.8, 2, 2, 0, '[]', '[]', NULL, NULL, 'stable', '[]', '[]', '{}',
                  '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
        """
    )
    conn.commit()
    conn.close()
    fake = FakeLLM()
    result = chat_turn(db_path, "你后来住哪", llm=fake, search=lambda query: HITS)
    assert result["reply"] == "我后来搬到上海了。"
    assert result["simulation"] is True
    assert result["target_name"] == "不吃香菜"
    assert result["evidence_message_ids"] == ["m-real"]
    assert result["cited_memory_ids"] == ["mem-shanghai"]
    assert "mem-fake" in result["rejected_ids"]
    assert "m-hallucinated" in result["rejected_ids"]
    assert {item["memory_id"] for item in result["retrieved_memories"]} == {"mem-shanghai"}
    assert result["retrieved_memories"][0]["cited"] is True
    assert result["confidence"] <= 0.5
    assert "说话很短" in fake.calls[0]["system"]
    assert "mem-fake" not in fake.calls[0]["system"]
    assert "m-hallucinated" not in fake.calls[0]["system"]

    again = chat_turn(
        db_path,
        "然后呢",
        session_id=result["session_id"],
        llm=fake,
        search=lambda query: HITS,
    )
    assert "你后来住哪" in fake.calls[1]["user"]
    assert again["session_id"] == result["session_id"]
    stored = init_db(db_path)
    try:
        count = stored.execute("SELECT COUNT(*) AS n FROM chat_messages").fetchone()["n"]
    finally:
        stored.close()
    assert count == 4


def test_weak_retrieval_caps_confidence_and_labels_the_simulation(tmp_path: Path):
    db_path = tmp_path / "weak.db"
    init_db(db_path).close()
    fake = FakeLLM(
        {
            "reply": "我住在火星。",
            "cited_memory_ids": ["mem-mars"],
            "cited_message_ids": ["m-mars"],
            "confidence": 0.99,
            "uncertainty": "",
        }
    )
    result = chat_turn(db_path, "你住在哪", llm=fake, search=lambda query: [])
    assert result["retrieval_weak"] is True
    assert result["confidence"] <= 0.35
    assert result["evidence_message_ids"] == []
    assert result["retrieved_memories"] == []
    assert "mem-mars" in result["rejected_ids"]
    assert "m-mars" in result["rejected_ids"]
    assert "AI 模拟" in result["reply"]
    assert result["uncertainty"]


def test_unstructured_model_text_cannot_introduce_evidence_ids(tmp_path: Path):
    db_path = tmp_path / "plain.db"
    init_db(db_path).close()
    fake = FakeLLM(plain="我后来搬到上海了，证据是 mem-fake 和 m-hallucinated。")
    result = chat_turn(db_path, "你后来住哪", llm=fake, search=lambda query: HITS)
    assert result["evidence_message_ids"] == []
    assert result["cited_memory_ids"] == []
    assert "mem-fake" not in result["evidence_message_ids"]
    assert "JSON" in (result["uncertainty"] or "")


def test_a_longer_question_still_hits_the_memory_phrase(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CHATTWIN_VECTOR_BACKEND", "memory")
    db_path = _indexed_memory(tmp_path / "slice.db")
    fake = FakeLLM()
    result = chat_turn(db_path, "你后来搬到了上海吗", llm=fake, use_vector=False)
    assert result["retrieval_weak"] is False
    assert result["cited_memory_ids"] == ["mem-shanghai"]
    assert result["evidence_message_ids"] == ["m-real"]
    assert result["confidence"] > 0.35


def test_hybrid_search_feeds_the_agent_without_a_live_model(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CHATTWIN_VECTOR_BACKEND", "memory")
    db_path = _indexed_memory(tmp_path / "index.db")
    fake = FakeLLM()
    result = chat_turn(db_path, "搬到了上海", llm=fake, use_vector=False)
    assert result["cited_memory_ids"] == ["mem-shanghai"]
    assert result["evidence_message_ids"] == ["m-real"]
    assert "mem-fake" in result["rejected_ids"]
    assert "搬到了上海" in fake.calls[0]["system"]


def _indexed_memory(db_path: Path) -> Path:
    conn = init_db(db_path)
    conn.execute(
        """
        INSERT INTO retrieval_documents (
            doc_id, collection, source_id, text, text_hash, content_hash,
            time_start, time_end, metadata_json, updated_at
        ) VALUES (
            'long_term_memories:mem-shanghai', 'long_term_memories', 'mem-shanghai',
            '[fact/active] 搬到了上海', 'h1', 'h2', NULL, NULL, ?,
            '2026-01-01T00:00:00+00:00'
        )
        """,
        (
            json.dumps(
                {"memory_type": "fact", "status": "active", "evidence_message_ids": ["m-real"]},
                ensure_ascii=False,
            ),
        ),
    )
    conn.commit()
    conn.close()
    return db_path


def test_parser_rejects_unknown_ids_and_keeps_known_ones():
    parsed = parse_twin_response(
        """```json
        {"reply":"好","cited_memory_ids":["mem-real","mem-nope"],"cited_message_ids":"m-real, m-nope","confidence":1.4}
        ```"""
    )
    grounded = ground_citations(
        parsed,
        allowed_memory_ids={"mem-real"},
        allowed_message_ids={"m-real"},
        memory_evidence={"mem-real": ["m-real"]},
    )
    assert parsed.structured is True
    assert parsed.confidence == 1
    assert grounded.cited_memory_ids == ["mem-real"]
    assert grounded.evidence_message_ids == ["m-real"]
    assert grounded.rejected_ids == ["mem-nope", "m-nope"]


def test_cli_one_shot_uses_the_same_grounding(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "cli.db"
    init_db(db_path).close()
    fake = FakeLLM()
    monkeypatch.setattr("chat.agent.build_llm_client", lambda **kwargs: fake)
    monkeypatch.setattr("chat.agent.search_hits", lambda *args, **kwargs: HITS)
    code = chat_main(["--db", str(db_path), "--message", "你后来住哪", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reply"] == "我后来搬到上海了。"
    assert payload["evidence_message_ids"] == ["m-real"]
    assert "m-hallucinated" not in payload["evidence_message_ids"]
    missing = chat_main(["--db", str(tmp_path / "missing.db"), "--message", "你好"])
    assert missing == 1


def test_empty_message_is_rejected(tmp_path: Path):
    db_path = tmp_path / "empty.db"
    init_db(db_path).close()
    with pytest.raises(ValueError, match="empty"):
        chat_turn(db_path, "  ", llm=FakeLLM(), search=lambda query: [])


def test_model_failure_keeps_the_user_turn(tmp_path: Path):
    db_path = tmp_path / "fail.db"
    init_db(db_path).close()

    class Broken:
        model = "broken"

        def complete(self, *, system: str, user: str) -> str:
            del system, user
            raise LLMError("LLM request timed out")

    with pytest.raises(Exception, match="timed out"):
        chat_turn(db_path, "你好", llm=Broken(), search=lambda query: [])
    conn = init_db(db_path)
    try:
        rows = conn.execute("SELECT role, content FROM chat_messages").fetchall()
    finally:
        conn.close()
    assert [(row["role"], row["content"]) for row in rows] == [("user", "你好")]


def test_serve_cannot_bind_a_public_host():
    with pytest.raises(SystemExit):
        serve_parser().parse_args(["--host", "0.0.0.0"])
