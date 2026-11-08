"""L1 distillation with a fake model: persist, skip, resume, and retry."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import httpx
import pytest

from common.database import init_db
from distillation.categories import L1_CATEGORIES
from distillation.engine import DistillationError, make_task_id, run_distillation
from distillation.llm import LLMError, OpenAICompatibleClient
from distillation.parser import ResponseParseError, parse_observations
from distillation.prompting import PROMPT_VERSION, load_system_prompt
from scripts.distill_l1 import main as distill_main
from segmentation.service import run_segmentation
from segmentation.splitter import SegmentParams
from tests.support import seed_messages


class FakeLLM:
    def __init__(self, responder) -> None:
        self.responder = responder
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self.responder(system, user)


def _fact_response(system: str, user: str) -> str:
    del system
    message_ids = re.findall(r"id=(\S+)", user)
    return json.dumps(
        {
            "observations": [
                {
                    "category": "Fact",
                    "summary": "提到了一件事",
                    "evidence_message_ids": [message_ids[0], "not-a-real-message"],
                    "confidence": 1.4,
                }
            ]
        },
        ensure_ascii=False,
    )


def _three_segments(tmp_path: Path) -> Path:
    db_path = tmp_path / "three.db"
    seed_messages(
        db_path,
        [
            {"message_id": "a", "conversation_id": "c1", "timestamp": 10, "content": "第一句"},
            {"message_id": "b", "conversation_id": "c2", "timestamp": 20, "content": "第二句"},
            {"message_id": "c", "conversation_id": "c3", "timestamp": 30, "content": "第三句"},
        ],
    )
    result = run_segmentation(db_path, params=SegmentParams(gap_seconds=1800, max_messages=50, overlap_messages=3))
    assert result["segment_count"] == 3
    return db_path


def test_prompt_lists_l1_categories_and_task_id_formula():
    text = load_system_prompt()
    assert text.startswith(f"prompt_version: {PROMPT_VERSION}")
    for category in L1_CATEGORIES:
        assert category in text
    assert len(L1_CATEGORIES) == 10
    assert make_task_id("seg", "l1-v1", "demo") == hashlib.sha256(b"seg|l1-v1|demo").hexdigest()


def test_parser_keeps_only_traceable_evidence():
    payload = """```json
Here you go:
{"observations":[
  {"category":"emotion","summary":"有点着急","evidence_message_ids":["m1","nope"],"confidence":"0.5","time_range":{"start":2000,"end":1000}},
  {"category":"not-real","summary":"丢弃","evidence_message_ids":["m1"]},
  {"category":"fact","summary":"   ","evidence_message_ids":["m1"]},
  {"category":"event","summary":"没有证据","evidence_message_ids":["missing"]}
]}
```"""
    observations, dropped = parse_observations(payload, allowed_message_ids={"m1", "m2"})
    assert dropped == 3
    assert len(observations) == 1
    assert observations[0].category == "emotion"
    assert observations[0].evidence_message_ids == ["m1"]
    assert observations[0].confidence == 0.5
    assert observations[0].time_range_start == 1000
    assert observations[0].time_range_end == 2000
    with pytest.raises(ResponseParseError):
        parse_observations("not json", allowed_message_ids={"m1"})


def test_parser_accepts_millisecond_times():
    observations, dropped = parse_observations(
        json.dumps(
            {
                "observations": [
                    {
                        "category": "interest",
                        "summary": "喜欢走路",
                        "evidence_ids": ["m1"],
                        "time_range": [1_679_061_252_000, 1_679_061_300_000],
                    }
                ]
            }
        ),
        allowed_message_ids={"m1"},
    )
    assert dropped == 0
    assert observations[0].time_range_start == 1_679_061_252
    assert observations[0].time_range_end == 1_679_061_300


def test_openai_compatible_client_posts_chat_completions_without_echoing_the_chat():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{\"observations\": []}"}}]})

    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:11434/v1/",
        model="qwen2.5:7b",
        api_key="secret",
        temperature=0,
        transport=httpx.MockTransport(handler),
    )
    assert client.complete(system="sys", user="user") == "{\"observations\": []}"
    assert seen["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["body"]["model"] == "qwen2.5:7b"

    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"echo": json.loads(request.content.decode())["messages"][1]["content"]})

    failing_client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5:7b",
        transport=httpx.MockTransport(failing),
    )
    with pytest.raises(LLMError) as caught:
        failing_client.complete(system="sys", user="PRIVATE_CHAT_LINE")
    assert "PRIVATE_CHAT_LINE" not in str(caught.value)
    assert "500" in str(caught.value)


def test_distillation_persists_evidence_and_skips_blank_segments(tmp_path: Path):
    db_path = tmp_path / "obs.db"
    seed_messages(
        db_path,
        [
            {"message_id": "kept", "conversation_id": "c1", "timestamp": 100, "content": "今天去散步"},
            {"message_id": "blank", "conversation_id": "c1", "timestamp": 100 + 1800, "content": "   "},
        ],
    )
    segmented = run_segmentation(
        db_path,
        params=SegmentParams(gap_seconds=1800, max_messages=50, overlap_messages=3),
    )
    assert segmented["segment_count"] == 2
    fake = FakeLLM(_fact_response)
    result = run_distillation(
        db_path,
        client=fake,
        batch_size=2,
        model_version="test-model",
        prompt_version="l1-v1",
    )
    assert result["status"] == "completed"
    assert result["attempts_completed"] == 1
    assert result["tasks"]["skipped"] == 1
    assert result["tasks"]["completed"] == 1
    assert len(fake.calls) == 1
    assert "fact" in fake.calls[0]["system"]
    assert "id=kept" in fake.calls[0]["user"]
    assert "id=blank" not in fake.calls[0]["user"]

    conn = init_db(db_path)
    try:
        observation = conn.execute("SELECT * FROM l1_observations").fetchone()
        assert observation["category"] == "fact"
        assert json.loads(observation["evidence_message_ids_json"]) == ["kept"]
        assert observation["confidence"] == 1
        assert observation["time_range_start"] == 100
        assert observation["time_range_end"] == 100
        task = conn.execute(
            "SELECT task_id, segment_id, status FROM distillation_tasks WHERE status = 'completed'"
        ).fetchone()
        assert task["task_id"] == make_task_id(task["segment_id"], "l1-v1", "test-model")
        segment_ids = json.loads(
            conn.execute(
                "SELECT message_ids_json FROM conversation_segments WHERE segment_id = ?",
                (task["segment_id"],),
            ).fetchone()["message_ids_json"]
        )
        assert "kept" in segment_ids
    finally:
        conn.close()

    run_distillation(db_path, client=fake, model_version="test-model", prompt_version="l1-v1")
    assert len(fake.calls) == 1


def test_distillation_resumes_after_max_tasks_and_a_running_task(tmp_path: Path):
    db_path = _three_segments(tmp_path)
    fake = FakeLLM(_fact_response)
    kwargs = {"client": fake, "batch_size": 1, "model_version": "test-model", "prompt_version": "l1-v1"}
    first = run_distillation(db_path, max_tasks=1, **kwargs)
    assert first["status"] == "paused"
    assert first["attempts_completed"] == 1
    assert first["tasks_remaining"] == 2
    assert len(fake.calls) == 1
    first_prompt = fake.calls[0]["user"]

    second = run_distillation(db_path, **kwargs)
    assert second["status"] == "completed"
    assert second["attempts_completed"] == 2
    assert len(fake.calls) == 3
    assert len({call["user"] for call in fake.calls}) == 3
    assert fake.calls[0]["user"] == first_prompt

    third = run_distillation(db_path, **kwargs)
    assert third["attempts_completed"] == 0
    assert len(fake.calls) == 3

    conn = init_db(db_path)
    try:
        observations = conn.execute("SELECT segment_id, evidence_message_ids_json FROM l1_observations").fetchall()
        assert len(observations) == 3
        for row in observations:
            evidence = json.loads(row["evidence_message_ids_json"])
            allowed = json.loads(
                conn.execute(
                    "SELECT message_ids_json FROM conversation_segments WHERE segment_id = ?",
                    (row["segment_id"],),
                ).fetchone()["message_ids_json"]
            )
            assert evidence
            assert set(evidence) <= set(allowed)
        task_id = conn.execute("SELECT task_id FROM distillation_tasks LIMIT 1").fetchone()["task_id"]
        conn.execute(
            "UPDATE distillation_tasks SET status = 'running', attempt_count = 1, "
            "updated_at = '2020-01-01T00:00:00+00:00' WHERE task_id = ?",
            (task_id,),
        )
        conn.execute("DELETE FROM l1_observations WHERE task_id = ?", (task_id,))
        conn.commit()
    finally:
        conn.close()

    resumed = run_distillation(db_path, **kwargs)
    assert resumed["attempts_completed"] == 1
    assert len(fake.calls) == 4
    conn = init_db(db_path)
    try:
        status = conn.execute(
            "SELECT status, attempt_count, observations_count FROM distillation_tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        assert status["status"] == "completed"
        assert status["attempt_count"] == 2
        assert status["observations_count"] == 1
    finally:
        conn.close()


def test_failed_tasks_retry_then_stop_until_requested(tmp_path: Path):
    db_path = _three_segments(tmp_path)

    def explode(system: str, user: str) -> str:
        del system, user
        raise LLMError("temporary")

    fake = FakeLLM(explode)
    common = {"client": fake, "max_attempts": 2, "model_version": "test-model", "prompt_version": "l1-v1"}
    paused = run_distillation(db_path, max_tasks=1, **common)
    assert paused["attempts_failed"] == 1
    assert len(fake.calls) == 1
    conn = init_db(db_path)
    try:
        row = conn.execute(
            "SELECT status, attempt_count FROM distillation_tasks WHERE attempt_count > 0"
        ).fetchone()
        assert row["status"] == "retrying"
        assert row["attempt_count"] == 1
    finally:
        conn.close()

    run_distillation(db_path, **common)
    run_distillation(db_path, **common)
    conn = init_db(db_path)
    try:
        statuses = [row["status"] for row in conn.execute("SELECT status FROM distillation_tasks")]
    finally:
        conn.close()
    assert statuses.count("failed") == 3
    calls_after_exhaustion = len(fake.calls)
    run_distillation(db_path, **common)
    assert len(fake.calls) == calls_after_exhaustion

    fake.responder = _fact_response
    retried = run_distillation(db_path, retry_failed=True, **common)
    assert retried["status"] == "completed"
    assert retried["observations"] == 3
    assert len(fake.calls) > calls_after_exhaustion


def test_prompt_version_is_part_of_the_task_id(tmp_path: Path):
    db_path = _three_segments(tmp_path)
    fake = FakeLLM(_fact_response)
    run_distillation(db_path, client=fake, model_version="demo", prompt_version="l1-v1", max_tasks=1)
    run_distillation(db_path, client=fake, model_version="demo", prompt_version="l1-v2", max_tasks=1)
    assert len(fake.calls) == 2
    conn = init_db(db_path)
    try:
        versions = {
            row["prompt_version"]
            for row in conn.execute("SELECT prompt_version, status FROM distillation_tasks WHERE status = 'completed'")
        }
    finally:
        conn.close()
    assert versions == {"l1-v1", "l1-v2"}


def test_rebuild_drops_observations(tmp_path: Path):
    db_path = _three_segments(tmp_path)
    fake = FakeLLM(_fact_response)
    run_distillation(db_path, client=fake, model_version="test-model", prompt_version="l1-v1")
    run_segmentation(
        db_path,
        params=SegmentParams(gap_seconds=1800, max_messages=50, overlap_messages=3),
        rebuild=True,
    )
    conn = init_db(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM l1_observations").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM distillation_tasks").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM conversation_segments").fetchone()["n"] == 3
    finally:
        conn.close()


def test_distill_requires_segments_and_cli_uses_the_injected_client(tmp_path: Path, monkeypatch):
    empty = tmp_path / "empty.db"
    seed_messages(empty, [{"message_id": "m", "timestamp": 1, "content": "hi"}])
    with pytest.raises(DistillationError):
        run_distillation(empty, client=FakeLLM(_fact_response), model_version="test-model")

    db_path = _three_segments(tmp_path)
    fake = FakeLLM(_fact_response)
    monkeypatch.setattr("distillation.engine.build_llm_client", lambda **kwargs: fake)
    code = distill_main(["--db", str(db_path), "--batch-size", "2", "--model-version", "cli-model"])
    assert code == 0
    assert len(fake.calls) == 3
    again = distill_main(["--db", str(db_path), "--batch-size", "2", "--model-version", "cli-model"])
    assert again == 0
    assert len(fake.calls) == 3


def test_all_invalid_evidence_completes_with_zero_observations(tmp_path: Path):
    db_path = _three_segments(tmp_path)

    def junk(system: str, user: str) -> str:
        del system, user
        return json.dumps(
            {"observations": [{"category": "fact", "summary": "凭空", "evidence_message_ids": ["nope"]}]}
        )

    result = run_distillation(
        db_path,
        client=FakeLLM(junk),
        max_tasks=1,
        model_version="test-model",
        prompt_version="l1-v1",
    )
    assert result["attempts_completed"] == 1
    assert result["observations"] == 0
