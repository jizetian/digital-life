"""L2 merge and L3 profile with a fake model."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from common.database import init_db
from distillation.engine import make_task_id as l1_task_id
from distillation.llm import LLMError
from personality.contradiction_detector import finalize_trait, parse_judgments
from personality.dimensions import DIMENSIONS, PROFILE_FILES, dimension_for_category
from personality.engine import make_task_id, run_personality
from personality.jsonutil import ResponseParseError
from personality.models import ObsView, classify_status, independent_group_count, trait_confidence
from personality.profile_builder import build_profile
from personality.prompting import PROMPT_VERSION, load_contradict_prompt, load_merge_prompt, load_reduce_prompt
from personality.trait_merger import parse_merge_response, parse_reduce_response
from scripts.build_personality import main as personality_main
from segmentation.service import run_segmentation
from segmentation.splitter import SegmentParams
from tests.support import seed_messages

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "l1_personality_observations.json"
_OBS_LINE = re.compile(r"observation_id=(\S+).*summary=(.*)$", re.MULTILINE)
_TRAIT_KEY = re.compile(r"trait_key=(\S+)")


class FakeLLM:
    def __init__(self, responder) -> None:
        self.responder = responder
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self.responder(system, user)


def _obs(
    observation_id: str,
    segment_id: str,
    messages: list[str],
    *,
    summary: str | None = None,
    start: int | None = None,
    end: int | None = None,
    confidence: float | None = 0.8,
    category: str = "communication",
) -> ObsView:
    return ObsView(
        observation_id=observation_id,
        segment_id=segment_id,
        category=category,
        dimension=dimension_for_category(category),
        summary=summary or observation_id,
        evidence_message_ids=messages,
        confidence=confidence,
        time_start=start,
        time_end=end if end is not None else start,
    )


def _fixture_responder(system: str, user: str) -> str:
    if "relation must be exactly one of" in system:
        keys = _TRAIT_KEY.findall(user)
        judgments = []
        for key in keys:
            for observation_id, summary in _OBS_LINE.findall(user):
                if "避免直接" in summary:
                    relation = "contradicts"
                elif "后来写得很长" in summary:
                    relation = "evolution"
                else:
                    relation = "unrelated"
                judgments.append(
                    {
                        "trait_key": key,
                        "observation_id": observation_id,
                        "relation": relation,
                        "situational_scope": None,
                        "reason": relation,
                    }
                )
        return json.dumps({"judgments": judgments}, ensure_ascii=False)
    if "separate batches" in system:
        keys = _TRAIT_KEY.findall(user)
        return json.dumps(
            {
                "traits": [
                    {
                        "statement": "先给短结论",
                        "situational_scope": "日常答复",
                        "trait_keys": keys,
                        "stages": [],
                    }
                ]
            },
            ensure_ascii=False,
        )
    dimension = ""
    for line in user.splitlines():
        if line.startswith("dimension: "):
            dimension = line.split(": ", 1)[1].strip()
    parsed = _OBS_LINE.findall(user)
    if dimension == "values":
        return json.dumps(
            {
                "traits": [
                    {
                        "statement": "把守时看得很重",
                        "situational_scope": "约定时间",
                        "observation_ids": [observation_id for observation_id, _ in parsed],
                        "stages": [],
                    }
                ]
            },
            ensure_ascii=False,
        )
    groups = {
        "先给短结论": [observation_id for observation_id, summary in parsed if "回复很短" in summary],
        "后来会把原因写长": [observation_id for observation_id, summary in parsed if "后来写得很长" in summary],
        "只提过一次周末": [observation_id for observation_id, summary in parsed if "只在一段" in summary],
    }
    traits = []
    for statement, observation_ids in groups.items():
        if not observation_ids:
            continue
        scope = "日常答复" if statement == "先给短结论" else "general"
        traits.append(
            {
                "statement": statement,
                "situational_scope": scope,
                "observation_ids": observation_ids,
                "stages": [],
            }
        )
    return json.dumps({"traits": traits}, ensure_ascii=False)


def _load_fixture_db(tmp_path: Path) -> Path:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    db_path = tmp_path / "personality.db"
    seed_messages(
        db_path,
        [
            {
                "message_id": item["message_id"],
                "timestamp": item["timestamp"],
                "content": item["content"],
                "conversation_id": "chat",
            }
            for item in payload["observations"]
        ],
    )
    segmented = run_segmentation(
        db_path,
        params=SegmentParams(gap_seconds=1, max_messages=1, overlap_messages=0),
    )
    assert segmented["segment_count"] == len(payload["observations"])
    conn = init_db(db_path)
    try:
        segments = {
            row["message_id"]: row["segment_id"]
            for row in conn.execute(
                "SELECT segment_id, message_id FROM conversation_segment_messages"
            )
        }
        now = "2026-01-01T00:00:00+00:00"
        for item in payload["observations"]:
            segment_id = segments[item["message_id"]]
            task_id = l1_task_id(segment_id, "l1-v1", "fixture")
            conn.execute(
                """
                INSERT INTO distillation_tasks (
                    task_id, segment_id, prompt_version, model_version, status,
                    attempt_count, observations_count, created_at, updated_at, finished_at
                ) VALUES (?, ?, 'l1-v1', 'fixture', 'completed', 1, 1, ?, ?, ?)
                ON CONFLICT(task_id) DO NOTHING
                """,
                (task_id, segment_id, now, now, now),
            )
            conn.execute(
                """
                INSERT INTO l1_observations (
                    observation_id, task_id, segment_id, category, summary,
                    evidence_message_ids_json, confidence, time_range_start, time_range_end,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["observation_id"],
                    task_id,
                    segment_id,
                    item["category"],
                    item["summary"],
                    json.dumps([item["message_id"]], ensure_ascii=False),
                    item["confidence"],
                    item["timestamp"],
                    item["timestamp"],
                    "{}",
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_prompt_versions_dimensions_and_task_id():
    for loader in (load_merge_prompt, load_reduce_prompt, load_contradict_prompt):
        assert loader().startswith(f"prompt_version: {PROMPT_VERSION}")
    assert len(DIMENSIONS) == 6
    assert dimension_for_category("value") == "values"
    assert dimension_for_category("opinion") == "thinking"
    assert dimension_for_category("fact") is None
    assert len(PROFILE_FILES) == 8
    raw = b"merge|communication|batch|l2-v1|demo"
    assert make_task_id("merge", "communication", "batch", "l2-v1", "demo") == hashlib.sha256(raw).hexdigest()


def test_independence_and_confidence_rules():
    assert independent_group_count([
        _obs("a", "s1", ["m1"]),
        _obs("b", "s1", ["m2"]),
    ]) == 1
    assert independent_group_count([
        _obs("a", "s1", ["m1"]),
        _obs("b", "s2", ["m1"]),
    ]) == 1
    assert independent_group_count([
        _obs("a", "s1", ["m1"]),
        _obs("b", "s2", ["m2"]),
    ]) == 2
    alone = trait_confidence([0.9, 0.8], 2, 0)
    countered = trait_confidence([0.9, 0.8], 2, 1)
    assert countered < alone
    expected = round(0.85 * (0.55 + 0.45 * 0.5) * (1 - 0.75 * (1 / 3)), 4)
    assert countered == expected
    assert classify_status(1, 0, min_independent=2) == "provisional"
    assert classify_status(2, 1, min_independent=2) == "stable"
    assert classify_status(2, 2, min_independent=2) == "contested"


def test_parsers_drop_unknown_ids_and_keep_uncited_batches():
    traits = parse_merge_response(
        '{"traits":[{"statement":"ok","observation_ids":["a","missing"]},{"statement":"  ","observation_ids":["a"]}]}',
        allowed_ids={"a"},
    )
    assert traits == [
        {"statement": "ok", "situational_scope": "general", "observation_ids": ["a"], "stages": []}
    ]
    with pytest.raises(ResponseParseError):
        parse_merge_response("not json", allowed_ids={"a"})
    local = {
        "b0t0": {"statement": "one", "situational_scope": "general", "observation_ids": ["a"], "stages": []},
        "b1t0": {"statement": "two", "situational_scope": "general", "observation_ids": ["b"], "stages": []},
    }
    reduced = parse_reduce_response(
        json.dumps({"traits": [{"statement": "one", "trait_keys": ["b0t0", "missing"], "situational_scope": "general"}]}),
        local_traits=local,
    )
    kept = sorted(oid for trait in reduced for oid in trait["observation_ids"])
    assert kept == ["a", "b"]
    judgments = parse_judgments(
        json.dumps(
            {
                "judgments": [
                    {"trait_key": "t0", "observation_id": "a", "relation": "contradicts", "reason": "no"},
                    {"trait_key": "nope", "observation_id": "a", "relation": "supports"},
                    {"trait_key": "t0", "observation_id": "missing", "relation": "supports"},
                ]
            }
        ),
        trait_keys={"t0"},
        allowed_observation_ids={"a"},
    )
    assert len(judgments) == 1
    assert judgments[0]["relation"] == "contradicts"


def test_time_splits_contradiction_from_evolution_and_scope_is_not_a_counter():
    observations = {
        "early": _obs("early", "s1", ["m1"], start=100, confidence=0.9),
        "mid": _obs("mid", "s2", ["m2"], start=200, confidence=0.8),
        "inside": _obs("inside", "s3", ["m3"], summary="反对", start=150),
        "later": _obs("later", "s4", ["m4"], summary="后来变了", start=5000),
        "scoped": _obs("scoped", "s5", ["m5"], summary="只在赶工时", start=160),
    }
    record = finalize_trait(
        trait_key="t0",
        dimension="communication",
        spec={
            "statement": "先给短结论",
            "situational_scope": "日常答复",
            "observation_ids": ["early", "mid"],
            "stages": [],
        },
        judgments=[
            {"trait_key": "t0", "observation_id": "inside", "relation": "evolution", "reason": "同期相反", "situational_scope": None},
            {"trait_key": "t0", "observation_id": "later", "relation": "contradicts", "reason": "时期不同", "situational_scope": None},
            {"trait_key": "t0", "observation_id": "scoped", "relation": "situational", "reason": "范围更窄", "situational_scope": "赶工时"},
        ],
        observations=observations,
        min_independent=2,
        prompt_version="l2-v1",
        model_version="demo",
    )
    assert record is not None
    assert record.status == "stable"
    assert record.evidence_ids == ["early", "mid"]
    assert record.counter_evidence_ids == ["inside"]
    assert record.counter_evidence_count == 1
    assert record.evidence_count == 2
    assert record.independent_evidence_count == 2
    assert [stage["observation_ids"] for stage in record.stages] == [["early", "mid"], ["later"]]
    assert record.variants[0]["situational_scope"] == "赶工时"
    assert record.valid_from == 100
    assert record.valid_to == 5000

    overlapping = finalize_trait(
        trait_key="t0",
        dimension="thinking",
        spec={
            "statement": "会比较",
            "situational_scope": "general",
            "observation_ids": ["early", "mid"],
            "stages": [
                {"statement": "早期", "observation_ids": ["early", "mid"]},
                {"statement": "同时", "observation_ids": ["early"]},
            ],
        },
        judgments=[],
        observations=observations,
        min_independent=2,
        prompt_version="l2-v1",
        model_version="demo",
    )
    assert overlapping is not None
    assert overlapping.stages == []


def test_profile_from_fixture_observations_is_stable_and_resumable(tmp_path: Path):
    db_path = _load_fixture_db(tmp_path)
    output = tmp_path / "profile"
    fake = FakeLLM(_fixture_responder)
    kwargs = {
        "client": fake,
        "output_dir": output,
        "model_version": "persona-test",
        "prompt_version": "l2-v1",
        "batch_size": 20,
    }
    result = run_personality(db_path, **kwargs)
    assert result["status"] == "completed"
    assert result["attempts_completed"] >= 1
    assert result["traits"]["stable"] == 2
    assert result["traits"]["provisional"] >= 1
    assert result["files"] == list(PROFILE_FILES)
    for name in PROFILE_FILES:
        assert (output / name).is_file()
    for call in fake.calls:
        blob = call["system"] + call["user"]
        assert "RAW_MESSAGE_" not in blob
        assert "observation_id=" in call["user"] or "trait_key=" in call["user"]

    evidence = json.loads((output / "personality_evidence.json").read_text(encoding="utf-8"))
    assert set(evidence["dimensions"]) == set(DIMENSIONS)
    communication = evidence["dimensions"]["communication"]["traits"]
    stable = [trait for trait in communication if trait["status"] == "stable"]
    assert len(stable) == 1
    trait = stable[0]
    assert trait["statement"] == "先给短结论"
    assert trait["evidence_count"] == 2
    assert trait["independent_evidence_count"] == 2
    assert trait["counter_evidence_count"] == 1
    assert trait["evidence_ids"] == ["obs-short-1", "obs-short-2"]
    assert trait["counter_evidence_ids"] == ["obs-avoid"]
    assert trait["valid_from"] == 1700000000
    assert trait["valid_to"] == 1704000000
    assert trait["situational_scope"] == "日常答复"
    assert {item["message_ids"][0] for item in trait["evidence"]} == {"m-short-1", "m-short-2"}
    assert evidence["context_observations"]["fact"] == 1
    assert evidence["evolution"]
    assert evidence["evolution"][0]["dimension"] == "communication"

    style = (output / "communication_style.md").read_text(encoding="utf-8")
    assert "confidence:" in style
    assert "evidence_count: 2" in style
    assert "counter_evidence_count: 1" in style
    assert "valid_from:" in style
    assert "valid_to:" in style
    assert "evidence_ids: obs-short-1, obs-short-2" in style
    assert "## Temporal evolution" in style
    assert "后来写得很长，会解释原因" in style
    stable_section, contested_section = style.split("## Contested traits", 1)
    assert "只提过一次周末" not in stable_section
    assert "先给短结论" in stable_section
    assert "避免直接表态" not in contested_section
    values = (output / "values.md").read_text(encoding="utf-8")
    assert "把守时看得很重" in values
    assert "evidence_count: 2" in values
    core = (output / "personality_core.md").read_text(encoding="utf-8")
    assert "fact: 1" in core
    assert "No stage change was supported." in (output / "emotional_patterns.md").read_text(encoding="utf-8")

    calls = len(fake.calls)
    again = run_personality(db_path, **kwargs)
    assert again["status"] == "completed"
    assert again["attempts_completed"] == 0
    assert len(fake.calls) == calls

    conn = init_db(db_path)
    try:
        task_id = conn.execute(
            """
            SELECT task_id FROM personality_tasks
            WHERE task_kind = 'merge' AND status = 'completed' AND active = 1
            ORDER BY dimension
            LIMIT 1
            """
        ).fetchone()["task_id"]
        conn.execute(
            """
            UPDATE personality_tasks
            SET status = 'running', result_json = NULL, finished_at = NULL
            WHERE task_id = ?
            """,
            (task_id,),
        )
        conn.commit()
    finally:
        conn.close()
    resumed = run_personality(db_path, **kwargs)
    assert resumed["status"] == "completed"
    assert resumed["attempts_completed"] >= 1
    assert len(fake.calls) > calls


def test_cross_batch_reduce_checkpoint_and_prompt_version(tmp_path: Path):
    db_path = tmp_path / "reduce.db"
    _seed_same_pattern(db_path, ["m1", "m2", "m3"])
    output = tmp_path / "out"
    fake = FakeLLM(_fixture_responder)
    common = {
        "client": fake,
        "output_dir": output,
        "model_version": "persona-test",
        "prompt_version": "l2-v1",
        "batch_size": 1,
    }
    paused = run_personality(db_path, max_tasks=1, **common)
    assert paused["status"] == "paused"
    assert paused["attempts_completed"] == 1
    assert paused["tasks_remaining"] >= 1
    assert paused["files"] == []
    assert not (output / "personality_core.md").exists()
    finished = run_personality(db_path, **common)
    assert finished["status"] == "completed"
    assert finished["traits"]["stable"] == 1
    assert any("separate batches" in call["system"] for call in fake.calls)
    evidence = json.loads((output / "personality_evidence.json").read_text(encoding="utf-8"))
    trait = evidence["dimensions"]["communication"]["traits"][0]
    assert trait["status"] == "stable"
    assert trait["evidence_count"] == 3
    assert trait["independent_evidence_count"] == 3

    other = tmp_path / "other"
    run_personality(
        db_path,
        client=fake,
        output_dir=other,
        model_version="persona-test",
        prompt_version="l2-v2",
        batch_size=1,
        max_tasks=1,
    )
    conn = init_db(db_path)
    try:
        versions = {
            row["prompt_version"]
            for row in conn.execute("SELECT DISTINCT prompt_version FROM personality_tasks")
        }
    finally:
        conn.close()
    assert versions == {"l2-v1", "l2-v2"}


def test_failed_tasks_retry_then_stop_until_requested(tmp_path: Path):
    db_path = tmp_path / "fail.db"
    _seed_same_pattern(db_path, ["m1", "m2"])
    output = tmp_path / "out"

    def explode(system: str, user: str) -> str:
        del system, user
        raise LLMError("temporary")

    fake = FakeLLM(explode)
    common = {
        "client": fake,
        "output_dir": output,
        "model_version": "persona-test",
        "prompt_version": "l2-v1",
        "max_attempts": 2,
        "batch_size": 1,
    }
    first = run_personality(db_path, max_tasks=1, **common)
    assert first["attempts_failed"] == 1
    assert first["status"] == "paused"
    conn = init_db(db_path)
    try:
        row = conn.execute(
            "SELECT status, attempt_count, error_message FROM personality_tasks WHERE attempt_count > 0"
        ).fetchone()
        assert row["status"] == "retrying"
        assert row["attempt_count"] == 1
        assert "回复很短" not in (row["error_message"] or "")
    finally:
        conn.close()
    run_personality(db_path, **common)
    run_personality(db_path, **common)
    conn = init_db(db_path)
    try:
        statuses = [row["status"] for row in conn.execute("SELECT status FROM personality_tasks WHERE active = 1")]
    finally:
        conn.close()
    assert "failed" in statuses
    calls = len(fake.calls)
    stalled = run_personality(db_path, **common)
    assert stalled["status"] == "failed"
    assert len(fake.calls) == calls
    assert not (output / "personality_core.md").exists()

    fake.responder = _fixture_responder
    rebuilt = run_personality(db_path, retry_failed=True, **common)
    assert rebuilt["status"] == "completed"
    assert rebuilt["traits"]["stable"] == 1
    assert (output / "decision_patterns.md").is_file()


def test_examples_document_the_profile_contract():
    root = Path(__file__).resolve().parents[1] / "personality" / "examples"
    for name in PROFILE_FILES:
        text = (root / name).read_text(encoding="utf-8")
        assert "prompt_version" in text
        if name.endswith(".md") and name != "personality_core.md":
            assert "## Stable traits" in text
            assert "## Temporal evolution" in text
    evidence = json.loads((root / "personality_evidence.json").read_text(encoding="utf-8"))
    assert set(evidence["dimensions"]) == set(DIMENSIONS)
    sample = evidence["dimensions"]["communication"]["traits"][0]
    for field in (
        "confidence",
        "evidence_count",
        "counter_evidence_count",
        "valid_from",
        "valid_to",
        "evidence_ids",
    ):
        assert field in sample


def test_cli_and_rebuild_drop_personality_rows(tmp_path: Path, monkeypatch):
    db_path = _load_fixture_db(tmp_path)
    output = tmp_path / "cli-profile"
    fake = FakeLLM(_fixture_responder)
    monkeypatch.setattr("personality.engine.build_llm_client", lambda **kwargs: fake)
    code = personality_main(
        ["--db", str(db_path), "--output", str(output), "--model-version", "cli-model", "--batch-size", "20"]
    )
    assert code == 0
    assert (output / "personality_evidence.json").is_file()
    again = personality_main(
        ["--db", str(db_path), "--output", str(output), "--model-version", "cli-model", "--batch-size", "20"]
    )
    assert again == 0
    calls = len(fake.calls)
    third = personality_main(
        ["--db", str(db_path), "--output", str(output), "--model-version", "cli-model", "--batch-size", "20"]
    )
    assert third == 0
    assert len(fake.calls) == calls

    run_segmentation(
        db_path,
        params=SegmentParams(gap_seconds=1, max_messages=1, overlap_messages=0),
        rebuild=True,
    )
    conn = init_db(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM l1_observations").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM l2_traits").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM personality_tasks").fetchone()["n"] == 0
    finally:
        conn.close()

    missing = personality_main(["--db", str(tmp_path / "missing.db"), "--model-version", "cli-model"])
    assert missing == 1
    with pytest.raises(FileNotFoundError):
        run_personality(tmp_path / "missing.db", client=fake, model_version="cli-model")


def test_profile_builder_writes_every_dimension_file(tmp_path: Path):
    obs = _obs("a", "s1", ["m1"], summary="一句", start=10)
    trait_id = "t"
    from personality.models import TraitRecord

    record = TraitRecord(
        trait_id=trait_id,
        dimension="decision",
        statement="先列条件再选",
        situational_scope="做选择时",
        evidence_ids=["a"],
        counter_evidence_ids=[],
        valid_from=10,
        valid_to=10,
        confidence=0.4,
        evidence_count=1,
        independent_evidence_count=1,
        counter_evidence_count=0,
        status="provisional",
        stages=[],
        variants=[],
        counter_details=[],
    )
    written = build_profile(
        output_dir=tmp_path / "built",
        traits=[record],
        observations={"a": obs},
        context_counts={"interest": 0, "event": 1, "fact": 0},
        prompt_version="l2-v1",
        model_version="demo",
        observation_fingerprint="fp",
        min_independent=2,
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert written["traits_provisional"] == 1
    decision = (tmp_path / "built" / "decision_patterns.md").read_text(encoding="utf-8")
    assert "先列条件再选" not in decision.split("## Contested traits", 1)[0]
    payload = json.loads((tmp_path / "built" / "personality_evidence.json").read_text(encoding="utf-8"))
    stored = payload["dimensions"]["decision"]["traits"][0]
    assert stored["status"] == "provisional"
    assert stored["evidence_ids"] == ["a"]
    assert stored["evidence"][0]["message_ids"] == ["m1"]


def _seed_same_pattern(db_path: Path, message_ids: list[str]) -> None:
    seed_messages(
        db_path,
        [
            {
                "message_id": message_id,
                "timestamp": 1_700_000_000 + index * 10_000,
                "content": f"RAW_MESSAGE_{message_id}",
                "conversation_id": "chat",
            }
            for index, message_id in enumerate(message_ids)
        ],
    )
    run_segmentation(db_path, params=SegmentParams(gap_seconds=1, max_messages=1, overlap_messages=0))
    conn = init_db(db_path)
    try:
        rows = conn.execute(
            "SELECT segment_id, message_id FROM conversation_segment_messages ORDER BY message_id"
        ).fetchall()
        now = "2026-01-01T00:00:00+00:00"
        for index, row in enumerate(rows):
            task_id = l1_task_id(row["segment_id"], "l1-v1", "fixture")
            conn.execute(
                """
                INSERT INTO distillation_tasks (
                    task_id, segment_id, prompt_version, model_version, status,
                    attempt_count, observations_count, created_at, updated_at, finished_at
                ) VALUES (?, ?, 'l1-v1', 'fixture', 'completed', 1, 1, ?, ?, ?)
                ON CONFLICT(task_id) DO NOTHING
                """,
                (task_id, row["segment_id"], now, now, now),
            )
            conn.execute(
                """
                INSERT INTO l1_observations (
                    observation_id, task_id, segment_id, category, summary,
                    evidence_message_ids_json, confidence, time_range_start, time_range_end,
                    payload_json, created_at
                ) VALUES (?, ?, ?, 'communication', ?, ?, 0.9, ?, ?, '{}', ?)
                """,
                (
                    f"obs-{index}",
                    task_id,
                    row["segment_id"],
                    "回复很短，先给结论",
                    json.dumps([row["message_id"]]),
                    1_700_000_000 + index * 10_000,
                    1_700_000_000 + index * 10_000,
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()
