"""Long-term memory with a fake model: evidence, conflicts, resume, timeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from common.database import init_db
from distillation.status import pipeline_status
from memory.memory_engine import MemoryEngineError, make_task_id, run_memory
from memory.memory_merger import apply_merge, resolve_actions
from memory.models import CandidateMemory, EntityMention, MergeAction, memory_identity
from memory.parser import ResponseParseError, parse_candidates, parse_merge_actions
from memory.prompting import PROMPT_VERSION, load_extract_prompt, load_merge_prompt
from memory.timeline_builder import build_timeline, render_timeline, search_memory_ids
from memory.types import MEMORY_TYPES
from scripts.build_memories import main as memory_main
from scripts.show_status import main as status_main
from scripts.show_timeline import main as timeline_main
from segmentation.service import run_segmentation
from segmentation.splitter import SegmentParams
from tests.support import seed_messages

RAW = "RAW_MEMORY_SOURCE_TOKEN"


class FakeLLM:
    def __init__(self, responder) -> None:
        self.responder = responder
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self.responder(system, user)


def _field(line: str, key: str) -> str | None:
    marker = f"{key}="
    start = line.find(marker)
    if start < 0:
        return None
    rest = line[start + len(marker) :]
    if key == "statement":
        return rest
    return rest.split()[0]


def _scripted(system: str, user: str) -> str:
    del system
    if user.startswith("task: extract"):
        if "message_id=m1 " in user:
            return json.dumps(
                {
                    "memories": [
                        {
                            "memory_type": "fact",
                            "statement": "住在北京",
                            "importance": 1.4,
                            "confidence": -0.2,
                            "evidence_message_ids": ["m1", "nope"],
                            "observation_ids": ["obs-beijing", "obs-missing"],
                            "entities": [
                                {"name": "owner", "type": "person", "role": "subject", "status": "confirmed"},
                                {"name": "北京", "type": "place", "role": "location", "status": "confirmed"},
                            ],
                        },
                        {
                            "memory_type": "fact",
                            "statement": "Works at Xinghe",
                            "importance": 0.6,
                            "confidence": 0.8,
                            "evidence_message_ids": ["m2"],
                            "entities": [
                                {"name": "owner", "type": "person", "role": "subject", "status": "confirmed"},
                                {"name": "Xinghe", "type": "organization", "role": "object", "status": "confirmed"},
                                {"name": "海淀", "type": "place", "role": "location", "status": "inferred"},
                            ],
                        },
                    ]
                },
                ensure_ascii=False,
            )
        if "message_id=m3 " in user:
            return json.dumps(
                {
                    "memories": [
                        {
                            "memory_type": "fact",
                            "statement": "搬到了上海",
                            "importance": 0.9,
                            "confidence": 0.85,
                            "evidence_message_ids": ["m3"],
                            "entities": [
                                {"name": "owner", "type": "person", "role": "subject", "status": "confirmed"},
                                {"name": "上海", "type": "place", "role": "location", "status": "confirmed"},
                            ],
                        },
                        {
                            "memory_type": "event",
                            "statement": "Left Xinghe",
                            "importance": 0.7,
                            "confidence": 0.9,
                            "evidence_message_ids": ["m4"],
                            "entities": [
                                {"name": "owner", "type": "person", "role": "subject", "status": "confirmed"},
                                {"name": "Xinghe", "type": "organization", "role": "object", "status": "confirmed"},
                            ],
                        },
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps({"memories": []})
    if user.startswith("task: merge"):
        existing: dict[str, str] = {}
        for line in user.splitlines():
            if line.startswith("- memory_id="):
                existing[_field(line, "statement") or ""] = _field(line, "memory_id") or ""
        actions = []
        for line in user.splitlines():
            if not line.startswith("- candidate="):
                continue
            index = int(_field(line, "candidate") or "-1")
            statement = _field(line, "statement") or ""
            if statement == "搬到了上海" and "住在北京" in existing:
                actions.append(
                    {
                        "candidate_index": index,
                        "action": "historical",
                        "target_memory_id": existing["住在北京"],
                        "note": "later city",
                    }
                )
            elif statement == "Left Xinghe" and "Works at Xinghe" in existing:
                actions.append(
                    {
                        "candidate_index": index,
                        "action": "supersede",
                        "target_memory_id": existing["Works at Xinghe"],
                        "note": "job ended",
                    }
                )
            else:
                actions.append({"candidate_index": index, "action": "insert", "target_memory_id": None})
        return json.dumps({"actions": actions})
    raise AssertionError("unexpected memory task")


def _prepare(tmp_path: Path, name: str = "memory.db") -> Path:
    db_path = tmp_path / name
    seed_messages(
        db_path,
        [
            {"message_id": "m1", "conversation_id": "c", "timestamp": 1000, "content": f"{RAW} 我住在北京。"},
            {"message_id": "m2", "conversation_id": "c", "timestamp": 1100, "content": "公司在海淀。"},
            {"message_id": "m3", "conversation_id": "c", "timestamp": 2900, "content": "我搬到上海了。"},
            {"message_id": "m4", "conversation_id": "c", "timestamp": 2910, "content": "星河的工作不做了。"},
            {"message_id": "blank", "conversation_id": "c2", "timestamp": 1, "content": "   "},
        ],
    )
    segmented = run_segmentation(
        db_path,
        params=SegmentParams(gap_seconds=1800, max_messages=50, overlap_messages=0),
    )
    assert segmented["segment_count"] == 3
    conn = init_db(db_path)
    try:
        segment_id = conn.execute(
            "SELECT segment_id FROM conversation_segment_messages WHERE message_id = 'm1'"
        ).fetchone()["segment_id"]
        now = "2026-01-01T00:00:00+00:00"
        conn.execute(
            """
            INSERT INTO distillation_tasks (
                task_id, segment_id, prompt_version, model_version, status,
                attempt_count, observations_count, created_at, updated_at, finished_at
            ) VALUES ('seed-l1', ?, 'l1-v1', 'fixture', 'completed', 1, 1, ?, ?, ?)
            """,
            (segment_id, now, now, now),
        )
        conn.execute(
            """
            INSERT INTO l1_observations (
                observation_id, task_id, segment_id, category, summary,
                evidence_message_ids_json, confidence, time_range_start, time_range_end,
                payload_json, created_at
            ) VALUES ('obs-beijing', 'seed-l1', ?, 'fact', '住在北京', ?, 0.8, 1000, 1000, '{}', ?)
            """,
            (segment_id, json.dumps(["m1"]), now),
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _candidate(
    statement: str,
    message_id: str,
    *,
    memory_type: str = "fact",
    entities: list[EntityMention] | None = None,
    situation: str | None = None,
    importance: float | None = 0.4,
    confidence: float | None = 0.6,
) -> CandidateMemory:
    return CandidateMemory(
        memory_type=memory_type,
        statement=statement,
        situation=situation,
        importance=importance,
        confidence=confidence,
        time_start=10,
        time_end=20,
        time_precision="range",
        evidence_message_ids=[message_id],
        observation_ids=[],
        entities=entities or [],
    )


def _apply(conn, candidates: list[CandidateMemory], actions: list[MergeAction], **kwargs) -> dict:
    stats = apply_merge(
        conn,
        segment_id=None,
        candidates=candidates,
        actions=actions,
        prompt_version=kwargs.get("prompt_version", "memory-v1"),
        model_version=kwargs.get("model_version", "unit"),
    )
    conn.commit()
    return stats.to_dict()


def test_prompts_and_task_id_formula():
    extract = load_extract_prompt()
    merge = load_merge_prompt()
    assert extract.startswith(f"prompt_version: {PROMPT_VERSION}")
    assert merge.startswith(f"prompt_version: {PROMPT_VERSION}")
    for memory_type in MEMORY_TYPES:
        assert memory_type in extract
    for action in ("insert", "duplicate", "supersede", "historical", "conflict", "related"):
        assert action in merge
    assert make_task_id("seg", "extract", "memory-v1", "demo") == hashlib.sha256(
        b"seg|extract|memory-v1|demo"
    ).hexdigest()


def test_parser_keeps_traceable_evidence_and_clamps_scores():
    payload = """```json
{"memories":[
  {"memory_type":"Fact","statement":"住在北京","evidence_message_ids":["m1","nope"],"importance":2,"confidence":-1,
   "time_range":[2000,1000],"entities":[{"name":"北京","type":"nope","role":"nope","status":"sure"}]},
  {"memory_type":"opinion","statement":"   ","evidence_message_ids":["m1"]},
  {"memory_type":"event","statement":"没有证据","evidence_message_ids":["missing"]},
  {"memory_type":"situational","statement":"周末才做饭","situation":"周末","observation_ids":["obs-1"],"evidence_message_ids":[]}
]}
```"""
    kept, dropped = parse_candidates(
        payload,
        allowed_message_ids={"m1"},
        allowed_observation_ids={"obs-1"},
    )
    assert dropped == 2
    assert len(kept) == 2
    assert kept[0].memory_type == "fact"
    assert kept[0].evidence_message_ids == ["m1"]
    assert kept[0].importance == 1.0
    assert kept[0].confidence == 0.0
    assert kept[0].time_start == 1000
    assert kept[0].time_end == 2000
    assert kept[0].entities[0].entity_type == "other"
    assert kept[0].entities[0].role == "mentioned"
    assert kept[0].entities[0].status == "unknown"
    assert kept[1].memory_type == "situational"
    assert kept[1].observation_ids == ["obs-1"]
    assert kept[1].situation == "周末"
    with pytest.raises(ResponseParseError):
        parse_candidates("not json", allowed_message_ids={"m1"}, allowed_observation_ids=set())


def test_merge_parser_and_resolve_fill_missing_candidates():
    actions = parse_merge_actions(
        json.dumps(
            {
                "actions": [
                    {"candidate_index": 1, "action": "supersedes", "target_memory_id": "known"},
                    {"candidate_index": 1, "action": "insert"},
                    {"candidate_index": 0, "action": "conflict", "target_memory_id": "missing"},
                    {"candidate_index": "nope", "action": "insert"},
                ]
            }
        )
    )
    candidates = [
        _candidate("a", "m1"),
        _candidate("b", "m1"),
        _candidate("c", "m1"),
    ]
    resolved = resolve_actions(candidates, actions, allowed_target_ids={"known"})
    assert [item.action for item in resolved] == ["insert", "supersede", "insert"]
    assert resolved[1].target_memory_id == "known"
    assert resolved[0].target_memory_id is None


def test_merger_keeps_conflicts_and_upgrades_entities(tmp_path: Path):
    db_path = tmp_path / "merge.db"
    seed_messages(
        db_path,
        [
            {"message_id": "ma", "timestamp": 10, "content": "alpha"},
            {"message_id": "mb", "timestamp": 20, "content": "beta"},
            {"message_id": "mc", "timestamp": 30, "content": "gamma"},
        ],
    )
    conn = init_db(db_path)
    try:
        typed = [
            _candidate(
                f"type-{memory_type}",
                "ma",
                memory_type=memory_type,
                situation="周末" if memory_type == "situational" else None,
            )
            for memory_type in MEMORY_TYPES
        ]
        inserted = _apply(conn, typed, [MergeAction(index, "insert", None, None) for index in range(len(typed))])
        assert inserted["inserted"] == len(MEMORY_TYPES)
        situational = conn.execute(
            "SELECT situation FROM memories WHERE memory_type = 'situational'"
        ).fetchone()
        assert situational["situation"] == "周末"

        beijing = _candidate("住在北京", "ma")
        _apply(conn, [beijing], [MergeAction(0, "insert", None, None)])
        beijing_id = memory_identity("memory-v1", "unit", "fact", "住在北京")
        conflict = _apply(
            conn,
            [_candidate("一直住在上海", "mb")],
            [MergeAction(0, "conflict", beijing_id, "unsure")],
        )
        assert conflict["conflicts"] == 1
        rows = {
            row["statement"]: row
            for row in conn.execute("SELECT statement, status FROM memories WHERE statement IN ('住在北京', '一直住在上海')")
        }
        assert rows["住在北京"]["status"] == "uncertain"
        assert rows["一直住在上海"]["status"] == "uncertain"
        relation = conn.execute(
            "SELECT relation_type FROM memory_relations WHERE relation_type = 'conflicts_with'"
        ).fetchone()
        assert relation is not None
        challenge = conn.execute(
            """
            SELECT message_id FROM memory_evidence
            WHERE memory_id = ? AND evidence_role = 'conflict'
            """,
            (beijing_id,),
        ).fetchone()
        assert challenge["message_id"] == "mb"
        support = json.loads(
            conn.execute(
                "SELECT evidence_message_ids_json FROM memories WHERE memory_id = ?",
                (beijing_id,),
            ).fetchone()["evidence_message_ids_json"]
        )
        assert support == ["ma"]

        _apply(
            conn,
            [_candidate("住在北京", "mc")],
            [MergeAction(0, "insert", None, None)],
        )
        evidence = json.loads(
            conn.execute(
                "SELECT evidence_message_ids_json FROM memories WHERE memory_id = ?",
                (beijing_id,),
            ).fetchone()["evidence_message_ids_json"]
        )
        assert evidence == ["ma", "mc"]
        assert conn.execute("SELECT COUNT(*) AS n FROM memories WHERE statement = '住在北京'").fetchone()["n"] == 1

        superseded = _apply(
            conn,
            [_candidate("不再住在北京", "mb")],
            [MergeAction(0, "supersede", beijing_id, "replaced")],
        )
        assert superseded["superseded"] == 1
        old = conn.execute("SELECT status, superseded_by FROM memories WHERE memory_id = ?", (beijing_id,)).fetchone()
        new_id = memory_identity("memory-v1", "unit", "fact", "不再住在北京")
        assert old["status"] == "superseded"
        assert old["superseded_by"] == new_id
        assert conn.execute("SELECT status FROM memories WHERE memory_id = ?", (new_id,)).fetchone()["status"] == "active"

        historical_target = memory_identity("memory-v1", "unit", "fact", "type-fact")
        historical = _apply(
            conn,
            [_candidate("后来换了城市", "mc")],
            [MergeAction(0, "historical", historical_target, "later")],
        )
        assert historical["historical"] == 1
        assert (
            conn.execute("SELECT status FROM memories WHERE memory_id = ?", (historical_target,)).fetchone()["status"]
            == "historical"
        )

        mentions = [
            _candidate(
                "first xinghe",
                "ma",
                entities=[EntityMention("Xinghe", "organization", "object", "inferred", aliases=["星河"])],
            ),
            _candidate(
                "second xinghe",
                "mb",
                entities=[EntityMention("XINGHE", "organization", "object", "confirmed", description="公司")],
            ),
            _candidate(
                "third xinghe",
                "mc",
                entities=[EntityMention("xinghe", "organization", "object", "unknown")],
            ),
        ]
        _apply(conn, mentions, [MergeAction(index, "insert", None, None) for index in range(3)])
        entity = conn.execute(
            "SELECT status, canonical_name, aliases_json, description FROM entities WHERE name_key = 'xinghe'"
        ).fetchone()
        assert entity["status"] == "confirmed"
        assert entity["canonical_name"] == "Xinghe"
        assert entity["description"] == "公司"
        assert "星河" in json.loads(entity["aliases_json"])

        dropped = _apply(conn, [_candidate("凭空", "missing")], [MergeAction(0, "insert", None, None)])
        assert dropped["dropped"] == 1

        found = search_memory_ids(conn, "Xinghe")
        found_statements = [
            conn.execute("SELECT statement FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()["statement"]
            for memory_id in found
        ]
        assert any("xinghe" in statement.casefold() for statement in found_statements)
    finally:
        conn.close()


def test_engine_builds_memories_from_segments_and_l1(tmp_path: Path):
    db_path = _prepare(tmp_path)
    fake = FakeLLM(_scripted)
    paused = run_memory(
        db_path,
        client=fake,
        model_version="mem-test",
        prompt_version="memory-v1",
        max_tasks=1,
    )
    assert paused["status"] == "paused"
    assert paused["tasks_remaining"] >= 1
    assert paused["memories"]["total"] == 0
    assert RAW not in json.dumps(paused, ensure_ascii=False)
    assert len(fake.calls) == 1
    assert "observation_id=obs-beijing" in fake.calls[0]["user"]

    finished = run_memory(
        db_path,
        client=fake,
        model_version="mem-test",
        prompt_version="memory-v1",
    )
    assert finished["status"] == "completed", finished
    assert finished["memories"]["total"] == 4
    assert finished["memories"]["by_status"]["historical"] == 1
    assert finished["memories"]["by_status"]["superseded"] == 1
    assert finished["memories"]["by_status"]["active"] == 2
    assert RAW not in json.dumps(finished, ensure_ascii=False)
    calls_after = len(fake.calls)

    again = run_memory(
        db_path,
        client=fake,
        model_version="mem-test",
        prompt_version="memory-v1",
    )
    assert again["status"] == "completed"
    assert again["attempts_completed"] == 0
    assert len(fake.calls) == calls_after

    conn = init_db(db_path)
    try:
        rows = {
            row["statement"]: row
            for row in conn.execute("SELECT * FROM memories")
        }
        assert set(rows) == {"住在北京", "Works at Xinghe", "搬到了上海", "Left Xinghe"}
        beijing = rows["住在北京"]
        assert beijing["status"] == "historical"
        assert beijing["importance"] == 1.0
        assert beijing["confidence"] == 0.0
        assert beijing["time_start"] == 1000
        assert beijing["time_end"] == 1000
        assert json.loads(beijing["evidence_message_ids_json"]) == ["m1"]
        assert json.loads(beijing["source_observation_ids_json"]) == ["obs-beijing"]
        job = rows["Works at Xinghe"]
        assert job["status"] == "superseded"
        assert job["superseded_by"] == rows["Left Xinghe"]["memory_id"]
        assert rows["Left Xinghe"]["status"] == "active"
        assert rows["搬到了上海"]["status"] == "active"
        assert rows["搬到了上海"]["time_start"] == 2900
        haidian = conn.execute("SELECT status FROM entities WHERE canonical_name = '海淀'").fetchone()
        assert haidian["status"] == "inferred"
        owner = conn.execute(
            "SELECT status FROM entities WHERE canonical_name = 'owner' AND entity_type = 'person'"
        ).fetchone()
        assert owner["status"] == "confirmed"
        relations = {
            row["relation_type"]
            for row in conn.execute("SELECT relation_type FROM memory_relations")
        }
        assert relations == {"historical_of", "supersedes"}
        timeline = build_timeline(conn)
        assert [item["statement"] for item in timeline] == [
            "住在北京",
            "Works at Xinghe",
            "搬到了上海",
            "Left Xinghe",
        ]
        active = build_timeline(conn, status="active")
        assert [item["statement"] for item in active] == ["搬到了上海", "Left Xinghe"]
        city = build_timeline(conn, entity="北京")
        assert [item["statement"] for item in city] == ["住在北京"]
        rendered = render_timeline(timeline)
        assert "住在北京" in rendered
        assert RAW not in rendered
        found = search_memory_ids(conn, "Works at Xinghe")
        assert found == [job["memory_id"]]
        assert search_memory_ids(conn, "   ") == []
        blob = " ".join(
            row["result_json"] or ""
            for row in conn.execute("SELECT result_json, error_message FROM memory_tasks")
        )
        assert RAW not in blob
    finally:
        conn.close()

    snapshot = pipeline_status(db_path)
    assert snapshot["memory"]["memories"]["total"] == 4
    assert snapshot["memory"]["entities"]["by_status"]["inferred"] == 1

    run_segmentation(
        db_path,
        params=SegmentParams(gap_seconds=1800, max_messages=50, overlap_messages=0),
        rebuild=True,
    )
    conn = init_db(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM entities").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM memory_tasks").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM memory_checkpoint").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM memories_fts").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM l1_observations").fetchone()["n"] == 0
    finally:
        conn.close()


def test_resume_after_crash_and_retry_failed(tmp_path: Path):
    db_path = _prepare(tmp_path)

    class Crash:
        def complete(self, *, system: str, user: str) -> str:
            del system, user
            raise SystemExit("stop")

    with pytest.raises(SystemExit):
        run_memory(db_path, client=Crash(), model_version="crash-model", prompt_version="memory-v1", max_tasks=1)
    conn = init_db(db_path)
    try:
        running = conn.execute(
            "SELECT COUNT(*) AS n FROM memory_tasks WHERE status = 'running'"
        ).fetchone()["n"]
    finally:
        conn.close()
    assert running == 1
    recovered = run_memory(
        db_path,
        client=FakeLLM(_scripted),
        model_version="crash-model",
        prompt_version="memory-v1",
    )
    assert recovered["status"] == "completed"
    assert recovered["memories"]["total"] == 4

    retry_db = _prepare(tmp_path, "retry.db")

    def bad(system: str, user: str) -> str:
        del system, user
        return "not-json"

    failed = run_memory(
        retry_db,
        client=FakeLLM(bad),
        model_version="retry-model",
        prompt_version="memory-v1",
        max_attempts=1,
    )
    assert failed["status"] == "failed"
    assert failed["attempts_failed"] == 2
    conn = init_db(retry_db)
    try:
        errors = [row["error_message"] for row in conn.execute("SELECT error_message FROM memory_tasks")]
    finally:
        conn.close()
    assert errors
    assert all(RAW not in (message or "") for message in errors)
    stalled = run_memory(
        retry_db,
        client=FakeLLM(_scripted),
        model_version="retry-model",
        prompt_version="memory-v1",
    )
    assert stalled["attempts_completed"] == 0
    retried = run_memory(
        retry_db,
        client=FakeLLM(_scripted),
        model_version="retry-model",
        prompt_version="memory-v1",
        retry_failed=True,
    )
    assert retried["status"] == "completed"
    assert retried["memories"]["total"] == 4


def test_prompt_version_is_part_of_the_task_id(tmp_path: Path):
    db_path = _prepare(tmp_path)
    fake = FakeLLM(_scripted)
    run_memory(db_path, client=fake, model_version="demo", prompt_version="memory-v1", max_tasks=1)
    run_memory(db_path, client=fake, model_version="demo", prompt_version="memory-v2", max_tasks=1)
    assert len(fake.calls) == 2
    conn = init_db(db_path)
    try:
        versions = {row["prompt_version"] for row in conn.execute("SELECT prompt_version FROM memory_tasks")}
    finally:
        conn.close()
    assert "memory-v1" in versions
    assert "memory-v2" in versions


def test_cli_prints_counts_without_raw_chat(tmp_path: Path, monkeypatch, capsys):
    db_path = _prepare(tmp_path)
    fake = FakeLLM(_scripted)
    monkeypatch.setattr("memory.memory_engine.build_llm_client", lambda **kwargs: fake)
    code = memory_main(["--db", str(db_path), "--model-version", "cli-model"])
    assert code == 0
    output = capsys.readouterr().out
    assert "status: completed" in output
    assert RAW not in output
    calls = len(fake.calls)
    again = memory_main(["--db", str(db_path), "--model-version", "cli-model"])
    assert again == 0
    assert len(fake.calls) == calls
    capsys.readouterr()
    assert timeline_main(["--db", str(db_path), "--status", "historical"]) == 0
    timeline = capsys.readouterr().out
    assert "住在北京" in timeline
    assert RAW not in timeline
    assert status_main(["--db", str(db_path)]) == 0
    assert "memories:" in capsys.readouterr().out
    assert memory_main(["--db", str(tmp_path / "missing.db"), "--model-version", "cli-model"]) == 1
    assert timeline_main(["--db", str(tmp_path / "missing.db")]) == 1
    with pytest.raises(FileNotFoundError):
        run_memory(tmp_path / "missing.db", client=fake, model_version="cli-model")

    empty = tmp_path / "empty.db"
    seed_messages(empty, [{"message_id": "m", "timestamp": 1, "content": "hi"}])
    with pytest.raises(MemoryEngineError):
        run_memory(empty, client=fake, model_version="cli-model")
