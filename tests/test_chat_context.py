"""Context assembly stays inside the character budget and prefers cited memories."""

from __future__ import annotations

from pathlib import Path

from chat.context import build_context
from chat.personality import PersonalityView, load_personality
from common.database import init_db


def _hit(collection: str, source_id: str, text: str, score: float, metadata: dict) -> dict:
    return {
        "collection": collection,
        "source_id": source_id,
        "text": text,
        "score": score,
        "metadata": metadata,
    }


def _personality() -> PersonalityView:
    return PersonalityView(source="l2", text="说话很短，先讲结论。", trait_ids=["t-stable"])


def _build(**overrides):
    hits = overrides.pop("hits", [])
    history = overrides.pop("history", [])
    user_message = overrides.pop("user_message", "你后来住在哪")
    personality = overrides.pop("personality", _personality())
    params = dict(
        target_name="不吃香菜",
        personality=personality,
        hits=hits,
        history=history,
        user_message=user_message,
        max_chars=900,
        personality_chars=200,
        history_chars=180,
        history_turns=2,
        user_chars=200,
    )
    params.update(overrides)
    return build_context(**params)


def test_evidence_backed_memory_outranks_a_higher_score_without_evidence():
    memory = _hit(
        "long_term_memories",
        "mem-keep",
        "搬到了上海",
        0.1,
        {"status": "active", "memory_type": "fact", "evidence_message_ids": ["m1"]},
    )
    bare = _hit(
        "long_term_memories",
        "mem-bare",
        "没有消息证据的说法",
        0.9,
        {"status": "active", "memory_type": "fact", "evidence_message_ids": []},
    )
    segment = _hit(
        "conversation_segments",
        "seg-big",
        ("原" * 4000) + "UNIQUESEGMENTTOKEN",
        0.8,
        {"message_ids": ["m9"]},
    )
    built = _build(hits=[segment, bare, memory])
    assert built.char_count <= 900
    assert built.char_count == len(built.budgeted_text)
    assert built.system.index("mem-keep") < built.system.index("mem-bare")
    assert "UNIQUESEGMENTTOKEN" not in built.system
    assert built.allowed_memory_ids[0] == "mem-keep"
    assert "m1" in built.allowed_message_ids
    assert "mem-keep" in {item["memory_id"] for item in built.retrieved_memories}
    assert built.retrieval_weak is False
    assert "不吃香菜" in built.system
    assert "Never invent" in built.system
    assert "说话很短" in built.system


def test_weak_retrieval_is_labeled_when_nothing_has_message_evidence():
    trait = _hit(
        "personality_observations",
        "trait-1",
        "说话很短",
        0.4,
        {"dimension": "communication", "status": "stable"},
    )
    built = _build(hits=[trait], max_chars=1200, history_chars=200)
    assert built.retrieval_weak is True
    assert "证据不足" in built.system
    assert built.uncertainty_note is not None
    assert built.allowed_message_ids == []
    assert built.allowed_memory_ids == []


def test_history_and_the_current_question_are_capped():
    history = [
        {"role": "user", "content": "OLDEST_TURN_TOKEN"},
        {"role": "assistant", "content": "中间一句"},
        {"role": "user", "content": "NEWEST_TURN_TOKEN"},
    ]
    built = _build(
        history=history,
        history_turns=1,
        user_message="QUESTION_TOKEN" + ("问" * 80),
        user_chars=16,
        max_chars=800,
    )
    assert "NEWEST_TURN_TOKEN" in built.user
    assert "OLDEST_TURN_TOKEN" not in built.user
    assert "QUESTION_TOKEN" in built.user
    assert "问" * 80 not in built.user
    assert built.truncated is True
    assert built.char_count <= 800


def test_stable_traits_are_loaded_and_provisional_traits_are_not(tmp_path: Path):
    db_path = tmp_path / "personality.db"
    conn = init_db(db_path)
    _trait(conn, "t-stable", "说话很短，先讲结论。", "stable")
    _trait(conn, "t-provisional", "临时特质不应该出现", "provisional")
    view = load_personality(conn, output_dir=tmp_path / "missing-profile", max_chars=500)
    conn.close()
    assert view.source == "l2"
    assert "说话很短" in view.text
    assert "临时特质不应该出现" not in view.text
    assert view.trait_ids == ["t-stable"]


def test_l3_profile_is_used_when_there_is_no_stable_trait(tmp_path: Path):
    profile = tmp_path / "personality"
    profile.mkdir()
    (profile / "personality_core.md").write_text(
        "\n".join(
            [
                "# Personality core",
                "",
                "## Stable traits",
                "",
                "- 喜欢把利弊摊开再说决定。",
                "",
                "## Contested traits",
                "",
                "- 有争议的说法不要进人格块。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    conn = init_db(tmp_path / "empty.db")
    view = load_personality(conn, output_dir=profile, max_chars=500)
    conn.close()
    assert view.source == "l3"
    assert "喜欢把利弊摊开" in view.text
    assert "有争议的说法不要进人格块" not in view.text


def _trait(conn, trait_id: str, statement: str, status: str) -> None:
    conn.execute(
        """
        INSERT INTO l2_traits (
            trait_id, prompt_version, model_version, dimension, statement,
            situational_scope, confidence, evidence_count, independent_evidence_count,
            counter_evidence_count, evidence_ids_json, counter_evidence_ids_json,
            valid_from, valid_to, status, stages_json, variants_json, payload_json,
            created_at, updated_at
        ) VALUES (?, 'l2-v1', 'test-model', 'communication', ?, '', 0.8, 2, 2, 0,
                  '[]', '[]', NULL, NULL, ?, '[]', '[]', '{}',
                  '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
        """,
        (trait_id, statement, status),
    )
    conn.commit()
