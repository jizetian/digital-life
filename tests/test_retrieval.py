"""Hybrid retrieval with a scripted embedder and an in-memory vector store.

These tests do not load BGE-M3 and do not open a Qdrant server. One case uses
the Qdrant client's in-process memory mode to exercise the real adapter.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from common.clock import utc_now
from common.database import init_db
from retrieval.embedding import HashEmbeddingClient, OpenAIEmbeddingClient
from retrieval.hybrid_search import SearchHit
from retrieval.reranker import HttpReranker
from retrieval.service import run_index, run_search
from retrieval.vector_store import InMemoryVectorStore, QdrantVectorStore, time_overlaps
from scripts.build_embeddings import main as build_main
from scripts.search import main as search_main
from scripts.show_status import main as status_main
from segmentation.service import run_segmentation


LIBRARY = "the library closes at nine"
BOTH = "BOTH_HIT purple bicycle parade"
VECTOR = "VECTOR_ONLY quiet study"


class ScriptedEmbedder:
    """Maps marker phrases onto orthogonal axes. Everything else is a zero vector."""

    model_name = "scripted"
    provider = "scripted"
    dimension = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            compact = " ".join(text.split())
            if compact in {LIBRARY, "query-vector"} or "VECTOR_ONLY" in text:
                vectors.append([0.0, 1.0, 0.0])
            elif "BOTH_HIT" in text:
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 0.0])
        return vectors


class OtherModel(ScriptedEmbedder):
    model_name = "scripted-v2"


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return dot / (left_norm * right_norm)


def _seed(db_path: Path) -> None:
    conn = init_db(db_path)
    now = utc_now()
    conn.execute(
        """
        INSERT INTO import_jobs (
            job_id, source_file, source_path, status, created_at, updated_at
        ) VALUES ('seed', 'seed.json', 'seed.json', 'completed', ?, ?)
        """,
        (now, now),
    )
    conn.execute(
        """
        INSERT INTO messages (
            message_id, conversation_id, sender_role, timestamp, message_type,
            content, raw_content, is_forwarded, source_file, source_position,
            job_id, created_at
        ) VALUES ('m-lib', 'conv', 'owner', 10, 'text', ?, '{}', 0, 'seed.json', 0, 'seed', ?)
        """,
        (LIBRARY, now),
    )
    conn.execute(
        """
        INSERT INTO conversation_segments (
            segment_id, conversation_id, params_hash, start_time, end_time,
            start_message_id, end_message_id, message_ids_json, overlap_message_ids_json,
            message_count, target_message_count, processing_status, opened_by, closed_by,
            created_at, updated_at
        ) VALUES (
            'seg-library', 'conv', 'test', 10, 10, 'm-lib', 'm-lib', '["m-lib"]', '[]',
            1, 1, 'completed', 'start', 'end', ?, ?
        )
        """,
        (now, now),
    )
    conn.execute(
        """
        INSERT INTO conversation_segment_messages (segment_id, message_id, position, is_overlap)
        VALUES ('seg-library', 'm-lib', 0, 0)
        """
    )
    conn.execute(
        """
        INSERT INTO memories (
            memory_id, memory_type, status, statement, importance, confidence,
            time_start, time_end, time_precision, evidence_message_ids_json,
            source_observation_ids_json, origin_segment_id, prompt_version, model_version,
            payload_json, created_at, updated_at
        ) VALUES (
            'mem-vector', 'fact', 'active', ?, 0.4, 0.8,
            1000, 1000, 'instant', '["m-lib"]', '[]', 'seg-library', 'memory-v1', 'test',
            '{}', ?, ?
        )
        """,
        (VECTOR, now, now),
    )
    conn.execute(
        """
        INSERT INTO l2_traits (
            trait_id, prompt_version, model_version, dimension, statement, situational_scope,
            confidence, evidence_count, independent_evidence_count, counter_evidence_count,
            evidence_ids_json, counter_evidence_ids_json, valid_from, valid_to, status,
            stages_json, variants_json, payload_json, created_at, updated_at
        ) VALUES (
            'trait-both', 'l2-v1', 'test', 'decision', ?, 'when choosing',
            0.9, 2, 2, 0, '["obs-1"]', '[]', 5000, 5000, 'stable',
            '[]', '[]', '{}', ?, ?
        )
        """,
        (BOTH, now, now),
    )
    conn.commit()
    conn.close()


def _index(db_path: Path, store, embedder=None):
    return run_index(db_path, embedder=embedder or ScriptedEmbedder(), store=store)


def test_hash_embeddings_overlap_on_shared_tokens():
    client = HashEmbeddingClient(dimension=128)
    moved, query, ride = client.embed(["去年从北京搬到了上海", "搬到了上海", "周末去西湖骑行"])
    again = client.embed(["搬到了上海"])[0]
    assert query == again
    assert _cosine(moved, query) > _cosine(moved, ride)
    assert _cosine(moved, query) > 0.5


def test_openai_embedding_client_preserves_input_order():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "bge-m3"
        assert request.url.path == "/v1/embeddings"
        items = [
            {"index": index, "embedding": [float(index + 1), 0.0]}
            for index, _text in enumerate(body["input"])
        ]
        items.reverse()
        return httpx.Response(200, json={"data": items, "model": "bge-m3"})

    client = OpenAIEmbeddingClient(
        base_url="http://embed.test/v1",
        model="bge-m3",
        transport=httpx.MockTransport(handler),
    )
    vectors = client.embed(["one", "two"])
    assert vectors == [[1.0, 0.0], [2.0, 0.0]]
    assert client.dimension == 2

    def fail(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(503, json={"error": "down"})

    broken = OpenAIEmbeddingClient(
        base_url="http://embed.test/v1",
        model="bge-m3",
        transport=httpx.MockTransport(fail),
    )
    with pytest.raises(Exception, match="HTTP 503"):
        broken.embed(["secret phrase"])


def test_http_reranker_uses_index_scores():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert [item["id"] for item in body["documents"]] == ["a", "b"]
        return httpx.Response(200, json={"results": [{"index": 1, "score": 0.2}, {"id": "a", "score": 0.9}]})

    reranker = HttpReranker("http://rerank.test/rank", transport=httpx.MockTransport(handler))
    first = SearchHit("a", "long_term_memories", "a", "alpha", 0.1, None, None, None, 1, None, None, {})
    second = SearchHit("b", "long_term_memories", "b", "beta", 0.1, None, None, None, 2, None, None, {})
    ordered = reranker.rerank("query", [first, second])
    assert [hit.doc_id for hit in ordered] == ["b", "a"]
    assert ordered[0].rerank_score == 0.2
    assert ordered[1].score == 0.9


def test_hybrid_search_merges_channels_and_dedupes(tmp_path: Path):
    db_path = tmp_path / "hybrid.db"
    _seed(db_path)
    store = InMemoryVectorStore()
    built = _index(db_path, store)
    rendered = json.dumps(built)
    assert LIBRARY not in rendered
    assert VECTOR not in rendered
    assert built["documents"]["total"] == 3
    assert built["embedded"] == 3
    assert built["skipped"] == 0
    assert built["documents"]["by_collection"] == {
        "conversation_segments": 1,
        "long_term_memories": 1,
        "personality_observations": 1,
    }

    again = _index(db_path, store)
    assert again["embedded"] == 0
    assert again["skipped"] == 3
    assert again["documents"]["unchanged"] == 3
    assert again["rebuilt_vectors"] is False

    held = InMemoryVectorStore()
    restored = run_index(db_path, embedder=ScriptedEmbedder(), store=held)
    assert restored["rebuilt_vectors"] is True
    assert restored["embedded"] == 3
    assert run_search(db_path, BOTH, embedder=ScriptedEmbedder(), store=held)["hits"][0]["source_id"] == "trait-both"

    both = run_search(db_path, BOTH, embedder=ScriptedEmbedder(), store=store)
    assert [hit["source_id"] for hit in both["hits"]] == ["trait-both"]
    assert both["hits"][0]["vector_rank"] == 1
    assert both["hits"][0]["fts_rank"] == 1
    assert both["hits"][0]["metadata"]["evidence_ids"] == ["obs-1"]
    assert len({hit["doc_id"] for hit in both["hits"]}) == len(both["hits"])

    mixed = run_search(db_path, LIBRARY, embedder=ScriptedEmbedder(), store=store)
    by_source = {hit["source_id"]: hit for hit in mixed["hits"]}
    assert set(by_source) == {"seg-library", "mem-vector"}
    assert by_source["seg-library"]["fts_rank"] == 1
    assert by_source["seg-library"]["vector_rank"] is None
    assert by_source["seg-library"]["metadata"]["message_ids"] == ["m-lib"]
    assert by_source["mem-vector"]["vector_rank"] == 1
    assert by_source["mem-vector"]["fts_rank"] is None
    assert by_source["mem-vector"]["metadata"]["evidence_message_ids"] == ["m-lib"]

    lexical = run_search(db_path, LIBRARY, embedder=ScriptedEmbedder(), store=store, use_vector=False)
    assert [hit["source_id"] for hit in lexical["hits"]] == ["seg-library"]
    dense = run_search(db_path, "query-vector", embedder=ScriptedEmbedder(), store=store, use_fts=False)
    assert [hit["source_id"] for hit in dense["hits"]] == ["mem-vector"]
    with pytest.raises(ValueError, match="both"):
        run_search(db_path, LIBRARY, embedder=ScriptedEmbedder(), store=store, use_vector=False, use_fts=False)


def test_filters_rerank_rebuild_and_model_change(tmp_path: Path):
    db_path = tmp_path / "filters.db"
    _seed(db_path)
    store = InMemoryVectorStore()
    embedder = ScriptedEmbedder()
    _index(db_path, store, embedder)

    historical = run_search(
        db_path,
        LIBRARY,
        embedder=embedder,
        store=store,
        memory_status="historical",
    )
    assert [hit["source_id"] for hit in historical["hits"]] == ["seg-library"]

    window = run_search(
        db_path,
        LIBRARY,
        embedder=embedder,
        store=store,
        time_start=900,
        time_end=1100,
    )
    assert [hit["source_id"] for hit in window["hits"]] == ["mem-vector"]

    trait = run_search(db_path, BOTH, embedder=embedder, store=store, dimension="decision")
    assert [hit["source_id"] for hit in trait["hits"]] == ["trait-both"]
    empty = run_search(db_path, BOTH, embedder=embedder, store=store, dimension="emotion")
    assert empty["hits"] == []
    with pytest.raises(ValueError, match="dimension"):
        run_search(db_path, BOTH, embedder=embedder, store=store, dimension="nope")

    class Reverse:
        name = "reverse"

        def rerank(self, query, hits):
            del query
            return list(reversed(hits))

    mixed = run_search(db_path, LIBRARY, embedder=embedder, store=store)
    reversed_hits = run_search(db_path, LIBRARY, embedder=embedder, store=store, reranker=Reverse())
    assert [hit["source_id"] for hit in reversed_hits["hits"]] == list(
        reversed([hit["source_id"] for hit in mixed["hits"]])
    )
    assert reversed_hits["reranker"] == "reverse"

    conn = init_db(db_path)
    conn.execute(
        "UPDATE l2_traits SET statement = ? WHERE trait_id = 'trait-both'",
        ("a different stable preference",),
    )
    conn.commit()
    conn.close()
    refreshed = _index(db_path, store, embedder)
    assert refreshed["embedded"] == 1
    assert refreshed["documents"]["upserted"] == 1
    found = run_search(db_path, "different stable preference", embedder=embedder, store=store, use_vector=False)
    assert [hit["source_id"] for hit in found["hits"]] == ["trait-both"]
    missed = run_search(db_path, BOTH, embedder=embedder, store=store)
    assert missed["hits"] == []

    conn = init_db(db_path)
    conn.execute("DELETE FROM memories WHERE memory_id = 'mem-vector'")
    conn.commit()
    conn.close()
    removed = _index(db_path, store, embedder)
    assert removed["documents"]["deleted"] == 1
    assert removed["vector_deleted"] == 1
    assert store.count("long_term_memories") == 0
    assert run_search(db_path, "query-vector", embedder=embedder, store=store)["hits"] == []

    changed = run_index(db_path, embedder=OtherModel(), store=store)
    assert changed["rebuilt_vectors"] is True
    assert changed["model"] == "scripted-v2"
    assert changed["embedded"] == 2


def test_hash_hybrid_search_finds_the_chinese_memory(tmp_path: Path):
    db_path = tmp_path / "zh.db"
    conn = init_db(db_path)
    now = utc_now()
    conn.execute(
        """
        INSERT INTO import_jobs (
            job_id, source_file, source_path, status, created_at, updated_at
        ) VALUES ('seed', 'seed.json', 'seed.json', 'completed', ?, ?)
        """,
        (now, now),
    )
    conn.execute(
        """
        INSERT INTO messages (
            message_id, conversation_id, sender_role, timestamp, message_type,
            content, raw_content, is_forwarded, source_file, source_position,
            job_id, created_at
        ) VALUES ('m-ride', 'conv', 'owner', 20, 'text', '周末去西湖骑行', '{}', 0, 'seed.json', 0, 'seed', ?)
        """,
        (now,),
    )
    conn.execute(
        """
        INSERT INTO conversation_segments (
            segment_id, conversation_id, params_hash, start_time, end_time,
            start_message_id, end_message_id, message_ids_json, overlap_message_ids_json,
            message_count, target_message_count, processing_status, opened_by, closed_by,
            created_at, updated_at
        ) VALUES (
            'seg-ride', 'conv', 'test', 20, 20, 'm-ride', 'm-ride', '["m-ride"]', '[]',
            1, 1, 'completed', 'start', 'end', ?, ?
        )
        """,
        (now, now),
    )
    conn.execute(
        """
        INSERT INTO conversation_segment_messages (segment_id, message_id, position, is_overlap)
        VALUES ('seg-ride', 'm-ride', 0, 0)
        """
    )
    conn.execute(
        """
        INSERT INTO memories (
            memory_id, memory_type, status, statement, importance, confidence,
            time_start, time_end, time_precision, evidence_message_ids_json,
            source_observation_ids_json, prompt_version, model_version,
            payload_json, created_at, updated_at
        ) VALUES (
            'mem-move', 'event', 'active', '去年从北京搬到了上海', 0.7, 0.9,
            2900, 2900, 'day', '["m-ride"]', '[]', 'memory-v1', 'test',
            '{}', ?, ?
        )
        """,
        (now, now),
    )
    conn.execute(
        """
        INSERT INTO l2_traits (
            trait_id, prompt_version, model_version, dimension, statement, situational_scope,
            confidence, evidence_count, independent_evidence_count, counter_evidence_count,
            evidence_ids_json, counter_evidence_ids_json, status,
            stages_json, variants_json, payload_json, created_at, updated_at
        ) VALUES (
            'trait-choice', 'l2-v1', 'test', 'decision', '做决定前会先列出利弊', '',
            0.8, 2, 2, 0, '[]', '[]', 'stable',
            '[]', '[]', '{}', ?, ?
        )
        """,
        (now, now),
    )
    conn.commit()
    conn.close()

    store = InMemoryVectorStore()
    embedder = HashEmbeddingClient(dimension=128, model_name="hash-v1")
    built = run_index(db_path, embedder=embedder, store=store)
    assert built["provider"] == "hash"
    found = run_search(db_path, "搬到了上海", embedder=embedder, store=store)
    assert found["hits"][0]["source_id"] == "mem-move"
    assert found["hits"][0]["collection"] == "long_term_memories"
    assert found["hits"][0]["vector_rank"] is not None
    assert found["hits"][0]["fts_rank"] is not None
    assert len({hit["doc_id"] for hit in found["hits"]}) == len(found["hits"])
    short = run_search(db_path, "西湖", embedder=embedder, store=store, use_vector=False)
    assert [hit["source_id"] for hit in short["hits"]] == ["seg-ride"]


def test_qdrant_adapter_roundtrip_uses_memory_mode(tmp_path: Path):
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient

    db_path = tmp_path / "qdrant.db"
    _seed(db_path)
    store = QdrantVectorStore(client=QdrantClient(":memory:"))
    embedder = ScriptedEmbedder()
    run_index(db_path, embedder=embedder, store=store)
    assert store.count("long_term_memories") == 1
    found = run_search(db_path, "query-vector", embedder=embedder, store=store, use_fts=False)
    assert [hit["source_id"] for hit in found["hits"]] == ["mem-vector"]
    assert found["hits"][0]["vector_score"] == pytest.approx(1.0)


def test_segmentation_rebuild_drops_the_retrieval_projection(tmp_path: Path):
    db_path = tmp_path / "rebuild.db"
    _seed(db_path)
    store = InMemoryVectorStore()
    _index(db_path, store)
    run_segmentation(db_path, rebuild=True)
    conn = init_db(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM retrieval_documents").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM retrieval_embeddings").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM retrieval_fts").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == 0
    finally:
        conn.close()


def test_cli_prints_hits_without_putting_them_in_the_index_report(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "cli.db"
    _seed(db_path)
    store = InMemoryVectorStore()
    embedder = ScriptedEmbedder()
    monkeypatch.setattr("retrieval.service.build_vector_store", lambda **kwargs: store)
    monkeypatch.setattr("retrieval.service.build_embedding_client", lambda **kwargs: embedder)
    assert build_main(["--db", str(db_path), "--provider", "hash", "--backend", "memory"]) == 0
    report = capsys.readouterr().out
    assert "status: completed" in report
    assert LIBRARY not in report
    assert search_main(["--db", str(db_path), "--query", LIBRARY]) == 0
    printed = capsys.readouterr().out
    assert "seg-library" in printed
    assert LIBRARY in printed
    assert status_main(["--db", str(db_path)]) == 0
    assert "retrieval:" in capsys.readouterr().out
    assert build_main(["--db", str(tmp_path / "missing.db")]) == 1
    assert search_main(["--db", str(tmp_path / "missing.db"), "--query", "x"]) == 1
    with pytest.raises(ValueError, match="empty"):
        run_search(db_path, "   ", embedder=embedder, store=store)
    with pytest.raises(ValueError, match="unknown collection"):
        run_search(db_path, LIBRARY, embedder=embedder, store=store, collections=["not-a-collection"])


def test_time_overlap_treats_a_single_bound_as_a_point():
    assert time_overlaps(1000, None, 900, 1100) is True
    assert time_overlaps(None, None, 900, 1100) is False
    assert time_overlaps(10, 10, None, None) is True
    assert time_overlaps(10, 10, 900, 1100) is False
