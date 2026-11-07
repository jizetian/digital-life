"""Search and index endpoints. Embeddings and Qdrant are both fakes."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from retrieval.vector_store import InMemoryVectorStore
from tests.test_retrieval import LIBRARY, ScriptedEmbedder, _seed


def _wait(client: TestClient, job_id: str, db_path: Path) -> dict:
    last = None
    for _ in range(100):
        response = client.get(f"/api/v1/embeddings/{job_id}", params={"db": str(db_path)})
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in {"completed", "failed", "paused"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {last}")


def test_embeddings_job_then_search(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "api.db"
    _seed(db_path)
    store = InMemoryVectorStore()
    embedder = ScriptedEmbedder()
    monkeypatch.setattr("retrieval.service.build_vector_store", lambda **kwargs: store)
    monkeypatch.setattr("retrieval.service.build_embedding_client", lambda **kwargs: embedder)
    client = TestClient(create_app())

    created = client.post("/api/v1/embeddings", json={"db_path": str(db_path), "provider": "hash"})
    assert created.status_code == 200, created.text
    job = _wait(client, created.json()["job_id"], db_path)
    assert job["status"] == "completed", job
    payload = json.dumps(job["payload"], ensure_ascii=False)
    assert LIBRARY not in payload
    assert job["payload"]["embedded"] == 3
    assert job["payload"]["documents"]["total"] == 3

    status = client.get("/api/v1/embeddings/status", params={"db": str(db_path)})
    assert status.status_code == 200
    assert status.json()["embedded"] == 3
    assert status.json()["by_collection"]["personality_observations"] == 1

    found = client.post(
        "/api/v1/search",
        json={"db_path": str(db_path), "query": LIBRARY, "limit": 5},
    )
    assert found.status_code == 200, found.text
    body = found.json()
    sources = {hit["source_id"] for hit in body["hits"]}
    assert "seg-library" in sources
    assert body["used_vector"] is True
    assert body["used_fts"] is True

    listed = client.get("/api/v1/search", params={"db": str(db_path), "q": "query-vector", "use_fts": False})
    assert listed.status_code == 200, listed.text
    assert [hit["source_id"] for hit in listed.json()["hits"]] == ["mem-vector"]

    filtered = client.post(
        "/api/v1/search",
        json={"db_path": str(db_path), "query": LIBRARY, "memory_status": "historical"},
    )
    assert [hit["source_id"] for hit in filtered.json()["hits"]] == ["seg-library"]


def test_search_routes_reject_a_missing_database_and_an_empty_query(tmp_path: Path, monkeypatch):
    store = InMemoryVectorStore()
    monkeypatch.setattr("retrieval.service.build_vector_store", lambda **kwargs: store)
    monkeypatch.setattr("retrieval.service.build_embedding_client", lambda **kwargs: ScriptedEmbedder())
    client = TestClient(create_app())
    missing = tmp_path / "missing.db"
    created = client.post("/api/v1/embeddings", json={"db_path": str(missing)})
    assert created.status_code == 400
    status = client.get("/api/v1/embeddings/status", params={"db": str(missing)})
    assert status.status_code == 404
    posted = client.post("/api/v1/search", json={"db_path": str(missing), "query": "hello"})
    assert posted.status_code == 400
    fetched = client.get("/api/v1/search", params={"db": str(missing), "q": "hello"})
    assert fetched.status_code == 404

    db_path = tmp_path / "empty-query.db"
    _seed(db_path)
    blank = client.post("/api/v1/search", json={"db_path": str(db_path), "query": "  "})
    assert blank.status_code == 400
    unknown = client.get("/api/v1/embeddings/does-not-exist", params={"db": str(db_path)})
    assert unknown.status_code == 404
