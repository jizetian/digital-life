"""Long-term memory over the local API."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from tests.test_memory import RAW, FakeLLM, _prepare, _scripted


def _wait(client: TestClient, job_id: str, db_path: Path) -> dict:
    last = None
    for _ in range(100):
        response = client.get(f"/api/v1/memory/{job_id}", params={"db": str(db_path)})
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in {"completed", "failed", "paused"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {last}")


def test_memory_job_builds_a_timeline(tmp_path: Path, monkeypatch):
    db_path = _prepare(tmp_path)
    fake = FakeLLM(_scripted)
    monkeypatch.setattr("memory.memory_engine.build_llm_client", lambda **kwargs: fake)
    client = TestClient(create_app())
    created = client.post(
        "/api/v1/memory",
        json={"db_path": str(db_path), "model_version": "api-model"},
    )
    assert created.status_code == 200, created.text
    job = _wait(client, created.json()["job_id"], db_path)
    assert job["status"] == "completed", job
    payload = json.dumps(job["payload"], ensure_ascii=False)
    assert RAW not in payload
    assert job["payload"]["memories"]["total"] == 4
    assert job["payload"]["memories"]["by_status"]["superseded"] == 1

    status = client.get("/api/v1/memory/status", params={"db": str(db_path)})
    assert status.status_code == 200
    assert status.json()["memories"]["by_status"]["historical"] == 1
    assert status.json()["entities"]["by_status"]["confirmed"] >= 1

    timeline = client.get(
        "/api/v1/memory/timeline",
        params={"db": str(db_path), "status": "active"},
    )
    assert timeline.status_code == 200
    body = timeline.json()
    assert body["count"] == 2
    rendered = json.dumps(body, ensure_ascii=False)
    assert "搬到了上海" in rendered
    assert RAW not in rendered
    assert "我住在北京" not in rendered

    pipeline = client.get("/api/v1/distill/status", params={"db": str(db_path)})
    assert pipeline.status_code == 200
    assert pipeline.json()["memory"]["memories"]["total"] == 4

    remembered = client.get(f"/api/v1/memory/{created.json()['job_id']}")
    assert remembered.status_code == 200


def test_memory_routes_reject_a_missing_database(tmp_path: Path):
    client = TestClient(create_app())
    missing = client.post("/api/v1/memory", json={"db_path": str(tmp_path / "missing.db")})
    assert missing.status_code == 400
    status = client.get("/api/v1/memory/status", params={"db": str(tmp_path / "missing.db")})
    assert status.status_code == 404
    timeline = client.get("/api/v1/memory/timeline", params={"db": str(tmp_path / "missing.db")})
    assert timeline.status_code == 404
    unknown = client.get("/api/v1/memory/does-not-exist", params={"db": str(tmp_path / "missing.db")})
    assert unknown.status_code == 404
    bad = client.get(
        "/api/v1/memory/timeline",
        params={"db": str(tmp_path / "missing.db"), "status": "nope"},
    )
    assert bad.status_code == 404
