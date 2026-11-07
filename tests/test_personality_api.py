"""Personality profile over the local API."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from tests.test_personality import FakeLLM, _fixture_responder, _load_fixture_db


def _wait(client: TestClient, job_id: str, db_path: Path) -> dict:
    last = None
    for _ in range(100):
        response = client.get(f"/api/v1/personality/{job_id}", params={"db": str(db_path)})
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in {"completed", "failed", "paused"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {last}")


def test_personality_job_writes_a_profile(tmp_path: Path, monkeypatch):
    db_path = _load_fixture_db(tmp_path)
    output = tmp_path / "api-profile"
    fake = FakeLLM(_fixture_responder)
    monkeypatch.setattr("personality.engine.build_llm_client", lambda **kwargs: fake)
    client = TestClient(create_app())
    created = client.post(
        "/api/v1/personality",
        json={
            "db_path": str(db_path),
            "output_dir": str(output),
            "batch_size": 20,
            "model_version": "api-model",
        },
    )
    assert created.status_code == 200, created.text
    job = _wait(client, created.json()["job_id"], db_path)
    assert job["status"] == "completed", job
    payload = json.dumps(job["payload"], ensure_ascii=False)
    assert "先给短结论" not in payload
    assert "RAW_MESSAGE_" not in payload
    assert job["payload"]["traits"]["stable"] == 2
    evidence = json.loads((output / "personality_evidence.json").read_text(encoding="utf-8"))
    assert evidence["dimensions"]["values"]["traits"][0]["evidence_count"] == 2

    status = client.get("/api/v1/personality/status", params={"db": str(db_path)})
    assert status.status_code == 200
    body = status.json()
    assert body["traits"]["by_status"]["stable"] == 2
    assert body["checkpoints"][0]["profile_current"] is True
    pipeline = client.get("/api/v1/distill/status", params={"db": str(db_path)})
    assert pipeline.status_code == 200
    assert pipeline.json()["personality"]["traits"]["total"] >= 2

    remembered = client.get(f"/api/v1/personality/{created.json()['job_id']}")
    assert remembered.status_code == 200


def test_personality_routes_reject_a_missing_database(tmp_path: Path):
    client = TestClient(create_app())
    missing = client.post("/api/v1/personality", json={"db_path": str(tmp_path / "missing.db")})
    assert missing.status_code == 400
    status = client.get("/api/v1/personality/status", params={"db": str(tmp_path / "missing.db")})
    assert status.status_code == 404
    unknown = client.get("/api/v1/personality/does-not-exist", params={"db": str(tmp_path / "missing.db")})
    assert unknown.status_code == 404
