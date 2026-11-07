"""FastAPI segmentation and distillation jobs."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from common.database import init_db
from importer.import_service import run_import
from tests.conftest import FIXTURE
from tests.test_distillation import FakeLLM, _fact_response


def _wait(client: TestClient, path: str, job_id: str, db_path: Path) -> dict:
    last = None
    for _ in range(100):
        response = client.get(path.format(job_id=job_id), params={"db": str(db_path)})
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in {"completed", "failed"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {last}")


def test_segment_and_distill_jobs(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "api.db"
    run_import(FIXTURE, db_path)
    fake = FakeLLM(_fact_response)
    monkeypatch.setattr("distillation.engine.build_llm_client", lambda **kwargs: fake)
    client = TestClient(create_app())

    created = client.post("/api/v1/segment", json={"db_path": str(db_path)})
    assert created.status_code == 200
    body = created.json()
    assert body["job_id"]
    segment_job = _wait(client, "/api/v1/segment/{job_id}", body["job_id"], db_path)
    assert segment_job["status"] == "completed"
    assert segment_job["payload"]["segment_count"] >= 2
    assert segment_job["payload"]["message_total"] == 300
    assert "你已添加了" not in json.dumps(segment_job["payload"], ensure_ascii=False)

    distilled = client.post(
        "/api/v1/distill",
        json={"db_path": str(db_path), "batch_size": 4, "model_version": "api-model"},
    )
    assert distilled.status_code == 200
    distill_job = _wait(client, "/api/v1/distill/{job_id}", distilled.json()["job_id"], db_path)
    assert distill_job["status"] == "completed", distill_job
    assert distill_job["payload"]["observations"] >= 1
    assert "提到了一件事" not in json.dumps(distill_job["payload"], ensure_ascii=False)
    assert fake.calls
    assert len(fake.calls) == distill_job["payload"]["attempts_completed"]

    status = client.get("/api/v1/distill/status", params={"db": str(db_path)})
    assert status.status_code == 200
    snapshot = status.json()
    assert snapshot["segments"]["total"] >= 2
    assert snapshot["observations"]["total"] >= 1
    assert snapshot["observations"]["by_category"]["fact"] >= 1
    assert snapshot["segmentation_checkpoint"]["status"] == "completed"

    conn = init_db(db_path)
    try:
        rows = conn.execute(
            "SELECT segment_id, evidence_message_ids_json FROM l1_observations"
        ).fetchall()
        assert rows
        for row in rows:
            evidence = json.loads(row["evidence_message_ids_json"])
            allowed = json.loads(
                conn.execute(
                    "SELECT message_ids_json FROM conversation_segments WHERE segment_id = ?",
                    (row["segment_id"],),
                ).fetchone()["message_ids_json"]
            )
            assert set(evidence) <= set(allowed)
    finally:
        conn.close()

    remembered = client.get(f"/api/v1/distill/{distilled.json()['job_id']}")
    assert remembered.status_code == 200


def test_pipeline_routes_reject_missing_files(tmp_path: Path):
    client = TestClient(create_app())
    missing = client.post("/api/v1/segment", json={"db_path": str(tmp_path / "missing.db")})
    assert missing.status_code == 400
    unknown = client.get("/api/v1/distill/does-not-exist", params={"db": str(tmp_path / "missing.db")})
    assert unknown.status_code == 404
    status = client.get("/api/v1/distill/status", params={"db": str(tmp_path / "missing.db")})
    assert status.status_code == 404
