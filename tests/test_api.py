"""FastAPI import job endpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from tests.conftest import FIXTURE


def _wait_job(client: TestClient, job_id: str, db_path: Path) -> dict:
    last = None
    for _ in range(100):
        response = client.get(f"/api/v1/import/{job_id}", params={"db": str(db_path)})
        assert response.status_code == 200
        last = response.json()
        if last["status"] in {"completed", "failed"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {last}")


def test_post_and_get_import_job(tmp_path: Path):
    db_path = tmp_path / "api.db"
    client = TestClient(create_app())
    created = client.post(
        "/api/v1/import",
        json={"input_path": str(FIXTURE), "db_path": str(db_path), "resume": False},
    )
    assert created.status_code == 200
    body = created.json()
    assert body["job_id"]
    assert body["status"] in {"pending", "running", "completed"}

    job = _wait_job(client, body["job_id"], db_path)
    assert job["status"] == "completed"
    assert job["stats"]["total_messages"] == 300
    assert job["stats"]["success_count"] == 300
    assert job["stats"]["failed_count"] == 0
    assert "target_person_ratio" in job["stats"]
    assert "text_ratio" in job["stats"]
    assert "kind_counts" in job["stats"]
    assert "type_name_counts" in job["stats"]

    # Same process can resolve the database without repeating ?db=.
    remembered = client.get(f"/api/v1/import/{body['job_id']}")
    assert remembered.status_code == 200
    assert remembered.json()["job_id"] == body["job_id"]


def test_missing_file_and_unknown_job(tmp_path: Path):
    client = TestClient(create_app())
    missing = client.post("/api/v1/import", json={"input_path": str(tmp_path / "nope.json")})
    assert missing.status_code == 400
    unknown = client.get("/api/v1/import/does-not-exist", params={"db": str(tmp_path / "missing.db")})
    assert unknown.status_code == 404


def test_api_records_schema_failure(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"metadata": {"application": "Nope"}, "messages": []}), encoding="utf-8")
    db_path = tmp_path / "bad.db"
    client = TestClient(create_app())
    created = client.post("/api/v1/import", json={"input_path": str(path), "db_path": str(db_path)})
    assert created.status_code == 200
    job = _wait_job(client, created.json()["job_id"], db_path)
    assert job["status"] == "failed"
    assert "Nope" in job["error_message"]
