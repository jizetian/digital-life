"""Import API. Bind it to localhost; chat exports stay on this machine."""

from __future__ import annotations

import sys
from pathlib import Path
from threading import Lock, Thread

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from common.config import default_db_path
from common.logger import configure_logging, get_logger
from importer.import_service import begin_import, get_job, run_import

log = get_logger(__name__)
_JOB_DB: dict[str, str] = {}
_LOCK = Lock()


class ImportRequest(BaseModel):
    input_path: str = Field(..., description="Local path to a Moect JSON export")
    db_path: str | None = Field(None, description="SQLite file. Defaults to the configured local database")
    resume: bool = False


def create_app() -> FastAPI:
    app = FastAPI(
        title="ChatTwin Import API",
        version="0.1.0",
        description="Local-first Moect export import. No distillation or chat agent.",
    )

    @app.post("/api/v1/import")
    def start_import(body: ImportRequest) -> dict:
        source = Path(body.input_path).expanduser()
        if not source.is_file():
            raise HTTPException(status_code=400, detail=f"export file not found: {source}")
        database = Path(body.db_path).expanduser() if body.db_path else default_db_path()
        try:
            job = begin_import(source, database, resume=body.resume)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=f"export file not found: {exc}") from exc
        except OSError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        job_id = job["job_id"]
        with _LOCK:
            _JOB_DB[job_id] = str(Path(database).expanduser().resolve())

        def _worker() -> None:
            try:
                run_import(source, database, resume=True, job_id=job_id)
            except Exception:
                log.exception("import job %s failed", job_id)

        Thread(target=_worker, name=f"chattwin-import-{job_id[:8]}", daemon=True).start()
        return {"job_id": job_id, "status": job["status"]}

    @app.get("/api/v1/import/{job_id}")
    def read_import(
        job_id: str,
        db: str | None = Query(None, description="SQLite path, if this process did not start the job"),
    ) -> dict:
        database = _database_for(job_id, db)
        job = get_job(database, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="import job not found")
        return job

    return app


def _database_for(job_id: str, db: str | None) -> str:
    if db:
        return str(Path(db).expanduser().resolve())
    with _LOCK:
        recorded = _JOB_DB.get(job_id)
    if recorded:
        return recorded
    raise HTTPException(status_code=404, detail="unknown job; pass ?db= pointing at the SQLite file")


configure_logging()
app = create_app()
