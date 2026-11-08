"""Local import, segmentation, distillation, personality, memory, search, and chat API.

Bind it to 127.0.0.1. Chat exports, the SQLite file, and the Qdrant directory
stay on this machine. The chat reply is synchronous JSON. It is not streamed.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from threading import Lock, Thread

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from chat.agent import ChatModelError, ChatNotFound, chat_turn
from chat.store import get_session, list_sessions
from common.clock import utc_now
from common.config import default_db_path
from common.database import init_db
from common.logger import configure_logging, get_logger
from distillation.engine import run_distillation
from distillation.llm import LLMError
from distillation.status import pipeline_status
from importer.import_service import begin_import, get_job, run_import
from memory.memory_engine import run_memory
from memory.status import memory_snapshot
from memory.timeline_builder import build_timeline
from personality.engine import run_personality
from personality.status import personality_snapshot
from retrieval.errors import RetrievalError
from retrieval.service import run_index, run_search
from retrieval.status import retrieval_snapshot_conn
from segmentation.service import run_segmentation

log = get_logger(__name__)
_JOB_DB: dict[str, str] = {}
_LOCK = Lock()


class ImportRequest(BaseModel):
    input_path: str = Field(..., description="Local path to a Moect JSON export")
    db_path: str | None = Field(None, description="SQLite file. Defaults to the configured local database")
    resume: bool = False


class SegmentRequest(BaseModel):
    db_path: str | None = None
    gap_minutes: float | None = None
    max_messages: int | None = None
    overlap_messages: int | None = None
    overlap_on_time_gap: bool | None = None
    rebuild: bool = False


class DistillRequest(BaseModel):
    db_path: str | None = None
    batch_size: int | None = None
    max_tasks: int | None = None
    retry_failed: bool = False
    model: str | None = None
    model_version: str | None = None
    base_url: str | None = None
    prompt_version: str | None = None


class PersonalityRequest(BaseModel):
    db_path: str | None = None
    output_dir: str | None = None
    batch_size: int | None = None
    max_tasks: int | None = None
    min_independent: int | None = None
    retry_failed: bool = False
    model: str | None = None
    model_version: str | None = None
    base_url: str | None = None
    prompt_version: str | None = None


class MemoryRequest(BaseModel):
    db_path: str | None = None
    batch_size: int | None = None
    max_tasks: int | None = None
    retry_failed: bool = False
    model: str | None = None
    model_version: str | None = None
    base_url: str | None = None
    prompt_version: str | None = None


class EmbedRequest(BaseModel):
    db_path: str | None = None
    rebuild: bool = False
    collections: list[str] | None = None
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    batch_size: int | None = None
    vector_backend: str | None = None
    qdrant_path: str | None = None
    qdrant_url: str | None = None


class SearchRequest(BaseModel):
    query: str
    db_path: str | None = None
    collections: list[str] | None = None
    limit: int | None = Field(None, ge=1, le=100)
    use_vector: bool = True
    use_fts: bool = True
    rerank: bool = False
    memory_status: str | None = None
    memory_type: str | None = None
    trait_status: str | None = None
    dimension: str | None = None
    time_start: int | None = None
    time_end: int | None = None
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    vector_backend: str | None = None
    qdrant_path: str | None = None
    qdrant_url: str | None = None


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    db_path: str | None = None
    target_name: str | None = None
    use_vector: bool = True
    use_fts: bool = True
    limit: int | None = Field(None, ge=1, le=100)


_UI_PATH = Path(__file__).resolve().parents[2] / "web" / "index.html"


def create_app() -> FastAPI:
    app = FastAPI(
        title="ChatTwin Local API",
        version="0.6.0",
        description=(
            "Local-first Moect import, segmentation, distillation, personality, long-term memory, "
            "hybrid search, and a digital-twin chat. The chat reply is synchronous JSON."
        ),
    )

    @app.get("/")
    def chat_page():
        if not _UI_PATH.is_file():
            raise HTTPException(status_code=404, detail="chat page is missing")
        return FileResponse(_UI_PATH, media_type="text/html; charset=utf-8")

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

    @app.post("/api/v1/segment")
    def start_segment(body: SegmentRequest) -> dict:
        database = _existing_db(body.db_path)
        return _launch(
            database,
            "segmentation",
            {"rebuild": body.rebuild},
            lambda: run_segmentation(
                database,
                gap_minutes=body.gap_minutes,
                max_messages=body.max_messages,
                overlap_messages=body.overlap_messages,
                overlap_on_time_gap=body.overlap_on_time_gap,
                rebuild=body.rebuild,
            ),
        )

    @app.get("/api/v1/segment/{job_id}")
    def read_segment(
        job_id: str,
        db: str | None = Query(None, description="SQLite path, if this process did not start the job"),
    ) -> dict:
        return _read_processing_job(job_id, db)

    @app.post("/api/v1/distill")
    def start_distill(body: DistillRequest) -> dict:
        database = _existing_db(body.db_path)
        return _launch(
            database,
            "distillation",
            {"batch_size": body.batch_size, "retry_failed": body.retry_failed},
            lambda: run_distillation(
                database,
                batch_size=body.batch_size,
                max_tasks=body.max_tasks,
                retry_failed=body.retry_failed,
                model=body.model,
                model_version=body.model_version,
                base_url=body.base_url,
                prompt_version=body.prompt_version,
            ),
        )

    @app.get("/api/v1/distill/status")
    def read_distill_status(
        db: str | None = Query(None, description="SQLite path. Defaults to the configured local database"),
    ) -> dict:
        database = Path(db).expanduser() if db else default_db_path()
        try:
            return pipeline_status(database)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"database not found: {exc}") from exc

    @app.get("/api/v1/distill/{job_id}")
    def read_distill(
        job_id: str,
        db: str | None = Query(None, description="SQLite path, if this process did not start the job"),
    ) -> dict:
        return _read_processing_job(job_id, db)

    @app.post("/api/v1/personality")
    def start_personality(body: PersonalityRequest) -> dict:
        database = _existing_db(body.db_path)
        return _launch(
            database,
            "personality",
            {"batch_size": body.batch_size, "retry_failed": body.retry_failed},
            lambda: run_personality(
                database,
                output_dir=body.output_dir,
                batch_size=body.batch_size,
                max_tasks=body.max_tasks,
                min_independent=body.min_independent,
                retry_failed=body.retry_failed,
                model=body.model,
                model_version=body.model_version,
                base_url=body.base_url,
                prompt_version=body.prompt_version,
            ),
        )

    @app.get("/api/v1/personality/status")
    def read_personality_status(
        db: str | None = Query(None, description="SQLite path. Defaults to the configured local database"),
    ) -> dict:
        database = Path(db).expanduser() if db else default_db_path()
        try:
            return personality_snapshot(database)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"database not found: {exc}") from exc

    @app.get("/api/v1/personality/{job_id}")
    def read_personality(
        job_id: str,
        db: str | None = Query(None, description="SQLite path, if this process did not start the job"),
    ) -> dict:
        return _read_processing_job(job_id, db)

    @app.post("/api/v1/memory")
    def start_memory(body: MemoryRequest) -> dict:
        database = _existing_db(body.db_path)
        return _launch(
            database,
            "memory",
            {"batch_size": body.batch_size, "retry_failed": body.retry_failed},
            lambda: run_memory(
                database,
                batch_size=body.batch_size,
                max_tasks=body.max_tasks,
                retry_failed=body.retry_failed,
                model=body.model,
                model_version=body.model_version,
                base_url=body.base_url,
                prompt_version=body.prompt_version,
            ),
        )

    @app.get("/api/v1/memory/status")
    def read_memory_status(
        db: str | None = Query(None, description="SQLite path. Defaults to the configured local database"),
    ) -> dict:
        database = Path(db).expanduser() if db else default_db_path()
        try:
            return memory_snapshot(database)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"database not found: {exc}") from exc

    @app.get("/api/v1/memory/timeline")
    def read_memory_timeline(
        db: str | None = Query(None, description="SQLite path. Defaults to the configured local database"),
        entity: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = Query(100, ge=1, le=500),
    ) -> dict:
        database = Path(db).expanduser() if db else default_db_path()
        if not database.is_file():
            raise HTTPException(status_code=404, detail=f"database not found: {database}")
        conn = init_db(database)
        try:
            try:
                items = build_timeline(
                    conn,
                    entity=entity,
                    memory_type=memory_type,
                    status=status,
                    limit=limit,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            conn.close()
        return {"count": len(items), "items": items}

    @app.get("/api/v1/memory/{job_id}")
    def read_memory(
        job_id: str,
        db: str | None = Query(None, description="SQLite path, if this process did not start the job"),
    ) -> dict:
        return _read_processing_job(job_id, db)

    @app.post("/api/v1/embeddings")
    def start_embeddings(body: EmbedRequest) -> dict:
        database = _existing_db(body.db_path)
        return _launch(
            database,
            "embeddings",
            {"rebuild": body.rebuild, "collections": body.collections},
            lambda: run_index(
                database,
                collections=body.collections,
                rebuild=body.rebuild,
                provider=body.provider,
                model=body.model,
                base_url=body.base_url,
                batch_size=body.batch_size,
                vector_backend=body.vector_backend,
                qdrant_path=body.qdrant_path,
                qdrant_url=body.qdrant_url,
            ),
        )

    @app.get("/api/v1/embeddings/status")
    def read_embeddings_status(
        db: str | None = Query(None, description="SQLite path. Defaults to the configured local database"),
    ) -> dict:
        database = Path(db).expanduser() if db else default_db_path()
        if not database.is_file():
            raise HTTPException(status_code=404, detail=f"database not found: {database}")
        conn = init_db(database)
        try:
            return retrieval_snapshot_conn(conn)
        finally:
            conn.close()

    @app.get("/api/v1/embeddings/{job_id}")
    def read_embeddings(
        job_id: str,
        db: str | None = Query(None, description="SQLite path, if this process did not start the job"),
    ) -> dict:
        return _read_processing_job(job_id, db)

    @app.post("/api/v1/search")
    def search_post(body: SearchRequest) -> dict:
        database = _existing_db(body.db_path)
        return _search(
            database,
            body.query,
            collections=body.collections,
            limit=body.limit,
            use_vector=body.use_vector,
            use_fts=body.use_fts,
            rerank=body.rerank,
            memory_status=body.memory_status,
            memory_type=body.memory_type,
            trait_status=body.trait_status,
            dimension=body.dimension,
            time_start=body.time_start,
            time_end=body.time_end,
            provider=body.provider,
            model=body.model,
            base_url=body.base_url,
            vector_backend=body.vector_backend,
            qdrant_path=body.qdrant_path,
            qdrant_url=body.qdrant_url,
        )

    @app.get("/api/v1/search")
    def search_get(
        q: str = Query(..., description="Search text"),
        db: str | None = Query(None, description="SQLite path. Defaults to the configured local database"),
        collection: list[str] | None = Query(None, description="Repeat to limit collections"),
        limit: int = Query(10, ge=1, le=100),
        use_vector: bool = True,
        use_fts: bool = True,
        rerank: bool = False,
        memory_status: str | None = None,
        memory_type: str | None = None,
        trait_status: str | None = None,
        dimension: str | None = None,
    ) -> dict:
        database = Path(db).expanduser() if db else default_db_path()
        if not database.is_file():
            raise HTTPException(status_code=404, detail=f"database not found: {database}")
        return _search(
            database,
            q,
            collections=collection,
            limit=limit,
            use_vector=use_vector,
            use_fts=use_fts,
            rerank=rerank,
            memory_status=memory_status,
            memory_type=memory_type,
            trait_status=trait_status,
            dimension=dimension,
        )

    @app.post("/api/v1/chat")
    def post_chat(body: ChatRequest) -> dict:
        database = _existing_db(body.db_path)
        try:
            return chat_turn(
                database,
                body.message,
                session_id=body.session_id,
                target_name=body.target_name,
                use_vector=body.use_vector,
                use_fts=body.use_fts,
                limit=body.limit,
            )
        except ChatNotFound:
            raise HTTPException(status_code=404, detail="chat session not found") from None
        except ChatModelError as exc:
            raise HTTPException(
                status_code=502,
                detail={"message": str(exc), "session_id": exc.session_id},
            ) from exc
        except (LLMError, RetrievalError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/chat/sessions")
    def read_chat_sessions(
        db: str | None = Query(None, description="SQLite path. Defaults to the configured local database"),
        limit: int = Query(50, ge=1, le=200),
    ) -> dict:
        database = _database_file(db)
        conn = init_db(database)
        try:
            sessions = list_sessions(conn, limit)
        finally:
            conn.close()
        return {"sessions": sessions}

    @app.get("/api/v1/chat/sessions/{session_id}")
    def read_chat_session(
        session_id: str,
        db: str | None = Query(None, description="SQLite path. Defaults to the configured local database"),
    ) -> dict:
        database = _database_file(db)
        conn = init_db(database)
        try:
            session = get_session(conn, session_id)
        finally:
            conn.close()
        if session is None:
            raise HTTPException(status_code=404, detail="chat session not found")
        return session

    return app


def _search(database: Path, query: str, **kwargs) -> dict:
    try:
        return run_search(database, query, **kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"database not found: {exc}") from exc
    except (RetrievalError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _database_file(db: str | None) -> Path:
    database = Path(db).expanduser() if db else default_db_path()
    if not database.is_file():
        raise HTTPException(status_code=404, detail=f"database not found: {database}")
    return database


def _existing_db(db_path: str | None) -> Path:
    database = Path(db_path).expanduser() if db_path else default_db_path()
    if not database.is_file():
        raise HTTPException(status_code=400, detail=f"database not found: {database}")
    return database.resolve()


def _launch(database: Path, job_type: str, payload: dict, worker) -> dict:
    job_id = uuid.uuid4().hex
    now = utc_now()
    conn = init_db(database)
    try:
        conn.execute(
            """
            INSERT INTO processing_jobs (job_id, job_type, status, payload_json, created_at, updated_at)
            VALUES (?, ?, 'pending', ?, ?, ?)
            """,
            (job_id, job_type, json.dumps(payload), now, now),
        )
        conn.commit()
    finally:
        conn.close()
    with _LOCK:
        _JOB_DB[job_id] = str(database)

    def _run() -> None:
        _set_job(database, job_id, "running", payload)
        try:
            result = worker()
            # A paused distillation still finished this HTTP call; tasks left behind
            # are resumed by the next POST. ``running`` would look like a live worker.
            job_status = result.get("status", "completed")
            if job_status == "running":
                job_status = "paused"
            _set_job(database, job_id, job_status, result)
        except Exception as exc:
            log.exception("%s job %s failed", job_type, job_id)
            _set_job(database, job_id, "failed", {"error": f"{type(exc).__name__}: {exc}"})

    Thread(target=_run, name=f"chattwin-{job_type}-{job_id[:8]}", daemon=True).start()
    return {"job_id": job_id, "status": "pending"}


def _set_job(database: Path, job_id: str, status: str, payload: dict) -> None:
    conn = init_db(database)
    try:
        conn.execute(
            """
            UPDATE processing_jobs
            SET status = ?, payload_json = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (status, json.dumps(payload, ensure_ascii=False), utc_now(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def _read_processing_job(job_id: str, db: str | None) -> dict:
    database = _database_for(job_id, db)
    if not Path(database).is_file():
        raise HTTPException(status_code=404, detail="database not found")
    conn = init_db(database)
    try:
        row = conn.execute("SELECT * FROM processing_jobs WHERE job_id = ?", (job_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    payload = json.loads(row["payload_json"]) if row["payload_json"] else None
    return {
        "job_id": row["job_id"],
        "job_type": row["job_type"],
        "status": row["status"],
        "payload": payload,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


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
