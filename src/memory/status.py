"""Counts for memory tasks, stored memories, entities, and the checkpoint."""

from __future__ import annotations

from pathlib import Path

from common.database import init_db
from common.statuses import PIPELINE_STATUSES
from memory.types import ENTITY_STATUSES, MEMORY_STATUSES, MEMORY_TYPES

TASK_KINDS = ("extract", "merge")
RELATION_TYPES = ("supersedes", "historical_of", "conflicts_with", "supports", "related_to")


def memory_snapshot(db_path: str | Path) -> dict:
    database = Path(db_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"database not found: {database}")
    conn = init_db(database)
    try:
        return memory_snapshot_conn(conn)
    finally:
        conn.close()


def memory_counts_for(conn, *, prompt_version: str, model_version: str) -> dict:
    return _memory_counts(conn, prompt_version=prompt_version, model_version=model_version)


def memory_snapshot_conn(conn) -> dict:
    return {
        "tasks": _task_counts(conn),
        "memories": _memory_counts(conn),
        "entities": _entity_counts(conn),
        "relations": _relation_counts(conn),
        "evidence": _evidence_count(conn),
        "checkpoints": _checkpoints(conn),
    }


def format_memory_status(status: dict) -> str:
    lines = ["memory_tasks: " + str(status["tasks"]["total"])]
    for name, count in status["tasks"]["by_status"].items():
        if count:
            lines.append(f"  {name}: {count}")
    kinds = status["tasks"]["by_kind"]
    if any(kinds.values()):
        lines.append("  kinds: " + ", ".join(f"{kind}={kinds[kind]}" for kind in TASK_KINDS if kinds[kind]))
    lines.append("memories: " + str(status["memories"]["total"]))
    for name, count in status["memories"]["by_status"].items():
        if count:
            lines.append(f"  {name}: {count}")
    for name, count in status["memories"]["by_type"].items():
        if count:
            lines.append(f"  {name}: {count}")
    lines.append("entities: " + str(status["entities"]["total"]))
    for name, count in status["entities"]["by_status"].items():
        if count:
            lines.append(f"  {name}: {count}")
    lines.append("memory_relations: " + str(status["relations"]["total"]))
    for name, count in status["relations"]["by_type"].items():
        if count:
            lines.append(f"  {name}: {count}")
    lines.append(f"memory_evidence: {status['evidence']}")
    checkpoints = status.get("checkpoints") or []
    if not checkpoints:
        lines.append("memory_checkpoint: (none)")
    for checkpoint in checkpoints:
        lines.append(
            "memory_checkpoint: "
            f"{checkpoint['status']} "
            f"prompt={checkpoint['prompt_version']} "
            f"model={checkpoint['model_version']} "
            f"memories={checkpoint['memories_written']}"
        )
    return "\n".join(lines)


def _task_counts(conn) -> dict:
    by_status = {status: 0 for status in PIPELINE_STATUSES}
    for row in conn.execute("SELECT status, COUNT(*) AS n FROM memory_tasks GROUP BY status"):
        if row["status"] in by_status:
            by_status[row["status"]] = int(row["n"])
    by_kind = {kind: 0 for kind in TASK_KINDS}
    for row in conn.execute("SELECT task_kind, COUNT(*) AS n FROM memory_tasks GROUP BY task_kind"):
        if row["task_kind"] in by_kind:
            by_kind[row["task_kind"]] = int(row["n"])
    return {"total": sum(by_status.values()), "by_status": by_status, "by_kind": by_kind}


def _memory_counts(conn, *, prompt_version: str | None = None, model_version: str | None = None) -> dict:
    clauses = []
    params: list = []
    if prompt_version is not None:
        clauses.append("prompt_version = ?")
        params.append(prompt_version)
    if model_version is not None:
        clauses.append("model_version = ?")
        params.append(model_version)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    by_status = {status: 0 for status in MEMORY_STATUSES}
    for row in conn.execute(f"SELECT status, COUNT(*) AS n FROM memories {where} GROUP BY status", params):
        if row["status"] in by_status:
            by_status[row["status"]] = int(row["n"])
    by_type = {memory_type: 0 for memory_type in MEMORY_TYPES}
    for row in conn.execute(
        f"SELECT memory_type, COUNT(*) AS n FROM memories {where} GROUP BY memory_type",
        params,
    ):
        if row["memory_type"] in by_type:
            by_type[row["memory_type"]] = int(row["n"])
    return {"total": sum(by_status.values()), "by_status": by_status, "by_type": by_type}


def _entity_counts(conn) -> dict:
    by_status = {status: 0 for status in ENTITY_STATUSES}
    for row in conn.execute("SELECT status, COUNT(*) AS n FROM entities GROUP BY status"):
        if row["status"] in by_status:
            by_status[row["status"]] = int(row["n"])
    return {"total": sum(by_status.values()), "by_status": by_status}


def _relation_counts(conn) -> dict:
    by_type = {relation: 0 for relation in RELATION_TYPES}
    for row in conn.execute("SELECT relation_type, COUNT(*) AS n FROM memory_relations GROUP BY relation_type"):
        if row["relation_type"] in by_type:
            by_type[row["relation_type"]] = int(row["n"])
    return {"total": sum(by_type.values()), "by_type": by_type}


def _evidence_count(conn) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM memory_evidence").fetchone()["n"])


def _checkpoints(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT prompt_version, model_version, status, extracts_completed, merges_completed,
               memories_written, updated_at
        FROM memory_checkpoint
        ORDER BY updated_at
        """
    ).fetchall()
    return [
        {
            "prompt_version": row["prompt_version"],
            "model_version": row["model_version"],
            "status": row["status"],
            "extracts_completed": row["extracts_completed"],
            "merges_completed": row["merges_completed"],
            "memories_written": row["memories_written"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]
