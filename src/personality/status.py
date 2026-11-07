"""Counts for personality tasks, L2 traits, and the last profile checkpoint."""

from __future__ import annotations

from pathlib import Path

from common.database import init_db
from common.statuses import PIPELINE_STATUSES

TRAIT_STATUSES = ("stable", "provisional", "contested")
TASK_KINDS = ("merge", "reduce", "contradict")


def personality_snapshot(db_path: str | Path) -> dict:
    database = Path(db_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"database not found: {database}")
    conn = init_db(database)
    try:
        return personality_snapshot_conn(conn)
    finally:
        conn.close()


def personality_snapshot_conn(conn) -> dict:
    tasks = _task_counts(conn)
    traits = _trait_counts(conn)
    checkpoints = []
    rows = conn.execute(
        """
        SELECT prompt_version, model_version, observation_fingerprint, status,
               output_dir, profile_written_at, min_independent,
               traits_stable, traits_provisional, traits_contested, updated_at
        FROM personality_checkpoint
        ORDER BY updated_at
        """
    ).fetchall()
    for row in rows:
        checkpoints.append(
            {
                "prompt_version": row["prompt_version"],
                "model_version": row["model_version"],
                "observation_fingerprint": row["observation_fingerprint"],
                "status": row["status"],
                "output_dir": row["output_dir"],
                "profile_written_at": row["profile_written_at"],
                "min_independent": row["min_independent"],
                "traits_stable": row["traits_stable"],
                "traits_provisional": row["traits_provisional"],
                "traits_contested": row["traits_contested"],
                "updated_at": row["updated_at"],
                "profile_current": row["status"] == "completed" and bool(row["profile_written_at"]),
            }
        )
    return {"tasks": tasks, "traits": traits, "checkpoints": checkpoints}


def format_personality_status(status: dict) -> str:
    lines = ["personality_tasks: " + str(status["tasks"]["total"])]
    for name, count in status["tasks"]["by_status"].items():
        if count:
            lines.append(f"  {name}: {count}")
    kinds = status["tasks"]["by_kind"]
    if any(kinds.values()):
        lines.append(
            "  kinds: "
            + ", ".join(f"{kind}={kinds[kind]}" for kind in TASK_KINDS if kinds[kind])
        )
    lines.append("personality_traits: " + str(status["traits"]["total"]))
    for name, count in status["traits"]["by_status"].items():
        if count:
            lines.append(f"  {name}: {count}")
    for dimension, count in status["traits"]["by_dimension"].items():
        if count:
            lines.append(f"  {dimension}: {count}")
    checkpoints = status.get("checkpoints") or []
    if not checkpoints:
        lines.append("personality_checkpoint: (none)")
    for checkpoint in checkpoints:
        lines.append(
            "personality_checkpoint: "
            f"{checkpoint['status']} "
            f"prompt={checkpoint['prompt_version']} "
            f"model={checkpoint['model_version']} "
            f"profile_current={checkpoint['profile_current']}"
        )
    return "\n".join(lines)


def _task_counts(conn) -> dict:
    by_status = {status: 0 for status in PIPELINE_STATUSES}
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS n
        FROM personality_tasks
        WHERE active = 1
        GROUP BY status
        """
    ).fetchall()
    for row in rows:
        by_status[row["status"]] = int(row["n"])
    by_kind = {kind: 0 for kind in TASK_KINDS}
    kind_rows = conn.execute(
        """
        SELECT task_kind, COUNT(*) AS n
        FROM personality_tasks
        WHERE active = 1
        GROUP BY task_kind
        """
    ).fetchall()
    for row in kind_rows:
        if row["task_kind"] in by_kind:
            by_kind[row["task_kind"]] = int(row["n"])
    return {"total": sum(by_status.values()), "by_status": by_status, "by_kind": by_kind}


def _trait_counts(conn) -> dict:
    by_status = {status: 0 for status in TRAIT_STATUSES}
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS n
        FROM l2_traits
        GROUP BY status
        """
    ).fetchall()
    for row in rows:
        if row["status"] in by_status:
            by_status[row["status"]] = int(row["n"])
    by_dimension: dict[str, int] = {}
    dimension_rows = conn.execute(
        """
        SELECT dimension, COUNT(*) AS n
        FROM l2_traits
        GROUP BY dimension
        ORDER BY dimension
        """
    ).fetchall()
    for row in dimension_rows:
        by_dimension[row["dimension"]] = int(row["n"])
    return {"total": sum(by_status.values()), "by_status": by_status, "by_dimension": by_dimension}
