"""Counts for the retrieval projection. Vectors live outside SQLite."""

from __future__ import annotations

from retrieval.collections import COLLECTIONS


def retrieval_snapshot_conn(conn) -> dict:
    counts = {name: 0 for name in COLLECTIONS}
    for row in conn.execute(
        "SELECT collection, COUNT(*) AS n FROM retrieval_documents GROUP BY collection"
    ):
        if row["collection"] in counts:
            counts[row["collection"]] = int(row["n"])
    embedded = int(conn.execute("SELECT COUNT(*) AS n FROM retrieval_embeddings").fetchone()["n"])
    models = [
        row["model_name"]
        for row in conn.execute(
            "SELECT DISTINCT model_name FROM retrieval_embeddings ORDER BY model_name"
        )
    ]
    return {
        "documents": sum(counts.values()),
        "embedded": embedded,
        "by_collection": counts,
        "models": models,
    }


def format_retrieval_status(status: dict) -> str:
    lines = [f"retrieval: documents {status['documents']} embedded {status['embedded']}"]
    for name, count in status["by_collection"].items():
        if count:
            lines.append(f"  {name}: {count}")
    if status["models"]:
        lines.append("  model: " + ", ".join(status["models"]))
    return "\n".join(lines)
