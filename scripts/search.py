"""Hybrid search over the local retrieval index.

    python -m scripts.search --db data/database/cleaned_messages.db --query "搬到了上海"

Dense vectors and SQLite FTS5 are merged and deduped. This does not start a
chat agent. Run ``build_embeddings`` first so the projection and vectors exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.logger import configure_logging  # noqa: E402
from retrieval.errors import RetrievalError  # noqa: E402
from retrieval.service import run_search  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search segments, memories, and personality traits.")
    parser.add_argument("--db", required=True, help="SQLite database")
    parser.add_argument("--query", required=True, help="Search text")
    parser.add_argument("--collection", action="append", default=None, help="Limit to one collection. Repeatable.")
    parser.add_argument("--limit", type=int, default=None, help="How many hits to print (1-100)")
    parser.add_argument("--no-vector", action="store_true", help="Skip the dense channel")
    parser.add_argument("--no-fts", action="store_true", help="Skip the SQLite FTS5 channel")
    parser.add_argument("--rerank", action="store_true", help="Call the configured rerank URL after fusion")
    parser.add_argument("--memory-status", default=None, help="active, historical, superseded, or uncertain")
    parser.add_argument("--memory-type", default=None, help="fact, event, relationship, opinion, or situational")
    parser.add_argument("--trait-status", default=None, help="stable, provisional, or contested")
    parser.add_argument("--dimension", default=None, help="Personality dimension, for example decision")
    parser.add_argument("--provider", default=None, help="openai, hash, or sentence_transformers")
    parser.add_argument("--model", default=None, help="Embedding model name. Must match the index.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL that serves /embeddings")
    parser.add_argument("--backend", default=None, help="qdrant or memory")
    parser.add_argument("--qdrant-path", default=None, help="Local Qdrant directory")
    parser.add_argument("--qdrant-url", default=None, help="Remote Qdrant URL")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON")
    return parser


def format_search(result: dict) -> str:
    lines = [
        "=== ChatTwin 检索 / search ===",
        f"model: {result['model']}",
        f"hits: {len(result['hits'])}",
        f"indexed_documents: {result['indexed_documents']}",
        f"vector: {result['used_vector']}",
        f"fts: {result['used_fts']}",
        f"reranker: {result['reranker']}",
    ]
    if result.get("vector_note"):
        lines.append(f"vector_note: {result['vector_note']}")
    for index, hit in enumerate(result["hits"], start=1):
        lines.append(
            f"{index}. score={hit['score']:.4f} collection={hit['collection']} "
            f"source={hit['source_id']} vector_rank={hit['vector_rank']} fts_rank={hit['fts_rank']}"
        )
        lines.append("   " + _snippet(hit["text"]))
        evidence = hit["metadata"].get("evidence_message_ids") or hit["metadata"].get("message_ids")
        if evidence:
            shown = ", ".join(str(item) for item in evidence[:8])
            lines.append(f"   evidence: {shown}")
    return "\n".join(lines)


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= 240:
        return flat
    return flat[:240] + "..."


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        result = run_search(
            args.db,
            args.query,
            collections=args.collection,
            limit=args.limit,
            use_vector=not args.no_vector,
            use_fts=not args.no_fts,
            rerank=args.rerank,
            memory_status=args.memory_status,
            memory_type=args.memory_type,
            trait_status=args.trait_status,
            dimension=args.dimension,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            vector_backend=args.backend,
            qdrant_path=args.qdrant_path,
            qdrant_url=args.qdrant_url,
        )
    except FileNotFoundError as exc:
        print(f"找不到数据库 / database not found: {exc}", file=sys.stderr)
        return 1
    except (RetrievalError, ValueError, OSError) as exc:
        print(f"检索失败 / search failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_search(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
