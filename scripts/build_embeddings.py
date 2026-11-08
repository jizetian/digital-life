"""Build or refresh the hybrid retrieval index.

    python -m scripts.build_embeddings --db data/database/cleaned_messages.db

SQLite is the source of truth. Qdrant only stores vectors and can be rebuilt
with ``--rebuild``. The default embedding model is BGE-M3 through an
OpenAI-compatible endpoint. ``--provider hash`` is a local stand-in for tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.logger import configure_logging  # noqa: E402
from retrieval.errors import RetrievalError  # noqa: E402
from retrieval.service import run_index  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed conversation segments, long-term memories, and personality traits."
    )
    parser.add_argument("--db", required=True, help="SQLite database that already holds the source rows")
    parser.add_argument("--rebuild", action="store_true", help="Drop vector collections and embed every document again")
    parser.add_argument(
        "--collection",
        action="append",
        default=None,
        help="Limit the sync to one collection. Repeat for several. Default is all three.",
    )
    parser.add_argument("--provider", default=None, help="openai, hash, or sentence_transformers")
    parser.add_argument("--model", default=None, help="Embedding model name. Default is bge-m3.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL that serves /embeddings")
    parser.add_argument("--batch-size", type=int, default=None, help="Texts per embedding request")
    parser.add_argument("--backend", default=None, help="qdrant or memory. memory is process-local and not saved.")
    parser.add_argument("--qdrant-path", default=None, help="Local Qdrant directory. Default is data/qdrant.")
    parser.add_argument("--qdrant-url", default=None, help="Remote Qdrant URL. Overrides --qdrant-path.")
    return parser


def format_index(result: dict) -> str:
    documents = result["documents"]
    lines = [
        "=== ChatTwin 检索索引 / retrieval index ===",
        f"status: {result['status']}",
        f"provider: {result['provider']}",
        f"model: {result['model']}",
        f"dimension: {result['dimension']}",
        f"rebuilt_vectors: {result['rebuilt_vectors']}",
        f"documents: {documents['total']}",
        f"upserted: {documents['upserted']}",
        f"unchanged: {documents['unchanged']}",
        f"deleted: {documents['deleted']}",
        f"embedded: {result['embedded']}",
        f"skipped: {result['skipped']}",
        f"vector_deleted: {result['vector_deleted']}",
        "collections:",
    ]
    for name, count in documents["by_collection"].items():
        lines.append(f"  {name}: {count}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        result = run_index(
            args.db,
            collections=args.collection,
            rebuild=args.rebuild,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            batch_size=args.batch_size,
            vector_backend=args.backend,
            qdrant_path=args.qdrant_path,
            qdrant_url=args.qdrant_url,
        )
    except FileNotFoundError as exc:
        print(f"找不到数据库 / database not found: {exc}", file=sys.stderr)
        return 1
    except (RetrievalError, ValueError, OSError) as exc:
        print(f"索引失败 / index failed: {exc}", file=sys.stderr)
        return 1
    print(format_index(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
