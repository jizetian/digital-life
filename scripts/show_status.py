"""Show segmentation, L1 distillation, personality, memory, and retrieval counts.

    python -m scripts.show_status --db data/database/cleaned_messages.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from distillation.status import format_pipeline_status, pipeline_status  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print local segmentation, distillation, personality, memory, and retrieval status."
    )
    parser.add_argument("--db", required=True, help="SQLite database path")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        status = pipeline_status(args.db)
    except FileNotFoundError as exc:
        print(f"找不到数据库 / database not found: {exc}", file=sys.stderr)
        return 1
    print(format_pipeline_status(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
