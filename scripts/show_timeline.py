"""Print the long-term memory timeline.

    python -m scripts.show_timeline --db data/database/cleaned_messages.db

This reads memories already stored in SQLite. It does not call a model.
Historical and superseded rows stay in the list unless --status filters them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.database import init_db  # noqa: E402
from memory.timeline_builder import build_timeline, render_timeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print the local memory timeline. No model call.")
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument("--entity", default=None, help="Entity id, canonical name, or normalized name")
    parser.add_argument("--type", default=None, help="fact, event, relationship, opinion, or situational")
    parser.add_argument("--status", default=None, help="active, historical, superseded, or uncertain")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to print")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    database = Path(args.db).expanduser()
    if not database.is_file():
        print(f"找不到数据库 / database not found: {database}", file=sys.stderr)
        return 1
    conn = init_db(database)
    try:
        try:
            items = build_timeline(
                conn,
                entity=args.entity,
                memory_type=args.type,
                status=args.status,
                limit=args.limit,
            )
        except ValueError as exc:
            print(f"时间线参数错误 / timeline failed: {exc}", file=sys.stderr)
            return 1
    finally:
        conn.close()
    print(render_timeline(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
