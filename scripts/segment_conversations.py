"""Split messages already stored in SQLite into conversation segments.

    python -m scripts.segment_conversations --db data/database/cleaned_messages.db
    python -m scripts.segment_conversations --db data/database/cleaned_messages.db --rebuild
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
from segmentation.service import SegmentationConflictError, run_segmentation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment a local ChatTwin SQLite database. Message text stays in the messages table."
    )
    parser.add_argument("--db", required=True, help="SQLite database produced by scripts.import_data")
    parser.add_argument("--gap-minutes", type=float, default=None, help="Start a segment after this gap (default 30)")
    parser.add_argument("--max-messages", type=int, default=None, help="Maximum messages stored on one segment (default 50)")
    parser.add_argument("--overlap", type=int, default=None, help="Messages copied onto the next size-split segment (default 4)")
    parser.add_argument(
        "--overlap-on-time-gap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also copy the overlap onto a segment that starts because of a time gap",
    )
    parser.add_argument("--page-size", type=int, default=None, help="Messages read per batch (default 2000)")
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Stop after this many newly closed segments and leave a resumable checkpoint",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing segments and distillation rows, then segment from the start",
    )
    return parser


def format_segmentation(result: dict) -> str:
    params = result["params"]
    return "\n".join(
        [
            "=== ChatTwin 切分 / segmentation ===",
            f"status: {result['status']}",
            f"unchanged: {result['unchanged']}",
            f"resumed: {result['resumed']}",
            f"messages: {result['messages_consumed']}/{result['message_total']}",
            f"segments_written: {result['segments_written']}",
            f"segment_count: {result['segment_count']}",
            f"gap_seconds: {params['gap_seconds']}",
            f"max_messages: {params['max_messages']}",
            f"overlap_messages: {params['overlap_messages']}",
            f"overlap_on_time_gap: {params['overlap_on_time_gap']}",
            f"params_hash: {result['params_hash']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        result = run_segmentation(
            args.db,
            gap_minutes=args.gap_minutes,
            max_messages=args.max_messages,
            overlap_messages=args.overlap,
            overlap_on_time_gap=args.overlap_on_time_gap,
            rebuild=args.rebuild,
            max_segments=args.max_segments,
            page_size=args.page_size,
        )
    except FileNotFoundError as exc:
        print(f"找不到数据库 / database not found: {exc}", file=sys.stderr)
        return 1
    except (SegmentationConflictError, ValueError, OSError) as exc:
        print(f"切分失败 / segmentation failed: {exc}", file=sys.stderr)
        return 1
    print(format_segmentation(result))
    if result["status"] not in {"completed", "running"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
