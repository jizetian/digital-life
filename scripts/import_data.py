"""Import a local Moect JSON export into SQLite.

    python -m scripts.import_data --input path/to/export.json --db data/database/cleaned_messages.db
    python -m scripts.import_data --input path/to/export.json --db data/database/cleaned_messages.db --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.checkpoint import ResumeConflictError  # noqa: E402
from common.logger import configure_logging  # noqa: E402
from importer.import_service import format_report, run_import  # noqa: E402
from importer.schema_detector import SchemaError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import a local Moect WeChat export into SQLite. The JSON file is not copied into git."
    )
    parser.add_argument("--input", required=True, help="Path to a Moect JSON export on this machine")
    parser.add_argument("--db", required=True, help="SQLite database path, for example data/database/cleaned_messages.db")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the newest unfinished job for this file",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Commit every N messages (default from settings)")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        job = run_import(
            args.input,
            args.db,
            resume=args.resume,
            batch_size=args.batch_size,
        )
    except FileNotFoundError as exc:
        print(f"找不到导出文件 / export file not found: {exc}", file=sys.stderr)
        return 1
    except (SchemaError, ResumeConflictError, OSError, UnicodeError, ValueError) as exc:
        print(f"导入失败 / import failed: {exc}", file=sys.stderr)
        return 1
    print(format_report(job))
    if job.get("status") != "completed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
