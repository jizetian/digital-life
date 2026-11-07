"""Run L1 distillation for segmented conversations.

    python -m scripts.distill_l1 --db data/database/cleaned_messages.db --batch-size 8

Completed tasks are skipped. Pending, retrying, and interrupted running tasks
are continued. Re-run the same command to resume.
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
from distillation.engine import DistillationError, run_distillation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract L1 observations for local conversation segments. No chat export is uploaded."
    )
    parser.add_argument("--db", required=True, help="SQLite database that already has conversation_segments")
    parser.add_argument("--batch-size", type=int, default=None, help="Segments claimed per round (default 8)")
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Stop after this many attempts and leave the rest for a later resume",
    )
    parser.add_argument("--max-attempts", type=int, default=None, help="Attempts before a task is marked failed (default 3)")
    parser.add_argument("--retry-failed", action="store_true", help="Reset failed tasks and try them again")
    parser.add_argument("--model", default=None, help="Model name sent to the OpenAI-compatible endpoint")
    parser.add_argument(
        "--model-version",
        default=None,
        help="Cache key mixed into task_id. Defaults to the model name.",
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, for example http://127.0.0.1:11434/v1")
    parser.add_argument("--prompt-version", default=None, help="Override the L1 prompt version used in task_id")
    return parser


def format_distillation(result: dict) -> str:
    lines = [
        "=== ChatTwin L1 蒸馏 / distillation ===",
        f"status: {result['status']}",
        f"prompt_version: {result['prompt_version']}",
        f"model: {result['model']}",
        f"model_version: {result['model_version']}",
        f"batch_size: {result['batch_size']}",
        f"attempts_completed: {result['attempts_completed']}",
        f"attempts_failed: {result['attempts_failed']}",
        f"tasks_remaining: {result['tasks_remaining']}",
        f"observations: {result['observations']}",
        "tasks:",
    ]
    for status, count in sorted(result["tasks"].items()):
        lines.append(f"  {status}: {count}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        result = run_distillation(
            args.db,
            batch_size=args.batch_size,
            max_tasks=args.max_tasks,
            max_attempts=args.max_attempts,
            retry_failed=args.retry_failed,
            model=args.model,
            model_version=args.model_version,
            base_url=args.base_url,
            prompt_version=args.prompt_version,
        )
    except FileNotFoundError as exc:
        print(f"找不到数据库 / database not found: {exc}", file=sys.stderr)
        return 1
    except (DistillationError, ValueError, OSError) as exc:
        print(f"蒸馏失败 / distillation failed: {exc}", file=sys.stderr)
        return 1
    print(format_distillation(result))
    if result["attempts_completed"] == 0 and result["attempts_failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
