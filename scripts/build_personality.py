"""Merge L1 observations into an L2/L3 personality profile.

    python -m scripts.build_personality \
      --db data/database/cleaned_messages.db \
      --output personality

Completed tasks are skipped. Pending, retrying, and interrupted running tasks
are continued. Re-run the same command to resume. The profile files are written
only when every task for this prompt and model has finished.
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
from personality.engine import PersonalityError, run_personality  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a local personality profile from L1 observations. Raw chat stays in SQLite."
    )
    parser.add_argument("--db", required=True, help="SQLite database that already has l1_observations")
    parser.add_argument("--output", default=None, help="Directory for the profile files (default: personality/)")
    parser.add_argument("--batch-size", type=int, default=None, help="L1 observations per merge call (default 20)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Stop after this many model calls and leave a checkpoint")
    parser.add_argument("--max-attempts", type=int, default=None, help="Attempts before a task is marked failed (default 3)")
    parser.add_argument("--min-independent", type=int, default=None, help="Independent evidence groups required for a stable trait (default 2)")
    parser.add_argument("--retry-failed", action="store_true", help="Reset failed tasks and try them again")
    parser.add_argument("--model", default=None, help="Model name sent to the OpenAI-compatible endpoint")
    parser.add_argument("--model-version", default=None, help="Cache key mixed into task_id. Defaults to the model name.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, for example http://127.0.0.1:11434/v1")
    parser.add_argument("--prompt-version", default=None, help="Override the L2 prompt version used in task_id")
    return parser


def format_personality(result: dict) -> str:
    lines = [
        "=== ChatTwin 人格 / personality ===",
        f"status: {result['status']}",
        f"prompt_version: {result['prompt_version']}",
        f"model: {result['model']}",
        f"model_version: {result['model_version']}",
        f"batch_size: {result['batch_size']}",
        f"min_independent: {result['min_independent']}",
        f"attempts_completed: {result['attempts_completed']}",
        f"attempts_failed: {result['attempts_failed']}",
        f"tasks_remaining: {result['tasks_remaining']}",
        f"traits_stable: {result['traits']['stable']}",
        f"traits_provisional: {result['traits']['provisional']}",
        f"traits_contested: {result['traits']['contested']}",
        f"output_dir: {result['output_dir']}",
        "tasks:",
    ]
    for status, count in sorted(result["tasks"].items()):
        lines.append(f"  {status}: {count}")
    if result["files"]:
        lines.append("files:")
        for name in result["files"]:
            lines.append(f"  {name}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        result = run_personality(
            args.db,
            output_dir=args.output,
            batch_size=args.batch_size,
            max_tasks=args.max_tasks,
            max_attempts=args.max_attempts,
            min_independent=args.min_independent,
            retry_failed=args.retry_failed,
            model=args.model,
            model_version=args.model_version,
            base_url=args.base_url,
            prompt_version=args.prompt_version,
        )
    except FileNotFoundError as exc:
        print(f"找不到数据库 / database not found: {exc}", file=sys.stderr)
        return 1
    except (PersonalityError, ValueError, OSError) as exc:
        print(f"人格蒸馏失败 / personality failed: {exc}", file=sys.stderr)
        return 1
    print(format_personality(result))
    if result["status"] == "failed":
        return 1
    if result["attempts_completed"] == 0 and result["attempts_failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
