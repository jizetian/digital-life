"""Talk to the local digital twin.

    python -m scripts.chat --db data/database/cleaned_messages.db
    python -m scripts.chat --db data/database/cleaned_messages.db --message "你后来搬去哪了"

The reply is one JSON object from the model, then a normal chat line here.
Retrieval uses the Phase 5 index. Without embeddings, full text still runs.
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

from chat.agent import ChatModelError, ChatNotFound, chat_turn  # noqa: E402
from common.logger import configure_logging  # noqa: E402
from distillation.llm import LLMError  # noqa: E402
from retrieval.errors import RetrievalError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with the local digital twin.")
    parser.add_argument("--db", required=True, help="SQLite database")
    parser.add_argument("--message", default=None, help="One turn, then exit. Omit this for a prompt.")
    parser.add_argument("--session", default=None, help="Continue this session id")
    parser.add_argument("--target-name", default=None, help="Twin display name. Default: 不吃香菜")
    parser.add_argument("--personality-dir", default=None, help="L3 profile directory")
    parser.add_argument("--limit", type=int, default=None, help="How many retrieval hits to consider")
    parser.add_argument("--max-chars", type=int, default=None, help="Prompt budget for personality, evidence, and history")
    parser.add_argument("--no-vector", action="store_true", help="Skip the dense channel")
    parser.add_argument("--no-fts", action="store_true", help="Skip the SQLite FTS5 channel")
    parser.add_argument("--json", action="store_true", help="Print each turn as JSON")
    return parser


def format_turn(result: dict) -> str:
    name = result.get("target_name") or "不吃香菜"
    lines = [f"{name}> {result['reply']}"]
    lines.append(f"置信度: {result['confidence']}")
    if result.get("retrieval_weak"):
        lines.append("检索: 证据不足")
    evidence = result.get("evidence_message_ids") or []
    lines.append("证据消息: " + (", ".join(evidence) if evidence else "(none)"))
    memories = result.get("retrieved_memories") or []
    if memories:
        shown = []
        for item in memories:
            mark = "cited" if item.get("cited") else "retrieved"
            shown.append(f"{item['memory_id']}({mark})")
        lines.append("记忆: " + ", ".join(shown))
    if result.get("rejected_ids"):
        lines.append("已丢弃编号: " + ", ".join(result["rejected_ids"]))
    if result.get("uncertainty"):
        lines.append("不确定: " + result["uncertainty"])
    lines.append(f"session: {result['session_id']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    session_id = args.session
    if args.message is not None:
        return _one(args, args.message, session_id)
    print("和「不吃香菜」的 AI 模拟说话。空行忽略，输入 /quit 退出。")
    while True:
        try:
            line = input("你> ")
        except EOFError:
            print()
            return 0
        if line.strip() in {"/quit", "/exit", "退出"}:
            return 0
        if not line.strip():
            continue
        code = _one(args, line, session_id)
        if code != 0:
            return code
        session_id = _LAST_SESSION or session_id


_LAST_SESSION: str | None = None


def _one(args, message: str, session_id: str | None) -> int:
    global _LAST_SESSION
    try:
        result = chat_turn(
            args.db,
            message,
            session_id=session_id,
            target_name=args.target_name,
            personality_dir=args.personality_dir,
            use_vector=not args.no_vector,
            use_fts=not args.no_fts,
            limit=args.limit,
            max_chars=args.max_chars,
        )
    except FileNotFoundError as exc:
        print(f"找不到数据库 / database not found: {exc}", file=sys.stderr)
        return 1
    except ChatNotFound as exc:
        print(f"没有这个会话 / session not found: {exc}", file=sys.stderr)
        return 1
    except (ChatModelError, LLMError, RetrievalError, ValueError, OSError) as exc:
        print(f"对话失败 / chat failed: {exc}", file=sys.stderr)
        return 1
    _LAST_SESSION = result["session_id"]
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_turn(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
