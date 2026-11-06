"""Load the L1 prompt and render one segment as a message-id transcript."""

from __future__ import annotations

from pathlib import Path

from distillation.categories import L1_CATEGORIES

PROMPT_VERSION = "l1-v1"
PROMPT_FILENAME = "l1_observation_v1.md"

_CACHED: str | None = None


def prompt_path() -> Path:
    return Path(__file__).resolve().parents[2] / "prompts" / PROMPT_FILENAME


def load_system_prompt() -> str:
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    path = prompt_path()
    text = path.read_text(encoding="utf-8")
    first = text.splitlines()[0].strip() if text.strip() else ""
    expected = f"prompt_version: {PROMPT_VERSION}"
    if first != expected:
        raise RuntimeError(f"{path.name} must start with {expected!r}")
    missing = [category for category in L1_CATEGORIES if category not in text]
    if missing:
        raise RuntimeError(f"{path.name} is missing categories: {', '.join(missing)}")
    _CACHED = text
    return text


def render_user_prompt(
    *,
    segment_id: str,
    conversation_id: str | None,
    messages: list[dict],
    max_chars: int,
) -> str:
    """One line per message so each observation can cite a message id."""
    lines = [
        "Extract L1 observations from this conversation segment.",
        f"segment_id: {segment_id}",
        f"conversation_id: {conversation_id or ''}",
        "messages:",
    ]
    for item in messages:
        text = item.get("content") or ""
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars] + "…"
        timestamp = "" if item.get("timestamp") is None else item["timestamp"]
        role = item.get("sender_role") or ""
        kind = item.get("message_type") or ""
        overlap = "true" if item.get("is_overlap") else "false"
        lines.append(
            f"- id={item['message_id']} time={timestamp} role={role} "
            f"type={kind} overlap={overlap} text={text}"
        )
    lines.append("Respond with JSON only.")
    return "\n".join(lines)
