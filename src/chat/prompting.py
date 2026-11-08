"""Load the twin system prompt. The version lives in this module, not in the database."""

from __future__ import annotations

from pathlib import Path

PROMPT_VERSION = "twin-v1"
PROMPT_FILENAME = "twin_chat_v1.md"

_CACHE: str | None = None


def prompt_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "prompts"


def load_prompt() -> str:
    global _CACHE
    if _CACHE is None:
        path = prompt_dir() / PROMPT_FILENAME
        text = path.read_text(encoding="utf-8")
        if not text.startswith(f"prompt_version: {PROMPT_VERSION}"):
            raise ValueError(f"{PROMPT_FILENAME} must start with prompt_version: {PROMPT_VERSION}")
        for token in ("cited_memory_ids", "cited_message_ids", "AI simulation", "{target_name}"):
            if token not in text:
                raise ValueError(f"{PROMPT_FILENAME} is missing {token}")
        _CACHE = text
    return _CACHE


def render_instructions(target_name: str) -> str:
    name = target_name.strip() or "不吃香菜"
    return load_prompt().replace("{target_name}", name)
