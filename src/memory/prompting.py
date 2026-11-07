"""Load the memory prompts and render one segment without storing the chat."""

from __future__ import annotations

from pathlib import Path

from memory.models import CandidateMemory
from memory.types import ENTITY_STATUSES, MEMORY_TYPES, MERGE_ACTIONS

PROMPT_VERSION = "memory-v1"
EXTRACT_FILENAME = "memory_extract_v1.md"
MERGE_FILENAME = "memory_merge_v1.md"

_CACHE: dict[str, str] = {}


def prompt_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "prompts"


def load_extract_prompt() -> str:
    return _load(EXTRACT_FILENAME, required=(*MEMORY_TYPES, *ENTITY_STATUSES, "evidence_message_ids"))


def load_merge_prompt() -> str:
    return _load(MERGE_FILENAME, required=MERGE_ACTIONS)


def render_extract_prompt(
    *,
    segment_id: str,
    conversation_id: str | None,
    messages: list[dict],
    observations: list[dict],
    max_chars: int,
) -> str:
    lines = [
        "task: extract",
        "Extract long-term memories from this conversation segment and its L1 observations.",
        f"segment_id: {segment_id}",
        f"conversation_id: {conversation_id or ''}",
        "messages:",
    ]
    for item in messages:
        lines.append(
            f"- message_id={item['message_id']} time={_stamp(item.get('timestamp'))} "
            f"role={item.get('sender_role') or ''} type={item.get('message_type') or ''} "
            f"overlap={'true' if item.get('is_overlap') else 'false'} "
            f"text={_one_line(item.get('content') or '', max_chars)}"
        )
    lines.append("observations:")
    if not observations:
        lines.append("- (none)")
    for item in observations:
        evidence = ",".join(item.get("evidence_message_ids") or [])
        lines.append(
            f"- observation_id={item['observation_id']} category={item.get('category') or ''} "
            f"confidence={_num(item.get('confidence'))} "
            f"time={_stamp(item.get('time_start'))}..{_stamp(item.get('time_end'))} "
            f"evidence={evidence} summary={_one_line(item.get('summary') or '', max_chars)}"
        )
    lines.append("Respond with JSON only.")
    return "\n".join(lines)


def render_merge_prompt(
    *,
    segment_id: str,
    existing: list[dict],
    candidates: list[CandidateMemory],
    max_chars: int,
) -> str:
    lines = [
        "task: merge",
        "Decide how each candidate memory relates to the existing store. Do not delete old memories.",
        f"segment_id: {segment_id}",
        "existing:",
    ]
    if not existing:
        lines.append("- (none)")
    for item in existing:
        situation = item.get("situation") or ""
        situation_bit = f" situation={_one_line(situation, 120)}" if situation else ""
        lines.append(
            f"- memory_id={item['memory_id']} type={item['memory_type']} status={item['status']} "
            f"time={_stamp(item.get('time_start'))}..{_stamp(item.get('time_end'))} "
            f"importance={_num(item.get('importance'))} confidence={_num(item.get('confidence'))} "
            f"entities={item.get('entities') or ''}{situation_bit} "
            f"statement={_one_line(item.get('statement') or '', max_chars)}"
        )
    lines.append("candidates:")
    for index, candidate in enumerate(candidates):
        entities = "|".join(
            f"{_safe(item.name)}:{item.role}:{item.status}:{item.entity_type}" for item in candidate.entities
        )
        situation_bit = f" situation={_one_line(candidate.situation, 120)}" if candidate.situation else ""
        lines.append(
            f"- candidate={index} type={candidate.memory_type} "
            f"importance={_num(candidate.importance)} confidence={_num(candidate.confidence)} "
            f"time={_stamp(candidate.time_start)}..{_stamp(candidate.time_end)} "
            f"precision={candidate.time_precision} entities={entities} "
            f"evidence={','.join(candidate.evidence_message_ids)} "
            f"observations={','.join(candidate.observation_ids)}"
            f"{situation_bit} statement={_one_line(candidate.statement, max_chars)}"
        )
    lines.append("Respond with JSON only.")
    return "\n".join(lines)


def _load(filename: str, *, required: tuple[str, ...]) -> str:
    cached = _CACHE.get(filename)
    if cached is not None:
        return cached
    path = prompt_dir() / filename
    text = path.read_text(encoding="utf-8")
    first = text.splitlines()[0].strip() if text.strip() else ""
    expected = f"prompt_version: {PROMPT_VERSION}"
    if first != expected:
        raise RuntimeError(f"{path.name} must start with {expected!r}")
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"{path.name} is missing: {', '.join(missing)}")
    _CACHE[filename] = text
    return text


def _one_line(text: str, max_chars: int) -> str:
    cleaned = " ".join(str(text).replace("\r", " ").replace("\n", " ").split())
    if max_chars > 0 and len(cleaned) > max_chars:
        return cleaned[:max_chars] + "…"
    return cleaned


def _safe(name: str) -> str:
    return _one_line(name, 80).replace("|", "/").replace(":", "/")


def _stamp(value) -> str:
    return "" if value is None else str(value)


def _num(value) -> str:
    return "" if value is None else str(value)
