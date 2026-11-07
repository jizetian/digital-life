"""Load the L2 prompts and render observation batches without raw chat text."""

from __future__ import annotations

from pathlib import Path

from personality.dimensions import DIMENSIONS
from personality.models import ObsView

PROMPT_VERSION = "l2-v1"

_MERGE_FILE = "l2_merge_v1.md"
_REDUCE_FILE = "l2_reduce_v1.md"
_CONTRADICT_FILE = "l2_contradict_v1.md"
_CACHE: dict[str, str] = {}

RELATIONS = ("supports", "contradicts", "evolution", "situational", "unrelated")


def prompt_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "prompts"


def load_merge_prompt() -> str:
    return _load(
        _MERGE_FILE,
        required=(*DIMENSIONS, "observation_ids", "situational_scope", "stages"),
    )


def load_reduce_prompt() -> str:
    return _load(_REDUCE_FILE, required=("trait_keys", "situational_scope", "stages"))


def load_contradict_prompt() -> str:
    return _load(_CONTRADICT_FILE, required=RELATIONS)


def render_merge_prompt(
    *,
    dimension: str,
    observations: list[ObsView],
    max_chars: int,
) -> str:
    lines = [
        "Merge these L1 observations into candidate traits for one dimension.",
        f"dimension: {dimension}",
        "observations:",
    ]
    lines.extend(_observation_lines(observations, max_chars=max_chars))
    lines.append("Respond with JSON only.")
    return "\n".join(lines)


def render_reduce_prompt(
    *,
    dimension: str,
    local_traits: list[tuple[str, dict]],
) -> str:
    lines = [
        "Merge candidate traits from separate batches of one dimension.",
        f"dimension: {dimension}",
        "traits:",
    ]
    for key, trait in local_traits:
        ids = ",".join(trait.get("observation_ids") or [])
        scope = _one_line(trait.get("situational_scope") or "general")
        statement = _one_line(trait.get("statement") or "")
        lines.append(
            f"- trait_key={key} scope={scope} observation_ids={ids} statement={statement}"
        )
    lines.append("Respond with JSON only.")
    return "\n".join(lines)


def render_contradict_prompt(
    *,
    dimension: str,
    traits: list[tuple[str, dict]],
    observations: list[ObsView],
    max_chars: int,
) -> str:
    lines = [
        "Judge observations against candidate traits in one dimension.",
        f"dimension: {dimension}",
        "traits:",
    ]
    for key, trait in traits:
        ids = ",".join(trait.get("observation_ids") or [])
        scope = _one_line(trait.get("situational_scope") or "general")
        statement = _one_line(trait.get("statement") or "")
        lines.append(
            f"- trait_key={key} scope={scope} evidence_ids={ids} statement={statement}"
        )
    lines.append("observations:")
    lines.extend(_observation_lines(observations, max_chars=max_chars))
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


def _observation_lines(observations: list[ObsView], *, max_chars: int) -> list[str]:
    lines = []
    for item in observations:
        summary = _one_line(item.summary)
        if max_chars > 0 and len(summary) > max_chars:
            summary = summary[:max_chars] + "…"
        start = "" if item.time_start is None else item.time_start
        end = "" if item.time_end is None else item.time_end
        confidence = "" if item.confidence is None else item.confidence
        lines.append(
            f"- observation_id={item.observation_id} segment_id={item.segment_id} "
            f"category={item.category} time_start={start} time_end={end} "
            f"confidence={confidence} summary={summary}"
        )
    return lines


def _one_line(text: str) -> str:
    return str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ").strip()
