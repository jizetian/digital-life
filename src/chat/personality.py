"""Stable voice for the twin: L2 traits, then the L3 profile if it adds anything.

Provisional traits stay out of this block. They do not yet have enough
independent evidence to speak as the person. Retrieval can still surface a
provisional trait as a hit when the question asks for it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PersonalityView:
    source: str
    text: str
    trait_ids: list[str]


_NONE = "尚无稳定人格档案。回答时不要假装已经掌握稳定的说话习惯。"


def load_personality(
    conn: sqlite3.Connection,
    *,
    output_dir: Path | None,
    max_chars: int,
) -> PersonalityView:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    traits = _stable_traits(conn)
    l2_lines = [_trait_line(row) for row in traits]
    l2_text = "\n".join(l2_lines).strip()
    l3_text = _l3_stable(output_dir)
    extra = _extra_lines(l3_text, l2_text)
    if not extra:
        l3_text = ""
    parts: list[str] = []
    if l2_text:
        parts.append(l2_text)
    if extra:
        parts.append("\n".join(extra))
    if l2_text and extra:
        source = "l2+l3"
    elif l2_text:
        source = "l2"
    elif extra or l3_text:
        source = "l3"
        if not parts and l3_text:
            parts.append(l3_text)
    else:
        source = "none"
        parts.append(_NONE)
    text = "\n\n".join(part for part in parts if part).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return PersonalityView(
        source=source,
        text=text,
        trait_ids=[row["trait_id"] for row in traits],
    )


def _stable_traits(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    version = _active_version(conn)
    if version is None:
        return []
    prompt_version, model_version = version
    return conn.execute(
        """
        SELECT trait_id, dimension, statement, situational_scope, confidence
        FROM l2_traits
        WHERE prompt_version = ? AND model_version = ? AND status = 'stable'
        ORDER BY dimension, confidence DESC, statement
        """,
        (prompt_version, model_version),
    ).fetchall()


def _active_version(conn: sqlite3.Connection) -> tuple[str, str] | None:
    checkpoint = conn.execute(
        """
        SELECT prompt_version, model_version
        FROM personality_checkpoint
        WHERE profile_written_at IS NOT NULL
        ORDER BY updated_at DESC
        LIMIT 1
        """
    ).fetchone()
    if checkpoint is not None:
        return checkpoint["prompt_version"], checkpoint["model_version"]
    row = conn.execute(
        """
        SELECT prompt_version, model_version
        FROM l2_traits
        GROUP BY prompt_version, model_version
        ORDER BY MAX(updated_at) DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    return row["prompt_version"], row["model_version"]


def _trait_line(row) -> str:
    statement = " ".join((row["statement"] or "").split())
    scope = " ".join((row["situational_scope"] or "").split())
    scope_bit = f" 情境：{scope}" if scope else ""
    confidence = row["confidence"]
    confidence_bit = "" if confidence is None else f", confidence={confidence}"
    return (
        f"- [{row['dimension']}] {statement}{scope_bit} "
        f"(trait_id={row['trait_id']}{confidence_bit})"
    )


def _l3_stable(output_dir: Path | None) -> str:
    if output_dir is None:
        return ""
    path = output_dir / "personality_core.md"
    if not path.is_file():
        return ""
    return _section(path.read_text(encoding="utf-8"), ("## Stable traits", "## 稳定特质"))


def _section(markdown: str, headings: tuple[str, ...]) -> str:
    lines = markdown.splitlines()
    start = None
    wanted = {heading.lower() for heading in headings}
    for index, line in enumerate(lines):
        if line.strip().lower() in wanted:
            start = index + 1
            break
    if start is None:
        return ""
    kept: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def _extra_lines(l3_text: str, l2_text: str) -> list[str]:
    if not l3_text:
        return []
    extra: list[str] = []
    for line in l3_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped in {"None.", "None"}:
            continue
        if stripped in l2_text:
            continue
        extra.append(line)
    return extra
