"""Assemble a bounded prompt from personality, retrieval hits, and recent turns.

``max_chars`` covers the personality block, the retrieved evidence, and the
recent turns. The fixed instruction file is not part of that budget. The
current user message is capped separately so a long question cannot push the
evidence out of the prompt.

Evidence-backed memories are packed before traits, segments, and memories
that have no message ids. A memory with a higher score still waits behind a
lower-scored memory that cites a message.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chat.personality import PersonalityView
from chat.prompting import render_instructions

_KIND = {
    "long_term_memories": "memory",
    "personality_observations": "trait",
    "conversation_segments": "segment",
}
_MIN_BLOCK = 48


@dataclass(slots=True)
class BuiltContext:
    system: str
    user: str
    personality_source: str
    retrieval_weak: bool
    truncated: bool
    char_count: int
    budgeted_text: str
    uncertainty_note: str | None
    retrieved_memories: list[dict] = field(default_factory=list)
    context_hits: list[dict] = field(default_factory=list)
    allowed_memory_ids: list[str] = field(default_factory=list)
    allowed_message_ids: list[str] = field(default_factory=list)
    memory_evidence: dict[str, list[str]] = field(default_factory=dict)


def build_context(
    *,
    target_name: str,
    personality: PersonalityView,
    hits: list,
    history: list[dict],
    user_message: str,
    max_chars: int,
    personality_chars: int,
    history_chars: int,
    history_turns: int,
    user_chars: int,
) -> BuiltContext:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if personality_chars < 1 or history_chars < 1 or user_chars < 1 or history_turns < 1:
        raise ValueError("chat budgets must be positive")
    name = target_name.strip() or "不吃香菜"
    question, question_clipped = _clip(user_message.strip(), user_chars)
    turns = list(history or [])[-history_turns:]
    reserved = min(history_chars, max_chars // 3)
    if reserved < 1:
        reserved = 1
    if reserved >= max_chars:
        reserved = max(max_chars // 3, 1)
    body_budget = max_chars - reserved
    header_room = 220
    personality_room = min(personality_chars, max(1, body_budget - header_room))
    personality_text, personality_clipped = _clip(personality.text.strip(), personality_room)
    candidates = _rank_hits(hits)
    selected, body_truncated = _pack_hits(name, personality, personality_text, candidates, body_budget)
    body = _render_body(name, personality, personality_text, selected)
    while selected and len(body) > body_budget:
        selected.pop()
        body_truncated = True
        body = _render_body(name, personality, personality_text, selected)
    if len(body) > body_budget:
        overflow = len(body) - body_budget
        personality_text, personality_clipped = _clip(personality_text, max(1, len(personality_text) - overflow - 1))
        body_truncated = True
        body = _render_body(name, personality, personality_text, selected)
    evidence_backed = any(
        block["evidence_message_ids"] and block["kind"] in {"memory", "segment"} for block in selected
    )
    retrieval_weak = not evidence_backed
    truncated = question_clipped or body_truncated or personality_clipped
    if retrieval_weak:
        note = "证据不足。没有可引用的消息证据。不要编造经历。"
    elif truncated:
        note = "上下文已按字数上限截断。没有放进提示词的内容不能引用。"
    else:
        note = None
    if note and note not in body:
        noted = body + "\n" + note
        if len(noted) <= body_budget:
            body = noted
        else:
            trimmed, _ = _clip(body, max(1, body_budget - len(note) - 1))
            body = trimmed + "\n" + note
            truncated = True
    leftover = body_budget - len(body)
    if leftover < 0:
        leftover = 0
    history_budget = reserved + leftover
    history_text, history_truncated = _pack_history(turns, max(0, history_budget - 1))
    if history_truncated:
        truncated = True
        if note is None:
            note = "上下文已按字数上限截断。没有放进提示词的内容不能引用。"
            if note not in body and len(body) + 1 + len(note) <= body_budget:
                body = body + "\n" + note
    budgeted = body + "\n" + history_text if history_text else body
    if len(budgeted) > max_chars:
        history_text, _ = _clip(history_text, max(0, max_chars - len(body) - 1))
        budgeted = body + "\n" + history_text if history_text else body[:max_chars]
        truncated = True
    system = render_instructions(name) + "\n\n" + body
    user = _user(history_text, question, question_clipped, target_name=name)
    memories = [block["memory"] for block in selected if block["memory"] is not None]
    return BuiltContext(
        system=system,
        user=user,
        personality_source=personality.source,
        retrieval_weak=retrieval_weak,
        truncated=truncated,
        char_count=len(budgeted),
        budgeted_text=budgeted,
        uncertainty_note=note,
        retrieved_memories=memories,
        context_hits=[block["hit"] for block in selected],
        allowed_memory_ids=[block["source_id"] for block in selected if block["kind"] == "memory"],
        allowed_message_ids=_allowed_messages(selected),
        memory_evidence={
            block["source_id"]: list(block["evidence_message_ids"])
            for block in selected
            if block["kind"] == "memory"
        },
    )


def _pack_hits(name, personality, personality_text: str, candidates: list[dict], body_budget: int) -> tuple[list[dict], bool]:
    """Keep higher-ranked hits that fit, then clip the first hit that did not.

    A long segment must not crowd out a later memory that still fits whole.
    """
    selected: list[dict] = []
    skipped: list[dict] = []
    for block in candidates:
        trial = selected + [block]
        if len(_render_body(name, personality, personality_text, trial)) <= body_budget:
            selected = trial
        else:
            skipped.append(block)
    if not skipped:
        return selected, False
    room = body_budget - len(_render_body(name, personality, personality_text, selected))
    clipped = _clip_block(skipped[0], room)
    if clipped is not None and len(_render_body(name, personality, personality_text, selected + [clipped])) <= body_budget:
        selected.append(clipped)
    return selected, True


def _clip_block(block: dict, room: int) -> dict | None:
    if room < _MIN_BLOCK:
        return None
    rendered = _block_text(block)
    if len(rendered) <= room:
        return block
    clipped = dict(block)
    keep = room - 1
    if keep < _MIN_BLOCK:
        return None
    clipped["rendered"] = rendered[:keep].rstrip() + "…"
    clipped["snippet"] = clipped["rendered"][:160]
    return clipped


def _render_body(name, personality, personality_text: str, selected: list[dict], status: str | None = None) -> str:
    del name
    lines: list[str] = []
    if personality is None:
        source = ""
    else:
        source = personality.source
    lines.append(f"人格来源: {source or 'none'}")
    lines.append(personality_text or "尚无稳定人格档案。")
    lines.append("")
    if status is None:
        weak = not any(block["evidence_message_ids"] and block["kind"] in {"memory", "segment"} for block in selected)
        status = "证据不足" if weak else "可以引用下列编号"
    lines.append(f"检索状态: {status}")
    if selected:
        lines.append("")
        lines.extend(block["rendered"] for block in selected)
    message_ids = _allowed_messages(selected)
    memory_ids = [block["source_id"] for block in selected if block["kind"] == "memory"]
    lines.append("")
    lines.append("可引用 memory_id: " + (", ".join(memory_ids) if memory_ids else "(none)"))
    lines.append("可引用 message_id: " + (", ".join(message_ids) if message_ids else "(none)"))
    return "\n".join(lines)


def _user(history_text: str, question: str, clipped: bool, *, target_name: str = "不吃香菜") -> str:
    history = history_text.strip() or "（这是会话的第一条消息）"
    note = "\n（用户这句话已按字数上限截断。）" if clipped else ""
    name = (target_name or "不吃香菜").strip() or "不吃香菜"
    return (
        f"最近对话:\n{history}\n\n"
        f"用户:\n{question}{note}\n\n"
        f"角色锁定：你现在就是「{name}」的数字分身在用微信回复，第一人称。\n"
        "禁止：通用 AI 助手口吻、自称 GLM/ChatGPT、说自己无法品尝食物。\n"
        '只返回一个 JSON 对象，键必须含 reply，例如 '
        '{"reply":"……","cited_memory_ids":[],"cited_message_ids":[],"confidence":0.6,"uncertainty":""}。'
        "不要使用 Markdown。"
    )


def _pack_history(turns: list[dict], budget: int) -> tuple[str, bool]:
    if budget < 1 or not turns:
        return "", bool(turns)
    pieces: list[str] = []
    used = 0
    truncated = False
    for turn in reversed(turns):
        role = "assistant" if turn.get("role") == "assistant" else "user"
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        piece = f"[{role}]\n{content}"
        extra = len(piece) if not pieces else len(piece) + 1
        if used + extra <= budget:
            pieces.append(piece)
            used += extra
            continue
        room = budget - used - (0 if not pieces else 1)
        clipped, did = _clip(piece, room)
        if clipped:
            pieces.append(clipped)
            truncated = True
        else:
            truncated = True
        break
    if len(pieces) < len([turn for turn in turns if str(turn.get("content") or "").strip()]):
        truncated = True
    pieces.reverse()
    return "\n".join(pieces), truncated


def _rank_hits(hits: list) -> list[dict]:
    ranked = [_normalize(hit) for hit in hits or []]
    ranked = [hit for hit in ranked if hit is not None]
    ranked.sort(key=lambda hit: (hit["group"], hit["status_rank"], -hit["score"], hit["source_id"]))
    return ranked


def _normalize(hit) -> dict | None:
    if hasattr(hit, "to_dict"):
        hit = hit.to_dict()
    if not isinstance(hit, dict):
        return None
    collection = str(hit.get("collection") or "")
    kind = _KIND.get(collection)
    if kind is None:
        return None
    source_id = str(hit.get("source_id") or "").strip()
    text = str(hit.get("text") or "").strip()
    if not source_id or not text:
        return None
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    try:
        score = float(hit.get("score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    if kind == "memory":
        evidence = _strings(metadata.get("evidence_message_ids"))
        group = 0 if evidence else 3
        status_rank = {"active": 0, "uncertain": 1, "historical": 2, "superseded": 4}.get(
            str(metadata.get("status") or ""),
            3,
        )
    elif kind == "trait":
        evidence = []
        group = 1
        status_rank = {"stable": 0, "contested": 1, "provisional": 2}.get(str(metadata.get("status") or ""), 3)
    else:
        evidence = _strings(metadata.get("message_ids"))
        group = 2 if evidence else 4
        status_rank = 0
    block = {
        "kind": kind,
        "collection": collection,
        "source_id": source_id,
        "text": text,
        "score": score,
        "metadata": metadata,
        "evidence_message_ids": evidence,
        "group": group,
        "status_rank": status_rank,
        "memory": None,
        "hit": None,
        "rendered": "",
        "snippet": "",
    }
    block["rendered"] = _block_text(block)
    block["snippet"] = " ".join(text.split())[:160]
    block["hit"] = {
        "kind": kind,
        "source_id": source_id,
        "score": score,
        "evidence_message_ids": list(evidence),
        "snippet": block["snippet"],
    }
    if kind == "memory":
        block["memory"] = {
            "memory_id": source_id,
            "text": text,
            "status": metadata.get("status"),
            "memory_type": metadata.get("memory_type"),
            "score": score,
            "evidence_message_ids": list(evidence),
            "cited": False,
        }
    return block


def _block_text(block: dict) -> str:
    metadata = block["metadata"]
    if block["kind"] == "memory":
        header = (
            f"[memory id={block['source_id']} status={metadata.get('status') or 'unknown'} "
            f"score={block['score']:.4f}]"
        )
        evidence = ", ".join(block["evidence_message_ids"]) if block["evidence_message_ids"] else "(none)"
        return f"{header}\n{block['text']}\nevidence_message_ids: {evidence}"
    if block["kind"] == "trait":
        header = (
            f"[trait id={block['source_id']} dimension={metadata.get('dimension') or 'unknown'} "
            f"status={metadata.get('status') or 'unknown'} score={block['score']:.4f}]"
        )
        return f"{header}\n{block['text']}"
    evidence = ", ".join(block["evidence_message_ids"]) if block["evidence_message_ids"] else "(none)"
    header = f"[segment id={block['source_id']} score={block['score']:.4f}]"
    return f"{header}\n{block['text']}\nmessage_ids: {evidence}"


def _allowed_messages(selected: list[dict]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for block in selected:
        if block["kind"] not in {"memory", "segment"}:
            continue
        for message_id in block["evidence_message_ids"]:
            if message_id not in seen:
                seen.add(message_id)
                found.append(message_id)
    return found


def _strings(value) -> list[str]:
    if not isinstance(value, list):
        return []
    found: list[str] = []
    seen: set[str] = set()
    for item in value:
        if item is None or isinstance(item, bool):
            continue
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        found.append(text)
    return found


def _clip(text: str, limit: int) -> tuple[str, bool]:
    if limit < 1:
        return "", bool(text)
    if len(text) <= limit:
        return text, False
    if limit == 1:
        return "…", True
    return text[: limit - 1].rstrip() + "…", True
