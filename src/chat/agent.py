"""One digital-twin turn.

Hybrid search supplies the evidence. The model writes the reply. Citation
ids that were not in that evidence are discarded before the reply is stored.
"""

from __future__ import annotations

from pathlib import Path

from common.config import ROOT, chat_config, personality_config
from common.database import init_db
from common.logger import get_logger
from distillation.llm import LLMError, build_llm_client
from chat.context import build_context
from chat.parser import ground_citations, parse_twin_response
from chat.personality import load_personality
from chat.prompting import PROMPT_VERSION
from chat.store import add_assistant_message, add_user_message, ensure_session, recent_turns

log = get_logger(__name__)

_WEAK_CONFIDENCE_CAP = 0.35
_REJECTED_CONFIDENCE_CAP = 0.5

_OOC_MARKERS = (
    "我是一个乐于助人",
    "乐于助人的AI",
    "AI助手",
    "AI 助手",
    "大语言模型",
    "我是 GLM",
    "由 Z.ai",
    "ChatGPT",
    "我无法品尝",
    "没有真正的喜好",
    "作为人工智能",
    "作为 AI",
)


def _looks_out_of_character(reply: str) -> bool:
    text = (reply or "").strip()
    if not text:
        return True
    return any(marker in text for marker in _OOC_MARKERS)



class ChatNotFound(KeyError):
    """The caller named a session that is not in this database."""


class ChatModelError(Exception):
    """The chat model failed after the user turn was stored."""

    def __init__(self, message: str, session_id: str) -> None:
        super().__init__(message)
        self.session_id = session_id


def chat_turn(
    db_path,
    message: str,
    *,
    session_id: str | None = None,
    target_name: str | None = None,
    llm=None,
    search=None,
    personality_dir: str | Path | None = None,
    use_vector: bool = True,
    use_fts: bool = True,
    limit: int | None = None,
    max_chars: int | None = None,
) -> dict:
    text = (message or "").strip()
    if not text:
        raise ValueError("message is empty")
    database = Path(db_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"database not found: {database}")
    cfg = chat_config()
    name = (target_name or str(cfg.get("target_name") or "不吃香菜")).strip() or "不吃香菜"
    retrieval_limit = int(cfg.get("retrieval_limit", 8) if limit is None else limit)
    budget = int(cfg.get("max_chars", 6000) if max_chars is None else max_chars)
    personality_chars = int(cfg.get("personality_chars", 1400))
    history_chars = int(cfg.get("history_chars", 1600))
    history_turns_limit = int(cfg.get("history_turns", 8))
    user_chars = int(cfg.get("user_chars", 2000))
    if retrieval_limit < 1 or budget < 1:
        raise ValueError("chat limits must be positive")
    if not use_vector and not use_fts:
        raise ValueError("chat needs vector search, full text, or both")

    if search is None:
        hits = search_hits(database, text, limit=retrieval_limit, use_vector=use_vector, use_fts=use_fts)
    else:
        hits = list(search(text))

    output_dir = _personality_dir(personality_dir)
    conn = init_db(database)
    try:
        try:
            session = ensure_session(conn, session_id, target_name=name)
        except KeyError as exc:
            raise ChatNotFound(str(exc)) from exc
        if session_id:
            name = session["target_name"] or name
        history = recent_turns(conn, session["session_id"], history_turns_limit)
        personality = load_personality(conn, output_dir=output_dir, max_chars=personality_chars)
        client = llm if llm is not None else build_llm_client(temperature=float(cfg.get("temperature", 0.4)))
        built = build_context(
            target_name=name,
            personality=personality,
            hits=hits,
            history=history,
            user_message=text,
            max_chars=budget,
            personality_chars=personality_chars,
            history_chars=history_chars,
            history_turns=history_turns_limit,
            user_chars=user_chars,
        )
        add_user_message(conn, session["session_id"], text)
        try:
            raw = client.complete(system=built.system, user=built.user)
        except LLMError as exc:
            raise ChatModelError(str(exc), session["session_id"]) from exc
        parsed = parse_twin_response(raw if isinstance(raw, str) else str(raw))
        if (not parsed.structured) or _looks_out_of_character(parsed.reply):
            repair = (
                built.user
                + "\n\n上次输出不合格：你又跳出角色了。现在必须用「"
                + name
                + "」的口吻，第一人称中文微信回复，"
                + 'JSON 只能是 {"reply":"...","cited_memory_ids":[],"cited_message_ids":[],"confidence":0.5,"uncertainty":""}。'
            )
            try:
                raw = client.complete(system=built.system, user=repair)
            except LLMError as exc:
                raise ChatModelError(str(exc), session["session_id"]) from exc
            repaired = parse_twin_response(raw if isinstance(raw, str) else str(raw))
            if repaired.reply and not _looks_out_of_character(repaired.reply):
                parsed = repaired
        grounded = ground_citations(
            parsed,
            allowed_memory_ids=set(built.allowed_memory_ids),
            allowed_message_ids=set(built.allowed_message_ids),
            memory_evidence=built.memory_evidence,
        )
        reply, confidence, uncertainty = _finalize(name, parsed, grounded, built)
        memories = []
        cited = set(grounded.cited_memory_ids)
        for item in built.retrieved_memories:
            row = dict(item)
            row["cited"] = row["memory_id"] in cited
            memories.append(row)
        model_name = str(getattr(client, "model", "") or "")
        session_id_value = session["session_id"]
        message_id = add_assistant_message(
            conn,
            session["session_id"],
            content=reply,
            retrieved_memories=memories,
            evidence_message_ids=grounded.evidence_message_ids,
            context_hits=built.context_hits,
            confidence=confidence,
            uncertainty=uncertainty,
            retrieval_weak=built.retrieval_weak,
            model=model_name,
        )
    finally:
        conn.close()
    log.info(
        "chat session=%s hits=%s memories=%s evidence=%s weak=%s confidence=%s rejected=%s",
        session_id_value,
        len(hits),
        len(memories),
        len(grounded.evidence_message_ids),
        built.retrieval_weak,
        confidence,
        len(grounded.rejected_ids),
    )
    return {
        "reply": reply,
        "session_id": session_id_value,
        "message_id": message_id,
        "target_name": name,
        "simulation": True,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "retrieval_weak": built.retrieval_weak,
        "truncated": built.truncated,
        "personality_source": built.personality_source,
        "retrieved_memories": memories,
        "evidence_message_ids": grounded.evidence_message_ids,
        "cited_memory_ids": grounded.cited_memory_ids,
        "context_hits": built.context_hits,
        "rejected_ids": grounded.rejected_ids,
        "prompt_version": PROMPT_VERSION,
        "model": model_name,
    }


def search_hits(db_path, query: str, *, limit: int, use_vector: bool, use_fts: bool) -> list[dict]:
    """Call Phase 5 search. Vectors use the whole question.

    Full text quotes the whole string, so a long question is not a phrase
    inside a short memory. When that search is empty, try shorter slices of
    the question with full text only. Those slices still go through
    ``run_search``.
    """
    hits = _search_once(db_path, query, limit=limit, use_vector=use_vector, use_fts=use_fts)
    if hits or not use_fts or len(query) < 4:
        return hits
    for piece in _fallback_slices(query):
        found = _search_once(db_path, piece, limit=limit, use_vector=False, use_fts=True)
        if found:
            log.info("chat retrieval used a shorter full-text slice chars=%s hits=%s", len(piece), len(found))
            return found
    return []


def _search_once(db_path, query: str, *, limit: int, use_vector: bool, use_fts: bool) -> list[dict]:
    from retrieval.errors import RetrievalError
    from retrieval.service import run_search

    try:
        result = run_search(db_path, query, limit=limit, use_vector=use_vector, use_fts=use_fts)
    except RetrievalError:
        if not (use_vector and use_fts):
            raise
        log.warning("vector search failed; retrying with full text only")
        result = run_search(db_path, query, limit=limit, use_vector=False, use_fts=True)
    return list(result.get("hits") or [])


def _fallback_slices(query: str) -> list[str]:
    text = "".join(ch for ch in query if not ch.isspace())
    found: list[str] = []
    seen: set[str] = set()
    for size in (8, 6, 5, 4):
        if len(text) < size:
            continue
        step = 2 if size >= 6 else 1
        for start in range(0, len(text) - size + 1, step):
            piece = text[start : start + size]
            if piece in seen:
                continue
            seen.add(piece)
            found.append(piece)
            if len(found) >= 12:
                return found
    return found


def _finalize(name: str, parsed, grounded, built) -> tuple[str, float, str | None]:
    reply = (parsed.reply or "").strip()
    if not reply:
        reply = "我现在没法回答。"
    if built.retrieval_weak and "模拟" not in reply and "AI" not in reply:
        reply = f"我是「{name}」的 AI 模拟，不是本人。这次没有足够的聊天证据。{reply}"
    confidence = float(parsed.confidence)
    notes: list[str] = []
    if parsed.uncertainty:
        notes.append(parsed.uncertainty)
    if grounded.rejected_ids:
        notes.append("模型给出的部分编号不在本次检索里，已丢弃。")
        confidence = min(confidence, _REJECTED_CONFIDENCE_CAP)
    if built.retrieval_weak:
        confidence = min(confidence, _WEAK_CONFIDENCE_CAP)
        if not any("证据" in note for note in notes):
            notes.append("检索没有带回可引用的消息证据。")
    if built.uncertainty_note and built.truncated and not built.retrieval_weak:
        if built.uncertainty_note not in notes:
            notes.append(built.uncertainty_note)
    if confidence < 0:
        confidence = 0.0
    if confidence > 1:
        confidence = 1.0
    uncertainty = " ".join(notes).strip() or None
    return reply, round(confidence, 4), uncertainty


def _personality_dir(override) -> Path | None:
    if override is not None and str(override).strip():
        path = Path(override).expanduser()
    else:
        configured = personality_config().get("output_dir") or "personality"
        path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path
