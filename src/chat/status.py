"""Session counts for ``show_status``. Message text stays in SQLite."""

from __future__ import annotations


def chat_snapshot_conn(conn) -> dict:
    sessions = int(conn.execute("SELECT COUNT(*) AS n FROM chat_sessions").fetchone()["n"])
    messages = int(conn.execute("SELECT COUNT(*) AS n FROM chat_messages").fetchone()["n"])
    return {"sessions": sessions, "messages": messages}


def format_chat_status(status: dict) -> str:
    return f"chat: sessions {status['sessions']} messages {status['messages']}"
