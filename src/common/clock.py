"""UTC timestamps stored in SQLite as ISO-8601 strings."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
