"""Track message ids already seen inside the current export file."""

from __future__ import annotations


class Deduplicator:
    def __init__(self) -> None:
        self._seen: set[str] = set()

    def seen_in_file(self, message_id: str) -> bool:
        """Return True when ``message_id`` already appeared in this file."""
        if message_id in self._seen:
            return True
        self._seen.add(message_id)
        return False

    def __len__(self) -> int:
        return len(self._seen)
