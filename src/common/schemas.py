"""Normalized message model and import-report counters.

Field mapping follows the Phase 1 spec. Unknown source fields stay empty;
nothing is inferred from a display name or from ``time_str``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


ROLE_OWNER = "owner"
ROLE_TARGET = "target_person"
ROLE_UNKNOWN = "unknown"

# Stable keys written to import_jobs.stats_json and printed by the CLI.
REPORT_FIELDS = (
    "total_messages",
    "success_count",
    "failed_count",
    "missing_timestamp_count",
    "missing_sender_count",
    "duplicate_count",
    "kind_counts",
    "type_name_counts",
    "target_person_ratio",
    "text_ratio",
)


@dataclass(slots=True)
class NormalizedMessage:
    message_id: str
    conversation_id: str | None
    local_id: int | None
    server_id: str | None
    shard: int | None
    sender_id: str | None
    sender_name: str | None
    sender_role: str
    is_sender: bool | None
    timestamp: int | None
    time_str: str | None
    message_type: str | None
    type_code: int | None
    raw_type: int | None
    type_name: str | None
    content: str | None
    raw_content: dict
    is_forwarded: bool
    resource_status: str | None
    resource_notice: str | None
    source_file: str
    source_position: int
    missing_timestamp: bool
    missing_sender: bool


@dataclass
class RunCounters:
    total_messages: int = 0
    failed_count: int = 0
    missing_timestamp_count: int = 0
    missing_sender_count: int = 0
    target_person_count: int = 0
    owner_count: int = 0
    unknown_role_count: int = 0
    text_count: int = 0
    kind_counts: Counter[str] = field(default_factory=Counter)
    type_name_counts: Counter[str] = field(default_factory=Counter)

    def observe(self, message: NormalizedMessage) -> None:
        if message.missing_timestamp:
            self.missing_timestamp_count += 1
        if message.missing_sender:
            self.missing_sender_count += 1
        if message.sender_role == ROLE_TARGET:
            self.target_person_count += 1
        elif message.sender_role == ROLE_OWNER:
            self.owner_count += 1
        else:
            self.unknown_role_count += 1
        kind_key = message.message_type if message.message_type is not None else "<missing>"
        type_name_key = message.type_name if message.type_name is not None else "<missing>"
        self.kind_counts[kind_key] += 1
        self.type_name_counts[type_name_key] += 1
        if message.message_type == "text":
            self.text_count += 1

    def to_report(self, success_count: int, duplicate_count: int) -> dict:
        classified = self.total_messages - self.failed_count
        report = {
            "total_messages": self.total_messages,
            "success_count": success_count,
            "failed_count": self.failed_count,
            "missing_timestamp_count": self.missing_timestamp_count,
            "missing_sender_count": self.missing_sender_count,
            "duplicate_count": duplicate_count,
            "kind_counts": dict(sorted(self.kind_counts.items())),
            "type_name_counts": dict(sorted(self.type_name_counts.items())),
            "target_person_count": self.target_person_count,
            "owner_count": self.owner_count,
            "unknown_role_count": self.unknown_role_count,
            "text_count": self.text_count,
            "target_person_ratio": _ratio(self.target_person_count, classified),
            "text_ratio": _ratio(self.text_count, classified),
        }
        missing = [key for key in REPORT_FIELDS if key not in report]
        if missing:
            raise RuntimeError(f"import report is missing fields: {missing}")
        return report


def _ratio(part: int, whole: int) -> float | None:
    if whole <= 0:
        return None
    return round(part / whole, 6)
