"""Split an ordered message stream into overlapping conversation segments.

The caller supplies messages already ordered within each conversation
(conversation id, then source position). This module does not read the
database and does not concatenate message text.

A new segment starts when:

- the conversation id changes (no overlap is carried across conversations)
- the forward time gap is at least ``gap_seconds`` (overlap only if
  ``overlap_on_time_gap`` is set)
- the open segment already holds ``max_messages`` messages, including any
  overlap prefix

``target_message_count`` is the number of primary messages: the messages this
segment owns, excluding the overlap prefix copied from the previous segment.
Overlap messages stay in ``message_ids`` so evidence can still cite them.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SegmentParams:
    gap_seconds: int = 30 * 60
    max_messages: int = 50
    overlap_messages: int = 4
    overlap_on_time_gap: bool = False

    def __post_init__(self) -> None:
        if self.gap_seconds < 0:
            raise ValueError("gap_seconds must be >= 0")
        if self.max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        if self.overlap_messages < 0:
            raise ValueError("overlap_messages must be >= 0")
        if self.overlap_messages >= self.max_messages:
            raise ValueError("overlap_messages must be smaller than max_messages")

    def digest(self) -> str:
        body = (
            f"{self.gap_seconds}|{self.max_messages}|"
            f"{self.overlap_messages}|{int(self.overlap_on_time_gap)}"
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict:
        return {
            "gap_seconds": self.gap_seconds,
            "max_messages": self.max_messages,
            "overlap_messages": self.overlap_messages,
            "overlap_on_time_gap": self.overlap_on_time_gap,
        }


@dataclass(slots=True)
class MessageRef:
    message_id: str
    conversation_id: str | None
    timestamp: int | None
    source_position: int

    def sort_key(self) -> tuple[str, int, str]:
        return (self.conversation_id or "", int(self.source_position), self.message_id)


@dataclass(slots=True)
class OpenSegment:
    messages: list[MessageRef]
    overlap_count: int
    opened_by: str

    @property
    def conversation_id(self) -> str | None:
        if not self.messages:
            return None
        return self.messages[0].conversation_id


@dataclass(slots=True)
class ClosedSegment:
    messages: list[MessageRef]
    overlap_count: int
    opened_by: str
    closed_by: str

    @property
    def conversation_id(self) -> str | None:
        return self.messages[0].conversation_id

    @property
    def overlap_messages(self) -> list[MessageRef]:
        return self.messages[: self.overlap_count]

    @property
    def primary_messages(self) -> list[MessageRef]:
        return self.messages[self.overlap_count :]

    @property
    def target_message_count(self) -> int:
        return len(self.primary_messages)


def segment_id_for(params_hash: str, conversation_id: str | None, primary_ids: list[str]) -> str:
    body = "\n".join([params_hash, conversation_id or "", *primary_ids])
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def time_bounds(messages: list[MessageRef]) -> tuple[int | None, int | None]:
    stamps = [int(item.timestamp) for item in messages if item.timestamp is not None]
    if not stamps:
        return None, None
    return min(stamps), max(stamps)


def open_segment_payload(segment: OpenSegment | None) -> dict | None:
    if segment is None:
        return None
    return {
        "overlap_count": segment.overlap_count,
        "opened_by": segment.opened_by,
        "messages": [asdict(item) for item in segment.messages],
    }


def open_segment_from_payload(payload: dict) -> OpenSegment:
    messages = [
        MessageRef(
            message_id=item["message_id"],
            conversation_id=item["conversation_id"],
            timestamp=None if item["timestamp"] is None else int(item["timestamp"]),
            source_position=int(item["source_position"]),
        )
        for item in payload["messages"]
    ]
    return OpenSegment(
        messages=messages,
        overlap_count=int(payload["overlap_count"]),
        opened_by=str(payload["opened_by"]),
    )


class Segmenter:
    """Incremental splitter. ``push`` consumes one message; ``finish`` closes the tail."""

    def __init__(self, params: SegmentParams, open_segment: OpenSegment | None = None) -> None:
        self.params = params
        self.open = open_segment

    def push(self, message: MessageRef) -> list[ClosedSegment]:
        if self.open is None or not self.open.messages:
            self.open = OpenSegment(messages=[message], overlap_count=0, opened_by="start")
            return []

        if message.conversation_id != self.open.conversation_id:
            closed = self._close("conversation_change")
            self.open = OpenSegment(messages=[message], overlap_count=0, opened_by="conversation_change")
            return [closed]

        if _is_time_gap(self.open.messages[-1], message, self.params.gap_seconds):
            closed = self._close("time_gap")
            tail = (
                _tail(closed.messages, self.params.overlap_messages)
                if self.params.overlap_on_time_gap
                else []
            )
            self.open = OpenSegment(
                messages=[*tail, message],
                overlap_count=len(tail),
                opened_by="time_gap",
            )
            return [closed]

        if len(self.open.messages) >= self.params.max_messages:
            closed = self._close("max_messages")
            tail = _tail(closed.messages, self.params.overlap_messages)
            self.open = OpenSegment(
                messages=[*tail, message],
                overlap_count=len(tail),
                opened_by="max_messages",
            )
            return [closed]

        self.open.messages.append(message)
        return []

    def finish(self) -> ClosedSegment | None:
        if self.open is None or not self.open.messages:
            self.open = None
            return None
        closed = self._close("end")
        self.open = None
        return closed

    def _close(self, reason: str) -> ClosedSegment:
        if self.open is None or len(self.open.messages) <= self.open.overlap_count:
            raise RuntimeError("cannot close a segment that has no primary messages")
        closed = ClosedSegment(
            messages=list(self.open.messages),
            overlap_count=self.open.overlap_count,
            opened_by=self.open.opened_by,
            closed_by=reason,
        )
        self.open = None
        return closed


def _is_time_gap(previous: MessageRef, current: MessageRef, gap_seconds: int) -> bool:
    if previous.timestamp is None or current.timestamp is None:
        return False
    return int(current.timestamp) - int(previous.timestamp) >= gap_seconds


def _tail(messages: list[MessageRef], overlap: int) -> list[MessageRef]:
    if overlap <= 0:
        return []
    return list(messages[-overlap:])
