"""Shared personality records and the deterministic evidence rules.

Independent evidence is what promotes an L1 note into an L3 trait.
Two observations are not independent when they come from the same segment
or when their message ids overlap (segment overlap copies messages forward).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ObsView:
    observation_id: str
    segment_id: str
    category: str
    dimension: str | None
    summary: str
    evidence_message_ids: list[str]
    confidence: float | None
    time_start: int | None
    time_end: int | None


@dataclass(slots=True)
class TraitRecord:
    trait_id: str
    dimension: str
    statement: str
    situational_scope: str
    evidence_ids: list[str]
    counter_evidence_ids: list[str]
    valid_from: int | None
    valid_to: int | None
    confidence: float
    evidence_count: int
    independent_evidence_count: int
    counter_evidence_count: int
    status: str
    stages: list[dict] = field(default_factory=list)
    variants: list[dict] = field(default_factory=list)
    counter_details: list[dict] = field(default_factory=list)


def independent_group_count(observations: list[ObsView]) -> int:
    """How many mutually independent evidence groups are in ``observations``."""
    if not observations:
        return 0
    parent = list(range(len(observations)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    message_sets = [set(item.evidence_message_ids) for item in observations]
    for index, item in enumerate(observations):
        for earlier in range(index):
            same_segment = item.segment_id == observations[earlier].segment_id
            shared_messages = bool(message_sets[index] & message_sets[earlier])
            if same_segment or shared_messages:
                union(index, earlier)
    return len({find(index) for index in range(len(observations))})


def time_span(observations: list[ObsView]) -> tuple[int | None, int | None]:
    starts = [item.time_start for item in observations if item.time_start is not None]
    ends = [item.time_end for item in observations if item.time_end is not None]
    if not starts and not ends:
        return None, None
    start = min(starts) if starts else min(ends)
    end = max(ends) if ends else max(starts)
    if start > end:
        start, end = end, start
    return start, end


def ranges_disjoint(
    left_start: int | None,
    left_end: int | None,
    right_start: int | None,
    right_end: int | None,
) -> bool:
    """True only when both ranges are known and do not touch."""
    if None in (left_start, left_end, right_start, right_end):
        return False
    return left_end < right_start or right_end < left_start


def trait_confidence(
    confidences: list[float | None],
    independent_groups: int,
    counter_count: int,
) -> float:
    """Blend mean L1 confidence with independent support and counter-evidence.

    Four independent groups saturate the evidence term. Each counter-evidence
    item pulls the score down in proportion to how much of the record it is.
    """
    known = [value for value in confidences if value is not None]
    base = sum(known) / len(known) if known else 0.5
    evidence_factor = min(max(independent_groups, 0), 4) / 4
    total = independent_groups + counter_count
    counter_ratio = (counter_count / total) if total else 0.0
    score = base * (0.55 + 0.45 * evidence_factor) * (1 - 0.75 * counter_ratio)
    if score < 0:
        score = 0.0
    if score > 1:
        score = 1.0
    return round(score, 4)


def classify_status(
    independent_groups: int,
    counter_count: int,
    *,
    min_independent: int,
) -> str:
    if independent_groups < min_independent:
        return "provisional"
    if counter_count > 0 and counter_count >= independent_groups:
        return "contested"
    return "stable"
