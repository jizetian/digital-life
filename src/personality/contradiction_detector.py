"""Apply counter-evidence, situational scope, and time-stage changes.

Relations come from the model. Time decides whether an opposition is a
contradiction inside one period or a change between periods. Missing times
are not treated as a clean stage change.
"""

from __future__ import annotations

import hashlib

from personality.jsonutil import ResponseParseError, load_json_payload
from personality.models import (
    ObsView,
    TraitRecord,
    classify_status,
    independent_group_count,
    ranges_disjoint,
    time_span,
    trait_confidence,
)
from personality.prompting import RELATIONS

_RELATION_RANK = {
    "contradicts": 4,
    "evolution": 3,
    "situational": 2,
    "supports": 1,
    "unrelated": 0,
}


def parse_judgments(
    text: str,
    *,
    trait_keys: set[str],
    allowed_observation_ids: set[str],
) -> list[dict]:
    payload = load_json_payload(text)
    raw_items = payload.get("judgments", [])
    if raw_items is None:
        raw_items = []
    if not isinstance(raw_items, list):
        raise ResponseParseError("judgments must be a list")
    judgments: list[dict] = []
    for item in raw_items:
        parsed = _one_judgment(item, trait_keys, allowed_observation_ids)
        if parsed is not None:
            judgments.append(parsed)
    return judgments


def finalize_traits(
    specs: list[tuple[str, dict]],
    judgments: list[dict],
    observations: dict[str, ObsView],
    *,
    dimension: str,
    min_independent: int,
    prompt_version: str,
    model_version: str,
) -> list[TraitRecord]:
    """Turn merged trait specs plus judgments into scored trait records."""
    by_trait: dict[str, list[dict]] = {key: [] for key, _ in specs}
    for judgment in judgments:
        key = judgment["trait_key"]
        if key in by_trait:
            by_trait[key].append(judgment)
    records: list[TraitRecord] = []
    seen: set[str] = set()
    for key, spec in specs:
        record = finalize_trait(
            trait_key=key,
            dimension=dimension,
            spec=spec,
            judgments=by_trait.get(key, []),
            observations=observations,
            min_independent=min_independent,
            prompt_version=prompt_version,
            model_version=model_version,
        )
        if record is None or record.trait_id in seen:
            continue
        seen.add(record.trait_id)
        records.append(record)
    return records


def finalize_trait(
    *,
    trait_key: str,
    dimension: str,
    spec: dict,
    judgments: list[dict],
    observations: dict[str, ObsView],
    min_independent: int,
    prompt_version: str,
    model_version: str,
) -> TraitRecord | None:
    del trait_key
    evidence_ids = [oid for oid in spec.get("observation_ids") or [] if oid in observations]
    statement = spec.get("statement") or ""
    if not evidence_ids or not statement:
        return None
    scope = spec.get("situational_scope") or "general"
    collapsed = _collapse(judgments)
    variants: list[dict] = []
    counter_ids: list[str] = []
    counter_details: list[dict] = []
    evolution: list[tuple[ObsView, str]] = []

    for observation_id, judgment in collapsed:
        if observation_id in evidence_ids or observation_id not in observations:
            continue
        relation = _relation_for(judgment, observations[observation_id], evidence_ids, observations)
        obs = observations[observation_id]
        reason = judgment.get("reason") or ""
        if relation == "supports":
            evidence_ids.append(observation_id)
        elif relation == "contradicts":
            counter_ids.append(observation_id)
            counter_details.append({"observation_id": observation_id, "reason": reason})
        elif relation == "evolution":
            evolution.append((obs, obs.summary))
        elif relation == "situational":
            variants.append(
                {
                    "observation_id": observation_id,
                    "situational_scope": judgment.get("situational_scope") or "unspecified situation",
                    "reason": reason,
                }
            )

    support = [observations[oid] for oid in evidence_ids]
    groups = independent_group_count(support)
    stages = _compose_stages(
        statement=statement,
        evidence_ids=evidence_ids,
        model_stages=spec.get("stages") or [],
        evolution=evolution,
        observations=observations,
    )
    # A stage that collides with the support window is counter-evidence.
    support_start, support_end = time_span(support)
    kept_stages: list[dict] = []
    for stage in stages:
        ids = stage["observation_ids"]
        if set(ids) <= set(evidence_ids):
            kept_stages.append(stage)
            continue
        if ranges_disjoint(support_start, support_end, stage["valid_from"], stage["valid_to"]):
            kept_stages.append(stage)
            continue
        for observation_id in ids:
            if observation_id not in evidence_ids and observation_id not in counter_ids:
                counter_ids.append(observation_id)
                counter_details.append(
                    {"observation_id": observation_id, "reason": "overlaps the trait period"}
                )
    stages = _sequential_stages(kept_stages)
    span_ids = list(evidence_ids)
    for stage in stages:
        for observation_id in stage["observation_ids"]:
            if observation_id not in span_ids:
                span_ids.append(observation_id)
    valid_from, valid_to = time_span([observations[oid] for oid in span_ids if oid in observations])
    confidence = trait_confidence(
        [item.confidence for item in support],
        groups,
        len(counter_ids),
    )
    status = classify_status(groups, len(counter_ids), min_independent=min_independent)
    trait_id = hashlib.sha256(
        "|".join(
            [
                dimension,
                statement,
                ",".join(sorted(evidence_ids)),
                prompt_version,
                model_version,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return TraitRecord(
        trait_id=trait_id,
        dimension=dimension,
        statement=statement,
        situational_scope=scope,
        evidence_ids=evidence_ids,
        counter_evidence_ids=counter_ids,
        valid_from=valid_from,
        valid_to=valid_to,
        confidence=confidence,
        evidence_count=len(evidence_ids),
        independent_evidence_count=groups,
        counter_evidence_count=len(counter_ids),
        status=status,
        stages=stages,
        variants=variants,
        counter_details=counter_details,
    )


def _one_judgment(item, trait_keys: set[str], allowed_observation_ids: set[str]) -> dict | None:
    if not isinstance(item, dict):
        return None
    trait_key = str(item.get("trait_key", "")).strip()
    observation_id = str(item.get("observation_id", "")).strip()
    relation = str(item.get("relation", "")).strip().lower()
    if trait_key not in trait_keys or observation_id not in allowed_observation_ids:
        return None
    if relation not in RELATIONS:
        return None
    reason = item.get("reason")
    if not isinstance(reason, str):
        reason = ""
    reason = " ".join(reason.split())[:200]
    scope = item.get("situational_scope")
    if not isinstance(scope, str) or not scope.strip():
        scope = None
    else:
        scope = " ".join(scope.split())[:200]
    return {
        "trait_key": trait_key,
        "observation_id": observation_id,
        "relation": relation,
        "situational_scope": scope,
        "reason": reason,
    }


def _collapse(judgments: list[dict]) -> list[tuple[str, dict]]:
    """One judgment per observation. Opposition outranks a softer label."""
    chosen: dict[str, dict] = {}
    for judgment in judgments:
        observation_id = judgment["observation_id"]
        current = chosen.get(observation_id)
        if current is None or _RELATION_RANK[judgment["relation"]] > _RELATION_RANK[current["relation"]]:
            chosen[observation_id] = judgment
    return [(observation_id, chosen[observation_id]) for observation_id in sorted(chosen)]


def _relation_for(
    judgment: dict,
    obs: ObsView,
    evidence_ids: list[str],
    observations: dict[str, ObsView],
) -> str:
    relation = judgment["relation"]
    if relation in {"unrelated", "supports", "situational"}:
        return relation
    support = [observations[oid] for oid in evidence_ids if oid in observations]
    start, end = time_span(support)
    disjoint = ranges_disjoint(start, end, obs.time_start, obs.time_end)
    if relation in {"contradicts", "evolution"}:
        if disjoint:
            return "evolution"
        return "contradicts"
    return "unrelated"


def _compose_stages(
    *,
    statement: str,
    evidence_ids: list[str],
    model_stages: list,
    evolution: list[tuple[ObsView, str]],
    observations: dict[str, ObsView],
) -> list[dict]:
    stages = _model_stages(model_stages, evidence_ids, observations)
    if not evolution:
        return stages
    if not stages:
        support = [observations[oid] for oid in evidence_ids if oid in observations]
        start, end = time_span(support)
        stages = [
            {
                "statement": statement,
                "observation_ids": list(evidence_ids),
                "valid_from": start,
                "valid_to": end,
            }
        ]
    for obs, stage_statement in evolution:
        _add_evolution_stage(stages, obs, stage_statement)
    return stages


def _model_stages(raw_stages, evidence_ids: list[str], observations: dict[str, ObsView]) -> list[dict]:
    if not isinstance(raw_stages, list):
        return []
    allowed = set(evidence_ids)
    stages: list[dict] = []
    for item in raw_stages:
        if not isinstance(item, dict):
            continue
        statement = item.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            continue
        ids = [
            oid
            for oid in item.get("observation_ids") or []
            if oid in allowed and oid in observations
        ]
        if not ids:
            continue
        start, end = time_span([observations[oid] for oid in ids])
        stages.append(
            {
                "statement": " ".join(statement.split()),
                "observation_ids": ids,
                "valid_from": start,
                "valid_to": end,
            }
        )
    return _sequential_stages(stages)


def _add_evolution_stage(stages: list[dict], obs: ObsView, statement: str) -> None:
    if any(obs.observation_id in stage["observation_ids"] for stage in stages):
        return
    # stages[0] is the baseline when we synthesized one. Later stages are other periods.
    for stage in stages[1:]:
        if not ranges_disjoint(stage["valid_from"], stage["valid_to"], obs.time_start, obs.time_end):
            if obs.observation_id not in stage["observation_ids"]:
                stage["observation_ids"].append(obs.observation_id)
            start_values = [value for value in (stage["valid_from"], obs.time_start) if value is not None]
            end_values = [value for value in (stage["valid_to"], obs.time_end) if value is not None]
            if start_values:
                stage["valid_from"] = min(start_values)
            if end_values:
                stage["valid_to"] = max(end_values)
            return
    stages.append(
        {
            "statement": statement,
            "observation_ids": [obs.observation_id],
            "valid_from": obs.time_start,
            "valid_to": obs.time_end,
        }
    )


def _sequential_stages(stages: list[dict]) -> list[dict]:
    """Keep stages only when each period is strictly before the next."""
    if len(stages) < 2:
        return []
    ordered = sorted(
        stages,
        key=lambda stage: (stage.get("valid_from") is None, stage.get("valid_from") or 0, stage.get("statement") or ""),
    )
    for earlier, later in zip(ordered, ordered[1:]):
        if not ranges_disjoint(
            earlier.get("valid_from"),
            earlier.get("valid_to"),
            later.get("valid_from"),
            later.get("valid_to"),
        ):
            return []
    return ordered
