"""Merge L1 observations into candidate traits, then merge those across batches.

The model proposes groupings. This module drops ids it was not given and
keeps any local trait the cross-batch step fails to mention, so a response
cannot erase a whole batch of evidence.
"""

from __future__ import annotations

import hashlib

from personality.jsonutil import ResponseParseError, load_json_payload
from personality.models import ObsView

_TRAIT_FIELDS = ("statement", "situational_scope", "observation_ids", "stages")


def batch_key_for(observation_ids: list[str]) -> str:
    raw = ",".join(sorted(observation_ids))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def observation_fingerprint(observation_ids: list[str]) -> str:
    raw = ",".join(sorted(observation_ids))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def plan_batches(observations: list[ObsView], batch_size: int) -> list[list[ObsView]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ordered = sorted(
        observations,
        key=lambda item: (item.time_start is None, item.time_start or 0, item.observation_id),
    )
    return [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]


def parse_merge_response(text: str, *, allowed_ids: set[str]) -> list[dict]:
    """Return trait dicts whose observation ids all belong to ``allowed_ids``."""
    payload = load_json_payload(text)
    raw_items = payload.get("traits", [])
    if raw_items is None:
        raw_items = []
    if not isinstance(raw_items, list):
        raise ResponseParseError("traits must be a list")
    traits: list[dict] = []
    seen: set[tuple] = set()
    for item in raw_items:
        parsed = _trait_from_merge(item, allowed_ids)
        if parsed is None:
            continue
        identity = (parsed["statement"], tuple(parsed["observation_ids"]))
        if identity in seen:
            continue
        seen.add(identity)
        traits.append(parsed)
    return traits


def parse_reduce_response(text: str, *, local_traits: dict[str, dict]) -> list[dict]:
    """Union local traits cited by the model, then keep any trait it skipped."""
    payload = load_json_payload(text)
    raw_items = payload.get("traits", [])
    if raw_items is None:
        raw_items = []
    if not isinstance(raw_items, list):
        raise ResponseParseError("traits must be a list")
    parsed: list[tuple[list[str], dict]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        keys = _string_list(item.get("trait_keys"))
        keys = [key for key in keys if key in local_traits]
        if not keys:
            continue
        statement = _statement(item.get("statement"))
        if statement is None:
            statement = _statement(local_traits[keys[0]].get("statement")) or ""
        if not statement:
            continue
        parsed.append(
            (
                keys,
                {
                    "statement": statement,
                    "situational_scope": _scope(item.get("situational_scope")),
                    "stages": item.get("stages"),
                },
            )
        )
    return resolve_reduced_traits(parsed, local_traits)


def resolve_reduced_traits(
    parsed: list[tuple[list[str], dict]],
    local_traits: dict[str, dict],
) -> list[dict]:
    """Build global traits. Observation ids already used are not repeated."""
    used: set[str] = set()
    cited_keys: set[str] = set()
    resolved: list[dict] = []
    for keys, spec in parsed:
        cited_keys.update(keys)
        observation_ids = _take_ids(keys, local_traits, used)
        if not observation_ids:
            continue
        resolved.append(
            _pack_trait(
                statement=spec["statement"],
                situational_scope=spec["situational_scope"],
                observation_ids=observation_ids,
                stages=_stages_for(spec.get("stages"), observation_ids),
            )
        )
    for key in sorted(local_traits):
        if key in cited_keys:
            continue
        trait = local_traits[key]
        observation_ids = [oid for oid in trait.get("observation_ids") or [] if oid not in used]
        if not observation_ids:
            continue
        used.update(observation_ids)
        resolved.append(
            _pack_trait(
                statement=_statement(trait.get("statement")) or "",
                situational_scope=_scope(trait.get("situational_scope")),
                observation_ids=observation_ids,
                stages=_stages_for(trait.get("stages"), observation_ids),
            )
        )
    return [item for item in resolved if item["statement"] and item["observation_ids"]]


def local_traits_from_results(merge_results: list[tuple[str, dict]]) -> dict[str, dict]:
    """Assign stable ``b{n}t{m}`` keys in batch_key order."""
    local: dict[str, dict] = {}
    ordered = sorted(merge_results, key=lambda item: item[0])
    for batch_index, (_, payload) in enumerate(ordered):
        traits = payload.get("traits") or []
        if not isinstance(traits, list):
            continue
        for trait_index, trait in enumerate(traits):
            if not isinstance(trait, dict):
                continue
            observation_ids = _string_list(trait.get("observation_ids"))
            statement = _statement(trait.get("statement"))
            if not statement or not observation_ids:
                continue
            local[f"b{batch_index}t{trait_index}"] = _pack_trait(
                statement=statement,
                situational_scope=_scope(trait.get("situational_scope")),
                observation_ids=observation_ids,
                stages=_stages_for(trait.get("stages"), observation_ids),
            )
    return local


def reduce_batch_key(local_traits: dict[str, dict]) -> str:
    parts = [f"{key}:{_trait_identity(local_traits[key])}" for key in sorted(local_traits)]
    raw = "|".join(parts) if parts else "none"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def trait_fingerprint(traits: list[dict]) -> str:
    parts = [f"{key}:{_trait_identity(trait)}" for key, trait in assign_trait_keys(traits)]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _trait_identity(trait: dict) -> str:
    """Content hash input so a rewritten statement invalidates later tasks."""
    stages = trait.get("stages") or []
    stage_bits: list[str] = []
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            ids = ",".join(stage.get("observation_ids") or [])
            stage_bits.append(f"{stage.get('statement')}:{ids}")
    return "|".join(
        [
            trait.get("statement") or "",
            trait.get("situational_scope") or "",
            ",".join(trait.get("observation_ids") or []),
            ";".join(stage_bits),
        ]
    )


def assign_trait_keys(traits: list[dict]) -> list[tuple[str, dict]]:
    ordered = sorted(
        traits,
        key=lambda trait: (
            trait.get("statement") or "",
            ",".join(trait.get("observation_ids") or []),
        ),
    )
    return [(f"t{index}", trait) for index, trait in enumerate(ordered)]


def _trait_from_merge(item, allowed_ids: set[str]) -> dict | None:
    if not isinstance(item, dict):
        return None
    statement = _statement(item.get("statement"))
    if statement is None:
        return None
    observation_ids = [oid for oid in _string_list(item.get("observation_ids")) if oid in allowed_ids]
    if not observation_ids:
        return None
    return _pack_trait(
        statement=statement,
        situational_scope=_scope(item.get("situational_scope")),
        observation_ids=observation_ids,
        stages=_stages_for(item.get("stages"), observation_ids),
    )


def _stages_for(raw, observation_ids: list[str]) -> list[dict]:
    if not isinstance(raw, list):
        return []
    allowed = set(observation_ids)
    stages: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        statement = _statement(item.get("statement"))
        if statement is None:
            continue
        ids = [oid for oid in _string_list(item.get("observation_ids")) if oid in allowed]
        if not ids:
            continue
        stages.append({"statement": statement, "observation_ids": ids})
    return stages


def _pack_trait(
    *,
    statement: str,
    situational_scope: str,
    observation_ids: list[str],
    stages: list[dict],
) -> dict:
    return {
        "statement": statement,
        "situational_scope": situational_scope,
        "observation_ids": observation_ids,
        "stages": stages,
    }


def _take_ids(keys: list[str], local_traits: dict[str, dict], used: set[str]) -> list[str]:
    taken: list[str] = []
    for key in keys:
        for observation_id in local_traits[key].get("observation_ids") or []:
            if observation_id in used or observation_id in taken:
                continue
            taken.append(observation_id)
            used.add(observation_id)
    return taken


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _statement(value) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text or None


def _scope(value) -> str:
    if not isinstance(value, str) or not value.strip():
        return "general"
    return " ".join(value.split())


# Re-exported so callers can see the stored shape next to the parser.
TRAIT_FIELDS = _TRAIT_FIELDS
