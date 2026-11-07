"""Write the L3 personality profile as markdown plus an evidence JSON file.

Stable traits are the profile. Contested traits stay visible with their
counter-evidence. Provisional traits stay in the JSON only: they do not
yet have enough independent evidence to be part of the profile.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from personality.dimensions import CONTEXT_CATEGORIES, DIMENSION_META, DIMENSIONS, PROFILE_FILES
from personality.models import ObsView, TraitRecord


def build_profile(
    *,
    output_dir: Path,
    traits: list[TraitRecord],
    observations: dict[str, ObsView],
    context_counts: dict[str, int],
    prompt_version: str,
    model_version: str,
    observation_fingerprint: str,
    min_independent: int,
    generated_at: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_dimension = {name: [] for name in DIMENSIONS}
    for trait in traits:
        if trait.dimension in by_dimension:
            by_dimension[trait.dimension].append(trait)
    for name in by_dimension:
        by_dimension[name].sort(key=lambda trait: trait.statement)

    _write(output_dir / "personality_core.md", _core(
        by_dimension,
        context_counts=context_counts,
        prompt_version=prompt_version,
        model_version=model_version,
        observation_fingerprint=observation_fingerprint,
        min_independent=min_independent,
        generated_at=generated_at,
    ))
    for name in DIMENSIONS:
        meta = DIMENSION_META[name]
        _write(
            output_dir / meta["file"],
            _dimension(
                name,
                by_dimension[name],
                prompt_version=prompt_version,
                model_version=model_version,
                generated_at=generated_at,
            ),
        )
    evidence = _evidence_document(
        by_dimension,
        observations,
        context_counts=context_counts,
        prompt_version=prompt_version,
        model_version=model_version,
        observation_fingerprint=observation_fingerprint,
        min_independent=min_independent,
        generated_at=generated_at,
    )
    _write(
        output_dir / "personality_evidence.json",
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
    )
    counts = _counts(traits)
    return {
        "output_dir": str(output_dir),
        "files": list(PROFILE_FILES),
        "traits_stable": counts["stable"],
        "traits_provisional": counts["provisional"],
        "traits_contested": counts["contested"],
    }


def _core(
    by_dimension: dict[str, list[TraitRecord]],
    *,
    context_counts: dict[str, int],
    prompt_version: str,
    model_version: str,
    observation_fingerprint: str,
    min_independent: int,
    generated_at: str,
) -> str:
    stable = [trait for traits in by_dimension.values() for trait in traits if trait.status == "stable"]
    contested = [trait for traits in by_dimension.values() for trait in traits if trait.status == "contested"]
    provisional = [trait for traits in by_dimension.values() for trait in traits if trait.status == "provisional"]
    lines = [
        "# Personality core",
        "",
        "Cross-segment traits distilled from L1 observations. "
        "A trait is stable only when independent evidence agrees.",
        "",
        f"- prompt_version: {prompt_version}",
        f"- model_version: {model_version}",
        f"- generated_at: {generated_at}",
        f"- observation_fingerprint: {observation_fingerprint}",
        f"- min_independent_evidence: {min_independent}",
        "",
        "## Stable traits",
        "",
    ]
    lines.extend(_trait_blocks(stable) or ["None.", ""])
    lines.extend(["## Contested traits", ""])
    lines.extend(_trait_blocks(contested) or ["None.", ""])
    lines.extend(
        [
            "## Provisional traits withheld",
            "",
            (
                f"{len(provisional)} trait(s) had fewer than {min_independent} independent "
                "evidence groups. They are stored in personality_evidence.json and are not part of the profile."
            ),
            "",
            "## Temporal evolution",
            "",
        ]
    )
    evolution = _evolution_lines(stable)
    lines.extend(evolution or ["No stage change was supported.", ""])
    lines.extend(["## Context notes", ""])
    lines.append(
        "L1 categories interest, event, and fact are not personality dimensions."
    )
    for category in CONTEXT_CATEGORIES:
        lines.append(f"- {category}: {int(context_counts.get(category, 0))}")
    lines.append("")
    return "\n".join(lines)


def _dimension(
    name: str,
    traits: list[TraitRecord],
    *,
    prompt_version: str,
    model_version: str,
    generated_at: str,
) -> str:
    meta = DIMENSION_META[name]
    stable = [trait for trait in traits if trait.status == "stable"]
    contested = [trait for trait in traits if trait.status == "contested"]
    lines = [
        f"# {meta['title']}",
        "",
        meta["heading"],
        "",
        f"- prompt_version: {prompt_version}",
        f"- model_version: {model_version}",
        f"- generated_at: {generated_at}",
        "",
        "## Stable traits",
        "",
    ]
    lines.extend(_trait_blocks(stable) or ["None.", ""])
    lines.extend(["## Contested traits", ""])
    lines.extend(_trait_blocks(contested) or ["None.", ""])
    lines.extend(["## Temporal evolution", ""])
    lines.extend(_evolution_lines(stable) or ["No stage change was supported.", ""])
    return "\n".join(lines)


def _trait_blocks(traits: list[TraitRecord]) -> list[str]:
    blocks: list[str] = []
    for trait in traits:
        blocks.extend(
            [
                f"### {trait.statement}",
                "",
                f"- dimension: {trait.dimension}",
                f"- status: {trait.status}",
                f"- confidence: {trait.confidence}",
                f"- evidence_count: {trait.evidence_count}",
                f"- independent_evidence_count: {trait.independent_evidence_count}",
                f"- counter_evidence_count: {trait.counter_evidence_count}",
                f"- valid_from: {_field_time(trait.valid_from)}",
                f"- valid_to: {_field_time(trait.valid_to)}",
                f"- evidence_ids: {_id_list(trait.evidence_ids)}",
                f"- counter_evidence_ids: {_id_list(trait.counter_evidence_ids)}",
                f"- situational_scope: {trait.situational_scope}",
                f"- trait_id: {trait.trait_id}",
                "",
            ]
        )
    return blocks


def _evolution_lines(traits: list[TraitRecord]) -> list[str]:
    lines: list[str] = []
    for trait in traits:
        if len(trait.stages) < 2:
            continue
        lines.append(f"### {trait.dimension}: {trait.statement}")
        lines.append("")
        for stage in trait.stages:
            lines.append(
                f"- {_field_time(stage.get('valid_from'))} – {_field_time(stage.get('valid_to'))}: "
                f"{stage.get('statement')}"
            )
            lines.append(f"  - evidence_ids: {_id_list(stage.get('observation_ids') or [])}")
        lines.append("")
    return lines


def _evidence_document(
    by_dimension: dict[str, list[TraitRecord]],
    observations: dict[str, ObsView],
    *,
    context_counts: dict[str, int],
    prompt_version: str,
    model_version: str,
    observation_fingerprint: str,
    min_independent: int,
    generated_at: str,
) -> dict:
    dimensions = {}
    evolution = []
    for name in DIMENSIONS:
        meta = DIMENSION_META[name]
        encoded = [_trait_json(trait, observations) for trait in by_dimension[name]]
        dimensions[name] = {
            "title": meta["title"],
            "file": meta["file"],
            "traits": encoded,
        }
        for trait in by_dimension[name]:
            if trait.status == "stable" and len(trait.stages) >= 2:
                evolution.append(
                    {
                        "dimension": name,
                        "trait_id": trait.trait_id,
                        "statement": trait.statement,
                        "stages": trait.stages,
                    }
                )
    return {
        "prompt_version": prompt_version,
        "model_version": model_version,
        "generated_at": generated_at,
        "observation_fingerprint": observation_fingerprint,
        "min_independent_evidence": min_independent,
        "dimensions": dimensions,
        "evolution": evolution,
        "context_observations": {category: int(context_counts.get(category, 0)) for category in CONTEXT_CATEGORIES},
    }


def _trait_json(trait: TraitRecord, observations: dict[str, ObsView]) -> dict:
    return {
        "trait_id": trait.trait_id,
        "dimension": trait.dimension,
        "statement": trait.statement,
        "situational_scope": trait.situational_scope,
        "status": trait.status,
        "confidence": trait.confidence,
        "evidence_count": trait.evidence_count,
        "independent_evidence_count": trait.independent_evidence_count,
        "counter_evidence_count": trait.counter_evidence_count,
        "valid_from": trait.valid_from,
        "valid_to": trait.valid_to,
        "evidence_ids": list(trait.evidence_ids),
        "counter_evidence_ids": list(trait.counter_evidence_ids),
        "stages": trait.stages,
        "variants": trait.variants,
        "evidence": [_obs_json(observations[oid]) for oid in trait.evidence_ids if oid in observations],
        "counter_evidence": [
            {
                **_obs_json(observations[oid]),
                "reason": _reason_for(trait, oid),
            }
            for oid in trait.counter_evidence_ids
            if oid in observations
        ],
    }


def _obs_json(obs: ObsView) -> dict:
    return {
        "observation_id": obs.observation_id,
        "segment_id": obs.segment_id,
        "category": obs.category,
        "summary": obs.summary,
        "message_ids": list(obs.evidence_message_ids),
        "time_start": obs.time_start,
        "time_end": obs.time_end,
        "confidence": obs.confidence,
    }


def _reason_for(trait: TraitRecord, observation_id: str) -> str:
    for item in trait.counter_details:
        if item.get("observation_id") == observation_id:
            return item.get("reason") or ""
    return ""


def _counts(traits: list[TraitRecord]) -> dict[str, int]:
    counts = {"stable": 0, "provisional": 0, "contested": 0}
    for trait in traits:
        if trait.status in counts:
            counts[trait.status] += 1
    return counts


def _field_time(value) -> str:
    if value is None:
        return "unknown"
    try:
        stamp = int(value)
    except (TypeError, ValueError):
        return "unknown"
    day = datetime.fromtimestamp(stamp, timezone.utc).strftime("%Y-%m-%d")
    return f"{stamp} ({day})"


def _id_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
