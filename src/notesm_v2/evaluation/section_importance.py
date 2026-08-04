from __future__ import annotations

from collections.abc import Mapping

from notesm_v2.data.section_parser import ParsedSection, merge_duplicate_sections_for_model


def canonical_attention_weights(
    sections: list[ParsedSection],
    occurrence_attention: Mapping[int, float],
) -> dict[str, float]:
    """Aggregate occurrence-level attention onto merged canonical sections."""
    weights: dict[str, float] = {}
    for index, section in enumerate(sections):
        weights[section.canonical] = weights.get(section.canonical, 0.0) + float(occurrence_attention.get(index, 0.0))
    return weights


def remove_canonical_section_occurrences(sections: list[ParsedSection], canonical: str) -> list[ParsedSection]:
    """Remove every occurrence of a canonical section for ablation."""
    return [section for section in sections if section.canonical != canonical]


def top_k_canonical_sections(sections: list[ParsedSection], scores: Mapping[str, float], k: int) -> list[ParsedSection]:
    """Keep top-k canonical section buckets after merging repeats."""
    merged = merge_duplicate_sections_for_model(sections)
    keep = {canonical for canonical, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]}
    return [section for section in merged if section.canonical in keep]
