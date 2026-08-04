from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from notesm_v2.data.lampr_dataset import read_lampr_split
from notesm_v2.data.section_parser import (
    fallback_required_for_model,
    merge_duplicate_sections_for_model,
    parse_sections,
    section_occurrence_counts,
)


@dataclass(frozen=True)
class SectionLengthConfig:
    min_sections: int = 2
    coverage_threshold: float = 0.50
    section_max_tokens: int = 2048


def tokenizer_length(text: str, tokenizer: Any | None = None) -> int:
    if tokenizer is None:
        return len(text.split())
    encoded = tokenizer(text, add_special_tokens=False, padding=False, truncation=False)
    return len(encoded.get("input_ids", []))


def build_section_cache_frame(
    examples: pd.DataFrame,
    split: str,
    tokenizer: Any | None = None,
    length_config: SectionLengthConfig = SectionLengthConfig(),
) -> pd.DataFrame:
    required = {"example_id", "text"}
    missing = required - set(examples.columns)
    if missing:
        raise ValueError(f"examples missing columns for section cache: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for _, row in examples.iterrows():
        parse_result = parse_sections(str(row["text"]))
        occurrence_counts = section_occurrence_counts(parse_result.sections)
        merged_sections = merge_duplicate_sections_for_model(parse_result.sections)
        max_merged_tokens = max((tokenizer_length(section.text, tokenizer) for section in merged_sections), default=0)
        use_fallback = fallback_required_for_model(
            parse_result,
            min_sections=length_config.min_sections,
            coverage_threshold=length_config.coverage_threshold,
            section_max_tokens=length_config.section_max_tokens,
            token_count=lambda text: tokenizer_length(text, tokenizer),
        )
        for section_order, section in enumerate(merged_sections):
            num_occurrences = int(occurrence_counts.get(section.canonical, 0))
            rows.append(
                {
                    "example_id": row["example_id"],
                    "split": split,
                    "canonical_section": section.canonical,
                    "section_order": section_order,
                    "merged_section_text": section.text,
                    "num_occurrences": num_occurrences,
                    "char_length": len(section.text),
                    "token_length": tokenizer_length(section.text, tokenizer),
                    "used_repeated_section_merge": num_occurrences > 1,
                    "use_chunk_fallback": use_fallback,
                    "max_merged_section_tokens": max_merged_tokens,
                }
            )
    return pd.DataFrame(rows)


def chunk_text_by_tokens(text: str, tokenizer: Any | None, max_tokens: int) -> list[tuple[str, int]]:
    words = text.split()
    if tokenizer is None:
        return [(" ".join(words[start : start + max_tokens]), min(max_tokens, len(words) - start)) for start in range(0, len(words), max_tokens)]
    token_ids = tokenizer(text, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    chunks: list[tuple[str, int]] = []
    for start in range(0, len(token_ids), max_tokens):
        ids = token_ids[start : start + max_tokens]
        chunks.append((tokenizer.decode(ids, skip_special_tokens=False), len(ids)))
    return chunks


def build_fallback_chunk_cache_frame(
    examples: pd.DataFrame,
    section_cache: pd.DataFrame,
    split: str,
    tokenizer: Any | None = None,
    chunk_max_tokens: int = 4096,
) -> pd.DataFrame:
    fallback_ids = set(section_cache.loc[section_cache["use_chunk_fallback"], "example_id"].unique())
    rows: list[dict[str, object]] = []
    if not fallback_ids:
        return pd.DataFrame(columns=["example_id", "split", "chunk_order", "chunk_text", "token_length"])
    for _, row in examples[examples["example_id"].isin(fallback_ids)].iterrows():
        for chunk_order, (chunk_text, token_length) in enumerate(chunk_text_by_tokens(str(row["text"]), tokenizer, chunk_max_tokens)):
            rows.append(
                {
                    "example_id": row["example_id"],
                    "split": split,
                    "chunk_order": chunk_order,
                    "chunk_text": chunk_text,
                    "token_length": token_length,
                }
            )
    return pd.DataFrame(rows)


def build_section_cache_for_split(
    benchmark_root: str | Path,
    split: str,
    output_dir: str | Path,
    tokenizer: Any | None = None,
    limit: int | None = None,
    length_config: SectionLengthConfig = SectionLengthConfig(),
) -> Path:
    examples = read_lampr_split(benchmark_root, split)
    if limit is not None:
        examples = examples.head(limit).copy()
    cache = build_section_cache_frame(examples, split=split, tokenizer=tokenizer, length_config=length_config)
    output_path = Path(output_dir) / f"section_cache_{split}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(output_path, index=False)
    fallback_cache = build_fallback_chunk_cache_frame(
        examples,
        section_cache=cache,
        split=split,
        tokenizer=tokenizer,
        chunk_max_tokens=length_config.section_max_tokens,
    )
    fallback_cache.to_parquet(Path(output_dir) / f"fallback_chunk_cache_{split}.parquet", index=False)
    return output_path
