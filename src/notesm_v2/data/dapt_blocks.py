from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


RAW_TEXT_COLUMNS = {
    "text",
    "sequence_text",
    "merged_section_text",
    "section_text",
    "chunk_text",
    "subject_id",
    "hadm_id",
    "note_id",
}


@dataclass(frozen=True)
class DaptBlockRecord:
    block_id: str
    sequence_row_index: int
    patient_index: int
    block_index_within_patient: int
    token_start: int
    token_end: int
    block_length: int
    is_final_block_for_patient: bool
    max_length: int
    split: str = "train"
    source: str = "dapt_train_sequences"
    subject_id_hash: str | None = None
    sequence_token_count: int = 0
    sequence_block_count: int = 0
    dropped_tokens_for_sequence: int = 0

    def to_metadata_row(self) -> dict[str, object]:
        row = asdict(self)
        if row["subject_id_hash"] is None:
            row.pop("subject_id_hash")
        return row


def _coerce_token_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    elif hasattr(tokenized, "keys") and "input_ids" in tokenized.keys():
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if tokenized and isinstance(tokenized[0], list):
        if len(tokenized) != 1:
            raise ValueError("Expected a single tokenized sequence")
        tokenized = tokenized[0]
    return [int(token) for token in tokenized]


def tokenize_patient_sequence(tokenizer: Any, sequence_text: str) -> list[int]:
    """Tokenize one patient sequence without truncation or added model tokens."""
    tokenized = tokenizer(str(sequence_text), add_special_tokens=False, truncation=False)
    return _coerce_token_ids(tokenized)


def split_tokens_into_patient_blocks(
    token_ids: list[int],
    *,
    sequence_row_index: int,
    patient_index: int,
    max_length: int,
    subject_id_hash: str | None = None,
    min_block_tokens: int = 1,
    drop_short_blocks: bool = False,
    split: str = "train",
    source: str = "dapt_train_sequences",
) -> tuple[list[DaptBlockRecord], int]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if min_block_tokens <= 0:
        raise ValueError("min_block_tokens must be positive")
    if min_block_tokens > max_length:
        raise ValueError("min_block_tokens must be <= max_length")

    token_count = len(token_ids)
    if token_count == 0:
        return [], 0

    kept_spans: list[tuple[int, int]] = []
    dropped_tokens = 0
    for start in range(0, token_count, max_length):
        end = min(start + max_length, token_count)
        block_length = end - start
        if drop_short_blocks and block_length < min_block_tokens:
            dropped_tokens += block_length
            continue
        kept_spans.append((start, end))

    records: list[DaptBlockRecord] = []
    sequence_block_count = len(kept_spans)
    for block_index, (start, end) in enumerate(kept_spans):
        records.append(
            DaptBlockRecord(
                block_id=f"seq{sequence_row_index:08d}_block{block_index:06d}",
                sequence_row_index=int(sequence_row_index),
                patient_index=int(patient_index),
                subject_id_hash=subject_id_hash,
                block_index_within_patient=int(block_index),
                token_start=int(start),
                token_end=int(end),
                block_length=int(end - start),
                is_final_block_for_patient=block_index == sequence_block_count - 1,
                max_length=int(max_length),
                split=split,
                source=source,
                sequence_token_count=int(token_count),
                sequence_block_count=int(sequence_block_count),
                dropped_tokens_for_sequence=int(dropped_tokens),
            )
        )
    return records, dropped_tokens


def validate_train_only_sequences(sequences: pd.DataFrame) -> None:
    if "split" in sequences.columns:
        bad = sequences.loc[sequences["split"].astype(str) != "train", "split"].unique()
        if len(bad):
            raise ValueError(f"DAPT sequence input contains non-train split values: {sorted(map(str, bad))}")


def normalize_sequence_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "block_id",
            "sequence_row_index",
            "patient_index",
            "block_index_within_patient",
            "token_start",
            "token_end",
            "block_length",
            "is_final_block_for_patient",
            "max_length",
            "split",
            "source",
            "sequence_token_count",
            "sequence_block_count",
            "dropped_tokens_for_sequence",
        ]
    )


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=float)))


def build_dapt_block_manifest(
    sequences: pd.DataFrame,
    tokenizer: Any,
    *,
    max_length: int,
    min_block_tokens: int = 1,
    drop_short_blocks: bool = False,
    source: str = "dapt_train_sequences",
    keep_token_blocks: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[list[int]]]:
    """Tokenize train-only patient sequences and build raw-text-free block metadata."""
    if "sequence_text" not in sequences.columns:
        raise ValueError("DAPT sequences missing required column: sequence_text")
    validate_train_only_sequences(sequences)

    records: list[DaptBlockRecord] = []
    token_blocks: list[list[int]] = []
    patient_rows: list[dict[str, object]] = []
    total_dropped = 0

    for patient_index, (_, row) in enumerate(sequences.reset_index(drop=True).iterrows()):
        sequence_row_index = int(patient_index)
        subject_id_hash = str(row["subject_id_hash"]) if "subject_id_hash" in sequences.columns and pd.notna(row["subject_id_hash"]) else None
        token_ids = tokenize_patient_sequence(tokenizer, normalize_sequence_text(row["sequence_text"]))
        block_records, dropped_tokens = split_tokens_into_patient_blocks(
            token_ids,
            sequence_row_index=sequence_row_index,
            patient_index=patient_index,
            subject_id_hash=subject_id_hash,
            max_length=max_length,
            min_block_tokens=min_block_tokens,
            drop_short_blocks=drop_short_blocks,
            split=str(row["split"]) if "split" in sequences.columns else "train",
            source=source,
        )
        total_dropped += dropped_tokens
        for record in block_records:
            records.append(record)
            if keep_token_blocks:
                token_blocks.append(token_ids[record.token_start : record.token_end])
        patient_row = {
            "sequence_row_index": sequence_row_index,
            "patient_index": int(patient_index),
            "split": str(row["split"]) if "split" in sequences.columns else "train",
            "token_count": int(len(token_ids)),
            "block_count": int(len(block_records)),
            "dropped_tokens": int(dropped_tokens),
            "max_block_length": int(max((record.block_length for record in block_records), default=0)),
            "has_blocks": bool(block_records),
            "source": source,
        }
        if subject_id_hash is not None:
            patient_row["subject_id_hash"] = subject_id_hash
        patient_rows.append(patient_row)

    manifest = pd.DataFrame([record.to_metadata_row() for record in records]) if records else _empty_manifest()
    patient_counts = pd.DataFrame(patient_rows)
    if not patient_counts.empty and "subject_id" in patient_counts.columns:
        patient_counts = patient_counts.drop(columns=["subject_id"])

    summary = make_block_packing_summary(
        patient_counts,
        manifest,
        max_length=max_length,
        min_block_tokens=min_block_tokens,
        drop_short_blocks=drop_short_blocks,
        dropped_tokens=total_dropped,
    )
    histogram = make_block_length_histogram(manifest)
    assert_no_raw_columns(manifest, "dapt_block_manifest")
    assert_no_raw_columns(patient_counts, "dapt_patient_block_counts")
    assert_no_raw_columns(summary, "dapt_block_packing_summary")
    assert_no_raw_columns(histogram, "dapt_block_length_histogram")
    return manifest, summary, histogram, patient_counts, token_blocks


def make_block_packing_summary(
    patient_counts: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    max_length: int,
    min_block_tokens: int,
    drop_short_blocks: bool,
    dropped_tokens: int,
) -> pd.DataFrame:
    token_counts = patient_counts["token_count"].astype(int).tolist() if "token_count" in patient_counts else []
    block_counts = patient_counts["block_count"].astype(int).tolist() if "block_count" in patient_counts else []
    block_lengths = manifest["block_length"].astype(int).tolist() if "block_length" in manifest else []
    n_sequences = len(patient_counts)
    n_empty = int((patient_counts["token_count"].astype(int) == 0).sum()) if "token_count" in patient_counts else 0
    short_final_blocks = 0
    if not manifest.empty:
        short_final_blocks = int(((manifest["is_final_block_for_patient"].astype(bool)) & (manifest["block_length"].astype(int) < max_length)).sum())
    row = {
        "n_sequences": int(n_sequences),
        "n_nonempty_sequences": int(n_sequences - n_empty),
        "n_empty_sequences": int(n_empty),
        "total_tokens": int(sum(token_counts)),
        "total_blocks": int(len(manifest)),
        "mean_tokens_per_sequence": _mean(token_counts),
        "median_tokens_per_sequence": _median(token_counts),
        "p90_tokens_per_sequence": _percentile(token_counts, 90),
        "p95_tokens_per_sequence": _percentile(token_counts, 95),
        "max_tokens_per_sequence": int(max(token_counts, default=0)),
        "mean_blocks_per_sequence": _mean(block_counts),
        "max_blocks_per_sequence": int(max(block_counts, default=0)),
        "mean_block_length": _mean(block_lengths),
        "median_block_length": _median(block_lengths),
        "p95_block_length": _percentile(block_lengths, 95),
        "max_block_length": int(max(block_lengths, default=0)),
        "short_final_blocks_count": int(short_final_blocks),
        "dropped_tokens": int(dropped_tokens),
        "cross_patient_mixing": False,
        "full_dapt_block_packing_implemented": True,
        "training_launched": False,
        "max_length": int(max_length),
        "min_block_tokens": int(min_block_tokens),
        "drop_short_blocks": bool(drop_short_blocks),
    }
    return pd.DataFrame([row])


def make_block_length_histogram(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(columns=["block_length", "count"])
    return (
        manifest["block_length"]
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("block_length")
        .reset_index(name="count")
    )


def assert_no_raw_columns(frame: pd.DataFrame, name: str) -> None:
    forbidden = RAW_TEXT_COLUMNS & set(frame.columns)
    if forbidden:
        raise ValueError(f"{name} contains raw/restricted columns: {sorted(forbidden)}")


def validate_dapt_block_manifest(
    manifest: pd.DataFrame,
    summary: pd.DataFrame | None = None,
    patient_counts: pd.DataFrame | None = None,
    *,
    max_length: int,
) -> None:
    assert_no_raw_columns(manifest, "dapt_block_manifest")
    if summary is not None:
        assert_no_raw_columns(summary, "dapt_block_packing_summary")
    if patient_counts is not None:
        assert_no_raw_columns(patient_counts, "dapt_patient_block_counts")
    if manifest.empty:
        if summary is not None and int(summary.iloc[0].get("total_blocks", 0)) != 0:
            raise ValueError("Summary reports blocks but manifest is empty")
        return

    required = {
        "block_id",
        "sequence_row_index",
        "patient_index",
        "block_index_within_patient",
        "token_start",
        "token_end",
        "block_length",
        "max_length",
        "split",
        "sequence_token_count",
        "dropped_tokens_for_sequence",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"DAPT block manifest missing columns: {sorted(missing)}")
    if manifest["block_id"].duplicated().any():
        raise ValueError("DAPT block manifest has duplicate block_id values")
    bad_splits = set(manifest["split"].astype(str)) - {"train"}
    if bad_splits:
        raise ValueError(f"DAPT block manifest contains non-train split values: {sorted(bad_splits)}")
    if (manifest["block_length"].astype(int) > max_length).any():
        raise ValueError("DAPT block manifest contains a block longer than max_length")
    if (manifest["block_length"].astype(int) <= 0).any():
        raise ValueError("DAPT block manifest contains a non-positive block length")
    if not ((manifest["token_end"].astype(int) - manifest["token_start"].astype(int)) == manifest["block_length"].astype(int)).all():
        raise ValueError("DAPT block manifest has token span/block_length mismatch")

    for _, group in manifest.sort_values(["sequence_row_index", "token_start"]).groupby("sequence_row_index", sort=False):
        starts = group["token_start"].astype(int).tolist()
        ends = group["token_end"].astype(int).tolist()
        if starts and starts[0] != 0:
            raise ValueError("First block for a patient does not start at token 0")
        if starts[1:] != ends[:-1]:
            raise ValueError("Token spans are not contiguous within a patient")
        sequence_token_count = int(group["sequence_token_count"].iloc[0])
        dropped = int(group["dropped_tokens_for_sequence"].iloc[0])
        if sum(group["block_length"].astype(int)) + dropped != sequence_token_count:
            raise ValueError("Covered tokens plus dropped tokens do not match sequence token count")

    if summary is not None and not summary.empty:
        row = summary.iloc[0]
        if int(row["total_blocks"]) != len(manifest):
            raise ValueError("Summary total_blocks does not match manifest rows")
        if int(row["max_block_length"]) > max_length:
            raise ValueError("Summary max_block_length exceeds max_length")
        if bool(row.get("cross_patient_mixing", False)):
            raise ValueError("Summary reports cross-patient mixing")
    if patient_counts is not None and not patient_counts.empty:
        if "split" in patient_counts and (patient_counts["split"].astype(str) != "train").any():
            raise ValueError("Patient block counts contain non-train split values")
        nonempty = patient_counts[patient_counts["token_count"].astype(int) > 0]
        if (nonempty["block_count"].astype(int) <= 0).any():
            raise ValueError("A nonempty patient sequence has no DAPT block")
        total_tokens = int(patient_counts["token_count"].astype(int).sum())
        total_dropped = int(patient_counts["dropped_tokens"].astype(int).sum())
        if int(manifest["block_length"].astype(int).sum()) + total_dropped != total_tokens:
            raise ValueError("Manifest coverage does not match patient token counts")


class DaptBlockDataset:
    def __init__(self, token_blocks: list[list[int]]) -> None:
        self.token_blocks = [[int(token) for token in block] for block in token_blocks]

    def __len__(self) -> int:
        return len(self.token_blocks)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {"input_ids": self.token_blocks[index]}


class DaptBlockIterableDataset:
    def __init__(self, token_blocks: list[list[int]]) -> None:
        self.token_blocks = token_blocks

    def __iter__(self):
        for block in self.token_blocks:
            yield {"input_ids": [int(token) for token in block]}


def causal_lm_collate_fn(batch: list[dict[str, list[int]] | list[int]], pad_token_id: int):
    import torch

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    blocks: list[list[int]] = []
    for item in batch:
        if isinstance(item, dict):
            block = item["input_ids"]
        else:
            block = item
        blocks.append([int(token) for token in block])
    max_len = max(len(block) for block in blocks)
    if max_len <= 0:
        raise ValueError("Cannot collate only empty token blocks")

    input_rows = []
    attention_rows = []
    label_rows = []
    for block in blocks:
        pad_count = max_len - len(block)
        input_rows.append(block + [int(pad_token_id)] * pad_count)
        attention_rows.append([1] * len(block) + [0] * pad_count)
        label_rows.append(block + [-100] * pad_count)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }
