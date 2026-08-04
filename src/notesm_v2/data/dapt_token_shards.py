from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from notesm_v2.data.dapt_blocks import (
    assert_no_raw_columns,
    make_block_packing_summary,
    normalize_sequence_text,
    split_tokens_into_patient_blocks,
    tokenize_patient_sequence,
    validate_train_only_sequences,
)


@dataclass(frozen=True)
class TokenShardSpec:
    output_dir: Path
    max_length: int
    shard_size: int
    pad_token_id: int
    min_block_tokens: int = 1
    drop_short_blocks: bool = False
    source: str = "dapt_train_sequences"


SHARD_METADATA_COLUMNS = [
    "block_id",
    "global_block_index",
    "shard_id",
    "row_index_in_shard",
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


def shard_array_path(shard_dir: Path, shard_id: int) -> Path:
    return shard_dir / f"dapt_token_blocks_shard_{shard_id:05d}.npy"


def shard_metadata_path(shard_dir: Path, shard_id: int) -> Path:
    return shard_dir / f"dapt_token_blocks_shard_{shard_id:05d}_metadata.parquet"


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _flush_shard(
    *,
    output_dir: Path,
    shard_id: int,
    token_buffer: list[list[int]],
    metadata_buffer: list[dict[str, object]],
    max_length: int,
    pad_token_id: int,
    dry_run: bool,
) -> dict[str, object]:
    if not token_buffer:
        raise ValueError("Cannot flush an empty DAPT token shard")
    array = np.full((len(token_buffer), max_length), int(pad_token_id), dtype=np.int32)
    for row_index, block in enumerate(token_buffer):
        block_length = len(block)
        if block_length > max_length:
            raise ValueError("Token block longer than max_length during shard flush")
        array[row_index, :block_length] = np.asarray(block, dtype=np.int32)
        metadata_buffer[row_index]["row_index_in_shard"] = int(row_index)
    metadata = pd.DataFrame(metadata_buffer)
    assert_no_raw_columns(metadata, f"dapt_token_blocks_shard_{shard_id:05d}_metadata")
    if not dry_run:
        np.save(shard_array_path(output_dir, shard_id), array)
        metadata.to_parquet(shard_metadata_path(output_dir, shard_id), index=False)
    return {
        "shard_id": int(shard_id),
        "rows": int(len(token_buffer)),
        "max_length": int(max_length),
        "pad_token_id": int(pad_token_id),
        "npy_path": shard_array_path(output_dir, shard_id).name,
        "metadata_path": shard_metadata_path(output_dir, shard_id).name,
        "planned_only": bool(dry_run),
        "files_written": bool(not dry_run),
    }


def write_dapt_token_shards_streaming(
    sequences: pd.DataFrame,
    tokenizer: Any,
    spec: TokenShardSpec,
    *,
    max_shards: int | None = None,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write padded token block shards while keeping only one shard buffer in memory."""
    if "sequence_text" not in sequences.columns:
        raise ValueError("DAPT sequences missing required column: sequence_text")
    if spec.max_length <= 0:
        raise ValueError("max_length must be positive")
    if spec.shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if spec.min_block_tokens > spec.max_length:
        raise ValueError("min_block_tokens must be <= max_length")
    validate_train_only_sequences(sequences)

    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_buffer: list[list[int]] = []
    metadata_buffer: list[dict[str, object]] = []
    shard_rows: list[dict[str, object]] = []
    all_metadata_rows: list[dict[str, object]] = []
    patient_rows: list[dict[str, object]] = []
    shard_id = 0
    global_block_index = 0
    stopped_by_max_shards = False

    for patient_index, (_, row) in enumerate(sequences.reset_index(drop=True).iterrows()):
        if max_shards is not None and shard_id >= max_shards:
            stopped_by_max_shards = True
            break
        sequence_row_index = int(patient_index)
        subject_id_hash = str(row["subject_id_hash"]) if "subject_id_hash" in sequences.columns and pd.notna(row["subject_id_hash"]) else None
        token_ids = tokenize_patient_sequence(tokenizer, normalize_sequence_text(row["sequence_text"]))
        block_records, dropped_tokens = split_tokens_into_patient_blocks(
            token_ids,
            sequence_row_index=sequence_row_index,
            patient_index=patient_index,
            subject_id_hash=subject_id_hash,
            max_length=spec.max_length,
            min_block_tokens=spec.min_block_tokens,
            drop_short_blocks=spec.drop_short_blocks,
            split=str(row["split"]) if "split" in sequences.columns else "train",
            source=spec.source,
        )
        patient_rows.append(
            {
                "sequence_row_index": sequence_row_index,
                "patient_index": int(patient_index),
                "split": str(row["split"]) if "split" in sequences.columns else "train",
                "token_count": int(len(token_ids)),
                "block_count": int(len(block_records)),
                "dropped_tokens": int(dropped_tokens),
                "max_block_length": int(max((record.block_length for record in block_records), default=0)),
                "has_blocks": bool(block_records),
                "source": spec.source,
                **({"subject_id_hash": subject_id_hash} if subject_id_hash is not None else {}),
            }
        )
        for record in block_records:
            if max_shards is not None and shard_id >= max_shards:
                stopped_by_max_shards = True
                break
            block = token_ids[record.token_start : record.token_end]
            metadata_row = record.to_metadata_row()
            metadata_row.update(
                {
                    "global_block_index": int(global_block_index),
                    "shard_id": int(shard_id),
                    "row_index_in_shard": int(len(token_buffer)),
                }
            )
            ordered_row = {column: metadata_row[column] for column in SHARD_METADATA_COLUMNS if column in metadata_row}
            if "subject_id_hash" in metadata_row:
                ordered_row["subject_id_hash"] = metadata_row["subject_id_hash"]
            token_buffer.append(block)
            metadata_buffer.append(ordered_row)
            all_metadata_rows.append(ordered_row.copy())
            global_block_index += 1
            if len(token_buffer) >= spec.shard_size:
                shard_rows.append(
                    _flush_shard(
                        output_dir=output_dir,
                        shard_id=shard_id,
                        token_buffer=token_buffer,
                        metadata_buffer=metadata_buffer,
                        max_length=spec.max_length,
                        pad_token_id=spec.pad_token_id,
                        dry_run=dry_run,
                    )
                )
                shard_id += 1
                token_buffer = []
                metadata_buffer = []
        if stopped_by_max_shards:
            break

    if token_buffer and not (max_shards is not None and shard_id >= max_shards):
        shard_rows.append(
            _flush_shard(
                output_dir=output_dir,
                shard_id=shard_id,
                token_buffer=token_buffer,
                metadata_buffer=metadata_buffer,
                max_length=spec.max_length,
                pad_token_id=spec.pad_token_id,
                dry_run=dry_run,
            )
        )
        shard_id += 1

    metadata = pd.DataFrame(all_metadata_rows)
    patient_counts = pd.DataFrame(patient_rows)
    summary = make_block_packing_summary(
        patient_counts,
        metadata,
        max_length=spec.max_length,
        min_block_tokens=spec.min_block_tokens,
        drop_short_blocks=spec.drop_short_blocks,
        dropped_tokens=int(patient_counts["dropped_tokens"].sum()) if not patient_counts.empty else 0,
    )
    summary = summary.assign(
        total_shards=int(len(shard_rows)),
        shard_size=int(spec.shard_size),
        pad_token_id=int(spec.pad_token_id),
        token_shards_written=bool(not dry_run),
        raw_text_free=True,
        dry_run=bool(dry_run),
        stopped_by_max_shards=bool(stopped_by_max_shards),
        partial_shard_write=bool(stopped_by_max_shards),
    )
    shard_summary = pd.DataFrame(shard_rows)
    shard_summary = pd.concat([summary, shard_summary], axis=1) if shard_summary.empty else summary
    assert_no_raw_columns(summary, "dapt_token_shard_summary")
    return summary, pd.DataFrame(shard_rows)


def read_token_shard_metadata(shard_dir: str | Path) -> pd.DataFrame:
    root = Path(shard_dir)
    frames = []
    for path in sorted(root.glob("dapt_token_blocks_shard_*_metadata.parquet")):
        frame = pd.read_parquet(path)
        assert_no_raw_columns(frame, path.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[*SHARD_METADATA_COLUMNS, "subject_id_hash"])
    metadata = pd.concat(frames, ignore_index=True)
    if "global_block_index" in metadata.columns:
        metadata = metadata.sort_values("global_block_index").reset_index(drop=True)
    return metadata


def validate_dapt_token_shards(
    shard_dir: str | Path,
    *,
    max_length: int,
    manifest: pd.DataFrame | None = None,
    summary: pd.DataFrame | None = None,
    expected_total_blocks: int | None = None,
    expected_total_tokens: int | None = None,
    pad_token_id: int | None = None,
) -> pd.DataFrame:
    root = Path(shard_dir)
    shard_paths = sorted(root.glob("dapt_token_blocks_shard_*.npy"))
    if not shard_paths:
        raise ValueError(f"No token shard .npy files found in {root}")
    if summary is not None and not summary.empty and pad_token_id is None and "pad_token_id" in summary.columns:
        pad_token_id = int(summary.iloc[0]["pad_token_id"])

    metadata_frames = []
    total_rows = 0
    total_true_tokens = 0
    for path in shard_paths:
        shard_id = int(path.stem.split("_")[-1])
        metadata_path = shard_metadata_path(root, shard_id)
        if not metadata_path.exists():
            raise ValueError(f"Missing metadata for shard {shard_id}: {metadata_path}")
        array = np.load(path)
        if array.ndim != 2:
            raise ValueError(f"{path} rank must be 2")
        if array.shape[1] != max_length:
            raise ValueError(f"{path} second dimension {array.shape[1]} != max_length {max_length}")
        if array.dtype not in (np.dtype("int32"), np.dtype("int64")):
            raise ValueError(f"{path} dtype must be int32 or int64, found {array.dtype}")
        metadata = pd.read_parquet(metadata_path)
        assert_no_raw_columns(metadata, metadata_path.name)
        if len(metadata) != array.shape[0]:
            raise ValueError(f"{metadata_path} rows do not match {path} rows")
        if "shard_id" not in metadata or not (metadata["shard_id"].astype(int) == shard_id).all():
            raise ValueError(f"{metadata_path} shard_id column mismatch")
        expected_rows = np.arange(len(metadata), dtype=int)
        if "row_index_in_shard" not in metadata or not np.array_equal(metadata["row_index_in_shard"].astype(int).to_numpy(), expected_rows):
            raise ValueError(f"{metadata_path} row_index_in_shard is not 0..N-1")
        if (metadata["block_length"].astype(int) > max_length).any():
            raise ValueError(f"{metadata_path} contains block_length > max_length")
        if (metadata["block_length"].astype(int) <= 0).any():
            raise ValueError(f"{metadata_path} contains non-positive block_length")
        if pad_token_id is not None:
            lengths = metadata["block_length"].astype(int).to_numpy()
            for row_index, length in enumerate(lengths):
                if length < max_length and not np.all(array[row_index, length:] == int(pad_token_id)):
                    raise ValueError(f"{path} row {row_index} has non-pad tokens beyond block_length")
        total_rows += len(metadata)
        total_true_tokens += int(metadata["block_length"].astype(int).sum())
        metadata_frames.append(metadata)

    metadata_all = pd.concat(metadata_frames, ignore_index=True)
    if metadata_all["block_id"].duplicated().any():
        raise ValueError("Duplicate block_id values across token shards")
    if metadata_all["global_block_index"].duplicated().any():
        raise ValueError("Duplicate global_block_index values across token shards")
    global_indices = metadata_all.sort_values("global_block_index")["global_block_index"].astype(int).to_numpy()
    if not np.array_equal(global_indices, np.arange(len(global_indices), dtype=int)):
        raise ValueError("global_block_index values are not monotonic 0..N-1")
    bad_splits = set(metadata_all["split"].astype(str)) - {"train"}
    if bad_splits:
        raise ValueError(f"Token shard metadata contains non-train split values: {sorted(bad_splits)}")
    if expected_total_blocks is not None and total_rows != expected_total_blocks:
        raise ValueError(f"Token shard rows {total_rows} != expected_total_blocks {expected_total_blocks}")
    if expected_total_tokens is not None and total_true_tokens != expected_total_tokens:
        raise ValueError(f"Token shard true tokens {total_true_tokens} != expected_total_tokens {expected_total_tokens}")

    if manifest is not None:
        assert_no_raw_columns(manifest, "dapt_block_manifest")
        if set(metadata_all["block_id"]) != set(manifest["block_id"]):
            raise ValueError("Token shard block_id set does not match report-only manifest")
        compare_cols = [
            "block_id",
            "block_length",
            "sequence_row_index",
            "patient_index",
            "block_index_within_patient",
            "token_start",
            "token_end",
        ]
        merged = metadata_all[compare_cols].merge(manifest[compare_cols], on="block_id", suffixes=("_token", "_manifest"), validate="one_to_one")
        if len(merged) != len(manifest):
            raise ValueError("Token shard metadata cannot be one-to-one aligned with manifest")
        for col in compare_cols[1:]:
            if not (merged[f"{col}_token"].astype(int).to_numpy() == merged[f"{col}_manifest"].astype(int).to_numpy()).all():
                raise ValueError(f"Token shard metadata mismatch with manifest column: {col}")

    if summary is not None and not summary.empty:
        row = summary.iloc[0]
        partial_shard_write = _as_bool(row.get("partial_shard_write", False)) or _as_bool(row.get("stopped_by_max_shards", False))
        if "total_blocks" in summary and not partial_shard_write and int(row["total_blocks"]) != total_rows:
            raise ValueError("Token shard rows do not match summary total_blocks")
        if "total_tokens" in summary and not partial_shard_write and int(row["total_tokens"]) != total_true_tokens:
            raise ValueError("Token shard true token lengths do not match summary total_tokens")
        if "dropped_tokens" in summary and int(row["dropped_tokens"]) != 0:
            raise ValueError("Token shard summary reports dropped tokens")
        if _as_bool(row.get("cross_patient_mixing", False)):
            raise ValueError("Token shard summary reports cross-patient mixing")
    return metadata_all
