from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.data.dapt_blocks import (  # noqa: E402
    assert_no_raw_columns,
    build_dapt_block_manifest,
    validate_dapt_block_manifest,
)
from notesm_v2.models.special_tokens import add_project_special_tokens  # noqa: E402


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def prepare_tokenizer(tokenizer_id: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    special_token_result = add_project_special_tokens(tokenizer)
    return tokenizer, special_token_result


def validate_train_only_against_lampr_artifacts(benchmark_root: Path, sequences: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if "split" in sequences.columns:
        bad = sequences.loc[sequences["split"].astype(str) != "train", "split"].unique()
        if len(bad):
            raise ValueError(f"dapt_train_sequences.parquet contains non-train split values: {sorted(map(str, bad))}")
        notes.append("Input split column is train-only.")
    else:
        notes.append("Input has no split column; using frozen train-only DAPT sequence artifact.")

    sequence_hashes = set(sequences["subject_id_hash"].astype(str)) if "subject_id_hash" in sequences.columns else set()
    dapt_manifest_path = benchmark_root / "dapt_train_manifest.parquet"
    if sequence_hashes and dapt_manifest_path.exists():
        dapt_manifest = pd.read_parquet(dapt_manifest_path)
        if "subject_id_hash" in dapt_manifest.columns:
            manifest_hashes = set(dapt_manifest["subject_id_hash"].astype(str))
            if not sequence_hashes.issubset(manifest_hashes):
                raise ValueError("DAPT sequence subject hashes are not a subset of dapt_train_manifest subject hashes")
            notes.append("DAPT sequence subject hashes are contained in dapt_train_manifest.")

    no_text_manifest_path = benchmark_root / "benchmark_manifest_no_text.parquet"
    if sequence_hashes and no_text_manifest_path.exists():
        manifest = pd.read_parquet(no_text_manifest_path)
        required = {"subject_id_hash", "split"}
        if required.issubset(manifest.columns):
            val_test_hashes = set(manifest.loc[manifest["split"].isin(["val", "test"]), "subject_id_hash"].astype(str))
            leakage = sequence_hashes & val_test_hashes
            if leakage:
                raise ValueError(f"DAPT train sequences overlap val/test subject hashes: {len(leakage)}")
            notes.append("No DAPT sequence subject hash appears in frozen val/test benchmark subjects.")
        else:
            missing = sorted(required - set(manifest.columns))
            notes.append(f"Skipped val/test subject-hash validation because benchmark_manifest_no_text.parquet is missing columns: {missing}.")
    elif sequence_hashes:
        notes.append("Val/test subject-hash artifact unavailable; input is treated as frozen train-only DAPT sequence artifact.")
    else:
        notes.append("No subject_id_hash column available; input is treated as frozen train-only DAPT sequence artifact.")
    return notes


def write_token_shards(output_dir: Path, manifest: pd.DataFrame, token_blocks: list[list[int]], shard_size: int) -> int:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if len(manifest) != len(token_blocks):
        raise ValueError("Manifest rows and token blocks differ")
    shard_count = 0
    for start in range(0, len(token_blocks), shard_size):
        end = min(start + shard_size, len(token_blocks))
        shard_manifest = manifest.iloc[start:end].reset_index(drop=True)
        shard = pd.DataFrame(
            {
                "block_id": shard_manifest["block_id"].astype(str),
                "sequence_row_index": shard_manifest["sequence_row_index"].astype(int),
                "patient_index": shard_manifest["patient_index"].astype(int),
                "block_index_within_patient": shard_manifest["block_index_within_patient"].astype(int),
                "input_ids": token_blocks[start:end],
            }
        )
        assert_no_raw_columns(shard, f"dapt_block_token_shard_{shard_count:05d}")
        shard.to_parquet(output_dir / f"dapt_block_token_shard_{shard_count:05d}.parquet", index=False)
        shard_count += 1
    return shard_count


def write_report(
    output_dir: Path,
    *,
    benchmark_root: Path,
    tokenizer_id: str,
    summary: pd.DataFrame,
    train_only_notes: list[str],
    special_token_result: dict[str, object],
    write_block_token_shards: bool,
    shard_count: int,
) -> None:
    row = summary.iloc[0].to_dict()
    lines = [
        "# DAPT Block Packing Report",
        "",
        "This report intentionally excludes raw clinical note text.",
        "",
        "## Inputs",
        "",
        f"- Benchmark root: `{benchmark_root}`",
        "- Input artifact: `dapt_train_sequences.parquet`",
        "- Input is the frozen train-only DAPT sequence artifact generated by LAMP-R v0.2.",
        f"- Tokenizer: `{tokenizer_id}`",
        f"- Project special tokens added: {int(special_token_result.get('num_added', 0))}",
        "",
        "## Train-Only Validation",
        "",
    ]
    lines.extend([f"- {note}" for note in train_only_notes])
    lines.extend(
        [
            "",
            "## Packing Summary",
            "",
            f"- Sequences: {int(row['n_sequences'])}",
            f"- Nonempty sequences: {int(row['n_nonempty_sequences'])}",
            f"- Empty sequences: {int(row['n_empty_sequences'])}",
            f"- Total tokens: {int(row['total_tokens'])}",
            f"- Total blocks: {int(row['total_blocks'])}",
            f"- Mean blocks per sequence: {float(row['mean_blocks_per_sequence']):.4f}",
            f"- Max blocks per sequence: {int(row['max_blocks_per_sequence'])}",
            f"- Max block length: {int(row['max_block_length'])}",
            f"- Dropped tokens: {int(row['dropped_tokens'])}",
            f"- Cross-patient mixing: {bool(row['cross_patient_mixing'])}",
            f"- Full DAPT block packing implemented: {bool(row['full_dapt_block_packing_implemented'])}",
            f"- Training launched: {bool(row['training_launched'])}",
            "",
            "## Token Block Artifacts",
            "",
            f"- Token shards written: {bool(write_block_token_shards)}",
            f"- Token shard count: {int(shard_count)}",
            "- Token shards are restricted local artifacts when written.",
        ]
    )
    (output_dir / "DAPT_BLOCK_PACKING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build strict train-only DAPT causal LM block manifest without launching training.")
    parser.add_argument("--benchmark-root", default="results/lampr_full_v0_2")
    parser.add_argument("--tokenizer-id", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--output-dir", default="results/dapt_blocks/lampr_v0_2_mamba2_len4096")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--debug-sample", type=int, default=None)
    parser.add_argument("--min-block-tokens", type=int, default=1)
    parser.add_argument("--drop-short-blocks", action="store_true")
    parser.add_argument("--write-block-token-shards", action="store_true")
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0, help="Reserved for future parallel tokenization; currently runs serially.")
    args = parser.parse_args()

    benchmark_root = resolve_path(args.benchmark_root)
    sequences_path = benchmark_root / "dapt_train_sequences.parquet"
    if not sequences_path.exists():
        raise FileNotFoundError(f"Missing frozen train-only DAPT sequence artifact: {sequences_path}")

    sequences = pd.read_parquet(sequences_path)
    train_only_notes = validate_train_only_against_lampr_artifacts(benchmark_root, sequences)
    if args.debug_sample is not None:
        sequences = sequences.head(args.debug_sample).copy()

    tokenizer, special_token_result = prepare_tokenizer(args.tokenizer_id)
    keep_token_blocks = bool(args.write_block_token_shards or args.debug_sample is not None)
    if args.write_block_token_shards and args.debug_sample is None:
        print(
            "WARNING: Full token shard writing currently materializes token blocks in memory and is intended only "
            "for small/debug runs until streaming shard writing is implemented."
        )
    manifest, summary, histogram, patient_counts, token_blocks = build_dapt_block_manifest(
        sequences,
        tokenizer,
        max_length=args.max_length,
        min_block_tokens=args.min_block_tokens,
        drop_short_blocks=args.drop_short_blocks,
        source="dapt_train_sequences",
        keep_token_blocks=keep_token_blocks,
    )
    validate_dapt_block_manifest(manifest, summary, patient_counts, max_length=args.max_length)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_dir / "dapt_block_manifest.parquet", index=False)
    summary.to_csv(output_dir / "dapt_block_packing_summary.csv", index=False)
    histogram.to_csv(output_dir / "dapt_block_length_histogram.csv", index=False)
    patient_counts.to_csv(output_dir / "dapt_patient_block_counts.csv", index=False)
    shard_count = 0
    if args.write_block_token_shards:
        shard_count = write_token_shards(output_dir, manifest, token_blocks, args.shard_size)
    write_report(
        output_dir,
        benchmark_root=benchmark_root,
        tokenizer_id=args.tokenizer_id,
        summary=summary,
        train_only_notes=train_only_notes,
        special_token_result=special_token_result,
        write_block_token_shards=args.write_block_token_shards,
        shard_count=shard_count,
    )

    row = summary.iloc[0]
    print(f"Wrote DAPT block packing outputs to {output_dir}")
    print(
        summary[
            [
                "n_sequences",
                "total_tokens",
                "total_blocks",
                "mean_blocks_per_sequence",
                "max_blocks_per_sequence",
                "max_block_length",
                "dropped_tokens",
                "cross_patient_mixing",
                "full_dapt_block_packing_implemented",
                "training_launched",
            ]
        ].to_string(index=False)
    )
    if int(row["dropped_tokens"]) == 0:
        print("Full token coverage check passed: dropped_tokens=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
