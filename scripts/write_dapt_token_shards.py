from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.data.dapt_token_shards import (  # noqa: E402
    TokenShardSpec,
    validate_dapt_token_shards,
    write_dapt_token_shards_streaming,
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
    add_project_special_tokens(tokenizer)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id after preparation")
    return tokenizer


def clean_outputs(output_dir: Path) -> None:
    patterns = [
        "dapt_token_blocks_shard_*.npy",
        "dapt_token_blocks_shard_*_metadata.parquet",
        "dapt_token_shard_summary.csv",
        "dapt_token_shard_index.csv",
        "DAPT_TOKEN_SHARD_REPORT.md",
    ]
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def assert_can_write(output_dir: Path, overwrite: bool) -> None:
    existing = list(output_dir.glob("dapt_token_blocks_shard_*.npy")) + list(output_dir.glob("dapt_token_blocks_shard_*_metadata.parquet"))
    if existing and not overwrite:
        raise FileExistsError(f"{output_dir} already contains DAPT token shards; pass --overwrite to replace script outputs")
    if overwrite:
        clean_outputs(output_dir)


def write_report(output_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, shard_index: pd.DataFrame) -> None:
    row = summary.iloc[0].to_dict()
    lines = [
        "# DAPT Token Shard Report",
        "",
        "This report intentionally excludes raw clinical note text.",
        "",
        "## Inputs",
        "",
        f"- Benchmark root: `{args.benchmark_root}`",
        "- Input artifact: `dapt_train_sequences.parquet`",
        f"- Tokenizer: `{args.tokenizer_id}`",
        "",
        "## Shard Summary",
        "",
        f"- Sequences processed: {int(row['n_sequences'])}",
        f"- Total tokens: {int(row['total_tokens'])}",
        f"- Total blocks: {int(row['total_blocks'])}",
        f"- Total shards: {int(row['total_shards'])}",
        f"- Shard size: {int(row['shard_size'])}",
        f"- Max length: {int(row['max_length'])}",
        f"- Max block length: {int(row['max_block_length'])}",
        f"- Dropped tokens: {int(row['dropped_tokens'])}",
        f"- Cross-patient mixing: {bool(row['cross_patient_mixing'])}",
        f"- Full DAPT block packing implemented: {bool(row['full_dapt_block_packing_implemented'])}",
        f"- Token shards written: {bool(row['token_shards_written'])}",
        f"- Training launched: {bool(row['training_launched'])}",
        f"- Raw text free: {bool(row['raw_text_free'])}",
        f"- Partial shard write: {bool(row.get('partial_shard_write', False))}",
        f"- Manifest supplied: {bool(args.manifest)}",
        "",
        "Full token shard generation should pass `--manifest results/dapt_blocks/lampr_v0_2_mamba_len4096_report_only/dapt_block_manifest.parquet` so shards can be validated against the report-only manifest.",
        "",
        "## Files",
        "",
        "- `dapt_token_blocks_shard_*.npy`",
        "- `dapt_token_blocks_shard_*_metadata.parquet`",
        "- `dapt_token_shard_summary.csv`",
        "- `dapt_token_shard_index.csv`",
    ]
    if not shard_index.empty:
        lines.extend(["", f"- First shard rows: {int(shard_index.iloc[0]['rows'])}", f"- Last shard rows: {int(shard_index.iloc[-1]['rows'])}"])
    (output_dir / "DAPT_TOKEN_SHARD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write strict train-only DAPT token block shards.")
    parser.add_argument("--benchmark-root", default="results/lampr_full_v0_2")
    parser.add_argument("--tokenizer-id", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default="results/dapt_token_shards/lampr_v0_2_mamba_len4096")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--debug-sample", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--min-block-tokens", type=int, default=1)
    parser.add_argument("--drop-short-blocks", action="store_true")
    parser.add_argument("--allow-drop-short-blocks-debug", action="store_true")
    args = parser.parse_args()
    if args.drop_short_blocks and not args.allow_drop_short_blocks_debug:
        raise ValueError("--drop-short-blocks is disabled for training shard generation; pass --allow-drop-short-blocks-debug only for explicit debug experiments")

    benchmark_root = resolve_path(args.benchmark_root)
    sequences_path = benchmark_root / "dapt_train_sequences.parquet"
    if not sequences_path.exists():
        raise FileNotFoundError(f"Missing train-only DAPT sequence artifact: {sequences_path}")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_can_write(output_dir, args.overwrite)

    sequences = pd.read_parquet(sequences_path)
    if args.debug_sample is not None:
        sequences = sequences.head(args.debug_sample).copy()
    tokenizer = prepare_tokenizer(args.tokenizer_id)
    spec = TokenShardSpec(
        output_dir=output_dir,
        max_length=args.max_length,
        shard_size=args.shard_size,
        pad_token_id=int(tokenizer.pad_token_id),
        min_block_tokens=args.min_block_tokens,
        drop_short_blocks=args.drop_short_blocks,
    )
    summary, shard_index = write_dapt_token_shards_streaming(
        sequences,
        tokenizer,
        spec,
        max_shards=args.max_shards,
        dry_run=args.dry_run,
    )
    summary.to_csv(output_dir / "dapt_token_shard_summary.csv", index=False)
    shard_index.to_csv(output_dir / "dapt_token_shard_index.csv", index=False)
    write_report(output_dir, args, summary, shard_index)

    if not args.dry_run:
        manifest = pd.read_parquet(resolve_path(args.manifest)) if args.manifest else None
        validate_dapt_token_shards(output_dir, max_length=args.max_length, manifest=manifest, summary=summary)

    print(f"Wrote DAPT token shard outputs to {output_dir}")
    print(
        summary[
            [
                "n_sequences",
                "total_tokens",
                "total_blocks",
                "total_shards",
                "shard_size",
                "max_length",
                "max_block_length",
                "dropped_tokens",
                "cross_patient_mixing",
                "token_shards_written",
                "training_launched",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
