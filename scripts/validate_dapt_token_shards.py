from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.data.dapt_token_shards import validate_dapt_token_shards  # noqa: E402


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DAPT token block shard arrays and metadata.")
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--expected-total-blocks", type=int, default=None)
    parser.add_argument("--expected-total-tokens", type=int, default=None)
    args = parser.parse_args()

    shard_dir = resolve_path(args.shard_dir)
    manifest = pd.read_parquet(resolve_path(args.manifest)) if args.manifest else None
    summary = None
    if args.summary:
        summary = pd.read_csv(resolve_path(args.summary))
    else:
        summary_path = shard_dir / "dapt_token_shard_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
    metadata = validate_dapt_token_shards(
        shard_dir,
        max_length=args.max_length,
        manifest=manifest,
        summary=summary,
        expected_total_blocks=args.expected_total_blocks,
        expected_total_tokens=args.expected_total_tokens,
    )
    print("DAPT token shard validation passed.")
    print(f"shard_dir={shard_dir}")
    print(f"n_blocks={len(metadata)}")
    print(f"total_tokens={int(metadata['block_length'].astype(int).sum())}")
    print(f"max_block_length={int(metadata['block_length'].astype(int).max()) if len(metadata) else 0}")
    print(f"n_shards={int(metadata['shard_id'].nunique()) if len(metadata) else 0}")
    print("raw_text_free=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
