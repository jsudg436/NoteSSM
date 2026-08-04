from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.data.dapt_blocks import validate_dapt_block_manifest  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Validate strict train-only DAPT block manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--patient-counts", default=None)
    parser.add_argument("--max-length", type=int, required=True)
    args = parser.parse_args()

    manifest_path = resolve_path(args.manifest)
    summary_path = resolve_path(args.summary)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing DAPT block manifest: {manifest_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing DAPT block packing summary: {summary_path}")

    patient_counts_path = resolve_path(args.patient_counts) if args.patient_counts else manifest_path.parent / "dapt_patient_block_counts.csv"
    manifest = pd.read_parquet(manifest_path)
    summary = pd.read_csv(summary_path)
    patient_counts = pd.read_csv(patient_counts_path) if patient_counts_path.exists() else None
    validate_dapt_block_manifest(manifest, summary, patient_counts, max_length=args.max_length)

    row = summary.iloc[0]
    print("DAPT block manifest validation passed.")
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")
    print(f"patient_counts={patient_counts_path if patient_counts_path.exists() else 'not_found'}")
    print(f"total_blocks={int(row['total_blocks'])}")
    print(f"total_tokens={int(row['total_tokens'])}")
    print(f"max_block_length={int(row['max_block_length'])}")
    print(f"dropped_tokens={int(row['dropped_tokens'])}")
    print(f"cross_patient_mixing={bool(row['cross_patient_mixing'])}")
    print(f"training_launched={bool(row['training_launched'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
