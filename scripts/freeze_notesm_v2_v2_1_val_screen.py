from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FORBIDDEN_COLUMNS = {"text", "sequence_text", "merged_section_text", "section_text", "chunk_text", "subject_id", "hadm_id", "note_id"}
METRICS = ["AUPRC", "AUROC", "Precision", "Recall", "F1", "Brier", "ECE", "threshold"]


def resolve_path(path_arg: str | Path) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if Path.cwd().resolve() == ROOT.resolve():
        return ROOT / path
    return ROOT / path


def check_text_free(path: Path, frame: pd.DataFrame) -> None:
    forbidden = sorted(FORBIDDEN_COLUMNS & set(frame.columns))
    if forbidden:
        raise ValueError(f"{path} contains forbidden raw/restricted columns: {forbidden}")


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required freeze input: {path}")
    frame = pd.read_csv(path)
    check_text_free(path, frame)
    return frame


def validate_per_seed(per_seed: pd.DataFrame, setting: str, seeds: list[int]) -> None:
    required = {"setting", "split", "seed", "n", "positives", *METRICS}
    missing = required - set(per_seed.columns)
    if missing:
        raise ValueError(f"Per-seed table missing columns: {sorted(missing)}")
    rows = per_seed[(per_seed["setting"].astype(str) == setting) & (per_seed["split"].astype(str) == "val")].copy()
    found_seeds = sorted(rows["seed"].astype(int).tolist())
    if found_seeds != sorted(seeds):
        raise ValueError(f"Per-seed table has seeds {found_seeds}, expected {sorted(seeds)}")
    if rows["n"].astype(int).nunique() != 1:
        raise ValueError("Per-seed validation n is not constant across seeds")
    if rows["positives"].astype(int).nunique() != 1:
        raise ValueError("Per-seed validation positives is not constant across seeds")


def validate_aggregate(aggregate: pd.DataFrame, setting: str, seeds: list[int]) -> pd.Series:
    required = {
        "setting",
        "split",
        "n_seeds",
        "seeds",
        "n_min",
        "n_max",
        "positives_min",
        "positives_max",
        *[f"{metric}_mean" for metric in METRICS],
        *[f"{metric}_std" for metric in METRICS],
    }
    missing = required - set(aggregate.columns)
    if missing:
        raise ValueError(f"Aggregate table missing columns: {sorted(missing)}")
    rows = aggregate[(aggregate["setting"].astype(str) == setting) & (aggregate["split"].astype(str) == "val")]
    if len(rows) != 1:
        raise ValueError(f"Expected one aggregate row for setting={setting!r}, split='val'; found {len(rows)}")
    row = rows.iloc[0]
    if int(row["n_seeds"]) != len(seeds):
        raise ValueError(f"Aggregate n_seeds={row['n_seeds']}, expected {len(seeds)}")
    aggregate_seeds = sorted(int(seed) for seed in str(row["seeds"]).split())
    if aggregate_seeds != sorted(seeds):
        raise ValueError(f"Aggregate seeds={aggregate_seeds}, expected {sorted(seeds)}")
    if int(row["n_min"]) != int(row["n_max"]):
        raise ValueError("Aggregate validation row count differs across seeds")
    if int(row["positives_min"]) != int(row["positives_max"]):
        raise ValueError("Aggregate positive count differs across seeds")
    return row


def metric_table(row: pd.Series) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        rows.append(
            {
                "metric": metric,
                "mean": float(row[f"{metric}_mean"]),
                "std": float(row[f"{metric}_std"]),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    output_dir: Path,
    aggregate_row: pd.Series,
    per_seed: pd.DataFrame,
    setting: str,
    seeds: list[int],
    timestamp: str,
) -> None:
    metrics = metric_table(aggregate_row)
    metrics_text = metrics.to_string(index=False)
    per_seed_cols = ["setting", "split", "seed", "n", "positives", "AUPRC", "AUROC", "Precision", "Recall", "F1", "Brier", "ECE", "threshold"]
    per_seed_text = per_seed[per_seed_cols].to_string(index=False)
    text = f"""# NoteSSM-v2.1 Three-View Concat Validation Lock

Freeze timestamp: {timestamp}

## Locked Result

- Benchmark: LAMP-R v0.2 frozen
- Candidate: NoteSSM-v2.1 `{setting}`
- Auxiliary loss weight: 0.0
- Split: validation only
- Seeds: {" ".join(str(seed) for seed in seeds)}
- Validation examples per seed: {int(aggregate_row["n_min"])}
- Validation positives per seed: {int(aggregate_row["positives_min"])}
- Input artifacts: existing frozen cached note/section/chunk embeddings
- No DAPT rerun
- No note/section/chunk embedding extraction rerun
- No test evaluation
- No class_weight or focal loss
- Raw clinical note text is not written to this freeze artifact

## Aggregate Metrics

```text
{metrics_text}
```

## Per-Seed Metrics

```text
{per_seed_text}
```

## Decision

Lock `three_view_concat` with `aux_loss_weight=0.0` as the selected NoteSSM-v2.1 validation-stage candidate. The validation signal is stable across seeds 17, 23, and 42, with low AUPRC/AUROC/F1 variability.

## Interpretation Constraints

- This is a validation-only architecture screen, not a final test result.
- The v2.1 design is a diagnostic-motivated follow-up after prior full-test analysis, so any later test result should be described as exploratory unless confirmed on a new locked holdout, LAMP-R v0.3, or an external cohort.
- Do not modify LAMP-R v0.2.
- Do not rerun full DAPT.
- Do not rerun section/chunk embedding extraction.
- Do not run class_weight for this locked validation result.
- Do not create the final paper-wide table from this artifact alone.
"""
    (output_dir / "NOTESSM_V2_V2_1_THREE_VIEW_CONCAT_VAL_LOCKED.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze NoteSSM-v2.1 three_view_concat validation-only 3-seed screen.")
    parser.add_argument("--aggregate-root", default="results/notesm_v2_v2_1_val_screen_3seed_concat")
    parser.add_argument("--output-dir", default="results/notesm_v2_v2_1_val_screen_3seed_concat_locked")
    parser.add_argument("--setting", default="three_view_concat")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 42])
    args = parser.parse_args()

    aggregate_root = resolve_path(args.aggregate_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = aggregate_root / "notesm_v2_v2_1_val_aggregate.csv"
    per_seed_path = aggregate_root / "notesm_v2_v2_1_val_per_seed.csv"
    source_md_path = aggregate_root / "NOTESM_V2_V2_1_VAL_SCREEN_AGGREGATE.md"
    aggregate = read_required_csv(aggregate_path)
    per_seed = read_required_csv(per_seed_path)
    validate_per_seed(per_seed, args.setting, args.seeds)
    aggregate_row = validate_aggregate(aggregate, args.setting, args.seeds)

    shutil.copy2(aggregate_path, output_dir / "notesm_v2_v2_1_val_aggregate_locked.csv")
    shutil.copy2(per_seed_path, output_dir / "notesm_v2_v2_1_val_per_seed_locked.csv")
    if source_md_path.exists():
        shutil.copy2(source_md_path, output_dir / "NOTESM_V2_V2_1_VAL_SCREEN_AGGREGATE_SOURCE.md")

    locked_summary = pd.DataFrame(
        [
            {
                "setting": args.setting,
                "split": "val",
                "seeds": " ".join(str(seed) for seed in sorted(args.seeds)),
                "n_seeds": len(args.seeds),
                "n": int(aggregate_row["n_min"]),
                "positives": int(aggregate_row["positives_min"]),
                **{f"{metric}_mean": float(aggregate_row[f"{metric}_mean"]) for metric in METRICS},
                **{f"{metric}_std": float(aggregate_row[f"{metric}_std"]) for metric in METRICS},
                "validation_only": True,
                "test_evaluation_launched": False,
                "aux_loss_weight": 0.0,
                "raw_text_free": True,
                "benchmark": "LAMP-R v0.2 frozen",
            }
        ]
    )
    locked_summary.to_csv(output_dir / "notesm_v2_v2_1_three_view_concat_val_locked_summary.csv", index=False)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_markdown(output_dir, aggregate_row, per_seed, args.setting, sorted(args.seeds), timestamp)
    print(f"Wrote NoteSSM-v2.1 validation lock artifacts to {output_dir}")
    print(locked_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
