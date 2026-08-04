from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FORBIDDEN_COLUMNS = {"text", "sequence_text", "merged_section_text", "section_text", "chunk_text", "subject_id", "hadm_id", "note_id"}
METRICS = ["AUPRC", "AUROC", "Precision", "Recall", "F1", "Brier", "ECE", "threshold"]


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if Path.cwd().resolve() == ROOT.resolve():
        return ROOT / path
    return ROOT / path


def read_seed_summary(summary_root: Path, seed: int, setting: str) -> pd.Series:
    path = summary_root / "notesm_v2_v2_1_val_main_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing seed summary table: {path}")
    frame = pd.read_csv(path)
    forbidden = sorted(FORBIDDEN_COLUMNS & set(frame.columns))
    if forbidden:
        raise ValueError(f"Seed summary contains restricted columns: {forbidden}")
    rows = frame[(frame["setting"].astype(str) == setting) & (frame["split"].astype(str) == "val")]
    if rows.empty:
        raise ValueError(f"Missing setting={setting!r}, split='val' in {path}")
    if len(rows) > 1:
        raise ValueError(f"Duplicate setting={setting!r}, split='val' rows in {path}")
    row = rows.iloc[0].copy()
    row["seed"] = int(seed)
    row["source_summary_file"] = str(path)
    return row


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting, group in per_seed.groupby("setting", sort=False):
        row: dict[str, object] = {
            "setting": setting,
            "split": "val",
            "n_seeds": int(group["seed"].nunique()),
            "seeds": " ".join(str(seed) for seed in sorted(group["seed"].astype(int).unique())),
        }
        if "n" in group.columns:
            row["n_min"] = int(pd.to_numeric(group["n"]).min())
            row["n_max"] = int(pd.to_numeric(group["n"]).max())
        if "positives" in group.columns:
            row["positives_min"] = int(pd.to_numeric(group["positives"]).min())
            row["positives_max"] = int(pd.to_numeric(group["positives"]).max())
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(output_dir: Path, per_seed: pd.DataFrame, aggregate_table: pd.DataFrame, setting: str) -> None:
    per_seed_text = per_seed.to_string(index=False)
    aggregate_text = aggregate_table.to_string(index=False)
    text = f"""# NoteSSM-v2.1 Validation Screen Aggregate

This aggregates validation-only architecture screening results for `{setting}`.

Constraints honored:
- LAMP-R v0.2 unchanged
- no DAPT rerun
- no note/section/chunk embedding extraction
- no test evaluation
- no class_weight or focal loss
- prediction and summary tables are raw-text-free

## Aggregate

```text
{aggregate_text}
```

## Per-Seed Rows

```text
{per_seed_text}
```

## Interpretation

Use this aggregate for validation-stage architecture selection only. A later test run remains exploratory because v2.1 was motivated after prior full-test diagnostic analysis.
"""
    (output_dir / "NOTESM_V2_V2_1_VAL_SCREEN_AGGREGATE.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate NoteSSM-v2.1 validation screen summaries across seeds.")
    parser.add_argument("--summary-root-template", default="results/notesm_v2_v2_1_val_screen_seed{seed}")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 42])
    parser.add_argument("--setting", default="three_view_concat")
    parser.add_argument("--output-dir", default="results/notesm_v2_v2_1_val_screen_3seed_concat")
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        summary_root = resolve_path(args.summary_root_template.format(seed=seed))
        rows.append(read_seed_summary(summary_root, seed, args.setting))
    per_seed = pd.DataFrame(rows)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "notesm_v2_v2_1_val_per_seed.csv", index=False)
    aggregate_table = aggregate(per_seed)
    aggregate_table.to_csv(output_dir / "notesm_v2_v2_1_val_aggregate.csv", index=False)
    write_markdown(output_dir, per_seed, aggregate_table, args.setting)
    print(f"Wrote NoteSSM-v2.1 validation aggregate to {output_dir}")
    print(aggregate_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
