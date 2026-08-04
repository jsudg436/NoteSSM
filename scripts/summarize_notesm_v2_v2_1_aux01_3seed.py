from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

METRICS = ["AUPRC", "AUROC", "Precision", "Recall", "F1", "Brier", "ECE", "threshold"]
FORBIDDEN_COLUMNS = {"text", "sequence_text", "merged_section_text", "section_text", "chunk_text", "subject_id", "hadm_id", "note_id"}


def resolve_path(path_arg: str | Path) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if Path.cwd().resolve() == ROOT.resolve():
        return ROOT / path
    return ROOT / path


def read_one(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary table: {path}")
    frame = pd.read_csv(path)
    forbidden = sorted(FORBIDDEN_COLUMNS & set(frame.columns))
    if forbidden:
        raise ValueError(f"{path} contains restricted columns: {forbidden}")
    rows = frame[(frame["setting"].astype(str) == "three_view_concat") & (frame["split"].astype(str) == "val")]
    if len(rows) != 1:
        raise ValueError(f"Expected one three_view_concat/val row in {path}; found {len(rows)}")
    return rows.iloc[0]


def metric_value(row: pd.Series, metric: str) -> float:
    if metric in row:
        return float(row[metric])
    mean_col = f"{metric}_mean"
    if mean_col in row:
        return float(row[mean_col])
    raise KeyError(metric)


def build_tables(aux01: pd.Series, aux00_locked: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    aux01_row = {
        "setting": "three_view_concat_aux0.1_3seed",
        "role": "limited 3-seed validation-only check",
        "split": "val",
        "aux_loss_weight": 0.1,
        "seeds": aux01.get("seeds", "17 23 42"),
        "n_seeds": int(aux01.get("n_seeds", 3)),
        "n": int(aux01.get("n_min", aux01.get("n", 0))),
        "positives": int(aux01.get("positives_min", aux01.get("positives", 0))),
    }
    aux00_row = {
        "setting": "three_view_concat_aux0.0_3seed_locked",
        "role": "locked validation reference",
        "split": "val",
        "aux_loss_weight": 0.0,
        "seeds": aux00_locked.get("seeds", "17 23 42"),
        "n_seeds": int(aux00_locked.get("n_seeds", 3)),
        "n": int(aux00_locked.get("n", aux00_locked.get("n_min", 0))),
        "positives": int(aux00_locked.get("positives", aux00_locked.get("positives_min", 0))),
    }
    for metric in METRICS:
        aux01_row[f"{metric}_mean"] = metric_value(aux01, metric)
        aux01_row[f"{metric}_std"] = float(aux01.get(f"{metric}_std", 0.0))
        aux00_row[f"{metric}_mean"] = metric_value(aux00_locked, metric)
        aux00_row[f"{metric}_std"] = float(aux00_locked.get(f"{metric}_std", 0.0))
    main = pd.DataFrame([aux01_row, aux00_row])
    delta = {"comparison": "three_view_concat_aux0.1_3seed vs three_view_concat_aux0.0_3seed_locked"}
    for metric in METRICS:
        delta[f"{metric}_mean_delta"] = float(aux01_row[f"{metric}_mean"]) - float(aux00_row[f"{metric}_mean"])
        delta[f"{metric}_std_delta"] = float(aux01_row[f"{metric}_std"]) - float(aux00_row[f"{metric}_std"])
    return main, pd.DataFrame([delta])


def decision_text(delta: pd.DataFrame) -> str:
    row = delta.iloc[0]
    auprc = float(row["AUPRC_mean_delta"])
    auroc = float(row["AUROC_mean_delta"])
    f1 = float(row["F1_mean_delta"])
    brier = float(row["Brier_mean_delta"])
    ece = float(row["ECE_mean_delta"])
    if auprc >= 0.005 and auroc >= 0.002 and f1 >= 0.002 and brier <= 0.001:
        return "Aux=0.1 shows enough 3-seed validation signal to consider as a test-evaluation candidate."
    if auprc > 0 and auroc > 0 and f1 >= 0 and brier <= 0.001 and ece <= 0.002:
        return "Aux=0.1 is close, but the 3-seed validation gain is modest. Treat as optional; aux=0.0 remains the cleaner locked candidate."
    return "Aux=0.1 does not show enough 3-seed validation improvement. Keep aux=0.0 as the locked validation-stage candidate."


def write_markdown(output_dir: Path, main: pd.DataFrame, delta: pd.DataFrame, decision: str) -> None:
    text = f"""# NoteSSM-v2.1 Aux=0.1 3-Seed Validation Check

This compares `three_view_concat` with `aux_loss_weight=0.1` against the locked `aux_loss_weight=0.0` validation reference.

Constraints honored:
- LAMP-R v0.2 unchanged
- no DAPT rerun
- no note/section/chunk embedding extraction
- no test evaluation
- seeds 17, 23, and 42 only
- raw clinical note text is not written to this summary

## Main Table

```text
{main.to_string(index=False)}
```

## Delta Table

```text
{delta.to_string(index=False)}
```

## Decision

{decision}
"""
    (output_dir / "NOTESSM_V2_V2_1_AUX01_3SEED_VAL_CHECK.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize aux=0.1 three-seed validation check against aux=0.0 locked reference.")
    parser.add_argument("--aux01-aggregate", default="results/notesm_v2_v2_1_val_screen_3seed_concat_aux01/notesm_v2_v2_1_val_aggregate.csv")
    parser.add_argument(
        "--aux00-locked-summary",
        default="results/notesm_v2_v2_1_val_screen_3seed_concat_locked/notesm_v2_v2_1_three_view_concat_val_locked_summary.csv",
    )
    parser.add_argument("--output-dir", default="results/notesm_v2_v2_1_aux01_3seed_val_check")
    args = parser.parse_args()

    aux01 = read_one(resolve_path(args.aux01_aggregate))
    aux00 = read_one(resolve_path(args.aux00_locked_summary))
    main_table, delta_table = build_tables(aux01, aux00)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_table.to_csv(output_dir / "notesm_v2_v2_1_aux01_3seed_val_main_table.csv", index=False)
    delta_table.to_csv(output_dir / "notesm_v2_v2_1_aux01_3seed_val_delta.csv", index=False)
    decision = decision_text(delta_table)
    write_markdown(output_dir, main_table, delta_table, decision)
    print(f"Wrote NoteSSM-v2.1 aux=0.1 three-seed validation check summary to {output_dir}")
    print(main_table.to_string(index=False))
    print(delta_table.to_string(index=False))
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
