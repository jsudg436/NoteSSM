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


def read_summary(path: Path, setting: str = "three_view_concat") -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary table: {path}")
    frame = pd.read_csv(path)
    forbidden = sorted(FORBIDDEN_COLUMNS & set(frame.columns))
    if forbidden:
        raise ValueError(f"{path} contains restricted columns: {forbidden}")
    rows = frame[(frame["setting"].astype(str) == setting) & (frame["split"].astype(str) == "val")]
    if len(rows) != 1:
        raise ValueError(f"Expected one {setting}/val row in {path}; found {len(rows)}")
    return rows.iloc[0]


def metric_value(row: pd.Series, metric: str) -> float:
    if metric in row:
        return float(row[metric])
    mean_col = f"{metric}_mean"
    if mean_col in row:
        return float(row[mean_col])
    raise KeyError(metric)


def build_tables(aux01_row: pd.Series, aux00_seed17: pd.Series, aux00_locked: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_rows = []
    for label, row, aux_weight, role in [
        ("three_view_concat_aux0.1_seed17", aux01_row, 0.1, "limited validation-only check"),
        ("three_view_concat_aux0.0_seed17", aux00_seed17, 0.0, "seed17 reference"),
        ("three_view_concat_aux0.0_3seed_locked", aux00_locked, 0.0, "locked 3-seed validation reference"),
    ]:
        out = {
            "setting": label,
            "role": role,
            "split": "val",
            "seed": row.get("seed", ""),
            "aux_loss_weight": aux_weight,
            "n": int(row["n"]) if "n" in row else "",
            "positives": int(row["positives"]) if "positives" in row else "",
        }
        for metric in METRICS:
            out[metric] = metric_value(row, metric)
        main_rows.append(out)
    main = pd.DataFrame(main_rows)

    aux01 = main.iloc[0]
    delta_rows = []
    for _, reference in main.iloc[1:].iterrows():
        delta = {"comparison": f"{aux01['setting']} vs {reference['setting']}"}
        for metric in METRICS:
            delta[f"{metric}_delta"] = float(aux01[metric]) - float(reference[metric])
        delta_rows.append(delta)
    return main, pd.DataFrame(delta_rows)


def decision_text(delta: pd.DataFrame) -> str:
    seed17 = delta[delta["comparison"].str.contains("aux0.0_seed17", regex=False)]
    if seed17.empty:
        return "No seed17 aux=0.0 reference was available."
    row = seed17.iloc[0]
    auprc_delta = float(row["AUPRC_delta"])
    auroc_delta = float(row["AUROC_delta"])
    f1_delta = float(row["F1_delta"])
    brier_delta = float(row["Brier_delta"])
    ece_delta = float(row["ECE_delta"])
    if auprc_delta >= 0.005 and auroc_delta >= 0.002 and f1_delta >= 0.002 and brier_delta <= 0.001:
        return (
            "Aux=0.1 shows a limited seed17 validation improvement signal versus aux=0.0. "
            "Consider repeating aux=0.1 on seeds 23/42 before any test evaluation."
        )
    return (
        "Aux=0.1 does not show a convincing seed17 validation improvement over aux=0.0. "
        "Keep aux=0.0 as the locked validation-stage candidate unless there is a specific reason to explore auxiliary losses further."
    )


def write_markdown(output_dir: Path, main: pd.DataFrame, delta: pd.DataFrame, decision: str) -> None:
    text = f"""# NoteSSM-v2.1 Aux=0.1 Seed17 Validation Check

This is a limited validation-only check of `three_view_concat` with `aux_loss_weight=0.1`.

Constraints honored:
- LAMP-R v0.2 unchanged
- no DAPT rerun
- no note/section/chunk embedding extraction
- no test evaluation
- seed17 only
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
    (output_dir / "NOTESSM_V2_V2_1_AUX01_SEED17_VAL_CHECK.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize limited NoteSSM-v2.1 aux=0.1 seed17 validation check.")
    parser.add_argument("--aux01-summary", default="results/notesm_v2_v2_1_val_screen_seed17_aux01/notesm_v2_v2_1_val_main_table.csv")
    parser.add_argument("--aux00-seed17-summary", default="results/notesm_v2_v2_1_val_screen_seed17/notesm_v2_v2_1_val_main_table.csv")
    parser.add_argument(
        "--aux00-locked-summary",
        default="results/notesm_v2_v2_1_val_screen_3seed_concat_locked/notesm_v2_v2_1_three_view_concat_val_locked_summary.csv",
    )
    parser.add_argument("--output-dir", default="results/notesm_v2_v2_1_aux01_seed17_val_check")
    args = parser.parse_args()

    aux01 = read_summary(resolve_path(args.aux01_summary))
    aux00_seed17 = read_summary(resolve_path(args.aux00_seed17_summary))
    aux00_locked = read_summary(resolve_path(args.aux00_locked_summary))
    main_table, delta_table = build_tables(aux01, aux00_seed17, aux00_locked)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_table.to_csv(output_dir / "notesm_v2_v2_1_aux01_seed17_val_main_table.csv", index=False)
    delta_table.to_csv(output_dir / "notesm_v2_v2_1_aux01_seed17_val_delta.csv", index=False)
    decision = decision_text(delta_table)
    write_markdown(output_dir, main_table, delta_table, decision)
    print(f"Wrote NoteSSM-v2.1 aux=0.1 seed17 validation check summary to {output_dir}")
    print(main_table.to_string(index=False))
    print(delta_table.to_string(index=False))
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
