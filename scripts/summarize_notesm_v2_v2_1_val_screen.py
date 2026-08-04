from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.evaluation.metrics import binary_classification_metrics, brier_score, expected_calibration_error  # noqa: E402


FORBIDDEN_COLUMNS = {"text", "sequence_text", "merged_section_text", "section_text", "chunk_text", "subject_id", "hadm_id", "note_id"}
SCREEN_MODES = ["three_view_concat", "three_view_residual", "set_transformer_residual"]


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if Path.cwd().resolve() == ROOT.resolve():
        return ROOT / path
    return ROOT / path


def read_text_free_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    forbidden = sorted(FORBIDDEN_COLUMNS & set(frame.columns))
    if forbidden:
        raise ValueError(f"Prediction file contains raw/restricted columns: {forbidden}")
    required = {"example_id", "split", "y_true", "y_prob"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {sorted(missing)}")
    return frame


def best_val_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float, float]:
    candidates = np.unique(y_prob)
    if len(candidates) == 0:
        return 0.5, 0.0, 0.0, 0.0
    best = (0.5, -1.0, 0.0, 0.0)
    for threshold in candidates:
        pred = (y_prob >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        if f1 > best[1] or (f1 == best[1] and threshold > best[0]):
            best = (float(threshold), float(f1), float(precision), float(recall))
    return best


def summarize_prediction_dir(directory: Path, mode: str, seed: int) -> dict[str, object]:
    pred_path = directory / f"notesm_v2_hybrid_cached_predictions_seed{seed}.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing validation prediction file: {pred_path}")
    predictions = read_text_free_predictions(pred_path)
    if set(predictions["split"].astype(str)) != {"val"}:
        raise ValueError(f"Stage 1 screen must contain validation predictions only: {pred_path}")
    y_true = predictions["y_true"].astype(int).to_numpy()
    y_prob = predictions["y_prob"].astype(float).to_numpy()
    threshold, f1, precision, recall = best_val_threshold(y_true, y_prob)
    metrics = binary_classification_metrics(y_true, y_prob)
    summary_path = directory / f"notesm_v2_hybrid_v2_1_training_summary_seed{seed}.csv"
    training = pd.read_csv(summary_path).iloc[0].to_dict() if summary_path.exists() else {}
    return {
        "setting": mode,
        "split": "val",
        "seed": seed,
        "n": len(predictions),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()) if len(y_true) else 0.0,
        "AUPRC": float(metrics["AUPRC"]),
        "AUROC": float(metrics["AUROC"]),
        "threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Brier": brier_score(y_true, y_prob),
        "ECE": expected_calibration_error(y_true, y_prob),
        "runtime_seconds": training.get("runtime_seconds", ""),
        "peak_gpu_memory_mb": training.get("peak_gpu_memory_mb", ""),
        "trainable_parameters": training.get("trainable_parameters", ""),
        "initial_residual_alpha": training.get("initial_residual_alpha", ""),
        "final_residual_alpha": training.get("final_residual_alpha", ""),
        "note_head_initialized_from_checkpoint": training.get("note_head_initialized_from_checkpoint", ""),
        "note_head_checkpoint": training.get("note_head_checkpoint", ""),
        "raw_text_free_validation_result": True,
        "source_prediction_file": str(pred_path),
    }


def source_row(path: Path, key_col: str, key: str, display_name: str, role: str) -> dict[str, object] | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "split" not in frame.columns:
        return None
    rows = frame[(frame["split"].astype(str) == "val") & (frame[key_col].astype(str) == key)]
    if "fallback_group" in rows.columns:
        rows = rows[rows["fallback_group"].astype(str) == "all"]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        "setting": key,
        "display_name": display_name,
        "role": role,
        "split": "val",
        "AUPRC": row.get("AUPRC_mean", row.get("AUPRC")),
        "AUROC": row.get("AUROC_mean", row.get("AUROC")),
        "F1": row.get("F1_mean", row.get("F1")),
        "Precision": row.get("Precision_mean", row.get("Precision")),
        "Recall": row.get("Recall_mean", row.get("Recall")),
        "Brier": row.get("Brier_mean", row.get("Brier")),
        "ECE": row.get("ECE_mean", row.get("ECE")),
        "source_file": str(path),
    }


def load_validation_references(no_dapt_table: Path, hybrid_table: Path) -> pd.DataFrame:
    rows = []
    for key, display, role in [
        ("concat_mlp", "Current hybrid concat_mlp", "current frozen hybrid reference"),
        ("current_section_aware", "Current section-aware", "diagnostic reference"),
    ]:
        row = source_row(hybrid_table, "setting", key, display, role)
        if row is not None:
            rows.append(row)
    for key, display, role in [
        ("bioclinicalbert_512", "BioClinicalBERT", "external comparator"),
        ("clinicalmamba_4096", "ClinicalMamba", "external comparator"),
        ("longformer_mean_4096", "Longformer", "external comparator"),
    ]:
        row = source_row(no_dapt_table, "model_key", key, display, role)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def metadata_usage(screen_root: Path, seed: int, modes: list[str]) -> pd.DataFrame:
    rows = []
    for mode in modes:
        path = screen_root / mode / f"notesm_v2_hybrid_metadata_features_used_seed{seed}.csv"
        if not path.exists():
            rows.append({"setting": mode, "metadata_usage_file_found": False})
            continue
        row = pd.read_csv(path).iloc[0].to_dict()
        row["setting"] = mode
        row["metadata_usage_file_found"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def decision_text(main: pd.DataFrame, references: pd.DataFrame) -> str:
    concat_ref = references[references["setting"] == "concat_mlp"]
    biobert_ref = references[references["setting"] == "bioclinicalbert_512"]
    lines = []
    if not concat_ref.empty:
        ref = concat_ref.iloc[0]
        best = main.sort_values("AUPRC", ascending=False).iloc[0]
        d_auprc = float(best["AUPRC"]) - float(ref["AUPRC"])
        d_auroc = float(best["AUROC"]) - float(ref["AUROC"])
        d_brier = float(best["Brier"]) - float(ref["Brier"])
        lines.append(
            f"Best v2.1 validation AUPRC variant: {best['setting']} "
            f"(AUPRC delta vs current concat_mlp {d_auprc:+.4f}, AUROC delta {d_auroc:+.4f}, Brier delta {d_brier:+.4f})."
        )
        if d_auprc >= 0.020:
            lines.append("Strong promotion signal by AUPRC delta >= +0.020.")
        elif d_auprc >= 0.010 and d_auroc >= 0.005 and d_brier <= 0.003:
            lines.append("Minimum promotion signal met.")
        else:
            lines.append("Minimum promotion signal not met.")
    if not biobert_ref.empty and not main.empty:
        best = main.sort_values("AUPRC", ascending=False).iloc[0]
        gap = float(best["AUPRC"]) - float(biobert_ref.iloc[0]["AUPRC"])
        lines.append(f"AUPRC gap vs BioClinicalBERT validation reference: {gap:+.4f}.")
    lines.append("Do not run test evaluation until these validation results are reviewed.")
    return "\n".join(lines)


def write_markdown(output_dir: Path, main: pd.DataFrame, references: pd.DataFrame, decision: str, seed: int) -> None:
    main_text = main.to_string(index=False) if not main.empty else "No v2.1 validation rows found."
    reference_text = references.to_string(index=False) if not references.empty else "No validation reference rows found."
    text = f"""# NoteSSM-v2.1 Validation Screen Seed{seed}

This is a Stage 1 validation-only architecture screen. It uses frozen LAMP-R v0.2 artifacts and existing cached embeddings only.

Constraints honored:
- no benchmark changes
- no DAPT rerun
- no note/section/chunk embedding extraction
- no test evaluation
- no test prediction files
- no class_weight or focal loss

## Validation Results

```text
{main_text}
```

## Validation References

```text
{reference_text}
```

## Decision Context

{decision}

## Caveat

The previous concat_mlp design was motivated by full-test diagnostics. The v2.1 screen should therefore be described as validation-selected follow-up work. Any later test result is exploratory unless confirmed on a new locked holdout, LAMP-R v0.3, or an external cohort.
"""
    (output_dir / f"NOTESM_V2_V2_1_VAL_SCREEN_SEED{seed}.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize NoteSSM-v2.1 validation-only screen.")
    parser.add_argument("--screen-root", default="results/downstream_notesm_v2_v2_1_val_screen_seed17")
    parser.add_argument("--output-dir", default="results/notesm_v2_v2_1_val_screen_seed17")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--modes", nargs="+", default=SCREEN_MODES, help="Validation screen modes to summarize.")
    parser.add_argument("--no-dapt-table", default="results/no_dapt_full_comparison_frozen/no_dapt_full_main_table.csv")
    parser.add_argument("--hybrid-table", default="results/notesm_v2_hybrid_concat_full_frozen/notesm_v2_hybrid_concat_main_table.csv")
    args = parser.parse_args()
    modes = list(args.modes)
    unsupported = sorted(set(modes) - set(SCREEN_MODES))
    if unsupported:
        raise ValueError(f"Unsupported validation screen modes: {unsupported}")

    screen_root = resolve_path(args.screen_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_prediction_dir(screen_root / mode, mode, args.seed) for mode in modes]
    main_table = pd.DataFrame(rows)
    main_table.to_csv(output_dir / "notesm_v2_v2_1_val_main_table.csv", index=False)

    references = load_validation_references(resolve_path(args.no_dapt_table), resolve_path(args.hybrid_table))
    references.to_csv(output_dir / "notesm_v2_v2_1_val_references.csv", index=False)

    concat_ref = references[references["setting"] == "concat_mlp"]
    if not concat_ref.empty:
        ref = concat_ref.iloc[0]
        delta_rows = []
        for _, row in main_table.iterrows():
            delta = {"setting": row["setting"]}
            for metric in ["AUPRC", "AUROC", "F1", "Precision", "Recall", "Brier", "ECE"]:
                delta[f"{metric}_delta_vs_concat"] = float(row[metric]) - float(ref[metric])
            delta_rows.append(delta)
        pd.DataFrame(delta_rows).to_csv(output_dir / "notesm_v2_v2_1_val_delta_vs_concat.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "notesm_v2_v2_1_val_delta_vs_concat.csv", index=False)

    usage = metadata_usage(screen_root, args.seed, modes)
    usage.to_csv(output_dir / "notesm_v2_v2_1_metadata_usage.csv", index=False)
    decision = decision_text(main_table, references)
    write_markdown(output_dir, main_table, references, decision, args.seed)
    print(f"Wrote NoteSSM-v2.1 validation-only summary to {output_dir}")
    print(main_table[["setting", "AUPRC", "AUROC", "F1", "Brier", "ECE"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
