from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.evaluation.metrics import binary_classification_metrics, brier_score, expected_calibration_error  # noqa: E402


FORBIDDEN_COLUMNS = {
    "text",
    "sequence_text",
    "merged_section_text",
    "section_text",
    "chunk_text",
    "subject_id",
    "hadm_id",
    "note_id",
}


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def candidate_thresholds(y_prob: np.ndarray, max_candidates: int) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=float)
    unique = np.unique(y_prob)
    if len(unique) <= max_candidates:
        values = unique
    else:
        quantiles = np.linspace(0.0, 1.0, max_candidates)
        values = np.unique(np.quantile(y_prob, quantiles))
    return np.unique(np.concatenate(([0.0], values, [1.0])))


def confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "predicted_positive": int(np.sum(y_pred == 1)),
    }


def metrics_at_threshold(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    y_true = frame["y_true"].astype(int).to_numpy()
    y_prob = frame["y_prob"].astype(float).to_numpy()
    metrics = binary_classification_metrics(y_true, y_prob, threshold=threshold)
    metrics["Brier"] = brier_score(y_true, y_prob)
    metrics["ECE"] = expected_calibration_error(y_true, y_prob)
    metrics.update(confusion_counts(y_true, y_prob, threshold))
    metrics["threshold"] = float(threshold)
    metrics["n"] = int(len(frame))
    metrics["prevalence"] = float(np.mean(y_true)) if len(frame) else 0.0
    return metrics


def choose_threshold(predictions: pd.DataFrame, objective: str, max_candidates: int) -> tuple[float, pd.DataFrame]:
    val = predictions[predictions["split"] == "val"].copy()
    if val.empty:
        raise ValueError("Prediction CSV has no validation rows")
    thresholds = candidate_thresholds(val["y_prob"].astype(float).to_numpy(), max_candidates)
    rows = []
    for threshold in thresholds:
        row = metrics_at_threshold(val, float(threshold))
        rows.append(row)
    grid = pd.DataFrame(rows)
    if objective == "f1":
        sort_cols = ["F1", "Recall", "Precision", "threshold"]
    elif objective == "youden":
        grid["YoudenJ"] = grid["Recall"] + (grid["tn"] / (grid["tn"] + grid["fp"]).clip(lower=1)) - 1.0
        sort_cols = ["YoudenJ", "F1", "Recall", "Precision"]
    else:
        raise ValueError(f"Unknown threshold objective: {objective}")
    best = grid.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]
    return float(best["threshold"]), grid


def validate_predictions(predictions: pd.DataFrame, path: Path) -> None:
    required = {"example_id", "split", "y_true", "y_prob"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    forbidden = FORBIDDEN_COLUMNS & set(predictions.columns)
    if forbidden:
        raise ValueError(f"{path} contains forbidden raw columns: {sorted(forbidden)}")
    if not set(predictions["split"]).issubset({"val", "test"}):
        raise ValueError(f"{path} contains non-evaluation splits: {sorted(set(predictions['split']))}")
    if not predictions["y_prob"].astype(float).between(0.0, 1.0).all():
        raise ValueError(f"{path} contains probabilities outside [0, 1]")


def tune_downstream_dir(downstream_dir: Path, model: str, seed: int, objective: str, max_candidates: int) -> dict[str, float | int | str]:
    prediction_path = downstream_dir / f"{model}_cached_predictions_seed{seed}.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {prediction_path}")
    predictions = pd.read_csv(prediction_path)
    validate_predictions(predictions, prediction_path)
    threshold, grid = choose_threshold(predictions, objective, max_candidates)
    tuned = predictions.copy()
    tuned["y_pred_tuned"] = (tuned["y_prob"].astype(float) >= threshold).astype(int)
    tuned["threshold"] = threshold
    rows = []
    for split in ["val", "test"]:
        split_frame = tuned[tuned["split"] == split].copy()
        if split_frame.empty:
            continue
        row = metrics_at_threshold(split_frame.rename(columns={"y_pred_tuned": "y_pred"}), threshold)
        row["split"] = split
        row["objective"] = objective
        row["downstream_dir"] = str(downstream_dir)
        rows.append(row)
    metrics = pd.DataFrame(rows)
    grid.to_csv(downstream_dir / f"{model}_threshold_grid_seed{seed}.csv", index=False)
    tuned[["example_id", "split", "y_true", "y_prob", "y_pred_tuned", "threshold"]].to_csv(
        downstream_dir / f"{model}_tuned_predictions_seed{seed}.csv",
        index=False,
    )
    metrics.to_csv(downstream_dir / f"{model}_threshold_tuned_metrics_seed{seed}.csv", index=False)
    best_val = metrics[metrics["split"] == "val"].iloc[0]
    has_test = bool((metrics["split"] == "test").any())
    best_test = metrics[metrics["split"] == "test"].iloc[0] if has_test else None
    return {
        "downstream_dir": str(downstream_dir),
        "objective": objective,
        "threshold": threshold,
        "val_F1": float(best_val["F1"]),
        "val_Precision": float(best_val["Precision"]),
        "val_Recall": float(best_val["Recall"]),
        "val_AUPRC": float(best_val["AUPRC"]),
        "val_AUROC": float(best_val["AUROC"]),
        "test_F1": float(best_test["F1"]) if best_test is not None else "",
        "test_Precision": float(best_test["Precision"]) if best_test is not None else "",
        "test_Recall": float(best_test["Recall"]) if best_test is not None else "",
        "test_AUPRC": float(best_test["AUPRC"]) if best_test is not None else "",
        "test_AUROC": float(best_test["AUROC"]) if best_test is not None else "",
        "val_predicted_positive": int(best_val["predicted_positive"]),
        "test_predicted_positive": int(best_test["predicted_positive"]) if best_test is not None else "",
    }


def collect_downstream_dirs(args: argparse.Namespace) -> list[Path]:
    dirs = [resolve_path(path) for path in args.downstream_dir]
    if args.downstream_root:
        root = resolve_path(args.downstream_root)
        dirs.extend(sorted(path for path in root.iterdir() if path.is_dir()))
    dirs = [path for path in dirs if not path.name.endswith("_extraction_only")]
    if args.name_contains:
        dirs = [path for path in dirs if args.name_contains in path.name]
    return dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune a binary decision threshold on validation predictions and apply it to test.")
    parser.add_argument("--downstream-dir", action="append", default=[], help="Downstream run directory. May be passed multiple times.")
    parser.add_argument("--downstream-root", default=None, help="Optional root; immediate subdirectories are tuned.")
    parser.add_argument("--name-contains", default=None, help="Optional substring filter for downstream-root subdirectories.")
    parser.add_argument("--model", default="notesm_v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--objective", choices=["f1", "youden"], default="f1")
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--output", default=None, help="Aggregate threshold tuning summary CSV.")
    args = parser.parse_args()

    downstream_dirs = collect_downstream_dirs(args)
    if not downstream_dirs:
        raise ValueError("No downstream directories were provided")
    rows = [tune_downstream_dir(path, args.model, args.seed, args.objective, args.max_candidates) for path in downstream_dirs]
    summary = pd.DataFrame(rows)
    output = resolve_path(args.output) if args.output else downstream_dirs[0].parent / f"{args.model}_threshold_tuning_summary_seed{args.seed}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
