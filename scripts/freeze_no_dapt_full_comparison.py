from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
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
METRICS = ["AUPRC", "AUROC", "F1", "Precision", "Recall", "Brier", "ECE"]
HIGHER_IS_BETTER = {
    "AUPRC": True,
    "AUROC": True,
    "F1": True,
    "Precision": True,
    "Recall": True,
    "Brier": False,
    "ECE": False,
}
DEFAULT_EXPECTED_EVAL_COUNTS = {"val": 29298, "test": 29460}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    role: str
    input_name: str
    max_length: int
    pooling: str
    downstream_dir: Path


@dataclass
class PreparedPredictions:
    y_true: np.ndarray
    y_prob: np.ndarray
    y_pred: np.ndarray
    desc_order: np.ndarray
    desc_starts: np.ndarray
    asc_order: np.ndarray
    asc_starts: np.ndarray

    @classmethod
    def from_arrays(cls, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> "PreparedPredictions":
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)
        y_pred = np.asarray(y_pred).astype(int)
        desc_order = np.argsort(-y_prob, kind="mergesort")
        asc_order = np.argsort(y_prob, kind="mergesort")
        desc_scores = y_prob[desc_order]
        asc_scores = y_prob[asc_order]
        desc_starts = np.r_[0, np.flatnonzero(np.diff(desc_scores) != 0) + 1]
        asc_starts = np.r_[0, np.flatnonzero(np.diff(asc_scores) != 0) + 1]
        return cls(y_true, y_prob, y_pred, desc_order, desc_starts, asc_order, asc_starts)


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def prediction_path(downstream_dir: Path, seed: int, model: str = "notesm_v1") -> Path:
    path = downstream_dir / f"{model}_tuned_predictions_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing tuned prediction CSV: {path}")
    return path


def constant_threshold(frame: pd.DataFrame, model_key: str, seed: int, split: str, path: Path | None = None) -> float:
    values = frame["threshold"].drop_duplicates()
    if len(values) != 1 or values.isna().any():
        location = f" in {path}" if path is not None else ""
        raise ValueError(
            f"Expected exactly one threshold for model={model_key} seed={seed} split={split}{location}, "
            f"found {len(values)} distinct values"
        )
    return float(values.iloc[0])


def validate_prediction_frame(
    frame: pd.DataFrame,
    spec: ModelSpec,
    seed: int,
    path: Path,
    expected_eval_counts: dict[str, int] | None,
) -> None:
    duplicate_count = int(frame.duplicated(["example_id", "split"]).sum())
    if duplicate_count:
        raise ValueError(f"{path} contains duplicated example_id/split prediction rows: {duplicate_count}")
    for split in ["val", "test"]:
        split_frame = frame[frame["split"] == split]
        if split_frame.empty:
            raise ValueError(f"{path} has no {split} predictions")
        constant_threshold(split_frame, spec.key, seed, split, path)
        if expected_eval_counts is not None:
            expected = expected_eval_counts.get(split)
            if expected is not None and len(split_frame) != expected:
                raise ValueError(
                    f"{path} has {len(split_frame)} {split} predictions, expected {expected} for frozen LAMP-R v0.2"
                )


def load_predictions(spec: ModelSpec, seed: int, expected_eval_counts: dict[str, int] | None = None) -> pd.DataFrame:
    path = prediction_path(spec.downstream_dir, seed)
    frame = pd.read_csv(path)
    required = {"example_id", "split", "y_true", "y_prob", "y_pred_tuned", "threshold"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    forbidden = FORBIDDEN_COLUMNS & set(frame.columns)
    if forbidden:
        raise ValueError(f"{path} contains forbidden raw columns: {sorted(forbidden)}")
    if not set(frame["split"]).issubset({"val", "test"}):
        raise ValueError(f"{path} contains non-evaluation split values")
    frame = frame.copy()
    frame["example_id"] = frame["example_id"].astype(str)
    frame["y_true"] = frame["y_true"].astype(int)
    frame["y_prob"] = frame["y_prob"].astype(float)
    frame["y_pred_tuned"] = frame["y_pred_tuned"].astype(int)
    frame["threshold"] = frame["threshold"].astype(float)
    validate_prediction_frame(frame, spec, seed, path, expected_eval_counts)
    frame["model_key"] = spec.key
    frame["seed"] = seed
    return frame


def weighted_auprc(prepared: PreparedPredictions, weights: np.ndarray) -> float:
    order = prepared.desc_order
    y = prepared.y_true[order]
    w = weights[order].astype(float)
    total_pos = float(np.sum(w * y))
    if total_pos <= 0:
        return np.nan
    group_tp = np.add.reduceat(w * y, prepared.desc_starts)
    group_fp = np.add.reduceat(w * (1 - y), prepared.desc_starts)
    cum_tp = np.cumsum(group_tp)
    cum_fp = np.cumsum(group_fp)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    recall = cum_tp / total_pos
    recall_delta = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_delta * precision))


def weighted_auroc(prepared: PreparedPredictions, weights: np.ndarray) -> float:
    order = prepared.asc_order
    y = prepared.y_true[order]
    w = weights[order].astype(float)
    group_pos = np.add.reduceat(w * y, prepared.asc_starts)
    group_neg = np.add.reduceat(w * (1 - y), prepared.asc_starts)
    total_pos = float(np.sum(group_pos))
    total_neg = float(np.sum(group_neg))
    if total_pos <= 0 or total_neg <= 0:
        return np.nan
    neg_before = np.cumsum(group_neg) - group_neg
    numerator = float(np.sum(group_pos * neg_before + 0.5 * group_pos * group_neg))
    return numerator / (total_pos * total_neg)


def weighted_brier(prepared: PreparedPredictions, weights: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if denom <= 0:
        return np.nan
    return float(np.sum(weights * (prepared.y_prob - prepared.y_true) ** 2) / denom)


def weighted_ece(prepared: PreparedPredictions, weights: np.ndarray, n_bins: int = 10) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return np.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right < 1.0:
            mask = (prepared.y_prob >= left) & (prepared.y_prob < right)
        else:
            mask = (prepared.y_prob >= left) & (prepared.y_prob <= right)
        bin_weight = float(np.sum(weights[mask]))
        if bin_weight <= 0:
            continue
        confidence = float(np.sum(weights[mask] * prepared.y_prob[mask]) / bin_weight)
        observed = float(np.sum(weights[mask] * prepared.y_true[mask]) / bin_weight)
        ece += (bin_weight / total) * abs(confidence - observed)
    return float(ece)


def weighted_threshold_metrics(prepared: PreparedPredictions, weights: np.ndarray) -> dict[str, float]:
    y = prepared.y_true
    pred = prepared.y_pred
    tp = float(np.sum(weights * ((y == 1) & (pred == 1))))
    fp = float(np.sum(weights * ((y == 0) & (pred == 1))))
    fn = float(np.sum(weights * ((y == 1) & (pred == 0))))
    tn = float(np.sum(weights * ((y == 0) & (pred == 0))))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(float(np.sum(weights)), 1.0)
    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def all_metrics(prepared: PreparedPredictions, weights: np.ndarray) -> dict[str, float]:
    out = weighted_threshold_metrics(prepared, weights)
    out["AUPRC"] = weighted_auprc(prepared, weights)
    out["AUROC"] = weighted_auroc(prepared, weights)
    out["Brier"] = weighted_brier(prepared, weights)
    out["ECE"] = weighted_ece(prepared, weights)
    return out


def metrics_for_frame(frame: pd.DataFrame) -> dict[str, float]:
    prepared = PreparedPredictions.from_arrays(frame["y_true"], frame["y_prob"], frame["y_pred_tuned"])
    weights = np.ones(len(frame), dtype=float)
    return all_metrics(prepared, weights)


def main_table(specs: list[ModelSpec], seeds: list[int], predictions: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        for seed in seeds:
            frame = predictions[(spec.key, seed)]
            for split in ["val", "test"]:
                split_frame = frame[frame["split"] == split].copy()
                metrics = metrics_for_frame(split_frame)
                rows.append(
                    {
                        "model_key": spec.key,
                        "display_name": spec.display_name,
                        "role": spec.role,
                        "input": spec.input_name,
                        "max_length": spec.max_length,
                        "pooling": spec.pooling,
                        "head": "MLP",
                        "loss": "natural_bce",
                        "seed": seed,
                        "split": split,
                        "threshold": constant_threshold(split_frame, spec.key, seed, split),
                        "n": int(len(split_frame)),
                        "prevalence": float(split_frame["y_true"].mean()),
                        **{metric: metrics[metric] for metric in ["AUPRC", "AUROC", "Accuracy", "Precision", "Recall", "F1", "Brier", "ECE"]},
                    }
                )
    per_seed = pd.DataFrame(rows)
    group_cols = ["model_key", "display_name", "role", "input", "max_length", "pooling", "head", "loss", "split"]
    metric_cols = ["AUPRC", "AUROC", "Accuracy", "Precision", "Recall", "F1", "Brier", "ECE", "threshold", "n", "prevalence"]
    agg = per_seed.groupby(group_cols, as_index=False)[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in agg.columns]
    return agg


def calibration_summary(specs: list[ModelSpec], seeds: list[int], predictions: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        for seed in seeds:
            frame = predictions[(spec.key, seed)]
            for split in ["val", "test"]:
                split_frame = frame[frame["split"] == split].copy()
                prepared = PreparedPredictions.from_arrays(split_frame["y_true"], split_frame["y_prob"], split_frame["y_pred_tuned"])
                weights = np.ones(len(split_frame), dtype=float)
                order = np.argsort(-prepared.y_prob)
                decile = order[: max(1, len(order) // 10)]
                positives = max(float(np.sum(prepared.y_true)), 1.0)
                rows.append(
                    {
                        "model_key": spec.key,
                        "display_name": spec.display_name,
                        "seed": seed,
                        "split": split,
                        "n": int(len(split_frame)),
                        "prevalence": float(np.mean(prepared.y_true)),
                        "mean_score": float(np.mean(prepared.y_prob)),
                        "brier": weighted_brier(prepared, weights),
                        "ece": weighted_ece(prepared, weights),
                        "top_decile_observed_rate": float(np.mean(prepared.y_true[decile])),
                        "top_decile_sensitivity": float(np.sum(prepared.y_true[decile]) / positives),
                    }
                )
    per_seed = pd.DataFrame(rows)
    group_cols = ["model_key", "display_name", "split"]
    metric_cols = ["n", "prevalence", "mean_score", "brier", "ece", "top_decile_observed_rate", "top_decile_sensitivity"]
    agg = per_seed.groupby(group_cols, as_index=False)[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in agg.columns]
    return agg


def error_summary(specs: list[ModelSpec], seeds: list[int], predictions: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        for seed in seeds:
            frame = predictions[(spec.key, seed)]
            for split in ["val", "test"]:
                split_frame = frame[frame["split"] == split].copy()
                prepared = PreparedPredictions.from_arrays(split_frame["y_true"], split_frame["y_prob"], split_frame["y_pred_tuned"])
                weights = np.ones(len(split_frame), dtype=float)
                metrics = weighted_threshold_metrics(prepared, weights)
                n = max(float(len(split_frame)), 1.0)
                rows.append(
                    {
                        "model_key": spec.key,
                        "display_name": spec.display_name,
                        "seed": seed,
                        "split": split,
                        "n": int(len(split_frame)),
                        "tp": metrics["tp"],
                        "fp": metrics["fp"],
                        "tn": metrics["tn"],
                        "fn": metrics["fn"],
                        "fp_rate": metrics["fp"] / n,
                        "fn_rate": metrics["fn"] / n,
                        "Precision": metrics["Precision"],
                        "Recall": metrics["Recall"],
                        "F1": metrics["F1"],
                    }
                )
    per_seed = pd.DataFrame(rows)
    group_cols = ["model_key", "display_name", "split"]
    metric_cols = ["n", "tp", "fp", "tn", "fn", "fp_rate", "fn_rate", "Precision", "Recall", "F1"]
    agg = per_seed.groupby(group_cols, as_index=False)[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in agg.columns]
    return agg


def align_pair(left: pd.DataFrame, right: pd.DataFrame) -> tuple[PreparedPredictions, PreparedPredictions]:
    lcols = ["example_id", "split", "y_true", "y_prob", "y_pred_tuned"]
    rcols = ["example_id", "split", "y_true", "y_prob", "y_pred_tuned"]
    merged = left[lcols].merge(right[rcols], on=["example_id", "split"], suffixes=("_left", "_right"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Prediction example_id/split mismatch in paired comparison")
    if not (merged["y_true_left"].astype(int).to_numpy() == merged["y_true_right"].astype(int).to_numpy()).all():
        raise ValueError("Paired predictions disagree on y_true")
    left_prepared = PreparedPredictions.from_arrays(merged["y_true_left"], merged["y_prob_left"], merged["y_pred_tuned_left"])
    right_prepared = PreparedPredictions.from_arrays(merged["y_true_left"], merged["y_prob_right"], merged["y_pred_tuned_right"])
    return left_prepared, right_prepared


def paired_bootstrap(
    comparison: tuple[str, str],
    seeds: list[int],
    predictions: dict[tuple[str, int], pd.DataFrame],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[dict[str, float | int | str]]:
    left_key, right_key = comparison
    prepared_pairs = []
    observed_seed_metrics = []
    for seed in seeds:
        left = predictions[(left_key, seed)]
        right = predictions[(right_key, seed)]
        left_test = left[left["split"] == "test"].copy()
        right_test = right[right["split"] == "test"].copy()
        left_prepared, right_prepared = align_pair(left_test, right_test)
        prepared_pairs.append((left_prepared, right_prepared))
        weights = np.ones(len(left_prepared.y_true), dtype=float)
        l_metrics = all_metrics(left_prepared, weights)
        r_metrics = all_metrics(right_prepared, weights)
        observed_seed_metrics.append({metric: l_metrics[metric] - r_metrics[metric] for metric in METRICS})
    observed = {metric: float(np.mean([row[metric] for row in observed_seed_metrics])) for metric in METRICS}

    boot_values = {metric: [] for metric in METRICS}
    for _ in range(n_bootstrap):
        seed_deltas = {metric: [] for metric in METRICS}
        for left_prepared, right_prepared in prepared_pairs:
            n = len(left_prepared.y_true)
            indices = rng.integers(0, n, size=n)
            weights = np.bincount(indices, minlength=n).astype(float)
            l_metrics = all_metrics(left_prepared, weights)
            r_metrics = all_metrics(right_prepared, weights)
            for metric in METRICS:
                delta = l_metrics[metric] - r_metrics[metric]
                if not np.isnan(delta):
                    seed_deltas[metric].append(float(delta))
        for metric in METRICS:
            if seed_deltas[metric]:
                boot_values[metric].append(float(np.mean(seed_deltas[metric])))
    rows = []
    for metric in METRICS:
        values = np.asarray(boot_values[metric], dtype=float)
        if len(values) == 0:
            ci_low = np.nan
            ci_high = np.nan
        else:
            ci_low, ci_high = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "comparison": f"{left_key}_vs_{right_key}",
                "left_model": left_key,
                "right_model": right_key,
                "metric": metric,
                "higher_is_better": HIGHER_IS_BETTER[metric],
                "observed_delta_left_minus_right": observed[metric],
                "bootstrap_delta_mean": float(np.nanmean(values)) if len(values) else np.nan,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0) if len(values) else False,
                "n_bootstrap": int(n_bootstrap),
                "n_bootstrap_valid": int(len(values)),
                "n_seeds": int(len(seeds)),
            }
        )
    return rows


def fmt_mean_std(row: pd.Series, metric: str) -> str:
    return f"{float(row[f'{metric}_mean']):.4f} +/- {float(row[f'{metric}_std']):.4f}"


def write_markdown(main: pd.DataFrame, bootstrap: pd.DataFrame, output: Path) -> None:
    test = main[main["split"] == "test"].sort_values("AUPRC_mean", ascending=False)
    lvc = bootstrap[
        (bootstrap["left_model"] == "longformer_mean_4096")
        & (bootstrap["right_model"] == "clinicalmamba_4096")
        & (bootstrap["metric"].isin(["AUPRC", "AUROC", "F1"]))
    ]
    all_longformer_positive = bool((lvc["ci_low"] > 0).all()) if len(lvc) == 3 else False
    if all_longformer_positive:
        decision = "Longformer mean 4096 is the numerically and bootstrap-supported best no-DAPT overall comparator against ClinicalMamba for AUPRC, AUROC, and F1."
    else:
        decision = "Longformer mean 4096 and ClinicalMamba are statistically comparable under the paired bootstrap rule because at least one AUPRC/AUROC/F1 CI overlaps zero."
    metric_notes = []
    if len(lvc) == 3:
        for metric in ["AUPRC", "AUROC", "F1"]:
            row = lvc[lvc["metric"] == metric].iloc[0]
            if float(row["ci_low"]) > 0:
                metric_notes.append(f"{metric} favors Longformer")
            elif float(row["ci_high"]) < 0:
                metric_notes.append(f"{metric} favors ClinicalMamba")
            else:
                metric_notes.append(f"{metric} CI overlaps zero")
    metric_note = "Metric-level Longformer vs ClinicalMamba note: " + "; ".join(metric_notes) + "." if metric_notes else ""

    rows = [
        "# No-DAPT Full Comparison Frozen",
        "",
        "- Benchmark: LAMP-R v0.2 frozen",
        "- Input: discharge summary",
        "- Embedding: frozen cached note embedding",
        "- Head: MLP",
        "- Loss: natural BCE",
        "- Seeds: 17, 23, 42",
        "- Threshold: selected on validation F1 and applied to test",
        "- AUPRC/AUROC/Brier/ECE: computed from raw probabilities",
        "- Precision/Recall/F1: computed from the validation-tuned threshold",
        "- Delta convention: positive AUPRC/AUROC/F1/Precision/Recall deltas favor the left model; negative Brier/ECE deltas favor the left model.",
        "- Main Mamba-family no-DAPT baseline: ClinicalMamba",
        "- Numerically best full no-DAPT comparator: Longformer mean 4096",
        "- Cheap clinical encoder baseline: BioClinicalBERT mean 512",
        "- General Mamba baseline: Mamba2",
        "- ClinicalMamba is a strong public clinical baseline, but may have MIMIC-derived pretraining overlap; interpret as practical baseline comparison, not contamination-free architecture comparison.",
        "",
        "## Test Main Table",
        "",
        "| Model | Role | Max length | Pooling | AUPRC | AUROC | F1 | Precision | Recall | Brier | ECE |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in test.iterrows():
        rows.append(
            f"| {row['display_name']} | {row['role']} | {int(row['max_length'])} | {row['pooling']} | "
            f"{fmt_mean_std(row, 'AUPRC')} | {fmt_mean_std(row, 'AUROC')} | {fmt_mean_std(row, 'F1')} | "
            f"{float(row['Precision_mean']):.4f} | {float(row['Recall_mean']):.4f} | "
            f"{float(row['Brier_mean']):.4f} | {float(row['ECE_mean']):.4f} |"
        )
    rows.extend(
        [
            "",
            "## Bootstrap Decision",
            "",
            decision,
            metric_note,
            "",
            "| Comparison | Metric | Delta left-right | 95% CI | Excludes 0 |",
            "|---|---|---:|---:|---|",
        ]
    )
    focus = bootstrap[bootstrap["metric"].isin(["AUPRC", "AUROC", "F1", "Brier", "ECE"])].copy()
    for _, row in focus.iterrows():
        rows.append(
            f"| {row['left_model']} vs {row['right_model']} | {row['metric']} | "
            f"{float(row['observed_delta_left_minus_right']):.4f} | "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] | {row['ci_excludes_zero']} |"
        )
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze full no-DAPT comparison with paired bootstrap CIs.")
    parser.add_argument("--mamba2-dir", default="results/downstream_head_mlp_3seed/mamba2_last_full_len4096_sharded_mlp_natural_bce")
    parser.add_argument("--clinicalmamba-dir", default="results/downstream_head_clinicalmamba_3seed/clinicalmamba_last_full_len4096_sharded_mlp_natural_bce")
    parser.add_argument("--bioclinicalbert-dir", default="results/downstream_head_bioclinicalbert_mean_3seed/bioclinicalbert_mean_full_len512_sharded_mlp_natural_bce")
    parser.add_argument("--longformer-dir", default="results/downstream_head_longformer_mean_3seed/longformer_mean_full_len4096_sharded_mlp_natural_bce")
    parser.add_argument("--output-dir", default="results/no_dapt_full_comparison_frozen")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 42])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    parser.add_argument("--expected-val-n", type=int, default=DEFAULT_EXPECTED_EVAL_COUNTS["val"])
    parser.add_argument("--expected-test-n", type=int, default=DEFAULT_EXPECTED_EVAL_COUNTS["test"])
    parser.add_argument(
        "--skip-expected-count-validation",
        action="store_true",
        help="Skip frozen LAMP-R v0.2 val/test prediction row count validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = [
        ModelSpec("mamba2_4096", "Mamba2 130M", "General Mamba baseline", "discharge summary", 4096, "last_token", resolve_path(args.mamba2_dir)),
        ModelSpec("clinicalmamba_4096", "ClinicalMamba 130M", "Main Mamba-family no-DAPT baseline", "discharge summary", 4096, "last_token", resolve_path(args.clinicalmamba_dir)),
        ModelSpec("bioclinicalbert_512", "BioClinicalBERT", "Cheap clinical encoder baseline", "discharge summary", 512, "mean", resolve_path(args.bioclinicalbert_dir)),
        ModelSpec("longformer_mean_4096", "Longformer", "Non-Mamba long-context comparator", "discharge summary", 4096, "mean", resolve_path(args.longformer_dir)),
    ]
    expected_eval_counts = None
    if not args.skip_expected_count_validation:
        expected_eval_counts = {"val": args.expected_val_n, "test": args.expected_test_n}
    predictions = {}
    for spec in specs:
        for seed in args.seeds:
            predictions[(spec.key, seed)] = load_predictions(spec, seed, expected_eval_counts)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main = main_table(specs, args.seeds, predictions)
    main.to_csv(output_dir / "no_dapt_full_main_table.csv", index=False)

    calibration = calibration_summary(specs, args.seeds, predictions)
    calibration.to_csv(output_dir / "no_dapt_full_calibration_summary.csv", index=False)

    errors = error_summary(specs, args.seeds, predictions)
    errors.to_csv(output_dir / "no_dapt_full_error_summary.csv", index=False)

    comparisons = [
        ("longformer_mean_4096", "clinicalmamba_4096"),
        ("bioclinicalbert_512", "clinicalmamba_4096"),
        ("bioclinicalbert_512", "mamba2_4096"),
        ("longformer_mean_4096", "bioclinicalbert_512"),
    ]
    rng = np.random.default_rng(args.bootstrap_seed)
    boot_rows = []
    for comparison in comparisons:
        boot_rows.extend(paired_bootstrap(comparison, args.seeds, predictions, args.n_bootstrap, rng))
    bootstrap = pd.DataFrame(boot_rows)
    bootstrap.to_csv(output_dir / "no_dapt_full_delta_bootstrap.csv", index=False)

    write_markdown(main, bootstrap, output_dir / "NO_DAPT_FULL_COMPARISON_FROZEN.md")

    print(f"Wrote {output_dir / 'NO_DAPT_FULL_COMPARISON_FROZEN.md'}")
    print(f"Wrote {output_dir / 'no_dapt_full_main_table.csv'}")
    print(f"Wrote {output_dir / 'no_dapt_full_delta_bootstrap.csv'}")
    print(f"Wrote {output_dir / 'no_dapt_full_calibration_summary.csv'}")
    print(f"Wrote {output_dir / 'no_dapt_full_error_summary.csv'}")
    print(main[main["split"] == "test"].sort_values("AUPRC_mean", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
