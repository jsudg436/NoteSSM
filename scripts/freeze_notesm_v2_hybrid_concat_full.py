from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HELPER_SPEC = importlib.util.spec_from_file_location(
    "freeze_no_dapt_full_comparison",
    ROOT / "scripts" / "freeze_no_dapt_full_comparison.py",
)
helper = importlib.util.module_from_spec(HELPER_SPEC)
assert HELPER_SPEC.loader is not None
sys.modules[HELPER_SPEC.name] = helper
HELPER_SPEC.loader.exec_module(helper)


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
METRICS = ["AUPRC", "AUROC", "Precision", "Recall", "F1", "Brier", "ECE"]
MODEL_SPECS = {
    "concat_mlp": ("notesm_v2_hybrid", "Hybrid concat_mlp"),
    "current_section_aware": ("notesm_v2", "Current section-aware"),
    "dapt_lasttoken": ("notesm_v1", "DAPT last-token"),
}


def resolve_path(path_arg: str | Path) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def check_raw_text_free(path: Path, columns: set[str]) -> None:
    forbidden = FORBIDDEN_COLUMNS & columns
    if forbidden:
        raise ValueError(f"{path} contains forbidden raw columns: {sorted(forbidden)}")


def safe_copy_seed_artifacts(
    source_dir: Path,
    target_dir: Path,
    seed: int,
    model_prefix: str = "notesm_v2_hybrid",
    overwrite: bool = False,
) -> list[Path]:
    patterns = [
        f"{model_prefix}_tuned_predictions_seed{seed}.csv",
        f"{model_prefix}_threshold_tuned_metrics_seed{seed}.csv",
        f"{model_prefix}_threshold_tuning_summary_seed{seed}.csv",
        f"{model_prefix}_cached_predictions_seed{seed}.csv",
        f"{model_prefix}_cached_embedding_metrics_seed{seed}.csv",
        f"{model_prefix}_training_plan_seed{seed}.csv",
        f"{model_prefix}_cached_hybrid_head_seed{seed}.pt",
        f"{model_prefix}_threshold_grid_seed{seed}.csv",
    ]
    copied: list[Path] = []
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in patterns:
        source = source_dir / name
        if not source.exists():
            continue
        target = target_dir / name
        if target.exists():
            if source.resolve() == target.resolve():
                continue
            if not overwrite and source.read_bytes() != target.read_bytes():
                raise FileExistsError(f"Refusing to overwrite existing different seed artifact: {target}")
        shutil.copy2(source, target)
        copied.append(target)
    required = [
        target_dir / f"{model_prefix}_tuned_predictions_seed{seed}.csv",
        target_dir / f"{model_prefix}_threshold_tuned_metrics_seed{seed}.csv",
        target_dir / f"{model_prefix}_threshold_tuning_summary_seed{seed}.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required seed{seed} artifacts after copy: {missing}")
    return copied


def load_fallback_map(section_cache_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ["train", "val", "test"]:
        path = section_cache_dir / f"section_cache_{split}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["example_id", "split", "use_chunk_fallback"])
        check_raw_text_free(path, set(frame.columns))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No section_cache_*.parquet files found under {section_cache_dir}")
    sections = pd.concat(frames, ignore_index=True)
    sections["example_id"] = sections["example_id"].astype(str)
    sections["split"] = sections["split"].astype(str)
    sections["use_chunk_fallback"] = sections["use_chunk_fallback"].astype(bool)
    fallback = sections.groupby(["example_id", "split"], as_index=False).agg(use_chunk_fallback=("use_chunk_fallback", "any"))
    fallback["fallback_group"] = np.where(fallback["use_chunk_fallback"], "fallback", "non_fallback")
    return fallback


def prediction_file(directory: Path, model_prefix: str, seed: int) -> Path:
    path = directory / f"{model_prefix}_tuned_predictions_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing tuned prediction CSV: {path}")
    return path


def load_predictions(
    directory: Path,
    model_prefix: str,
    setting: str,
    seed: int,
    fallback: pd.DataFrame,
    expected_counts: dict[str, int],
) -> pd.DataFrame:
    path = prediction_file(directory, model_prefix, seed)
    frame = pd.read_csv(path)
    required = {"example_id", "split", "y_true", "y_prob", "y_pred_tuned", "threshold"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    check_raw_text_free(path, set(frame.columns))
    frame = frame.copy()
    frame["example_id"] = frame["example_id"].astype(str)
    frame["split"] = frame["split"].astype(str)
    frame["y_true"] = frame["y_true"].astype(int)
    frame["y_prob"] = frame["y_prob"].astype(float)
    frame["y_pred_tuned"] = frame["y_pred_tuned"].astype(int)
    frame["threshold"] = frame["threshold"].astype(float)
    if frame.duplicated(["example_id", "split"]).any():
        raise ValueError(f"{path} contains duplicated example_id/split rows")
    if not set(frame["split"]).issubset({"val", "test"}):
        raise ValueError(f"{path} contains non-evaluation split values")
    if not frame["y_prob"].between(0.0, 1.0).all():
        raise ValueError(f"{path} contains y_prob outside [0, 1]")
    if not set(frame["y_pred_tuned"].astype(int).tolist()).issubset({0, 1}):
        raise ValueError(f"{path} contains non-binary tuned predictions")
    for split, expected in expected_counts.items():
        split_frame = frame[frame["split"] == split]
        if len(split_frame) != expected:
            raise ValueError(f"{path} has {len(split_frame)} {split} rows, expected {expected}")
        helper.constant_threshold(split_frame, setting, seed, split, path)
    merged = frame.merge(fallback, on=["example_id", "split"], how="left", validate="many_to_one")
    if merged["fallback_group"].isna().any():
        missing_ids = merged.loc[merged["fallback_group"].isna(), "example_id"].head(5).tolist()
        raise ValueError(f"{path} contains example_ids absent from fallback map: {missing_ids}")
    merged["setting"] = setting
    merged["display_name"] = MODEL_SPECS[setting][1]
    merged["seed"] = seed
    return merged


def enforce_pairwise_consistency(predictions: dict[tuple[str, int], pd.DataFrame], seeds: list[int]) -> None:
    for seed in seeds:
        base = predictions[("concat_mlp", seed)][["example_id", "split", "y_true"]].sort_values(["split", "example_id"]).reset_index(drop=True)
        for setting in ["current_section_aware", "dapt_lasttoken"]:
            other = predictions[(setting, seed)][["example_id", "split", "y_true"]].sort_values(["split", "example_id"]).reset_index(drop=True)
            if not base[["example_id", "split"]].equals(other[["example_id", "split"]]):
                raise ValueError(f"Prediction example_id/split mismatch for seed={seed} setting={setting}")
            if not (base["y_true"].astype(int).to_numpy() == other["y_true"].astype(int).to_numpy()).all():
                raise ValueError(f"Prediction y_true mismatch for seed={seed} setting={setting}")


def threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(float(len(y_true)), 1.0)
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def metrics_for(frame: pd.DataFrame, setting: str, seed: int, split: str, fallback_group: str) -> dict[str, object]:
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_prob = frame["y_prob"].to_numpy(dtype=float)
    y_pred = frame["y_pred_tuned"].to_numpy(dtype=int)
    thresholds = frame["threshold"].dropna().astype(float).unique()
    row: dict[str, object] = {
        "setting": setting,
        "display_name": MODEL_SPECS[setting][1],
        "seed": seed,
        "split": split,
        "fallback_group": fallback_group,
        "n": int(len(frame)),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()) if len(frame) else np.nan,
        "threshold": float(thresholds[0]) if len(thresholds) == 1 else np.nan,
        "predicted_positive": int(y_pred.sum()),
    }
    row.update(threshold_metrics(y_true, y_pred))
    prepared = helper.PreparedPredictions.from_arrays(y_true, y_prob, y_pred)
    weights = np.ones(len(frame), dtype=float)
    row["AUPRC"] = helper.weighted_auprc(prepared, weights)
    row["AUROC"] = helper.weighted_auroc(prepared, weights)
    row["Brier"] = helper.weighted_brier(prepared, weights)
    row["ECE"] = helper.weighted_ece(prepared, weights)
    return row


def per_seed_tables(predictions: dict[tuple[str, int], pd.DataFrame], seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for setting in ["concat_mlp", "current_section_aware", "dapt_lasttoken"]:
        for seed in seeds:
            frame = predictions[(setting, seed)]
            for split in ["val", "test"]:
                split_frame = frame[frame["split"] == split].copy()
                rows.append(metrics_for(split_frame, setting, seed, split, "all"))
                for group_name, group in split_frame.groupby("fallback_group", sort=True):
                    rows.append(metrics_for(group, setting, seed, split, str(group_name)))
    table = pd.DataFrame(rows)
    return table[table["fallback_group"] == "all"].copy(), table[table["fallback_group"].isin(["fallback", "non_fallback"])].copy()


def aggregate_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    numeric = ["n", "positives", "prevalence", "threshold", "predicted_positive", "Accuracy", *METRICS, "tp", "fp", "tn", "fn"]
    rows = []
    for keys, group in per_seed.groupby(["setting", "display_name", "split", "fallback_group"], sort=True):
        setting, display_name, split, fallback_group = keys
        row: dict[str, object] = {
            "setting": setting,
            "display_name": display_name,
            "split": split,
            "fallback_group": fallback_group,
            "n_seeds": int(group["seed"].nunique()),
        }
        for col in numeric:
            row[f"{col}_mean"] = float(group[col].astype(float).mean())
            row[f"{col}_std"] = float(group[col].astype(float).std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def delta_table(aggregate: pd.DataFrame, reference: str, suffix: str) -> pd.DataFrame:
    ref = aggregate[(aggregate["setting"] == reference) & (aggregate["split"] == "test")]
    rows = []
    concat = aggregate[(aggregate["setting"] == "concat_mlp") & (aggregate["split"] == "test")]
    for _, row in concat.iterrows():
        match = ref[ref["fallback_group"] == row["fallback_group"]]
        if match.empty:
            continue
        ref_row = match.iloc[0]
        out = {"setting": "concat_mlp", "split": "test", "fallback_group": row["fallback_group"], "reference": reference}
        for metric in METRICS:
            out[f"{metric}_concat_mean"] = float(row[f"{metric}_mean"])
            out[f"{metric}_{suffix}_mean"] = float(ref_row[f"{metric}_mean"])
            out[f"{metric}_delta_vs_{suffix}"] = float(row[f"{metric}_mean"] - ref_row[f"{metric}_mean"])
        rows.append(out)
    return pd.DataFrame(rows)


def bootstrap_metric(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    prepared = helper.PreparedPredictions.from_arrays(y_true, y_score, y_pred)
    weights = np.ones(len(y_true), dtype=float)
    return helper.all_metrics(prepared, weights)[metric]


def paired_bootstrap(
    predictions: dict[tuple[str, int], pd.DataFrame],
    right_setting: str,
    seeds: list[int],
    n_bootstrap: int,
    rng_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    observed = {metric: [] for metric in METRICS}
    boot = {metric: [] for metric in METRICS}
    prepared = []
    for seed in seeds:
        left = predictions[("concat_mlp", seed)]
        right = predictions[(right_setting, seed)]
        merged = left[left["split"] == "test"].merge(
            right[right["split"] == "test"],
            on=["example_id", "split", "y_true"],
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        prepared.append(merged)
        y_true = merged["y_true"].to_numpy(dtype=int)
        for metric in METRICS:
            left_value = bootstrap_metric(y_true, merged["y_prob_left"].to_numpy(float), merged["y_pred_tuned_left"].to_numpy(int), metric)
            right_value = bootstrap_metric(y_true, merged["y_prob_right"].to_numpy(float), merged["y_pred_tuned_right"].to_numpy(int), metric)
            observed[metric].append(left_value - right_value)
    for _ in range(n_bootstrap):
        for metric in METRICS:
            seed_deltas = []
            for merged in prepared:
                idx = rng.integers(0, len(merged), size=len(merged))
                sample = merged.iloc[idx]
                y_true = sample["y_true"].to_numpy(dtype=int)
                left_value = bootstrap_metric(y_true, sample["y_prob_left"].to_numpy(float), sample["y_pred_tuned_left"].to_numpy(int), metric)
                right_value = bootstrap_metric(y_true, sample["y_prob_right"].to_numpy(float), sample["y_pred_tuned_right"].to_numpy(int), metric)
                seed_deltas.append(left_value - right_value)
            boot[metric].append(float(np.mean(seed_deltas)))
    rows = []
    for metric in METRICS:
        values = np.asarray(boot[metric], dtype=float)
        rows.append(
            {
                "comparison": f"concat_mlp_minus_{right_setting}",
                "left_setting": "concat_mlp",
                "right_setting": right_setting,
                "metric": metric,
                "observed_delta_mean": float(np.mean(observed[metric])),
                "bootstrap_delta_mean": float(values.mean()),
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "ci_excludes_zero": bool(np.quantile(values, 0.025) > 0 or np.quantile(values, 0.975) < 0),
                "n_bootstrap": int(n_bootstrap),
                "n_seeds": int(len(seeds)),
            }
        )
    return pd.DataFrame(rows)


def artifact_validation(concat_dir: Path, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        path = concat_dir / f"hybrid_output_validation_seed{seed}.csv"
        if not path.exists():
            rows.append({"artifact": "validation_file", "seed": seed, "validation_passed": False, "reason": f"missing {path}"})
            continue
        frame = pd.read_csv(path)
        check_raw_text_free(path, set(frame.columns))
        if "validation_passed" in frame.columns and not frame["validation_passed"].map(lambda x: str(x).lower() in {"true", "1"}).all():
            rows.append({"artifact": "validation_file", "seed": seed, "validation_passed": False, "reason": "contains failed validation rows"})
        else:
            rows.append({"artifact": "validation_file", "seed": seed, "validation_passed": True, "reason": ""})
    return pd.DataFrame(rows)


def decision_table(aggregate: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    test = aggregate[aggregate["split"] == "test"].copy()
    def row(setting: str, group: str) -> pd.Series:
        match = test[(test["setting"] == setting) & (test["fallback_group"] == group)]
        if match.empty:
            raise ValueError(f"Missing aggregate row setting={setting} group={group}")
        return match.iloc[0]

    concat_all = row("concat_mlp", "all")
    current_all = row("current_section_aware", "all")
    concat_nf = row("concat_mlp", "non_fallback")
    current_nf = row("current_section_aware", "non_fallback")
    concat_fb = row("concat_mlp", "fallback")
    current_fb = row("current_section_aware", "fallback")
    auprc_delta = float(concat_all["AUPRC_mean"] - current_all["AUPRC_mean"])
    auroc_delta = float(concat_all["AUROC_mean"] - current_all["AUROC_mean"])
    f1_delta = float(concat_all["F1_mean"] - current_all["F1_mean"])
    brier_delta = float(concat_all["Brier_mean"] - current_all["Brier_mean"])
    ece_delta = float(concat_all["ECE_mean"] - current_all["ECE_mean"])
    non_fallback_f1_delta = float(concat_nf["F1_mean"] - current_nf["F1_mean"])
    non_fallback_recall_delta = float(concat_nf["Recall_mean"] - current_nf["Recall_mean"])
    fallback_f1_delta = float(concat_fb["F1_mean"] - current_fb["F1_mean"])
    key_bootstrap = bootstrap[
        (bootstrap["comparison"] == "concat_mlp_minus_current_section_aware")
        & (bootstrap["metric"].isin(["AUPRC", "AUROC", "F1"]))
        & (bootstrap["ci_lower"].astype(float) > 0)
    ]
    passes_overall = bool(
        auprc_delta >= 0
        and auroc_delta >= 0
        and f1_delta >= -0.002
        and brier_delta <= 0.001
        and ece_delta <= 0.005
    )
    passes_non_fallback = bool(non_fallback_f1_delta >= 0.05 or non_fallback_recall_delta >= 0.10)
    passes_fallback = bool(fallback_f1_delta >= -0.01)
    passes_bootstrap = bool(not key_bootstrap.empty)
    if passes_overall and passes_non_fallback and passes_fallback and passes_bootstrap:
        category = "confirmed_hybrid_candidate"
    elif passes_non_fallback and (not passes_overall or not passes_bootstrap):
        category = "diagnostic_only"
    else:
        category = "reject"
    return pd.DataFrame(
        [
            {
                "setting": "concat_mlp",
                "AUPRC_delta_vs_current": auprc_delta,
                "AUROC_delta_vs_current": auroc_delta,
                "F1_delta_vs_current": f1_delta,
                "Brier_delta_vs_current": brier_delta,
                "ECE_delta_vs_current": ece_delta,
                "non_fallback_F1_delta_vs_current": non_fallback_f1_delta,
                "non_fallback_Recall_delta_vs_current": non_fallback_recall_delta,
                "fallback_F1_delta_vs_current": fallback_f1_delta,
                "passes_overall_rule": passes_overall,
                "passes_non_fallback_rule": passes_non_fallback,
                "passes_fallback_rule": passes_fallback,
                "passes_key_bootstrap_rule": passes_bootstrap,
                "decision_category": category,
            }
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(output: Path, main: pd.DataFrame, groups: pd.DataFrame, delta_current: pd.DataFrame, bootstrap: pd.DataFrame, decision: pd.DataFrame) -> None:
    test_main = main[main["split"] == "test"].copy()
    test_groups = groups[groups["split"] == "test"].copy()
    lines = [
        "# NoteSSM-v2 Hybrid Concat Full Frozen",
        "",
        "- Benchmark: LAMP-R v0.2 frozen.",
        "- Model: NoteSSM-v2 hybrid concat_mlp.",
        "- Inputs: existing full section/chunk embedding cache plus existing DAPT last-token note embedding cache.",
        "- Head/loss: MLP + natural BCE.",
        "- Seeds: 17, 23, 42.",
        "- Threshold: selected on validation F1 and applied to test.",
        "- AUPRC/AUROC/Brier/ECE use raw probabilities; Precision/Recall/F1 use validation-tuned threshold.",
        "- This hybrid design was motivated by prior branch-specific diagnostic analysis.",
        "- Interpret as a diagnostic-motivated follow-up, not an unbiased final benchmark improvement.",
        "- LAMP-R v0.2 is unchanged. No new embeddings or DAPT training were run.",
        "- Post-hoc branch policy is diagnostic context only, not a trained model.",
        "- No raw clinical text is included in this report.",
        "",
        "## Main Test Table",
        "",
    ]
    main_cols = ["setting", "display_name", "AUPRC_mean", "AUPRC_std", "AUROC_mean", "AUROC_std", "Precision_mean", "Recall_mean", "F1_mean", "F1_std", "Brier_mean", "ECE_mean", "threshold_mean"]
    lines.append(markdown_table(test_main[main_cols].sort_values("AUPRC_mean", ascending=False), main_cols))
    lines.extend(["", "## Group Test Table", ""])
    group_cols = ["setting", "fallback_group", "AUPRC_mean", "AUROC_mean", "Precision_mean", "Recall_mean", "F1_mean", "Brier_mean", "ECE_mean"]
    lines.append(markdown_table(test_groups[group_cols].sort_values(["fallback_group", "F1_mean"], ascending=[True, False]), group_cols))
    lines.extend(["", "## Delta vs Current Section-Aware", ""])
    delta_cols = ["setting", "fallback_group", "AUPRC_delta_vs_current", "AUROC_delta_vs_current", "F1_delta_vs_current", "Brier_delta_vs_current", "ECE_delta_vs_current"]
    lines.append(markdown_table(delta_current[delta_cols], delta_cols))
    lines.extend(["", "## Paired Bootstrap", ""])
    boot_cols = ["comparison", "metric", "observed_delta_mean", "ci_lower", "ci_upper", "ci_excludes_zero"]
    lines.append(markdown_table(bootstrap[bootstrap["metric"].isin(METRICS)][boot_cols], boot_cols))
    lines.extend(["", "## Decision", ""])
    dcols = [
        "setting",
        "AUPRC_delta_vs_current",
        "AUROC_delta_vs_current",
        "F1_delta_vs_current",
        "non_fallback_F1_delta_vs_current",
        "fallback_F1_delta_vs_current",
        "passes_key_bootstrap_rule",
        "decision_category",
    ]
    lines.append(markdown_table(decision[dcols], dcols))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze 3-seed full NoteSSM-v2 hybrid concat_mlp comparison.")
    parser.add_argument("--concat-dir", default="results/downstream_notesm_v2_hybrid_full/concat_mlp_natural_bce")
    parser.add_argument("--seed17-source-dir", default=None)
    parser.add_argument("--current-section-dir", default="results/downstream_notesm_v2_full/dapt1epoch_mamba2_section_full_mlp_natural_bce")
    parser.add_argument("--dapt-lasttoken-dir", default="results/downstream_head_dapt1epoch_mamba2_3seed/dapt1epoch_mamba2_last_full_len4096_mlp_natural_bce")
    parser.add_argument("--section-cache-dir", default="results/notesm_v2_section_cache/lampr_v0_2_full")
    parser.add_argument("--output-dir", default="results/notesm_v2_hybrid_concat_full_frozen")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 42])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument("--expected-val-rows", type=int, default=29298)
    parser.add_argument("--expected-test-rows", type=int, default=29460)
    parser.add_argument("--overwrite-seed17-copy", action="store_true")
    parser.add_argument("--copy-seed17-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    concat_dir = resolve_path(args.concat_dir)
    if args.seed17_source_dir is not None:
        copied = safe_copy_seed_artifacts(resolve_path(args.seed17_source_dir), concat_dir, 17, overwrite=args.overwrite_seed17_copy)
        print(f"Copied/verified seed17 artifacts into {concat_dir}: {len(copied)} files")
    if args.copy_seed17_only:
        return 0
    fallback = load_fallback_map(resolve_path(args.section_cache_dir))
    expected_counts = {"val": args.expected_val_rows, "test": args.expected_test_rows}
    directories = {
        "concat_mlp": concat_dir,
        "current_section_aware": resolve_path(args.current_section_dir),
        "dapt_lasttoken": resolve_path(args.dapt_lasttoken_dir),
    }
    predictions: dict[tuple[str, int], pd.DataFrame] = {}
    for setting, directory in directories.items():
        model_prefix, _ = MODEL_SPECS[setting]
        for seed in args.seeds:
            predictions[(setting, seed)] = load_predictions(directory, model_prefix, setting, seed, fallback, expected_counts)
    enforce_pairwise_consistency(predictions, args.seeds)

    main_seed, group_seed = per_seed_tables(predictions, args.seeds)
    main = aggregate_table(main_seed)
    groups = aggregate_table(group_seed)
    delta_current = delta_table(pd.concat([main, groups], ignore_index=True), "current_section_aware", "current")
    delta_dapt = delta_table(pd.concat([main, groups], ignore_index=True), "dapt_lasttoken", "dapt_lasttoken")
    bootstrap = pd.concat(
        [
            paired_bootstrap(predictions, "current_section_aware", args.seeds, args.n_bootstrap, args.bootstrap_seed),
            paired_bootstrap(predictions, "dapt_lasttoken", args.seeds, args.n_bootstrap, args.bootstrap_seed + 1),
        ],
        ignore_index=True,
    )
    validation = artifact_validation(concat_dir, args.seeds)
    if not validation["validation_passed"].astype(bool).all():
        raise ValueError("Hybrid concat artifact validation is incomplete or failed")
    decision = decision_table(pd.concat([main, groups], ignore_index=True), bootstrap)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main.to_csv(output_dir / "notesm_v2_hybrid_concat_main_table.csv", index=False)
    groups.to_csv(output_dir / "notesm_v2_hybrid_concat_group_table.csv", index=False)
    delta_current.to_csv(output_dir / "notesm_v2_hybrid_concat_delta_vs_current.csv", index=False)
    delta_dapt.to_csv(output_dir / "notesm_v2_hybrid_concat_delta_vs_dapt_lasttoken.csv", index=False)
    bootstrap.to_csv(output_dir / "notesm_v2_hybrid_concat_bootstrap.csv", index=False)
    validation.to_csv(output_dir / "notesm_v2_hybrid_concat_artifact_validation.csv", index=False)
    decision.to_csv(output_dir / "notesm_v2_hybrid_concat_decision.csv", index=False)
    write_markdown(output_dir / "NOTESM_V2_HYBRID_CONCAT_FULL_FROZEN.md", main, groups, delta_current, bootstrap, decision)
    print(f"Wrote NoteSSM-v2 hybrid concat full frozen outputs to {output_dir}")
    print(decision.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
