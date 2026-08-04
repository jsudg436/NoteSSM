from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEEDS = (17, 23, 42)
METRICS = ("AUPRC", "AUROC", "Accuracy", "Precision", "Recall", "F1", "Brier", "ECE")
HIGHER_IS_BETTER = {
    "AUPRC": True,
    "AUROC": True,
    "Accuracy": True,
    "Precision": True,
    "Recall": True,
    "F1": True,
    "Brier": False,
    "ECE": False,
}
FORBIDDEN_COLUMNS = {
    "text",
    "sequence_text",
    "merged_section_text",
    "section_text",
    "chunk_text",
    "subject_id",
    "hadm_id",
    "note_id",
    "patient_id",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    directory: Path
    filename_pattern: str

    def prediction_path(self, seed: int) -> Path:
        return self.directory / self.filename_pattern.format(seed=seed)


class MetricComputer:
    """Compute metrics from integer bootstrap multiplicities without expanding rows."""

    def __init__(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> None:
        self.y = np.asarray(y_true, dtype=np.float64)
        self.p = np.asarray(y_prob, dtype=np.float64)
        self.pred = np.asarray(y_pred, dtype=np.float64)
        self.ap_order, self.ap_starts = self._score_groups(descending=True)
        self.auc_order, self.auc_starts = self._score_groups(descending=False)
        self.ece_bins = np.minimum((self.p * 10).astype(np.int32), 9)

    def _score_groups(self, *, descending: bool) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(self.p, kind="mergesort")
        if descending:
            order = order[::-1]
        sorted_prob = self.p[order]
        starts = np.r_[0, np.flatnonzero(sorted_prob[1:] != sorted_prob[:-1]) + 1]
        return order, starts.astype(np.int32)

    def compute(self, weights: np.ndarray) -> dict[str, np.ndarray]:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim == 1:
            w = w[None, :]
        if w.shape[1] != len(self.y):
            raise ValueError(f"Weight width {w.shape[1]} does not match prediction rows {len(self.y)}")

        total = w.sum(axis=1)
        pos_total = w @ self.y
        neg_total = total - pos_total
        tp = w @ (self.y * self.pred)
        fp = w @ ((1.0 - self.y) * self.pred)
        fn = pos_total - tp
        tn = neg_total - fp

        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(tp),
            where=(precision + recall) > 0,
        )
        accuracy = np.divide(tp + tn, total, out=np.full_like(total, np.nan), where=total > 0)
        brier = np.divide(
            w @ ((self.p - self.y) ** 2),
            total,
            out=np.full_like(total, np.nan),
            where=total > 0,
        )

        ece_numerator = np.zeros(len(w), dtype=np.float64)
        for bin_id in range(10):
            mask = (self.ece_bins == bin_id).astype(np.float64)
            observed = w @ (mask * self.y)
            predicted = w @ (mask * self.p)
            ece_numerator += np.abs(observed - predicted)
        ece = np.divide(
            ece_numerator,
            total,
            out=np.full_like(total, np.nan),
            where=total > 0,
        )

        ap_w = w[:, self.ap_order]
        ap_y = self.y[self.ap_order]
        ap_group_pos = np.add.reduceat(ap_w * ap_y, self.ap_starts, axis=1)
        ap_group_total = np.add.reduceat(ap_w, self.ap_starts, axis=1)
        ap_cum_pos = np.cumsum(ap_group_pos, axis=1)
        ap_cum_total = np.cumsum(ap_group_total, axis=1)
        ap_precision = np.divide(
            ap_cum_pos,
            ap_cum_total,
            out=np.zeros_like(ap_cum_pos),
            where=ap_cum_total > 0,
        )
        auprc = np.divide(
            np.sum(ap_precision * ap_group_pos, axis=1),
            pos_total,
            out=np.full_like(pos_total, np.nan),
            where=pos_total > 0,
        )

        auc_w = w[:, self.auc_order]
        auc_y = self.y[self.auc_order]
        auc_group_pos = np.add.reduceat(auc_w * auc_y, self.auc_starts, axis=1)
        auc_group_neg = np.add.reduceat(auc_w * (1.0 - auc_y), self.auc_starts, axis=1)
        neg_before = np.cumsum(auc_group_neg, axis=1) - auc_group_neg
        auc_numerator = np.sum(auc_group_pos * (neg_before + 0.5 * auc_group_neg), axis=1)
        auroc = np.divide(
            auc_numerator,
            pos_total * neg_total,
            out=np.full_like(pos_total, np.nan),
            where=(pos_total > 0) & (neg_total > 0),
        )

        return {
            "AUPRC": auprc,
            "AUROC": auroc,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Brier": brier,
            "ECE": ece,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze seed-matched, patient-cluster paired-bootstrap confidence intervals for "
            "multi-span NoteSSM versus Longformer and ClinicalMamba."
        )
    )
    parser.add_argument("--project-root", default=None, help="Project root. Defaults to the repository root when run from scripts/.")
    parser.add_argument(
        "--notesm-dir",
        default="results/frozen/dapt_mamba3_joint_D_final_valtest_3seed_aux00_20260803/artifact",
    )
    parser.add_argument(
        "--notesm-pattern",
        default="notesm_v2_hybrid_tuned_predictions_seed{seed}.csv",
    )
    parser.add_argument(
        "--longformer-dir",
        default=(
            "results/downstream_head_longformer_mean_3seed/"
            "longformer_mean_full_len4096_sharded_mlp_natural_bce"
        ),
    )
    parser.add_argument(
        "--clinicalmamba-dir",
        default=(
            "results/downstream_head_clinicalmamba_3seed/"
            "clinicalmamba_last_full_len4096_sharded_mlp_natural_bce"
        ),
    )
    parser.add_argument(
        "--baseline-pattern",
        default="notesm_v1_tuned_predictions_seed{seed}.csv",
    )
    parser.add_argument(
        "--manifest",
        default="results/lampr_full_v0_2/benchmark_manifest_no_text.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="results/notesm_multispan_external_patient_bootstrap_frozen",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--bootstrap-batch-size", type=int, default=20)
    parser.add_argument("--expected-test-n", type=int, default=29460)
    parser.add_argument("--expected-test-patients", type=int, default=13454)
    return parser.parse_args()


def find_project_root(explicit_root: str | None) -> Path:
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        if not (root / "scripts").is_dir():
            raise FileNotFoundError(f"--project-root does not contain scripts/: {root}")
        return root
    candidates = [Path.cwd().resolve(), Path(__file__).resolve().parents[1]]
    for root in candidates:
        if (root / "scripts").is_dir() and (root / "results").is_dir():
            return root
    raise FileNotFoundError("Could not locate the project root. Pass --project-root PATH.")


def resolve_under_root(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_prediction(path: Path, expected_test_n: int) -> tuple[pd.DataFrame, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing tuned prediction CSV: {path}")
    frame = pd.read_csv(path)
    required = {"example_id", "split", "y_true", "y_prob", "y_pred_tuned"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    forbidden = FORBIDDEN_COLUMNS & set(frame.columns)
    if forbidden:
        raise ValueError(f"{path} contains forbidden identifiers or raw text: {sorted(forbidden)}")
    if frame.duplicated(["example_id", "split"]).any():
        raise ValueError(f"{path} contains duplicate example_id/split rows")

    test = frame[frame["split"].astype(str).eq("test")].copy()
    if len(test) != expected_test_n:
        raise ValueError(f"Expected {expected_test_n:,} test rows in {path}, found {len(test):,}")
    test["example_id"] = test["example_id"].astype(str)
    test["y_true"] = pd.to_numeric(test["y_true"], errors="raise").astype(np.int8)
    test["y_prob"] = pd.to_numeric(test["y_prob"], errors="raise").astype(np.float64)
    test["y_pred_tuned"] = pd.to_numeric(test["y_pred_tuned"], errors="raise").astype(np.int8)
    if test["example_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate test example_id values")
    if not set(test["y_true"].unique()).issubset({0, 1}):
        raise ValueError(f"{path} has non-binary y_true values")
    if not set(test["y_pred_tuned"].unique()).issubset({0, 1}):
        raise ValueError(f"{path} has non-binary y_pred_tuned values")
    if not np.isfinite(test["y_prob"].to_numpy()).all():
        raise ValueError(f"{path} has non-finite y_prob values")
    if not test["y_prob"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"{path} has y_prob values outside [0, 1]")

    threshold = np.nan
    if "threshold" in test.columns:
        values = pd.to_numeric(test["threshold"], errors="raise").drop_duplicates()
        if len(values) != 1:
            raise ValueError(f"Expected one validation-selected threshold in {path}, found {len(values)}")
        threshold = float(values.iloc[0])

    inventory = {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "columns": "|".join(map(str, frame.columns)),
        "test_n": int(len(test)),
        "test_positive_n": int(test["y_true"].sum()),
        "test_prevalence": float(test["y_true"].mean()),
        "threshold": threshold,
        "raw_text_and_direct_identifier_free": True,
    }
    return test[["example_id", "y_true", "y_prob", "y_pred_tuned"]], inventory


def load_and_align_predictions(
    specs: tuple[ModelSpec, ...],
    seeds: list[int],
    expected_test_n: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    pd.DataFrame,
]:
    loaded: dict[tuple[str, int], pd.DataFrame] = {}
    inventory_rows: list[dict[str, object]] = []
    for spec in specs:
        for seed in seeds:
            path = spec.prediction_path(seed)
            frame, inventory = load_prediction(path, expected_test_n)
            loaded[(spec.key, seed)] = frame
            inventory_rows.append(
                {"model_key": spec.key, "display_name": spec.display_name, "seed": seed, **inventory}
            )

    canonical_key = (specs[0].key, seeds[0])
    canonical = loaded[canonical_key].sort_values("example_id").reset_index(drop=True)
    example_ids = canonical["example_id"].to_numpy(dtype=str)
    y_true = canonical["y_true"].to_numpy(dtype=np.int8)
    expected_ids = set(example_ids)
    arrays: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    for key, frame in loaded.items():
        candidate_ids = set(frame["example_id"].astype(str))
        if candidate_ids != expected_ids:
            missing = len(expected_ids - candidate_ids)
            extra = len(candidate_ids - expected_ids)
            raise ValueError(f"Prediction example_id set mismatch for {key}: missing={missing}, extra={extra}")
        aligned = frame.set_index("example_id").reindex(example_ids)
        candidate_y = aligned["y_true"].to_numpy(dtype=np.int8)
        if not np.array_equal(candidate_y, y_true):
            raise ValueError(f"y_true mismatch after alignment for {key}")
        arrays[key] = (
            aligned["y_prob"].to_numpy(dtype=np.float64),
            aligned["y_pred_tuned"].to_numpy(dtype=np.int8),
        )
    return example_ids, y_true, arrays, pd.DataFrame(inventory_rows)


def load_patient_index(
    manifest_path: Path,
    example_ids: np.ndarray,
    expected_test_n: int,
    expected_test_patients: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing benchmark manifest: {manifest_path}")
    manifest = pd.read_parquet(manifest_path, columns=["example_id", "split", "subject_id_hash"])
    test = manifest[manifest["split"].astype(str).eq("test")].copy()
    test["example_id"] = test["example_id"].astype(str)
    if len(test) != expected_test_n:
        raise ValueError(
            f"Expected {expected_test_n:,} test rows in benchmark manifest, found {len(test):,}"
        )
    if test["example_id"].duplicated().any():
        raise ValueError("Benchmark manifest contains duplicate test example_id values")
    if set(test["example_id"]) != set(example_ids):
        raise ValueError("Benchmark manifest and predictions do not have the same test example_id set")
    aligned = test.set_index("example_id").reindex(example_ids)
    if aligned["subject_id_hash"].isna().any():
        raise ValueError("Missing subject_id_hash after aligning benchmark manifest")
    subject_hashes = aligned["subject_id_hash"].astype(str).to_numpy()
    unique_subjects, patient_index = np.unique(subject_hashes, return_inverse=True)
    if expected_test_patients > 0 and len(unique_subjects) != expected_test_patients:
        raise ValueError(
            f"Expected {expected_test_patients:,} unique test patients, found {len(unique_subjects):,}"
        )
    info = {
        "path": str(manifest_path),
        "size_bytes": int(manifest_path.stat().st_size),
        "sha256": sha256_file(manifest_path),
        "test_n": int(len(test)),
        "test_patient_n": int(len(unique_subjects)),
        "patient_column": "subject_id_hash",
    }
    return unique_subjects, patient_index.astype(np.int32), info


def observed_tables(
    specs: tuple[ModelSpec, ...],
    seeds: list[int],
    y_true: np.ndarray,
    computers: dict[tuple[str, int], MetricComputer],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ones = np.ones((1, len(y_true)), dtype=np.float64)
    rows = []
    display = {spec.key: spec.display_name for spec in specs}
    for spec in specs:
        for seed in seeds:
            values = computers[(spec.key, seed)].compute(ones)
            rows.append(
                {
                    "model_key": spec.key,
                    "display_name": display[spec.key],
                    "seed": seed,
                    "split": "test",
                    **{metric: float(values[metric][0]) for metric in METRICS},
                }
            )
    per_seed = pd.DataFrame(rows)
    aggregate_rows = []
    for (model_key, display_name), group in per_seed.groupby(["model_key", "display_name"], sort=False):
        row: dict[str, object] = {
            "model_key": model_key,
            "display_name": display_name,
            "split": "test",
            "n_seeds": len(group),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows).sort_values("AUPRC_mean", ascending=False)
    return per_seed, aggregate


def sample_patient_cluster_weights(
    rng: np.random.Generator,
    patient_index: np.ndarray,
    n_patients: int,
    n_replicates: int,
) -> np.ndarray:
    weights = np.empty((n_replicates, len(patient_index)), dtype=np.float64)
    for row in weights:
        sampled_patients = rng.integers(0, n_patients, size=n_patients)
        patient_multiplicity = np.bincount(sampled_patients, minlength=n_patients)
        row[:] = patient_multiplicity[patient_index]
    return weights


def run_patient_cluster_bootstrap(
    specs: tuple[ModelSpec, ...],
    seeds: list[int],
    y_true: np.ndarray,
    arrays: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    patient_index: np.ndarray,
    n_patients: int,
    n_bootstrap: int,
    bootstrap_seed: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    computers = {
        key: MetricComputer(y_true, y_prob, y_pred)
        for key, (y_prob, y_pred) in arrays.items()
    }
    per_seed, aggregate = observed_tables(specs, seeds, y_true, computers)
    display = {spec.key: spec.display_name for spec in specs}
    comparisons = (
        ("notesm_multispan", "longformer"),
        ("notesm_multispan", "clinicalmamba"),
    )

    ones = np.ones((1, len(y_true)), dtype=np.float64)
    observed = {key: computer.compute(ones) for key, computer in computers.items()}
    raw = {
        comparison: {metric: np.empty(n_bootstrap, dtype=np.float64) for metric in METRICS}
        for comparison in comparisons
    }
    rng = np.random.default_rng(bootstrap_seed)

    for start in range(0, n_bootstrap, batch_size):
        size = min(batch_size, n_bootstrap - start)
        weights = sample_patient_cluster_weights(rng, patient_index, n_patients, size)
        batch_metrics = {key: computer.compute(weights) for key, computer in computers.items()}
        for left, right in comparisons:
            for metric in METRICS:
                seed_matched_delta = np.stack(
                    [
                        batch_metrics[(left, seed)][metric] - batch_metrics[(right, seed)][metric]
                        for seed in seeds
                    ],
                    axis=0,
                )
                raw[(left, right)][metric][start : start + size] = seed_matched_delta.mean(axis=0)
        completed = start + size
        if completed == n_bootstrap or completed % max(100, batch_size) == 0:
            print(f"patient-cluster bootstrap {completed}/{n_bootstrap}", flush=True)

    raw_rows = []
    summary_rows = []
    for left, right in comparisons:
        left_agg = aggregate[aggregate["model_key"].eq(left)].iloc[0]
        right_agg = aggregate[aggregate["model_key"].eq(right)].iloc[0]
        for replicate in range(n_bootstrap):
            raw_rows.append(
                {
                    "replicate": replicate,
                    "comparison": f"{left}_vs_{right}",
                    "left_model": left,
                    "right_model": right,
                    **{f"delta_{metric}": raw[(left, right)][metric][replicate] for metric in METRICS},
                }
            )
        for metric in METRICS:
            observed_delta = float(
                np.mean(
                    [
                        observed[(left, seed)][metric][0] - observed[(right, seed)][metric][0]
                        for seed in seeds
                    ]
                )
            )
            values = raw[(left, right)][metric]
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite bootstrap deltas for {left} vs {right}, {metric}")
            ci_low, ci_high = np.quantile(values, [0.025, 0.975])
            favorable = values > 0 if HIGHER_IS_BETTER[metric] else values < 0
            summary_rows.append(
                {
                    "resampling_unit": "subject_id_hash_cluster",
                    "comparison": f"{left}_vs_{right}",
                    "left_model": left,
                    "left_display_name": display[left],
                    "right_model": right,
                    "right_display_name": display[right],
                    "metric": metric,
                    "higher_is_better": HIGHER_IS_BETTER[metric],
                    "left_observed_mean": float(left_agg[f"{metric}_mean"]),
                    "right_observed_mean": float(right_agg[f"{metric}_mean"]),
                    "observed_delta_left_minus_right": observed_delta,
                    "bootstrap_delta_mean": float(values.mean()),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                    "bootstrap_probability_left_favorable": float(favorable.mean()),
                    "n_bootstrap": n_bootstrap,
                    "n_bootstrap_valid": int(np.isfinite(values).sum()),
                    "n_seeds": len(seeds),
                    "n_examples": len(y_true),
                    "n_patients": n_patients,
                    "seed_matched": True,
                    "common_patient_resample_across_models_and_seeds": True,
                    "cluster_multiplicity_preserved_for_all_patient_episodes": True,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(raw_rows)


def format_mean_std(row: pd.Series, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}"


def write_markdown(
    output_dir: Path,
    aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
    n_examples: int,
    n_patients: int,
    seeds: list[int],
    n_bootstrap: int,
) -> None:
    lines = [
        "# Final Multi-Span NoteSSM Patient-Cluster Bootstrap",
        "",
        f"- Test episodes: {n_examples:,}",
        f"- Unique test patients: {n_patients:,}",
        f"- Seeds: {', '.join(map(str, seeds))}",
        f"- Patient-cluster bootstrap replicates: {n_bootstrap:,}",
        "- Each replicate samples patients with replacement and includes every episode for each sampled patient with its sampled multiplicity.",
        "- The same patient sample is used across all models and matched seeds within each replicate.",
        "- Reported deltas are multi-span NoteSSM minus comparator.",
        "- AUPRC/AUROC/Brier/ECE use raw probabilities; thresholded metrics use validation-selected thresholds.",
        "",
        "## Test Performance",
        "",
        "| Model | AUPRC | AUROC | F1 | Precision | Recall | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            f"| {row['display_name']} | {format_mean_std(row, 'AUPRC')} | "
            f"{format_mean_std(row, 'AUROC')} | {format_mean_std(row, 'F1')} | "
            f"{format_mean_std(row, 'Precision')} | {format_mean_std(row, 'Recall')} | "
            f"{format_mean_std(row, 'Brier')} | {format_mean_std(row, 'ECE')} |"
        )
    lines.extend(
        [
            "",
            "## Primary Patient-Cluster Inference",
            "",
            "| Comparison | Delta AUPRC | 95% CI | CI excludes 0 | P(NoteSSM favorable) |",
            "|---|---:|---:|---|---:|",
        ]
    )
    primary = bootstrap[bootstrap["metric"].eq("AUPRC")]
    for _, row in primary.iterrows():
        lines.append(
            f"| Multi-span NoteSSM vs {row['right_display_name']} | "
            f"{row['observed_delta_left_minus_right']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{row['ci_excludes_zero']} | {row['bootstrap_probability_left_favorable']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "A comparator is considered bootstrap-supported inferior on the primary endpoint only when the 95% patient-cluster CI for Delta AUPRC is entirely above zero. Point-estimate superiority alone is reported as numerical superiority.",
            "",
        ]
    )
    (output_dir / "NOTESSM_MULTISPAN_PATIENT_BOOTSTRAP_FROZEN.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be positive")
    if args.bootstrap_batch_size < 1:
        raise ValueError("--bootstrap-batch-size must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")

    root = find_project_root(args.project_root)
    specs = (
        ModelSpec(
            "notesm_multispan",
            "Multi-span NoteSSM",
            resolve_under_root(root, args.notesm_dir),
            args.notesm_pattern,
        ),
        ModelSpec(
            "longformer",
            "Longformer",
            resolve_under_root(root, args.longformer_dir),
            args.baseline_pattern,
        ),
        ModelSpec(
            "clinicalmamba",
            "ClinicalMamba",
            resolve_under_root(root, args.clinicalmamba_dir),
            args.baseline_pattern,
        ),
    )
    manifest_path = resolve_under_root(root, args.manifest)
    output_dir = resolve_under_root(root, args.output_dir)

    example_ids, y_true, arrays, inventory = load_and_align_predictions(
        specs, args.seeds, args.expected_test_n
    )
    unique_subjects, patient_index, manifest_info = load_patient_index(
        manifest_path,
        example_ids,
        args.expected_test_n,
        args.expected_test_patients,
    )
    computers = {
        key: MetricComputer(y_true, y_prob, y_pred)
        for key, (y_prob, y_pred) in arrays.items()
    }
    per_seed, aggregate = observed_tables(specs, args.seeds, y_true, computers)

    bootstrap, raw_bootstrap = run_patient_cluster_bootstrap(
        specs,
        args.seeds,
        y_true,
        arrays,
        patient_index,
        len(unique_subjects),
        args.n_bootstrap,
        args.bootstrap_seed,
        args.bootstrap_batch_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_dir / "prediction_inventory_and_sha256.csv", index=False)
    per_seed.to_csv(output_dir / "test_metrics_per_seed.csv", index=False)
    aggregate.to_csv(output_dir / "test_metrics_3seed_mean_std.csv", index=False)
    bootstrap.to_csv(output_dir / "patient_cluster_paired_bootstrap_ci.csv", index=False)
    raw_bootstrap.to_csv(output_dir / "patient_cluster_bootstrap_replicate_deltas.csv", index=False)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "command": sys.argv,
        "project_root": str(root),
        "seeds": args.seeds,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "resampling_unit": "subject_id_hash_cluster",
        "seed_matched": True,
        "common_patient_resample_across_models_and_seeds": True,
        "cluster_rule": (
            "Sample unique test patients with replacement; include all episodes belonging to each sampled "
            "patient with the patient's sampled multiplicity."
        ),
        "delta_convention": "multi-span NoteSSM minus comparator",
        "manifest": manifest_info,
        "prediction_files": inventory.to_dict(orient="records"),
    }
    (output_dir / "bootstrap_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    write_markdown(
        output_dir,
        aggregate,
        bootstrap,
        len(y_true),
        len(unique_subjects),
        args.seeds,
        args.n_bootstrap,
    )

    print(f"Wrote frozen patient-cluster bootstrap artifacts to {output_dir}")
    print(bootstrap[bootstrap["metric"].eq("AUPRC")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
