from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SPEC = importlib.util.spec_from_file_location("summarize_note_baseline_run", ROOT / "scripts" / "summarize_note_baseline_run.py")
summary_script = importlib.util.module_from_spec(SUMMARY_SPEC)
assert SUMMARY_SPEC.loader is not None
SUMMARY_SPEC.loader.exec_module(summary_script)

FORBIDDEN_COLUMNS = summary_script.FORBIDDEN_COLUMNS


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"{name} missing required columns: {sorted(missing)}")


def load_benchmark(root: Path) -> pd.DataFrame:
    benchmark = summary_script.load_benchmark_index(root)
    if not benchmark["example_id"].is_unique:
        raise AssertionError("Frozen benchmark example_id values are not unique")
    return benchmark


def validate_embedding_cache(cache_dir: Path, cache_name: str, benchmark: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    rows_extracted, hidden_size, metadata, metadata_path = summary_script.load_embedding_cache_summary(cache_dir, cache_name)
    if rows_extracted != len(benchmark):
        raise AssertionError(f"Embedding rows {rows_extracted} != benchmark rows {len(benchmark)}")
    if hidden_size is None or hidden_size <= 0:
        raise AssertionError(f"Invalid hidden size: {hidden_size}")
    if FORBIDDEN_COLUMNS & set(metadata.columns):
        raise AssertionError(f"Embedding metadata contains forbidden columns: {sorted(FORBIDDEN_COLUMNS & set(metadata.columns))}")
    aligned = summary_script.metadata_with_split(metadata, None if "split" in metadata.columns else None)
    if "split" not in aligned.columns or "label" not in aligned.columns:
        aligned = summary_script.align_metadata_with_benchmark(metadata, benchmark)
    if len(aligned) != rows_extracted:
        raise AssertionError("Aligned metadata row count does not match embedding row count")
    if aligned["example_id"].duplicated().any():
        duplicated = aligned.loc[aligned["example_id"].duplicated(), "example_id"].head(5).tolist()
        raise AssertionError(f"Embedding metadata has duplicated example_id values: {duplicated}")
    expected_ids = set(benchmark["example_id"].astype(str))
    observed_ids = set(aligned["example_id"].astype(str))
    if observed_ids != expected_ids:
        missing = list(expected_ids - observed_ids)[:5]
        extra = list(observed_ids - expected_ids)[:5]
        raise AssertionError(f"Embedding example_id mismatch. missing={missing} extra={extra}")
    if metadata_path is not None and not summary_script.raw_text_free(metadata_path):
        raise AssertionError(f"Embedding metadata is not raw-text-free: {metadata_path}")
    split_counts = aligned["split"].value_counts().to_dict()
    expected_split_counts = benchmark["split"].value_counts().to_dict()
    if split_counts != expected_split_counts:
        raise AssertionError(f"Embedding split counts {split_counts} != benchmark split counts {expected_split_counts}")
    return aligned, {
        "rows_extracted": int(rows_extracted),
        "hidden_size": int(hidden_size),
        "train_rows": int(split_counts.get("train", 0)),
        "val_rows": int(split_counts.get("val", 0)),
        "test_rows": int(split_counts.get("test", 0)),
        "metadata_raw_text_free": True,
    }


def validate_prediction_file(
    prediction_path: Path,
    benchmark: pd.DataFrame,
    prediction_column: str,
    require_threshold: bool,
) -> dict[str, int | bool]:
    predictions = pd.read_csv(prediction_path)
    required = {"example_id", "split", "y_true", "y_prob", prediction_column}
    if require_threshold:
        required.add("threshold")
    require_columns(predictions, required, str(prediction_path))
    forbidden = FORBIDDEN_COLUMNS & set(predictions.columns)
    if forbidden:
        raise AssertionError(f"Prediction CSV contains forbidden columns: {sorted(forbidden)}")
    if not set(predictions["split"]).issubset({"val", "test"}):
        raise AssertionError(f"Prediction CSV contains non-evaluation splits: {sorted(set(predictions['split']))}")
    if predictions["example_id"].duplicated().any():
        duplicated = predictions.loc[predictions["example_id"].duplicated(), "example_id"].head(5).tolist()
        raise AssertionError(f"Prediction CSV has duplicated example_id values: {duplicated}")
    if not predictions["y_prob"].between(0.0, 1.0).all():
        raise AssertionError("Prediction probabilities must be in [0, 1]")
    predicted_labels = pd.to_numeric(predictions[prediction_column], errors="coerce")
    if predicted_labels.isna().any() or not set(predicted_labels.astype(int).tolist()).issubset({0, 1}):
        raise AssertionError("Predicted labels must be binary")
    if not (predicted_labels == predicted_labels.astype(int)).all():
        raise AssertionError("Predicted labels must be binary")
    threshold_constant_by_split = False
    if require_threshold:
        for split, split_frame in predictions.groupby("split"):
            if split_frame["threshold"].nunique(dropna=False) != 1:
                raise AssertionError(f"Tuned threshold is not constant within split={split}")
        threshold_constant_by_split = True
    eval_benchmark = benchmark[benchmark["split"].isin(["val", "test"])].copy()
    expected_eval_ids = set(eval_benchmark["example_id"].astype(str))
    observed_eval_ids = set(predictions["example_id"].astype(str))
    if observed_eval_ids != expected_eval_ids:
        missing = list(expected_eval_ids - observed_eval_ids)[:5]
        extra = list(observed_eval_ids - expected_eval_ids)[:5]
        raise AssertionError(f"Prediction example_id mismatch. missing={missing} extra={extra}")
    aligned = predictions.copy()
    aligned["example_id"] = aligned["example_id"].astype(str)
    expected = eval_benchmark[["example_id", "split", "label"]].copy()
    expected["example_id"] = expected["example_id"].astype(str)
    aligned = aligned.merge(expected, on="example_id", how="left", validate="one_to_one", suffixes=("", "_benchmark"))
    if (aligned["split"].astype(str) != aligned["split_benchmark"].astype(str)).any():
        raise AssertionError("Prediction split values disagree with frozen benchmark")
    if (aligned["y_true"].astype(int) != aligned["label"].astype(int)).any():
        raise AssertionError("Prediction y_true values disagree with frozen benchmark labels")
    counts = predictions["split"].value_counts().to_dict()
    expected_counts = eval_benchmark["split"].value_counts().to_dict()
    if counts != expected_counts:
        raise AssertionError(f"Prediction split counts {counts} != benchmark eval split counts {expected_counts}")
    return {
        "raw_text_free": True,
        "val_prediction_rows": int(counts.get("val", 0)),
        "test_prediction_rows": int(counts.get("test", 0)),
        "threshold_constant_by_split": threshold_constant_by_split,
    }


def validate_metrics_file(metrics_path: Path, tuned: bool) -> None:
    metrics = pd.read_csv(metrics_path)
    required = {"split", "AUPRC", "AUROC"}
    if tuned:
        required |= {"F1", "Precision", "Recall", "Brier", "ECE"}
    require_columns(metrics, required, str(metrics_path))
    if set(metrics["split"]) != {"val", "test"}:
        raise AssertionError(f"Metrics CSV must contain val/test rows, got {sorted(set(metrics['split']))}")


def validate_checkpoint_summary(
    cache_dir: Path,
    cache_name: str,
    expected_global_step: int | None,
    expected_format: str | None,
) -> dict[str, object]:
    paths = sorted(cache_dir.rglob(f"{cache_name}_checkpoint_load_summary.csv"))
    if not paths:
        return {
            "checkpoint_metadata_available": False,
            "checkpoint_summary_files": 0,
            "checkpoint_global_step": None,
            "checkpoint_format": None,
            "checkpoint_global_step_match": None,
            "checkpoint_format_match": None,
        }
    frames = [pd.read_csv(path) for path in paths]
    summary = pd.concat(frames, ignore_index=True)
    steps = sorted({int(step) for step in summary["checkpoint_global_step"].dropna().tolist()}) if "checkpoint_global_step" in summary.columns else []
    formats = sorted({str(fmt) for fmt in summary["checkpoint_format"].dropna().tolist()}) if "checkpoint_format" in summary.columns else []
    step_match = None
    format_match = None
    if expected_global_step is not None:
        step_match = steps == [int(expected_global_step)]
        if not step_match:
            raise AssertionError(f"Checkpoint global step mismatch: observed={steps}, expected={expected_global_step}")
    if expected_format is not None:
        format_match = formats == [str(expected_format)]
        if not format_match:
            raise AssertionError(f"Checkpoint format mismatch: observed={formats}, expected={expected_format}")
    return {
        "checkpoint_metadata_available": True,
        "checkpoint_summary_files": len(paths),
        "checkpoint_global_step": ";".join(str(step) for step in steps),
        "checkpoint_format": ";".join(formats),
        "checkpoint_global_step_match": step_match,
        "checkpoint_format_match": format_match,
    }


def validate_predictions(downstream_dir: Path, model: str, seed: int, benchmark: pd.DataFrame) -> dict[str, int | bool]:
    cached_prediction_path = downstream_dir / f"{model}_cached_predictions_seed{seed}.csv"
    tuned_prediction_path = downstream_dir / f"{model}_tuned_predictions_seed{seed}.csv"
    cached_metrics_path = downstream_dir / f"{model}_cached_embedding_metrics_seed{seed}.csv"
    tuned_metrics_path = downstream_dir / f"{model}_threshold_tuned_metrics_seed{seed}.csv"

    if not cached_prediction_path.exists() and not tuned_prediction_path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {cached_prediction_path} or {tuned_prediction_path}")

    cached_summary = None
    tuned_summary = None
    if cached_prediction_path.exists():
        cached_summary = validate_prediction_file(cached_prediction_path, benchmark, "y_pred", require_threshold=False)
    if tuned_prediction_path.exists():
        tuned_summary = validate_prediction_file(tuned_prediction_path, benchmark, "y_pred_tuned", require_threshold=True)

    if cached_metrics_path.exists():
        validate_metrics_file(cached_metrics_path, tuned=False)
    if tuned_metrics_path.exists():
        validate_metrics_file(tuned_metrics_path, tuned=True)
    if not cached_metrics_path.exists() and not tuned_metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {cached_metrics_path} or {tuned_metrics_path}")

    primary_summary = tuned_summary or cached_summary or {}
    metrics_path = downstream_dir / f"{model}_cached_embedding_metrics_seed{seed}.csv"
    return {
        "prediction_csv_found": True,
        "prediction_csv_raw_text_free": bool((cached_summary or tuned_summary)),
        "cached_prediction_found": cached_prediction_path.exists(),
        "cached_prediction_raw_text_free": bool(cached_summary["raw_text_free"]) if cached_summary else False,
        "tuned_prediction_found": tuned_prediction_path.exists(),
        "tuned_prediction_raw_text_free": bool(tuned_summary["raw_text_free"]) if tuned_summary else False,
        "val_prediction_rows": int(primary_summary.get("val_prediction_rows", 0)),
        "test_prediction_rows": int(primary_summary.get("test_prediction_rows", 0)),
        "threshold_constant_by_split": bool(tuned_summary["threshold_constant_by_split"]) if tuned_summary else False,
        "metrics_found": cached_metrics_path.exists() or tuned_metrics_path.exists(),
        "cached_metrics_found": cached_metrics_path.exists(),
        "tuned_metrics_found": tuned_metrics_path.exists(),
        "metrics_path": str(tuned_metrics_path if tuned_metrics_path.exists() else metrics_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate text-free no-DAPT note baseline embedding and prediction artifacts.")
    parser.add_argument("--benchmark-root", default="results/lampr_full_v0_2")
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--cache-name", required=True)
    parser.add_argument("--downstream-dir", action="append", default=[], help="Downstream run directory. May be passed multiple times.")
    parser.add_argument("--downstream-root", default=None, help="Optional directory; all immediate subdirectories are validated as downstream runs.")
    parser.add_argument("--model", default="notesm_v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--embedding-only", action="store_true", help="Validate only the merged embedding cache and skip downstream prediction checks.")
    parser.add_argument("--expected-checkpoint-global-step", type=int, default=None)
    parser.add_argument("--expected-checkpoint-format", default=None)
    parser.add_argument("--output", default=None, help="Optional validation summary CSV path.")
    args = parser.parse_args()

    benchmark_root = resolve_path(args.benchmark_root)
    cache_dir = resolve_path(args.embedding_cache_dir)
    benchmark = load_benchmark(benchmark_root)
    _, embedding_summary = validate_embedding_cache(cache_dir, args.cache_name, benchmark)
    checkpoint_summary = validate_checkpoint_summary(
        cache_dir,
        args.cache_name,
        args.expected_checkpoint_global_step,
        args.expected_checkpoint_format,
    )

    downstream_dirs = [resolve_path(path) for path in args.downstream_dir]
    if args.downstream_root:
        root = resolve_path(args.downstream_root)
        downstream_dirs.extend(sorted(path for path in root.iterdir() if path.is_dir()))
    if args.embedding_only:
        downstream_dirs = []
    downstream_dirs = [path for path in downstream_dirs if not path.name.endswith("_extraction_only")]

    rows = []
    if downstream_dirs:
        for downstream_dir in downstream_dirs:
            prediction_summary = validate_predictions(downstream_dir, args.model, args.seed, benchmark)
            row = {
                "downstream_dir": str(downstream_dir),
                **embedding_summary,
                **checkpoint_summary,
                **prediction_summary,
            }
            rows.append(row)
    else:
        rows.append({"downstream_dir": None, **embedding_summary, **checkpoint_summary})

    summary = pd.DataFrame(rows)
    if args.output:
        output = resolve_path(args.output)
    else:
        output = cache_dir / "artifact_validation_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print("All note baseline artifact validation checks passed.")
    print(f"benchmark_root={benchmark_root}")
    print(f"embedding_cache_dir={cache_dir}")
    print(f"cache_name={args.cache_name}")
    print(summary.to_string(index=False))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
