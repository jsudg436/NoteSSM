from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.data.lampr_dataset import read_lampr_split  # noqa: E402


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


def resolve_path(path_arg: str | Path) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def parquet_schema_columns(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema.names)


def check_raw_text_free(path: Path, columns: set[str]) -> None:
    forbidden = FORBIDDEN_COLUMNS & columns
    if forbidden:
        raise ValueError(f"{path} contains forbidden columns: {sorted(forbidden)}")


def benchmark_eval(root: Path) -> pd.DataFrame:
    frames = []
    for split in ["val", "test"]:
        frame = read_lampr_split(root, split)[["example_id", "split", "label"]].copy()
        frame["example_id"] = frame["example_id"].astype(str)
        frame["split"] = frame["split"].astype(str)
        frame["label"] = frame["label"].astype(int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def expected_eval_frame(benchmark_root: Path, example_id_file: Path | None) -> pd.DataFrame:
    benchmark = benchmark_eval(benchmark_root)
    if example_id_file is None:
        return benchmark
    subset = pd.read_csv(example_id_file)
    required = {"example_id", "split", "label"}
    missing = required - set(subset.columns)
    if missing:
        raise ValueError(f"{example_id_file} missing columns: {sorted(missing)}")
    check_raw_text_free(example_id_file, set(subset.columns))
    subset = subset[subset["split"].isin(["val", "test"])][["example_id", "split", "label"]].copy()
    subset["example_id"] = subset["example_id"].astype(str)
    subset["split"] = subset["split"].astype(str)
    subset["label"] = subset["label"].astype(int)
    merged = subset.merge(
        benchmark,
        on="example_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_benchmark"),
    )
    if merged["label_benchmark"].isna().any():
        raise ValueError("Subset contains examples absent from frozen benchmark")
    if (merged["split"] != merged["split_benchmark"]).any() or (merged["label"] != merged["label_benchmark"]).any():
        raise ValueError("Subset split/label disagrees with frozen benchmark")
    return subset


def metadata_paths(root: Path, prefix: str, cache_name: str | None = None) -> list[Path]:
    if prefix == "note":
        name = cache_name or ""
        single = root / f"{name}_metadata.parquet"
        paths = sorted(root.rglob(f"{name}_metadata_shard_*.parquet"))
    else:
        single = root / f"{prefix}_embedding_metadata.parquet"
        paths = sorted(root.glob(f"{prefix}_embedding_metadata_shard_*.parquet"))
    if single.exists():
        return [single]
    if paths:
        return paths
    raise FileNotFoundError(f"Missing {prefix} metadata under {root}")


def validate_metadata(root: Path, prefix: str, cache_name: str | None = None) -> dict[str, object]:
    paths = metadata_paths(root, prefix, cache_name=cache_name)
    rows = 0
    for path in paths:
        columns = parquet_schema_columns(path)
        check_raw_text_free(path, columns)
        import pyarrow.parquet as pq

        rows += int(pq.ParquetFile(path).metadata.num_rows)
    return {
        "artifact": f"{prefix}_metadata",
        "path": str(root),
        "rows": rows,
        "n_files": len(paths),
        "raw_text_free": True,
        "validation_passed": True,
    }


def validate_predictions(
    expected: pd.DataFrame,
    downstream_dir: Path,
    model: str,
    seed: int,
    expected_val_rows: int | None,
    expected_test_rows: int | None,
) -> dict[str, object]:
    path = downstream_dir / f"{model}_tuned_predictions_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing tuned predictions: {path}")
    predictions = pd.read_csv(path)
    required = {"example_id", "split", "y_true", "y_prob", "y_pred_tuned", "threshold"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    check_raw_text_free(path, set(predictions.columns))
    predictions = predictions.copy()
    predictions["example_id"] = predictions["example_id"].astype(str)
    predictions["split"] = predictions["split"].astype(str)
    if predictions.duplicated(["example_id", "split"]).any():
        raise ValueError(f"{path} contains duplicated example_id/split rows")
    if not predictions["y_prob"].astype(float).between(0.0, 1.0).all():
        raise ValueError(f"{path} contains y_prob outside [0, 1]")
    if not set(predictions["y_pred_tuned"].astype(int).tolist()).issubset({0, 1}):
        raise ValueError(f"{path} contains non-binary y_pred_tuned")
    for split, group in predictions.groupby("split"):
        if group["threshold"].nunique(dropna=False) != 1:
            raise ValueError(f"{path} threshold is not constant for split={split}")
    counts = predictions["split"].value_counts().to_dict()
    for split, expected_count in {"val": expected_val_rows, "test": expected_test_rows}.items():
        if expected_count is not None and int(counts.get(split, 0)) != expected_count:
            raise ValueError(f"{path} split={split} rows={counts.get(split, 0)}, expected {expected_count}")
    expected_ids = set(expected["example_id"].astype(str))
    observed_ids = set(predictions["example_id"].astype(str))
    if observed_ids != expected_ids:
        missing_ids = sorted(expected_ids - observed_ids)[:5]
        extra_ids = sorted(observed_ids - expected_ids)[:5]
        raise ValueError(f"Prediction example_id mismatch: missing={missing_ids} extra={extra_ids}")
    aligned = predictions.merge(expected, on="example_id", how="left", validate="one_to_one", suffixes=("", "_expected"))
    if (aligned["split"] != aligned["split_expected"]).any():
        raise ValueError("Prediction split disagrees with expected manifest")
    if (aligned["y_true"].astype(int) != aligned["label"].astype(int)).any():
        raise ValueError("Prediction y_true disagrees with expected labels")
    return {
        "artifact": "predictions",
        "seed": seed,
        "rows": int(len(predictions)),
        "val_rows": int(counts.get("val", 0)),
        "test_rows": int(counts.get("test", 0)),
        "raw_text_free": True,
        "validation_passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate NoteSSM-v2 hybrid cached-output artifacts.")
    parser.add_argument("--benchmark-root", default="results/lampr_full_v0_2")
    parser.add_argument("--downstream-dir", default=None)
    parser.add_argument("--hybrid-downstream-dir", default=None)
    parser.add_argument("--section-embedding-dir", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--cache-name", required=True)
    parser.add_argument("--model", default="notesm_v2_hybrid")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--example-id-file", default=None)
    parser.add_argument("--expected-val-rows", type=int, default=None)
    parser.add_argument("--expected-test-rows", type=int, default=None)
    parser.add_argument("--expected-settings", nargs="*", default=[])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    downstream_arg = args.downstream_dir or args.hybrid_downstream_dir
    if downstream_arg is None:
        raise ValueError("--downstream-dir or --hybrid-downstream-dir is required")
    downstream_dir = resolve_path(downstream_arg)
    expected = expected_eval_frame(resolve_path(args.benchmark_root), resolve_path(args.example_id_file) if args.example_id_file else None)
    rows: list[dict[str, object]] = []
    rows.append(validate_metadata(resolve_path(args.section_embedding_dir), "section"))
    chunk_dir = resolve_path(args.section_embedding_dir)
    if (chunk_dir / "chunk_embedding_metadata.parquet").exists() or list(chunk_dir.glob("chunk_embedding_metadata_shard_*.parquet")):
        rows.append(validate_metadata(chunk_dir, "chunk"))
    rows.append(validate_metadata(resolve_path(args.embedding_cache_dir), "note", cache_name=args.cache_name))
    seeds = args.seeds if args.seeds is not None else [args.seed if args.seed is not None else 17]
    for seed in seeds:
        rows.append(
            validate_predictions(
                expected=expected,
                downstream_dir=downstream_dir,
                model=args.model,
                seed=seed,
                expected_val_rows=args.expected_val_rows,
                expected_test_rows=args.expected_test_rows,
            )
        )
    for setting in args.expected_settings:
        setting_dir = downstream_dir / setting
        if not setting_dir.exists():
            raise FileNotFoundError(f"Missing expected hybrid setting directory: {setting_dir}")
    summary = pd.DataFrame(rows)
    output = resolve_path(args.output) if args.output else downstream_dir / "notesm_v2_hybrid_validation_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print("NoteSSM-v2 hybrid validation passed.")
    print(summary.to_string(index=False))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
