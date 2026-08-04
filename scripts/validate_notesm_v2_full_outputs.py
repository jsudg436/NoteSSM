from __future__ import annotations

import argparse
from pathlib import Path
import sys

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

SECTION_REQUIRED_KEYS = ["example_id", "split", "canonical_section", "section_order"]
SECTION_OPTIONAL_KEYS = ["token_length", "use_chunk_fallback"]
CHUNK_REQUIRED_KEYS = ["example_id", "split", "chunk_order"]
CHUNK_OPTIONAL_KEYS = ["token_length"]


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


def parquet_num_rows(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except ImportError:
        # Fallback reads only an existing safe key column, not text-bearing columns.
        columns = list(parquet_schema_columns(path))
        safe_columns = [col for col in columns if col not in FORBIDDEN_COLUMNS]
        if not safe_columns:
            raise RuntimeError(f"pyarrow is required to count rows for {path} without reading restricted text columns")
        return int(len(pd.read_parquet(path, columns=[safe_columns[0]])))


def parquet_schema_columns(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema.names)
    except ImportError:
        # nrows=0 is unavailable for parquet; this reads no text if the first
        # safe key column exists, but cannot discover every column. Server env
        # includes pyarrow, so this is only a narrow fallback.
        try:
            return set(pd.read_parquet(path, columns=[]).columns)
        except Exception:
            raise RuntimeError("pyarrow is required for parquet schema inspection") from None


def read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    schema = parquet_schema_columns(path)
    missing = set(columns) - schema
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    check_raw_text_free(path, set(columns))
    return pd.read_parquet(path, columns=columns)


def benchmark_eval(root: Path) -> pd.DataFrame:
    frames = []
    for split in ["val", "test"]:
        frame = read_lampr_split(root, split)[["example_id", "split", "label"]].copy()
        frame["example_id"] = frame["example_id"].astype(str)
        frame["split"] = frame["split"].astype(str)
        frame["label"] = frame["label"].astype(int)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    if out["example_id"].duplicated().any():
        raise ValueError("Frozen benchmark val/test contains duplicated example_id values")
    return out


def count_section_cache_rows(section_cache_dir: Path) -> tuple[int, int]:
    section_rows = 0
    chunk_rows = 0
    for split in ["train", "val", "test"]:
        section_rows += parquet_num_rows(section_cache_dir / f"section_cache_{split}.parquet")
        chunk_path = section_cache_dir / f"fallback_chunk_cache_{split}.parquet"
        if chunk_path.exists():
            chunk_rows += parquet_num_rows(chunk_path)
    return section_rows, chunk_rows


def metadata_paths(root: Path, prefix: str) -> list[Path]:
    single = root / f"{prefix}_embedding_metadata.parquet"
    if single.exists():
        return [single]
    paths = sorted(root.glob(f"{prefix}_embedding_metadata_shard_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Missing {prefix} embedding metadata under {root}")
    return paths


def read_embedding_metadata(root: Path, prefix: str) -> pd.DataFrame:
    frames = []
    for path in metadata_paths(root, prefix):
        schema = parquet_schema_columns(path)
        check_raw_text_free(path, schema)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def count_embedding_rows(root: Path, prefix: str, expected_hidden_size: int) -> tuple[int, int, int]:
    single_npy = root / f"{prefix}_embeddings.npy"
    single_meta = root / f"{prefix}_embedding_metadata.parquet"
    if single_npy.exists() and single_meta.exists():
        rows = parquet_num_rows(single_meta)
        check_raw_text_free(single_meta, parquet_schema_columns(single_meta))
        array = np.load(single_npy, mmap_mode="r")
        if int(array.shape[0]) != rows:
            raise ValueError(f"{prefix} row mismatch: npy={array.shape[0]} metadata={rows}")
        hidden = int(array.shape[1]) if array.ndim == 2 else -1
        if hidden != expected_hidden_size:
            raise ValueError(f"{prefix} hidden size={hidden}, expected {expected_hidden_size}")
        return rows, hidden, 1

    npy_paths = sorted(root.glob(f"{prefix}_embeddings_shard_*.npy"))
    meta_paths = sorted(root.glob(f"{prefix}_embedding_metadata_shard_*.parquet"))
    if not npy_paths and not meta_paths:
        raise FileNotFoundError(f"Missing {prefix} embedding files under {root}")
    if len(npy_paths) != len(meta_paths):
        raise FileNotFoundError(f"Mismatched {prefix} shard counts: npy={len(npy_paths)} metadata={len(meta_paths)}")
    rows = 0
    hidden = None
    global_indices: set[int] = set()
    for npy_path, meta_path in zip(npy_paths, meta_paths, strict=True):
        check_raw_text_free(meta_path, parquet_schema_columns(meta_path))
        meta_rows = parquet_num_rows(meta_path)
        metadata = pd.read_parquet(meta_path, columns=["global_row_index"]) if "global_row_index" in parquet_schema_columns(meta_path) else pd.DataFrame()
        array = np.load(npy_path, mmap_mode="r")
        if int(array.shape[0]) != meta_rows:
            raise ValueError(f"{prefix} shard row mismatch: {npy_path.name} vs {meta_path.name}")
        shard_hidden = int(array.shape[1]) if array.ndim == 2 else -1
        if shard_hidden != expected_hidden_size:
            raise ValueError(f"{prefix} hidden size={shard_hidden}, expected {expected_hidden_size}")
        hidden = shard_hidden if hidden is None else hidden
        if not metadata.empty:
            shard_indices = set(metadata["global_row_index"].astype(int).tolist())
            overlap = global_indices & shard_indices
            if overlap:
                raise ValueError(f"{prefix} duplicated global_row_index values: {sorted(overlap)[:5]}")
            global_indices.update(shard_indices)
        rows += meta_rows
    return int(rows), int(hidden or expected_hidden_size), len(npy_paths)


def safe_key_columns(path: Path, required: list[str], optional: list[str]) -> list[str]:
    schema = parquet_schema_columns(path)
    missing = set(required) - schema
    if missing:
        raise ValueError(f"{path} missing required key columns: {sorted(missing)}")
    return [*required, *[col for col in optional if col in schema]]


def read_cache_keys(section_cache_dir: Path, prefix: str, required: list[str], optional: list[str]) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    checked_columns: list[str] | None = None
    for split in ["train", "val", "test"]:
        name = f"section_cache_{split}.parquet" if prefix == "section" else f"fallback_chunk_cache_{split}.parquet"
        path = section_cache_dir / name
        if prefix == "chunk" and not path.exists():
            continue
        cols = safe_key_columns(path, required, optional)
        if checked_columns is None:
            checked_columns = cols
        else:
            checked_columns = [col for col in checked_columns if col in cols]
        frames.append(read_parquet_columns(path, cols))
    if not frames:
        return pd.DataFrame(columns=required), required
    common = checked_columns or required
    return pd.concat([frame[common] for frame in frames], ignore_index=True), common


def normalize_key_frame(frame: pd.DataFrame, columns: list[str], sort_columns: list[str]) -> pd.DataFrame:
    out = frame[columns].copy()
    for col in ["example_id", "split", "canonical_section"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    for col in ["section_order", "chunk_order", "token_length"]:
        if col in out.columns:
            out[col] = out[col].astype(int)
    if "use_chunk_fallback" in out.columns:
        out["use_chunk_fallback"] = out["use_chunk_fallback"].astype(bool)
    return out.sort_values(sort_columns).reset_index(drop=True)


def validate_key_alignment(
    section_cache_dir: Path,
    section_embedding_dir: Path,
    prefix: str,
    required: list[str],
    optional: list[str],
    sort_columns: list[str],
) -> tuple[bool, str]:
    cache_keys, cache_columns = read_cache_keys(section_cache_dir, prefix, required, optional)
    metadata = read_embedding_metadata(section_embedding_dir, prefix)
    missing = set(required) - set(metadata.columns)
    if missing:
        raise ValueError(f"{prefix} embedding metadata missing required key columns: {sorted(missing)}")
    columns = [col for col in cache_columns if col in metadata.columns]
    missing_from_metadata = set(required) - set(columns)
    if missing_from_metadata:
        raise ValueError(f"{prefix} embedding metadata missing required key columns: {sorted(missing_from_metadata)}")
    cache_norm = normalize_key_frame(cache_keys, columns, sort_columns)
    meta_norm = normalize_key_frame(metadata, columns, sort_columns)
    if len(cache_norm) != len(meta_norm):
        raise ValueError(f"{prefix} key row count mismatch: cache={len(cache_norm)} metadata={len(meta_norm)}")
    if not cache_norm.equals(meta_norm):
        mismatch = (cache_norm != meta_norm).any(axis=1)
        first = int(np.where(mismatch.to_numpy())[0][0]) if mismatch.any() else 0
        raise ValueError(
            f"{prefix} key alignment mismatch at sorted row {first}: "
            f"cache={cache_norm.iloc[first].to_dict()} metadata={meta_norm.iloc[first].to_dict()}"
        )
    return True, ",".join(columns)


def validate_metrics(metrics_path: Path) -> None:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing tuned metrics file: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    required = {"split", "AUPRC", "AUROC", "Precision", "Recall", "F1", "Brier", "ECE"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"{metrics_path} missing columns: {sorted(missing)}")
    check_raw_text_free(metrics_path, set(metrics.columns))
    if set(metrics["split"].astype(str)) != {"val", "test"}:
        raise ValueError(f"{metrics_path} must contain exactly val/test rows")


def validate_predictions(
    benchmark: pd.DataFrame,
    downstream_dir: Path,
    model: str,
    seed: int,
    expected_val_rows: int | None = None,
    expected_test_rows: int | None = None,
) -> dict[str, object]:
    prediction_path = downstream_dir / f"{model}_tuned_predictions_seed{seed}.csv"
    metrics_path = downstream_dir / f"{model}_threshold_tuned_metrics_seed{seed}.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing tuned prediction file: {prediction_path}")
    predictions = pd.read_csv(prediction_path)
    required = {"example_id", "split", "y_true", "y_prob", "y_pred_tuned", "threshold"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{prediction_path} missing columns: {sorted(missing)}")
    check_raw_text_free(prediction_path, set(predictions.columns))
    predictions = predictions.copy()
    predictions["example_id"] = predictions["example_id"].astype(str)
    predictions["split"] = predictions["split"].astype(str)
    if predictions["example_id"].duplicated().any():
        duplicated = predictions.loc[predictions["example_id"].duplicated(), "example_id"].head(5).tolist()
        raise ValueError(f"{prediction_path} has duplicated example_id values: {duplicated}")
    if not set(predictions["split"]).issubset({"val", "test"}):
        raise ValueError(f"{prediction_path} contains non-evaluation splits")
    counts = predictions["split"].value_counts().to_dict()
    expected_counts = benchmark["split"].value_counts().to_dict()
    if expected_val_rows is not None:
        expected_counts["val"] = expected_val_rows
    if expected_test_rows is not None:
        expected_counts["test"] = expected_test_rows
    for split in ["val", "test"]:
        observed = int(counts.get(split, 0))
        expected = int(expected_counts.get(split, 0))
        if observed != expected:
            raise ValueError(f"{prediction_path} split={split} rows={observed}, expected {expected}")
    if not predictions["y_prob"].astype(float).between(0.0, 1.0).all():
        raise ValueError(f"{prediction_path} contains y_prob outside [0, 1]")
    y_pred = pd.to_numeric(predictions["y_pred_tuned"], errors="coerce")
    if y_pred.isna().any() or not set(y_pred.astype(int).tolist()).issubset({0, 1}):
        raise ValueError(f"{prediction_path} contains non-binary y_pred_tuned values")
    for split, split_frame in predictions.groupby("split"):
        if split_frame["threshold"].nunique(dropna=False) != 1:
            raise ValueError(f"{prediction_path} threshold is not constant within split={split}")

    expected_ids = set(benchmark["example_id"].astype(str))
    observed_ids = set(predictions["example_id"].astype(str))
    if observed_ids != expected_ids:
        missing_ids = sorted(expected_ids - observed_ids)[:5]
        extra_ids = sorted(observed_ids - expected_ids)[:5]
        raise ValueError(f"Prediction example_id mismatch: missing={missing_ids} extra={extra_ids}")
    aligned = predictions.merge(
        benchmark[["example_id", "split", "label"]],
        on="example_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_benchmark"),
    )
    if (aligned["split"].astype(str) != aligned["split_benchmark"].astype(str)).any():
        raise ValueError("Prediction split disagrees with frozen benchmark")
    if (aligned["y_true"].astype(int) != aligned["label"].astype(int)).any():
        raise ValueError("Prediction y_true disagrees with frozen benchmark labels")
    validate_metrics(metrics_path)
    return {
        "artifact": "predictions",
        "seed": seed,
        "rows": int(len(predictions)),
        "expected_rows": int(len(benchmark)),
        "val_rows": int(counts.get("val", 0)),
        "test_rows": int(counts.get("test", 0)),
        "validation_passed": True,
        "raw_text_free": True,
    }


def validate_full_outputs(
    benchmark_root: Path,
    downstream_dir: Path,
    section_embedding_dir: Path,
    section_cache_dir: Path,
    model: str,
    seed: int,
    expected_hidden_size: int,
    expected_section_rows: int | None = None,
    expected_chunk_rows: int | None = None,
    expected_val_rows: int | None = None,
    expected_test_rows: int | None = None,
    embedding_only: bool = False,
) -> pd.DataFrame:
    section_cache_rows, chunk_cache_rows = count_section_cache_rows(section_cache_dir)
    if expected_section_rows is not None and section_cache_rows != expected_section_rows:
        raise ValueError(f"section_cache rows={section_cache_rows}, expected {expected_section_rows}")
    if expected_chunk_rows is not None and chunk_cache_rows != expected_chunk_rows:
        raise ValueError(f"fallback_chunk_cache rows={chunk_cache_rows}, expected {expected_chunk_rows}")
    section_rows, section_hidden, section_shards = count_embedding_rows(section_embedding_dir, "section", expected_hidden_size)
    chunk_rows, chunk_hidden, chunk_shards = count_embedding_rows(section_embedding_dir, "chunk", expected_hidden_size)
    if section_rows != section_cache_rows:
        raise ValueError(f"section_embeddings rows={section_rows}, section_cache rows={section_cache_rows}")
    if chunk_rows != chunk_cache_rows:
        raise ValueError(f"chunk_embeddings rows={chunk_rows}, fallback_chunk_cache rows={chunk_cache_rows}")
    section_key_ok, section_key_columns = validate_key_alignment(
        section_cache_dir,
        section_embedding_dir,
        "section",
        SECTION_REQUIRED_KEYS,
        SECTION_OPTIONAL_KEYS,
        ["example_id", "split", "section_order", "canonical_section"],
    )
    chunk_key_ok, chunk_key_columns = validate_key_alignment(
        section_cache_dir,
        section_embedding_dir,
        "chunk",
        CHUNK_REQUIRED_KEYS,
        CHUNK_OPTIONAL_KEYS,
        ["example_id", "split", "chunk_order"],
    )
    rows: list[dict[str, object]] = [
        {
            "artifact": "section_cache",
            "seed": seed,
            "rows": section_cache_rows,
            "expected_rows": expected_section_rows or section_cache_rows,
            "hidden_size": "",
            "shards": "",
            "section_key_alignment_passed": section_key_ok,
            "chunk_key_alignment_passed": "",
            "section_key_columns_checked": section_key_columns,
            "chunk_key_columns_checked": "",
            "validation_passed": True,
            "raw_text_free": False,
        },
        {
            "artifact": "fallback_chunk_cache",
            "seed": seed,
            "rows": chunk_cache_rows,
            "expected_rows": expected_chunk_rows or chunk_cache_rows,
            "hidden_size": "",
            "shards": "",
            "section_key_alignment_passed": "",
            "chunk_key_alignment_passed": chunk_key_ok,
            "section_key_columns_checked": "",
            "chunk_key_columns_checked": chunk_key_columns,
            "validation_passed": True,
            "raw_text_free": False,
        },
        {
            "artifact": "section_embeddings",
            "seed": seed,
            "rows": section_rows,
            "expected_rows": section_cache_rows,
            "hidden_size": section_hidden,
            "shards": section_shards,
            "section_key_alignment_passed": section_key_ok,
            "chunk_key_alignment_passed": "",
            "section_key_columns_checked": section_key_columns,
            "chunk_key_columns_checked": "",
            "validation_passed": True,
            "raw_text_free": True,
        },
        {
            "artifact": "chunk_embeddings",
            "seed": seed,
            "rows": chunk_rows,
            "expected_rows": chunk_cache_rows,
            "hidden_size": chunk_hidden,
            "shards": chunk_shards,
            "section_key_alignment_passed": "",
            "chunk_key_alignment_passed": chunk_key_ok,
            "section_key_columns_checked": "",
            "chunk_key_columns_checked": chunk_key_columns,
            "validation_passed": True,
            "raw_text_free": True,
        },
    ]
    if not embedding_only:
        benchmark = benchmark_eval(benchmark_root)
        rows.append(validate_predictions(
            benchmark,
            downstream_dir,
            model,
            seed,
            expected_val_rows=expected_val_rows,
            expected_test_rows=expected_test_rows,
        ))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate full NoteSSM-v2 section-aware artifacts.")
    parser.add_argument("--benchmark-root", default="results/lampr_full_v0_2")
    parser.add_argument("--downstream-dir", default="results/downstream_notesm_v2_full/dapt1epoch_mamba2_section_full_mlp_natural_bce")
    parser.add_argument("--section-embedding-dir", default="results/section_embedding_cache/dapt1epoch_mamba2_section_full")
    parser.add_argument("--section-cache-dir", default="results/notesm_v2_section_cache/lampr_v0_2_full")
    parser.add_argument("--model", default="notesm_v2")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--expected-hidden-size", type=int, default=768)
    parser.add_argument("--expected-section-rows", type=int, default=None)
    parser.add_argument("--expected-chunk-rows", type=int, default=None)
    parser.add_argument("--expected-val-rows", type=int, default=None)
    parser.add_argument("--expected-test-rows", type=int, default=None)
    parser.add_argument("--embedding-only", action="store_true", help="Validate only section/chunk cache and embedding artifacts; skip downstream predictions.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    downstream_dir = resolve_path(args.downstream_dir)
    summary = validate_full_outputs(
        benchmark_root=resolve_path(args.benchmark_root),
        downstream_dir=downstream_dir,
        section_embedding_dir=resolve_path(args.section_embedding_dir),
        section_cache_dir=resolve_path(args.section_cache_dir),
        model=args.model,
        seed=args.seed,
        expected_hidden_size=args.expected_hidden_size,
        expected_section_rows=args.expected_section_rows,
        expected_chunk_rows=args.expected_chunk_rows,
        expected_val_rows=args.expected_val_rows,
        expected_test_rows=args.expected_test_rows,
        embedding_only=bool(args.embedding_only),
    )
    if args.output:
        output = resolve_path(args.output)
    elif args.embedding_only:
        output = resolve_path(args.section_embedding_dir) / "notesm_v2_full_embedding_validation.csv"
    else:
        output = downstream_dir / f"notesm_v2_full_artifact_validation_seed{args.seed}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    if args.embedding_only:
        print("NoteSSM-v2 full embedding artifact validation passed.")
    else:
        print("NoteSSM-v2 full artifact validation passed.")
    print(summary.to_string(index=False))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
