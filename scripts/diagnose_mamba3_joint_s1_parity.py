from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

EXTRACTOR_PATH = ROOT / "scripts" / "extract_mamba3_joint_s1_span2.py"
SPEC = importlib.util.spec_from_file_location("extract_mamba3_joint_s1_span2_for_parity", EXTRACTOR_PATH)
extractor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


FORBIDDEN_REPORT_COLUMNS = {
    "text",
    "sequence_text",
    "merged_section_text",
    "section_text",
    "chunk_text",
    "patient_id",
    "subject_id",
    "hadm_id",
    "note_id",
    "input_ids",
    "token_ids",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def key_columns(input_kind: str) -> list[str]:
    if input_kind == "section":
        return ["example_id", "section_order"]
    if input_kind == "chunk":
        return ["example_id", "chunk_order"]
    raise ValueError(f"Unsupported input_kind: {input_kind}")


def load_sharded(root: Path, prefix: str) -> tuple[np.ndarray, pd.DataFrame]:
    embedding_paths = sorted(root.glob(f"{prefix}_embeddings_shard_*.npy"))
    metadata_paths = sorted(root.glob(f"{prefix}_embedding_metadata_shard_*.parquet"))
    if not embedding_paths or len(embedding_paths) != len(metadata_paths):
        raise FileNotFoundError(f"Missing or mismatched {prefix} shards under {root}")
    arrays: list[np.ndarray] = []
    frames: list[pd.DataFrame] = []
    offset = 0
    for embedding_path, metadata_path in zip(embedding_paths, metadata_paths, strict=True):
        array = np.load(embedding_path)
        metadata = pd.read_parquet(metadata_path).copy()
        forbidden = FORBIDDEN_REPORT_COLUMNS & set(metadata.columns)
        if forbidden:
            raise ValueError(f"{metadata_path} contains restricted metadata columns: {sorted(forbidden)}")
        metadata["row_index"] = np.arange(offset, offset + len(metadata), dtype=int)
        arrays.append(array.astype(np.float32, copy=False))
        frames.append(metadata)
        offset += len(metadata)
    return np.concatenate(arrays, axis=0), pd.concat(frames, ignore_index=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_old_checkpoint_contract(old_root: Path, input_kind: str) -> dict[str, Any]:
    candidates = sorted(old_root.glob(f"*{input_kind}*checkpoint_load_summary*.csv"))
    if not candidates:
        candidates = sorted(old_root.glob("*checkpoint_load_summary*.csv"))
    if not candidates:
        return {}
    frame = pd.read_csv(candidates[0])
    return frame.iloc[0].to_dict() if not frame.empty else {}


def compare_prefix(
    *,
    old_root: Path,
    joint_root: Path,
    input_kind: str,
    max_abs_tolerance: float,
    minimum_cosine_tolerance: float,
) -> dict[str, Any]:
    prefix = input_kind
    row: dict[str, Any] = {
        "input_kind": input_kind,
        "status": "ok",
        "max_abs_tolerance": float(max_abs_tolerance),
        "minimum_cosine_tolerance": float(minimum_cosine_tolerance),
        "raw_text_stored": False,
        "patient_identifier_stored": False,
    }
    try:
        old_embeddings, old_meta = load_sharded(old_root, prefix)
        joint_embeddings, joint_meta = load_sharded(joint_root, prefix)
        keys = key_columns(input_kind)
        for metadata_name, metadata in [("old", old_meta), ("joint", joint_meta)]:
            missing = [col for col in keys if col not in metadata.columns]
            if missing:
                raise ValueError(f"{metadata_name} metadata missing key columns: {missing}")
            metadata["example_id"] = metadata["example_id"].astype(str)
            duplicated = metadata.duplicated(keys, keep=False)
            if duplicated.any():
                raise ValueError(f"{metadata_name} metadata has duplicated keys: count={int(duplicated.sum())}")

        old_keys = old_meta[keys + ["row_index"]].rename(columns={"row_index": "_old_row"})
        joint_keys = joint_meta[keys + ["row_index"]].rename(columns={"row_index": "_joint_row"})
        aligned = joint_keys.merge(old_keys, on=keys, how="left", validate="one_to_one")
        missing_mask = aligned["_old_row"].isna()
        n_expected = int(len(joint_meta))
        n_missing = int(missing_mask.sum())
        n_aligned = int(n_expected - n_missing)
        shape_match = bool(old_embeddings.shape[1:] == joint_embeddings.shape[1:])

        max_abs = np.nan
        mean_abs = np.nan
        min_cosine = np.nan
        mean_cosine = np.nan
        if n_aligned and shape_match:
            old_rows = aligned.loc[~missing_mask, "_old_row"].to_numpy(dtype=int)
            joint_rows = aligned.loc[~missing_mask, "_joint_row"].to_numpy(dtype=int)
            old_vectors = old_embeddings[old_rows].astype(np.float32, copy=False)
            joint_vectors = joint_embeddings[joint_rows].astype(np.float32, copy=False)
            diff = np.abs(old_vectors - joint_vectors)
            max_abs = float(np.max(diff)) if diff.size else 0.0
            mean_abs = float(np.mean(diff)) if diff.size else 0.0
            numerator = np.sum(old_vectors * joint_vectors, axis=1)
            denominator = np.linalg.norm(old_vectors, axis=1) * np.linalg.norm(joint_vectors, axis=1)
            cosine = numerator / np.clip(denominator, 1e-12, None)
            min_cosine = float(np.min(cosine)) if cosine.size else 1.0
            mean_cosine = float(np.mean(cosine)) if cosine.size else 1.0

        old_contract = read_old_checkpoint_contract(old_root, input_kind)
        joint_provenance = read_json(joint_root / f"{input_kind}_span2_provenance.json")
        joint_checkpoint = joint_provenance.get("clean_checkpoint", {}) if isinstance(joint_provenance, dict) else {}
        old_global_step = old_contract.get("checkpoint_global_step", "")
        joint_global_step = joint_checkpoint.get("global_step", "")
        checkpoint_contract_match = (
            bool(str(old_global_step) == str(joint_global_step))
            if old_global_step != "" and joint_global_step != ""
            else "unknown_old_cache"
        )

        row.update(
            {
                "n_expected": n_expected,
                "n_aligned": n_aligned,
                "n_missing": n_missing,
                "shape_match": shape_match,
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "min_cosine": min_cosine,
                "mean_cosine": mean_cosine,
                "within_tolerance": bool(
                    n_missing == 0
                    and shape_match
                    and not np.isnan(max_abs)
                    and max_abs <= max_abs_tolerance
                    and not np.isnan(min_cosine)
                    and min_cosine >= minimum_cosine_tolerance
                ),
                "checkpoint_contract_match": checkpoint_contract_match,
                "tokenizer_contract_match": "unknown_old_cache",
                "max_length_contract_match": "unknown_old_cache",
                "dtype_contract_match": "unknown_old_cache",
                "marker_contract_match": "unknown_old_cache",
                "joint_checkpoint_global_step": joint_global_step,
                "joint_max_length": joint_provenance.get("max_length", ""),
                "joint_dtype": joint_provenance.get("dtype", ""),
                "joint_marker": joint_provenance.get("marker", ""),
                "old_cache_fingerprint": extractor.cache_file_manifest_fingerprint(old_root, prefix),
                "joint_cache_fingerprint": extractor.cache_file_manifest_fingerprint(joint_root, prefix),
            }
        )
        if n_missing:
            row["status"] = "error"
            row["error"] = f"Could not align {n_missing} / {n_expected} joint S1 rows to old S1 cache"
        elif not shape_match:
            row["status"] = "error"
            row["error"] = f"Embedding hidden shape mismatch: old={old_embeddings.shape[1:]} joint={joint_embeddings.shape[1:]}"
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "error": repr(exc),
                "n_expected": 0,
                "n_aligned": 0,
                "n_missing": 0,
                "shape_match": False,
                "within_tolerance": False,
                "old_cache_fingerprint": "",
                "joint_cache_fingerprint": "",
            }
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostic parity report for old Mamba3 S1 cache vs joint S1 cache.")
    parser.add_argument("--old-s1-cache-dir", default="results/section_embedding_cache/dapt1epoch_mamba3_siso187_section_chunk_full_len512")
    parser.add_argument("--joint-cache-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-3)
    parser.add_argument("--minimum-cosine-tolerance", type=float, default=0.99999)
    args = parser.parse_args()

    old_root = resolve_path(args.old_s1_cache_dir)
    joint_root = resolve_path(args.joint_cache_dir)
    out_root = resolve_path(args.output_dir) if args.output_dir else joint_root
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for input_kind in ["section", "chunk"]:
        row = compare_prefix(
            old_root=old_root,
            joint_root=joint_root,
            input_kind=input_kind,
            max_abs_tolerance=args.max_abs_tolerance,
            minimum_cosine_tolerance=args.minimum_cosine_tolerance,
        )
        rows.append(row)
        path = out_root / f"{input_kind}_old_s1_vs_joint_s1_parity.csv"
        pd.DataFrame([row]).to_csv(path, index=False)
        print(pd.DataFrame([row]).to_string(index=False), flush=True)

    pd.DataFrame(rows).to_csv(out_root / "old_s1_vs_joint_s1_parity_summary.csv", index=False)
    if any(row.get("status") == "error" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
