from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

EXTRACTOR_PATH = ROOT / "scripts" / "extract_mamba3_joint_s1_span2.py"
SPEC = importlib.util.spec_from_file_location("extract_mamba3_joint_s1_span2_for_validation", EXTRACTOR_PATH)
extractor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def shard_indices(root: Path, prefix: str) -> set[int]:
    out: set[int] = set()
    pattern = re.compile(rf"^{re.escape(prefix)}_embeddings_shard_(\d+)\.npy$")
    for path in root.glob(f"{prefix}_embeddings_shard_*.npy"):
        match = pattern.match(path.name)
        if match:
            out.add(int(match.group(1)))
    return out


def read_summary_rows(root: Path, prefix: str) -> int | None:
    path = root / f"{prefix}_embedding_shards_summary.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "n_rows" not in frame.columns:
        raise ValueError(f"Invalid summary file: {path}")
    return int(frame.iloc[0]["n_rows"])


def validate_prefix(root: Path, input_kind: str, allow_test_accessed: bool = False) -> dict[str, Any]:
    s1_prefix = input_kind
    s2_prefix = f"{input_kind}_span2"
    s1_representation = "sec_sum_1" if input_kind == "section" else "chunk_sum_1"
    s1_indices = shard_indices(root, s1_prefix)
    s2_indices = shard_indices(root, s2_prefix)
    if not s1_indices:
        raise FileNotFoundError(f"Missing {s1_prefix} S1 shards under {root}")
    if not s2_indices:
        raise FileNotFoundError(f"Missing {s2_prefix} S2 shards under {root}")
    if s1_indices != s2_indices:
        raise ValueError(f"S1/S2 shard index mismatch for {input_kind}: {sorted(s1_indices)[:5]} != {sorted(s2_indices)[:5]}")

    s1_rows = 0
    s2_rows = 0
    run_ids: set[str] = set()
    forward_contracts: set[str] = set()
    hidden_sizes: set[int] = set()
    provenance_path = root / f"{input_kind}_span2_provenance.json"
    if not provenance_path.exists():
        raise FileNotFoundError(f"Missing provenance file: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for shard_idx in sorted(s1_indices):
        s1_paths = extractor.shard_data_paths(root, s1_prefix, shard_idx)
        s2_paths = extractor.shard_data_paths(root, s2_prefix, shard_idx)
        complete_path = extractor.joint_completion_path(root, input_kind, shard_idx)
        missing = [path for path in [s1_paths["embeddings"], s1_paths["metadata"], s2_paths["embeddings"], s2_paths["metadata"], complete_path] if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing joint files for {input_kind} shard {shard_idx}: {[str(path) for path in missing]}")
        completion = json.loads(complete_path.read_text(encoding="utf-8"))
        if int(completion.get("schema_version", -1)) != extractor.COMPLETION_SCHEMA_VERSION:
            raise ValueError(f"Invalid schema_version in {complete_path}: {completion.get('schema_version')}")
        if completion.get("input_kind") != input_kind:
            raise ValueError(f"Invalid input_kind in {complete_path}: {completion.get('input_kind')}")
        if completion.get("raw_text_stored") is not False:
            raise ValueError(f"Invalid privacy/split flags in {complete_path}")
        if completion.get("test_accessed") is True and not allow_test_accessed:
            raise ValueError(f"Test-accessed artifact requires --allow-test-accessed: {complete_path}")
        if completion.get("test_accessed") not in {False, True}:
            raise ValueError(f"Invalid test_accessed flag in {complete_path}: {completion.get('test_accessed')}")
        if not completion.get("joint_run_id"):
            raise ValueError(f"Empty joint_run_id in {complete_path}")
        if not completion.get("forward_contract_sha256"):
            raise ValueError(f"Empty forward_contract_sha256 in {complete_path}")
        if completion.get("s1_prefix") != s1_prefix:
            raise ValueError(f"Invalid S1 prefix in {complete_path}: {completion.get('s1_prefix')}")
        if completion.get("s2_prefix") != s2_prefix:
            raise ValueError(f"Invalid S2 prefix in {complete_path}: {completion.get('s2_prefix')}")
        if completion.get("s1_pooling_contract", {}).get("representation") != s1_representation:
            raise ValueError(f"Invalid S1 pooling contract in {complete_path}")
        if completion.get("s2_span_partition_pooling_contract", {}).get("representation") != "span_mean_2":
            raise ValueError(f"Invalid S2 pooling contract in {complete_path}")
        if completion.get("s1_embeddings_sha256") != extractor.sha256_file(s1_paths["embeddings"]):
            raise ValueError(f"S1 embedding SHA mismatch: {s1_paths['embeddings']}")
        if completion.get("s1_metadata_sha256") != extractor.sha256_file(s1_paths["metadata"]):
            raise ValueError(f"S1 metadata SHA mismatch: {s1_paths['metadata']}")
        if completion.get("s2_embeddings_sha256") != extractor.sha256_file(s2_paths["embeddings"]):
            raise ValueError(f"S2 embedding SHA mismatch: {s2_paths['embeddings']}")
        if completion.get("s2_metadata_sha256") != extractor.sha256_file(s2_paths["metadata"]):
            raise ValueError(f"S2 metadata SHA mismatch: {s2_paths['metadata']}")
        s1_array = np.load(s1_paths["embeddings"], mmap_mode="r")
        s2_array = np.load(s2_paths["embeddings"], mmap_mode="r")
        if s1_array.ndim != 2 or s2_array.ndim != 2:
            raise ValueError(f"Embedding arrays must be rank-2 for {input_kind} shard {shard_idx}")
        if s1_array.dtype != np.float32 or s2_array.dtype != np.float32:
            raise ValueError(f"Embedding dtype must be float32 for {input_kind} shard {shard_idx}: {s1_array.dtype}, {s2_array.dtype}")
        if s1_array.shape[1] != s2_array.shape[1]:
            raise ValueError(f"S1/S2 hidden size mismatch for {input_kind} shard {shard_idx}: {s1_array.shape[1]} != {s2_array.shape[1]}")
        hidden_sizes.add(int(s1_array.shape[1]))
        s1_meta = pd.read_parquet(s1_paths["metadata"])
        s2_meta = pd.read_parquet(s2_paths["metadata"])
        extractor.validate_text_free_metadata(s1_meta, f"{input_kind} S1 validation")
        extractor.validate_text_free_metadata(s2_meta, f"{input_kind} S2 validation")
        if s1_array.shape[0] != len(s1_meta):
            raise ValueError(f"S1 embedding row count does not match metadata for {s1_paths['embeddings']}")
        if s2_array.shape[0] != len(s2_meta):
            raise ValueError(f"S2 embedding row count does not match metadata for {s2_paths['embeddings']}")
        if len(s1_meta) != int(completion.get("s1_rows", -1)):
            raise ValueError(f"S1 row count mismatch in {complete_path}")
        if len(s2_meta) != int(completion.get("s2_rows", -1)):
            raise ValueError(f"S2 row count mismatch in {complete_path}")
        if set(s1_meta.get("representation", pd.Series(dtype=str)).astype(str)) != {s1_representation}:
            raise ValueError(f"S1 metadata representation mismatch in {s1_paths['metadata']}")
        if set(s2_meta.get("representation", pd.Series(dtype=str)).astype(str)) != {"span_mean_2"}:
            raise ValueError(f"S2 metadata representation mismatch in {s2_paths['metadata']}")
        s1_rows += len(s1_meta)
        s2_rows += len(s2_meta)
        run_ids.add(str(completion.get("joint_run_id")))
        forward_contracts.add(str(completion.get("forward_contract_sha256")))

    s1_summary_rows = read_summary_rows(root, s1_prefix)
    s2_summary_rows = read_summary_rows(root, s2_prefix)
    if s1_summary_rows is not None and s1_summary_rows != s1_rows:
        raise ValueError(f"S1 summary row mismatch for {input_kind}: {s1_summary_rows} != {s1_rows}")
    if s2_summary_rows is not None and s2_summary_rows != s2_rows:
        raise ValueError(f"S2 summary row mismatch for {input_kind}: {s2_summary_rows} != {s2_rows}")
    if len(run_ids) != 1:
        raise ValueError(f"{input_kind} shards do not share exactly one joint_run_id: {sorted(run_ids)}")
    if len(forward_contracts) != 1:
        raise ValueError(f"{input_kind} shards do not share exactly one forward_contract_sha256: {sorted(forward_contracts)}")
    if len(hidden_sizes) != 1:
        raise ValueError(f"{input_kind} shards do not share exactly one hidden size: {sorted(hidden_sizes)}")
    if str(provenance.get("joint_run_id")) != sorted(run_ids)[0]:
        raise ValueError(f"Provenance joint_run_id mismatch for {input_kind}")
    if str(provenance.get("forward_contract_sha256")) != sorted(forward_contracts)[0]:
        raise ValueError(f"Provenance forward_contract_sha256 mismatch for {input_kind}")

    return {
        "input_kind": input_kind,
        "n_shards": len(s1_indices),
        "s1_rows": int(s1_rows),
        "s2_rows": int(s2_rows),
        "hidden_size": sorted(hidden_sizes)[0],
        "dtype": "float32",
        "joint_run_id": sorted(run_ids)[0],
        "forward_contract_sha256": sorted(forward_contracts)[0],
        "raw_text_stored": False,
        "test_accessed": bool(provenance.get("test_accessed", False)),
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DAPT Mamba3 joint S1/S2 sharded artifacts.")
    parser.add_argument("--joint-cache-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-test-accessed", action="store_true")
    args = parser.parse_args()

    root = resolve_path(args.joint_cache_dir)
    rows = [validate_prefix(root, input_kind, allow_test_accessed=args.allow_test_accessed) for input_kind in ["section", "chunk"]]
    out = pd.DataFrame(rows)
    if out["joint_run_id"].nunique() != 1:
        raise ValueError("Section/chunk artifacts do not share one joint_run_id")
    output_path = resolve_path(args.output) if args.output else root / "joint_s1_s2_artifact_validation_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(out.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
