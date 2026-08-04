from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.data.local_representations import (  # noqa: E402
    assert_exact_token_coverage,
    content_positions_from_ids,
    evidence_spans_for_positions,
)
from notesm_v2.models.notesm_v2 import extract_marker_hidden_states  # noqa: E402
from notesm_v2.models.special_tokens import add_project_special_tokens  # noqa: E402


MODEL_ID = "state-spaces/mamba3-siso-187m"
TOKENIZER_ID = "meta-llama/Llama-3.1-8B"
FORBIDDEN_METADATA = {
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
SAFE_SECTION_COLUMNS = [
    "example_id",
    "split",
    "canonical_section",
    "section_order",
    "num_occurrences",
    "char_length",
    "token_length",
    "used_repeated_section_merge",
    "use_chunk_fallback",
]
SAFE_CHUNK_COLUMNS = ["example_id", "split", "chunk_order", "token_length"]
SPAN_COLUMNS = [
    "representation",
    "evidence_slot",
    "within_segment_position",
    "token_start_position",
    "token_end_position",
    "token_count",
]
COMPLETION_SCHEMA_VERSION = 5


def _normalize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return ""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (int, float, bool)):
        return value
    return str(value)


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_sha256(value: Any) -> str:
    return hashlib.sha256(str(_normalize_scalar(value)).encode("utf-8")).hexdigest()


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - exercised only when tqdm is absent.

    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, **_: Any) -> None:
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or [])

        def __enter__(self):
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def update(self, _: int) -> None:
            return None


@dataclass(frozen=True)
class ExtractionSpec:
    input_kind: str
    text_col: str
    marker_token: str
    order_col: str
    safe_cols: list[str]

    @property
    def prefix(self) -> str:
        return self.input_kind


@dataclass
class S1AuditStats:
    n_compared: int = 0
    max_abs_diff: float = 0.0
    sum_abs_diff: float = 0.0
    n_values: int = 0
    min_cosine: float = 1.0
    sum_cosine: float = 0.0

    def update(self, computed: np.ndarray, expected: np.ndarray) -> None:
        if computed.shape != expected.shape:
            raise ValueError(f"S1 audit shape mismatch: computed={computed.shape} expected={expected.shape}")
        if computed.size == 0:
            return
        diff = np.abs(computed.astype(np.float32) - expected.astype(np.float32))
        cos_num = np.sum(computed * expected, axis=1)
        cos_den = np.linalg.norm(computed, axis=1) * np.linalg.norm(expected, axis=1)
        cosine = cos_num / np.clip(cos_den, 1e-12, None)
        self.n_compared += int(computed.shape[0])
        self.max_abs_diff = max(self.max_abs_diff, float(np.max(diff)))
        self.sum_abs_diff += float(np.sum(diff))
        self.n_values += int(diff.size)
        self.min_cosine = min(self.min_cosine, float(np.min(cosine)))
        self.sum_cosine += float(np.sum(cosine))

    def merge(self, other: "S1AuditStats") -> None:
        self.n_compared += int(other.n_compared)
        self.max_abs_diff = max(float(self.max_abs_diff), float(other.max_abs_diff))
        self.sum_abs_diff += float(other.sum_abs_diff)
        self.n_values += int(other.n_values)
        self.min_cosine = min(float(self.min_cosine), float(other.min_cosine))
        self.sum_cosine += float(other.sum_cosine)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "S1AuditStats":
        out = cls()
        out.n_compared = int(values.get("n_compared", 0))
        out.max_abs_diff = float(values.get("max_abs_diff", 0.0))
        out.n_values = int(values.get("n_values", 0))
        out.sum_abs_diff = float(values.get("sum_abs_diff", out.max_abs_diff * out.n_values))
        out.min_cosine = float(values.get("min_cosine", 1.0))
        out.sum_cosine = float(values.get("sum_cosine", values.get("mean_cosine", 1.0) * max(out.n_compared, 1)))
        return out

    @property
    def mean_abs_diff(self) -> float:
        return self.sum_abs_diff / max(self.n_values, 1)

    @property
    def mean_cosine(self) -> float:
        return self.sum_cosine / max(self.n_compared, 1)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_compared": self.n_compared,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "sum_abs_diff": self.sum_abs_diff,
            "n_values": self.n_values,
            "min_cosine": self.min_cosine,
            "mean_cosine": self.mean_cosine,
            "sum_cosine": self.sum_cosine,
        }


class S1AuditIndex:
    def __init__(self, root: Path, prefix: str, key_cols: list[str]) -> None:
        self.root = root
        self.prefix = prefix
        self.key_cols = key_cols
        self.embedding_paths = sorted(root.glob(f"{prefix}_embeddings_shard_*.npy"))
        metadata_paths = sorted(root.glob(f"{prefix}_embedding_metadata_shard_*.parquet"))
        if len(self.embedding_paths) != len(metadata_paths) or not self.embedding_paths:
            raise FileNotFoundError(f"Missing or mismatched S1 {prefix} shards under {root}")
        frames: list[pd.DataFrame] = []
        for shard_idx, metadata_path in enumerate(metadata_paths):
            metadata = pd.read_parquet(metadata_path)
            missing = [col for col in key_cols if col not in metadata.columns]
            if missing:
                raise ValueError(f"S1 metadata {metadata_path} is missing key columns: {missing}")
            frame = metadata[key_cols].copy()
            frame["example_id"] = frame["example_id"].astype(str)
            if "row_index_in_shard" in metadata.columns:
                frame["_s1_row"] = metadata["row_index_in_shard"].astype(int).to_numpy()
            else:
                frame["_s1_row"] = np.arange(len(metadata), dtype=int)
            frame["_s1_shard"] = shard_idx
            frames.append(frame)
        index = pd.concat(frames, ignore_index=True)
        duplicated = index.duplicated(key_cols, keep=False)
        if duplicated.any():
            sample = index.loc[duplicated, key_cols].head(5).to_dict("records")
            raise ValueError(f"S1 audit keys are not unique for {prefix}: {sample}")
        self.index = index
        self.lookup: dict[tuple[Any, ...], tuple[int, int]] = {}
        for row_dict in index.to_dict("records"):
            key = tuple(_normalize_scalar(row_dict[col]) for col in key_cols)
            self.lookup[key] = (int(row_dict["_s1_shard"]), int(row_dict["_s1_row"]))

    def vectors_for(self, keys: pd.DataFrame) -> np.ndarray:
        query = keys[self.key_cols].copy()
        query["example_id"] = query["example_id"].astype(str)
        entries: list[tuple[int, int, int]] = []
        missing: list[dict[str, Any]] = []
        for query_order, row in enumerate(query.to_dict("records")):
            key = tuple(_normalize_scalar(row[col]) for col in self.key_cols)
            value = self.lookup.get(key)
            if value is None:
                missing.append(row)
                continue
            entries.append((query_order, value[0], value[1]))
        if missing:
            raise ValueError(f"S1 audit could not align computed rows by key: {missing[:5]}")
        merged = pd.DataFrame(entries, columns=["_query_order", "_s1_shard", "_s1_row"])
        out: np.ndarray | None = None
        for shard_idx, group in merged.groupby("_s1_shard", sort=False):
            array = np.load(self.embedding_paths[int(shard_idx)])
            if out is None:
                out = np.empty((len(merged), array.shape[1]), dtype=np.float32)
            out[group["_query_order"].to_numpy(dtype=int)] = array[group["_s1_row"].to_numpy(dtype=int)]
        if out is None:
            return np.zeros((0, 0), dtype=np.float32)
        return out


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_file_manifest_fingerprint(root: Path, prefix: str) -> str:
    patterns = [
        f"{prefix}_embedding_shards_summary.csv",
        f"{prefix}_embeddings_shard_*.npy",
        f"{prefix}_embedding_metadata_shard_*.parquet",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.glob(pattern)))
    if not files:
        raise FileNotFoundError(f"Cannot fingerprint missing {prefix} S1 cache under {root}")
    records = []
    for path in sorted(files):
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return canonical_json_sha256({"prefix": prefix, "files": records})


def source_cache_basename(input_kind: str) -> str:
    if input_kind == "section":
        return "section_cache"
    if input_kind == "chunk":
        return "fallback_chunk_cache"
    raise ValueError(f"Unsupported input_kind: {input_kind}")


def manifest_records_for_files(root: Path, files: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def source_kind_cache_file_manifest_fingerprint(root: Path, input_kind: str, split: str, allow_test: bool = False) -> str:
    filename = source_cache_basename(input_kind)
    files = []
    for split_name in validate_split(split, allow_test=allow_test):
        path = root / f"{filename}_{split_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        files.append(path)
    records = manifest_records_for_files(root, files)
    return canonical_json_sha256({"input_kind": input_kind, "split": split, "files": records})


def source_bundle_cache_file_manifest_fingerprint(root: Path, split: str, allow_test: bool = False) -> str:
    files = []
    for input_kind in ["section", "chunk"]:
        filename = source_cache_basename(input_kind)
        for split_name in validate_split(split, allow_test=allow_test):
            path = root / f"{filename}_{split_name}.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            files.append(path)
    records = manifest_records_for_files(root, sorted(files))
    return canonical_json_sha256(
        {
            "bundle": "section_cache_and_fallback_chunk_cache",
            "split": split,
            "files": records,
        }
    )


def compare_sharded_embedding_dirs(
    left: Path,
    right: Path,
    prefix: str,
    *,
    max_abs_tol: float = 1e-6,
) -> dict[str, Any]:
    left_embeddings = sorted(left.glob(f"{prefix}_embeddings_shard_*.npy"))
    right_embeddings = sorted(right.glob(f"{prefix}_embeddings_shard_*.npy"))
    left_metadata = sorted(left.glob(f"{prefix}_embedding_metadata_shard_*.parquet"))
    right_metadata = sorted(right.glob(f"{prefix}_embedding_metadata_shard_*.parquet"))
    if not left_embeddings or not right_embeddings:
        raise ValueError(f"Missing {prefix} embedding shards for comparison")
    left_embedding_names = [path.name for path in left_embeddings]
    right_embedding_names = [path.name for path in right_embeddings]
    left_metadata_names = [path.name for path in left_metadata]
    right_metadata_names = [path.name for path in right_metadata]
    if left_embedding_names != right_embedding_names:
        raise ValueError(f"Embedding shard list mismatch: {left_embedding_names} != {right_embedding_names}")
    if left_metadata_names != right_metadata_names:
        raise ValueError(f"Metadata shard list mismatch: {left_metadata_names} != {right_metadata_names}")
    if len(left_embeddings) != len(left_metadata):
        raise ValueError(f"Mismatched {prefix} embedding/metadata counts in {left}")

    global_max_abs = 0.0
    total_rows = 0
    for left_embedding, right_embedding, left_meta_path, right_meta_path in zip(
        left_embeddings,
        right_embeddings,
        left_metadata,
        right_metadata,
        strict=True,
    ):
        left_array = np.load(left_embedding)
        right_array = np.load(right_embedding)
        if left_array.shape != right_array.shape:
            raise ValueError(f"Shape mismatch for {left_embedding.name}: {left_array.shape} != {right_array.shape}")
        max_abs = float(np.max(np.abs(left_array.astype(np.float32) - right_array.astype(np.float32)))) if left_array.size else 0.0
        global_max_abs = max(global_max_abs, max_abs)
        if max_abs > max_abs_tol:
            raise ValueError(f"Embedding determinism failed for {left_embedding.name}: max_abs={max_abs} tol={max_abs_tol}")
        left_meta = pd.read_parquet(left_meta_path)
        right_meta = pd.read_parquet(right_meta_path)
        pd.testing.assert_frame_equal(left_meta, right_meta, check_dtype=False)
        total_rows += int(len(left_meta))
    return {
        "prefix": prefix,
        "n_shards": int(len(left_embeddings)),
        "n_rows": int(total_rows),
        "max_abs": float(global_max_abs),
        "max_abs_tol": float(max_abs_tol),
    }


def shard_source_key_digest(shard: pd.DataFrame, spec: ExtractionSpec) -> str:
    safe_cols = [col for col in spec.safe_cols if col in shard.columns]
    cols = [*safe_cols, spec.text_col]
    seen: set[str] = set()
    cols = [col for col in cols if not (col in seen or seen.add(col))]
    missing = [col for col in ["example_id", "split", spec.order_col, spec.text_col] if col not in shard.columns]
    if missing:
        raise ValueError(f"Cannot compute source-key digest; missing columns: {missing}")
    records = [
        {
            **{col: _normalize_scalar(row[col]) for col in safe_cols},
            f"{spec.text_col}_sha256": text_sha256(row[spec.text_col]),
        }
        for row in shard[cols].to_dict("records")
    ]
    return canonical_json_sha256(
        {
            "safe_columns": safe_cols,
            "text_column": spec.text_col,
            "records": records,
            "raw_text_stored": False,
        }
    )


def tokenizer_contract(tokenizer: Any, tokenizer_id: str, marker_token: str) -> dict[str, Any]:
    marker_id = tokenizer.convert_tokens_to_ids(marker_token)
    if marker_id is None or marker_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"Marker token is not in tokenizer vocabulary: {marker_token}")
    project_marker_ids = {}
    for token in ["<SEC_SUM>", "<CHUNK_SUM>"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        project_marker_ids[token] = None if token_id is None else int(token_id)
    try:
        vocab_sha256 = canonical_json_sha256(tokenizer.get_vocab())
    except Exception:
        vocab_sha256 = ""
    try:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        backend_text = backend.to_str() if backend is not None and hasattr(backend, "to_str") else ""
        backend_tokenizer_sha256 = hashlib.sha256(backend_text.encode("utf-8")).hexdigest() if backend_text else ""
    except Exception:
        backend_tokenizer_sha256 = ""
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    try:
        init_kwargs_sha256 = canonical_json_sha256(init_kwargs)
    except Exception:
        init_kwargs_sha256 = ""
    tokenizer_commit_hash = str(
        getattr(tokenizer, "_commit_hash", "")
        or init_kwargs.get("_commit_hash", "")
        or init_kwargs.get("revision", "")
    )
    payload = {
        "tokenizer_id": tokenizer_id,
        "tokenizer_class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(len(tokenizer)),
        "vocab_sha256": vocab_sha256,
        "backend_tokenizer_sha256": backend_tokenizer_sha256,
        "init_kwargs_sha256": init_kwargs_sha256,
        "tokenizer_commit_hash": tokenizer_commit_hash,
        "pad_token_id": None if getattr(tokenizer, "pad_token_id", None) is None else int(tokenizer.pad_token_id),
        "eos_token_id": None if getattr(tokenizer, "eos_token_id", None) is None else int(tokenizer.eos_token_id),
        "bos_token_id": None if getattr(tokenizer, "bos_token_id", None) is None else int(tokenizer.bos_token_id),
        "unk_token_id": None if getattr(tokenizer, "unk_token_id", None) is None else int(tokenizer.unk_token_id),
        "all_special_ids": sorted({int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None}),
        "marker_token": marker_token,
        "marker_id": int(marker_id),
        "project_marker_ids": project_marker_ids,
    }
    run_payload = dict(payload)
    run_payload.pop("marker_token", None)
    run_payload.pop("marker_id", None)
    return {
        "tokenizer_vocab_size": payload["vocab_size"],
        "marker_id": payload["marker_id"],
        "tokenizer_backend_sha256": payload["backend_tokenizer_sha256"],
        "tokenizer_commit_hash": payload["tokenizer_commit_hash"],
        "tokenizer_fingerprint": canonical_json_sha256(payload),
        "tokenizer_run_fingerprint": canonical_json_sha256(run_payload),
    }


def extraction_spec(input_kind: str) -> ExtractionSpec:
    if input_kind == "section":
        return ExtractionSpec("section", "merged_section_text", "<SEC_SUM>", "section_order", SAFE_SECTION_COLUMNS)
    if input_kind == "chunk":
        return ExtractionSpec("chunk", "chunk_text", "<CHUNK_SUM>", "chunk_order", SAFE_CHUNK_COLUMNS)
    raise ValueError(f"Unsupported input_kind: {input_kind}")


def validate_split(split: str, allow_test: bool = False) -> list[str]:
    if split == "train_val":
        return ["train", "val"]
    if split in {"train", "val"}:
        return [split]
    if split == "test" and allow_test:
        return ["test"]
    raise ValueError("This extractor is train/validation only; test/all splits are rejected")


def read_source_cache(
    section_cache_dir: Path,
    input_kind: str,
    split: str,
    limit_examples: int | None,
    allow_test: bool = False,
) -> pd.DataFrame:
    spec = extraction_spec(input_kind)
    frames: list[pd.DataFrame] = []
    for split_name in validate_split(split, allow_test=allow_test):
        filename = "section_cache" if input_kind == "section" else "fallback_chunk_cache"
        path = section_cache_dir / f"{filename}_{split_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        if "split" not in frame.columns:
            frame["split"] = split_name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        return combined
    missing = [col for col in ["example_id", spec.text_col, spec.order_col] if col not in combined.columns]
    if missing:
        raise ValueError(f"Source cache is missing required columns: {missing}")
    combined["example_id"] = combined["example_id"].astype(str)
    if limit_examples is not None:
        selected = set(combined["example_id"].drop_duplicates().head(limit_examples).tolist())
        combined = combined[combined["example_id"].isin(selected)].copy()
    return combined.reset_index(drop=True)


def validate_text_free_metadata(metadata: pd.DataFrame, label: str) -> None:
    forbidden = sorted(FORBIDDEN_METADATA & set(metadata.columns))
    if forbidden:
        raise ValueError(f"{label} metadata contains restricted columns: {forbidden}")


def load_official_mamba3(
    model_id: str,
    tokenizer_id: str,
    checkpoint_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any, dict[str, Any]]:
    from notesm_v2.models.backbone_registry import get_backbone_spec, load_tokenizer_and_model

    spec = get_backbone_spec("mamba3_siso_187m")
    if model_id != spec.model_id or tokenizer_id != spec.tokenizer_id:
        raise ValueError(
            "Mamba3 S1/S2 locked extractor must use the registered Mamba3 backbone: "
            f"got model_id={model_id} tokenizer_id={tokenizer_id}"
        )
    tokenizer, model = load_tokenizer_and_model(spec, torch_dtype=dtype)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    add_project_special_tokens(tokenizer, model=model, init="mean")
    tokenizer.padding_side = "right"
    state_path = checkpoint_dir / "pytorch_training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if "model_state" not in state:
        raise ValueError(f"{state_path} does not contain model_state")
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    checkpoint_info = {
        "path": str(checkpoint_dir),
        "state_file": str(state_path),
        "global_step": state.get("global_step"),
        "scheduler_total_steps": state.get("args", {}).get("scheduler_total_steps"),
        "sha256": sha256_file(state_path),
    }
    return tokenizer, model, checkpoint_info


def append_marker_and_pad(
    rows: Sequence[Sequence[int]],
    marker_id: int,
    pad_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    marked = [list(row) + [int(marker_id)] for row in rows]
    max_len = max((len(row) for row in marked), default=1)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    for row in marked:
        pad_count = max_len - len(row)
        input_ids.append(row + [int(pad_id)] * pad_count)
        attention_mask.append([1] * len(row) + [0] * pad_count)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
    }


def encode_with_marker(
    texts: Sequence[str],
    tokenizer: Any,
    marker_token: str,
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    marker_id = tokenizer.convert_tokens_to_ids(marker_token)
    if marker_id is None or marker_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"Marker token is not in tokenizer vocabulary: {marker_token}")
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=max_length - 1,
    )
    return append_marker_and_pad(encoded["input_ids"], int(marker_id), int(tokenizer.pad_token_id), device), int(marker_id)


def last_hidden_state(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    if isinstance(outputs, tuple):
        return outputs[0]
    raise TypeError("Could not extract hidden states from model outputs")


def backbone_hidden(model: Any, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    input_ids = encoded["input_ids"]
    if callable(model):
        try:
            return last_hidden_state(model(**encoded, use_cache=False))
        except TypeError:
            try:
                return last_hidden_state(model(**encoded))
            except TypeError:
                pass
    if hasattr(model, "backbone"):
        try:
            return last_hidden_state(model.backbone(input_ids))
        except TypeError:
            return last_hidden_state(model.backbone(input_ids=input_ids))
    if callable(model):
        try:
            return last_hidden_state(model(input_ids=input_ids, use_cache=False))
        except TypeError:
            return last_hidden_state(model(input_ids=input_ids))
    raise TypeError("Model is neither callable nor exposes a backbone")


def valid_positions_by_row(encoded: dict[str, torch.Tensor], tokenizer: Any, marker_id: int) -> list[np.ndarray]:
    input_ids = encoded["input_ids"].detach().cpu().tolist()
    attention = encoded["attention_mask"].detach().cpu().tolist()
    excluded = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None)
    excluded.add(int(marker_id))
    if getattr(tokenizer, "pad_token_id", None) is not None:
        excluded.add(int(tokenizer.pad_token_id))
    return [
        content_positions_from_ids(row_ids, row_mask, excluded)
        for row_ids, row_mask in zip(input_ids, attention, strict=True)
    ]


def content_mask_tensor(encoded: dict[str, torch.Tensor], tokenizer: Any, marker_id: int) -> torch.Tensor:
    input_ids = encoded["input_ids"]
    mask = encoded["attention_mask"].bool()
    excluded = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None)
    excluded.add(int(marker_id))
    if getattr(tokenizer, "pad_token_id", None) is not None:
        excluded.add(int(tokenizer.pad_token_id))
    for token_id in sorted(excluded):
        mask = mask & input_ids.ne(int(token_id))
    return mask


def span_specs_from_positions(valid_positions: Sequence[np.ndarray]) -> tuple[list[tuple[int, np.ndarray]], list[dict[str, int]]]:
    specs: list[tuple[int, np.ndarray]] = []
    rows: list[dict[str, int]] = []
    for local_row, positions in enumerate(valid_positions):
        spans = evidence_spans_for_positions(positions, "span_mean_2")
        assert_exact_token_coverage(positions, [span for span, _ in spans])
        for span_positions, metadata in spans:
            specs.append((local_row, np.asarray(span_positions, dtype=int)))
        rows.append({"local_row": local_row, **metadata})
    return specs, rows


def pool_span_mean2_gpu(hidden: torch.Tensor, content_mask: torch.Tensor) -> tuple[torch.Tensor, pd.DataFrame]:
    hidden_float = hidden.float()
    batch, seq_len, hidden_size = hidden_float.shape
    if not bool(content_mask.any()):
        empty = pd.DataFrame(
            columns=[
                "local_row",
                "representation",
                "evidence_slot",
                "within_segment_position",
                "token_start_position",
                "token_end_position",
                "token_count",
            ]
        )
        return torch.zeros((0, hidden_size), dtype=torch.float32, device=hidden.device), empty

    counts_by_row = content_mask.sum(dim=1)
    ranks = content_mask.long().cumsum(dim=1) - 1
    first_span_lengths = (counts_by_row + 1) // 2
    slot_matrix = (ranks >= first_span_lengths.unsqueeze(1)).long()
    coords = torch.nonzero(content_mask, as_tuple=False)
    row_idx = coords[:, 0]
    token_pos = coords[:, 1]
    evidence_slot = slot_matrix[row_idx, token_pos]
    span_ids = row_idx * 2 + evidence_slot
    values = hidden_float[row_idx, token_pos]

    n_possible_spans = batch * 2
    sums = torch.zeros((n_possible_spans, hidden_size), dtype=torch.float32, device=hidden.device)
    sums.index_add_(0, span_ids, values)
    token_counts = torch.bincount(span_ids, minlength=n_possible_spans).to(hidden.device)
    if int(token_counts.sum().detach().cpu().item()) != int(content_mask.sum().detach().cpu().item()):
        raise ValueError("span_mean_2 assignment lost content tokens")

    starts = torch.full((n_possible_spans,), seq_len, dtype=torch.long, device=hidden.device)
    ends = torch.full((n_possible_spans,), -1, dtype=torch.long, device=hidden.device)
    starts.scatter_reduce_(0, span_ids, token_pos, reduce="amin", include_self=True)
    ends.scatter_reduce_(0, span_ids, token_pos, reduce="amax", include_self=True)

    active = torch.nonzero(token_counts > 0, as_tuple=False).squeeze(-1)
    pooled = sums[active] / token_counts[active].float().unsqueeze(-1).clamp(min=1.0)
    active_cpu = active.detach().cpu().numpy().astype(int)
    starts_cpu = starts[active].detach().cpu().numpy().astype(int)
    ends_cpu = ends[active].detach().cpu().numpy().astype(int)
    counts_cpu = token_counts[active].detach().cpu().numpy().astype(int)
    rows = [
        {
            "local_row": int(span_id // 2),
            "representation": "span_mean_2",
            "evidence_slot": int(span_id % 2),
            "within_segment_position": int(span_id % 2),
            "token_start_position": int(start),
            "token_end_position": int(end) + 1,
            "token_count": int(count),
        }
        for span_id, start, end, count in zip(active_cpu, starts_cpu, ends_cpu, counts_cpu, strict=True)
    ]
    return pooled, pd.DataFrame(rows)


def pool_span_specs(hidden: torch.Tensor, specs: Sequence[tuple[int, np.ndarray]]) -> torch.Tensor:
    hidden_float = hidden.float()
    hidden_size = int(hidden_float.shape[-1])
    if not specs:
        return torch.zeros((0, hidden_size), dtype=torch.float32, device=hidden.device)
    flat_rows: list[int] = []
    flat_positions: list[int] = []
    flat_spans: list[int] = []
    for span_idx, (row_idx, positions) in enumerate(specs):
        for pos in positions.tolist():
            flat_rows.append(int(row_idx))
            flat_positions.append(int(pos))
            flat_spans.append(span_idx)
    row_tensor = torch.tensor(flat_rows, dtype=torch.long, device=hidden.device)
    pos_tensor = torch.tensor(flat_positions, dtype=torch.long, device=hidden.device)
    span_tensor = torch.tensor(flat_spans, dtype=torch.long, device=hidden.device)
    values = hidden_float[row_tensor, pos_tensor]
    sums = torch.zeros((len(specs), hidden_size), dtype=torch.float32, device=hidden.device)
    sums.index_add_(0, span_tensor, values)
    counts = torch.bincount(span_tensor, minlength=len(specs)).to(hidden.device).float().clamp(min=1.0)
    return sums / counts.unsqueeze(-1)


def extract_joint_batch(
    batch: pd.DataFrame,
    *,
    spec: ExtractionSpec,
    tokenizer: Any,
    model: Any,
    max_length: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    encoded, marker_id = encode_with_marker(batch[spec.text_col].astype(str).tolist(), tokenizer, spec.marker_token, max_length, device)
    with torch.inference_mode():
        hidden = backbone_hidden(model, encoded)
        s1 = extract_marker_hidden_states(hidden.float(), encoded["input_ids"], int(marker_id))
        content_mask = content_mask_tensor(encoded, tokenizer, int(marker_id))
        s2, span_rows = pool_span_mean2_gpu(hidden, content_mask)
    s1_cpu = s1.detach().cpu().numpy().astype(np.float32, copy=False)
    s2_cpu = s2.detach().cpu().numpy().astype(np.float32, copy=False)
    return s1_cpu, s2_cpu, span_rows


def build_span_metadata(
    batch: pd.DataFrame,
    span_rows: pd.DataFrame,
    spec: ExtractionSpec,
    shard_idx: int,
    output_offset: int,
) -> pd.DataFrame:
    safe_cols = [col for col in spec.safe_cols if col in batch.columns]
    base = batch[safe_cols].copy()
    base["_local_row"] = np.arange(len(base), dtype=int)
    if span_rows.empty:
        columns = [*safe_cols, *SPAN_COLUMNS, "shard_index", "row_index_in_shard", "global_row_index", "row_index"]
        return pd.DataFrame(columns=columns)
    expanded = span_rows.merge(base, left_on="local_row", right_on="_local_row", how="left", validate="many_to_one")
    expanded = expanded.drop(columns=["local_row", "_local_row"])
    expanded["representation"] = "span_mean_2"
    ordered = [col for col in safe_cols if col in expanded.columns] + SPAN_COLUMNS
    metadata = expanded[ordered].copy()
    metadata["example_id"] = metadata["example_id"].astype(str)
    metadata["shard_index"] = int(shard_idx)
    metadata["row_index_in_shard"] = np.arange(len(metadata), dtype=int)
    metadata["global_row_index"] = np.arange(output_offset, output_offset + len(metadata), dtype=int)
    metadata["row_index"] = metadata["global_row_index"]
    validate_text_free_metadata(metadata, f"{spec.prefix} span2")
    return metadata


def build_s1_metadata(batch: pd.DataFrame, spec: ExtractionSpec, shard_idx: int, output_offset: int) -> pd.DataFrame:
    safe_cols = [col for col in spec.safe_cols if col in batch.columns]
    metadata = batch[safe_cols].copy()
    metadata["example_id"] = metadata["example_id"].astype(str)
    metadata["representation"] = s1_representation(spec)
    metadata["shard_index"] = int(shard_idx)
    metadata["row_index_in_shard"] = np.arange(len(metadata), dtype=int)
    metadata["global_row_index"] = np.arange(output_offset, output_offset + len(metadata), dtype=int)
    metadata["row_index"] = metadata["global_row_index"]
    validate_text_free_metadata(metadata, f"{spec.prefix} s1")
    return metadata


def shard_paths(output_dir: Path, prefix: str, shard_idx: int) -> dict[str, Path]:
    return {
        "embeddings": output_dir / f"{prefix}_embeddings_shard_{shard_idx:03d}.npy",
        "metadata": output_dir / f"{prefix}_embedding_metadata_shard_{shard_idx:03d}.parquet",
        "complete": output_dir / f"{prefix}_embedding_shard_{shard_idx:03d}.complete.json",
    }


def shard_data_paths(output_dir: Path, prefix: str, shard_idx: int) -> dict[str, Path]:
    return {
        "embeddings": output_dir / f"{prefix}_embeddings_shard_{shard_idx:03d}.npy",
        "metadata": output_dir / f"{prefix}_embedding_metadata_shard_{shard_idx:03d}.parquet",
    }


def joint_completion_path(output_dir: Path, input_kind: str, shard_idx: int) -> Path:
    return output_dir / f"{input_kind}_joint_embedding_shard_{shard_idx:03d}.complete.json"


def check_resume_state(paths: dict[str, Path], resume: bool) -> str:
    exists = {name: path.exists() for name, path in paths.items()}
    if all(exists.values()):
        return "complete" if resume else "exists"
    if any(exists.values()):
        raise ValueError(f"Partial shard files are present and cannot be resumed safely: {paths}")
    return "missing"


def check_joint_resume_state(
    s1_paths: dict[str, Path] | None,
    s2_paths: dict[str, Path],
    complete_path: Path,
    resume: bool,
    *,
    stale_label: str | None = None,
) -> str:
    required = [s2_paths["embeddings"], s2_paths["metadata"], complete_path]
    if s1_paths is not None:
        required.extend([s1_paths["embeddings"], s1_paths["metadata"]])
    present = [path.exists() for path in required]
    if all(present):
        return "complete" if resume else "exists"
    if any(present):
        if resume and stale_label:
            target = move_partial_joint_shard_to_stale(required, stale_label)
            print(f"Moved partial joint shard to stale directory: {target}", flush=True)
            return "missing"
        missing = [str(path) for path, exists in zip(required, present, strict=True) if not exists]
        raise ValueError(f"Partial joint shard files are present and cannot be resumed safely. Missing: {missing}")
    return "missing"


def move_partial_joint_shard_to_stale(paths: Sequence[Path], stale_label: str) -> Path:
    present = [path for path in paths if path.exists()]
    if not present:
        raise ValueError("No partial joint shard files were present to move")
    output_dir = present[0].parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stale_root = output_dir / "_stale_partial_joint_shards"
    stale_root.mkdir(parents=True, exist_ok=True)
    target = stale_root / f"{stale_label}_{stamp}"
    counter = 0
    while target.exists():
        counter += 1
        target = stale_root / f"{stale_label}_{stamp}_{counter}"
    target.mkdir(parents=True)
    for path in present:
        shutil.move(str(path), str(target / path.name))
    tmp_candidates = list(output_dir.glob("*.tmp.*")) + list(output_dir.glob(".*.tmp.*"))
    for tmp in dict.fromkeys(tmp_candidates):
        if any(tmp.name.startswith(f".{path.name}.tmp.") for path in paths):
            shutil.move(str(tmp), str(target / tmp.name))
    return target


def s1_representation(spec: ExtractionSpec) -> str:
    return "sec_sum_1" if spec.input_kind == "section" else "chunk_sum_1"


def s2_prefix(spec: ExtractionSpec, args: argparse.Namespace) -> str:
    if getattr(args, "write_s1", False):
        suffix = str(getattr(args, "s2_prefix_suffix", "span2")).strip("_")
        if not suffix:
            raise ValueError("--s2-prefix-suffix must not be empty when --write-s1 is enabled")
        return f"{spec.prefix}_{suffix}"
    return spec.prefix


def s1_pooling_contract(spec: ExtractionSpec) -> dict[str, Any]:
    return {
        "representation": s1_representation(spec),
        "pooling": "marker_hidden_state",
        "marker_token": spec.marker_token,
        "vectors_per_source_row": 1,
    }


def s2_pooling_contract() -> dict[str, Any]:
    return {
        "representation": "span_mean_2",
        "partition": "ordered_contiguous_content_tokens_into_two_spans",
        "pooling": "mean_hidden_state_per_nonempty_span",
        "short_input_policy": "store_nonempty_spans_only",
        "coverage": "exact_no_missing_no_duplicate_content_tokens",
    }


def split_accessed_test(split: str) -> bool:
    return split == "test"


def forward_contract_payload(
    *,
    spec: ExtractionSpec,
    args: argparse.Namespace,
    checkpoint_info: dict[str, Any],
    tokenizer_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": "mamba3_joint_s1_s2_len512_train_val",
        "input_kind": spec.input_kind,
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id,
        "tokenizer_fingerprint": tokenizer_info["tokenizer_fingerprint"],
        "tokenizer_vocab_size": int(tokenizer_info["tokenizer_vocab_size"]),
        "tokenizer_backend_sha256": tokenizer_info["tokenizer_backend_sha256"],
        "tokenizer_commit_hash": tokenizer_info["tokenizer_commit_hash"],
        "checkpoint_sha256": checkpoint_info["sha256"],
        "checkpoint_global_step": checkpoint_info.get("global_step"),
        "max_length": int(args.max_length),
        "tokenizer_content_max_length": int(args.max_length) - 1,
        "dtype": args.dtype,
        "marker": spec.marker_token,
        "marker_id": int(tokenizer_info["marker_id"]),
        "s1_pooling_contract": s1_pooling_contract(spec),
        "s2_span_partition_pooling_contract": s2_pooling_contract(),
        "one_backbone_pass_per_batch": True,
        "raw_text_stored": False,
        "token_ids_stored": False,
        "test_accessed": split_accessed_test(args.split),
    }


def forward_contract_sha256(
    *,
    spec: ExtractionSpec,
    args: argparse.Namespace,
    checkpoint_info: dict[str, Any],
    tokenizer_info: dict[str, Any],
) -> str:
    return canonical_json_sha256(
        forward_contract_payload(
            spec=spec,
            args=args,
            checkpoint_info=checkpoint_info,
            tokenizer_info=tokenizer_info,
        )
    )


def deterministic_joint_run_id(
    *,
    args: argparse.Namespace,
    checkpoint_info: dict[str, Any],
    tokenizer_info: dict[str, Any],
) -> str:
    payload = {
        "protocol": "mamba3_joint_s1_s2_len512_train_val",
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id,
        "tokenizer_run_fingerprint": tokenizer_info.get(
            "tokenizer_run_fingerprint",
            tokenizer_info["tokenizer_fingerprint"],
        ),
        "checkpoint_sha256": checkpoint_info["sha256"],
        "checkpoint_global_step": checkpoint_info.get("global_step"),
        "source_cache": str(resolve_path(args.section_cache_dir)),
        "source_bundle_manifest_sha256": getattr(args, "source_bundle_manifest_sha256", ""),
        "split": args.split,
        "limit_examples": getattr(args, "limit_examples", None),
        "limit_rows": getattr(args, "limit_rows", None),
        "max_length": int(args.max_length),
        "tokenizer_content_max_length": int(args.max_length) - 1,
        "dtype": args.dtype,
    }
    return canonical_json_sha256(payload)


def write_completion(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def temp_path_for(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = temp_path_for(path)
    if tmp.exists():
        tmp.unlink()
    try:
        with tmp.open("wb") as handle:
            np.save(handle, array)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    tmp = temp_path_for(path)
    if tmp.exists():
        tmp.unlink()
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    tmp = temp_path_for(path)
    if tmp.exists():
        tmp.unlink()
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def expected_joint_completion_fields(
    *,
    spec: ExtractionSpec,
    shard_idx: int,
    shard: pd.DataFrame,
    args: argparse.Namespace,
    checkpoint_info: dict[str, Any],
    tokenizer_info: dict[str, Any],
    audit_enabled: bool,
    s1_cache_fingerprint: str,
) -> dict[str, Any]:
    forward_sha = forward_contract_sha256(
        spec=spec,
        args=args,
        checkpoint_info=checkpoint_info,
        tokenizer_info=tokenizer_info,
    )
    source_digest = shard_source_key_digest(shard, spec)
    return {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "joint_run_id": deterministic_joint_run_id(args=args, checkpoint_info=checkpoint_info, tokenizer_info=tokenizer_info),
        "forward_contract_sha256": forward_sha,
        "input_kind": spec.input_kind,
        "input_rows": int(len(shard)),
        "shard_index": int(shard_idx),
        "source_digest_sha256": source_digest,
        "source_key_sha256": source_digest,
        "source_bundle_manifest_sha256": getattr(args, "source_bundle_manifest_sha256", ""),
        "source_kind_manifest_sha256": getattr(args, "source_kind_manifest_sha256", ""),
        "source_manifest_sha256": getattr(args, "source_kind_manifest_sha256", ""),
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id,
        "tokenizer_run_fingerprint": tokenizer_info.get("tokenizer_run_fingerprint", tokenizer_info["tokenizer_fingerprint"]),
        "tokenizer_fingerprint": tokenizer_info["tokenizer_fingerprint"],
        "tokenizer_vocab_size": int(tokenizer_info["tokenizer_vocab_size"]),
        "tokenizer_backend_sha256": tokenizer_info["tokenizer_backend_sha256"],
        "tokenizer_commit_hash": tokenizer_info["tokenizer_commit_hash"],
        "checkpoint_sha256": checkpoint_info["sha256"],
        "checkpoint_global_step": checkpoint_info.get("global_step"),
        "max_length": int(args.max_length),
        "tokenizer_content_max_length": int(args.max_length) - 1,
        "dtype": args.dtype,
        "marker": spec.marker_token,
        "marker_id": int(tokenizer_info["marker_id"]),
        "batch_size": int(args.batch_size),
        "s1_prefix": spec.prefix,
        "s1_pooling_contract": s1_pooling_contract(spec),
        "s2_prefix": s2_prefix(spec, args),
        "s2_span_partition_pooling_contract": s2_pooling_contract(),
        "audit_max_abs_tol": float(args.audit_max_abs_tol),
        "audit_min_cosine": float(args.audit_min_cosine),
        "s1_cache_fingerprint": s1_cache_fingerprint,
        "one_backbone_pass_per_batch": True,
        "s1_written": bool(getattr(args, "write_s1", False)),
        "test_accessed": split_accessed_test(args.split),
        "raw_text_stored": False,
        "token_ids_stored": False,
        "s1_audit_enabled": bool(audit_enabled),
    }


def load_completed_joint_metadata(
    *,
    s1_paths: dict[str, Path] | None,
    s2_paths: dict[str, Path],
    complete_path: Path,
    expected: dict[str, Any],
    require_audit: bool,
) -> tuple[pd.DataFrame | None, pd.DataFrame, dict[str, Any]]:
    completion = json.loads(complete_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if completion.get(key) != value:
            raise ValueError(f"Completed joint shard {complete_path} has mismatched {key}: {completion.get(key)!r} != {value!r}")

    if s1_paths is not None:
        actual_s1_embedding_sha = sha256_file(s1_paths["embeddings"])
        actual_s1_metadata_sha = sha256_file(s1_paths["metadata"])
        if completion.get("s1_embeddings_sha256") != actual_s1_embedding_sha:
            raise ValueError(f"Completed joint shard S1 embedding SHA256 mismatch: {s1_paths['embeddings']}")
        if completion.get("s1_metadata_sha256") != actual_s1_metadata_sha:
            raise ValueError(f"Completed joint shard S1 metadata SHA256 mismatch: {s1_paths['metadata']}")
        s1_metadata = pd.read_parquet(s1_paths["metadata"])
        if len(s1_metadata) != int(completion.get("s1_rows", -1)):
            raise ValueError("Completed joint shard S1 metadata row count does not match completion")
    else:
        s1_metadata = None

    actual_s2_embedding_sha = sha256_file(s2_paths["embeddings"])
    actual_s2_metadata_sha = sha256_file(s2_paths["metadata"])
    if completion.get("s2_embeddings_sha256") != actual_s2_embedding_sha:
        raise ValueError(f"Completed joint shard S2 embedding SHA256 mismatch: {s2_paths['embeddings']}")
    if completion.get("s2_metadata_sha256") != actual_s2_metadata_sha:
        raise ValueError(f"Completed joint shard S2 metadata SHA256 mismatch: {s2_paths['metadata']}")
    if require_audit and "s1_audit" not in completion:
        raise ValueError(f"Completed joint shard lacks S1 audit stats: {complete_path}")
    s2_metadata = pd.read_parquet(s2_paths["metadata"])
    if len(s2_metadata) != int(completion.get("s2_rows", -1)):
        raise ValueError("Completed joint shard S2 metadata row count does not match completion")
    return s1_metadata, s2_metadata, completion


def iter_shards(frame: pd.DataFrame, shard_size: int) -> Iterable[tuple[int, int, pd.DataFrame]]:
    for shard_idx, start in enumerate(range(0, len(frame), shard_size)):
        yield shard_idx, start, frame.iloc[start : start + shard_size].copy()


def audit_s1_vectors(
    audit_index: S1AuditIndex,
    spec: ExtractionSpec,
    shard: pd.DataFrame,
    computed_s1: np.ndarray,
    stats: S1AuditStats,
    max_abs_tol: float,
    min_cosine_tol: float,
) -> None:
    key_cols = ["example_id", spec.order_col]
    keys = shard[key_cols].copy()
    expected = audit_index.vectors_for(keys)
    stats.update(computed_s1, expected)
    if stats.max_abs_diff > max_abs_tol or stats.min_cosine < min_cosine_tol:
        raise ValueError(
            "S1 audit mismatch exceeds tolerance: "
            f"max_abs={stats.max_abs_diff:.6g} tol={max_abs_tol:.6g}; "
            f"min_cosine={stats.min_cosine:.6g} tol={min_cosine_tol:.6g}"
        )


def benchmark_batch_sizes(
    frame: pd.DataFrame,
    *,
    spec: ExtractionSpec,
    tokenizer: Any,
    model: Any,
    max_length: int,
    device: torch.device,
    batch_sizes: Sequence[int],
    output_dir: Path,
    repeats: int = 1,
    warmup_rows: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        if warmup_rows > 0 and not frame.empty:
            warmup = frame.head(int(warmup_rows)).copy()
            for batch in iter_batches(warmup, batch_size):
                extract_joint_batch(batch, spec=spec, tokenizer=tokenizer, model=model, max_length=max_length, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        status = "ok"
        rows_processed = len(frame)
        error = ""
        elapsed_values: list[float] = []
        row_rate_values: list[float] = []
        try:
            for repeat in range(max(int(repeats), 1)):
                start = time.time()
                processed_this_repeat = 0
                for batch in iter_batches(frame, batch_size):
                    extract_joint_batch(batch, spec=spec, tokenizer=tokenizer, model=model, max_length=max_length, device=device)
                    processed_this_repeat += len(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = max(time.time() - start, 1e-9)
                elapsed_values.append(elapsed)
                row_rate_values.append(processed_this_repeat / elapsed)
        except RuntimeError as exc:
            status = "oom" if "out of memory" in str(exc).lower() else "error"
            error = repr(exc)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(
            {
                "input_kind": spec.input_kind,
                "batch_size": int(batch_size),
                "status": status,
                "rows": int(rows_processed),
                "repeats": int(max(int(repeats), 1)),
                "warmup_rows": int(warmup_rows),
                "elapsed_seconds": float(np.median(elapsed_values)) if elapsed_values else "",
                "rows_per_second": float(np.median(row_rate_values)) if row_rate_values else 0.0,
                "rows_per_second_values": " ".join(f"{value:.6f}" for value in row_rate_values),
                "error": error,
            }
        )
    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / f"{spec.prefix}_batch_benchmark.csv", index=False)
    print(out.to_string(index=False), flush=True)
    return out


def iter_batches(frame: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size]


def write_summary(
    output_dir: Path,
    spec: ExtractionSpec,
    *,
    split: str,
    prefix: str,
    representation: str,
    input_rows: int,
    fresh_input_rows: int,
    resumed_input_rows: int,
    output_rows: int,
    shard_size: int,
    n_shards: int,
    batch_size: int,
    elapsed_seconds: float,
    peak_gpu_memory_mb: float | str,
) -> None:
    summary = pd.DataFrame(
        [
            {
                "prefix": prefix,
                "source_prefix": spec.prefix,
                "representation": representation,
                "n_input_rows": int(input_rows),
                "n_fresh_input_rows": int(fresh_input_rows),
                "n_resumed_input_rows": int(resumed_input_rows),
                "n_rows": int(output_rows),
                "shard_size": int(shard_size),
                "n_shards": int(n_shards),
                "batch_size": int(batch_size),
                "elapsed_seconds": elapsed_seconds,
                "rows_per_second": fresh_input_rows / max(elapsed_seconds, 1e-9) if fresh_input_rows else 0.0,
                "rows_per_second_scope": "fresh_rows_this_process",
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                "single_file_mode": False,
                "test_accessed": split_accessed_test(split),
            }
        ]
    )
    summary.to_csv(output_dir / f"{prefix}_embedding_shards_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def write_provenance(
    output_dir: Path,
    spec: ExtractionSpec,
    args: argparse.Namespace,
    checkpoint_info: dict[str, Any],
    tokenizer_info: dict[str, Any],
    s1_cache_fingerprint: str,
) -> None:
    forward_contract = forward_contract_payload(
        spec=spec,
        args=args,
        checkpoint_info=checkpoint_info,
        tokenizer_info=tokenizer_info,
    )
    provenance = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "joint_run_id": deterministic_joint_run_id(args=args, checkpoint_info=checkpoint_info, tokenizer_info=tokenizer_info),
        "forward_contract_sha256": canonical_json_sha256(forward_contract),
        "forward_contract": forward_contract,
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id,
        "tokenizer_vocab_size": int(tokenizer_info["tokenizer_vocab_size"]),
        "tokenizer_run_fingerprint": tokenizer_info.get("tokenizer_run_fingerprint", tokenizer_info["tokenizer_fingerprint"]),
        "tokenizer_fingerprint": tokenizer_info["tokenizer_fingerprint"],
        "tokenizer_backend_sha256": tokenizer_info["tokenizer_backend_sha256"],
        "tokenizer_commit_hash": tokenizer_info["tokenizer_commit_hash"],
        "clean_checkpoint": checkpoint_info,
        "source_cache": str(resolve_path(args.section_cache_dir)),
        "source_bundle_manifest_sha256": getattr(args, "source_bundle_manifest_sha256", ""),
        "source_kind_manifest_sha256": getattr(args, "source_kind_manifest_sha256", ""),
        "source_manifest_sha256": getattr(args, "source_kind_manifest_sha256", ""),
        "existing_s1_cache": str(resolve_path(args.s1_cache_dir)) if args.s1_cache_dir else "",
        "s1_cache_fingerprint": s1_cache_fingerprint,
        "input_kind": spec.input_kind,
        "s1_prefix": spec.prefix,
        "s1_representation": s1_representation(spec),
        "s2_prefix": s2_prefix(spec, args),
        "s2_representation": "span_mean_2",
        "split": args.split,
        "max_length": int(args.max_length),
        "tokenizer_content_max_length": int(args.max_length) - 1,
        "marker": spec.marker_token,
        "marker_id": int(tokenizer_info["marker_id"]),
        "dtype": args.dtype,
        "batch_size": int(args.batch_size),
        "audit_max_abs_tol": float(args.audit_max_abs_tol),
        "audit_min_cosine": float(args.audit_min_cosine),
        "one_backbone_pass_per_batch": True,
        "s1_written": bool(getattr(args, "write_s1", False)),
        "s2_written": True,
        "span_mean_4_written": False,
        "test_accessed": split_accessed_test(args.split),
        "raw_text_exported": False,
        "token_ids_exported": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{spec.prefix}_span2_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def extract_to_shards(
    frame: pd.DataFrame,
    *,
    spec: ExtractionSpec,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
    checkpoint_info: dict[str, Any],
    tokenizer_info: dict[str, Any],
    device: torch.device,
) -> None:
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    write_s1 = bool(getattr(args, "write_s1", False))
    s1_prefix_name = spec.prefix
    s1_repr = s1_representation(spec)
    s2_prefix_name = s2_prefix(spec, args)
    if write_s1 and s1_prefix_name == s2_prefix_name:
        raise ValueError("S1 and S2 prefixes must be distinct when --write-s1 is enabled")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    audit_index = None
    audit_stats = S1AuditStats()
    s1_cache_fingerprint = ""
    if args.s1_cache_dir and not args.no_s1_audit:
        s1_cache_root = resolve_path(args.s1_cache_dir)
        audit_index = S1AuditIndex(s1_cache_root, spec.prefix, ["example_id", spec.order_col])
        s1_cache_fingerprint = cache_file_manifest_fingerprint(s1_cache_root, spec.prefix)
    total_s1_rows = 0
    total_s2_rows = 0
    completed_input_rows = 0
    resumed_input_rows = 0
    fresh_input_rows = 0
    start_time = time.time()
    n_shards = int((len(frame) + args.shard_size - 1) // args.shard_size)
    progress_label = f"{spec.prefix} joint_s1_s2" if write_s1 else f"{spec.prefix} span2"
    with tqdm(total=len(frame), initial=0, unit="rows", desc=progress_label) as progress:
        for shard_idx, _, shard in iter_shards(frame, args.shard_size):
            s2_paths = shard_data_paths(output_dir, s2_prefix_name, shard_idx)
            s1_paths = shard_data_paths(output_dir, s1_prefix_name, shard_idx) if write_s1 else None
            complete_path = joint_completion_path(output_dir, spec.prefix, shard_idx)
            state = check_joint_resume_state(
                s1_paths,
                s2_paths,
                complete_path,
                bool(args.resume),
                stale_label=f"{spec.prefix}_joint_shard_{shard_idx:03d}",
            )
            joint_completion_expected = expected_joint_completion_fields(
                spec=spec,
                shard_idx=shard_idx,
                shard=shard,
                args=args,
                checkpoint_info=checkpoint_info,
                tokenizer_info=tokenizer_info,
                audit_enabled=audit_index is not None,
                s1_cache_fingerprint=s1_cache_fingerprint,
            )
            if state == "complete":
                s1_metadata, s2_metadata, completion = load_completed_joint_metadata(
                    s1_paths=s1_paths,
                    s2_paths=s2_paths,
                    complete_path=complete_path,
                    expected=joint_completion_expected,
                    require_audit=audit_index is not None,
                )
                total_s2_rows += len(s2_metadata)
                if write_s1 and s1_metadata is not None:
                    total_s1_rows += len(s1_metadata)
                completed_input_rows += len(shard)
                resumed_input_rows += len(shard)
                if audit_index is not None:
                    audit_stats.merge(S1AuditStats.from_dict(completion["s1_audit"]))
                progress.update(len(shard))
                print(f"[{spec.prefix}] skipped complete shard {shard_idx:03d}", flush=True)
                continue
            if state == "exists":
                raise FileExistsError(f"Shard already exists; use --resume or a clean output dir: {s2_paths}")
            shard_s1_parts: list[np.ndarray] = []
            shard_s2_parts: list[np.ndarray] = []
            shard_s1_meta_parts: list[pd.DataFrame] = []
            shard_s2_meta_parts: list[pd.DataFrame] = []
            s1_output_offset = total_s1_rows
            s2_output_offset = total_s2_rows
            for batch in iter_batches(shard, args.batch_size):
                s1_vectors, s2_vectors, span_rows = extract_joint_batch(
                    batch,
                    spec=spec,
                    tokenizer=tokenizer,
                    model=model,
                    max_length=args.max_length,
                    device=device,
                )
                shard_s1_parts.append(s1_vectors)
                shard_s1_meta_parts.append(
                    build_s1_metadata(
                        batch,
                        spec,
                        shard_idx,
                        s1_output_offset + sum(len(part) for part in shard_s1_meta_parts),
                    )
                )
                s2_metadata = build_span_metadata(
                    batch,
                    span_rows,
                    spec,
                    shard_idx,
                    s2_output_offset + sum(len(part) for part in shard_s2_meta_parts),
                )
                shard_s2_parts.append(s2_vectors)
                shard_s2_meta_parts.append(s2_metadata)
                progress.update(len(batch))
            shard_s1 = np.vstack(shard_s1_parts).astype(np.float32, copy=False) if shard_s1_parts else np.zeros((0, 0), dtype=np.float32)
            shard_s2 = np.vstack(shard_s2_parts).astype(np.float32, copy=False) if shard_s2_parts else np.zeros((0, 0), dtype=np.float32)
            shard_s1_metadata = pd.concat(shard_s1_meta_parts, ignore_index=True) if shard_s1_meta_parts else build_s1_metadata(shard, spec, shard_idx, total_s1_rows)
            shard_s2_metadata = pd.concat(shard_s2_meta_parts, ignore_index=True) if shard_s2_meta_parts else build_span_metadata(shard, pd.DataFrame(), spec, shard_idx, total_s2_rows)
            shard_s1_metadata["row_index_in_shard"] = np.arange(len(shard_s1_metadata), dtype=int)
            shard_s1_metadata["global_row_index"] = np.arange(total_s1_rows, total_s1_rows + len(shard_s1_metadata), dtype=int)
            shard_s1_metadata["row_index"] = shard_s1_metadata["global_row_index"]
            shard_s2_metadata["row_index_in_shard"] = np.arange(len(shard_s2_metadata), dtype=int)
            shard_s2_metadata["global_row_index"] = np.arange(total_s2_rows, total_s2_rows + len(shard_s2_metadata), dtype=int)
            shard_s2_metadata["row_index"] = shard_s2_metadata["global_row_index"]
            if len(shard_s1) != len(shard_s1_metadata):
                raise ValueError(f"S1 row count mismatch for shard {shard_idx}: embeddings={len(shard_s1)} metadata={len(shard_s1_metadata)}")
            if len(shard_s2) != len(shard_s2_metadata):
                raise ValueError(f"S2 row count mismatch for shard {shard_idx}: embeddings={len(shard_s2)} metadata={len(shard_s2_metadata)}")
            shard_audit_stats = S1AuditStats()
            if audit_index is not None:
                audit_s1_vectors(
                    audit_index,
                    spec,
                    shard,
                    shard_s1,
                    shard_audit_stats,
                    args.audit_max_abs_tol,
                    args.audit_min_cosine,
                )
                audit_stats.merge(shard_audit_stats)
            if write_s1:
                assert s1_paths is not None
                atomic_save_npy(s1_paths["embeddings"], shard_s1.astype(np.float32, copy=False))
                atomic_write_parquet(s1_paths["metadata"], shard_s1_metadata)
            atomic_save_npy(s2_paths["embeddings"], shard_s2.astype(np.float32, copy=False))
            atomic_write_parquet(s2_paths["metadata"], shard_s2_metadata)
            joint_completion = {
                **joint_completion_expected,
                "s1_rows": int(len(shard_s1_metadata)) if write_s1 else 0,
                "s2_rows": int(len(shard_s2_metadata)),
                "s1_embeddings_sha256": sha256_file(s1_paths["embeddings"]) if write_s1 and s1_paths is not None else "",
                "s1_metadata_sha256": sha256_file(s1_paths["metadata"]) if write_s1 and s1_paths is not None else "",
                "s2_embeddings_sha256": sha256_file(s2_paths["embeddings"]),
                "s2_metadata_sha256": sha256_file(s2_paths["metadata"]),
                "s1_audit": shard_audit_stats.to_dict(),
            }
            write_completion(complete_path, joint_completion)
            if write_s1:
                total_s1_rows += len(shard_s1_metadata)
            total_s2_rows += len(shard_s2_metadata)
            completed_input_rows += len(shard)
            fresh_input_rows += len(shard)
            print(
                f"[{spec.prefix}] wrote shard {shard_idx + 1}/{n_shards} "
                f"input_rows={len(shard)} s1_rows={len(shard_s1_metadata)} s2_rows={len(shard_s2_metadata)}",
                flush=True,
            )
    elapsed = time.time() - start_time
    peak_gpu = ""
    if device.type == "cuda":
        peak_gpu = torch.cuda.max_memory_allocated(device) / (1024**2)
    if write_s1:
        write_summary(
            output_dir,
            spec,
            split=args.split,
            prefix=s1_prefix_name,
            representation=s1_repr,
            input_rows=completed_input_rows,
            fresh_input_rows=fresh_input_rows,
            resumed_input_rows=resumed_input_rows,
            output_rows=total_s1_rows,
            shard_size=args.shard_size,
            n_shards=n_shards,
            batch_size=args.batch_size,
            elapsed_seconds=elapsed,
            peak_gpu_memory_mb=peak_gpu,
        )
    write_summary(
        output_dir,
        spec,
        split=args.split,
        prefix=s2_prefix_name,
        representation="span_mean_2",
        input_rows=completed_input_rows,
        fresh_input_rows=fresh_input_rows,
        resumed_input_rows=resumed_input_rows,
        output_rows=total_s2_rows,
        shard_size=args.shard_size,
        n_shards=n_shards,
        batch_size=args.batch_size,
        elapsed_seconds=elapsed,
        peak_gpu_memory_mb=peak_gpu,
    )
    write_provenance(output_dir, spec, args, checkpoint_info, tokenizer_info, s1_cache_fingerprint)
    if audit_index is not None:
        audit = pd.DataFrame([{**audit_stats.to_dict(), "prefix": spec.prefix}])
        audit.to_csv(output_dir / f"{spec.prefix}_s1_consistency_audit.csv", index=False)
        print(audit.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Mamba3 S2 span_mean_2 from one shared S1/S2 backbone pass.")
    parser.add_argument("--section-cache-dir", default="results/notesm_v2_section_cache/lampr_v0_2_full")
    parser.add_argument("--s1-cache-dir", default="results/section_embedding_cache/dapt1epoch_mamba3_siso187_section_chunk_full_len512")
    parser.add_argument("--output-dir", default="results/section_embedding_cache/dapt1epoch_mamba3_siso187_section_chunk_span2_full_len512_train_val")
    parser.add_argument("--input-kind", choices=["section", "chunk"], required=True)
    parser.add_argument("--split", default="train_val")
    parser.add_argument(
        "--allow-test-split",
        action="store_true",
        help="Allow final locked-model test embedding extraction. Test remains rejected unless this flag is set.",
    )
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    parser.add_argument("--model-checkpoint", default="checkpoints/dapt_mamba3_siso187_lampr_v0_2_len2048_1epoch_lr1e4_min2_clean/checkpoint-step-25338")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--no-s1-audit", action="store_true")
    parser.add_argument("--write-s1", action="store_true", help="Save marker-pooled S1 shards from the same forward pass as S2.")
    parser.add_argument("--s2-prefix-suffix", default="span2", help="Suffix used for S2 shards when --write-s1 is enabled.")
    parser.add_argument("--audit-max-abs-tol", type=float, default=1e-3)
    parser.add_argument("--audit-min-cosine", type=float, default=0.99999)
    parser.add_argument("--benchmark-batch-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--benchmark-repeats", type=int, default=1)
    parser.add_argument("--benchmark-warmup-rows", type=int, default=0)
    parser.add_argument("--benchmark-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_names = validate_split(args.split, allow_test=args.allow_test_split)
    if args.max_length != 512:
        raise ValueError("This protocol lock is for max_length=512 only")
    spec = extraction_spec(args.input_kind)
    source_root = resolve_path(args.section_cache_dir)
    args.source_bundle_manifest_sha256 = source_bundle_cache_file_manifest_fingerprint(
        source_root,
        args.split,
        allow_test=args.allow_test_split,
    )
    args.source_kind_manifest_sha256 = source_kind_cache_file_manifest_fingerprint(
        source_root,
        args.input_kind,
        args.split,
        allow_test=args.allow_test_split,
    )
    args.source_manifest_sha256 = args.source_kind_manifest_sha256
    frame = read_source_cache(
        source_root,
        args.input_kind,
        args.split,
        args.limit_examples,
        allow_test=args.allow_test_split,
    )
    if args.limit_rows is not None:
        frame = frame.head(int(args.limit_rows)).copy()
    print(f"input_kind={spec.input_kind} splits={','.join(split_names)} rows={len(frame)}", flush=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if args.dtype == "bf16" and device.type == "cuda" else torch.float32
    tokenizer, model, checkpoint_info = load_official_mamba3(
        args.model_id,
        args.tokenizer_id,
        resolve_path(args.model_checkpoint),
        device,
        dtype,
    )
    tokenizer_info = tokenizer_contract(tokenizer, args.tokenizer_id, spec.marker_token)
    if args.benchmark_batch_sizes:
        benchmark_batch_sizes(
            frame,
            spec=spec,
            tokenizer=tokenizer,
            model=model,
            max_length=args.max_length,
            device=device,
            batch_sizes=args.benchmark_batch_sizes,
            output_dir=resolve_path(args.output_dir),
            repeats=args.benchmark_repeats,
            warmup_rows=args.benchmark_warmup_rows,
        )
        if args.benchmark_only:
            return 0
    extract_to_shards(
        frame,
        spec=spec,
        tokenizer=tokenizer,
        model=model,
        args=args,
        checkpoint_info=checkpoint_info,
        tokenizer_info=tokenizer_info,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
