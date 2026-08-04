from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.data.lampr_dataset import balanced_downsample_indices, read_lampr_split  # noqa: E402
from notesm_v2.evaluation.metrics import binary_classification_metrics, brier_score, expected_calibration_error  # noqa: E402
from notesm_v2.models.notesm_v2 import (  # noqa: E402
    NoteSSMV2Config,
    NoteSSMV2HybridClassifier,
    NoteSSMV2HybridConfig,
    NoteSSMV2SectionAwareClassifier,
)


METRICS = ["AUPRC", "AUROC", "Accuracy", "Precision", "Recall", "F1"]
FINAL_METRICS = [*METRICS, "Brier", "ECE"]
RESTRICTED_METADATA_COLUMNS = {
    "text",
    "sequence_text",
    "merged_section_text",
    "section_text",
    "chunk_text",
    "patient_id",
    "subject_id",
    "hadm_id",
    "note_id",
}
SECTION_ORDER_BUCKET_LIMITS = [1, 2, 3, 4, 8, 16, 32]
LENGTH_BUCKET_LIMITS = [64, 128, 256, 512, 1024, 2048, 4096]
OCCURRENCE_BUCKET_LIMITS = [1, 2, 3, 4, 5, 8, 16]


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def resolve_device(device_arg: str = "auto") -> torch.device:
    normalized = str(device_arg).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested, but CUDA is not available")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"--device {device_arg} was requested, but CUDA is not available")
    if normalized == "cpu" or normalized.startswith("cuda"):
        return torch.device(normalized)
    raise ValueError(f"Unsupported --device value: {device_arg}")


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def load_benchmark_index(root: Path, splits: tuple[str, ...] = ("train", "val", "test")) -> pd.DataFrame:
    frames = [read_lampr_split(root, split) for split in splits]
    benchmark = pd.concat(frames, ignore_index=True)[["example_id", "split", "label"]].copy()
    benchmark["example_id"] = benchmark["example_id"].astype(str)
    if benchmark["example_id"].duplicated().any():
        raise ValueError("Frozen benchmark contains duplicated example_id values")
    return benchmark


def benchmark_splits_for_training(eval_splits: tuple[str, ...]) -> tuple[str, ...]:
    ordered = ["train", "val", "test"]
    needed = {"train", *eval_splits}
    return tuple(split for split in ordered if split in needed)


def load_note_embedding_cache(cache_dir: Path, cache_name: str) -> tuple[np.ndarray, pd.DataFrame]:
    embedding_path = cache_dir / f"{cache_name}.npy"
    metadata_path = cache_dir / f"{cache_name}_metadata.parquet"
    if embedding_path.exists() and metadata_path.exists():
        return np.load(embedding_path), pd.read_parquet(metadata_path)

    embedding_paths = sorted(cache_dir.rglob(f"{cache_name}_embeddings_shard_*.npy"))
    metadata_paths = sorted(cache_dir.rglob(f"{cache_name}_metadata_shard_*.parquet"))
    if not embedding_paths and not metadata_paths:
        raise FileNotFoundError(f"Missing embedding cache files for {cache_name} under {cache_dir}")
    if len(embedding_paths) != len(metadata_paths):
        raise FileNotFoundError(f"Mismatched note embedding/metadata shard counts under {cache_dir}")

    arrays = []
    frames = []
    offset = 0
    for embedding_shard, metadata_shard in zip(embedding_paths, metadata_paths, strict=True):
        array = np.load(embedding_shard)
        metadata = pd.read_parquet(metadata_shard).copy()
        if len(metadata) != array.shape[0]:
            raise ValueError(f"Shard row mismatch: {embedding_shard} vs {metadata_shard}")
        metadata["row_index"] = np.arange(offset, offset + len(metadata), dtype=int)
        arrays.append(array)
        frames.append(metadata)
        offset += len(metadata)
    embeddings = np.vstack(arrays) if arrays else np.zeros((0, 0), dtype=np.float32)
    metadata = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return embeddings, metadata


def validate_text_free_metadata(metadata: pd.DataFrame, artifact_name: str) -> None:
    restricted = RESTRICTED_METADATA_COLUMNS & set(metadata.columns)
    if restricted:
        raise ValueError(f"{artifact_name} metadata contains restricted columns: {sorted(restricted)}")


def align_embedding_metadata(metadata: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Attach authoritative benchmark label/split to embedding rows.

    Metadata may contain label/split for convenience, but those values are
    checked only for consistency. The frozen benchmark remains authoritative.
    """
    if "example_id" not in metadata.columns:
        raise ValueError("Embedding metadata must include example_id")
    meta = metadata.copy()
    meta["example_id"] = meta["example_id"].astype(str)
    label_meta = meta["label"].copy() if "label" in meta.columns else None
    split_meta = meta["split"].copy() if "split" in meta.columns else None
    meta = meta.drop(columns=[col for col in ["label", "split"] if col in meta.columns])
    aligned = meta.merge(benchmark, on="example_id", how="left", validate="many_to_one")
    if aligned["label"].isna().any() or aligned["split"].isna().any():
        missing = aligned.loc[aligned["label"].isna() | aligned["split"].isna(), "example_id"].head(5).tolist()
        raise ValueError(f"Embedding metadata contains examples absent from frozen benchmark: {missing}")
    if label_meta is not None:
        mismatch = label_meta.astype(int).to_numpy() != aligned["label"].astype(int).to_numpy()
        if mismatch.any():
            raise ValueError("Embedding metadata label values disagree with frozen benchmark labels")
    if split_meta is not None:
        mismatch = split_meta.astype(str).to_numpy() != aligned["split"].astype(str).to_numpy()
        if mismatch.any():
            raise ValueError("Embedding metadata split values disagree with frozen benchmark splits")
    aligned["label"] = aligned["label"].astype(int)
    aligned["split"] = aligned["split"].astype(str)
    return aligned


def focal_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float | None = None) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    loss = (1 - pt).pow(gamma) * bce
    if alpha is not None:
        alpha_t = torch.where(targets > 0.5, torch.tensor(alpha, device=targets.device), torch.tensor(1.0 - alpha, device=targets.device))
        loss = alpha_t * loss
    return loss.mean()


def make_head(kind: str, hidden_size: int, dropout: float) -> torch.nn.Module:
    if kind == "linear":
        return torch.nn.Linear(hidden_size, 1)
    if kind == "mlp":
        return torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, 1),
        )
    raise ValueError(f"Unknown head type: {kind}")


def build_notesm_v2_config(hidden_size: int, num_section_types: int, args: argparse.Namespace) -> NoteSSMV2Config:
    return NoteSSMV2Config(
        hidden_size=hidden_size,
        num_section_types=max(num_section_types, 2),
        dropout=args.dropout,
        head_type=getattr(args, "head", "linear"),
        mlp_hidden_size=getattr(args, "mlp_hidden_size", None),
        branch_mode=getattr(args, "notesm_v2_branch_mode", "shared"),
        use_section_type_embedding=getattr(args, "use_section_type_embedding", True),
        section_pooling=getattr(args, "section_pooling", "attention"),
        fallback_pooling=getattr(args, "fallback_pooling", "attention"),
    )


def build_notesm_v2_hybrid_config(
    section_hidden_size: int,
    note_hidden_size: int,
    num_section_types: int,
    args: argparse.Namespace,
) -> NoteSSMV2HybridConfig:
    return NoteSSMV2HybridConfig(
        hidden_size=section_hidden_size,
        note_hidden_size=note_hidden_size,
        num_section_types=max(num_section_types, 2),
        dropout=args.dropout,
        head_type=getattr(args, "head", "mlp"),
        mlp_hidden_size=getattr(args, "mlp_hidden_size", None),
        hybrid_mode=getattr(args, "hybrid_mode", "hard_routed_mlp"),
        section_pooling=getattr(args, "section_pooling", "attention"),
        fallback_pooling=getattr(args, "fallback_pooling", "attention"),
        use_section_type_embedding=getattr(args, "use_section_type_embedding", True),
        use_fallback_indicator=getattr(args, "use_fallback_indicator", True),
        set_transformer_depth=getattr(args, "set_transformer_depth", 1),
        set_transformer_heads=getattr(args, "set_transformer_heads", 4),
        set_transformer_ff_mult=getattr(args, "set_transformer_ff_mult", 2),
        residual_alpha_init=getattr(args, "residual_alpha_init", 0.1),
    )


def notesm_v2_summary_fields(args: argparse.Namespace) -> dict[str, object]:
    return {
        "notesm_v2_head_type": getattr(args, "head", "linear"),
        "notesm_v2_mlp_hidden_size": getattr(args, "mlp_hidden_size", None) or "",
        "notesm_v2_branch_mode": getattr(args, "notesm_v2_branch_mode", "shared"),
        "section_pooling": getattr(args, "section_pooling", "attention"),
        "fallback_pooling": getattr(args, "fallback_pooling", "attention"),
        "use_section_type_embedding": bool(getattr(args, "use_section_type_embedding", True)),
        "chunk_embedding_dir": getattr(args, "chunk_embedding_dir", None) or "",
    }


def notesm_v2_hybrid_summary_fields(args: argparse.Namespace) -> dict[str, object]:
    return {
        "hybrid_mode": getattr(args, "hybrid_mode", "hard_routed_mlp"),
        "notesm_v2_head_type": getattr(args, "head", "mlp"),
        "notesm_v2_mlp_hidden_size": getattr(args, "mlp_hidden_size", None) or "",
        "section_pooling": getattr(args, "section_pooling", "attention"),
        "fallback_pooling": getattr(args, "fallback_pooling", "attention"),
        "use_section_type_embedding": bool(getattr(args, "use_section_type_embedding", True)),
        "use_fallback_indicator": bool(getattr(args, "use_fallback_indicator", True)),
        "set_transformer_depth": getattr(args, "set_transformer_depth", ""),
        "set_transformer_heads": getattr(args, "set_transformer_heads", ""),
        "aux_loss_weight": getattr(args, "aux_loss_weight", 0.0),
        "note_head_checkpoint": getattr(args, "note_head_checkpoint", "") or "",
        "chunk_embedding_dir": getattr(args, "chunk_embedding_dir", None) or "",
    }


def selected_train_indices(labels: np.ndarray, splits: np.ndarray, sampling: str, seed: int) -> np.ndarray:
    train_rows = np.where(splits == "train")[0]
    if sampling == "balanced":
        local = balanced_downsample_indices(pd.Series(labels[train_rows]), seed=seed)
        return train_rows[local]
    return train_rows


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, loss_name: str, pos_weight: torch.Tensor | None, focal_alpha: float | None) -> torch.Tensor:
    if loss_name == "focal":
        return focal_loss_with_logits(logits, targets, gamma=2.0, alpha=focal_alpha)
    if loss_name == "class_weight":
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)


def parse_eval_splits(value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return ("val", "test")
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.replace(",", " ").split() if piece.strip()]
    else:
        pieces = [str(piece).strip() for piece in value if str(piece).strip()]
    allowed = {"val", "test"}
    invalid = sorted(set(pieces) - allowed)
    if invalid:
        raise ValueError(f"Unsupported evaluation splits: {invalid}")
    if not pieces:
        raise ValueError("At least one evaluation split is required")
    return tuple(dict.fromkeys(pieces))


def effective_eval_splits(args: argparse.Namespace) -> tuple[str, ...]:
    splits = parse_eval_splits(getattr(args, "eval_splits", "val,test"))
    if getattr(args, "skip_test", False):
        splits = tuple(split for split in splits if split != "test")
    if not splits:
        raise ValueError("No evaluation split remains after --skip-test")
    return splits


def bucket_values(values: np.ndarray, limits: list[int], max_bucket: int | None = None) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0).clip(lower=0).to_numpy()
    buckets = np.searchsorted(np.asarray(limits, dtype=float), numeric, side="right")
    if max_bucket is not None:
        buckets = np.clip(buckets, 0, max_bucket - 1)
    return buckets.astype(np.int64)


def section_length_values(group: pd.DataFrame) -> np.ndarray:
    if "token_length" in group.columns:
        return group["token_length"].to_numpy()
    if "char_length" in group.columns:
        return group["char_length"].to_numpy()
    return np.zeros(len(group), dtype=int)


def metadata_features_used(section_meta: pd.DataFrame, chunk_meta: pd.DataFrame | None) -> dict[str, bool]:
    return {
        "canonical_section": "canonical_section" in section_meta.columns,
        "section_order": "section_order" in section_meta.columns,
        "section_token_length": "token_length" in section_meta.columns,
        "section_char_length": "char_length" in section_meta.columns,
        "num_occurrences": "num_occurrences" in section_meta.columns,
        "used_repeated_section_merge": "used_repeated_section_merge" in section_meta.columns,
        "use_chunk_fallback": "use_chunk_fallback" in section_meta.columns,
        "chunk_order": chunk_meta is not None and "chunk_order" in chunk_meta.columns,
        "chunk_token_length": chunk_meta is not None and "token_length" in chunk_meta.columns,
        "evidence_slot": "evidence_slot" in section_meta.columns or (chunk_meta is not None and "evidence_slot" in chunk_meta.columns),
        "within_segment_position": "within_segment_position" in section_meta.columns
        or (chunk_meta is not None and "within_segment_position" in chunk_meta.columns),
    }


def compute_downstream_loss(
    out: dict[str, torch.Tensor],
    labels: torch.Tensor,
    args: argparse.Namespace,
    pos_weight: torch.Tensor | None,
    chunk_available: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    fused = compute_loss(out["logits"], labels, args.loss, pos_weight, args.focal_alpha)
    aux_weight = float(getattr(args, "aux_loss_weight", 0.0))
    components = {"loss_fused": float(fused.detach().cpu().item())}
    if aux_weight <= 0.0:
        components.update({"loss_note": 0.0, "loss_section": 0.0, "loss_chunk": 0.0})
        return fused, components
    note_loss = compute_loss(out["note_logits"], labels, args.loss, pos_weight, args.focal_alpha) if "note_logits" in out else labels.new_tensor(0.0)
    section_loss = (
        compute_loss(out["section_logits"], labels, args.loss, pos_weight, args.focal_alpha)
        if "section_logits" in out
        else labels.new_tensor(0.0)
    )
    chunk_loss = labels.new_tensor(0.0)
    if "chunk_logits" in out and chunk_available is not None:
        mask = chunk_available.bool()
        if mask.any():
            chunk_loss = compute_loss(out["chunk_logits"][mask], labels[mask], args.loss, pos_weight, args.focal_alpha)
    total = fused + aux_weight * (note_loss + section_loss + chunk_loss)
    components.update(
        {
            "loss_note": float(note_loss.detach().cpu().item()),
            "loss_section": float(section_loss.detach().cpu().item()),
            "loss_chunk": float(chunk_loss.detach().cpu().item()),
        }
    )
    return total, components


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def hybrid_forward(model: NoteSSMV2HybridClassifier, batch: dict[str, object]) -> dict[str, torch.Tensor]:
    return model(
        section_embeddings=batch["section_embeddings"],
        section_mask=batch["section_mask"],
        section_type_ids=batch["section_type_ids"],
        fallback_chunk_embeddings=batch["fallback_chunk_embeddings"],
        fallback_chunk_mask=batch["fallback_chunk_mask"],
        use_chunk_fallback=batch["use_chunk_fallback"],
        note_embedding=batch["note_embedding"],
        section_order_bucket_ids=batch.get("section_order_bucket_ids"),
        section_length_bucket_ids=batch.get("section_length_bucket_ids"),
        occurrence_bucket_ids=batch.get("occurrence_bucket_ids"),
        repeated_section_ids=batch.get("repeated_section_ids"),
        chunk_order_bucket_ids=batch.get("chunk_order_bucket_ids"),
        chunk_length_bucket_ids=batch.get("chunk_length_bucket_ids"),
    )


def train_note_embedding_head(
    embeddings: np.ndarray,
    aligned: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    device = resolve_device(getattr(args, "device", "auto"))
    labels = aligned["label"].astype(int).to_numpy()
    splits = aligned["split"].astype(str).to_numpy()
    train_indices = selected_train_indices(labels, splits, args.sampling, args.seed)
    x = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    train_dataset = TensorDataset(x[train_indices], y[train_indices])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    torch.manual_seed(args.seed)
    model = make_head(args.head, x.shape[1], args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pos = float(y[train_indices].sum().item())
    neg = float(len(train_indices) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)]) if args.loss == "class_weight" else None
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    for _ in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = compute_loss(logits, yb, args.loss, pos_weight, args.focal_alpha)
            loss.backward()
            optimizer.step()
    eval_splits = effective_eval_splits(args)
    eval_indices = np.where(np.isin(splits, eval_splits))[0]
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x[eval_indices].to(device)).squeeze(-1)).detach().cpu().numpy()
    write_eval_outputs(
        aligned.iloc[eval_indices].reset_index(drop=True),
        probs,
        labels[eval_indices],
        splits[eval_indices],
        output_dir,
        args.model,
        args.seed,
        eval_splits,
    )
    torch.save(cpu_state_dict(model), output_dir / f"{args.model}_cached_embedding_head_seed{args.seed}.pt")


def load_sharded_embeddings(root: Path, prefix: str) -> tuple[np.ndarray, pd.DataFrame] | None:
    embedding_paths = sorted(root.glob(f"{prefix}_embeddings_shard_*.npy"))
    metadata_paths = sorted(root.glob(f"{prefix}_embedding_metadata_shard_*.parquet"))
    if not embedding_paths and not metadata_paths:
        return None
    if len(embedding_paths) != len(metadata_paths):
        raise FileNotFoundError(f"Mismatched {prefix} embedding/metadata shard counts under {root}")
    arrays: list[np.ndarray] = []
    frames: list[pd.DataFrame] = []
    offset = 0
    for embedding_path, metadata_path in zip(embedding_paths, metadata_paths, strict=True):
        array = np.load(embedding_path)
        metadata = pd.read_parquet(metadata_path).copy()
        metadata["row_index"] = np.arange(offset, offset + len(metadata), dtype=int)
        arrays.append(array)
        frames.append(metadata)
        offset += len(metadata)
    return np.concatenate(arrays, axis=0), pd.concat(frames, ignore_index=True)


def filter_embedding_rows_by_examples(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    example_ids: list[str] | set[str] | pd.Series,
    artifact_name: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    if "example_id" not in metadata.columns:
        raise ValueError(f"{artifact_name} metadata must include example_id")
    meta = metadata.copy()
    meta["example_id"] = meta["example_id"].astype(str)
    if "row_index" not in meta.columns:
        meta["row_index"] = np.arange(len(meta), dtype=int)
    meta["row_index"] = meta["row_index"].astype(int)
    allowed = {str(example_id) for example_id in list(example_ids)}
    keep = meta["example_id"].isin(allowed)
    if not keep.any():
        raise ValueError(f"{artifact_name} cache has no rows matching authoritative benchmark examples")
    selected = meta.loc[keep].copy()
    row_indices = selected["row_index"].to_numpy(dtype=int)
    if len(row_indices) and (row_indices.min() < 0 or row_indices.max() >= embeddings.shape[0]):
        raise ValueError(f"{artifact_name} metadata row_index is out of bounds for embeddings")
    filtered_embeddings = embeddings[row_indices].astype(np.float32, copy=False)
    selected = selected.reset_index(drop=True)
    selected["row_index"] = np.arange(len(selected), dtype=int)
    return filtered_embeddings, selected


def load_section_embeddings(root: Path, prefix: str = "section") -> tuple[np.ndarray, pd.DataFrame]:
    embeddings_path = root / "section_embeddings.npy"
    metadata_path = root / "section_embedding_metadata.parquet"
    if prefix == "section" and embeddings_path.exists() and metadata_path.exists():
        metadata = pd.read_parquet(metadata_path)
        forbidden = {"merged_section_text", "section_text", "chunk_text", "patient_id", "subject_id", "hadm_id", "note_id"}
        if forbidden & set(metadata.columns):
            raise ValueError(f"Section embedding metadata contains restricted columns: {sorted(forbidden & set(metadata.columns))}")
        return np.load(embeddings_path), metadata
    sharded = load_sharded_embeddings(root, prefix)
    if sharded is None:
        raise FileNotFoundError(f"Missing {prefix} embedding cache files under {root}")
    embeddings, metadata = sharded
    forbidden = {"merged_section_text", "section_text", "chunk_text", "patient_id", "subject_id", "hadm_id", "note_id"}
    if forbidden & set(metadata.columns):
        raise ValueError(f"Section embedding metadata contains restricted columns: {sorted(forbidden & set(metadata.columns))}")
    return embeddings, metadata


def load_chunk_embeddings(root: Path, prefix: str = "chunk") -> tuple[np.ndarray | None, pd.DataFrame | None]:
    embeddings_path = root / "chunk_embeddings.npy"
    metadata_path = root / "chunk_embedding_metadata.parquet"
    if prefix == "chunk" and embeddings_path.exists() and metadata_path.exists():
        metadata = pd.read_parquet(metadata_path)
        forbidden = {"merged_section_text", "section_text", "chunk_text", "patient_id", "subject_id", "hadm_id", "note_id"}
        if forbidden & set(metadata.columns):
            raise ValueError(f"Chunk embedding metadata contains restricted columns: {sorted(forbidden & set(metadata.columns))}")
        return np.load(embeddings_path), metadata
    sharded = load_sharded_embeddings(root, prefix)
    if sharded is None:
        return None, None
    embeddings, metadata = sharded
    forbidden = {"merged_section_text", "section_text", "chunk_text", "patient_id", "subject_id", "hadm_id", "note_id"}
    if forbidden & set(metadata.columns):
        raise ValueError(f"Chunk embedding metadata contains restricted columns: {sorted(forbidden & set(metadata.columns))}")
    return embeddings, metadata


def combined_embedding_roots(primary: Path, extra_dirs: list[str] | tuple[str, ...] | None) -> list[Path]:
    roots = [primary]
    for value in extra_dirs or []:
        if value:
            roots.append(resolve_path(value))
    return roots


def concatenate_embedding_caches(
    parts: list[tuple[np.ndarray, pd.DataFrame]],
    artifact_name: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    if not parts:
        raise FileNotFoundError(f"No embedding cache parts found for {artifact_name}")
    arrays: list[np.ndarray] = []
    frames: list[pd.DataFrame] = []
    offset = 0
    hidden_size: int | None = None
    for array, metadata in parts:
        if array.ndim != 2:
            raise ValueError(f"{artifact_name} embeddings must be rank-2")
        if len(metadata) != array.shape[0]:
            raise ValueError(f"{artifact_name} metadata row count does not match embeddings")
        if hidden_size is None:
            hidden_size = int(array.shape[1])
        elif int(array.shape[1]) != hidden_size:
            raise ValueError(f"{artifact_name} hidden size mismatch across cache roots")
        frame = metadata.copy().reset_index(drop=True)
        frame["row_index"] = np.arange(offset, offset + len(frame), dtype=int)
        arrays.append(array)
        frames.append(frame)
        offset += len(frame)
    return np.concatenate(arrays, axis=0).astype(np.float32, copy=False), pd.concat(frames, ignore_index=True)


def load_section_embeddings_from_roots(roots: list[Path], prefix: str = "section") -> tuple[np.ndarray, pd.DataFrame]:
    return concatenate_embedding_caches(
        [load_section_embeddings(root, prefix=prefix) for root in roots],
        f"{prefix} section embedding",
    )


def load_chunk_embeddings_from_roots(roots: list[Path], prefix: str = "chunk") -> tuple[np.ndarray | None, pd.DataFrame | None]:
    parts: list[tuple[np.ndarray, pd.DataFrame]] = []
    for root in roots:
        embeddings, metadata = load_chunk_embeddings(root, prefix=prefix)
        if embeddings is not None and metadata is not None:
            parts.append((embeddings, metadata))
    if not parts:
        return None, None
    return concatenate_embedding_caches(parts, f"{prefix} chunk embedding")


def section_type_mapping(names: pd.Series) -> dict[str, int]:
    ordered = sorted(set(names.astype(str)))
    return {name: idx + 1 for idx, name in enumerate(ordered)}


def sort_metadata_group(group: pd.DataFrame, primary_order_col: str) -> pd.DataFrame:
    keys = [primary_order_col]
    for optional in ["evidence_slot", "within_segment_position", "token_start_position", "row_index"]:
        if optional in group.columns and optional not in keys:
            keys.append(optional)
    return group.sort_values(keys, kind="mergesort")


HybridRecord = dict[str, np.ndarray]


def precompute_hybrid_section_records(
    example_ids: list[str],
    section_groups: dict[str, pd.DataFrame],
    section_type_map: dict[str, int],
) -> dict[str, HybridRecord]:
    records: dict[str, HybridRecord] = {}
    for example_id in example_ids:
        group = section_groups.get(str(example_id))
        if group is None or group.empty:
            continue
        group = sort_metadata_group(group, "section_order")
        indices = group["row_index"].astype(int).to_numpy(copy=True)
        if "canonical_section" in group.columns:
            type_ids = np.array(
                [section_type_map.get(str(name), 0) for name in group["canonical_section"].astype(str).tolist()],
                dtype=np.int64,
            )
        else:
            type_ids = np.zeros(len(indices), dtype=np.int64)
        if len(type_ids) != len(indices):
            type_ids = np.zeros(len(indices), dtype=np.int64)
        records[str(example_id)] = {
            "section_indices": indices,
            "section_type_ids": type_ids,
            "section_order_bucket_ids": bucket_values(
                group["section_order"].to_numpy() if "section_order" in group.columns else np.zeros(len(group)),
                SECTION_ORDER_BUCKET_LIMITS,
                max_bucket=8,
            ),
            "section_length_bucket_ids": bucket_values(section_length_values(group), LENGTH_BUCKET_LIMITS, max_bucket=8),
            "occurrence_bucket_ids": bucket_values(
                group["num_occurrences"].to_numpy() if "num_occurrences" in group.columns else np.ones(len(group)),
                OCCURRENCE_BUCKET_LIMITS,
                max_bucket=8,
            ),
            "repeated_section_ids": (
                group["used_repeated_section_merge"].fillna(False).astype(bool).astype(np.int64).to_numpy(copy=True)
                if "used_repeated_section_merge" in group.columns
                else np.zeros(len(group), dtype=np.int64)
            ),
        }
    return records


def precompute_hybrid_chunk_records(
    example_ids: list[str],
    chunk_groups: dict[str, pd.DataFrame],
    fallback_examples: set[str],
) -> dict[str, HybridRecord]:
    records: dict[str, HybridRecord] = {}
    for example_id in example_ids:
        example_id = str(example_id)
        if example_id not in fallback_examples:
            continue
        group = chunk_groups.get(example_id)
        if group is None or group.empty:
            continue
        group = sort_metadata_group(group, "chunk_order")
        records[example_id] = {
            "chunk_indices": group["row_index"].astype(int).to_numpy(copy=True),
            "chunk_order_bucket_ids": bucket_values(
                group["chunk_order"].to_numpy() if "chunk_order" in group.columns else np.zeros(len(group)),
                SECTION_ORDER_BUCKET_LIMITS,
                max_bucket=8,
            ),
            "chunk_length_bucket_ids": bucket_values(
                group["token_length"].to_numpy() if "token_length" in group.columns else np.zeros(len(group)),
                LENGTH_BUCKET_LIMITS,
                max_bucket=8,
            ),
        }
    return records


def pack_grouped_embeddings(
    example_ids: list[str],
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    order_col: str,
    type_map: dict[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped = {str(example_id): sort_metadata_group(group, order_col) for example_id, group in metadata.groupby("example_id", sort=False)}
    hidden = embeddings.shape[1]
    max_len = max((len(grouped.get(example_id, [])) for example_id in example_ids), default=1)
    max_len = max(max_len, 1)
    values = torch.zeros((len(example_ids), max_len, hidden), dtype=torch.float32)
    mask = torch.zeros((len(example_ids), max_len), dtype=torch.long)
    type_ids = torch.zeros((len(example_ids), max_len), dtype=torch.long)
    for row, example_id in enumerate(example_ids):
        group = grouped.get(str(example_id))
        if group is None or group.empty:
            continue
        indices = group["row_index"].astype(int).to_numpy()
        count = len(indices)
        values[row, :count] = torch.tensor(embeddings[indices], dtype=torch.float32)
        mask[row, :count] = 1
        if type_map is not None and "canonical_section" in group:
            mapped = [type_map.get(str(name), 0) for name in group["canonical_section"].tolist()]
            type_ids[row, :count] = torch.tensor(mapped, dtype=torch.long)
    return values, mask, type_ids


class CachedSectionDataset(Dataset):
    def __init__(
        self,
        example_ids: list[str],
        labels: np.ndarray,
        splits: np.ndarray,
        section_groups: dict[str, pd.DataFrame],
        section_embeddings: np.ndarray,
        section_type_map: dict[str, int],
        chunk_groups: dict[str, pd.DataFrame],
        chunk_embeddings: np.ndarray | None,
        fallback_examples: set[str],
    ) -> None:
        self.example_ids = example_ids
        self.labels = labels.astype(int)
        self.splits = splits.astype(str)
        self.section_groups = section_groups
        self.section_embeddings = section_embeddings
        self.section_type_map = section_type_map
        self.chunk_groups = chunk_groups
        self.chunk_embeddings = chunk_embeddings
        self.fallback_examples = fallback_examples
        self.hidden_size = int(section_embeddings.shape[1])

    def __len__(self) -> int:
        return len(self.example_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        example_id = self.example_ids[index]
        section_group = self.section_groups.get(example_id)
        if section_group is None or section_group.empty:
            raise ValueError(f"No section embeddings found for example_id={example_id}")
        section_group = sort_metadata_group(section_group, "section_order")
        section_indices = section_group["row_index"].astype(int).to_numpy()
        section_type_ids = np.array(
            [self.section_type_map.get(str(name), 0) for name in section_group.get("canonical_section", pd.Series([], dtype=str)).astype(str).tolist()],
            dtype=np.int64,
        )
        if len(section_type_ids) != len(section_indices):
            section_type_ids = np.zeros(len(section_indices), dtype=np.int64)
        section_order_bucket_ids = bucket_values(
            section_group["section_order"].to_numpy() if "section_order" in section_group.columns else np.zeros(len(section_group)),
            SECTION_ORDER_BUCKET_LIMITS,
            max_bucket=8,
        )
        section_length_bucket_ids = bucket_values(section_length_values(section_group), LENGTH_BUCKET_LIMITS, max_bucket=8)
        occurrence_bucket_ids = bucket_values(
            section_group["num_occurrences"].to_numpy() if "num_occurrences" in section_group.columns else np.ones(len(section_group)),
            OCCURRENCE_BUCKET_LIMITS,
            max_bucket=8,
        )
        repeated_section_ids = (
            section_group["used_repeated_section_merge"].fillna(False).astype(bool).astype(np.int64).to_numpy()
            if "used_repeated_section_merge" in section_group.columns
            else np.zeros(len(section_group), dtype=np.int64)
        )
        chunk_array = np.zeros((0, self.hidden_size), dtype=np.float32)
        chunk_order_bucket_ids = np.zeros(0, dtype=np.int64)
        chunk_length_bucket_ids = np.zeros(0, dtype=np.int64)
        use_fallback = example_id in self.fallback_examples
        if use_fallback:
            chunk_group = self.chunk_groups.get(example_id)
            if chunk_group is None or chunk_group.empty or self.chunk_embeddings is None:
                raise ValueError(f"Fallback requested but chunk embeddings are missing for example_id={example_id}")
            chunk_group = sort_metadata_group(chunk_group, "chunk_order")
            chunk_array = self.chunk_embeddings[chunk_group["row_index"].astype(int).to_numpy()].astype(np.float32, copy=False)
            chunk_order_bucket_ids = bucket_values(
                chunk_group["chunk_order"].to_numpy() if "chunk_order" in chunk_group.columns else np.zeros(len(chunk_group)),
                SECTION_ORDER_BUCKET_LIMITS,
                max_bucket=8,
            )
            chunk_length_bucket_ids = bucket_values(
                chunk_group["token_length"].to_numpy() if "token_length" in chunk_group.columns else np.zeros(len(chunk_group)),
                LENGTH_BUCKET_LIMITS,
                max_bucket=8,
            )
        return {
            "example_id": example_id,
            "split": self.splits[index],
            "label": float(self.labels[index]),
            "section_embeddings": self.section_embeddings[section_indices].astype(np.float32, copy=False),
            "section_type_ids": section_type_ids,
            "section_order_bucket_ids": section_order_bucket_ids,
            "section_length_bucket_ids": section_length_bucket_ids,
            "occurrence_bucket_ids": occurrence_bucket_ids,
            "repeated_section_ids": repeated_section_ids,
            "chunk_embeddings": chunk_array,
            "chunk_order_bucket_ids": chunk_order_bucket_ids,
            "chunk_length_bucket_ids": chunk_length_bucket_ids,
            "use_fallback": use_fallback,
        }


def collate_section_batch(items: list[dict[str, object]]) -> dict[str, object]:
    batch = len(items)
    hidden_size = int(items[0]["section_embeddings"].shape[1])
    max_sections = max(int(item["section_embeddings"].shape[0]) for item in items)
    max_chunks = max(max(int(item["chunk_embeddings"].shape[0]), 1) for item in items)
    sections = torch.zeros((batch, max_sections, hidden_size), dtype=torch.float32)
    section_mask = torch.zeros((batch, max_sections), dtype=torch.long)
    section_type_ids = torch.zeros((batch, max_sections), dtype=torch.long)
    section_order_bucket_ids = torch.zeros((batch, max_sections), dtype=torch.long)
    section_length_bucket_ids = torch.zeros((batch, max_sections), dtype=torch.long)
    occurrence_bucket_ids = torch.zeros((batch, max_sections), dtype=torch.long)
    repeated_section_ids = torch.zeros((batch, max_sections), dtype=torch.long)
    chunks = torch.zeros((batch, max_chunks, hidden_size), dtype=torch.float32)
    chunk_mask = torch.zeros((batch, max_chunks), dtype=torch.long)
    chunk_order_bucket_ids = torch.zeros((batch, max_chunks), dtype=torch.long)
    chunk_length_bucket_ids = torch.zeros((batch, max_chunks), dtype=torch.long)
    labels = torch.zeros(batch, dtype=torch.float32)
    use_fallback = torch.zeros(batch, dtype=torch.long)
    example_ids: list[str] = []
    splits: list[str] = []
    for row, item in enumerate(items):
        sec = torch.tensor(item["section_embeddings"], dtype=torch.float32)
        sec_types = torch.tensor(item["section_type_ids"], dtype=torch.long)
        sections[row, : sec.shape[0]] = sec
        section_mask[row, : sec.shape[0]] = 1
        section_type_ids[row, : sec_types.shape[0]] = sec_types
        for key, target in [
            ("section_order_bucket_ids", section_order_bucket_ids),
            ("section_length_bucket_ids", section_length_bucket_ids),
            ("occurrence_bucket_ids", occurrence_bucket_ids),
            ("repeated_section_ids", repeated_section_ids),
        ]:
            values = torch.tensor(item.get(key, np.zeros(sec.shape[0], dtype=np.int64)), dtype=torch.long)
            target[row, : values.shape[0]] = values
        chunk = torch.tensor(item["chunk_embeddings"], dtype=torch.float32)
        if chunk.numel():
            chunks[row, : chunk.shape[0]] = chunk
            chunk_mask[row, : chunk.shape[0]] = 1
            for key, target in [
                ("chunk_order_bucket_ids", chunk_order_bucket_ids),
                ("chunk_length_bucket_ids", chunk_length_bucket_ids),
            ]:
                values = torch.tensor(item.get(key, np.zeros(chunk.shape[0], dtype=np.int64)), dtype=torch.long)
                target[row, : values.shape[0]] = values
        labels[row] = float(item["label"])
        use_fallback[row] = int(bool(item["use_fallback"]))
        example_ids.append(str(item["example_id"]))
        splits.append(str(item["split"]))
    return {
        "example_ids": example_ids,
        "splits": splits,
        "section_embeddings": sections,
        "section_mask": section_mask,
        "section_type_ids": section_type_ids,
        "section_order_bucket_ids": section_order_bucket_ids,
        "section_length_bucket_ids": section_length_bucket_ids,
        "occurrence_bucket_ids": occurrence_bucket_ids,
        "repeated_section_ids": repeated_section_ids,
        "fallback_chunk_embeddings": chunks,
        "fallback_chunk_mask": chunk_mask,
        "chunk_order_bucket_ids": chunk_order_bucket_ids,
        "chunk_length_bucket_ids": chunk_length_bucket_ids,
        "use_chunk_fallback": use_fallback,
        "section_available": section_mask.any(dim=1).long(),
        "chunk_available": chunk_mask.any(dim=1).long(),
        "labels": labels,
    }


class CachedHybridDataset(Dataset):
    def __init__(
        self,
        example_ids: list[str],
        labels: np.ndarray,
        splits: np.ndarray,
        section_groups: dict[str, pd.DataFrame],
        section_embeddings: np.ndarray,
        section_type_map: dict[str, int],
        chunk_groups: dict[str, pd.DataFrame],
        chunk_embeddings: np.ndarray | None,
        fallback_examples: set[str],
        note_embeddings: np.ndarray,
        note_row_by_example: dict[str, int],
        section_records: dict[str, HybridRecord] | None = None,
        chunk_records: dict[str, HybridRecord] | None = None,
    ) -> None:
        self.example_ids = example_ids
        self.labels = labels.astype(int)
        self.splits = splits.astype(str)
        self.section_groups = section_groups
        self.section_embeddings = section_embeddings
        self.section_type_map = section_type_map
        self.chunk_groups = chunk_groups
        self.chunk_embeddings = chunk_embeddings
        self.fallback_examples = fallback_examples
        self.note_embeddings = note_embeddings
        self.note_row_by_example = note_row_by_example
        self.section_hidden_size = int(section_embeddings.shape[1])
        self.note_hidden_size = int(note_embeddings.shape[1])
        self.section_records = (
            section_records
            if section_records is not None
            else precompute_hybrid_section_records(self.example_ids, self.section_groups, self.section_type_map)
        )
        self.chunk_records = (
            chunk_records
            if chunk_records is not None
            else precompute_hybrid_chunk_records(self.example_ids, self.chunk_groups, self.fallback_examples)
        )

    def __len__(self) -> int:
        return len(self.example_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        example_id = self.example_ids[index]
        section_record = self.section_records.get(example_id)
        if section_record is None:
            raise ValueError(f"No section embeddings found for example_id={example_id}")
        if example_id not in self.note_row_by_example:
            raise ValueError(f"No note embedding found for example_id={example_id}")
        section_indices = section_record["section_indices"]
        chunk_array = np.zeros((0, self.section_hidden_size), dtype=np.float32)
        chunk_order_bucket_ids = np.zeros(0, dtype=np.int64)
        chunk_length_bucket_ids = np.zeros(0, dtype=np.int64)
        use_fallback = example_id in self.fallback_examples
        if use_fallback:
            chunk_record = self.chunk_records.get(example_id)
            if chunk_record is None or self.chunk_embeddings is None:
                raise ValueError(f"Fallback requested but chunk embeddings are missing for example_id={example_id}")
            chunk_array = self.chunk_embeddings[chunk_record["chunk_indices"]].astype(np.float32, copy=False)
            chunk_order_bucket_ids = chunk_record["chunk_order_bucket_ids"]
            chunk_length_bucket_ids = chunk_record["chunk_length_bucket_ids"]
        return {
            "example_id": example_id,
            "split": self.splits[index],
            "label": float(self.labels[index]),
            "section_embeddings": self.section_embeddings[section_indices].astype(np.float32, copy=False),
            "section_type_ids": section_record["section_type_ids"],
            "section_order_bucket_ids": section_record["section_order_bucket_ids"],
            "section_length_bucket_ids": section_record["section_length_bucket_ids"],
            "occurrence_bucket_ids": section_record["occurrence_bucket_ids"],
            "repeated_section_ids": section_record["repeated_section_ids"],
            "chunk_embeddings": chunk_array,
            "chunk_order_bucket_ids": chunk_order_bucket_ids,
            "chunk_length_bucket_ids": chunk_length_bucket_ids,
            "use_fallback": use_fallback,
            "note_embedding": self.note_embeddings[self.note_row_by_example[example_id]].astype(np.float32, copy=False),
        }


def collate_hybrid_batch(items: list[dict[str, object]]) -> dict[str, object]:
    batch = collate_section_batch(items)
    note_hidden_size = int(items[0]["note_embedding"].shape[0])
    note_embeddings = torch.zeros((len(items), note_hidden_size), dtype=torch.float32)
    for row, item in enumerate(items):
        note_embeddings[row] = torch.tensor(item["note_embedding"], dtype=torch.float32)
    batch["note_embedding"] = note_embeddings
    return batch


def subset_dataset(dataset: CachedSectionDataset, indices: np.ndarray) -> CachedSectionDataset:
    return CachedSectionDataset(
        [dataset.example_ids[int(index)] for index in indices],
        dataset.labels[indices],
        dataset.splits[indices],
        dataset.section_groups,
        dataset.section_embeddings,
        dataset.section_type_map,
        dataset.chunk_groups,
        dataset.chunk_embeddings,
        dataset.fallback_examples,
    )


def subset_hybrid_dataset(dataset: CachedHybridDataset, indices: np.ndarray) -> CachedHybridDataset:
    return CachedHybridDataset(
        [dataset.example_ids[int(index)] for index in indices],
        dataset.labels[indices],
        dataset.splits[indices],
        dataset.section_groups,
        dataset.section_embeddings,
        dataset.section_type_map,
        dataset.chunk_groups,
        dataset.chunk_embeddings,
        dataset.fallback_examples,
        dataset.note_embeddings,
        dataset.note_row_by_example,
        section_records=dataset.section_records,
        chunk_records=dataset.chunk_records,
    )


def train_section_embedding_head(
    section_cache_dir: Path,
    benchmark: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    device = resolve_device(getattr(args, "device", "auto"))
    section_prefix = getattr(args, "section_embedding_prefix", "section")
    chunk_prefix = getattr(args, "chunk_embedding_prefix", "chunk")
    section_embeddings, section_meta = load_section_embeddings(section_cache_dir, prefix=section_prefix)
    section_meta["example_id"] = section_meta["example_id"].astype(str)
    section_meta["row_index"] = section_meta["row_index"].astype(int)
    authoritative_example_ids = benchmark["example_id"].astype(str).drop_duplicates().tolist()
    section_embeddings, section_meta = filter_embedding_rows_by_examples(
        section_embeddings,
        section_meta,
        authoritative_example_ids,
        "Section embedding",
    )
    example_ids = section_meta["example_id"].drop_duplicates().tolist()
    aligned = align_embedding_metadata(pd.DataFrame({"example_id": example_ids}), benchmark)
    labels = aligned["label"].astype(int).to_numpy()
    splits = aligned["split"].astype(str).to_numpy()
    type_map = section_type_mapping(section_meta["canonical_section"]) if "canonical_section" in section_meta.columns else {}
    section_groups = {str(example_id): group.copy() for example_id, group in section_meta.groupby("example_id", sort=False)}
    chunk_cache_dir = resolve_path(args.chunk_embedding_dir) if getattr(args, "chunk_embedding_dir", None) else section_cache_dir
    chunk_embeddings, chunk_meta = load_chunk_embeddings(chunk_cache_dir, prefix=chunk_prefix)
    chunk_groups: dict[str, pd.DataFrame] = {}
    if chunk_embeddings is not None and chunk_meta is not None:
        chunk_meta["example_id"] = chunk_meta["example_id"].astype(str)
        chunk_meta["row_index"] = chunk_meta["row_index"].astype(int)
        chunk_embeddings, chunk_meta = filter_embedding_rows_by_examples(
            chunk_embeddings,
            chunk_meta,
            authoritative_example_ids,
            "Chunk embedding",
        )
        chunk_groups = {str(example_id): group.copy() for example_id, group in chunk_meta.groupby("example_id", sort=False)}
    fallback_examples = set()
    if "use_chunk_fallback" in section_meta.columns:
        fallback_examples = set(section_meta.loc[section_meta["use_chunk_fallback"].astype(bool), "example_id"].astype(str))
    if fallback_examples and (chunk_embeddings is None or not chunk_groups):
        raise ValueError("Fallback examples are present, but chunk embedding cache is missing")
    missing_chunks = sorted(example_id for example_id in fallback_examples if example_id not in chunk_groups)
    if missing_chunks:
        raise ValueError(f"Fallback examples are missing chunk embeddings: {missing_chunks[:5]}")
    train_indices = selected_train_indices(labels, splits, args.sampling, args.seed)
    full_dataset = CachedSectionDataset(
        example_ids=example_ids,
        labels=labels,
        splits=splits,
        section_groups=section_groups,
        section_embeddings=section_embeddings,
        section_type_map=type_map,
        chunk_groups=chunk_groups,
        chunk_embeddings=chunk_embeddings,
        fallback_examples=fallback_examples,
    )
    train_dataset = subset_dataset(full_dataset, train_indices)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_section_batch)
    torch.manual_seed(args.seed)
    model = NoteSSMV2SectionAwareClassifier(
        build_notesm_v2_config(
            hidden_size=section_embeddings.shape[-1],
            num_section_types=len(type_map) + 1,
            args=args,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    y_all = torch.tensor(labels, dtype=torch.float32)
    pos = float(y_all[train_indices].sum().item())
    neg = float(len(train_indices) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)]) if args.loss == "class_weight" else None
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    for _ in range(args.epochs):
        model.train()
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            out = model(
                section_embeddings=batch["section_embeddings"],
                section_mask=batch["section_mask"],
                section_type_ids=batch["section_type_ids"],
                fallback_chunk_embeddings=batch["fallback_chunk_embeddings"],
                fallback_chunk_mask=batch["fallback_chunk_mask"],
                use_chunk_fallback=batch["use_chunk_fallback"],
            )
            loss = compute_loss(out["logits"], batch["labels"], args.loss, pos_weight, args.focal_alpha)
            loss.backward()
            optimizer.step()
    model.eval()
    eval_splits = effective_eval_splits(args)
    eval_indices = np.where(np.isin(splits, eval_splits))[0]
    eval_dataset = subset_dataset(full_dataset, eval_indices)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_section_batch)
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    splits_list: list[str] = []
    example_id_list: list[str] = []
    with torch.no_grad():
        for batch in eval_loader:
            batch = move_batch_to_device(batch, device)
            logits = model(
                section_embeddings=batch["section_embeddings"],
                section_mask=batch["section_mask"],
                section_type_ids=batch["section_type_ids"],
                fallback_chunk_embeddings=batch["fallback_chunk_embeddings"],
                fallback_chunk_mask=batch["fallback_chunk_mask"],
                use_chunk_fallback=batch["use_chunk_fallback"],
            )["logits"]
            probs_list.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels_list.append(batch["labels"].detach().cpu().numpy().astype(int))
            splits_list.extend(batch["splits"])
            example_id_list.extend(batch["example_ids"])
    probs = np.concatenate(probs_list) if probs_list else np.zeros(0, dtype=float)
    eval_labels = np.concatenate(labels_list) if labels_list else np.zeros(0, dtype=int)
    eval_aligned = pd.DataFrame({"example_id": example_id_list, "split": splits_list, "label": eval_labels})
    write_eval_outputs(eval_aligned, probs, eval_labels, np.array(splits_list), output_dir, args.model, args.seed, eval_splits)
    torch.save(cpu_state_dict(model), output_dir / f"{args.model}_cached_section_head_seed{args.seed}.pt")


def train_hybrid_embedding_head(
    section_cache_dir: Path,
    note_cache_dir: Path,
    cache_name: str,
    benchmark: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    started_at = time.time()
    def log_stage(message: str) -> None:
        elapsed = time.time() - started_at
        print(f"[hybrid] {message} elapsed={elapsed:.1f}s", flush=True)

    device = resolve_device(getattr(args, "device", "auto"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    section_prefix = getattr(args, "section_embedding_prefix", "section")
    chunk_prefix = getattr(args, "chunk_embedding_prefix", "chunk")
    log_stage(f"loading section roots prefix={section_prefix}")
    section_roots = combined_embedding_roots(section_cache_dir, getattr(args, "section_embedding_extra_dir", []))
    section_embeddings, section_meta = load_section_embeddings_from_roots(section_roots, prefix=section_prefix)
    log_stage(f"loaded section embeddings rows={section_embeddings.shape[0]}")
    validate_text_free_metadata(section_meta, "Section embedding")
    if "row_index" not in section_meta.columns:
        section_meta = section_meta.copy()
        section_meta["row_index"] = np.arange(len(section_meta), dtype=int)
    section_meta["example_id"] = section_meta["example_id"].astype(str)
    section_meta["row_index"] = section_meta["row_index"].astype(int)
    authoritative_example_ids = benchmark["example_id"].astype(str).drop_duplicates().tolist()
    section_embeddings, section_meta = filter_embedding_rows_by_examples(
        section_embeddings,
        section_meta,
        authoritative_example_ids,
        "Section embedding",
    )
    section_example_ids = section_meta["example_id"].drop_duplicates().tolist()
    section_aligned = align_embedding_metadata(pd.DataFrame({"example_id": section_example_ids}), benchmark)

    log_stage("loading note embeddings")
    note_embeddings, note_meta = load_note_embedding_cache(note_cache_dir, cache_name)
    log_stage(f"loaded note embeddings rows={note_embeddings.shape[0]}")
    validate_text_free_metadata(note_meta, "Note embedding")
    if "row_index" not in note_meta.columns:
        note_meta = note_meta.copy()
        note_meta["row_index"] = np.arange(len(note_meta), dtype=int)
    note_meta["example_id"] = note_meta["example_id"].astype(str)
    note_meta["row_index"] = note_meta["row_index"].astype(int)
    note_embeddings, note_meta = filter_embedding_rows_by_examples(
        note_embeddings,
        note_meta,
        authoritative_example_ids,
        "Note embedding",
    )
    if note_meta["example_id"].duplicated().any():
        duplicates = note_meta.loc[note_meta["example_id"].duplicated(), "example_id"].head(5).tolist()
        raise ValueError(f"Note embedding metadata contains duplicated example_id values: {duplicates}")
    note_aligned = align_embedding_metadata(note_meta, benchmark)
    section_set = set(section_example_ids)
    note_set = set(note_aligned["example_id"].astype(str))
    missing_note = sorted(section_set - note_set)
    missing_section = sorted(note_set - section_set)
    if missing_note or missing_section:
        raise ValueError(
            "Hybrid modality example_id mismatch: "
            f"missing_note={missing_note[:5]} missing_section={missing_section[:5]}"
        )
    section_keyed = section_aligned.set_index("example_id").loc[section_example_ids].reset_index()
    note_keyed = note_aligned.set_index("example_id").loc[section_example_ids].reset_index()
    if not (section_keyed["label"].astype(int).to_numpy() == note_keyed["label"].astype(int).to_numpy()).all():
        raise ValueError("Hybrid section/note label alignment mismatch")
    if not (section_keyed["split"].astype(str).to_numpy() == note_keyed["split"].astype(str).to_numpy()).all():
        raise ValueError("Hybrid section/note split alignment mismatch")
    if len(note_aligned) != note_embeddings.shape[0]:
        raise ValueError("Note embedding row count does not match metadata row count")
    if section_embeddings.shape[0] != len(section_meta):
        raise ValueError("Section embedding row count does not match metadata row count")

    type_map = section_type_mapping(section_meta["canonical_section"]) if "canonical_section" in section_meta.columns else {}
    log_stage("building section groups")
    section_groups = {str(example_id): group.copy() for example_id, group in section_meta.groupby("example_id", sort=False)}
    chunk_cache_dir = resolve_path(args.chunk_embedding_dir) if getattr(args, "chunk_embedding_dir", None) else section_cache_dir
    chunk_roots = combined_embedding_roots(chunk_cache_dir, getattr(args, "chunk_embedding_extra_dir", []))
    log_stage(f"loading chunk roots prefix={chunk_prefix}")
    chunk_embeddings, chunk_meta = load_chunk_embeddings_from_roots(chunk_roots, prefix=chunk_prefix)
    chunk_groups: dict[str, pd.DataFrame] = {}
    if chunk_embeddings is not None and chunk_meta is not None:
        log_stage(f"loaded chunk embeddings rows={chunk_embeddings.shape[0]}")
        validate_text_free_metadata(chunk_meta, "Chunk embedding")
        if "row_index" not in chunk_meta.columns:
            chunk_meta = chunk_meta.copy()
            chunk_meta["row_index"] = np.arange(len(chunk_meta), dtype=int)
        chunk_meta["example_id"] = chunk_meta["example_id"].astype(str)
        chunk_meta["row_index"] = chunk_meta["row_index"].astype(int)
        chunk_embeddings, chunk_meta = filter_embedding_rows_by_examples(
            chunk_embeddings,
            chunk_meta,
            authoritative_example_ids,
            "Chunk embedding",
        )
        log_stage("building chunk groups")
        chunk_groups = {str(example_id): group.copy() for example_id, group in chunk_meta.groupby("example_id", sort=False)}
    metadata_used = metadata_features_used(section_meta, chunk_meta)
    pd.DataFrame([metadata_used]).to_csv(output_dir / f"{args.model}_metadata_features_used_seed{args.seed}.csv", index=False)
    fallback_examples: set[str] = set()
    if "use_chunk_fallback" in section_meta.columns:
        fallback_flags = section_meta.groupby("example_id")["use_chunk_fallback"].any()
        fallback_examples = set(fallback_flags[fallback_flags.astype(bool)].index.astype(str))
    if fallback_examples and (chunk_embeddings is None or not chunk_groups):
        raise ValueError("Fallback examples are present, but chunk embedding cache is missing")
    missing_chunks = sorted(example_id for example_id in fallback_examples if example_id not in chunk_groups)
    if missing_chunks:
        raise ValueError(f"Fallback examples are missing chunk embeddings: {missing_chunks[:5]}")
    note_row_by_example = {
        str(row.example_id): int(row.row_index)
        for row in note_keyed[["example_id", "row_index"]].itertuples(index=False)
    }
    labels = section_keyed["label"].astype(int).to_numpy()
    splits = section_keyed["split"].astype(str).to_numpy()
    train_indices = selected_train_indices(labels, splits, args.sampling, args.seed)
    log_stage("constructing cached hybrid dataset")
    full_dataset = CachedHybridDataset(
        example_ids=section_example_ids,
        labels=labels,
        splits=splits,
        section_groups=section_groups,
        section_embeddings=section_embeddings,
        section_type_map=type_map,
        chunk_groups=chunk_groups,
        chunk_embeddings=chunk_embeddings,
        fallback_examples=fallback_examples,
        note_embeddings=note_embeddings,
        note_row_by_example=note_row_by_example,
    )
    train_dataset = subset_hybrid_dataset(full_dataset, train_indices)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_hybrid_batch)
    torch.manual_seed(args.seed)
    model = NoteSSMV2HybridClassifier(
        build_notesm_v2_hybrid_config(
            section_hidden_size=section_embeddings.shape[-1],
            note_hidden_size=note_embeddings.shape[-1],
            num_section_types=len(type_map) + 1,
            args=args,
        )
    )
    note_head_initialized = False
    note_head_checkpoint = getattr(args, "note_head_checkpoint", None)
    if note_head_checkpoint:
        checkpoint_path = resolve_path(note_head_checkpoint)
        model.load_note_head_checkpoint(checkpoint_path)
        note_head_initialized = True
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    y_all = torch.tensor(labels, dtype=torch.float32)
    pos = float(y_all[train_indices].sum().item())
    neg = float(len(train_indices) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)]) if args.loss == "class_weight" else None
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    component_rows: list[dict[str, float | int]] = []
    log_stage(f"starting training epochs={args.epochs} train_examples={len(train_dataset)}")
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            out = hybrid_forward(model, batch)
            loss, components = compute_downstream_loss(
                out,
                batch["labels"],
                args,
                pos_weight,
                chunk_available=batch.get("chunk_available"),
            )
            loss.backward()
            if getattr(args, "require_residual_grad_check", False) and getattr(model, "residual_head", None) is not None:
                if not any(parameter.grad is not None and parameter.grad.detach().abs().sum().item() > 0 for parameter in model.residual_head.parameters()):
                    raise ValueError("Residual head parameters did not receive nonzero gradients on the first optimizer step")
                args.require_residual_grad_check = False
            optimizer.step()
            component_rows.append({"step": len(component_rows) + 1, **components, "loss_total": float(loss.detach().cpu().item())})
        log_stage(f"finished epoch {epoch + 1}/{args.epochs} steps={len(component_rows)}")
    model.eval()
    eval_splits = effective_eval_splits(args)
    eval_indices = np.where(np.isin(splits, eval_splits))[0]
    eval_dataset = subset_hybrid_dataset(full_dataset, eval_indices)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_hybrid_batch)
    log_stage(f"starting evaluation examples={len(eval_dataset)} splits={','.join(eval_splits)}")
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    splits_list: list[str] = []
    example_id_list: list[str] = []
    with torch.no_grad():
        for batch in eval_loader:
            batch = move_batch_to_device(batch, device)
            logits = hybrid_forward(model, batch)["logits"]
            probs_list.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels_list.append(batch["labels"].detach().cpu().numpy().astype(int))
            splits_list.extend(batch["splits"])
            example_id_list.extend(batch["example_ids"])
    probs = np.concatenate(probs_list) if probs_list else np.zeros(0, dtype=float)
    eval_labels = np.concatenate(labels_list) if labels_list else np.zeros(0, dtype=int)
    eval_aligned = pd.DataFrame({"example_id": example_id_list, "split": splits_list, "label": eval_labels})
    write_eval_outputs(eval_aligned, probs, eval_labels, np.array(splits_list), output_dir, args.model, args.seed, eval_splits)
    torch.save(cpu_state_dict(model), output_dir / f"{args.model}_cached_hybrid_head_seed{args.seed}.pt")
    if component_rows:
        pd.DataFrame(component_rows).to_csv(output_dir / f"{args.model}_loss_components_seed{args.seed}.csv", index=False)
    peak_gpu_memory_mb = ""
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    log_stage("finished hybrid downstream")
    pd.DataFrame(
        [
            {
                "hybrid_mode": args.hybrid_mode,
                "seed": args.seed,
                "evaluation_splits": ",".join(eval_splits),
                "test_evaluation_launched": "test" in eval_splits,
                "architecture_screening_only": bool(getattr(args, "architecture_screening_only", False)),
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "trainable_parameters": trainable_parameter_count(model),
                "initial_residual_alpha": getattr(args, "residual_alpha_init", ""),
                "final_residual_alpha": float(model.residual_alpha.detach().cpu().item()) if getattr(model, "residual_alpha", None) is not None else "",
                "note_head_initialized_from_checkpoint": note_head_initialized,
                "note_head_checkpoint": str(note_head_checkpoint or ""),
                "runtime_seconds": time.time() - started_at,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
            }
        ]
    ).to_csv(output_dir / f"{args.model}_v2_1_training_summary_seed{args.seed}.csv", index=False)


def write_eval_outputs(
    aligned: pd.DataFrame,
    probs: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    output_dir: Path,
    model_name: str,
    seed: int,
    evaluation_splits: tuple[str, ...] = ("val", "test"),
) -> None:
    preds = (probs >= 0.5).astype(int)
    prediction_rows = pd.DataFrame(
        {
            "example_id": aligned["example_id"].astype(str).to_numpy(),
            "split": splits,
            "y_true": labels.astype(int),
            "y_prob": probs.astype(float),
            "y_pred": preds.astype(int),
        }
    )
    prediction_rows[prediction_rows["split"].isin(evaluation_splits)].to_csv(
        output_dir / f"{model_name}_cached_predictions_seed{seed}.csv",
        index=False,
    )
    rows = []
    for split_name in evaluation_splits:
        mask = splits == split_name
        if not np.any(mask):
            continue
        metrics = binary_classification_metrics(labels[mask], probs[mask])
        metrics["Brier"] = brier_score(labels[mask], probs[mask])
        metrics["ECE"] = expected_calibration_error(labels[mask], probs[mask])
        metrics["split"] = split_name
        rows.append(metrics)
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / f"{model_name}_cached_embedding_metrics_seed{seed}.csv", index=False)
    print(results.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Downstream LAMP-R training entrypoint.")
    parser.add_argument("--benchmark-root", default="results/lampr_full_v0_2")
    parser.add_argument("--model", choices=["notesm_v1", "notesm_v2", "notesm_v2_hybrid"], default="notesm_v1")
    parser.add_argument("--loss", choices=["bce", "class_weight", "focal"], default="class_weight")
    parser.add_argument("--sampling", choices=["balanced", "natural"], default="balanced")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", default="results/downstream")
    parser.add_argument("--embedding-cache-dir", default=None)
    parser.add_argument("--section-embedding-dir", default=None)
    parser.add_argument("--chunk-embedding-dir", default=None, help="Optional chunk embedding cache dir; defaults to --section-embedding-dir.")
    parser.add_argument(
        "--section-embedding-extra-dir",
        action="append",
        default=[],
        help="Additional section embedding cache dir(s) concatenated after --section-embedding-dir.",
    )
    parser.add_argument(
        "--chunk-embedding-extra-dir",
        action="append",
        default=[],
        help="Additional chunk embedding cache dir(s) concatenated after --chunk-embedding-dir.",
    )
    parser.add_argument("--section-embedding-prefix", default="section", help="Sharded section embedding prefix to load.")
    parser.add_argument("--chunk-embedding-prefix", default="chunk", help="Sharded chunk embedding prefix to load.")
    parser.add_argument("--cache-name", default="debug")
    parser.add_argument("--head", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--mlp-hidden-size", type=int, default=None)
    parser.add_argument(
        "--notesm-v2-branch-mode",
        choices=["shared", "branch_indicator", "branch_specific_bias", "branch_specific_heads"],
        default="shared",
    )
    parser.add_argument("--section-pooling", choices=["attention", "mean", "max"], default="attention")
    parser.add_argument("--fallback-pooling", choices=["attention", "mean", "max"], default="attention")
    parser.add_argument("--use-section-type-embedding", type=parse_bool, default=True)
    parser.add_argument(
        "--hybrid-mode",
        choices=[
            "hard_routed_mlp",
            "concat_mlp",
            "learned_gate_mlp",
            "branch_specific_heads_hybrid",
            "three_view_concat",
            "three_view_residual",
            "set_transformer_residual",
        ],
        default="hard_routed_mlp",
    )
    parser.add_argument("--use-fallback-indicator", type=parse_bool, default=True)
    parser.add_argument("--set-transformer-depth", type=int, default=1)
    parser.add_argument("--set-transformer-heads", type=int, default=4)
    parser.add_argument("--set-transformer-ff-mult", type=int, default=2)
    parser.add_argument("--residual-alpha-init", type=float, default=0.1)
    parser.add_argument("--note-head-checkpoint", default=None)
    parser.add_argument("--aux-loss-weight", type=float, choices=[0.0, 0.1], default=0.0)
    parser.add_argument("--eval-splits", default="val,test")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--architecture-screening-only", action="store_true")
    parser.add_argument("--require-residual-grad-check", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--focal-alpha", type=float, default=None)
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-training", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)

    eval_splits = effective_eval_splits(args)
    root = resolve_path(args.benchmark_root)
    benchmark = load_benchmark_index(root, splits=benchmark_splits_for_training(eval_splits))
    train = benchmark[benchmark["split"] == "train"]
    effective_train = train.iloc[balanced_downsample_indices(train["label"], seed=args.seed)] if args.sampling == "balanced" else train
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_row = {
        "model": args.model,
        "loss": args.loss,
        "sampling": args.sampling,
        "head": args.head,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "train_examples": len(train),
        "effective_train_examples": len(effective_train),
        "val_examples": int((benchmark["split"] == "val").sum()),
        "test_examples": int((benchmark["split"] == "test").sum()) if "test" in set(benchmark["split"]) else 0,
        "metrics": ",".join(METRICS),
        "final_metrics": ",".join(FINAL_METRICS),
        "training_launched": bool(args.run_training),
        "evaluation_splits": ",".join(eval_splits),
        "test_evaluation_launched": "test" in eval_splits,
        "architecture_screening_only": bool(args.architecture_screening_only),
        "device_requested": args.device,
        "device_resolved": str(device),
        "cuda_available": torch.cuda.is_available(),
        "section_embedding_prefix": args.section_embedding_prefix,
        "chunk_embedding_prefix": args.chunk_embedding_prefix,
        "section_embedding_extra_dir": ",".join(args.section_embedding_extra_dir),
        "chunk_embedding_extra_dir": ",".join(args.chunk_embedding_extra_dir),
    }
    if args.model == "notesm_v2":
        summary_row.update(notesm_v2_summary_fields(args))
    if args.model == "notesm_v2_hybrid":
        summary_row.update(notesm_v2_hybrid_summary_fields(args))
    summary = pd.DataFrame([summary_row])
    summary.to_csv(output_dir / f"{args.model}_training_plan_seed{args.seed}.csv", index=False)
    print(summary.to_string(index=False))
    if args.dry_run or not args.run_training:
        print("Downstream dry run complete. Heavy training not launched.")
        return 0
    if args.model == "notesm_v2":
        if args.section_embedding_dir is None:
            raise ValueError("--section-embedding-dir is required for notesm_v2 cached-section training")
        train_section_embedding_head(resolve_path(args.section_embedding_dir), benchmark, args, output_dir)
        return 0
    if args.model == "notesm_v2_hybrid":
        if args.section_embedding_dir is None:
            raise ValueError("--section-embedding-dir is required for notesm_v2_hybrid training")
        if args.embedding_cache_dir is None:
            raise ValueError("--embedding-cache-dir is required for notesm_v2_hybrid training")
        train_hybrid_embedding_head(
            section_cache_dir=resolve_path(args.section_embedding_dir),
            note_cache_dir=resolve_path(args.embedding_cache_dir),
            cache_name=args.cache_name,
            benchmark=benchmark,
            args=args,
            output_dir=output_dir,
        )
        return 0
    if args.embedding_cache_dir is None:
        raise ValueError("--embedding-cache-dir is required for cached embedding training")
    cache_dir = resolve_path(args.embedding_cache_dir)
    embeddings, metadata = load_note_embedding_cache(cache_dir, args.cache_name)
    embeddings, metadata = filter_embedding_rows_by_examples(
        embeddings,
        metadata,
        benchmark["example_id"].astype(str).drop_duplicates().tolist(),
        "Note embedding",
    )
    aligned = align_embedding_metadata(metadata, benchmark)
    if len(aligned) != embeddings.shape[0]:
        raise ValueError("Embedding row count does not match metadata row count")
    train_note_embedding_head(embeddings, aligned, args, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
