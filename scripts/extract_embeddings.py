from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.data.lampr_dataset import read_lampr_split  # noqa: E402
from notesm_v2.data.local_representations import content_positions_from_ids, evidence_spans_for_positions  # noqa: E402
from notesm_v2.models.backbone_registry import get_backbone_spec, last_hidden_state_from_outputs, load_tokenizer_and_model  # noqa: E402
from notesm_v2.models.notesm_v1 import last_valid_token_pool  # noqa: E402
from notesm_v2.models.notesm_v2 import extract_marker_hidden_states  # noqa: E402
from notesm_v2.models.special_tokens import add_project_special_tokens  # noqa: E402
from notesm_v2.training.embedding_cache import EmbeddingCacheManager, EmbeddingCacheSpec  # noqa: E402


SECTION_METADATA_COLUMNS = [
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
CHUNK_METADATA_COLUMNS = ["example_id", "split", "chunk_order", "token_length"]
LOCAL_REPRESENTATION_METADATA_COLUMNS = [
    "representation",
    "evidence_slot",
    "within_segment_position",
    "token_start_position",
    "token_end_position",
    "token_count",
]
FORBIDDEN_METADATA = {"text", "sequence_text", "merged_section_text", "section_text", "chunk_text", "subject_id", "hadm_id", "note_id"}
LOCAL_REPRESENTATIONS = {"masked_mean_1", "span_mean_2", "span_mean_4"}


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def deterministic_debug_embeddings(example_ids: list[str], hidden_size: int) -> np.ndarray:
    rows = []
    for example_id in example_ids:
        seed = int(hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        rows.append(rng.normal(size=hidden_size).astype(np.float32))
    return np.vstack(rows) if rows else np.zeros((0, hidden_size), dtype=np.float32)


def read_example_id_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing example id manifest: {path}")
    frame = pd.read_csv(path)
    if "example_id" not in frame.columns:
        raise ValueError(f"{path} must contain an example_id column")
    frame = frame.copy()
    frame["example_id"] = frame["example_id"].astype(str)
    if "split" in frame.columns:
        frame["split"] = frame["split"].astype(str)
    return frame


def filter_examples_by_manifest(examples: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    out = examples.copy()
    out["example_id"] = out["example_id"].astype(str)
    if "split" in manifest.columns:
        keys = manifest[["example_id", "split"]].drop_duplicates()
        selected = out.merge(keys, on=["example_id", "split"], how="inner")
    else:
        ids = set(manifest["example_id"].astype(str))
        selected = out[out["example_id"].isin(ids)].copy()
    if selected.empty:
        raise ValueError("No benchmark examples matched --example-id-file")
    if selected["example_id"].duplicated().any():
        duplicated = selected.loc[selected["example_id"].duplicated(), "example_id"].head(5).tolist()
        raise ValueError(f"Filtered examples contain duplicated example_id values: {duplicated}")
    order = manifest["example_id"].drop_duplicates().reset_index().rename(columns={"index": "_manifest_order"})
    split_order = {"train": 0, "val": 1, "test": 2}
    selected = selected.merge(order, on="example_id", how="left")
    selected["_split_order"] = selected["split"].map(split_order).fillna(99).astype(int)
    selected = selected.sort_values(["_split_order", "_manifest_order"]).drop(columns=["_split_order", "_manifest_order"])
    return selected.reset_index(drop=True)


def read_examples(root: Path, split: str, limit: int | None, example_id_file: Path | None = None) -> pd.DataFrame:
    if split in {"all", "train_val"}:
        frames = []
        split_names = ["train", "val"] if split == "train_val" else ["train", "val", "test"]
        for split_name in split_names:
            frame = read_lampr_split(root, split_name)
            frames.append(frame.head(limit).copy() if limit is not None and example_id_file is None else frame)
        examples = pd.concat(frames, ignore_index=True)
    else:
        frame = read_lampr_split(root, split)
        examples = frame.head(limit).copy() if limit is not None and example_id_file is None else frame
    if example_id_file is not None:
        examples = filter_examples_by_manifest(examples, read_example_id_manifest(example_id_file))
        if limit is not None:
            examples = examples.head(limit).copy()
    return examples


def prepare_tokenizer(tokenizer, model=None) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            if model is not None and hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
    add_project_special_tokens(tokenizer, model=model, init="mean")
    tokenizer.padding_side = "right"


def model_device(model, requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return device


def load_state_dict_checkpoint_into_model(model, checkpoint_dir: Path) -> dict:
    state_path = checkpoint_dir / "pytorch_training_state.pt"
    trainer_state_path = checkpoint_dir / "trainer_state.pt"
    if state_path.exists():
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")
        if "model_state" not in state:
            raise ValueError(f"{state_path} does not contain model_state")
        model.load_state_dict(state["model_state"])
        return state
    if trainer_state_path.exists():
        raise ValueError(
            f"{checkpoint_dir} looks like a save_pretrained checkpoint; pass it through --model-checkpoint only after "
            "adding save_pretrained loading support, or use the state_dict checkpoint directory."
        )
    raise FileNotFoundError(f"No supported state_dict checkpoint found under {checkpoint_dir}")


def iter_batches(frame: pd.DataFrame, batch_size: int):
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size]


def pool_hidden_states(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "last_token":
        return last_valid_token_pool(hidden, attention_mask)
    if pooling == "cls":
        return hidden[:, 0]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    raise ValueError(f"Unsupported note pooling mode: {pooling}")


def maybe_add_global_attention(encoded: dict[str, torch.Tensor], backbone_name: str, enabled: bool) -> dict[str, torch.Tensor]:
    if enabled and "longformer" in backbone_name.lower() and "input_ids" in encoded:
        global_attention_mask = torch.zeros_like(encoded["input_ids"])
        if global_attention_mask.numel():
            global_attention_mask[:, 0] = 1
        encoded["global_attention_mask"] = global_attention_mask
    return encoded


def backbone_hidden(model, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return token hidden states for encoder, HF causal LM, or official Mamba3 models."""
    if hasattr(model, "backbone"):
        try:
            return last_hidden_state_from_outputs(model.backbone(encoded["input_ids"]))
        except TypeError:
            return last_hidden_state_from_outputs(model.backbone(input_ids=encoded["input_ids"]))
    try:
        return last_hidden_state_from_outputs(model(**encoded))
    except TypeError:
        return last_hidden_state_from_outputs(model(input_ids=encoded["input_ids"]))


def extract_note_embeddings(
    frame: pd.DataFrame,
    tokenizer,
    model,
    max_length: int,
    batch_size: int,
    device: torch.device,
    pooling: str,
    backbone_name: str,
    use_longformer_global_attention: bool,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in iter_batches(frame, batch_size):
            encoded = tokenizer(
                batch["text"].astype(str).tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            encoded = maybe_add_global_attention(encoded, backbone_name, use_longformer_global_attention)
            hidden = backbone_hidden(model, encoded)
            pooled = pool_hidden_states(hidden, encoded["attention_mask"], pooling)
            outputs.append(pooled.detach().cpu().float().numpy())
    return np.vstack(outputs) if outputs else np.zeros((0, model.config.hidden_size), dtype=np.float32)


def write_note_cache_shards(
    output_dir: Path,
    frame: pd.DataFrame,
    cache_name: str,
    tokenizer,
    model,
    max_length: int,
    batch_size: int,
    device: torch.device,
    pooling: str,
    backbone_name: str,
    use_longformer_global_attention: bool,
    shard_size: int,
    resume: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    total_shards = int((len(frame) + shard_size - 1) // shard_size)
    total_rows = 0
    completed_rows = 0
    for shard_idx, start in enumerate(range(0, len(frame), shard_size)):
        shard = frame.iloc[start : start + shard_size].copy()
        embedding_path = output_dir / f"{cache_name}_embeddings_shard_{shard_idx:03d}.npy"
        metadata_path = output_dir / f"{cache_name}_metadata_shard_{shard_idx:03d}.parquet"
        print(f"[note-shard] {cache_name} shard {shard_idx + 1}/{total_shards} rows={len(shard)} start={start}", flush=True)
        if resume and embedding_path.exists() and metadata_path.exists():
            print(f"[note-shard] skipping existing shard {shard_idx:03d}", flush=True)
            completed_rows += len(pd.read_parquet(metadata_path))
            total_rows += len(shard)
            continue
        embeddings = extract_note_embeddings(
            shard,
            tokenizer,
            model,
            max_length,
            batch_size,
            device,
            pooling,
            backbone_name,
            use_longformer_global_attention,
        )
        metadata = shard[["example_id", "split"]].copy()
        metadata["example_id"] = metadata["example_id"].astype(str)
        metadata["shard_index"] = shard_idx
        metadata["row_index_in_shard"] = list(range(len(metadata)))
        metadata["global_row_index"] = list(range(start, start + len(metadata)))
        metadata["row_index"] = metadata["global_row_index"]
        np.save(embedding_path, embeddings.astype(np.float32, copy=False))
        metadata.to_parquet(metadata_path, index=False)
        completed_rows += len(metadata)
        total_rows += len(metadata)
        print(f"[note-shard] wrote {embedding_path}", flush=True)
        print(f"[note-shard] wrote {metadata_path}", flush=True)
    summary = pd.DataFrame(
        [
            {
                "cache_name": cache_name,
                "n_rows": completed_rows,
                "expected_rows": len(frame),
                "shard_size": shard_size,
                "n_shards": total_shards,
                "single_file_mode": False,
                "resume": bool(resume),
            }
        ]
    )
    summary.to_csv(output_dir / f"{cache_name}_shards_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def encode_with_marker(texts: list[str], tokenizer, marker_token: str, max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    marker_id = tokenizer.convert_tokens_to_ids(marker_token)
    if marker_id is None or marker_id == tokenizer.unk_token_id:
        raise ValueError(f"Marker token is not in tokenizer vocabulary: {marker_token}")
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=max_length - 1,
    )
    rows = [list(ids) + [int(marker_id)] for ids in encoded["input_ids"]]
    max_len = max(len(row) for row in rows) if rows else 1
    pad_id = int(tokenizer.pad_token_id)
    input_ids = []
    attention_mask = []
    for row in rows:
        pad = max_len - len(row)
        input_ids.append(row + [pad_id] * pad)
        attention_mask.append([1] * len(row) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
    }


def extract_marker_embeddings(
    frame: pd.DataFrame,
    text_col: str,
    marker_token: str,
    tokenizer,
    model,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    marker_id = tokenizer.convert_tokens_to_ids(marker_token)
    with torch.no_grad():
        for batch in iter_batches(frame, batch_size):
            encoded = encode_with_marker(batch[text_col].astype(str).tolist(), tokenizer, marker_token, max_length, device)
            hidden = backbone_hidden(model, encoded)
            pooled = extract_marker_hidden_states(hidden, encoded["input_ids"], marker_id)
            outputs.append(pooled.detach().cpu().float().numpy())
    return np.vstack(outputs) if outputs else np.zeros((0, model.config.hidden_size), dtype=np.float32)


def content_token_mask(encoded: dict[str, torch.Tensor], tokenizer, marker_id: int | None) -> torch.Tensor:
    input_ids = encoded["input_ids"]
    special_ids = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None)
    if marker_id is not None:
        special_ids.add(int(marker_id))
    if tokenizer.pad_token_id is not None:
        special_ids.add(int(tokenizer.pad_token_id))
    mask = torch.zeros_like(encoded["attention_mask"], dtype=torch.bool)
    for row_idx in range(input_ids.shape[0]):
        positions = content_positions_from_ids(
            input_ids[row_idx].detach().cpu().tolist(),
            encoded["attention_mask"][row_idx].detach().cpu().tolist(),
            special_ids,
        )
        if len(positions):
            mask[row_idx, torch.tensor(positions, dtype=torch.long, device=mask.device)] = True
    return mask


def representation_vectors_for_row(
    hidden_row: torch.Tensor,
    valid_positions: np.ndarray,
    representation: str,
) -> tuple[list[np.ndarray], list[dict[str, int]]]:
    if len(valid_positions) == 0:
        return [], []
    vectors: list[np.ndarray] = []
    rows: list[dict[str, int]] = []
    for positions, row in evidence_spans_for_positions(valid_positions, representation):
        positions = np.asarray(positions, dtype=int)
        pooled = hidden_row[torch.tensor(positions, dtype=torch.long, device=hidden_row.device)].mean(dim=0)
        vectors.append(pooled.detach().cpu().float().numpy())
        rows.append(row)
    return vectors, rows


def extract_local_representation_embeddings(
    frame: pd.DataFrame,
    text_col: str,
    marker_token: str,
    representation: str,
    tokenizer,
    model,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, pd.DataFrame]:
    outputs: list[np.ndarray] = []
    metadata_rows: list[dict[str, int]] = []
    marker_id = tokenizer.convert_tokens_to_ids(marker_token)
    with torch.no_grad():
        for batch in iter_batches(frame, batch_size):
            encoded = encode_with_marker(batch[text_col].astype(str).tolist(), tokenizer, marker_token, max_length, device)
            hidden = backbone_hidden(model, encoded)
            valid_mask = content_token_mask(encoded, tokenizer, int(marker_id))
            for local_row, (_, source_row) in enumerate(batch.iterrows()):
                valid_positions = torch.nonzero(valid_mask[local_row], as_tuple=False).squeeze(-1).detach().cpu().numpy().astype(int)
                vectors, rows = representation_vectors_for_row(hidden[local_row], valid_positions, representation)
                for vector, row in zip(vectors, rows, strict=True):
                    outputs.append(vector)
                    metadata_rows.append(
                        {
                            "source_row_index": int(source_row.name),
                            "representation": representation,
                            **row,
                        }
                    )
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    embeddings = np.vstack(outputs).astype(np.float32, copy=False) if outputs else np.zeros((0, hidden_size), dtype=np.float32)
    return embeddings, pd.DataFrame(metadata_rows)


def limit_by_examples(frame: pd.DataFrame, limit_examples: int | None, example_ids: list[str] | None = None) -> pd.DataFrame:
    if frame.empty or "example_id" not in frame.columns:
        return frame.copy()
    if example_ids is None:
        if limit_examples is None:
            return frame.copy()
        example_ids = frame["example_id"].astype(str).drop_duplicates().head(limit_examples).tolist()
    selected = set(str(example_id) for example_id in example_ids)
    return frame[frame["example_id"].astype(str).isin(selected)].copy()


def read_section_cache(
    section_cache_dir: Path,
    split: str,
    limit_examples: int | None,
    example_ids: list[str] | None = None,
) -> pd.DataFrame:
    frames = []
    if split == "all":
        splits = ["train", "val", "test"]
    elif split == "train_val":
        splits = ["train", "val"]
    else:
        splits = [split]
    for split_name in splits:
        path = section_cache_dir / f"section_cache_{split_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        frame = pd.read_parquet(path)
        if "split" not in frame.columns:
            frame["split"] = split_name
        if example_ids is None:
            frames.append(limit_by_examples(frame, limit_examples))
        else:
            frames.append(limit_by_examples(frame, limit_examples=None, example_ids=example_ids))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_chunk_cache(
    section_cache_dir: Path,
    split: str,
    limit_examples: int | None,
    example_ids: list[str] | None = None,
) -> pd.DataFrame:
    frames = []
    if split == "all":
        splits = ["train", "val", "test"]
    elif split == "train_val":
        splits = ["train", "val"]
    else:
        splits = [split]
    for split_name in splits:
        path = section_cache_dir / f"fallback_chunk_cache_{split_name}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if "split" not in frame.columns:
            frame["split"] = split_name
        if example_ids is None:
            frames.append(limit_by_examples(frame, limit_examples))
        else:
            frames.append(limit_by_examples(frame, limit_examples=None, example_ids=example_ids))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*CHUNK_METADATA_COLUMNS, "chunk_text"])
    return combined


def limited_section_example_ids(section_cache_dir: Path, split: str, limit_examples: int | None) -> list[str] | None:
    if limit_examples is None:
        return None
    try:
        sections = read_section_cache(section_cache_dir, split, limit_examples)
    except FileNotFoundError:
        return None
    return sections["example_id"].astype(str).drop_duplicates().tolist()


def write_marker_cache(output_dir: Path, embeddings: np.ndarray, metadata: pd.DataFrame, prefix: str) -> None:
    if FORBIDDEN_METADATA & set(metadata.columns):
        raise ValueError(f"{prefix} metadata contains restricted columns: {sorted(FORBIDDEN_METADATA & set(metadata.columns))}")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{prefix}_embeddings.npy", embeddings.astype(np.float32, copy=False))
    metadata = metadata.copy()
    metadata["row_index"] = list(range(len(metadata)))
    metadata.to_parquet(output_dir / f"{prefix}_embedding_metadata.parquet", index=False)
    print(f"Wrote {output_dir / f'{prefix}_embeddings.npy'}")
    print(f"Wrote {output_dir / f'{prefix}_embedding_metadata.parquet'}")


def expanded_local_metadata(frame: pd.DataFrame, keep_cols: list[str], rep_meta: pd.DataFrame) -> pd.DataFrame:
    if rep_meta.empty:
        columns = [*keep_cols, *LOCAL_REPRESENTATION_METADATA_COLUMNS]
        return pd.DataFrame(columns=columns)
    source = frame[keep_cols].copy()
    source["_source_row_index"] = source.index.astype(int)
    expanded = rep_meta.merge(source, left_on="source_row_index", right_on="_source_row_index", how="left", validate="many_to_one")
    expanded = expanded.drop(columns=["source_row_index", "_source_row_index"])
    ordered = [col for col in keep_cols if col in expanded.columns] + [col for col in LOCAL_REPRESENTATION_METADATA_COLUMNS if col in expanded.columns]
    return expanded[ordered].copy()


def write_local_representation_cache(
    output_dir: Path,
    frame: pd.DataFrame,
    text_col: str,
    keep_cols: list[str],
    prefix: str,
    marker_token: str,
    representation: str,
    tokenizer,
    model,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> None:
    embeddings, rep_meta = extract_local_representation_embeddings(frame, text_col, marker_token, representation, tokenizer, model, max_length, batch_size, device)
    metadata = expanded_local_metadata(frame, keep_cols, rep_meta)
    write_marker_cache(output_dir, embeddings, metadata, prefix)


def write_marker_cache_shards(
    output_dir: Path,
    frame: pd.DataFrame,
    text_col: str,
    keep_cols: list[str],
    prefix: str,
    marker_token: str,
    tokenizer,
    model,
    max_length: int,
    batch_size: int,
    device: torch.device,
    shard_size: int,
    resume: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    total_rows = 0
    for shard_idx, start in enumerate(range(0, len(frame), shard_size)):
        embedding_path = output_dir / f"{prefix}_embeddings_shard_{shard_idx:03d}.npy"
        metadata_path = output_dir / f"{prefix}_embedding_metadata_shard_{shard_idx:03d}.parquet"
        shard = frame.iloc[start : start + shard_size].copy()
        if resume and embedding_path.exists() and metadata_path.exists():
            total_rows += len(shard)
            print(f"[{prefix}-shard] skipping existing shard {shard_idx:03d}", flush=True)
            continue
        embeddings = extract_marker_embeddings(shard, text_col, marker_token, tokenizer, model, max_length, batch_size, device)
        metadata = shard[keep_cols].copy()
        if FORBIDDEN_METADATA & set(metadata.columns):
            raise ValueError(f"{prefix} shard metadata contains restricted columns: {sorted(FORBIDDEN_METADATA & set(metadata.columns))}")
        metadata["shard_index"] = shard_idx
        metadata["row_index_in_shard"] = list(range(len(metadata)))
        metadata["global_row_index"] = list(range(start, start + len(metadata)))
        metadata["row_index"] = metadata["global_row_index"]
        np.save(embedding_path, embeddings.astype(np.float32, copy=False))
        metadata.to_parquet(metadata_path, index=False)
        total_rows += len(metadata)
    summary = pd.DataFrame(
        [
            {
                "prefix": prefix,
                "n_rows": total_rows,
                "shard_size": shard_size,
                "n_shards": int((len(frame) + shard_size - 1) // shard_size),
                "single_file_mode": False,
            }
        ]
    )
    summary.to_csv(output_dir / f"{prefix}_embedding_shards_summary.csv", index=False)
    print(summary.to_string(index=False))


def write_local_representation_cache_shards(
    output_dir: Path,
    frame: pd.DataFrame,
    text_col: str,
    keep_cols: list[str],
    prefix: str,
    marker_token: str,
    representation: str,
    tokenizer,
    model,
    max_length: int,
    batch_size: int,
    device: torch.device,
    shard_size: int,
    resume: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    total_input_rows = 0
    total_output_rows = 0
    output_offset = 0
    n_shards = int((len(frame) + shard_size - 1) // shard_size)
    for shard_idx, start in enumerate(range(0, len(frame), shard_size)):
        embedding_path = output_dir / f"{prefix}_embeddings_shard_{shard_idx:03d}.npy"
        metadata_path = output_dir / f"{prefix}_embedding_metadata_shard_{shard_idx:03d}.parquet"
        shard = frame.iloc[start : start + shard_size].copy()
        if resume and embedding_path.exists() and metadata_path.exists():
            metadata = pd.read_parquet(metadata_path)
            total_input_rows += len(shard)
            total_output_rows += len(metadata)
            output_offset += len(metadata)
            print(f"[{prefix}-shard] skipping existing shard {shard_idx:03d}", flush=True)
            continue
        embeddings, rep_meta = extract_local_representation_embeddings(shard, text_col, marker_token, representation, tokenizer, model, max_length, batch_size, device)
        metadata = expanded_local_metadata(shard, keep_cols, rep_meta)
        if FORBIDDEN_METADATA & set(metadata.columns):
            raise ValueError(f"{prefix} shard metadata contains restricted columns: {sorted(FORBIDDEN_METADATA & set(metadata.columns))}")
        metadata["shard_index"] = shard_idx
        metadata["row_index_in_shard"] = list(range(len(metadata)))
        metadata["global_row_index"] = list(range(output_offset, output_offset + len(metadata)))
        metadata["row_index"] = metadata["global_row_index"]
        np.save(embedding_path, embeddings.astype(np.float32, copy=False))
        metadata.to_parquet(metadata_path, index=False)
        total_input_rows += len(shard)
        total_output_rows += len(metadata)
        output_offset += len(metadata)
        print(f"[{prefix}-shard] wrote {embedding_path}", flush=True)
        print(f"[{prefix}-shard] wrote {metadata_path}", flush=True)
    summary = pd.DataFrame(
        [
            {
                "prefix": prefix,
                "representation": representation,
                "n_input_rows": total_input_rows,
                "n_rows": total_output_rows,
                "shard_size": shard_size,
                "n_shards": n_shards,
                "single_file_mode": False,
            }
        ]
    )
    summary.to_csv(output_dir / f"{prefix}_embedding_shards_summary.csv", index=False)
    print(summary.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract or smoke-test text-free embedding caches.")
    parser.add_argument("--benchmark-root", default="results/lampr_full_v0_2")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all", "train_val"])
    parser.add_argument("--output-dir", default="results/embedding_cache")
    parser.add_argument("--cache-name", default="debug")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Limit note examples. For section/chunk caches, this is treated as an example-level limit for backward compatibility.")
    parser.add_argument("--example-id-file", default=None, help="CSV manifest with example_id and optional split/label columns for subset extraction.")
    parser.add_argument("--limit-examples", type=int, default=None, help="Example-level limit for section/chunk caches; keeps all rows for selected example_id values.")
    parser.add_argument("--debug-random", action="store_true", help="Write deterministic random note embeddings for pipeline smoke tests.")
    parser.add_argument("--run-backbone", action="store_true", help="Run actual frozen backbone extraction.")
    parser.add_argument(
        "--backbone",
        choices=["mamba2_130m", "clinical_mamba_130m", "mamba3_siso_187m", "bioclinicalbert", "longformer", "modernbert_base"],
        default="mamba2_130m",
    )
    parser.add_argument("--input-kind", choices=["note", "section", "chunk"], default="note")
    parser.add_argument(
        "--pooling",
        choices=["auto", "last_token", "cls", "mean", "sec_sum", "chunk_sum", "masked_mean_1", "span_mean_2", "span_mean_4"],
        default="auto",
    )
    parser.add_argument("--section-cache-dir", default="results/lampr_full_v0_2/section_cache")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=None, help="Write section/chunk embeddings as shard files for larger extraction jobs.")
    parser.add_argument("--resume", action="store_true", help="Skip existing shard files when --shard-size is used.")
    parser.add_argument("--longformer-global-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-checkpoint", default=None, help="Optional state_dict DAPT checkpoint directory with pytorch_training_state.pt.")
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    if args.debug_random:
        example_id_file = resolve_path(args.example_id_file) if args.example_id_file else None
        examples = read_examples(resolve_path(args.benchmark_root), args.split, args.limit, example_id_file=example_id_file)
        example_ids = examples["example_id"].astype(str).tolist()
        embeddings = deterministic_debug_embeddings(example_ids, hidden_size=args.hidden_size)
        manager = EmbeddingCacheManager(output_dir, EmbeddingCacheSpec(name=args.cache_name, hidden_size=args.hidden_size))
        manager.write(example_ids, embeddings, extra_metadata=examples[["example_id", "split"]].copy())
        print(f"Wrote embedding cache to {manager.root}")
        print(f"rows={len(example_ids)} hidden_size={args.hidden_size}")
        return 0
    if not args.run_backbone:
        raise NotImplementedError("Use --debug-random for smoke tests or --run-backbone for actual extraction.")

    spec = get_backbone_spec(args.backbone)
    tokenizer, model = load_tokenizer_and_model(spec)
    prepare_tokenizer(tokenizer, model=model)
    checkpoint_state = None
    checkpoint_dir = None
    if args.model_checkpoint:
        checkpoint_dir = resolve_path(args.model_checkpoint)
        checkpoint_state = load_state_dict_checkpoint_into_model(model, checkpoint_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_summary = pd.DataFrame(
            [
                {
                    "model_checkpoint": str(checkpoint_dir),
                    "checkpoint_global_step": checkpoint_state.get("global_step", ""),
                    "checkpoint_format": checkpoint_state.get("checkpoint_format", ""),
                    "checkpoint_scheduler_total_steps": (checkpoint_state.get("args") or {}).get("scheduler_total_steps", ""),
                    "checkpoint_max_steps": (checkpoint_state.get("args") or {}).get("max_steps", ""),
                }
            ]
        )
        checkpoint_summary.to_csv(output_dir / f"{args.cache_name}_checkpoint_load_summary.csv", index=False)
        print(checkpoint_summary.to_string(index=False), flush=True)
    device = model_device(model, args.device)
    max_length = args.max_length or spec.max_length

    if args.input_kind == "note":
        pooling = args.pooling
        if pooling == "auto":
            pooling = "last_token" if spec.family in {"mamba", "mamba3"} else "cls"
        if pooling not in {"last_token", "cls", "mean"}:
            raise ValueError("note input-kind supports --pooling auto/last_token/cls/mean")
        example_id_file = resolve_path(args.example_id_file) if args.example_id_file else None
        examples = read_examples(resolve_path(args.benchmark_root), args.split, args.limit, example_id_file=example_id_file)
        if args.shard_size is not None:
            write_note_cache_shards(
                output_dir,
                examples,
                args.cache_name,
                tokenizer,
                model,
                max_length,
                args.batch_size,
                device,
                pooling,
                spec.name,
                args.longformer_global_attention,
                args.shard_size,
                args.resume,
            )
            return 0
        embeddings = extract_note_embeddings(
            examples,
            tokenizer,
            model,
            max_length,
            args.batch_size,
            device,
            pooling,
            spec.name,
            args.longformer_global_attention,
        )
        manager = EmbeddingCacheManager(output_dir, EmbeddingCacheSpec(name=args.cache_name, hidden_size=embeddings.shape[1]))
        manager.write(examples["example_id"].astype(str).tolist(), embeddings, extra_metadata=examples[["example_id", "split"]].copy())
        print(f"Wrote embedding cache to {manager.root}")
        print(f"rows={len(examples)} hidden_size={embeddings.shape[1]}")
        return 0

    section_cache_dir = resolve_path(args.section_cache_dir)
    section_limit = args.limit_examples if args.limit_examples is not None else args.limit
    example_manifest = read_example_id_manifest(resolve_path(args.example_id_file)) if args.example_id_file else None
    if example_manifest is not None and "split" in example_manifest.columns:
        invalid_splits = sorted(set(example_manifest["split"].astype(str)) - {"train", "val", "test"})
        if invalid_splits:
            raise ValueError(f"--example-id-file contains unsupported split values: {invalid_splits}")
    manifest_ids = example_manifest["example_id"].astype(str).drop_duplicates().tolist() if example_manifest is not None else None
    if args.input_kind == "section":
        if args.pooling not in {"auto", "sec_sum", *LOCAL_REPRESENTATIONS}:
            raise ValueError("section input-kind requires --pooling sec_sum/masked_mean_1/span_mean_2/span_mean_4")
        pooling = "sec_sum" if args.pooling == "auto" else args.pooling
        sections = read_section_cache(section_cache_dir, args.split, section_limit, example_ids=manifest_ids)
        keep_cols = [col for col in SECTION_METADATA_COLUMNS if col in sections.columns]
        if args.shard_size is not None:
            if pooling in LOCAL_REPRESENTATIONS:
                write_local_representation_cache_shards(output_dir, sections, "merged_section_text", keep_cols, "section", "<SEC_SUM>", pooling, tokenizer, model, max_length, args.batch_size, device, args.shard_size, resume=args.resume)
            else:
                write_marker_cache_shards(output_dir, sections, "merged_section_text", keep_cols, "section", "<SEC_SUM>", tokenizer, model, max_length, args.batch_size, device, args.shard_size, resume=args.resume)
            return 0
        if pooling in LOCAL_REPRESENTATIONS:
            write_local_representation_cache(output_dir, sections, "merged_section_text", keep_cols, "section", "<SEC_SUM>", pooling, tokenizer, model, max_length, args.batch_size, device)
        else:
            embeddings = extract_marker_embeddings(sections, "merged_section_text", "<SEC_SUM>", tokenizer, model, max_length, args.batch_size, device)
            write_marker_cache(output_dir, embeddings, sections[keep_cols].copy(), "section")
        return 0
    if args.pooling not in {"auto", "chunk_sum", *LOCAL_REPRESENTATIONS}:
        raise ValueError("chunk input-kind requires --pooling chunk_sum/masked_mean_1/span_mean_2/span_mean_4")
    pooling = "chunk_sum" if args.pooling == "auto" else args.pooling
    example_ids = manifest_ids if manifest_ids is not None else limited_section_example_ids(section_cache_dir, args.split, section_limit)
    chunks = read_chunk_cache(section_cache_dir, args.split, section_limit, example_ids=example_ids)
    keep_cols = [col for col in CHUNK_METADATA_COLUMNS if col in chunks.columns]
    if args.shard_size is not None:
        if pooling in LOCAL_REPRESENTATIONS:
            write_local_representation_cache_shards(output_dir, chunks, "chunk_text", keep_cols, "chunk", "<CHUNK_SUM>", pooling, tokenizer, model, max_length, args.batch_size, device, args.shard_size, resume=args.resume)
        else:
            write_marker_cache_shards(output_dir, chunks, "chunk_text", keep_cols, "chunk", "<CHUNK_SUM>", tokenizer, model, max_length, args.batch_size, device, args.shard_size, resume=args.resume)
        return 0
    if pooling in LOCAL_REPRESENTATIONS:
        write_local_representation_cache(output_dir, chunks, "chunk_text", keep_cols, "chunk", "<CHUNK_SUM>", pooling, tokenizer, model, max_length, args.batch_size, device)
    else:
        embeddings = extract_marker_embeddings(chunks, "chunk_text", "<CHUNK_SUM>", tokenizer, model, max_length, args.batch_size, device)
        write_marker_cache(output_dir, embeddings, chunks[keep_cols].copy(), "chunk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
