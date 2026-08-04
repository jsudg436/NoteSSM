from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmbeddingCacheSpec:
    name: str
    hidden_size: int
    dtype: str = "float32"


class EmbeddingCacheManager:
    """Disk-backed embedding cache with text-free metadata."""

    def __init__(self, root: str | Path, spec: EmbeddingCacheSpec) -> None:
        self.root = Path(root)
        self.spec = spec
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def embedding_path(self) -> Path:
        return self.root / f"{self.spec.name}.npy"

    @property
    def metadata_path(self) -> Path:
        return self.root / f"{self.spec.name}_metadata.parquet"

    def write(self, example_ids: list[str], embeddings: np.ndarray, extra_metadata: pd.DataFrame | None = None) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be rank-2 [n_examples, hidden_size]")
        if embeddings.shape[0] != len(example_ids):
            raise ValueError("number of embeddings must match example_ids")
        if embeddings.shape[1] != self.spec.hidden_size:
            raise ValueError(f"hidden size mismatch: {embeddings.shape[1]} != {self.spec.hidden_size}")
        np.save(self.embedding_path, embeddings.astype(self.spec.dtype, copy=False))
        metadata = pd.DataFrame({"example_id": example_ids, "row_index": list(range(len(example_ids)))})
        if extra_metadata is not None:
            forbidden = {
                "text",
                "sequence_text",
                "merged_section_text",
                "section_text",
                "chunk_text",
                "subject_id",
                "hadm_id",
                "note_id",
            }
            if forbidden & set(extra_metadata.columns):
                raise ValueError(f"embedding metadata cannot contain raw text columns: {sorted(forbidden & set(extra_metadata.columns))}")
            metadata = metadata.merge(extra_metadata, on="example_id", how="left")
        metadata.to_parquet(self.metadata_path, index=False)

    def read(self) -> tuple[pd.DataFrame, np.ndarray]:
        if not self.embedding_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing embedding cache files under {self.root}")
        return pd.read_parquet(self.metadata_path), np.load(self.embedding_path)
