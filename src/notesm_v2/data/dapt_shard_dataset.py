from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from notesm_v2.data.dapt_blocks import causal_lm_collate_fn
from notesm_v2.data.dapt_token_shards import read_token_shard_metadata, shard_array_path


class DaptTokenShardDataset:
    def __init__(
        self,
        shard_dir: str | Path,
        *,
        shuffle: bool = False,
        seed: int | None = None,
        preload_shard_count: int = 2,
        min_block_tokens: int = 1,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.min_block_tokens = max(int(min_block_tokens), 1)
        metadata = read_token_shard_metadata(self.shard_dir).reset_index(drop=True)
        self.source_row_count = int(len(metadata))
        if metadata.empty:
            raise ValueError(f"No DAPT token shard metadata found in {self.shard_dir}")
        if self.min_block_tokens > 1:
            metadata = metadata[metadata["block_length"].astype(int) >= self.min_block_tokens].reset_index(drop=True)
        self.skipped_short_block_count = int(self.source_row_count - len(metadata))
        self.metadata = metadata
        if self.metadata.empty:
            raise ValueError(
                f"No DAPT token shard rows remain in {self.shard_dir} after min_block_tokens={self.min_block_tokens} filtering"
            )
        self.indices = np.arange(len(self.metadata), dtype=int)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)
        self.preload_shard_count = max(int(preload_shard_count), 1)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.indices)

    def _load_shard(self, shard_id: int) -> np.ndarray:
        shard_id = int(shard_id)
        if shard_id in self._cache:
            array = self._cache.pop(shard_id)
            self._cache[shard_id] = array
            return array
        path = shard_array_path(self.shard_dir, shard_id)
        if not path.exists():
            raise FileNotFoundError(f"Missing DAPT token shard array: {path}")
        array = np.load(path)
        self._cache[shard_id] = array
        while len(self._cache) > self.preload_shard_count:
            self._cache.popitem(last=False)
        return array

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        metadata_index = int(self.indices[index])
        row = self.metadata.iloc[metadata_index]
        shard_id = int(row["shard_id"])
        row_index = int(row["row_index_in_shard"])
        block_length = int(row["block_length"])
        array = self._load_shard(shard_id)
        block = array[row_index, :block_length].astype(np.int64, copy=False).tolist()
        return {"input_ids": [int(token) for token in block], "block_id": str(row["block_id"])}


class DaptTokenShardIterableDataset:
    def __init__(self, shard_dir: str | Path, *, shuffle: bool = False, seed: int | None = None) -> None:
        self.dataset = DaptTokenShardDataset(shard_dir, shuffle=shuffle, seed=seed)

    def __iter__(self):
        for index in range(len(self.dataset)):
            yield self.dataset[index]


def dapt_token_collate_fn(batch: list[dict[str, list[int]]], pad_token_id: int):
    return causal_lm_collate_fn(batch, pad_token_id=pad_token_id)
