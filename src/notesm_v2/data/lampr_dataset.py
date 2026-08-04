from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TEXT_COLUMN = "text"


def read_lampr_split(root: str | Path, split: str) -> pd.DataFrame:
    path = Path(root) / f"benchmark_examples_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing LAMP-R split parquet: {path}")
    frame = pd.read_parquet(path)
    if "split" not in frame.columns:
        frame["split"] = split
    return frame


def read_lampr_splits(root: str | Path, splits: tuple[str, ...] = ("train", "val", "test")) -> pd.DataFrame:
    return pd.concat([read_lampr_split(root, split) for split in splits], ignore_index=True)


@dataclass(frozen=True)
class LamprExample:
    example_id: str
    split: str
    label: int
    text: str
    metadata: dict[str, Any]


class LamprDownstreamDataset:
    """Lightweight downstream dataset over frozen LAMP-R parquet examples.

    This class intentionally does not log or report raw text. Callers may pass a
    tokenizer to get tokenized tensors/dicts at item access time.
    """

    def __init__(
        self,
        examples: pd.DataFrame,
        tokenizer: Any | None = None,
        max_length: int = 4096,
        include_text: bool = True,
    ) -> None:
        required = {"example_id", "split", "label", TEXT_COLUMN}
        missing = required - set(examples.columns)
        if missing:
            raise ValueError(f"LAMP-R examples missing columns: {sorted(missing)}")
        self.examples = examples.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_text = include_text

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.examples.iloc[index]
        item: dict[str, Any] = {
            "example_id": row["example_id"],
            "split": row["split"],
            "label": int(row["label"]),
        }
        if self.include_text:
            item["text"] = str(row[TEXT_COLUMN])
        if self.tokenizer is not None:
            item["tokens"] = self.tokenizer(
                str(row[TEXT_COLUMN]),
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
        return item


def balanced_downsample_indices(labels: list[int] | pd.Series, seed: int = 17) -> list[int]:
    labels_series = pd.Series(labels).reset_index(drop=True)
    positives = labels_series[labels_series == 1].index.to_series()
    negatives = labels_series[labels_series == 0].index.to_series()
    if positives.empty or negatives.empty:
        return labels_series.index.tolist()
    n = min(len(positives), len(negatives))
    pos_sample = positives.sample(n=n, random_state=seed)
    neg_sample = negatives.sample(n=n, random_state=seed)
    return sorted(pd.concat([pos_sample, neg_sample]).astype(int).tolist())
