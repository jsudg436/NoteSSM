from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationHeadConfig:
    hidden_size: int
    dropout: float = 0.1
    num_labels: int = 1
