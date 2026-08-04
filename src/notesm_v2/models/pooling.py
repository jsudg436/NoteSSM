from __future__ import annotations

import numpy as np


def mean_pool(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(float)
    denom = np.maximum(mask.sum(axis=1), 1.0)
    return (hidden_states * mask).sum(axis=1) / denom


def last_valid_pool(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    indices = np.maximum(attention_mask.sum(axis=1) - 1, 0).astype(int)
    return hidden_states[np.arange(hidden_states.shape[0]), indices]
