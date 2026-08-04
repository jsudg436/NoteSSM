from __future__ import annotations

import numpy as np


def binary_focal_loss_numpy(y_true: np.ndarray, y_prob: np.ndarray, gamma: float = 2.0, alpha: float = 0.25) -> float:
    eps = 1e-8
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    loss_pos = -alpha * ((1.0 - y_prob) ** gamma) * y_true * np.log(y_prob)
    loss_neg = -(1.0 - alpha) * (y_prob**gamma) * (1.0 - y_true) * np.log(1.0 - y_prob)
    return float(np.mean(loss_pos + loss_neg))
