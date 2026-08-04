from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= left) & (y_prob < right if right < 1.0 else y_prob <= right)
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(accuracy - confidence)
    return ece


def risk_deciles(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    order = np.argsort(-y_prob)
    chunks = np.array_split(order, n_bins)
    rows: list[dict[str, float]] = []
    total_pos = max(float(np.sum(y_true)), 1.0)
    for idx, chunk in enumerate(chunks, start=1):
        positives = float(np.sum(y_true[chunk]))
        rows.append(
            {
                "risk_bin": idx,
                "n": float(len(chunk)),
                "mean_score": float(np.mean(y_prob[chunk])) if len(chunk) else 0.0,
                "observed_rate": positives / float(len(chunk)) if len(chunk) else 0.0,
                "sensitivity": positives / total_pos,
            }
        )
    return rows


def binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(y_true), 1)
    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "AUROC": float(auroc_score(y_true, y_prob)),
        "AUPRC": float(auprc_score(y_true, y_prob)),
    }


def auroc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def auprc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.sum(y_true) == 0:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))
