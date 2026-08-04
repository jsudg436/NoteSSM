from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_METRICS = ("AUPRC", "AUROC", "Accuracy", "Precision", "Recall", "F1", "Brier", "ECE")
HIGHER_IS_BETTER = {
    "AUPRC": True,
    "AUROC": True,
    "Accuracy": True,
    "Precision": True,
    "Recall": True,
    "F1": True,
    "Brier": False,
    "ECE": False,
}


@dataclass(frozen=True)
class PredictionSet:
    name: str
    y_true: np.ndarray
    y_prob: np.ndarray
    y_pred: np.ndarray


class WeightedMetricComputer:
    """Compute binary metrics under row multiplicities without expanding rows."""

    def __init__(self, y_true: Sequence[int | float], y_prob: Sequence[float], y_pred: Sequence[int | float]) -> None:
        self.y = np.asarray(y_true, dtype=np.float64)
        self.p = np.asarray(y_prob, dtype=np.float64)
        self.pred = np.asarray(y_pred, dtype=np.float64)
        if not (len(self.y) == len(self.p) == len(self.pred)):
            raise ValueError("y_true, y_prob, and y_pred must have the same length")
        self.ap_order, self.ap_starts = self._score_groups(descending=True)
        self.auc_order, self.auc_starts = self._score_groups(descending=False)
        self.ece_bins = np.minimum((self.p * 10).astype(np.int32), 9)

    def _score_groups(self, *, descending: bool) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(self.p, kind="mergesort")
        if descending:
            order = order[::-1]
        sorted_prob = self.p[order]
        starts = np.r_[0, np.flatnonzero(sorted_prob[1:] != sorted_prob[:-1]) + 1]
        return order, starts.astype(np.int32)

    def compute(self, weights: np.ndarray) -> dict[str, np.ndarray]:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim == 1:
            w = w[None, :]
        if w.shape[1] != len(self.y):
            raise ValueError(f"Weight width {w.shape[1]} does not match prediction rows {len(self.y)}")

        total = w.sum(axis=1)
        pos_total = w @ self.y
        neg_total = total - pos_total
        tp = w @ (self.y * self.pred)
        fp = w @ ((1.0 - self.y) * self.pred)
        fn = pos_total - tp
        tn = neg_total - fp

        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
        accuracy = np.divide(tp + tn, total, out=np.full_like(total, np.nan), where=total > 0)
        brier = np.divide(w @ ((self.p - self.y) ** 2), total, out=np.full_like(total, np.nan), where=total > 0)

        ece_numerator = np.zeros(len(w), dtype=np.float64)
        for bin_id in range(10):
            mask = (self.ece_bins == bin_id).astype(np.float64)
            observed = w @ (mask * self.y)
            predicted = w @ (mask * self.p)
            ece_numerator += np.abs(observed - predicted)
        ece = np.divide(ece_numerator, total, out=np.full_like(total, np.nan), where=total > 0)

        ap_w = w[:, self.ap_order]
        ap_y = self.y[self.ap_order]
        ap_group_pos = np.add.reduceat(ap_w * ap_y, self.ap_starts, axis=1)
        ap_group_total = np.add.reduceat(ap_w, self.ap_starts, axis=1)
        ap_cum_pos = np.cumsum(ap_group_pos, axis=1)
        ap_cum_total = np.cumsum(ap_group_total, axis=1)
        ap_precision = np.divide(ap_cum_pos, ap_cum_total, out=np.zeros_like(ap_cum_pos), where=ap_cum_total > 0)
        auprc = np.divide(
            np.sum(ap_precision * ap_group_pos, axis=1),
            pos_total,
            out=np.full_like(pos_total, np.nan),
            where=pos_total > 0,
        )

        auc_w = w[:, self.auc_order]
        auc_y = self.y[self.auc_order]
        auc_group_pos = np.add.reduceat(auc_w * auc_y, self.auc_starts, axis=1)
        auc_group_neg = np.add.reduceat(auc_w * (1.0 - auc_y), self.auc_starts, axis=1)
        neg_before = np.cumsum(auc_group_neg, axis=1) - auc_group_neg
        auc_numerator = np.sum(auc_group_pos * (neg_before + 0.5 * auc_group_neg), axis=1)
        auroc = np.divide(
            auc_numerator,
            pos_total * neg_total,
            out=np.full_like(pos_total, np.nan),
            where=(pos_total > 0) & (neg_total > 0),
        )

        return {
            "AUPRC": auprc,
            "AUROC": auroc,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Brier": brier,
            "ECE": ece,
        }


def patient_cluster_weights(
    rng: np.random.Generator,
    patient_index: np.ndarray,
    n_patients: int,
    n_replicates: int,
) -> np.ndarray:
    """Sample patients with replacement and map patient multiplicities to episode rows."""
    if n_patients <= 0:
        raise ValueError("n_patients must be positive")
    weights = np.empty((n_replicates, len(patient_index)), dtype=np.float64)
    for row in weights:
        sampled = rng.integers(0, n_patients, size=n_patients)
        multiplicity = np.bincount(sampled, minlength=n_patients)
        row[:] = multiplicity[patient_index]
    return weights


def paired_patient_cluster_bootstrap(
    predictions: Mapping[str, PredictionSet],
    patient_ids: Sequence[str],
    *,
    n_bootstrap: int = 1000,
    seed: int = 20260803,
    batch_size: int = 20,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Return paired patient-cluster bootstrap CIs for all model pairs.

    The same patient resample is applied to every model in every replicate.
    This is the contract used for NoteSSM paper comparisons.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    names = list(predictions)
    if len(names) < 2:
        raise ValueError("At least two prediction sets are required")

    unique_patients, patient_index = np.unique(np.asarray(patient_ids, dtype=str), return_inverse=True)
    row_count = len(patient_index)
    computers = {}
    for name, pred in predictions.items():
        if len(pred.y_true) != row_count:
            raise ValueError(f"{name} row count does not match patient_ids")
        computers[name] = WeightedMetricComputer(pred.y_true, pred.y_prob, pred.y_pred)

    rng = np.random.default_rng(seed)
    deltas: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for left in names:
        for right in names:
            if left == right:
                continue
            for metric in metrics:
                deltas[(left, right, metric)] = []

    for start in range(0, n_bootstrap, batch_size):
        size = min(batch_size, n_bootstrap - start)
        weights = patient_cluster_weights(rng, patient_index.astype(np.int32), len(unique_patients), size)
        values = {name: computers[name].compute(weights) for name in names}
        for left in names:
            for right in names:
                if left == right:
                    continue
                for metric in metrics:
                    deltas[(left, right, metric)].append(values[left][metric] - values[right][metric])

    rows = []
    for (left, right, metric), chunks in deltas.items():
        values = np.concatenate(chunks)
        lower, upper = np.quantile(values, [0.025, 0.975])
        favorable = values > 0 if HIGHER_IS_BETTER.get(metric, True) else values < 0
        rows.append(
            {
                "left_model": left,
                "right_model": right,
                "metric": metric,
                "delta_mean": float(np.mean(values)),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "ci_excludes_zero": bool((lower > 0) or (upper < 0)),
                "probability_left_favorable": float(np.mean(favorable)),
                "n_bootstrap": int(n_bootstrap),
                "n_patients": int(len(unique_patients)),
                "resampling_unit": "patient_cluster",
                "common_resample_across_models": True,
            }
        )
    return pd.DataFrame(rows)
