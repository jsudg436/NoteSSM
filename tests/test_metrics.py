from __future__ import annotations

import numpy as np

from notesm_v2.evaluation.metrics import auprc_score, auroc_score, expected_calibration_error, risk_deciles


def test_ece_is_zero_for_perfect_bin_confidence() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y_true, y_prob, n_bins=2) == 0.0


def test_risk_decile_output_sums_to_sample_count() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    rows = risk_deciles(y_true, y_prob, n_bins=5)
    assert sum(int(row["n"]) for row in rows) == 5
    assert sum(row["sensitivity"] for row in rows) == 1.0


def test_tied_scores_match_sklearn_expected_values() -> None:
    y_true = np.array([0, 1])
    y_prob = np.array([0.5, 0.5])
    assert auroc_score(y_true, y_prob) == 0.5
    assert auprc_score(y_true, y_prob) == 0.5
