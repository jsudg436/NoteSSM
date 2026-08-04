from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_no_dapt_full_comparison",
    ROOT / "scripts" / "freeze_no_dapt_full_comparison.py",
)
freezer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = freezer
SPEC.loader.exec_module(freezer)

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ModuleNotFoundError:
    average_precision_score = None
    roc_auc_score = None


def write_predictions(path: Path, thresholds: list[float] | None = None) -> None:
    thresholds = thresholds or [0.5, 0.5, 0.5, 0.5]
    pd.DataFrame(
        {
            "example_id": ["v0", "v1", "t0", "t1"],
            "split": ["val", "val", "test", "test"],
            "y_true": [0, 1, 0, 1],
            "y_prob": [0.2, 0.8, 0.3, 0.7],
            "y_pred_tuned": [0, 1, 0, 1],
            "threshold": thresholds,
        }
    ).to_csv(path, index=False)


def test_unweighted_auprc_auroc_match_sklearn_values() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8])
    prepared = freezer.PreparedPredictions.from_arrays(y_true, y_prob, np.array([0, 0, 0, 1]))
    weights = np.ones(len(y_true), dtype=float)

    expected_auprc = 5.0 / 6.0
    expected_auroc = 0.75
    if average_precision_score is not None and roc_auc_score is not None:
        expected_auprc = float(average_precision_score(y_true, y_prob))
        expected_auroc = float(roc_auc_score(y_true, y_prob))

    assert np.isclose(freezer.weighted_auprc(prepared, weights), expected_auprc)
    assert np.isclose(freezer.weighted_auroc(prepared, weights), expected_auroc)


def test_tied_score_auprc_auroc_match_sklearn_expected_values() -> None:
    y_true = np.array([0, 1])
    y_prob = np.array([0.5, 0.5])
    prepared = freezer.PreparedPredictions.from_arrays(y_true, y_prob, np.array([0, 0]))
    weights = np.ones(len(y_true), dtype=float)

    expected_auprc = 0.5
    expected_auroc = 0.5
    if average_precision_score is not None and roc_auc_score is not None:
        expected_auprc = float(average_precision_score(y_true, y_prob))
        expected_auroc = float(roc_auc_score(y_true, y_prob))

    assert np.isclose(freezer.weighted_auprc(prepared, weights), expected_auprc)
    assert np.isclose(freezer.weighted_auroc(prepared, weights), expected_auroc)


def test_load_predictions_validates_counts_and_constant_threshold() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        prediction_path = root / "notesm_v1_tuned_predictions_seed17.csv"
        write_predictions(prediction_path)
        spec = freezer.ModelSpec("debug", "Debug", "debug", "note", 128, "mean", root)

        frame = freezer.load_predictions(spec, 17, {"val": 2, "test": 2})

        assert len(frame) == 4
        assert freezer.constant_threshold(frame[frame["split"] == "val"], "debug", 17, "val") == 0.5

        write_predictions(prediction_path, thresholds=[0.4, 0.6, 0.5, 0.5])
        try:
            freezer.load_predictions(spec, 17, {"val": 2, "test": 2})
            raise AssertionError("expected non-constant validation threshold to fail")
        except ValueError as exc:
            assert "exactly one threshold" in str(exc)

        write_predictions(prediction_path)
        try:
            freezer.load_predictions(spec, 17, {"val": 3, "test": 2})
            raise AssertionError("expected split count mismatch to fail")
        except ValueError as exc:
            assert "expected 3" in str(exc)


def test_align_pair_uses_example_id_and_split() -> None:
    left = pd.DataFrame(
        {
            "example_id": ["same"],
            "split": ["test"],
            "y_true": [1],
            "y_prob": [0.8],
            "y_pred_tuned": [1],
        }
    )
    right = pd.DataFrame(
        {
            "example_id": ["same"],
            "split": ["val"],
            "y_true": [1],
            "y_prob": [0.7],
            "y_pred_tuned": [1],
        }
    )

    try:
        freezer.align_pair(left, right)
        raise AssertionError("expected mismatched split alignment to fail")
    except ValueError as exc:
        assert "example_id/split mismatch" in str(exc)


def test_frozen_markdown_documents_delta_direction(tmp_path: Path) -> None:
    main = pd.DataFrame(
        {
            "model_key": ["longformer_mean_4096"],
            "display_name": ["Longformer"],
            "role": ["Comparator"],
            "input": ["discharge summary"],
            "max_length": [4096],
            "pooling": ["mean"],
            "head": ["MLP"],
            "loss": ["natural_bce"],
            "split": ["test"],
            "AUPRC_mean": [0.3],
            "AUPRC_std": [0.01],
            "AUROC_mean": [0.6],
            "AUROC_std": [0.01],
            "F1_mean": [0.4],
            "F1_std": [0.01],
            "Precision_mean": [0.3],
            "Recall_mean": [0.6],
            "Brier_mean": [0.15],
            "ECE_mean": [0.02],
        }
    )
    bootstrap = pd.DataFrame(
        {
            "left_model": ["longformer_mean_4096", "longformer_mean_4096", "longformer_mean_4096"],
            "right_model": ["clinicalmamba_4096", "clinicalmamba_4096", "clinicalmamba_4096"],
            "metric": ["AUPRC", "AUROC", "F1"],
            "observed_delta_left_minus_right": [0.0, 0.1, 0.1],
            "ci_low": [-0.1, 0.01, 0.01],
            "ci_high": [0.1, 0.2, 0.2],
            "ci_excludes_zero": [False, True, True],
        }
    )
    output = tmp_path / "NO_DAPT_FULL_COMPARISON_FROZEN.md"

    freezer.write_markdown(main, bootstrap, output)
    text = output.read_text(encoding="utf-8")

    assert "positive AUPRC/AUROC/F1/Precision/Recall deltas favor the left model" in text
    assert "negative Brier/ECE deltas favor the left model" in text
    assert "AUPRC CI overlaps zero" in text
