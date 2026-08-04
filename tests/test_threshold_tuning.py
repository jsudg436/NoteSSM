from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tune_validation_threshold", ROOT / "scripts" / "tune_validation_threshold.py")
tuner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tuner)


def write_predictions(root: Path) -> None:
    pd.DataFrame(
        {
            "example_id": ["v0", "v1", "v2", "v3", "t0", "t1", "t2", "t3"],
            "split": ["val", "val", "val", "val", "test", "test", "test", "test"],
            "y_true": [0, 1, 0, 1, 0, 1, 0, 1],
            "y_prob": [0.1, 0.8, 0.2, 0.7, 0.15, 0.75, 0.3, 0.65],
            "y_pred": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    ).to_csv(root / "notesm_v1_cached_predictions_seed17.csv", index=False)


def test_threshold_tuning_selects_validation_threshold_and_writes_outputs() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        write_predictions(root)

        row = tuner.tune_downstream_dir(root, "notesm_v1", 17, "f1", max_candidates=20)

        assert 0.0 <= row["threshold"] <= 1.0
        assert row["val_F1"] == 1.0
        assert row["threshold"] == 0.7
        assert row["test_F1"] == 2 / 3
        assert (root / "notesm_v1_threshold_grid_seed17.csv").exists()
        assert (root / "notesm_v1_tuned_predictions_seed17.csv").exists()
        assert (root / "notesm_v1_threshold_tuned_metrics_seed17.csv").exists()
        tuned = pd.read_csv(root / "notesm_v1_tuned_predictions_seed17.csv")
        assert {"example_id", "split", "y_true", "y_prob", "y_pred_tuned", "threshold"} <= set(tuned.columns)
        assert "text" not in tuned.columns


def test_threshold_tuning_leaves_test_fields_blank_when_predictions_are_validation_only() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        pd.DataFrame(
            {
                "example_id": ["v0", "v1", "v2", "v3"],
                "split": ["val", "val", "val", "val"],
                "y_true": [0, 1, 0, 1],
                "y_prob": [0.1, 0.8, 0.2, 0.7],
                "y_pred": [0, 1, 0, 1],
            }
        ).to_csv(root / "notesm_v1_cached_predictions_seed17.csv", index=False)

        row = tuner.tune_downstream_dir(root, "notesm_v1", 17, "f1", max_candidates=20)

        assert row["val_F1"] == 1.0
        assert row["test_F1"] == ""
        assert row["test_AUPRC"] == ""
        metrics = pd.read_csv(root / "notesm_v1_threshold_tuned_metrics_seed17.csv")
        assert set(metrics["split"]) == {"val"}


def test_threshold_tuning_rejects_raw_text_predictions() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        write_predictions(root)
        predictions = pd.read_csv(root / "notesm_v1_cached_predictions_seed17.csv")
        predictions["text"] = "raw"
        predictions.to_csv(root / "notesm_v1_cached_predictions_seed17.csv", index=False)

        try:
            tuner.tune_downstream_dir(root, "notesm_v1", 17, "f1", max_candidates=20)
            raise AssertionError("expected raw text predictions to fail")
        except ValueError as exc:
            assert "forbidden raw columns" in str(exc)
