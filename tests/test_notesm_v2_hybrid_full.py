from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hybrid_freezer", ROOT / "scripts" / "freeze_notesm_v2_hybrid_concat_full.py")
hybrid_freezer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = hybrid_freezer
SPEC.loader.exec_module(hybrid_freezer)


def write_prediction_dir(
    directory: Path,
    model: str,
    seed: int,
    y_prob: list[float],
    y_pred: list[int] | None = None,
    y_true_offset: int = 0,
    raw_text: bool = False,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    y_true = np.array([1, 0, 1, 0], dtype=int)
    if y_true_offset:
        y_true = (y_true + y_true_offset) % 2
    frame = pd.DataFrame(
        {
            "example_id": ["v_fb", "v_nf", "t_fb", "t_nf"],
            "split": ["val", "val", "test", "test"],
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred_tuned": y_pred if y_pred is not None else [1, 0, 1, 0],
            "threshold": [0.5, 0.5, 0.5, 0.5],
        }
    )
    if raw_text:
        frame["text"] = "raw"
    frame.to_csv(directory / f"{model}_tuned_predictions_seed{seed}.csv", index=False)
    pd.DataFrame({"split": ["val", "test"], "F1": [1.0, 1.0]}).to_csv(
        directory / f"{model}_threshold_tuned_metrics_seed{seed}.csv",
        index=False,
    )
    pd.DataFrame({"seed": [seed], "threshold": [0.5]}).to_csv(
        directory / f"{model}_threshold_tuning_summary_seed{seed}.csv",
        index=False,
    )


def fallback_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["v_fb", "v_nf", "t_fb", "t_nf"],
            "split": ["val", "val", "test", "test"],
            "use_chunk_fallback": [True, False, True, False],
            "fallback_group": ["fallback", "non_fallback", "fallback", "non_fallback"],
        }
    )


def test_canonical_seed17_copy_does_not_overwrite_different_files(tmp_path: Path) -> None:
    source = tmp_path / "seed17"
    target = tmp_path / "canonical"
    write_prediction_dir(source, "notesm_v2_hybrid", 17, [0.9, 0.1, 0.8, 0.2])
    copied = hybrid_freezer.safe_copy_seed_artifacts(source, target, 17)
    assert target.joinpath("notesm_v2_hybrid_tuned_predictions_seed17.csv").exists()
    assert copied

    target.joinpath("notesm_v2_hybrid_tuned_predictions_seed17.csv").write_text("different\n", encoding="utf-8")
    try:
        hybrid_freezer.safe_copy_seed_artifacts(source, target, 17)
        raise AssertionError("expected different existing artifact to be protected")
    except FileExistsError:
        pass


def test_load_predictions_rejects_raw_text_columns(tmp_path: Path) -> None:
    directory = tmp_path / "preds"
    write_prediction_dir(directory, "notesm_v2_hybrid", 17, [0.9, 0.1, 0.8, 0.2], raw_text=True)

    try:
        hybrid_freezer.load_predictions(directory, "notesm_v2_hybrid", "concat_mlp", 17, fallback_frame(), {"val": 2, "test": 2})
        raise AssertionError("expected raw text prediction column to fail")
    except ValueError as exc:
        assert "forbidden raw columns" in str(exc)


def test_load_predictions_rejects_missing_val_test_examples(tmp_path: Path) -> None:
    directory = tmp_path / "preds"
    write_prediction_dir(directory, "notesm_v2_hybrid", 17, [0.9, 0.1, 0.8, 0.2])

    try:
        hybrid_freezer.load_predictions(directory, "notesm_v2_hybrid", "concat_mlp", 17, fallback_frame(), {"val": 2, "test": 3})
        raise AssertionError("expected count mismatch to fail")
    except ValueError as exc:
        assert "expected 3" in str(exc)


def test_pairwise_consistency_rejects_y_true_mismatch(tmp_path: Path) -> None:
    concat = tmp_path / "concat"
    current = tmp_path / "current"
    dapt = tmp_path / "dapt"
    write_prediction_dir(concat, "notesm_v2_hybrid", 17, [0.9, 0.1, 0.8, 0.2])
    write_prediction_dir(current, "notesm_v2", 17, [0.8, 0.2, 0.7, 0.3], y_true_offset=1)
    write_prediction_dir(dapt, "notesm_v1", 17, [0.7, 0.3, 0.6, 0.4])
    fallback = fallback_frame()
    predictions = {
        ("concat_mlp", 17): hybrid_freezer.load_predictions(concat, "notesm_v2_hybrid", "concat_mlp", 17, fallback, {"val": 2, "test": 2}),
        ("current_section_aware", 17): hybrid_freezer.load_predictions(current, "notesm_v2", "current_section_aware", 17, fallback, {"val": 2, "test": 2}),
        ("dapt_lasttoken", 17): hybrid_freezer.load_predictions(dapt, "notesm_v1", "dapt_lasttoken", 17, fallback, {"val": 2, "test": 2}),
    }

    try:
        hybrid_freezer.enforce_pairwise_consistency(predictions, [17])
        raise AssertionError("expected y_true mismatch to fail")
    except ValueError as exc:
        assert "y_true mismatch" in str(exc)


def test_group_summary_separates_fallback_and_nonfallback(tmp_path: Path) -> None:
    directories = {}
    for setting, model, probs in [
        ("concat_mlp", "notesm_v2_hybrid", [0.9, 0.1, 0.8, 0.2]),
        ("current_section_aware", "notesm_v2", [0.8, 0.2, 0.7, 0.3]),
        ("dapt_lasttoken", "notesm_v1", [0.7, 0.3, 0.6, 0.4]),
    ]:
        directories[setting] = tmp_path / setting
        write_prediction_dir(directories[setting], model, 17, probs)
    fallback = fallback_frame()
    predictions = {
        (setting, 17): hybrid_freezer.load_predictions(directory, model, setting, 17, fallback, {"val": 2, "test": 2})
        for setting, directory in directories.items()
        for model in [hybrid_freezer.MODEL_SPECS[setting][0]]
    }
    _, groups = hybrid_freezer.per_seed_tables(predictions, [17])

    test_groups = groups[groups["split"] == "test"]
    assert set(test_groups["fallback_group"]) == {"fallback", "non_fallback"}
    assert set(test_groups["n"].astype(int)) == {1}


def test_bootstrap_pairing_uses_example_id_and_split(tmp_path: Path) -> None:
    fallback = fallback_frame()
    concat = tmp_path / "concat"
    current = tmp_path / "current"
    dapt = tmp_path / "dapt"
    write_prediction_dir(concat, "notesm_v2_hybrid", 17, [0.9, 0.1, 0.8, 0.2])
    write_prediction_dir(current, "notesm_v2", 17, [0.6, 0.4, 0.6, 0.4])
    write_prediction_dir(dapt, "notesm_v1", 17, [0.6, 0.4, 0.6, 0.4])
    predictions = {
        ("concat_mlp", 17): hybrid_freezer.load_predictions(concat, "notesm_v2_hybrid", "concat_mlp", 17, fallback, {"val": 2, "test": 2}),
        ("current_section_aware", 17): hybrid_freezer.load_predictions(current, "notesm_v2", "current_section_aware", 17, fallback, {"val": 2, "test": 2}),
        ("dapt_lasttoken", 17): hybrid_freezer.load_predictions(dapt, "notesm_v1", "dapt_lasttoken", 17, fallback, {"val": 2, "test": 2}),
    }
    boot = hybrid_freezer.paired_bootstrap(predictions, "current_section_aware", [17], n_bootstrap=5, rng_seed=1)

    assert set(boot["comparison"]) == {"concat_mlp_minus_current_section_aware"}
    assert {"AUPRC", "AUROC", "F1"} <= set(boot["metric"])


def test_decision_rule_classifies_pass_and_diagnostic() -> None:
    aggregate = pd.DataFrame(
        [
            {"setting": "concat_mlp", "display_name": "Hybrid", "split": "test", "fallback_group": "all", "AUPRC_mean": 0.31, "AUROC_mean": 0.66, "F1_mean": 0.39, "Brier_mean": 0.14, "ECE_mean": 0.01, "Recall_mean": 0.5},
            {"setting": "current_section_aware", "display_name": "Current", "split": "test", "fallback_group": "all", "AUPRC_mean": 0.30, "AUROC_mean": 0.65, "F1_mean": 0.39, "Brier_mean": 0.14, "ECE_mean": 0.01, "Recall_mean": 0.5},
            {"setting": "concat_mlp", "display_name": "Hybrid", "split": "test", "fallback_group": "non_fallback", "AUPRC_mean": 0.20, "AUROC_mean": 0.60, "F1_mean": 0.20, "Brier_mean": 0.10, "ECE_mean": 0.01, "Recall_mean": 0.50},
            {"setting": "current_section_aware", "display_name": "Current", "split": "test", "fallback_group": "non_fallback", "AUPRC_mean": 0.10, "AUROC_mean": 0.50, "F1_mean": 0.10, "Brier_mean": 0.11, "ECE_mean": 0.01, "Recall_mean": 0.30},
            {"setting": "concat_mlp", "display_name": "Hybrid", "split": "test", "fallback_group": "fallback", "AUPRC_mean": 0.32, "AUROC_mean": 0.67, "F1_mean": 0.40, "Brier_mean": 0.15, "ECE_mean": 0.01, "Recall_mean": 0.50},
            {"setting": "current_section_aware", "display_name": "Current", "split": "test", "fallback_group": "fallback", "AUPRC_mean": 0.32, "AUROC_mean": 0.67, "F1_mean": 0.40, "Brier_mean": 0.15, "ECE_mean": 0.01, "Recall_mean": 0.50},
        ]
    )
    bootstrap = pd.DataFrame(
        {
            "comparison": ["concat_mlp_minus_current_section_aware"],
            "metric": ["AUPRC"],
            "ci_lower": [0.001],
        }
    )

    decision = hybrid_freezer.decision_table(aggregate, bootstrap)
    assert decision["decision_category"].iloc[0] == "confirmed_hybrid_candidate"

    no_bootstrap = bootstrap.copy()
    no_bootstrap["ci_lower"] = -0.001
    decision = hybrid_freezer.decision_table(aggregate, no_bootstrap)
    assert decision["decision_category"].iloc[0] == "diagnostic_only"
