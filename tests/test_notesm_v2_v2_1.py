from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from notesm_v2.models.notesm_v2 import NoteSSMV2HybridClassifier, NoteSSMV2HybridConfig


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_downstream_script", ROOT / "scripts" / "train_downstream.py")
train_downstream = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(train_downstream)

SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_notesm_v2_v2_1_val_screen",
    ROOT / "scripts" / "summarize_notesm_v2_v2_1_val_screen.py",
)
summary_script = importlib.util.module_from_spec(SUMMARY_SPEC)
assert SUMMARY_SPEC.loader is not None
SUMMARY_SPEC.loader.exec_module(summary_script)

AGG_SPEC = importlib.util.spec_from_file_location(
    "aggregate_notesm_v2_v2_1_val_screen_seeds",
    ROOT / "scripts" / "aggregate_notesm_v2_v2_1_val_screen_seeds.py",
)
aggregate_script = importlib.util.module_from_spec(AGG_SPEC)
assert AGG_SPEC.loader is not None
AGG_SPEC.loader.exec_module(aggregate_script)

FREEZE_SPEC = importlib.util.spec_from_file_location(
    "freeze_notesm_v2_v2_1_val_screen",
    ROOT / "scripts" / "freeze_notesm_v2_v2_1_val_screen.py",
)
freeze_script = importlib.util.module_from_spec(FREEZE_SPEC)
assert FREEZE_SPEC.loader is not None
FREEZE_SPEC.loader.exec_module(freeze_script)

AUX01_SPEC = importlib.util.spec_from_file_location(
    "summarize_notesm_v2_v2_1_aux01_seed17",
    ROOT / "scripts" / "summarize_notesm_v2_v2_1_aux01_seed17.py",
)
aux01_script = importlib.util.module_from_spec(AUX01_SPEC)
assert AUX01_SPEC.loader is not None
AUX01_SPEC.loader.exec_module(aux01_script)

AUX01_3SEED_SPEC = importlib.util.spec_from_file_location(
    "summarize_notesm_v2_v2_1_aux01_3seed",
    ROOT / "scripts" / "summarize_notesm_v2_v2_1_aux01_3seed.py",
)
aux01_3seed_script = importlib.util.module_from_spec(AUX01_3SEED_SPEC)
assert AUX01_3SEED_SPEC.loader is not None
AUX01_3SEED_SPEC.loader.exec_module(aux01_3seed_script)


def make_batch(batch_size: int = 2, hidden_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "section_embeddings": torch.randn(batch_size, 3, hidden_size),
        "section_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)[:batch_size],
        "section_type_ids": torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long)[:batch_size],
        "fallback_chunk_embeddings": torch.randn(batch_size, 2, hidden_size),
        "fallback_chunk_mask": torch.tensor([[1, 0], [1, 1]], dtype=torch.long)[:batch_size],
        "use_chunk_fallback": torch.tensor([0, 1], dtype=torch.long)[:batch_size],
        "note_embedding": torch.randn(batch_size, hidden_size),
        "section_order_bucket_ids": torch.tensor([[0, 1, 0], [0, 1, 2]], dtype=torch.long)[:batch_size],
        "section_length_bucket_ids": torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long)[:batch_size],
        "occurrence_bucket_ids": torch.tensor([[0, 1, 0], [0, 1, 2]], dtype=torch.long)[:batch_size],
        "repeated_section_ids": torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.long)[:batch_size],
        "chunk_order_bucket_ids": torch.tensor([[0, 0], [0, 1]], dtype=torch.long)[:batch_size],
        "chunk_length_bucket_ids": torch.tensor([[1, 0], [1, 2]], dtype=torch.long)[:batch_size],
    }


def test_v2_1_three_view_concat_forward_shape() -> None:
    model = NoteSSMV2HybridClassifier(
        NoteSSMV2HybridConfig(
            hidden_size=4,
            note_hidden_size=4,
            hybrid_mode="three_view_concat",
            head_type="mlp",
            mlp_hidden_size=8,
            use_section_type_embedding=True,
            dropout=0.0,
        )
    )
    out = model(**make_batch())
    assert out["logits"].shape == (2,)
    assert out["z_note"].shape == (2, 4)
    assert out["z_sec"].shape == (2, 4)
    assert out["z_chunk"].shape == (2, 4)


def test_v2_1_three_view_residual_alpha_and_gradients() -> None:
    model = NoteSSMV2HybridClassifier(
        NoteSSMV2HybridConfig(
            hidden_size=4,
            note_hidden_size=4,
            hybrid_mode="three_view_residual",
            head_type="mlp",
            mlp_hidden_size=8,
            use_section_type_embedding=False,
            dropout=0.0,
        )
    )
    assert torch.allclose(model.residual_alpha.detach(), torch.tensor(0.1))
    out = model(**make_batch())
    assert out["logits"].shape == (2,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], torch.tensor([0.0, 1.0]))
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.residual_head.parameters())


def test_v2_1_set_transformer_residual_padding_mask_ignores_padded_values() -> None:
    model = NoteSSMV2HybridClassifier(
        NoteSSMV2HybridConfig(
            hidden_size=4,
            note_hidden_size=4,
            hybrid_mode="set_transformer_residual",
            head_type="mlp",
            mlp_hidden_size=8,
            use_section_type_embedding=False,
            set_transformer_heads=2,
            dropout=0.0,
        )
    )
    model.eval()
    batch = make_batch(batch_size=1)
    batch_alt = {key: value.clone() for key, value in batch.items()}
    batch["section_embeddings"][0, 2] = 1000.0
    batch_alt["section_embeddings"][0, 2] = -1000.0
    with torch.no_grad():
        first = model(**batch)["z_sec"]
        second = model(**batch_alt)["z_sec"]
    assert torch.allclose(first, second, atol=1e-5)


def test_hybrid_collator_keeps_both_views_and_marks_missing_chunk() -> None:
    items = [
        {
            "example_id": "fallback",
            "split": "train",
            "label": 1.0,
            "section_embeddings": np.ones((2, 4), dtype=np.float32),
            "section_type_ids": np.array([1, 2], dtype=np.int64),
            "section_order_bucket_ids": np.array([0, 1], dtype=np.int64),
            "section_length_bucket_ids": np.array([1, 2], dtype=np.int64),
            "occurrence_bucket_ids": np.array([0, 1], dtype=np.int64),
            "repeated_section_ids": np.array([0, 1], dtype=np.int64),
            "chunk_embeddings": np.ones((3, 4), dtype=np.float32),
            "chunk_order_bucket_ids": np.array([0, 1, 2], dtype=np.int64),
            "chunk_length_bucket_ids": np.array([1, 1, 2], dtype=np.int64),
            "use_fallback": True,
            "note_embedding": np.ones(4, dtype=np.float32),
        },
        {
            "example_id": "nonfallback",
            "split": "val",
            "label": 0.0,
            "section_embeddings": np.ones((1, 4), dtype=np.float32),
            "section_type_ids": np.array([1], dtype=np.int64),
            "chunk_embeddings": np.zeros((0, 4), dtype=np.float32),
            "use_fallback": False,
            "note_embedding": np.ones(4, dtype=np.float32),
        },
    ]
    batch = train_downstream.collate_hybrid_batch(items)
    assert batch["section_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["fallback_chunk_mask"].tolist() == [[1, 1, 1], [0, 0, 0]]
    assert batch["use_chunk_fallback"].tolist() == [1, 0]
    assert batch["chunk_available"].tolist() == [1, 0]


def test_metadata_buckets_are_deterministic() -> None:
    values = np.array([0, 1, 2, 65, 4097])
    first = train_downstream.bucket_values(values, [1, 2, 64, 4096], max_bucket=5)
    second = train_downstream.bucket_values(values, [1, 2, 64, 4096], max_bucket=5)
    assert first.tolist() == second.tolist()
    assert first.tolist() == [0, 1, 2, 3, 4]


def test_metadata_usage_summary_is_raw_text_free() -> None:
    section_meta = pd.DataFrame(
        {
            "example_id": ["a"],
            "canonical_section": ["PAST_MEDICAL_HISTORY"],
            "section_order": [0],
            "char_length": [12],
            "num_occurrences": [1],
            "used_repeated_section_merge": [False],
            "use_chunk_fallback": [False],
        }
    )
    chunk_meta = pd.DataFrame({"example_id": ["a"], "chunk_order": [0], "token_length": [128]})
    usage = train_downstream.metadata_features_used(section_meta, chunk_meta)
    assert usage["canonical_section"]
    assert usage["chunk_token_length"]
    assert not ({"text", "subject_id", "hadm_id", "note_id"} & set(usage))


def test_masked_chunk_auxiliary_loss_handles_batch_without_chunks() -> None:
    labels = torch.tensor([0.0, 1.0])
    out = {
        "logits": torch.tensor([0.1, -0.1]),
        "note_logits": torch.tensor([0.1, -0.1]),
        "section_logits": torch.tensor([0.1, -0.1]),
        "chunk_logits": torch.tensor([float("nan"), float("nan")]),
    }
    args = Namespace(loss="bce", focal_alpha=None, aux_loss_weight=0.1)
    loss, components = train_downstream.compute_downstream_loss(
        out,
        labels,
        args,
        pos_weight=None,
        chunk_available=torch.tensor([0, 0]),
    )
    assert torch.isfinite(loss)
    assert components["loss_chunk"] == 0.0


def test_note_head_checkpoint_loads_into_residual_head() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "head.pt"
        source = NoteSSMV2HybridClassifier(
            NoteSSMV2HybridConfig(hidden_size=4, note_hidden_size=4, hybrid_mode="three_view_residual", head_type="mlp", mlp_hidden_size=8)
        )
        torch.save(source.note_classifier.state_dict(), path)
        target = NoteSSMV2HybridClassifier(
            NoteSSMV2HybridConfig(hidden_size=4, note_hidden_size=4, hybrid_mode="three_view_residual", head_type="mlp", mlp_hidden_size=8)
        )
        target.load_note_head_checkpoint(path)
        for left, right in zip(source.note_classifier.parameters(), target.note_classifier.parameters(), strict=True):
            assert torch.allclose(left, right)


def test_val_only_note_training_writes_no_test_predictions_or_metrics() -> None:
    benchmark = pd.DataFrame(
        {
            "example_id": ["a", "b", "c", "d"],
            "split": ["train", "train", "val", "test"],
            "label": [0, 1, 0, 1],
        }
    )
    aligned = train_downstream.align_embedding_metadata(pd.DataFrame({"example_id": ["a", "b", "c", "d"]}), benchmark)
    embeddings = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
    args = Namespace(
        sampling="natural",
        seed=3,
        batch_size=2,
        head="linear",
        dropout=0.0,
        lr=1e-2,
        loss="bce",
        focal_alpha=None,
        epochs=1,
        model="notesm_v1",
        eval_splits="val",
        skip_test=False,
        device="cpu",
    )
    with TemporaryDirectory() as tmpdir:
        output = Path(tmpdir)
        train_downstream.train_note_embedding_head(embeddings, aligned, args, output)
        predictions = pd.read_csv(output / "notesm_v1_cached_predictions_seed3.csv")
        metrics = pd.read_csv(output / "notesm_v1_cached_embedding_metrics_seed3.csv")
    assert set(predictions["split"]) == {"val"}
    assert set(metrics["split"]) == {"val"}


def test_v2_1_summary_supports_concat_only_mode() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        mode_dir = root / "three_view_concat"
        mode_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "example_id": ["a", "b", "c", "d"],
                "split": ["val", "val", "val", "val"],
                "y_true": [0, 1, 0, 1],
                "y_prob": [0.1, 0.8, 0.2, 0.7],
                "y_pred": [0, 1, 0, 1],
            }
        ).to_csv(mode_dir / "notesm_v2_hybrid_cached_predictions_seed23.csv", index=False)
        pd.DataFrame({"runtime_seconds": [12.0], "peak_gpu_memory_mb": [42.0]}).to_csv(
            mode_dir / "notesm_v2_hybrid_v2_1_training_summary_seed23.csv",
            index=False,
        )

        row = summary_script.summarize_prediction_dir(mode_dir, "three_view_concat", 23)
        usage = summary_script.metadata_usage(root, 23, ["three_view_concat"])

    assert row["setting"] == "three_view_concat"
    assert row["split"] == "val"
    assert row["n"] == 4
    assert usage["setting"].tolist() == ["three_view_concat"]


def test_v2_1_seed_aggregate_computes_mean_and_std() -> None:
    rows = []
    for seed, auprc in [(17, 0.30), (23, 0.32), (42, 0.34)]:
        rows.append(
            {
                "setting": "three_view_concat",
                "split": "val",
                "seed": seed,
                "n": 10,
                "positives": 2,
                "AUPRC": auprc,
                "AUROC": 0.60 + auprc,
                "Precision": 0.2,
                "Recall": 0.7,
                "F1": 0.31,
                "Brier": 0.15,
                "ECE": 0.02,
                "threshold": 0.4,
            }
        )
    result = aggregate_script.aggregate(pd.DataFrame(rows))

    row = result.iloc[0]
    assert row["setting"] == "three_view_concat"
    assert row["n_seeds"] == 3
    assert abs(row["AUPRC_mean"] - 0.32) < 1e-12
    assert row["AUPRC_std"] > 0.0


def test_v2_1_freeze_validation_checks_seed_coverage_and_aggregate() -> None:
    per_seed = pd.DataFrame(
        [
            {
                "setting": "three_view_concat",
                "split": "val",
                "seed": seed,
                "n": 10,
                "positives": 2,
                "AUPRC": 0.3,
                "AUROC": 0.6,
                "Precision": 0.2,
                "Recall": 0.7,
                "F1": 0.31,
                "Brier": 0.15,
                "ECE": 0.02,
                "threshold": 0.4,
            }
            for seed in [17, 23, 42]
        ]
    )
    aggregate = aggregate_script.aggregate(per_seed)

    freeze_script.validate_per_seed(per_seed, "three_view_concat", [17, 23, 42])
    row = freeze_script.validate_aggregate(aggregate, "three_view_concat", [17, 23, 42])

    assert int(row["n_seeds"]) == 3


def test_v2_1_freeze_validation_rejects_missing_seed() -> None:
    per_seed = pd.DataFrame(
        {
            "setting": ["three_view_concat", "three_view_concat"],
            "split": ["val", "val"],
            "seed": [17, 23],
            "n": [10, 10],
            "positives": [2, 2],
            "AUPRC": [0.3, 0.31],
            "AUROC": [0.6, 0.61],
            "Precision": [0.2, 0.2],
            "Recall": [0.7, 0.7],
            "F1": [0.31, 0.31],
            "Brier": [0.15, 0.15],
            "ECE": [0.02, 0.02],
            "threshold": [0.4, 0.4],
        }
    )
    try:
        freeze_script.validate_per_seed(per_seed, "three_view_concat", [17, 23, 42])
        raise AssertionError("expected incomplete seed coverage to fail")
    except ValueError as exc:
        assert "expected" in str(exc)


def test_v2_1_aux01_summary_compares_against_aux00_references() -> None:
    aux01 = pd.Series(
        {
            "setting": "three_view_concat",
            "split": "val",
            "seed": 17,
            "n": 10,
            "positives": 2,
            "AUPRC": 0.34,
            "AUROC": 0.68,
            "Precision": 0.3,
            "Recall": 0.7,
            "F1": 0.42,
            "Brier": 0.14,
            "ECE": 0.01,
            "threshold": 0.2,
        }
    )
    aux00_seed17 = aux01.copy()
    aux00_seed17["AUPRC"] = 0.32
    aux00_seed17["AUROC"] = 0.66
    aux00_seed17["F1"] = 0.40
    aux00_locked = pd.Series(
        {
            "setting": "three_view_concat",
            "split": "val",
            "n": 10,
            "positives": 2,
            "AUPRC_mean": 0.321,
            "AUROC_mean": 0.661,
            "Precision_mean": 0.29,
            "Recall_mean": 0.62,
            "F1_mean": 0.401,
            "Brier_mean": 0.145,
            "ECE_mean": 0.016,
            "threshold_mean": 0.18,
        }
    )

    main, delta = aux01_script.build_tables(aux01, aux00_seed17, aux00_locked)

    assert len(main) == 3
    assert len(delta) == 2
    assert float(delta.iloc[0]["AUPRC_delta"]) > 0


def test_v2_1_aux01_3seed_summary_compares_against_locked_aux00() -> None:
    aux01 = pd.Series(
        {
            "setting": "three_view_concat",
            "split": "val",
            "seeds": "17 23 42",
            "n_seeds": 3,
            "n_min": 10,
            "positives_min": 2,
            "AUPRC_mean": 0.325,
            "AUPRC_std": 0.001,
            "AUROC_mean": 0.683,
            "AUROC_std": 0.001,
            "Precision_mean": 0.30,
            "Precision_std": 0.01,
            "Recall_mean": 0.62,
            "Recall_std": 0.02,
            "F1_mean": 0.402,
            "F1_std": 0.001,
            "Brier_mean": 0.145,
            "Brier_std": 0.001,
            "ECE_mean": 0.016,
            "ECE_std": 0.004,
            "threshold_mean": 0.18,
            "threshold_std": 0.01,
        }
    )
    aux00 = pd.Series(
        {
            "setting": "three_view_concat",
            "split": "val",
            "seeds": "17 23 42",
            "n_seeds": 3,
            "n": 10,
            "positives": 2,
            "AUPRC_mean": 0.322,
            "AUPRC_std": 0.001,
            "AUROC_mean": 0.682,
            "AUROC_std": 0.001,
            "Precision_mean": 0.29,
            "Precision_std": 0.01,
            "Recall_mean": 0.62,
            "Recall_std": 0.02,
            "F1_mean": 0.401,
            "F1_std": 0.001,
            "Brier_mean": 0.145,
            "Brier_std": 0.001,
            "ECE_mean": 0.016,
            "ECE_std": 0.004,
            "threshold_mean": 0.18,
            "threshold_std": 0.01,
        }
    )

    main, delta = aux01_3seed_script.build_tables(aux01, aux00)

    assert len(main) == 2
    assert float(delta.iloc[0]["AUPRC_mean_delta"]) > 0
