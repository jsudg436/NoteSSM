from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_downstream_script", ROOT / "scripts" / "train_downstream.py")
train_downstream = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(train_downstream)


def test_resolve_device_auto_uses_cpu_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(train_downstream.torch.cuda, "is_available", lambda: False)
    assert train_downstream.resolve_device("auto").type == "cpu"


def test_resolve_device_cuda_fails_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(train_downstream.torch.cuda, "is_available", lambda: False)
    try:
        train_downstream.resolve_device("cuda")
        raise AssertionError("expected unavailable CUDA request to fail")
    except ValueError as exc:
        assert "CUDA is not available" in str(exc)


def test_move_batch_to_device_moves_only_tensors() -> None:
    batch = {"x": torch.tensor([1.0]), "example_ids": ["a"]}
    moved = train_downstream.move_batch_to_device(batch, torch.device("cpu"))
    assert moved["x"].device.type == "cpu"
    assert moved["example_ids"] == ["a"]


def benchmark_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["a", "b", "c", "d", "e", "f"],
            "split": ["train", "train", "train", "train", "val", "test"],
            "label": [0, 1, 0, 1, 0, 1],
        }
    )


def test_align_embedding_metadata_uses_benchmark_without_metadata_labels() -> None:
    metadata = pd.DataFrame({"example_id": ["a", "b", "e"], "row_index": [0, 1, 2]})
    aligned = train_downstream.align_embedding_metadata(metadata, benchmark_index())
    assert aligned["label"].tolist() == [0, 1, 0]
    assert aligned["split"].tolist() == ["train", "train", "val"]


def test_align_embedding_metadata_checks_optional_metadata_labels() -> None:
    metadata = pd.DataFrame({"example_id": ["a"], "label": [1], "split": ["train"]})
    try:
        train_downstream.align_embedding_metadata(metadata, benchmark_index())
        raise AssertionError("expected label mismatch to fail")
    except ValueError as exc:
        assert "label values disagree" in str(exc)


def test_cached_note_training_writes_text_free_predictions() -> None:
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        metadata = train_downstream.align_embedding_metadata(pd.DataFrame({"example_id": list("abcdef")}), benchmark_index())
        embeddings = np.arange(6 * 4, dtype=np.float32).reshape(6, 4) / 10.0
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
            device="cpu",
        )
        train_downstream.train_note_embedding_head(embeddings, metadata, args, output_dir)
        predictions = pd.read_csv(output_dir / "notesm_v1_cached_predictions_seed3.csv")
        assert {"example_id", "split", "y_true", "y_prob", "y_pred"} <= set(predictions.columns)
        assert "text" not in predictions.columns
        assert set(predictions["split"]) == {"val", "test"}


def test_load_note_embedding_cache_reads_split_subdirectory_shards() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for split, example_id, value in [("train", "a", 1.0), ("val", "e", 2.0), ("test", "f", 3.0)]:
            split_dir = root / split
            split_dir.mkdir()
            np.save(split_dir / "debug_embeddings_shard_000.npy", np.full((1, 4), value, dtype=np.float32))
            pd.DataFrame(
                {
                    "example_id": [example_id],
                    "split": [split],
                    "shard_index": [0],
                    "row_index_in_shard": [0],
                    "global_row_index": [0],
                    "row_index": [0],
                }
            ).to_parquet(split_dir / "debug_metadata_shard_000.parquet", index=False)

        embeddings, metadata = train_downstream.load_note_embedding_cache(root, "debug")

        assert embeddings.shape == (3, 4)
        assert metadata["example_id"].tolist() == ["f", "a", "e"] or set(metadata["example_id"]) == {"a", "e", "f"}
        assert set(metadata["split"]) == {"train", "val", "test"}
        assert metadata["row_index"].tolist() == [0, 1, 2]


def test_section_and_chunk_loaders_accept_custom_shard_prefixes() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        np.save(root / "section_span2_embeddings_shard_000.npy", np.ones((2, 4), dtype=np.float32))
        pd.DataFrame(
            {
                "example_id": ["a", "b"],
                "row_index": [10, 11],
                "canonical_section": ["HPI", "PMH"],
                "section_order": [0, 1],
            }
        ).to_parquet(root / "section_span2_embedding_metadata_shard_000.parquet", index=False)
        np.save(root / "chunk_span2_embeddings_shard_000.npy", np.ones((1, 4), dtype=np.float32))
        pd.DataFrame({"example_id": ["b"], "row_index": [7], "chunk_order": [0]}).to_parquet(
            root / "chunk_span2_embedding_metadata_shard_000.parquet",
            index=False,
        )

        section_embeddings, section_meta = train_downstream.load_section_embeddings(root, prefix="section_span2")
        chunk_embeddings, chunk_meta = train_downstream.load_chunk_embeddings(root, prefix="chunk_span2")

        assert section_embeddings.shape == (2, 4)
        assert section_meta["row_index"].tolist() == [0, 1]
        assert chunk_embeddings is not None
        assert chunk_meta is not None
        assert chunk_embeddings.shape == (1, 4)
        assert chunk_meta["row_index"].tolist() == [0]


def test_validation_only_benchmark_omits_test_split() -> None:
    assert train_downstream.benchmark_splits_for_training(("val",)) == ("train", "val")


def test_joint_abcd_seed17_script_uses_locked_prefix_grid_and_validation_only() -> None:
    script = (ROOT / "scripts" / "run_dapt_mamba3_joint_abcd_seed17_val.sh").read_text(encoding="utf-8")
    assert "run_arm A section chunk" in script
    assert "run_arm B section_span2 chunk" in script
    assert "run_arm C section chunk_span2" in script
    assert "run_arm D section_span2 chunk_span2" in script
    assert "--section-embedding-prefix \"${section_prefix}\"" in script
    assert "--chunk-embedding-prefix \"${chunk_prefix}\"" in script
    assert "--eval-splits val" in script
    assert "--skip-test" in script
    assert "--architecture-screening-only" in script
    assert "SEED=17" in script
    assert "dapt1epoch_mamba3_siso187_note_full_len2048" in script
    assert "validate_mamba3_joint_s1_s2_artifacts.py" in script
    assert "joint_abcd_seed17_winner.txt" in script


def test_notesm_v2_fails_when_fallback_cache_missing() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        np.save(root / "section_embeddings.npy", np.ones((1, 4), dtype=np.float32))
        pd.DataFrame(
            {
                "example_id": ["a"],
                "row_index": [0],
                "canonical_section": ["PAST_MEDICAL_HISTORY"],
                "section_order": [0],
                "use_chunk_fallback": [True],
            }
        ).to_parquet(root / "section_embedding_metadata.parquet", index=False)
        args = Namespace(
            sampling="natural",
            seed=3,
            batch_size=1,
            dropout=0.0,
            lr=1e-2,
            loss="bce",
            focal_alpha=None,
            epochs=1,
            model="notesm_v2",
            device="cpu",
        )
        try:
            train_downstream.train_section_embedding_head(root, benchmark_index(), args, root)
            raise AssertionError("expected missing chunk fallback cache to fail")
        except ValueError as exc:
            assert "chunk embedding cache is missing" in str(exc)


def test_notesm_v2_config_uses_downstream_head_and_structure_args() -> None:
    args = Namespace(
        dropout=0.2,
        head="mlp",
        mlp_hidden_size=12,
        notesm_v2_branch_mode="branch_indicator",
        use_section_type_embedding=False,
        section_pooling="mean",
        fallback_pooling="max",
    )
    config = train_downstream.build_notesm_v2_config(hidden_size=8, num_section_types=5, args=args)
    assert config.head_type == "mlp"
    assert config.mlp_hidden_size == 12
    assert config.branch_mode == "branch_indicator"
    assert not config.use_section_type_embedding
    assert config.section_pooling == "mean"
    assert config.fallback_pooling == "max"


def write_hybrid_artifacts(root: Path, note_root: Path, include_note_f: bool = True, raw_note_column: bool = False) -> None:
    section_examples = ["a", "b", "e", "f"]
    np.save(root / "section_embeddings.npy", np.arange(4 * 4, dtype=np.float32).reshape(4, 4) / 10.0)
    pd.DataFrame(
        {
            "example_id": section_examples,
            "row_index": [0, 1, 2, 3],
            "canonical_section": ["PAST_MEDICAL_HISTORY"] * 4,
            "section_order": [0, 0, 0, 0],
            "use_chunk_fallback": [False, True, False, False],
        }
    ).to_parquet(root / "section_embedding_metadata.parquet", index=False)
    np.save(root / "chunk_embeddings.npy", np.ones((1, 4), dtype=np.float32))
    pd.DataFrame({"example_id": ["b"], "row_index": [0], "chunk_order": [0]}).to_parquet(
        root / "chunk_embedding_metadata.parquet",
        index=False,
    )
    note_examples = ["a", "b", "e", "f"] if include_note_f else ["a", "b", "e"]
    np.save(note_root / "debug.npy", np.ones((len(note_examples), 4), dtype=np.float32))
    metadata = pd.DataFrame({"example_id": note_examples, "row_index": list(range(len(note_examples)))})
    if raw_note_column:
        metadata["text"] = ["raw"] * len(metadata)
    metadata.to_parquet(note_root / "debug_metadata.parquet", index=False)


def hybrid_args(**overrides) -> Namespace:
    values = dict(
        sampling="natural",
        seed=3,
        batch_size=2,
        head="mlp",
        mlp_hidden_size=8,
        hybrid_mode="hard_routed_mlp",
        section_pooling="mean",
        fallback_pooling="mean",
        use_section_type_embedding=False,
        use_fallback_indicator=True,
        dropout=0.0,
        lr=1e-2,
        loss="bce",
        focal_alpha=None,
        epochs=1,
        model="notesm_v2_hybrid",
        device="cpu",
    )
    values.update(overrides)
    return Namespace(**values)


def test_hybrid_training_aligns_section_and_note_embeddings_and_writes_predictions() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        section_root = root / "section"
        note_root = root / "note"
        section_root.mkdir()
        note_root.mkdir()
        write_hybrid_artifacts(section_root, note_root)

        train_downstream.train_hybrid_embedding_head(section_root, note_root, "debug", benchmark_index(), hybrid_args(), root)

        predictions = pd.read_csv(root / "notesm_v2_hybrid_cached_predictions_seed3.csv")
        assert {"example_id", "split", "y_true", "y_prob", "y_pred"} <= set(predictions.columns)
        assert "text" not in predictions.columns
        assert set(predictions["split"]) == {"val", "test"}


def test_hybrid_training_can_load_chunks_from_separate_cache_dir() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        section_root = root / "section"
        chunk_root = root / "chunk"
        note_root = root / "note"
        section_root.mkdir()
        chunk_root.mkdir()
        note_root.mkdir()
        write_hybrid_artifacts(section_root, note_root)
        (chunk_root / "chunk_embeddings.npy").write_bytes((section_root / "chunk_embeddings.npy").read_bytes())
        pd.read_parquet(section_root / "chunk_embedding_metadata.parquet").to_parquet(
            chunk_root / "chunk_embedding_metadata.parquet",
            index=False,
        )
        (section_root / "chunk_embeddings.npy").unlink()
        (section_root / "chunk_embedding_metadata.parquet").unlink()

        args = hybrid_args(chunk_embedding_dir=str(chunk_root))
        train_downstream.train_hybrid_embedding_head(section_root, note_root, "debug", benchmark_index(), args, root)

        predictions = pd.read_csv(root / "notesm_v2_hybrid_cached_predictions_seed3.csv")
        assert set(predictions["split"]) == {"val", "test"}


def test_hybrid_validation_only_filters_all_split_section_note_and_chunk_caches() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        section_root = root / "section"
        chunk_root = root / "chunk"
        note_root = root / "note"
        section_root.mkdir()
        chunk_root.mkdir()
        note_root.mkdir()
        write_hybrid_artifacts(section_root, note_root)
        (chunk_root / "chunk_embeddings.npy").write_bytes((section_root / "chunk_embeddings.npy").read_bytes())
        pd.read_parquet(section_root / "chunk_embedding_metadata.parquet").to_parquet(
            chunk_root / "chunk_embedding_metadata.parquet",
            index=False,
        )
        (section_root / "chunk_embeddings.npy").unlink()
        (section_root / "chunk_embedding_metadata.parquet").unlink()

        benchmark_train_val = benchmark_index()[lambda frame: frame["split"].isin(["train", "val"])].copy()
        args = hybrid_args(chunk_embedding_dir=str(chunk_root), eval_splits="val")
        train_downstream.train_hybrid_embedding_head(section_root, note_root, "debug", benchmark_train_val, args, root)

        predictions = pd.read_csv(root / "notesm_v2_hybrid_cached_predictions_seed3.csv")
        assert set(predictions["split"]) == {"val"}
        assert set(predictions["example_id"]) == {"e"}


def test_hybrid_training_fails_when_note_embedding_missing() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        section_root = root / "section"
        note_root = root / "note"
        section_root.mkdir()
        note_root.mkdir()
        write_hybrid_artifacts(section_root, note_root, include_note_f=False)

        try:
            train_downstream.train_hybrid_embedding_head(section_root, note_root, "debug", benchmark_index(), hybrid_args(), root)
            raise AssertionError("expected missing note modality to fail")
        except ValueError as exc:
            assert "missing_note" in str(exc)


def test_hybrid_training_fails_on_raw_text_note_metadata() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        section_root = root / "section"
        note_root = root / "note"
        section_root.mkdir()
        note_root.mkdir()
        write_hybrid_artifacts(section_root, note_root, raw_note_column=True)

        try:
            train_downstream.train_hybrid_embedding_head(section_root, note_root, "debug", benchmark_index(), hybrid_args(), root)
            raise AssertionError("expected raw text metadata to fail")
        except ValueError as exc:
            assert "restricted columns" in str(exc)


def test_hybrid_collate_batch_shapes() -> None:
    items = [
        {
            "example_id": "a",
            "split": "train",
            "label": 0.0,
            "section_embeddings": np.ones((2, 4), dtype=np.float32),
            "section_type_ids": np.array([1, 2], dtype=np.int64),
            "chunk_embeddings": np.zeros((0, 4), dtype=np.float32),
            "use_fallback": False,
            "note_embedding": np.ones(4, dtype=np.float32),
        },
        {
            "example_id": "b",
            "split": "train",
            "label": 1.0,
            "section_embeddings": np.ones((1, 4), dtype=np.float32),
            "section_type_ids": np.array([1], dtype=np.int64),
            "chunk_embeddings": np.ones((3, 4), dtype=np.float32),
            "use_fallback": True,
            "note_embedding": np.ones(4, dtype=np.float32),
        },
    ]
    batch = train_downstream.collate_hybrid_batch(items)
    assert batch["section_embeddings"].shape == (2, 2, 4)
    assert batch["fallback_chunk_embeddings"].shape == (2, 3, 4)
    assert batch["note_embedding"].shape == (2, 4)
    assert batch["labels"].shape == (2,)
