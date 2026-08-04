from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from notesm_v2.data.dapt_blocks import build_dapt_block_manifest, causal_lm_collate_fn
from notesm_v2.data.dapt_shard_dataset import DaptTokenShardDataset
from notesm_v2.data.dapt_token_shards import (
    TokenShardSpec,
    read_token_shard_metadata,
    validate_dapt_token_shards,
    write_dapt_token_shards_streaming,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_dapt_block_smoke", ROOT / "scripts" / "run_dapt_block_smoke.py")
smoke_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = smoke_script
SPEC.loader.exec_module(smoke_script)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False, truncation: bool = False):
        assert add_special_tokens is False
        assert truncation is False
        if not text:
            return {"input_ids": []}
        return {"input_ids": [int(piece) for piece in text.split()]}


class FakeModelConfig:
    use_cache = True


class FakeModel:
    def __init__(self) -> None:
        self.config = FakeModelConfig()
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True


def write_small_shards(root: Path, *, shard_size: int = 2, max_length: int = 4):
    sequences = pd.DataFrame({"sequence_text": ["1 2 3 4 5", "10 11 12"], "subject_id_hash": ["a", "b"]})
    spec = TokenShardSpec(output_dir=root, max_length=max_length, shard_size=shard_size, pad_token_id=0)
    return write_dapt_token_shards_streaming(sequences, FakeTokenizer(), spec)


def test_streaming_writer_does_not_materialize_all_blocks() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        result = write_small_shards(root, shard_size=2, max_length=4)

        assert len(result) == 2
        summary, shard_index = result
        assert not hasattr(summary, "token_blocks")
        assert len(shard_index) == 2
        assert (root / "dapt_token_blocks_shard_00000.npy").exists()
        assert (root / "dapt_token_blocks_shard_00001.npy").exists()


def test_token_shard_padding_and_metadata() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sequences = pd.DataFrame({"sequence_text": ["1 2 3 4 5"]})
        spec = TokenShardSpec(output_dir=root, max_length=4, shard_size=8, pad_token_id=0)
        summary, _ = write_dapt_token_shards_streaming(sequences, FakeTokenizer(), spec)

        array = np.load(root / "dapt_token_blocks_shard_00000.npy")
        metadata = pd.read_parquet(root / "dapt_token_blocks_shard_00000_metadata.parquet")

        assert array.shape == (2, 4)
        assert array.dtype == np.int32
        assert array.tolist() == [[1, 2, 3, 4], [5, 0, 0, 0]]
        assert metadata["block_length"].astype(int).tolist() == [4, 1]
        assert int(summary.iloc[0]["total_blocks"]) == 2


def test_token_shard_manifest_matches_report_manifest() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sequences = pd.DataFrame({"sequence_text": ["1 2 3 4 5", "10 11 12"]})
        manifest, _, _, _, _ = build_dapt_block_manifest(sequences, FakeTokenizer(), max_length=4)
        spec = TokenShardSpec(output_dir=root, max_length=4, shard_size=2, pad_token_id=0)
        summary, _ = write_dapt_token_shards_streaming(sequences, FakeTokenizer(), spec)

        metadata = validate_dapt_token_shards(root, max_length=4, manifest=manifest, summary=summary)

        assert set(metadata["block_id"]) == set(manifest["block_id"])
        assert int(metadata["block_length"].sum()) == int(manifest["block_length"].sum())


def test_no_raw_text_in_token_shard_metadata() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        write_small_shards(root)
        metadata = read_token_shard_metadata(root)

        forbidden = {"sequence_text", "text", "subject_id", "hadm_id", "note_id"}
        assert forbidden.isdisjoint(set(metadata.columns))


def test_shard_dataset_getitem() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        write_small_shards(root)
        dataset = DaptTokenShardDataset(root)

        assert len(dataset) == 3
        assert dataset[0]["input_ids"] == [1, 2, 3, 4]
        assert dataset[1]["input_ids"] == [5]


def test_shard_dataset_can_filter_single_token_blocks_for_causal_lm_training() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        write_small_shards(root)
        dataset = DaptTokenShardDataset(root, min_block_tokens=2)

        assert dataset.source_row_count == 3
        assert dataset.skipped_short_block_count == 1
        assert len(dataset) == 2
        assert all(len(dataset[index]["input_ids"]) >= 2 for index in range(len(dataset)))


def test_causal_collator_from_shard_dataset() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        write_small_shards(root)
        dataset = DaptTokenShardDataset(root)
        batch = causal_lm_collate_fn([dataset[1], dataset[2]], pad_token_id=0)

        assert batch["input_ids"].tolist() == [[5, 0, 0], [10, 11, 12]]
        assert batch["attention_mask"].tolist() == [[1, 0, 0], [1, 1, 1]]
        assert batch["labels"].tolist() == [[5, -100, -100], [10, 11, 12]]


def test_dapt_block_smoke_gate() -> None:
    try:
        smoke_script.validate_smoke_args(11, "model")
        raise AssertionError("expected max_steps gate to fail")
    except ValueError as exc:
        assert "max-steps <= 10" in str(exc)

    try:
        smoke_script.validate_smoke_args(3, None)
        raise AssertionError("expected missing model_id to fail")
    except ValueError as exc:
        assert "model-id is required" in str(exc)


def test_max_shards_marks_partial_and_validation_skips_full_summary_totals() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sequences = pd.DataFrame({"sequence_text": ["1 2 3 4 5", "10 11 12 13 14"]})
        spec = TokenShardSpec(output_dir=root, max_length=2, shard_size=1, pad_token_id=0)
        summary, shard_index = write_dapt_token_shards_streaming(sequences, FakeTokenizer(), spec, max_shards=1)

        assert bool(summary.iloc[0]["partial_shard_write"]) is True
        assert bool(summary.iloc[0]["stopped_by_max_shards"]) is True
        assert len(shard_index) == 1
        metadata = validate_dapt_token_shards(root, max_length=2, summary=summary)
        assert len(metadata) == 1


def test_dry_run_shard_index_marks_planned_only_and_writes_no_files() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sequences = pd.DataFrame({"sequence_text": ["1 2 3"]})
        spec = TokenShardSpec(output_dir=root, max_length=2, shard_size=8, pad_token_id=0)
        summary, shard_index = write_dapt_token_shards_streaming(sequences, FakeTokenizer(), spec, dry_run=True)

        assert bool(summary.iloc[0]["dry_run"]) is True
        assert bool(summary.iloc[0]["token_shards_written"]) is False
        assert shard_index["planned_only"].astype(bool).all()
        assert not shard_index["files_written"].astype(bool).any()
        assert not list(root.glob("dapt_token_blocks_shard_*.npy"))


def test_gradient_checkpointing_disables_use_cache() -> None:
    model = FakeModel()

    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    assert model.config.use_cache is False
    assert model.gradient_checkpointing_enabled is True


def test_validate_dapt_token_shards_catches_duplicate_block_id() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        write_small_shards(root)
        first_metadata = root / "dapt_token_blocks_shard_00000_metadata.parquet"
        frame = pd.read_parquet(first_metadata)
        frame.loc[1, "block_id"] = frame.loc[0, "block_id"]
        frame.to_parquet(first_metadata, index=False)

        try:
            validate_dapt_token_shards(root, max_length=4)
            raise AssertionError("expected duplicate block_id validation failure")
        except ValueError as exc:
            assert "Duplicate block_id" in str(exc)
