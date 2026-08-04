from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("extract_embeddings_script", ROOT / "scripts" / "extract_embeddings.py")
extract_embeddings = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(extract_embeddings)


class DummyTokenizer:
    pad_token_id = 0
    unk_token_id = -1
    eos_token = "<EOS>"
    pad_token = "<PAD>"
    padding_side = "right"

    def get_vocab(self):
        return {"<SEC_SUM>": 99, "<CHUNK_SUM>": 98, "<PAD>": 0}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.get_vocab().get(token, self.unk_token_id)

    def __call__(self, texts, add_special_tokens=True, padding=False, truncation=True, max_length=None, return_tensors=None):
        rows = []
        for text in texts:
            ids = list(range(1, len(str(text).split()) + 1))
            if truncation and max_length is not None:
                ids = ids[:max_length]
            rows.append(ids)
        if return_tensors == "pt":
            max_len = max(len(row) for row in rows) if rows else 1
            input_ids = []
            attention_mask = []
            for row in rows:
                pad = max_len - len(row)
                input_ids.append(row + [self.pad_token_id] * pad)
                attention_mask.append([1] * len(row) + [0] * pad)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": rows}


class DummyConfig:
    hidden_size = 3


class DummyModel:
    config = DummyConfig()

    def __call__(self, input_ids, attention_mask):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, self.config.hidden_size)
        return type("DummyOutputs", (), {"hidden_states": (hidden,)})


def test_marker_token_is_retained_after_truncation() -> None:
    encoded = extract_embeddings.encode_with_marker(
        ["one two three four five"],
        DummyTokenizer(),
        "<SEC_SUM>",
        max_length=4,
        device=torch.device("cpu"),
    )
    assert encoded["input_ids"].shape[1] == 4
    assert encoded["input_ids"][0, -1].item() == 99


def test_span_mean_2_covers_each_encoded_content_token_once_in_order() -> None:
    frame = pd.DataFrame(
        {
            "example_id": ["a"],
            "split": ["train"],
            "merged_section_text": ["one two three four five"],
        }
    )
    embeddings, metadata = extract_embeddings.extract_local_representation_embeddings(
        frame=frame,
        text_col="merged_section_text",
        marker_token="<SEC_SUM>",
        representation="span_mean_2",
        tokenizer=DummyTokenizer(),
        model=DummyModel(),
        max_length=6,
        batch_size=1,
        device=torch.device("cpu"),
    )

    assert embeddings.shape == (2, 3)
    assert metadata["evidence_slot"].tolist() == [0, 1]
    assert metadata["within_segment_position"].tolist() == [0, 1]
    assert metadata["token_start_position"].tolist() == [0, 3]
    assert metadata["token_end_position"].tolist() == [3, 5]
    assert metadata["token_count"].tolist() == [3, 2]
    assert int(metadata["token_count"].sum()) == 5
    assert np.allclose(embeddings[:, 0], np.asarray([2.0, 4.5], dtype=np.float32))


def test_note_pooling_modes_shape() -> None:
    hidden = torch.randn(2, 5, 7)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    for pooling in ["last_token", "cls", "mean"]:
        pooled = extract_embeddings.pool_hidden_states(hidden, mask, pooling)
        assert pooled.shape == (2, 7)


def test_state_dict_checkpoint_loads_into_embedding_model() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        checkpoint = root / "checkpoint-step-1"
        checkpoint.mkdir()
        source = torch.nn.Linear(2, 1)
        with torch.no_grad():
            source.weight.fill_(5.0)
            source.bias.fill_(2.0)
        torch.save(
            {
                "model_state": source.state_dict(),
                "global_step": 1,
                "checkpoint_format": "state_dict",
                "args": {"scheduler_total_steps": 14764, "max_steps": 2000},
            },
            checkpoint / "pytorch_training_state.pt",
        )

        target = torch.nn.Linear(2, 1)
        state = extract_embeddings.load_state_dict_checkpoint_into_model(target, checkpoint)

        assert state["global_step"] == 1
        assert torch.allclose(target.weight, torch.full_like(target.weight, 5.0))
        assert torch.allclose(target.bias, torch.full_like(target.bias, 2.0))


def test_section_shard_metadata_has_no_raw_text_or_ids() -> None:
    frame = pd.DataFrame(
        {
            "example_id": ["a", "b"],
            "split": ["train", "train"],
            "canonical_section": ["A", "B"],
            "section_order": [0, 0],
            "merged_section_text": ["alpha beta gamma", "delta epsilon"],
            "use_chunk_fallback": [False, False],
        }
    )
    with TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        extract_embeddings.write_marker_cache_shards(
            output_dir=out,
            frame=frame,
            text_col="merged_section_text",
            keep_cols=["example_id", "split", "canonical_section", "section_order", "use_chunk_fallback"],
            prefix="section",
            marker_token="<SEC_SUM>",
            tokenizer=DummyTokenizer(),
            model=DummyModel(),
            max_length=4,
            batch_size=1,
            device=torch.device("cpu"),
            shard_size=1,
        )
        metadata = pd.read_parquet(out / "section_embedding_metadata_shard_000.parquet")
        forbidden = {"merged_section_text", "section_text", "chunk_text", "subject_id", "hadm_id", "note_id"}
        assert forbidden.isdisjoint(metadata.columns)
        assert (out / "section_embeddings_shard_000.npy").exists()
        assert np.load(out / "section_embeddings_shard_000.npy").shape == (1, 3)


def test_note_shards_are_text_free_and_resumable() -> None:
    frame = pd.DataFrame(
        {
            "example_id": ["a", "b", "c"],
            "split": ["train", "val", "test"],
            "text": ["alpha beta", "gamma delta", "epsilon zeta"],
        }
    )
    with TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        extract_embeddings.write_note_cache_shards(
            output_dir=out,
            frame=frame,
            cache_name="debug",
            tokenizer=DummyTokenizer(),
            model=DummyModel(),
            max_length=4,
            batch_size=1,
            device=torch.device("cpu"),
            pooling="last_token",
            backbone_name="dummy",
            use_longformer_global_attention=False,
            shard_size=2,
            resume=False,
        )
        first = pd.read_parquet(out / "debug_metadata_shard_000.parquet")
        second = pd.read_parquet(out / "debug_metadata_shard_001.parquet")
        assert "text" not in first.columns
        assert first["global_row_index"].tolist() == [0, 1]
        assert second["global_row_index"].tolist() == [2]
        assert np.load(out / "debug_embeddings_shard_000.npy").shape == (2, 3)

        extract_embeddings.write_note_cache_shards(
            output_dir=out,
            frame=frame,
            cache_name="debug",
            tokenizer=DummyTokenizer(),
            model=DummyModel(),
            max_length=4,
            batch_size=1,
            device=torch.device("cpu"),
            pooling="last_token",
            backbone_name="dummy",
            use_longformer_global_attention=False,
            shard_size=2,
            resume=True,
        )
        summary = pd.read_csv(out / "debug_shards_summary.csv")
        assert int(summary.loc[0, "n_rows"]) == 3


def test_chunk_shard_creates_one_file_when_one_shard_is_expected() -> None:
    frame = pd.DataFrame(
        {
            "example_id": ["a", "a"],
            "split": ["train", "train"],
            "chunk_order": [0, 1],
            "chunk_text": ["alpha beta", "gamma delta"],
            "token_length": [2, 2],
        }
    )
    with TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        extract_embeddings.write_marker_cache_shards(
            output_dir=out,
            frame=frame,
            text_col="chunk_text",
            keep_cols=["example_id", "split", "chunk_order", "token_length"],
            prefix="chunk",
            marker_token="<CHUNK_SUM>",
            tokenizer=DummyTokenizer(),
            model=DummyModel(),
            max_length=4,
            batch_size=1,
            device=torch.device("cpu"),
            shard_size=10,
        )
        shard_files = sorted(out.glob("chunk_embeddings_shard_*.npy"))
        metadata_files = sorted(out.glob("chunk_embedding_metadata_shard_*.parquet"))
        assert [path.name for path in shard_files] == ["chunk_embeddings_shard_000.npy"]
        assert [path.name for path in metadata_files] == ["chunk_embedding_metadata_shard_000.parquet"]
        assert np.load(shard_files[0]).shape == (2, 3)


def test_chunk_shard_metadata_has_non_overlapping_global_row_index() -> None:
    frame = pd.DataFrame(
        {
            "example_id": ["a", "b", "c"],
            "split": ["train", "train", "train"],
            "chunk_order": [0, 0, 0],
            "chunk_text": ["alpha", "beta", "gamma"],
            "token_length": [1, 1, 1],
        }
    )
    with TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        extract_embeddings.write_marker_cache_shards(
            output_dir=out,
            frame=frame,
            text_col="chunk_text",
            keep_cols=["example_id", "split", "chunk_order", "token_length"],
            prefix="chunk",
            marker_token="<CHUNK_SUM>",
            tokenizer=DummyTokenizer(),
            model=DummyModel(),
            max_length=4,
            batch_size=1,
            device=torch.device("cpu"),
            shard_size=2,
        )
        first = pd.read_parquet(out / "chunk_embedding_metadata_shard_000.parquet")
        second = pd.read_parquet(out / "chunk_embedding_metadata_shard_001.parquet")
        assert {"shard_index", "row_index_in_shard", "global_row_index", "row_index"} <= set(first.columns)
        assert set(first["global_row_index"]).isdisjoint(set(second["global_row_index"]))
        assert first["global_row_index"].tolist() == [0, 1]
        assert second["global_row_index"].tolist() == [2]


def test_section_and_chunk_limit_is_example_level() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        pd.DataFrame(
            {
                "example_id": ["a", "a", "b"],
                "split": ["train", "train", "train"],
                "canonical_section": ["A", "B", "A"],
                "section_order": [0, 1, 0],
                "merged_section_text": ["a one", "a two", "b one"],
            }
        ).to_parquet(root / "section_cache_train.parquet", index=False)
        pd.DataFrame(
            {
                "example_id": ["a", "a", "b"],
                "split": ["train", "train", "train"],
                "chunk_order": [0, 1, 0],
                "chunk_text": ["a c1", "a c2", "b c1"],
                "token_length": [2, 2, 2],
            }
        ).to_parquet(root / "fallback_chunk_cache_train.parquet", index=False)

        sections = extract_embeddings.read_section_cache(root, split="train", limit_examples=1)
        assert sections["example_id"].tolist() == ["a", "a"]
        selected_ids = sections["example_id"].astype(str).drop_duplicates().tolist()
        chunks = extract_embeddings.read_chunk_cache(root, split="train", limit_examples=1, example_ids=selected_ids)
        assert chunks["example_id"].tolist() == ["a", "a"]


def test_section_limit_applies_per_split_for_all() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for split in ["train", "val", "test"]:
            pd.DataFrame(
                {
                    "example_id": [f"{split}_a", f"{split}_a", f"{split}_b"],
                    "split": [split, split, split],
                    "canonical_section": ["A", "B", "A"],
                    "section_order": [0, 1, 0],
                    "merged_section_text": ["one", "two", "three"],
                }
            ).to_parquet(root / f"section_cache_{split}.parquet", index=False)

        sections = extract_embeddings.read_section_cache(root, split="all", limit_examples=1)
        assert sections["split"].tolist() == ["train", "train", "val", "val", "test", "test"]
        assert sections["example_id"].tolist() == ["train_a", "train_a", "val_a", "val_a", "test_a", "test_a"]


def test_missing_fallback_chunk_cache_returns_empty_schema_and_writes_empty_summary() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        chunks = extract_embeddings.read_chunk_cache(root, split="train", limit_examples=1)
        assert chunks.empty
        assert [*extract_embeddings.CHUNK_METADATA_COLUMNS, "chunk_text"] == list(chunks.columns)
        out = root / "out"
        extract_embeddings.write_marker_cache_shards(
            output_dir=out,
            frame=chunks,
            text_col="chunk_text",
            keep_cols=[col for col in extract_embeddings.CHUNK_METADATA_COLUMNS if col in chunks.columns],
            prefix="chunk",
            marker_token="<CHUNK_SUM>",
            tokenizer=DummyTokenizer(),
            model=DummyModel(),
            max_length=4,
            batch_size=1,
            device=torch.device("cpu"),
            shard_size=2,
        )
        summary = pd.read_csv(out / "chunk_embedding_shards_summary.csv")
        assert int(summary.loc[0, "n_rows"]) == 0
        assert int(summary.loc[0, "n_shards"]) == 0
        assert not list(out.glob("chunk_embeddings_shard_*.npy"))


def test_read_examples_filters_by_example_id_manifest_and_preserves_split() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for split in ["train", "val", "test"]:
            pd.DataFrame(
                {
                    "example_id": [f"{split}_a", f"{split}_b"],
                    "label": [1, 0],
                    "text": [f"{split} alpha", f"{split} beta"],
                }
            ).to_parquet(root / f"benchmark_examples_{split}.parquet", index=False)
        manifest = root / "manifest.csv"
        pd.DataFrame(
            {
                "example_id": ["train_b", "test_a"],
                "split": ["train", "test"],
                "label": [0, 1],
            }
        ).to_csv(manifest, index=False)

        examples = extract_embeddings.read_examples(root, split="all", limit=None, example_id_file=manifest)
        assert examples[["example_id", "split"]].to_dict("records") == [
            {"example_id": "train_b", "split": "train"},
            {"example_id": "test_a", "split": "test"},
        ]
        assert "text" in examples.columns


def test_train_val_reads_only_train_and_validation_benchmark_files() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for split in ["train", "val"]:
            pd.DataFrame(
                {
                    "example_id": [f"{split}_a", f"{split}_b"],
                    "label": [1, 0],
                    "text": [f"{split} alpha", f"{split} beta"],
                }
            ).to_parquet(root / f"benchmark_examples_{split}.parquet", index=False)
        manifest = root / "manifest.csv"
        pd.DataFrame(
            {
                "example_id": ["train_b", "val_a"],
                "split": ["train", "val"],
                "label": [0, 1],
            }
        ).to_csv(manifest, index=False)

        examples = extract_embeddings.read_examples(root, split="train_val", limit=None, example_id_file=manifest)

        assert not (root / "benchmark_examples_test.parquet").exists()
        assert examples[["example_id", "split"]].to_dict("records") == [
            {"example_id": "train_b", "split": "train"},
            {"example_id": "val_a", "split": "val"},
        ]


def test_train_val_section_manifest_filter_does_not_require_test_cache() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for split in ["train", "val"]:
            pd.DataFrame(
                {
                    "example_id": [f"{split}_a", f"{split}_b"],
                    "split": [split, split],
                    "canonical_section": ["A", "B"],
                    "section_order": [0, 0],
                    "merged_section_text": ["one two", "three four"],
                }
            ).to_parquet(root / f"section_cache_{split}.parquet", index=False)

        sections = extract_embeddings.read_section_cache(
            root,
            split="train_val",
            limit_examples=None,
            example_ids=["train_b", "val_a"],
        )

        assert not (root / "section_cache_test.parquet").exists()
        assert sections[["example_id", "split"]].to_dict("records") == [
            {"example_id": "train_b", "split": "train"},
            {"example_id": "val_a", "split": "val"},
        ]
