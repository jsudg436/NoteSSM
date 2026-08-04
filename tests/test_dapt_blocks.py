from __future__ import annotations

import pandas as pd
import numpy as np

from notesm_v2.data.dapt_blocks import (
    assert_no_raw_columns,
    build_dapt_block_manifest,
    causal_lm_collate_fn,
    split_tokens_into_patient_blocks,
    validate_dapt_block_manifest,
    validate_train_only_sequences,
)


class FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False, truncation: bool = False):
        assert add_special_tokens is False
        assert truncation is False
        if not text:
            return {"input_ids": []}
        return {"input_ids": [int(piece) for piece in text.split()]}


class FakeBatchEncoding:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids

    def keys(self):
        return ["input_ids"]

    def __getitem__(self, key: str):
        if key != "input_ids":
            raise KeyError(key)
        return self.input_ids


class FakeBatchEncodingTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False, truncation: bool = False):
        assert add_special_tokens is False
        assert truncation is False
        if not text:
            return FakeBatchEncoding([])
        return FakeBatchEncoding([int(piece) for piece in text.split()])


def test_tokenizer_batch_encoding_object_is_supported() -> None:
    sequences = pd.DataFrame({"sequence_text": ["1 2 3 4"]})
    manifest, summary, _, _, token_blocks = build_dapt_block_manifest(
        sequences,
        FakeBatchEncodingTokenizer(),
        max_length=3,
        keep_token_blocks=True,
    )

    assert list(manifest["block_length"].astype(int)) == [3, 1]
    assert token_blocks == [[1, 2, 3], [4]]
    assert int(summary.iloc[0]["total_tokens"]) == 4


def test_split_tokens_no_cross_patient_mixing() -> None:
    sequences = pd.DataFrame(
        {
            "sequence_text": ["1 2 3 4 5", "101 102 103 104 105"],
            "subject_id_hash": ["a", "b"],
        }
    )
    manifest, summary, _, _, token_blocks = build_dapt_block_manifest(sequences, FakeTokenizer(), max_length=3, keep_token_blocks=True)

    assert list(manifest.groupby("patient_index")["block_length"].sum()) == [5, 5]
    assert all(all(token < 100 for token in block) for block in token_blocks[:2])
    assert all(all(token >= 100 for token in block) for block in token_blocks[2:])
    assert bool(summary.iloc[0]["cross_patient_mixing"]) is False


def test_full_token_coverage() -> None:
    records, dropped = split_tokens_into_patient_blocks(
        list(range(10)),
        sequence_row_index=0,
        patient_index=0,
        max_length=4,
    )

    assert [record.block_length for record in records] == [4, 4, 2]
    assert sum(record.block_length for record in records) == 10
    assert dropped == 0


def test_final_short_block_kept() -> None:
    records, dropped = split_tokens_into_patient_blocks(
        list(range(5)),
        sequence_row_index=0,
        patient_index=0,
        max_length=4,
    )

    assert [record.block_length for record in records] == [4, 1]
    assert records[-1].is_final_block_for_patient
    assert dropped == 0


def test_empty_sequence_handling() -> None:
    sequences = pd.DataFrame({"sequence_text": [""]})
    manifest, summary, _, patient_counts, token_blocks = build_dapt_block_manifest(sequences, FakeTokenizer(), max_length=4)

    assert manifest.empty
    assert token_blocks == []
    assert int(summary.iloc[0]["n_empty_sequences"]) == 1
    assert int(summary.iloc[0]["total_blocks"]) == 0
    assert int(patient_counts.iloc[0]["token_count"]) == 0


def test_causal_lm_collator_padding_labels() -> None:
    batch = causal_lm_collate_fn([[1, 2, 3], [4]], pad_token_id=0)

    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[1, 2, 3], [4, -100, -100]]


def test_manifest_has_no_raw_text() -> None:
    sequences = pd.DataFrame({"sequence_text": ["1 2 3"], "subject_id_hash": ["hash_a"]})
    manifest, summary, histogram, patient_counts, _ = build_dapt_block_manifest(sequences, FakeTokenizer(), max_length=2)

    assert_no_raw_columns(manifest, "manifest")
    assert_no_raw_columns(summary, "summary")
    assert_no_raw_columns(histogram, "histogram")
    assert_no_raw_columns(patient_counts, "patient_counts")
    assert "sequence_text" not in manifest.columns
    assert "subject_id_hash" in manifest.columns


def test_train_only_split_validation() -> None:
    sequences = pd.DataFrame({"sequence_text": ["1 2 3"], "split": ["val"]})

    try:
        validate_train_only_sequences(sequences)
        raise AssertionError("expected non-train split to fail")
    except ValueError as exc:
        assert "non-train" in str(exc)


def test_max_block_length_never_exceeds_max_length() -> None:
    sequences = pd.DataFrame({"sequence_text": [" ".join(str(i) for i in range(17))]})
    manifest, summary, _, patient_counts, _ = build_dapt_block_manifest(sequences, FakeTokenizer(), max_length=5)

    assert int(manifest["block_length"].max()) <= 5
    validate_dapt_block_manifest(manifest, summary, patient_counts, max_length=5)


def test_full_report_only_does_not_keep_token_blocks() -> None:
    sequences = pd.DataFrame({"sequence_text": ["1 2 3 4 5 6 7"]})
    manifest, summary, _, _, token_blocks = build_dapt_block_manifest(
        sequences,
        FakeTokenizer(),
        max_length=3,
        keep_token_blocks=False,
    )

    assert token_blocks == []
    assert list(manifest["block_length"].astype(int)) == [3, 3, 1]
    assert int(summary.iloc[0]["total_tokens"]) == 7
    assert int(summary.iloc[0]["total_blocks"]) == 3


def test_debug_keep_token_blocks() -> None:
    sequences = pd.DataFrame({"sequence_text": ["1 2 3 4 5 6 7"]})
    manifest, _, _, _, token_blocks = build_dapt_block_manifest(
        sequences,
        FakeTokenizer(),
        max_length=3,
        keep_token_blocks=True,
    )

    assert token_blocks == [[1, 2, 3], [4, 5, 6], [7]]
    assert sum(len(block) for block in token_blocks) == int(manifest["block_length"].sum())


def test_nan_sequence_text_treated_as_empty() -> None:
    sequences = pd.DataFrame({"sequence_text": [np.nan]})
    manifest, summary, _, patient_counts, token_blocks = build_dapt_block_manifest(
        sequences,
        FakeTokenizer(),
        max_length=4,
        keep_token_blocks=True,
    )

    assert manifest.empty
    assert token_blocks == []
    assert int(summary.iloc[0]["n_empty_sequences"]) == 1
    assert int(patient_counts.iloc[0]["token_count"]) == 0


def test_min_block_tokens_greater_than_max_length_fails() -> None:
    try:
        split_tokens_into_patient_blocks(
            [1, 2, 3],
            sequence_row_index=0,
            patient_index=0,
            max_length=2,
            min_block_tokens=3,
        )
        raise AssertionError("expected min_block_tokens > max_length to fail")
    except ValueError as exc:
        assert "min_block_tokens must be <= max_length" in str(exc)


def test_duplicate_dataframe_index_does_not_create_duplicate_block_id() -> None:
    sequences = pd.DataFrame({"sequence_text": ["1 2 3 4", "5 6 7 8"]})
    sequences.index = [7, 7]

    manifest, _, _, _, _ = build_dapt_block_manifest(sequences, FakeTokenizer(), max_length=2)

    assert not manifest["block_id"].duplicated().any()
    assert set(manifest["sequence_row_index"].astype(int)) == {0, 1}
