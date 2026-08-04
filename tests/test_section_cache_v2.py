from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from notesm_v2.data.local_input_v2 import (
    PROJECT_SPECIAL_TOKENS,
    sha256_path,
    validate_augmented_tokenizer_v2,
    validate_runtime_tokenizer_binding_v2,
)
from notesm_v2.data.section_cache_v2 import (
    EXTRACTION_READY_READINESS,
    PARSER_V2_CACHE_SCHEMA_VERSION,
    RESTRICTED_CACHE_FILENAMES,
    LockedTokenizerBindingV2,
    _vocab_mapping_sha256,
    bind_locked_tokenizer_to_model,
    build_parser_v2_cache_frames,
    write_parser_v2_cache,
)
from notesm_v2.utils.parser_v2_privacy import validate_public_artifact


class _FakeBackendTokenizer:
    def __init__(self, serialized: str) -> None:
        self._serialized = serialized

    def to_str(self) -> str:
        return self._serialized


class FakeFastTokenizer:
    """Small immutable character tokenizer with exact source offsets."""

    def __init__(self) -> None:
        ordinary_tokens = ["<PAD>", "<EOS>", "<UNK>"]
        ordinary_tokens.extend(chr(codepoint) for codepoint in range(32, 127))
        tokens = list(dict.fromkeys([*ordinary_tokens, *PROJECT_SPECIAL_TOKENS]))
        self._vocab = {token: index for index, token in enumerate(tokens)}
        payload = {
            "fake_tokenizer": {
                "kind": "character",
                "vocab": self._vocab,
            }
        }
        self._serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.backend_tokenizer = _FakeBackendTokenizer(self._serialized)
        self.is_fast = True
        self.padding_side = "right"
        self.pad_token_id = self._vocab["<PAD>"]
        self.eos_token_id = self._vocab["<EOS>"]
        self.unk_token_id = self._vocab["<UNK>"]

    @property
    def serialized(self) -> str:
        return self._serialized

    def __len__(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.get(token, self.unk_token_id)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("the parser-v2 test tokenizer forbids automatic specials")
        if text in self._vocab and text in PROJECT_SPECIAL_TOKENS:
            return [self._vocab[text]]
        return list(
            self(
                text,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_offsets_mapping=False,
            )["input_ids"]
        )

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if add_special_tokens or padding or truncation:
            raise AssertionError("direct parser-v2 tokenization contract was changed")
        token_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for index, character in enumerate(text):
            if character.isspace():
                continue
            token_ids.append(self._vocab.get(character, self.unk_token_id))
            offsets.append((index, index + 1))
        result: dict[str, Any] = {"input_ids": token_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class NativeZeroWidthFastTokenizer(FakeFastTokenizer):
    """Emit native zero-width whitespace anchors without changing token order."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if add_special_tokens or padding or truncation:
            raise AssertionError("direct parser-v2 tokenization contract was changed")

        token_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(text):
            if text[cursor].isspace():
                run_end = cursor + 1
                while run_end < len(text) and text[run_end].isspace():
                    run_end += 1
                for character in text[cursor:run_end]:
                    token_ids.append(self._vocab.get(character, self.unk_token_id))
                    offsets.append((run_end, run_end))
                cursor = run_end
                continue
            token_ids.append(self._vocab.get(text[cursor], self.unk_token_id))
            offsets.append((cursor, cursor + 1))
            cursor += 1

        result: dict[str, Any] = {"input_ids": token_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class EphemeralBackendStateFastTokenizer(FakeFastTokenizer):
    """Model the tokenizers backend's mutable per-call padding state."""

    def __init__(self) -> None:
        super().__init__()
        payload = json.loads(self._serialized)
        payload["padding"] = {
            "strategy": {"Fixed": 1024},
            "direction": "Right",
            "pad_id": self.pad_token_id,
            "pad_type_id": 0,
            "pad_token": "<PAD>",
            "pad_to_multiple_of": None,
        }
        payload["truncation"] = {
            "direction": "Right",
            "max_length": 1024,
            "strategy": "LongestFirst",
            "stride": 0,
        }
        self._serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.backend_tokenizer._serialized = self._serialized

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, Any]:
        result = super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            return_offsets_mapping=return_offsets_mapping,
        )
        payload = json.loads(self.backend_tokenizer._serialized)
        payload["padding"] = None
        payload["truncation"] = None
        self.backend_tokenizer._serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return result


class DelayedSemanticMutationFastTokenizer(FakeFastTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.mutate_on_next_call = False

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, Any]:
        result = super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            return_offsets_mapping=return_offsets_mapping,
        )
        if self.mutate_on_next_call:
            payload = json.loads(self.backend_tokenizer._serialized)
            payload["fake_tokenizer"]["kind"] = "delayed-mutated-model"
            self.backend_tokenizer._serialized = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            self.mutate_on_next_call = False
        return result


class _FakeWeight:
    def __init__(self, rows: int) -> None:
        self.shape = (rows, 4)


class _FakeEmbeddings:
    def __init__(self, rows: int) -> None:
        self.weight = _FakeWeight(rows)


class FakeModel:
    def __init__(self, embedding_rows: int) -> None:
        self._embeddings = _FakeEmbeddings(embedding_rows)

    def get_input_embeddings(self) -> _FakeEmbeddings:
        return self._embeddings


def _write_fake_tokenizer(
    tmp_path: Path,
    *,
    model: FakeModel | None = None,
    tokenizer: FakeFastTokenizer | None = None,
) -> tuple[FakeFastTokenizer, Path, Any]:
    tokenizer = tokenizer or FakeFastTokenizer()
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text(tokenizer.serialized, encoding="utf-8")
    preflight = validate_augmented_tokenizer_v2(
        tokenizer,
        tokenizer_json_path=tokenizer_json,
        expected_tokenizer_json_sha256=sha256_path(tokenizer_json),
        model=model,
    )
    return tokenizer, tokenizer_json, preflight


def _binding(
    tokenizer: FakeFastTokenizer,
    tokenizer_json: Path,
    preflight: Any,
) -> LockedTokenizerBindingV2:
    return LockedTokenizerBindingV2(
        tokenizer=tokenizer,
        preflight=preflight,
        tokenizer_json_path=tokenizer_json.resolve(),
        vocab_mapping_sha256=_vocab_mapping_sha256(tokenizer),
        tokenizer_object_identity=id(tokenizer),
    )


def _train_examples() -> pd.DataFrame:
    direct_note = (
        "Chief Complaint:\n"
        "AB\n"
        "Brief Hospital Course:\n"
        "CD"
    )
    repeated_note = (
        "Past Medical History:\n"
        "EF\n"
        "Brief Hospital Course:\n"
        "GH\n"
        "Past Medical History:\n"
        "IJ"
    )
    return pd.DataFrame(
        {
            "example_id": ["direct", "repeated", "zero-slot"],
            "split": ["train", "train", "train"],
            "text": [direct_note, repeated_note, ""],
        }
    )


@pytest.fixture
def implementation_cache(tmp_path: Path) -> tuple[Any, LockedTokenizerBindingV2, pd.DataFrame]:
    tokenizer, tokenizer_json, preflight = _write_fake_tokenizer(tmp_path)
    tokenizer_only_binding = _binding(tokenizer, tokenizer_json, preflight)
    binding = bind_locked_tokenizer_to_model(
        tokenizer_only_binding,
        FakeModel(len(tokenizer)),
    )
    examples = _train_examples()
    frames = build_parser_v2_cache_frames(
        examples,
        tokenizer_binding=binding,
    )
    return frames, binding, examples


@pytest.fixture
def native_zero_width_cache(
    tmp_path: Path,
) -> tuple[Any, LockedTokenizerBindingV2, pd.DataFrame]:
    tokenizer = NativeZeroWidthFastTokenizer()
    tokenizer, tokenizer_json, preflight = _write_fake_tokenizer(
        tmp_path,
        tokenizer=tokenizer,
    )
    binding = bind_locked_tokenizer_to_model(
        _binding(tokenizer, tokenizer_json, preflight),
        FakeModel(len(tokenizer)),
    )
    examples = pd.DataFrame(
        {
            "example_id": ["native-zero"],
            "split": ["train"],
            "text": ["Chief Complaint:\nA  B"],
        }
    )
    frames = build_parser_v2_cache_frames(
        examples,
        tokenizer_binding=binding,
    )
    return frames, binding, examples


def test_note_index_is_one_row_per_example_and_zero_slot_fallback_is_preserved(
    implementation_cache: tuple[Any, LockedTokenizerBindingV2, pd.DataFrame],
) -> None:
    frames, _, examples = implementation_cache

    assert len(frames.note_index) == len(examples)
    assert not frames.note_index["example_id"].duplicated().any()
    assert set(frames.note_index["example_id"]) == set(examples["example_id"])

    zero_note = frames.note_index.set_index("example_id").loc["zero-slot"]
    assert int(zero_note["model_slot_count"]) == 0
    assert bool(zero_note["use_chunk_fallback"])
    assert int(zero_note["fallback_chunk_count"]) == 1

    zero_chunks = frames.fallback_chunks.loc[
        frames.fallback_chunks["example_id"] == "zero-slot"
    ]
    assert len(zero_chunks) == 1
    zero_chunk = zero_chunks.iloc[0]
    assert bool(zero_chunk["empty_source_chunk"])
    assert zero_chunk["content_token_ids"] == []
    assert zero_chunk["source_offset_starts"] == []
    assert zero_chunk["source_offset_ends"] == []
    assert zero_chunk["source_offset_zero_width_mask"] == []
    assert int(zero_chunk["zero_width_token_count"]) == 0
    assert zero_chunk["input_ids"] == [int(zero_chunk["chunk_sum_token_id"])]
    assert frames.summary["zero_model_slot_fallback_preserved"] is True


def test_restricted_ledgers_do_not_export_raw_text_columns(
    implementation_cache: tuple[Any, LockedTokenizerBindingV2, pd.DataFrame],
) -> None:
    frames, _, examples = implementation_cache

    assert set(RESTRICTED_CACHE_FILENAMES) == {
        "note_index",
        "occurrence_ledger",
        "span_ledger",
        "token_ledger",
        "fallback_chunks",
    }
    assert {"heading_start", "body_start", "content_start"} <= set(
        frames.occurrence_ledger.columns
    )
    assert {"span_start", "span_end", "span_kind"} <= set(
        frames.span_ledger.columns
    )
    assert {
        "original_content_ids",
        "source_offset_starts",
        "source_offset_ends",
        "source_offset_zero_width_mask",
        "retained_mask",
        "input_ids",
        "retained_source_offset_zero_width_mask",
        "zero_width_token_count",
        "retained_zero_width_token_count",
        "dropped_zero_width_token_count",
    } <= set(frames.token_ledger.columns)
    assert {
        "content_token_ids",
        "source_offset_starts",
        "source_offset_ends",
        "source_offset_zero_width_mask",
        "zero_width_token_count",
        "input_ids",
    } <= set(frames.fallback_chunks.columns)
    assert {
        "zero_width_token_count",
        "retained_zero_width_token_count",
        "dropped_zero_width_token_count",
    } <= set(frames.occurrence_ledger.columns)

    forbidden_columns = {
        "text",
        "raw_text",
        "note_text",
        "heading",
        "raw_heading",
        "body",
        "section_text",
    }
    raw_notes = set(examples["text"].tolist())
    for frame in [
        frames.note_index,
        frames.occurrence_ledger,
        frames.span_ledger,
        frames.token_ledger,
        frames.fallback_chunks,
    ]:
        assert forbidden_columns.isdisjoint(frame.columns)
        for column in frame.select_dtypes(include=["object"]).columns:
            assert not any(
                isinstance(value, str) and value in raw_notes and value
                for value in frame[column].tolist()
            )


def test_direct_token_ids_use_absolute_source_offsets(
    implementation_cache: tuple[Any, LockedTokenizerBindingV2, pd.DataFrame],
) -> None:
    frames, binding, examples = implementation_cache
    note = str(examples.set_index("example_id").loc["direct", "text"])
    body_start = note.index("AB")
    token_row = frames.token_ledger.loc[
        (frames.token_ledger["example_id"] == "direct")
        & (
            frames.token_ledger["canonical_section"]
            == "CHIEF_COMPLAINT"
        )
    ].iloc[0]

    expected_content_ids = [
        binding.tokenizer.convert_tokens_to_ids("A"),
        binding.tokenizer.convert_tokens_to_ids("B"),
    ]
    expected_marker = binding.tokenizer.convert_tokens_to_ids("<SEC_SUM>")
    assert token_row["original_content_ids"] == expected_content_ids
    assert token_row["retained_content_ids"] == expected_content_ids
    assert token_row["input_ids"] == [*expected_content_ids, expected_marker]
    assert token_row["attention_mask"] == [1, 1, 1]
    assert token_row["source_offset_starts"] == [body_start, body_start + 1]
    assert token_row["source_offset_ends"] == [body_start + 1, body_start + 2]
    assert token_row["source_offset_zero_width_mask"] == [False, False]
    assert token_row["retained_source_offset_zero_width_mask"] == [False, False]
    assert int(token_row["zero_width_token_count"]) == 0
    assert int(token_row["retained_zero_width_token_count"]) == 0
    assert int(token_row["dropped_zero_width_token_count"]) == 0
    assert token_row["retained_mask"] == [True, True]


def test_native_zero_width_masks_counts_and_same_anchor_order_reach_both_ledgers(
    native_zero_width_cache: tuple[Any, LockedTokenizerBindingV2, pd.DataFrame],
) -> None:
    frames, _, examples = native_zero_width_cache

    assert (
        PARSER_V2_CACHE_SCHEMA_VERSION
        == "parser_v2_occurrence_direct_ids_v2_native_zero_width"
    )
    assert set(frames.note_index["cache_schema_version"]) == {
        PARSER_V2_CACHE_SCHEMA_VERSION
    }

    note = str(examples.iloc[0]["text"])
    body_start = note.index("A")
    body_anchor = note.index("B")
    token_row = frames.token_ledger.iloc[0]

    assert token_row["source_offset_starts"] == [
        body_start,
        body_anchor,
        body_anchor,
        body_anchor,
    ]
    assert token_row["source_offset_ends"] == [
        body_start + 1,
        body_anchor,
        body_anchor,
        body_anchor + 1,
    ]
    assert token_row["source_offset_zero_width_mask"] == [
        False,
        True,
        True,
        False,
    ]
    assert token_row["retained_source_offset_zero_width_mask"] == [
        False,
        True,
        True,
        False,
    ]
    assert int(token_row["zero_width_token_count"]) == 2
    assert int(token_row["retained_zero_width_token_count"]) == 2
    assert int(token_row["dropped_zero_width_token_count"]) == 0
    assert token_row["retained_mask"] == [True, True, True, True]

    occurrence = frames.occurrence_ledger.loc[
        frames.occurrence_ledger["is_model_slot"].astype(bool)
    ].iloc[0]
    assert int(occurrence["zero_width_token_count"]) == 2
    assert int(occurrence["retained_zero_width_token_count"]) == 2
    assert int(occurrence["dropped_zero_width_token_count"]) == 0

    fallback = frames.fallback_chunks.iloc[0]
    fallback_mask = list(fallback["source_offset_zero_width_mask"])
    fallback_starts = list(fallback["source_offset_starts"])
    fallback_ends = list(fallback["source_offset_ends"])
    assert len(fallback_mask) == len(fallback["content_token_ids"])
    assert fallback_mask == [
        start == end
        for start, end in zip(fallback_starts, fallback_ends, strict=True)
    ]
    assert int(fallback["zero_width_token_count"]) == sum(fallback_mask) == 4
    two_space_token_indexes = [
        index
        for index, (start, end) in enumerate(
            zip(fallback_starts, fallback_ends, strict=True)
        )
        if start == body_anchor and end == body_anchor
    ]
    assert len(two_space_token_indexes) == 2
    assert two_space_token_indexes[1] == two_space_token_indexes[0] + 1


def test_native_zero_width_restricted_ledgers_survive_parquet_roundtrip(
    native_zero_width_cache: tuple[Any, LockedTokenizerBindingV2, pd.DataFrame],
    tmp_path: Path,
) -> None:
    frames, binding, _ = native_zero_width_cache
    frames.summary.update(
        {
            "strict_model_state_load": True,
            "model_config_sha256": "b" * 64,
            "checkpoint_state_sha256": "c" * 64,
            "base_tokenizer_json_sha256": "d" * 64,
            "benchmark_columns_opened": "example_id|text",
            "outcome_label_column_opened": False,
            "patient_identifier_column_opened": False,
        }
    )
    output_dir = tmp_path / "native-zero-width-cache"

    write_parser_v2_cache(
        frames,
        output_dir=output_dir,
        protocol_sha256="a" * 64,
        code_sha256={"synthetic.py": "e" * 64},
        tokenizer_preflight=binding.preflight,
    )

    token_roundtrip = pd.read_parquet(
        output_dir / RESTRICTED_CACHE_FILENAMES["token_ledger"]
    ).iloc[0]
    fallback_roundtrip = pd.read_parquet(
        output_dir / RESTRICTED_CACHE_FILENAMES["fallback_chunks"]
    ).iloc[0]
    occurrence_roundtrip = pd.read_parquet(
        output_dir / RESTRICTED_CACHE_FILENAMES["occurrence_ledger"]
    )
    occurrence_roundtrip = occurrence_roundtrip.loc[
        occurrence_roundtrip["is_model_slot"].astype(bool)
    ].iloc[0]

    assert token_roundtrip["source_offset_zero_width_mask"].tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert token_roundtrip[
        "retained_source_offset_zero_width_mask"
    ].tolist() == [False, True, True, False]
    assert int(token_roundtrip["zero_width_token_count"]) == 2
    assert int(token_roundtrip["retained_zero_width_token_count"]) == 2
    assert int(token_roundtrip["dropped_zero_width_token_count"]) == 0

    fallback_mask = fallback_roundtrip[
        "source_offset_zero_width_mask"
    ].tolist()
    assert int(fallback_roundtrip["zero_width_token_count"]) == sum(
        bool(value) for value in fallback_mask
    )
    assert int(occurrence_roundtrip["zero_width_token_count"]) == 2
    assert int(occurrence_roundtrip["retained_zero_width_token_count"]) == 2
    assert int(occurrence_roundtrip["dropped_zero_width_token_count"]) == 0


def test_native_zero_width_cache_truncates_by_token_index_at_511(
    tmp_path: Path,
) -> None:
    tokenizer = NativeZeroWidthFastTokenizer()
    tokenizer, tokenizer_json, preflight = _write_fake_tokenizer(
        tmp_path,
        tokenizer=tokenizer,
    )
    binding = bind_locked_tokenizer_to_model(
        _binding(tokenizer, tokenizer_json, preflight),
        FakeModel(len(tokenizer)),
    )
    examples = pd.DataFrame(
        {
            "example_id": ["truncation-zero-width"],
            "split": ["train"],
            "text": ["Chief Complaint:\n" + ("A" * 510) + "   B"],
        }
    )

    frames = build_parser_v2_cache_frames(
        examples,
        tokenizer_binding=binding,
    )
    token_row = frames.token_ledger.iloc[0]

    assert int(token_row["full_content_token_count"]) == 514
    assert int(token_row["retained_content_token_count"]) == 511
    assert int(token_row["dropped_content_token_count"]) == 3
    assert len(token_row["retained_content_ids"]) == 511
    assert token_row["retained_content_ids"] == token_row[
        "original_content_ids"
    ][:511]
    assert token_row["retained_mask"] == ([True] * 511) + ([False] * 3)
    assert token_row["source_offset_zero_width_mask"][510:513] == [
        True,
        True,
        True,
    ]
    assert token_row["retained_source_offset_zero_width_mask"] == token_row[
        "source_offset_zero_width_mask"
    ][:511]
    assert int(token_row["zero_width_token_count"]) == 3
    assert int(token_row["retained_zero_width_token_count"]) == 1
    assert int(token_row["dropped_zero_width_token_count"]) == 2
    assert len(token_row["input_ids"]) == 512

    occurrence = frames.occurrence_ledger.loc[
        frames.occurrence_ledger["is_model_slot"].astype(bool)
    ].iloc[0]
    assert int(occurrence["zero_width_token_count"]) == 3
    assert int(occurrence["retained_zero_width_token_count"]) == 1
    assert int(occurrence["dropped_zero_width_token_count"]) == 2


def test_repeated_canonical_occurrences_are_not_merged(
    implementation_cache: tuple[Any, LockedTokenizerBindingV2, pd.DataFrame],
) -> None:
    frames, binding, _ = implementation_cache
    repeated_occurrences = frames.occurrence_ledger.loc[
        (frames.occurrence_ledger["example_id"] == "repeated")
        & (
            frames.occurrence_ledger["canonical_section"]
            == "PAST_MEDICAL_HISTORY"
        )
        & frames.occurrence_ledger["is_model_slot"].astype(bool)
    ].sort_values("source_order")
    repeated_tokens = frames.token_ledger.loc[
        (frames.token_ledger["example_id"] == "repeated")
        & (
            frames.token_ledger["canonical_section"]
            == "PAST_MEDICAL_HISTORY"
        )
    ].sort_values("source_order")

    assert len(repeated_occurrences) == 2
    assert len(repeated_tokens) == 2
    assert repeated_occurrences["canonical_occurrence_count"].tolist() == [2, 2]
    assert repeated_occurrences["is_repeated_canonical"].astype(bool).all()
    assert repeated_occurrences["source_order"].is_monotonic_increasing
    repeat_marker_id = binding.tokenizer.convert_tokens_to_ids("<SEC_REPEAT>")
    assert all(
        repeat_marker_id not in input_ids
        for input_ids in repeated_tokens["input_ids"].tolist()
    )
    assert frames.summary["synthetic_repeat_markers_injected"] == 0


@pytest.mark.parametrize("requested_split", ["validation", "test"])
def test_requested_validation_or_test_split_is_blocked(
    tmp_path: Path,
    requested_split: str,
) -> None:
    tokenizer, tokenizer_json, preflight = _write_fake_tokenizer(tmp_path)
    binding = _binding(tokenizer, tokenizer_json, preflight)

    with pytest.raises(ValueError, match="train only"):
        build_parser_v2_cache_frames(
            _train_examples(),
            tokenizer_binding=binding,
            split=requested_split,
        )


@pytest.mark.parametrize("row_split", ["validation", "test"])
def test_non_train_row_membership_is_blocked(
    tmp_path: Path,
    row_split: str,
) -> None:
    tokenizer, tokenizer_json, preflight = _write_fake_tokenizer(tmp_path)
    binding = _binding(tokenizer, tokenizer_json, preflight)
    examples = pd.DataFrame(
        {"example_id": ["blocked"], "split": [row_split], "text": ["AB"]}
    )

    with pytest.raises(ValueError, match="non-train row"):
        build_parser_v2_cache_frames(examples, tokenizer_binding=binding)


@pytest.mark.parametrize("invalid_text", [None, 17, ["not", "a", "string"]])
def test_null_and_non_string_text_are_blocked(
    tmp_path: Path,
    invalid_text: object,
) -> None:
    tokenizer, tokenizer_json, preflight = _write_fake_tokenizer(tmp_path)
    tokenizer_only_binding = _binding(tokenizer, tokenizer_json, preflight)
    binding = bind_locked_tokenizer_to_model(
        tokenizer_only_binding,
        FakeModel(len(tokenizer)),
    )
    examples = pd.DataFrame(
        {"example_id": ["invalid"], "split": ["train"], "text": [invalid_text]}
    )

    with pytest.raises(ValueError, match="text"):
        build_parser_v2_cache_frames(examples, tokenizer_binding=binding)


def test_runtime_binding_and_extraction_readiness_are_fail_closed(
    tmp_path: Path,
) -> None:
    tokenizer, tokenizer_json, tokenizer_only = _write_fake_tokenizer(tmp_path)
    binding = _binding(tokenizer, tokenizer_json, tokenizer_only)

    assert tokenizer_only.extraction_ready is False
    validate_runtime_tokenizer_binding_v2(
        tokenizer,
        tokenizer_only,
        require_extraction_ready=False,
    )
    with pytest.raises(ValueError, match="compatibility"):
        validate_runtime_tokenizer_binding_v2(
            tokenizer,
            tokenizer_only,
            require_extraction_ready=True,
        )
    with pytest.raises(ValueError, match="compatibility"):
        build_parser_v2_cache_frames(
            _train_examples(),
            tokenizer_binding=binding,
        )

    replacement_tokenizer = FakeFastTokenizer()
    model_bound_binding = bind_locked_tokenizer_to_model(
        binding,
        FakeModel(len(tokenizer)),
    )
    replaced_binding = replace(
        model_bound_binding,
        tokenizer=replacement_tokenizer,
    )
    with pytest.raises(ValueError, match="identity"):
        build_parser_v2_cache_frames(
            _train_examples(),
            tokenizer_binding=replaced_binding,
        )

    model = FakeModel(len(tokenizer))
    model_bound = validate_augmented_tokenizer_v2(
        tokenizer,
        tokenizer_json_path=tokenizer_json,
        expected_tokenizer_json_sha256=sha256_path(tokenizer_json),
        model=model,
    )
    assert model_bound.embedding_rows_match is True
    assert model_bound.model_compatibility_checked is True
    assert model_bound.extraction_ready is True
    validate_runtime_tokenizer_binding_v2(
        tokenizer,
        model_bound,
        require_extraction_ready=True,
    )

    direct_model_bound_binding = _binding(tokenizer, tokenizer_json, model_bound)
    frames = build_parser_v2_cache_frames(
        _train_examples(),
        tokenizer_binding=direct_model_bound_binding,
    )
    assert frames.summary["extraction_ready"] is True
    assert frames.summary["model_embedding_rows_checked"] is True
    with pytest.raises(ValueError, match="extraction-ready"):
        build_parser_v2_cache_frames(
            _train_examples(),
            tokenizer_binding=model_bound_binding,
            readiness="implementation_only",
        )

    with pytest.raises(ValueError, match="embedding rows"):
        validate_augmented_tokenizer_v2(
            tokenizer,
            tokenizer_json_path=tokenizer_json,
            expected_tokenizer_json_sha256=sha256_path(tokenizer_json),
            model=FakeModel(len(tokenizer) + 1),
        )


def test_model_bound_cache_survives_only_ephemeral_backend_call_state(
    tmp_path: Path,
) -> None:
    tokenizer = EphemeralBackendStateFastTokenizer()
    tokenizer, tokenizer_json, tokenizer_only = _write_fake_tokenizer(
        tmp_path,
        tokenizer=tokenizer,
    )
    binding = _binding(tokenizer, tokenizer_json, tokenizer_only)

    # Tokenizer preflight itself performs a call, so the backend JSON is now
    # locked to the explicit disabled padding/truncation policy.
    assert tokenizer.backend_tokenizer.to_str() != tokenizer.serialized

    model_bound = bind_locked_tokenizer_to_model(
        binding,
        FakeModel(len(tokenizer)),
    )
    assert (
        model_bound.preflight.runtime_tokenizer_semantic_sha256
        == model_bound.preflight.file_tokenizer_semantic_sha256
    )
    frames = build_parser_v2_cache_frames(
        _train_examples(),
        tokenizer_binding=model_bound,
    )
    assert frames.summary["extraction_ready"] is True
    assert frames.summary["example_count"] == len(_train_examples())

    # Re-enabling a runtime truncation policy after binding must fail closed.
    payload = json.loads(tokenizer.backend_tokenizer.to_str())
    payload["truncation"] = {
        "direction": "Right",
        "max_length": 512,
        "strategy": "LongestFirst",
        "stride": 0,
    }
    tokenizer.backend_tokenizer._serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with pytest.raises(ValueError, match="changed after preflight"):
        build_parser_v2_cache_frames(
            _train_examples(),
            tokenizer_binding=model_bound,
        )

    # Mutating a tokenizer-semantic backend field also remains fail closed.
    payload = json.loads(tokenizer.backend_tokenizer.to_str())
    payload["truncation"] = None
    payload["fake_tokenizer"]["kind"] = "mutated-character-model"
    tokenizer.backend_tokenizer._serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with pytest.raises(ValueError, match="changed after preflight"):
        build_parser_v2_cache_frames(
            _train_examples(),
            tokenizer_binding=model_bound,
        )


def test_model_promotion_preserves_tokenizer_fingerprints(
    tmp_path: Path,
) -> None:
    tokenizer, tokenizer_json, tokenizer_only = _write_fake_tokenizer(tmp_path)
    binding = _binding(tokenizer, tokenizer_json, tokenizer_only)
    model_bound = bind_locked_tokenizer_to_model(
        binding,
        FakeModel(len(tokenizer)),
    )

    fingerprint_fields = (
        "tokenizer_json_sha256",
        "expected_tokenizer_json_sha256",
        "file_tokenizer_canonical_sha256",
        "runtime_tokenizer_canonical_sha256",
        "file_tokenizer_semantic_sha256",
        "runtime_tokenizer_semantic_sha256",
        "runtime_tokenizer_matches_file",
        "runtime_padding_disabled",
        "runtime_truncation_disabled",
        "vocab_size",
        "pad_token_id",
        "eos_token_id",
        "sec_sum_token_id",
    )
    for field_name in fingerprint_fields:
        assert getattr(model_bound.preflight, field_name) == getattr(
            tokenizer_only,
            field_name,
        )
    assert tokenizer_only.extraction_ready is False
    assert model_bound.preflight.extraction_ready is True
    assert model_bound.preflight.embedding_rows_match is True


def test_cache_final_gate_detects_delayed_tokenizer_semantic_mutation(
    tmp_path: Path,
) -> None:
    tokenizer = DelayedSemanticMutationFastTokenizer()
    tokenizer, tokenizer_json, tokenizer_only = _write_fake_tokenizer(
        tmp_path,
        tokenizer=tokenizer,
    )
    binding = _binding(tokenizer, tokenizer_json, tokenizer_only)
    model_bound = bind_locked_tokenizer_to_model(
        binding,
        FakeModel(len(tokenizer)),
    )
    tokenizer.mutate_on_next_call = True

    with pytest.raises(ValueError, match="changed after preflight"):
        build_parser_v2_cache_frames(
            _train_examples(),
            tokenizer_binding=model_bound,
        )


def test_write_cache_fails_before_touching_existing_output_directory(
    implementation_cache: tuple[Any, LockedTokenizerBindingV2, pd.DataFrame],
    tmp_path: Path,
) -> None:
    frames, binding, _ = implementation_cache
    output_dir = tmp_path / "already-exists"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        write_parser_v2_cache(
            frames,
            output_dir=output_dir,
            protocol_sha256="a" * 64,
            code_sha256={},
            tokenizer_preflight=binding.preflight,
        )
    assert list(output_dir.iterdir()) == []
    assert frames.summary["cache_readiness"] == EXTRACTION_READY_READINESS


def test_write_cache_verifies_manifest_before_implementation_marker(
    implementation_cache: tuple[Any, LockedTokenizerBindingV2, pd.DataFrame],
    tmp_path: Path,
) -> None:
    frames, binding, _ = implementation_cache
    frames.summary.update(
        {
            "strict_model_state_load": True,
            "model_config_sha256": "b" * 64,
            "checkpoint_state_sha256": "c" * 64,
            "base_tokenizer_json_sha256": "d" * 64,
            "benchmark_columns_opened": "example_id|text",
            "outcome_label_column_opened": False,
            "patient_identifier_column_opened": False,
        }
    )
    output_dir = tmp_path / "new-cache"

    result = write_parser_v2_cache(
        frames,
        output_dir=output_dir,
        protocol_sha256="a" * 64,
        code_sha256={"synthetic.py": "e" * 64},
        tokenizer_preflight=binding.preflight,
    )

    manifest = output_dir / "PARSER_V2_CACHE_SHA256SUMS.txt"
    marker = output_dir / "PARSER_V2_CACHE_IMPLEMENTATION.complete"
    assert result["status"] == "COMPLETE"
    assert sha256_path(manifest) == result["manifest_sha256"]
    assert marker.is_file()
    assert marker.name not in manifest.read_text(encoding="utf-8")
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_payload["manifest_sha256"] == result["manifest_sha256"]
    provenance_path = output_dir / "PARSER_V2_CACHE_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert "restricted_artifact_sha256" in provenance
    assert "restricted_artifacts" not in provenance
    assert validate_public_artifact(provenance_path).passed is True
