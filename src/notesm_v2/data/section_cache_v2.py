from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from notesm_v2.data.local_input_v2 import (
    LOCKED_AUGMENTED_TOKENIZER_JSON_SHA256,
    LOCKED_LOCAL_CONTENT_TOKENS,
    OFFSET_CONTRACT_VERSION,
    PROJECT_SPECIAL_TOKENS,
    LocalInputV2,
    TokenizerPreflightV2,
    build_local_input_v2,
    sha256_path,
    validate_augmented_tokenizer_v2,
    validate_model_bound_tokenizer_v2,
    validate_offsets_v2,
    validate_runtime_tokenizer_binding_v2,
)
from notesm_v2.data.section_parser_v2 import (
    PARSER_V2_VERSION,
    ParserV2Config,
    parse_sections_v2,
)


PARSER_V2_CACHE_SCHEMA_VERSION = (
    "parser_v2_occurrence_direct_ids_v2_native_zero_width"
)
IMPLEMENTATION_STAGE_SPLIT = "train"
EXTRACTION_READY_READINESS = "extraction_ready"
CHUNK_SUM_MARKER = "<CHUNK_SUM>"

NOTE_INDEX_COLUMNS = [
    "example_id",
    "split",
    "parser_version",
    "cache_schema_version",
    "occurrence_count",
    "model_slot_count",
    "heading_candidate_count",
    "recognized_heading_count",
    "unrecognized_heading_count",
    "empty_body_occurrence_count",
    "max_section_tokens",
    "few_sections",
    "too_long",
    "structurally_usable",
    "use_chunk_fallback",
    "evidence_preservation_coverage",
    "recognized_canonical_coverage",
    "source_character_count",
    "nonheading_character_count",
    "recognized_canonical_character_count",
    "omitted_character_count",
    "duplicated_character_count",
    "overlap_span_count",
    "synthetic_repeat_markers_injected",
    "fallback_chunk_count",
    "occurrence_zero_width_token_count",
    "retained_occurrence_zero_width_token_count",
    "dropped_occurrence_zero_width_token_count",
    "fallback_zero_width_token_count",
    "has_occurrence_zero_width",
    "augmented_tokenizer_sha256",
]

OCCURRENCE_LEDGER_COLUMNS = [
    "example_id",
    "split",
    "parser_version",
    "cache_schema_version",
    "occurrence_index",
    "source_order",
    "canonical_section",
    "heading_kind",
    "occurrence_origin",
    "heading_start",
    "heading_end",
    "body_start",
    "body_end",
    "content_start",
    "content_end",
    "is_model_slot",
    "empty_body",
    "full_content_token_count",
    "retained_content_token_count",
    "dropped_content_token_count",
    "zero_width_token_count",
    "retained_zero_width_token_count",
    "dropped_zero_width_token_count",
    "canonical_occurrence_index",
    "canonical_occurrence_count",
    "canonical_model_slot_index",
    "canonical_model_slot_count",
    "model_slot_index",
    "is_repeated_canonical",
    "use_chunk_fallback",
    "augmented_tokenizer_sha256",
]

SPAN_LEDGER_COLUMNS = [
    "example_id",
    "split",
    "parser_version",
    "cache_schema_version",
    "span_order",
    "span_kind",
    "span_start",
    "span_end",
    "occurrence_index",
    "canonical_section",
]

TOKEN_LEDGER_COLUMNS = [
    "example_id",
    "split",
    "parser_version",
    "cache_schema_version",
    "occurrence_index",
    "source_order",
    "canonical_section",
    "original_content_ids",
    "source_offset_starts",
    "source_offset_ends",
    "source_offset_zero_width_mask",
    "retained_mask",
    "retained_content_ids",
    "retained_source_offset_starts",
    "retained_source_offset_ends",
    "retained_source_offset_zero_width_mask",
    "input_ids",
    "attention_mask",
    "full_content_token_count",
    "retained_content_token_count",
    "dropped_content_token_count",
    "zero_width_token_count",
    "retained_zero_width_token_count",
    "dropped_zero_width_token_count",
    "sec_sum_token_id",
    "augmented_tokenizer_sha256",
]

FALLBACK_CHUNK_COLUMNS = [
    "example_id",
    "split",
    "parser_version",
    "cache_schema_version",
    "chunk_order",
    "source_token_start_index",
    "source_token_end_index",
    "content_token_ids",
    "source_offset_starts",
    "source_offset_ends",
    "source_offset_zero_width_mask",
    "input_ids",
    "attention_mask",
    "content_token_count",
    "token_length",
    "zero_width_token_count",
    "empty_source_chunk",
    "chunk_sum_token_id",
    "augmented_tokenizer_sha256",
]

RESTRICTED_CACHE_FILENAMES = {
    "note_index": "parser_v2_note_index_train.parquet",
    "occurrence_ledger": "parser_v2_occurrence_ledger_train.parquet",
    "span_ledger": "parser_v2_span_ledger_train.parquet",
    "token_ledger": "parser_v2_token_ledger_train.parquet",
    "fallback_chunks": "parser_v2_fallback_chunk_token_cache_train.parquet",
}


@dataclass(frozen=True, slots=True)
class ParserV2CacheFrames:
    note_index: pd.DataFrame = field(repr=False)
    occurrence_ledger: pd.DataFrame = field(repr=False)
    span_ledger: pd.DataFrame = field(repr=False)
    token_ledger: pd.DataFrame = field(repr=False)
    fallback_chunks: pd.DataFrame = field(repr=False)
    summary: dict[str, object]


@dataclass(frozen=True, slots=True)
class LockedTokenizerBindingV2:
    tokenizer: Any = field(repr=False)
    preflight: TokenizerPreflightV2
    tokenizer_json_path: Path
    vocab_mapping_sha256: str
    tokenizer_object_identity: int

    def __iter__(self):
        # Preserve convenient two-value unpacking while keeping the binding
        # object available to cache builders.
        yield self.tokenizer
        yield self.preflight


def _vocab_mapping_sha256(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()
    canonical = json.dumps(
        sorted((str(token), int(token_id)) for token, token_id in vocab.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_tokenizer_binding_v2(
    binding: LockedTokenizerBindingV2,
    *,
    require_extraction_ready: bool,
) -> None:
    validate_runtime_tokenizer_binding_v2(
        binding.tokenizer,
        binding.preflight,
        require_extraction_ready=require_extraction_ready,
    )
    if id(binding.tokenizer) != binding.tokenizer_object_identity:
        raise ValueError("tokenizer object identity changed after preflight")
    observed_file_sha = sha256_path(binding.tokenizer_json_path)
    if observed_file_sha != binding.preflight.tokenizer_json_sha256:
        raise ValueError("bound augmented tokenizer artifact changed after preflight")
    if _vocab_mapping_sha256(binding.tokenizer) != binding.vocab_mapping_sha256:
        raise ValueError("bound tokenizer vocabulary changed after preflight")
    if int(len(binding.tokenizer)) != int(binding.preflight.vocab_size):
        raise ValueError("bound tokenizer vocabulary size changed after preflight")
    if not bool(getattr(binding.tokenizer, "is_fast", False)):
        raise ValueError("bound tokenizer is no longer a fast tokenizer")
    if getattr(binding.tokenizer, "padding_side", None) != "right":
        raise ValueError("bound tokenizer padding policy changed after preflight")


def verify_protocol_sidecar(
    protocol_path: str | Path,
    sidecar_path: str | Path,
) -> str:
    protocol = Path(protocol_path)
    sidecar = Path(sidecar_path)
    if not protocol.is_file():
        raise FileNotFoundError("locked parser-v2 protocol is missing")
    if not sidecar.is_file():
        raise FileNotFoundError("locked parser-v2 protocol SHA-256 sidecar is missing")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if not fields or len(fields[0]) != 64:
        raise ValueError("locked parser-v2 protocol SHA-256 sidecar is malformed")
    expected = fields[0].lower()
    observed = sha256_path(protocol)
    if observed != expected:
        raise ValueError("locked parser-v2 protocol SHA-256 mismatch")
    return observed


def validate_locked_bytelevel_offset_contract(
    tokenizer_json_path: str | Path,
) -> dict[str, object]:
    """Fail closed unless the locked tokenizer has the audited ByteLevel policy."""
    path = Path(tokenizer_json_path)
    if not path.is_file():
        raise FileNotFoundError("locked tokenizer.json is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError("locked tokenizer.json is malformed") from None

    pre_tokenizer = payload.get("pre_tokenizer")
    post_processor = payload.get("post_processor")
    if not isinstance(pre_tokenizer, dict) or not isinstance(
        post_processor,
        dict,
    ):
        raise ValueError("locked tokenizer lacks explicit offset components")
    if pre_tokenizer.get("type") != "ByteLevel":
        raise ValueError("locked tokenizer pre-tokenizer is not ByteLevel")
    if post_processor.get("type") != "ByteLevel":
        raise ValueError("locked tokenizer post-processor is not ByteLevel")
    if pre_tokenizer.get("trim_offsets") is not True:
        raise ValueError("locked ByteLevel pre-tokenizer trim_offsets changed")
    if post_processor.get("trim_offsets") is not True:
        raise ValueError("locked ByteLevel post-processor trim_offsets changed")
    for component_name, component in (
        ("pre-tokenizer", pre_tokenizer),
        ("post-processor", post_processor),
    ):
        if component.get("add_prefix_space") is not False:
            raise ValueError(
                f"locked ByteLevel {component_name} add_prefix_space changed"
            )
        if component.get("use_regex") is not True:
            raise ValueError(
                f"locked ByteLevel {component_name} use_regex changed"
            )

    return {
        "offset_contract_version": OFFSET_CONTRACT_VERSION,
        "native_zero_width_offsets_allowed": True,
        "native_zero_width_offsets_preserved": True,
        "offset_repair_applied": False,
        "pre_tokenizer_type": "ByteLevel",
        "pre_tokenizer_trim_offsets": True,
        "pre_tokenizer_add_prefix_space": False,
        "pre_tokenizer_use_regex": True,
        "post_processor_type": "ByteLevel",
        "post_processor_trim_offsets": True,
        "post_processor_add_prefix_space": False,
        "post_processor_use_regex": True,
    }


def load_locked_augmented_tokenizer(
    tokenizer_dir: str | Path,
    *,
    expected_tokenizer_json_sha256: str = LOCKED_AUGMENTED_TOKENIZER_JSON_SHA256,
) -> LockedTokenizerBindingV2:
    root = Path(tokenizer_dir)
    tokenizer_json = root / "tokenizer.json"
    if not root.is_dir() or not tokenizer_json.is_file():
        raise FileNotFoundError("locked augmented tokenizer artifact is missing")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(root),
        local_files_only=True,
        use_fast=True,
    )
    preflight = validate_augmented_tokenizer_v2(
        tokenizer,
        tokenizer_json_path=tokenizer_json,
        expected_tokenizer_json_sha256=expected_tokenizer_json_sha256,
    )
    return LockedTokenizerBindingV2(
        tokenizer=tokenizer,
        preflight=preflight,
        tokenizer_json_path=tokenizer_json.resolve(),
        vocab_mapping_sha256=_vocab_mapping_sha256(tokenizer),
        tokenizer_object_identity=id(tokenizer),
    )


def bind_locked_tokenizer_to_model(
    binding: LockedTokenizerBindingV2,
    model: Any,
) -> LockedTokenizerBindingV2:
    validate_tokenizer_binding_v2(
        binding,
        require_extraction_ready=False,
    )
    preflight = validate_model_bound_tokenizer_v2(
        binding.tokenizer,
        binding.preflight,
        model,
    )
    model_bound = LockedTokenizerBindingV2(
        tokenizer=binding.tokenizer,
        preflight=preflight,
        tokenizer_json_path=binding.tokenizer_json_path,
        vocab_mapping_sha256=binding.vocab_mapping_sha256,
        tokenizer_object_identity=binding.tokenizer_object_identity,
    )
    validate_tokenizer_binding_v2(
        model_bound,
        require_extraction_ready=True,
    )
    return model_bound


def canonical_row_set_sha256(example_ids: pd.Series, split: str) -> str:
    values = example_ids.astype(str)
    if values.duplicated().any():
        raise ValueError("parser-v2 input contains duplicated example identifiers")
    payload = "\n".join(
        f"{split}\0{example_id}" for example_id in sorted(values.tolist())
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ensure_train_only(examples: pd.DataFrame, split: str) -> None:
    if split != IMPLEMENTATION_STAGE_SPLIT:
        raise ValueError("parser-v2 implementation cache is locked to train only")
    required = {"example_id", "text"}
    missing = required - set(examples.columns)
    if missing:
        raise ValueError("parser-v2 examples are missing required columns")
    if examples["example_id"].isna().any():
        raise ValueError("parser-v2 examples contain null identifiers")
    if examples["text"].isna().any():
        raise ValueError("parser-v2 examples contain null note text")
    if not examples["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("parser-v2 note text must contain strings only")
    unexpected = set(examples.columns) - {"example_id", "split", "text"}
    if unexpected:
        raise ValueError("parser-v2 implementation input contains unexpected columns")
    if examples["example_id"].astype(str).duplicated().any():
        raise ValueError("parser-v2 examples contain duplicated identifiers")
    if "split" in examples.columns:
        split_values = examples["split"].astype(str)
        if not split_values.eq(IMPLEMENTATION_STAGE_SPLIT).all():
            raise ValueError("parser-v2 implementation input contains a non-train row")


def _flat_ids(value: Any, *, field: str) -> list[int]:
    if value is None:
        raise ValueError(f"tokenizer output is missing {field}")
    rows = value.tolist() if hasattr(value, "tolist") else value
    if rows and isinstance(rows[0], (list, tuple)):
        if len(rows) != 1:
            raise ValueError(f"{field} unexpectedly contains a batch")
        rows = rows[0]
    return [int(item) for item in rows]


def _flat_offsets(value: Any) -> list[tuple[int, int]]:
    if value is None:
        raise ValueError("tokenizer output is missing offset_mapping")
    rows = value.tolist() if hasattr(value, "tolist") else value
    if (
        rows
        and isinstance(rows[0], (list, tuple))
        and rows[0]
        and isinstance(rows[0][0], (list, tuple))
    ):
        if len(rows) != 1:
            raise ValueError("offset_mapping unexpectedly contains a batch")
        rows = rows[0]
    return [(int(item[0]), int(item[1])) for item in rows]


def validate_native_zero_width_semantics_v2(
    text: str,
    token_ids: list[int] | tuple[int, ...],
    offsets: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    zero_width_mask: list[bool] | tuple[bool, ...],
    tokenizer: Any,
) -> dict[str, int | bool]:
    """Validate native zero-width anchors without exporting token or text values."""
    if not (
        len(token_ids) == len(offsets) == len(zero_width_mask)
    ):
        raise ValueError("native token, offset, and zero-width ledgers differ")
    expected_mask = tuple(start == end for start, end in offsets)
    if tuple(bool(value) for value in zero_width_mask) != expected_mask:
        raise ValueError("native zero-width mask is inconsistent")

    special_ids = {
        int(value)
        for value in (getattr(tokenizer, "all_special_ids", None) or [])
    }
    for token in PROJECT_SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None:
            special_ids.add(int(token_id))

    zero_width_count = 0
    for token_id, (start, end), is_zero_width in zip(
        token_ids,
        offsets,
        zero_width_mask,
        strict=True,
    ):
        if not is_zero_width:
            continue
        zero_width_count += 1
        if start != end:
            raise ValueError("native zero-width mask is inconsistent")
        if int(token_id) in special_ids:
            raise ValueError("native zero-width token is special or synthetic")
        try:
            decoded = tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            raise ValueError(
                "native zero-width token could not be decoded"
            ) from None
        if not isinstance(decoded, str) or not decoded or not decoded.isspace():
            raise ValueError("native zero-width token is not whitespace-only")
        adjacent_whitespace = (
            (start > 0 and text[start - 1].isspace())
            or (start < len(text) and text[start].isspace())
        )
        if not adjacent_whitespace:
            raise ValueError(
                "native zero-width anchor is not adjacent to whitespace"
            )

    return {
        "zero_width_token_count": int(zero_width_count),
        "zero_width_semantics_validated": True,
        "zero_width_non_whitespace_token_count": 0,
        "zero_width_special_token_count": 0,
        "zero_width_nonadjacent_anchor_count": 0,
    }


def _tokenize_source_once(
    text: str,
    tokenizer: Any,
    *,
    validate_zero_width_semantics: bool = False,
) -> tuple[list[int], list[tuple[int, int]], list[bool]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        token_ids = _flat_ids(encoded.get("input_ids"), field="input_ids")
        offsets = _flat_offsets(encoded.get("offset_mapping"))
        validate_offsets_v2(
            offsets,
            text_length=len(text),
            expected_count=len(token_ids),
        )
    except Exception:
        raise ValueError("locked tokenizer failed direct source tokenization") from None
    zero_width_mask = [start == end for start, end in offsets]
    if validate_zero_width_semantics:
        validate_native_zero_width_semantics_v2(
            text,
            token_ids,
            offsets,
            zero_width_mask,
            tokenizer,
        )
    return token_ids, offsets, zero_width_mask


def _single_marker_id(tokenizer: Any, marker: str) -> int:
    marker_id = tokenizer.convert_tokens_to_ids(marker)
    if marker_id is None or marker_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("required summary marker is absent")
    encoded = tokenizer.encode(marker, add_special_tokens=False)
    if [int(item) for item in encoded] != [int(marker_id)]:
        raise ValueError("required summary marker is not exactly one token")
    return int(marker_id)


def _absolute_offsets(
    local_offsets: tuple[tuple[int, int], ...],
    body_start: int,
) -> tuple[list[int], list[int]]:
    starts = [int(body_start + start) for start, _ in local_offsets]
    ends = [int(body_start + end) for _, end in local_offsets]
    return starts, ends


def _token_row(
    *,
    example_id: str,
    split: str,
    occurrence: Any,
    local_input: LocalInputV2,
    tokenizer_sha256: str,
) -> dict[str, object]:
    starts, ends = _absolute_offsets(local_input.original_offsets, occurrence.body_start)
    retained_starts, retained_ends = _absolute_offsets(
        local_input.retained_offsets,
        occurrence.body_start,
    )
    retained_count = local_input.retained_content_count
    return {
        "example_id": example_id,
        "split": split,
        "parser_version": PARSER_V2_VERSION,
        "cache_schema_version": PARSER_V2_CACHE_SCHEMA_VERSION,
        "occurrence_index": int(occurrence.occurrence_index),
        "source_order": int(occurrence.source_order),
        "canonical_section": occurrence.canonical,
        "original_content_ids": list(local_input.original_content_ids),
        "source_offset_starts": starts,
        "source_offset_ends": ends,
        "source_offset_zero_width_mask": list(
            local_input.original_zero_width_mask
        ),
        "retained_mask": [
            index < retained_count
            for index in range(local_input.original_content_count)
        ],
        "retained_content_ids": list(local_input.retained_content_ids),
        "retained_source_offset_starts": retained_starts,
        "retained_source_offset_ends": retained_ends,
        "retained_source_offset_zero_width_mask": list(
            local_input.retained_zero_width_mask
        ),
        "input_ids": list(local_input.input_ids),
        "attention_mask": list(local_input.attention_mask),
        "full_content_token_count": int(local_input.original_content_count),
        "retained_content_token_count": int(retained_count),
        "dropped_content_token_count": int(local_input.dropped_content_count),
        "zero_width_token_count": int(local_input.zero_width_token_count),
        "retained_zero_width_token_count": int(
            local_input.retained_zero_width_token_count
        ),
        "dropped_zero_width_token_count": int(
            local_input.dropped_zero_width_token_count
        ),
        "sec_sum_token_id": int(local_input.sec_sum_token_id),
        "augmented_tokenizer_sha256": tokenizer_sha256,
    }


def _fallback_chunk_rows(
    *,
    example_id: str,
    text: str,
    split: str,
    tokenizer: Any,
    tokenizer_sha256: str,
    validate_zero_width_semantics: bool = False,
) -> list[dict[str, object]]:
    token_ids, offsets, zero_width_mask = _tokenize_source_once(
        text,
        tokenizer,
        validate_zero_width_semantics=validate_zero_width_semantics,
    )
    marker_id = _single_marker_id(tokenizer, CHUNK_SUM_MARKER)
    starts = list(range(0, len(token_ids), LOCKED_LOCAL_CONTENT_TOKENS))
    if not starts:
        starts = [0]
    rows: list[dict[str, object]] = []
    for chunk_order, start in enumerate(starts):
        stop = min(start + LOCKED_LOCAL_CONTENT_TOKENS, len(token_ids))
        content_ids = token_ids[start:stop]
        content_offsets = offsets[start:stop]
        content_zero_width_mask = zero_width_mask[start:stop]
        input_ids = [*content_ids, marker_id]
        rows.append(
            {
                "example_id": example_id,
                "split": split,
                "parser_version": PARSER_V2_VERSION,
                "cache_schema_version": PARSER_V2_CACHE_SCHEMA_VERSION,
                "chunk_order": int(chunk_order),
                "source_token_start_index": int(start),
                "source_token_end_index": int(stop),
                "content_token_ids": list(content_ids),
                "source_offset_starts": [int(item[0]) for item in content_offsets],
                "source_offset_ends": [int(item[1]) for item in content_offsets],
                "source_offset_zero_width_mask": list(
                    content_zero_width_mask
                ),
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "content_token_count": int(len(content_ids)),
                "token_length": int(len(content_ids)),
                "zero_width_token_count": int(
                    sum(content_zero_width_mask)
                ),
                "empty_source_chunk": len(content_ids) == 0,
                "chunk_sum_token_id": marker_id,
                "augmented_tokenizer_sha256": tokenizer_sha256,
            }
        )
    return rows


def build_parser_v2_cache_frames(
    examples: pd.DataFrame,
    *,
    tokenizer_binding: LockedTokenizerBindingV2,
    split: str = IMPLEMENTATION_STAGE_SPLIT,
    parser_config: ParserV2Config = ParserV2Config(),
    readiness: str = EXTRACTION_READY_READINESS,
    offset_contract_metadata: dict[str, object] | None = None,
) -> ParserV2CacheFrames:
    _ensure_train_only(examples, split)
    if readiness != EXTRACTION_READY_READINESS:
        raise ValueError("actual parser-v2 cache construction must be extraction-ready")
    validate_tokenizer_binding_v2(
        tokenizer_binding,
        require_extraction_ready=True,
    )
    tokenizer = tokenizer_binding.tokenizer
    tokenizer_preflight = tokenizer_binding.preflight
    if not tokenizer_preflight.passed:
        raise ValueError("augmented tokenizer preflight has not passed")
    if (
        tokenizer_preflight.embedding_rows_match is not True
        or not tokenizer_preflight.model_compatibility_checked
        or not tokenizer_preflight.extraction_ready
    ):
        raise ValueError("model/checkpoint tokenizer compatibility gate has not passed")
    if parser_config.max_content_tokens != LOCKED_LOCAL_CONTENT_TOKENS:
        raise ValueError("parser-v2 content-token budget must remain 511")
    validate_zero_width_semantics = offset_contract_metadata is not None
    if validate_zero_width_semantics:
        expected_contract = {
            "offset_contract_version": OFFSET_CONTRACT_VERSION,
            "native_zero_width_offsets_allowed": True,
            "native_zero_width_offsets_preserved": True,
            "offset_repair_applied": False,
            "pre_tokenizer_type": "ByteLevel",
            "pre_tokenizer_trim_offsets": True,
            "pre_tokenizer_add_prefix_space": False,
            "pre_tokenizer_use_regex": True,
            "post_processor_type": "ByteLevel",
            "post_processor_trim_offsets": True,
            "post_processor_add_prefix_space": False,
            "post_processor_use_regex": True,
        }
        if any(
            offset_contract_metadata.get(key) != value
            for key, value in expected_contract.items()
        ):
            raise ValueError("native zero-width offset contract metadata changed")

    note_rows: list[dict[str, object]] = []
    occurrence_rows: list[dict[str, object]] = []
    span_rows: list[dict[str, object]] = []
    token_rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []
    tokenizer_sha = tokenizer_preflight.tokenizer_json_sha256

    for row_ordinal, (_, row) in enumerate(examples.iterrows()):
        example_id = str(row["example_id"])
        text = row["text"]
        prepared_inputs: list[LocalInputV2] = []

        def token_count(body: str) -> int:
            try:
                prepared = build_local_input_v2(body, tokenizer)
                if validate_zero_width_semantics:
                    validate_native_zero_width_semantics_v2(
                        body,
                        prepared.original_content_ids,
                        prepared.original_offsets,
                        prepared.original_zero_width_mask,
                        tokenizer,
                    )
            except Exception:
                raise ValueError("locked tokenizer failed occurrence tokenization") from None
            prepared_inputs.append(prepared)
            return prepared.original_content_count

        try:
            parsed = parse_sections_v2(
                text,
                token_count=token_count,
                config=parser_config,
            )
        except Exception:
            raise RuntimeError(
                f"parser-v2 cache construction failed at row ordinal {row_ordinal}"
            ) from None

        model_occurrences = list(parsed.model_sections)
        if len(model_occurrences) != len(prepared_inputs):
            raise RuntimeError("parser-v2 occurrence/token ledger alignment failed")

        prepared_by_occurrence = {
            int(occurrence.occurrence_index): prepared
            for occurrence, prepared in zip(
                model_occurrences,
                prepared_inputs,
                strict=True,
            )
        }

        if parsed.fallback_required:
            chunks_for_note = _fallback_chunk_rows(
                example_id=example_id,
                text=text,
                split=split,
                tokenizer=tokenizer,
                tokenizer_sha256=tokenizer_sha,
                validate_zero_width_semantics=validate_zero_width_semantics,
            )
            chunk_rows.extend(chunks_for_note)
        else:
            chunks_for_note = []

        for occurrence in parsed.occurrences:
            prepared = prepared_by_occurrence.get(int(occurrence.occurrence_index))
            full_count = int(prepared.original_content_count) if prepared else 0
            retained_count = int(prepared.retained_content_count) if prepared else 0
            dropped_count = int(prepared.dropped_content_count) if prepared else 0
            zero_width_count = (
                int(prepared.zero_width_token_count) if prepared else 0
            )
            retained_zero_width_count = (
                int(prepared.retained_zero_width_token_count)
                if prepared
                else 0
            )
            dropped_zero_width_count = (
                int(prepared.dropped_zero_width_token_count)
                if prepared
                else 0
            )
            if full_count != int(occurrence.token_length):
                raise RuntimeError("parser-v2 occurrence token count mismatch")
            occurrence_rows.append(
                {
                    "example_id": example_id,
                    "split": split,
                    "parser_version": PARSER_V2_VERSION,
                    "cache_schema_version": PARSER_V2_CACHE_SCHEMA_VERSION,
                    "occurrence_index": int(occurrence.occurrence_index),
                    "source_order": int(occurrence.source_order),
                    "canonical_section": occurrence.canonical,
                    "heading_kind": occurrence.heading_kind,
                    "occurrence_origin": occurrence.occurrence_origin,
                    "heading_start": occurrence.heading_start,
                    "heading_end": occurrence.heading_end,
                    "body_start": int(occurrence.body_start),
                    "body_end": int(occurrence.body_end),
                    "content_start": int(occurrence.content_start),
                    "content_end": int(occurrence.content_end),
                    "is_model_slot": bool(occurrence.is_model_slot),
                    "empty_body": not bool(occurrence.is_model_slot),
                    "full_content_token_count": full_count,
                    "retained_content_token_count": retained_count,
                    "dropped_content_token_count": dropped_count,
                    "zero_width_token_count": zero_width_count,
                    "retained_zero_width_token_count": (
                        retained_zero_width_count
                    ),
                    "dropped_zero_width_token_count": (
                        dropped_zero_width_count
                    ),
                    "canonical_occurrence_index": int(
                        occurrence.canonical_occurrence_index
                    ),
                    "canonical_occurrence_count": int(
                        occurrence.canonical_occurrence_count
                    ),
                    "canonical_model_slot_index": (
                        None
                        if occurrence.canonical_model_slot_index is None
                        else int(occurrence.canonical_model_slot_index)
                    ),
                    "canonical_model_slot_count": int(
                        occurrence.canonical_model_slot_count
                    ),
                    "model_slot_index": (
                        None
                        if occurrence.model_slot_index is None
                        else int(occurrence.model_slot_index)
                    ),
                    "is_repeated_canonical": bool(occurrence.repeated_section),
                    "use_chunk_fallback": bool(parsed.fallback_required),
                    "augmented_tokenizer_sha256": tokenizer_sha,
                }
            )
            if prepared is not None:
                token_rows.append(
                    _token_row(
                        example_id=example_id,
                        split=split,
                        occurrence=occurrence,
                        local_input=prepared,
                        tokenizer_sha256=tokenizer_sha,
                    )
                )

        for span_order, span in enumerate(parsed.ledger):
            span_rows.append(
                {
                    "example_id": example_id,
                    "split": split,
                    "parser_version": PARSER_V2_VERSION,
                    "cache_schema_version": PARSER_V2_CACHE_SCHEMA_VERSION,
                    "span_order": int(span_order),
                    "span_kind": span.kind,
                    "span_start": int(span.start),
                    "span_end": int(span.end),
                    "occurrence_index": span.occurrence_index,
                    "canonical_section": span.canonical,
                }
            )

        occurrence_zero_width_count = int(
            sum(prepared.zero_width_token_count for prepared in prepared_inputs)
        )
        retained_occurrence_zero_width_count = int(
            sum(
                prepared.retained_zero_width_token_count
                for prepared in prepared_inputs
            )
        )
        dropped_occurrence_zero_width_count = int(
            sum(
                prepared.dropped_zero_width_token_count
                for prepared in prepared_inputs
            )
        )
        fallback_zero_width_count = int(
            sum(
                int(chunk["zero_width_token_count"])
                for chunk in chunks_for_note
            )
        )
        note_rows.append(
            {
                "example_id": example_id,
                "split": split,
                "parser_version": PARSER_V2_VERSION,
                "cache_schema_version": PARSER_V2_CACHE_SCHEMA_VERSION,
                "occurrence_count": int(len(parsed.occurrences)),
                "model_slot_count": int(len(model_occurrences)),
                "heading_candidate_count": int(parsed.heading_candidate_count),
                "recognized_heading_count": int(parsed.recognized_heading_count),
                "unrecognized_heading_count": int(parsed.unrecognized_heading_count),
                "empty_body_occurrence_count": int(
                    parsed.empty_body_occurrence_count
                ),
                "max_section_tokens": int(parsed.max_section_tokens),
                "few_sections": bool(parsed.few_sections),
                "too_long": bool(parsed.too_long),
                "structurally_usable": bool(parsed.structurally_usable),
                "use_chunk_fallback": bool(parsed.fallback_required),
                "evidence_preservation_coverage": float(
                    parsed.evidence_preservation_coverage
                ),
                "recognized_canonical_coverage": (
                    None
                    if parsed.recognized_canonical_coverage is None
                    else float(parsed.recognized_canonical_coverage)
                ),
                "source_character_count": int(parsed.source_character_count),
                "nonheading_character_count": int(
                    parsed.nonheading_character_count
                ),
                "recognized_canonical_character_count": int(
                    parsed.recognized_canonical_character_count
                ),
                "omitted_character_count": int(parsed.omitted_character_count),
                "duplicated_character_count": int(
                    parsed.duplicated_character_count
                ),
                "overlap_span_count": int(parsed.overlap_span_count),
                "synthetic_repeat_markers_injected": int(
                    parsed.synthetic_repeat_marker_count
                ),
                "fallback_chunk_count": int(len(chunks_for_note)),
                "occurrence_zero_width_token_count": (
                    occurrence_zero_width_count
                ),
                "retained_occurrence_zero_width_token_count": (
                    retained_occurrence_zero_width_count
                ),
                "dropped_occurrence_zero_width_token_count": (
                    dropped_occurrence_zero_width_count
                ),
                "fallback_zero_width_token_count": (
                    fallback_zero_width_count
                ),
                "has_occurrence_zero_width": bool(
                    occurrence_zero_width_count > 0
                ),
                "augmented_tokenizer_sha256": tokenizer_sha,
            }
        )

    note_index = pd.DataFrame(note_rows, columns=NOTE_INDEX_COLUMNS)
    occurrence_ledger = pd.DataFrame(
        occurrence_rows,
        columns=OCCURRENCE_LEDGER_COLUMNS,
    )
    span_ledger = pd.DataFrame(span_rows, columns=SPAN_LEDGER_COLUMNS)
    token_ledger = pd.DataFrame(token_rows, columns=TOKEN_LEDGER_COLUMNS)
    fallback_chunks = pd.DataFrame(chunk_rows, columns=FALLBACK_CHUNK_COLUMNS)

    if len(note_index) != len(examples):
        raise RuntimeError("parser-v2 note index is not one row per input example")
    if note_index["example_id"].astype(str).duplicated().any():
        raise RuntimeError("parser-v2 note index contains duplicate identifiers")
    fallback_expected = int(note_index["use_chunk_fallback"].astype(bool).sum())
    fallback_observed = int(
        fallback_chunks["example_id"].astype(str).nunique()
        if not fallback_chunks.empty
        else 0
    )
    if fallback_observed != fallback_expected:
        raise RuntimeError("parser-v2 fallback chunk membership is incomplete")

    example_count = int(len(note_index))
    fallback_count = int(note_index["use_chunk_fallback"].astype(bool).sum())
    zero_slot_count = int((note_index["model_slot_count"].astype(int) == 0).sum())
    occurrence_content_token_count = int(
        token_ledger["full_content_token_count"].astype(int).sum()
    )
    occurrence_zero_width_token_count = int(
        token_ledger["zero_width_token_count"].astype(int).sum()
    )
    retained_occurrence_zero_width_token_count = int(
        token_ledger["retained_zero_width_token_count"].astype(int).sum()
    )
    dropped_occurrence_zero_width_token_count = int(
        token_ledger["dropped_zero_width_token_count"].astype(int).sum()
    )
    fallback_content_token_count = int(
        fallback_chunks["content_token_count"].astype(int).sum()
    )
    fallback_zero_width_token_count = int(
        fallback_chunks["zero_width_token_count"].astype(int).sum()
    )
    occurrence_zero_width_example_count = int(
        note_index["has_occurrence_zero_width"].astype(bool).sum()
    )
    contract_metadata = {
        "offset_contract_version": OFFSET_CONTRACT_VERSION,
        "native_zero_width_offsets_allowed": True,
        "native_zero_width_offsets_preserved": True,
        "offset_repair_applied": False,
        "zero_width_semantics_validated": bool(
            validate_zero_width_semantics
        ),
        "zero_width_non_whitespace_token_count": 0,
        "zero_width_special_token_count": 0,
        "zero_width_nonadjacent_anchor_count": 0,
    }
    if offset_contract_metadata is not None:
        contract_metadata.update(offset_contract_metadata)
    summary = {
        "status": "EXTRACTION_READY_VALID",
        "stage": "PARSER_V2_IMPLEMENTATION_TRAIN_ONLY_CACHE",
        "cache_readiness": readiness,
        "extraction_ready": True,
        "checkpoint_identity_bound": True,
        "model_embedding_rows_checked": True,
        "split": split,
        "parser_version": PARSER_V2_VERSION,
        "cache_schema_version": PARSER_V2_CACHE_SCHEMA_VERSION,
        "example_count": example_count,
        "occurrence_count": int(len(occurrence_ledger)),
        "model_slot_count": int(len(token_ledger)),
        "span_count": int(len(span_ledger)),
        "fallback_example_count": fallback_count,
        "fallback_example_rate": (
            float(fallback_count / example_count) if example_count else 0.0
        ),
        "fallback_chunk_count": int(len(fallback_chunks)),
        **contract_metadata,
        "occurrence_content_token_count": occurrence_content_token_count,
        "occurrence_zero_width_token_count": (
            occurrence_zero_width_token_count
        ),
        "occurrence_zero_width_token_rate": (
            float(
                occurrence_zero_width_token_count
                / occurrence_content_token_count
            )
            if occurrence_content_token_count
            else 0.0
        ),
        "retained_occurrence_zero_width_token_count": (
            retained_occurrence_zero_width_token_count
        ),
        "dropped_occurrence_zero_width_token_count": (
            dropped_occurrence_zero_width_token_count
        ),
        "occurrence_zero_width_example_count": (
            occurrence_zero_width_example_count
        ),
        "occurrence_zero_width_example_rate": (
            float(occurrence_zero_width_example_count / example_count)
            if example_count
            else 0.0
        ),
        "fallback_content_token_count": fallback_content_token_count,
        "fallback_zero_width_token_count": fallback_zero_width_token_count,
        "fallback_zero_width_token_rate": (
            float(fallback_zero_width_token_count / fallback_content_token_count)
            if fallback_content_token_count
            else 0.0
        ),
        "offset_count_mismatch_count": 0,
        "offset_reversed_count": 0,
        "offset_out_of_range_count": 0,
        "offset_positive_width_overlap_count": 0,
        "zero_model_slot_example_count": zero_slot_count,
        "zero_model_slot_fallback_preserved": bool(
            (
                note_index.loc[
                    note_index["model_slot_count"].astype(int) == 0,
                    "use_chunk_fallback",
                ]
                .astype(bool)
                .all()
            )
        ),
        "evidence_preservation_all_one": bool(
            note_index["evidence_preservation_coverage"].eq(1.0).all()
        ),
        "omitted_character_count": int(
            note_index["omitted_character_count"].astype(int).sum()
        ),
        "duplicated_character_count": int(
            note_index["duplicated_character_count"].astype(int).sum()
        ),
        "overlap_span_count": int(
            note_index["overlap_span_count"].astype(int).sum()
        ),
        "synthetic_repeat_markers_injected": int(
            note_index["synthetic_repeat_markers_injected"].astype(int).sum()
        ),
        "canonical_row_set_sha256": canonical_row_set_sha256(
            examples["example_id"],
            split,
        ),
        "augmented_tokenizer_sha256": tokenizer_sha,
        "test_rows_opened": 0,
        "validation_rows_opened": 0,
    }
    if (
        not summary["zero_model_slot_fallback_preserved"]
        or not summary["evidence_preservation_all_one"]
        or summary["omitted_character_count"] != 0
        or summary["duplicated_character_count"] != 0
        or summary["overlap_span_count"] != 0
        or summary["synthetic_repeat_markers_injected"] != 0
        or int(
            note_index["occurrence_zero_width_token_count"]
            .astype(int)
            .sum()
        )
        != occurrence_zero_width_token_count
        or int(
            note_index["retained_occurrence_zero_width_token_count"]
            .astype(int)
            .sum()
        )
        != retained_occurrence_zero_width_token_count
        or int(
            note_index["dropped_occurrence_zero_width_token_count"]
            .astype(int)
            .sum()
        )
        != dropped_occurrence_zero_width_token_count
        or int(
            note_index["fallback_zero_width_token_count"].astype(int).sum()
        )
        != fallback_zero_width_token_count
        or (
            retained_occurrence_zero_width_token_count
            + dropped_occurrence_zero_width_token_count
            != occurrence_zero_width_token_count
        )
    ):
        raise RuntimeError("parser-v2 cache conservation gate failed")
    validate_tokenizer_binding_v2(
        tokenizer_binding,
        require_extraction_ready=True,
    )

    return ParserV2CacheFrames(
        note_index=note_index,
        occurrence_ledger=occurrence_ledger,
        span_ledger=span_ledger,
        token_ledger=token_ledger,
        fallback_chunks=fallback_chunks,
        summary=summary,
    )


def _write_sha256_manifest(paths: list[Path], output_path: Path) -> str:
    lines = [
        f"{sha256_path(path)}  {path.name}"
        for path in sorted(paths, key=lambda item: item.name)
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_path(output_path)


def _verify_sha256_manifest(paths: list[Path], manifest_path: Path) -> str:
    expected = {
        path.name: sha256_path(path)
        for path in sorted(paths, key=lambda item: item.name)
    }
    observed: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError("parser-v2 cache SHA-256 manifest is malformed")
        digest, name = fields[0], fields[1].strip()
        if name in observed:
            raise RuntimeError("parser-v2 cache SHA-256 manifest has a duplicate entry")
        observed[name] = digest
    if observed != expected:
        raise RuntimeError("parser-v2 cache SHA-256 manifest verification failed")
    return sha256_path(manifest_path)


def write_parser_v2_cache(
    frames: ParserV2CacheFrames,
    *,
    output_dir: str | Path,
    protocol_sha256: str,
    code_sha256: dict[str, str],
    tokenizer_preflight: TokenizerPreflightV2,
    offset_amendment_sha256: str | None = None,
) -> dict[str, object]:
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError("parser-v2 output directory already exists")
    if (
        not tokenizer_preflight.passed
        or not tokenizer_preflight.extraction_ready
        or tokenizer_preflight.embedding_rows_match is not True
        or frames.summary.get("extraction_ready") is not True
        or not frames.summary.get("strict_model_state_load", False)
        or (
            offset_amendment_sha256 is not None
            and frames.summary.get("zero_width_semantics_validated") is not True
        )
    ):
        raise ValueError("parser-v2 cache cannot be written without model-bound readiness")
    root.mkdir(parents=True, exist_ok=False)

    frame_by_name = {
        "note_index": frames.note_index,
        "occurrence_ledger": frames.occurrence_ledger,
        "span_ledger": frames.span_ledger,
        "token_ledger": frames.token_ledger,
        "fallback_chunks": frames.fallback_chunks,
    }
    restricted_paths: list[Path] = []
    for name, filename in RESTRICTED_CACHE_FILENAMES.items():
        path = root / filename
        frame_by_name[name].to_parquet(path, index=False)
        restricted_paths.append(path)

    restriction_notice = root / "RESTRICTED_ARTIFACTS_DO_NOT_SYNC.txt"
    restriction_notice.write_text(
        "This directory contains identifier-, offset-, and token-level restricted artifacts.\n"
        "Only separately privacy-validated aggregates may leave the trusted environment.\n",
        encoding="utf-8",
    )
    restricted_paths.append(restriction_notice)

    summary_path = root / "PARSER_V2_CACHE_SUMMARY_RESTRICTED.csv"
    pd.DataFrame([frames.summary]).to_csv(summary_path, index=False)

    preflight_public = {
        "passed": bool(tokenizer_preflight.passed),
        "tokenizer_json_sha256": tokenizer_preflight.tokenizer_json_sha256,
        "expected_tokenizer_json_sha256": (
            tokenizer_preflight.expected_tokenizer_json_sha256
        ),
        "vocab_size": int(tokenizer_preflight.vocab_size),
        "is_fast": bool(tokenizer_preflight.is_fast),
        "offset_mapping_supported": bool(
            tokenizer_preflight.offset_mapping_supported
        ),
        "all_project_tokens_single": bool(
            tokenizer_preflight.all_project_tokens_single
        ),
        "padding_side": tokenizer_preflight.padding_side,
        "add_special_tokens_policy": bool(
            tokenizer_preflight.add_special_tokens_policy
        ),
    }
    provenance = {
        "status": "EXTRACTION_READY_VALID",
        "stage": "PARSER_V2_IMPLEMENTATION_TRAIN_ONLY_CACHE",
        "cache_readiness": frames.summary.get(
            "cache_readiness",
            EXTRACTION_READY_READINESS,
        ),
        "extraction_ready": True,
        "checkpoint_identity_bound": True,
        "model_embedding_rows_checked": True,
        "protocol_sha256": protocol_sha256,
        "offset_amendment_sha256": offset_amendment_sha256,
        "parser_version": PARSER_V2_VERSION,
        "cache_schema_version": PARSER_V2_CACHE_SCHEMA_VERSION,
        "offset_contract_version": frames.summary.get(
            "offset_contract_version"
        ),
        "native_zero_width_offsets_allowed": bool(
            frames.summary.get("native_zero_width_offsets_allowed", False)
        ),
        "native_zero_width_offsets_preserved": bool(
            frames.summary.get("native_zero_width_offsets_preserved", False)
        ),
        "offset_repair_applied": bool(
            frames.summary.get("offset_repair_applied", False)
        ),
        "pre_tokenizer_type": frames.summary.get("pre_tokenizer_type"),
        "pre_tokenizer_trim_offsets": frames.summary.get(
            "pre_tokenizer_trim_offsets"
        ),
        "pre_tokenizer_add_prefix_space": frames.summary.get(
            "pre_tokenizer_add_prefix_space"
        ),
        "pre_tokenizer_use_regex": frames.summary.get(
            "pre_tokenizer_use_regex"
        ),
        "post_processor_type": frames.summary.get("post_processor_type"),
        "post_processor_trim_offsets": frames.summary.get(
            "post_processor_trim_offsets"
        ),
        "post_processor_add_prefix_space": frames.summary.get(
            "post_processor_add_prefix_space"
        ),
        "post_processor_use_regex": frames.summary.get(
            "post_processor_use_regex"
        ),
        "zero_width_semantics_validated": bool(
            frames.summary.get("zero_width_semantics_validated", False)
        ),
        "zero_width_non_whitespace_token_count": frames.summary.get(
            "zero_width_non_whitespace_token_count"
        ),
        "zero_width_special_token_count": frames.summary.get(
            "zero_width_special_token_count"
        ),
        "zero_width_nonadjacent_anchor_count": frames.summary.get(
            "zero_width_nonadjacent_anchor_count"
        ),
        "occurrence_content_token_count": frames.summary.get(
            "occurrence_content_token_count"
        ),
        "occurrence_zero_width_token_count": frames.summary.get(
            "occurrence_zero_width_token_count"
        ),
        "occurrence_zero_width_token_rate": frames.summary.get(
            "occurrence_zero_width_token_rate"
        ),
        "retained_occurrence_zero_width_token_count": frames.summary.get(
            "retained_occurrence_zero_width_token_count"
        ),
        "dropped_occurrence_zero_width_token_count": frames.summary.get(
            "dropped_occurrence_zero_width_token_count"
        ),
        "occurrence_zero_width_example_count": frames.summary.get(
            "occurrence_zero_width_example_count"
        ),
        "occurrence_zero_width_example_rate": frames.summary.get(
            "occurrence_zero_width_example_rate"
        ),
        "fallback_content_token_count": frames.summary.get(
            "fallback_content_token_count"
        ),
        "fallback_zero_width_token_count": frames.summary.get(
            "fallback_zero_width_token_count"
        ),
        "fallback_zero_width_token_rate": frames.summary.get(
            "fallback_zero_width_token_rate"
        ),
        "offset_count_mismatch_count": frames.summary.get(
            "offset_count_mismatch_count"
        ),
        "offset_reversed_count": frames.summary.get("offset_reversed_count"),
        "offset_out_of_range_count": frames.summary.get(
            "offset_out_of_range_count"
        ),
        "offset_positive_width_overlap_count": frames.summary.get(
            "offset_positive_width_overlap_count"
        ),
        "canonical_row_set_sha256": frames.summary[
            "canonical_row_set_sha256"
        ],
        "base_tokenizer_json_sha256": frames.summary.get(
            "base_tokenizer_json_sha256"
        ),
        "benchmark_columns_opened": frames.summary.get(
            "benchmark_columns_opened"
        ),
        "outcome_label_column_opened": bool(
            frames.summary.get("outcome_label_column_opened", False)
        ),
        "patient_identifier_column_opened": bool(
            frames.summary.get("patient_identifier_column_opened", False)
        ),
        "model_config_sha256": frames.summary.get("model_config_sha256"),
        "checkpoint_state_sha256": frames.summary.get(
            "checkpoint_state_sha256"
        ),
        "checkpoint_global_step": frames.summary.get(
            "checkpoint_global_step"
        ),
        "checkpoint_format": frames.summary.get("checkpoint_format"),
        "model_embedding_rows": frames.summary.get("model_embedding_rows"),
        "tokenizer_vocab_size": frames.summary.get("tokenizer_vocab_size"),
        "strict_model_state_load": bool(
            frames.summary.get("strict_model_state_load", False)
        ),
        "tokenizer_preflight": preflight_public,
        "code_sha256": dict(sorted(code_sha256.items())),
        "restricted_artifact_sha256": {
            path.name: sha256_path(path) for path in restricted_paths
        },
        "validation_rows_opened": 0,
        "test_rows_opened": 0,
        "raw_note_exported": False,
        "raw_heading_exported": False,
    }
    provenance_path = root / "PARSER_V2_CACHE_PROVENANCE.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_path = root / "PARSER_V2_CACHE_SHA256SUMS.txt"
    canonical_paths = [*restricted_paths, summary_path, provenance_path]
    _write_sha256_manifest(canonical_paths, manifest_path)
    manifest_sha = _verify_sha256_manifest(canonical_paths, manifest_path)
    marker_path = root / "PARSER_V2_CACHE_IMPLEMENTATION.complete"
    marker_path.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "stage": "PARSER_V2_IMPLEMENTATION_TRAIN_ONLY_CACHE",
                "manifest_sha256": manifest_sha,
                "parser_version": PARSER_V2_VERSION,
                "cache_schema_version": PARSER_V2_CACHE_SCHEMA_VERSION,
                "offset_contract_version": frames.summary.get(
                    "offset_contract_version"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "COMPLETE",
        "output_dir": str(root),
        "manifest_sha256": manifest_sha,
        "summary": frames.summary,
    }


def tokenizer_preflight_public_dict(
    preflight: TokenizerPreflightV2,
) -> dict[str, object]:
    value = asdict(preflight)
    for restricted_field in [
        "pad_token_id",
        "eos_token_id",
        "sec_sum_token_id",
        "embedding_rows",
    ]:
        value.pop(restricted_field, None)
    return value
