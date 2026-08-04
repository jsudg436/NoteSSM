from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from notesm_v2.data.dapt_packing import SPECIAL_TOKENS
from notesm_v2.models.notesm_v2 import SECTION_SPECIAL_TOKENS


LOCKED_BASE_TOKENIZER_JSON_SHA256 = (
    "b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf"
)
LOCKED_AUGMENTED_TOKENIZER_JSON_SHA256 = (
    "1c28570c9380134bbe49610d984439c9479aa841a8482c4187d44e4f4edecb66"
)
LOCKED_LOCAL_CONTENT_TOKENS = 511
LOCKED_SUMMARY_MARKER = "<SEC_SUM>"
OFFSET_CONTRACT_VERSION = "native_zero_width_anchor_v1"
PROJECT_SPECIAL_TOKENS = tuple(dict.fromkeys([*SPECIAL_TOKENS, *SECTION_SPECIAL_TOKENS]))
RUNTIME_EPHEMERAL_BACKEND_FIELDS = frozenset({"padding", "truncation"})


@dataclass(frozen=True, slots=True)
class TokenizerPreflightV2:
    tokenizer_json_sha256: str
    expected_tokenizer_json_sha256: str
    file_tokenizer_canonical_sha256: str
    runtime_tokenizer_canonical_sha256: str
    file_tokenizer_semantic_sha256: str
    runtime_tokenizer_semantic_sha256: str
    runtime_tokenizer_matches_file: bool
    runtime_padding_disabled: bool
    runtime_truncation_disabled: bool
    vocab_size: int
    embedding_rows: int | None = field(repr=False)
    pad_token_id: int = field(repr=False)
    eos_token_id: int = field(repr=False)
    sec_sum_token_id: int = field(repr=False)
    padding_side: str
    is_fast: bool
    offset_mapping_supported: bool
    all_project_tokens_single: bool
    embedding_rows_match: bool | None
    model_compatibility_checked: bool
    extraction_ready: bool
    add_special_tokens_policy: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LocalInputV2:
    original_content_ids: tuple[int, ...] = field(repr=False)
    retained_content_ids: tuple[int, ...] = field(repr=False)
    dropped_content_count: int
    original_offsets: tuple[tuple[int, int], ...] = field(repr=False)
    retained_offsets: tuple[tuple[int, int], ...] = field(repr=False)
    original_zero_width_mask: tuple[bool, ...] = field(repr=False)
    retained_zero_width_mask: tuple[bool, ...] = field(repr=False)
    input_ids: tuple[int, ...] = field(repr=False)
    attention_mask: tuple[int, ...] = field(repr=False)
    sec_sum_token_id: int = field(repr=False)

    def __post_init__(self) -> None:
        if not (
            len(self.original_content_ids)
            == len(self.original_offsets)
            == len(self.original_zero_width_mask)
        ):
            raise ValueError("original token, offset, and zero-width ledgers differ")
        if not (
            len(self.retained_content_ids)
            == len(self.retained_offsets)
            == len(self.retained_zero_width_mask)
        ):
            raise ValueError("retained token, offset, and zero-width ledgers differ")
        if self.dropped_content_count != (
            len(self.original_content_ids) - len(self.retained_content_ids)
        ):
            raise ValueError("dropped token count is inconsistent")
        if tuple(start == end for start, end in self.original_offsets) != (
            self.original_zero_width_mask
        ):
            raise ValueError("original zero-width mask is inconsistent")
        if tuple(start == end for start, end in self.retained_offsets) != (
            self.retained_zero_width_mask
        ):
            raise ValueError("retained zero-width mask is inconsistent")
        if self.input_ids != (*self.retained_content_ids, self.sec_sum_token_id):
            raise ValueError("direct local input contract is inconsistent")
        if len(self.attention_mask) != len(self.input_ids) or not all(
            value == 1 for value in self.attention_mask
        ):
            raise ValueError("direct local attention mask is inconsistent")

    @property
    def original_content_count(self) -> int:
        return len(self.original_content_ids)

    @property
    def retained_content_count(self) -> int:
        return len(self.retained_content_ids)

    @property
    def zero_width_token_count(self) -> int:
        return sum(self.original_zero_width_mask)

    @property
    def retained_zero_width_token_count(self) -> int:
        return sum(self.retained_zero_width_mask)

    @property
    def dropped_zero_width_token_count(self) -> int:
        return self.zero_width_token_count - self.retained_zero_width_token_count


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: str) -> str:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        raise ValueError("tokenizer JSON is malformed") from None
    canonical = json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _semantic_tokenizer_json_sha256(value: str) -> str:
    """Hash tokenizer semantics while excluding mutable execution state.

    Hugging Face fast-tokenizer calls can update the backend JSON's top-level
    ``padding`` and ``truncation`` fields even when each call explicitly uses
    ``padding=False`` and ``truncation=False``. Those fields are runtime
    execution state, not tokenizer vocabulary/normalizer/pre-tokenizer/model
    semantics. The full tokenizer file remains SHA-locked and must exactly
    match the freshly loaded runtime backend during preflight.
    """

    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        raise ValueError("tokenizer JSON is malformed") from None
    if not isinstance(parsed, dict):
        raise ValueError("tokenizer JSON root must be an object")
    semantic = dict(parsed)
    for field_name in RUNTIME_EPHEMERAL_BACKEND_FIELDS:
        semantic.pop(field_name, None)
    canonical = json.dumps(
        semantic,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _runtime_tokenizer_json(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if not callable(to_str):
        raise ValueError("runtime tokenizer does not expose its backend JSON")
    try:
        return to_str()
    except Exception:
        raise ValueError("runtime tokenizer backend serialization failed") from None


def _runtime_tokenizer_canonical_sha256(tokenizer: Any) -> str:
    return _canonical_json_sha256(_runtime_tokenizer_json(tokenizer))


def _runtime_tokenizer_semantic_sha256(tokenizer: Any) -> str:
    return _semantic_tokenizer_json_sha256(_runtime_tokenizer_json(tokenizer))


def _runtime_backend_policy_disabled(tokenizer: Any) -> tuple[bool, bool]:
    try:
        parsed = json.loads(_runtime_tokenizer_json(tokenizer))
    except json.JSONDecodeError:
        raise ValueError("runtime tokenizer backend JSON is malformed") from None
    if not isinstance(parsed, dict):
        raise ValueError("runtime tokenizer backend JSON root must be an object")
    return parsed.get("padding") is None, parsed.get("truncation") is None


def _flat_int_list(value: Any, *, name: str) -> list[int]:
    if value is None:
        raise ValueError(f"tokenizer output is missing {name}")
    rows = value.tolist() if hasattr(value, "tolist") else value
    if rows and isinstance(rows[0], (list, tuple)):
        if len(rows) != 1:
            raise ValueError(f"{name} unexpectedly contains a batch")
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
    offsets: list[tuple[int, int]] = []
    for item in rows:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("offset_mapping contains a malformed offset")
        offsets.append((int(item[0]), int(item[1])))
    return offsets


def validate_offsets_v2(
    offsets: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    text_length: int,
    expected_count: int,
) -> None:
    if len(offsets) != expected_count:
        raise ValueError("token and offset counts differ")
    previous_end = 0
    for start, end in offsets:
        if start < 0 or end < start or end > text_length:
            raise ValueError("token offset is outside the source body")
        if start < previous_end:
            raise ValueError("token offsets are not monotonic")
        previous_end = end


def _embedding_rows(model: Any | None) -> int | None:
    if model is None:
        return None
    if not hasattr(model, "get_input_embeddings"):
        raise ValueError("model does not expose input embeddings")
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) < 1:
        raise ValueError("model input embeddings do not expose a row count")
    return int(shape[0])


def validate_augmented_tokenizer_v2(
    tokenizer: Any,
    *,
    tokenizer_json_path: str | Path,
    expected_tokenizer_json_sha256: str = LOCKED_AUGMENTED_TOKENIZER_JSON_SHA256,
    model: Any | None = None,
) -> TokenizerPreflightV2:
    tokenizer_json = Path(tokenizer_json_path)
    tokenizer_json_sha256 = sha256_path(tokenizer_json)
    if tokenizer_json_sha256 != expected_tokenizer_json_sha256:
        raise ValueError("augmented tokenizer.json SHA-256 mismatch")
    tokenizer_json_text = tokenizer_json.read_text(encoding="utf-8")
    file_canonical_sha256 = _canonical_json_sha256(tokenizer_json_text)
    file_semantic_sha256 = _semantic_tokenizer_json_sha256(tokenizer_json_text)
    pristine_runtime_canonical_sha256 = _runtime_tokenizer_canonical_sha256(
        tokenizer
    )
    pristine_runtime_semantic_sha256 = _runtime_tokenizer_semantic_sha256(
        tokenizer
    )
    if pristine_runtime_canonical_sha256 != file_canonical_sha256:
        raise ValueError("runtime tokenizer does not match the verified tokenizer.json")
    if pristine_runtime_semantic_sha256 != file_semantic_sha256:
        raise ValueError("runtime tokenizer semantics do not match tokenizer.json")
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError("parser-v2 requires a fast tokenizer")
    if getattr(tokenizer, "padding_side", None) != "right":
        raise ValueError("parser-v2 requires right padding")
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise ValueError("parser-v2 tokenizer requires pad_token_id")
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise ValueError("parser-v2 tokenizer requires eos_token_id")

    special_token_ids: dict[str, int] = {}
    for token in PROJECT_SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
            raise ValueError("project special token is absent from the tokenizer")
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if [int(item) for item in encoded] != [int(token_id)]:
            raise ValueError("project special token is not exactly one token")
        special_token_ids[token] = int(token_id)
    if len(set(special_token_ids.values())) != len(special_token_ids):
        raise ValueError("project special tokens do not have unique token IDs")

    probe = "alpha beta"
    try:
        encoded_probe = tokenizer(
            probe,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_offsets_mapping=True,
        )
    except Exception as exc:
        raise ValueError("tokenizer does not support return_offsets_mapping") from exc
    probe_ids = _flat_int_list(encoded_probe.get("input_ids"), name="input_ids")
    probe_offsets = _flat_offsets(encoded_probe.get("offset_mapping"))
    validate_offsets_v2(probe_offsets, text_length=len(probe), expected_count=len(probe_ids))
    runtime_canonical_sha256 = _runtime_tokenizer_canonical_sha256(tokenizer)
    runtime_semantic_sha256 = _runtime_tokenizer_semantic_sha256(tokenizer)
    if runtime_semantic_sha256 != pristine_runtime_semantic_sha256:
        raise ValueError("runtime tokenizer semantics changed during preflight")
    padding_disabled, truncation_disabled = _runtime_backend_policy_disabled(tokenizer)
    if not padding_disabled or not truncation_disabled:
        raise ValueError(
            "runtime tokenizer padding/truncation policy was not disabled"
        )

    vocab_size = int(len(tokenizer))
    rows = _embedding_rows(model)
    embedding_rows_match = None if rows is None else rows == vocab_size
    if embedding_rows_match is False:
        raise ValueError("tokenizer vocab size and model embedding rows differ")
    model_compatibility_checked = rows is not None
    extraction_ready = bool(model_compatibility_checked and embedding_rows_match)

    return TokenizerPreflightV2(
        tokenizer_json_sha256=tokenizer_json_sha256,
        expected_tokenizer_json_sha256=expected_tokenizer_json_sha256,
        file_tokenizer_canonical_sha256=file_canonical_sha256,
        runtime_tokenizer_canonical_sha256=runtime_canonical_sha256,
        file_tokenizer_semantic_sha256=file_semantic_sha256,
        runtime_tokenizer_semantic_sha256=runtime_semantic_sha256,
        runtime_tokenizer_matches_file=True,
        runtime_padding_disabled=padding_disabled,
        runtime_truncation_disabled=truncation_disabled,
        vocab_size=vocab_size,
        embedding_rows=rows,
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        sec_sum_token_id=special_token_ids[LOCKED_SUMMARY_MARKER],
        padding_side=str(tokenizer.padding_side),
        is_fast=True,
        offset_mapping_supported=True,
        all_project_tokens_single=True,
        embedding_rows_match=embedding_rows_match,
        model_compatibility_checked=model_compatibility_checked,
        extraction_ready=extraction_ready,
        add_special_tokens_policy=False,
        passed=True,
    )


def validate_runtime_tokenizer_binding_v2(
    tokenizer: Any,
    preflight: TokenizerPreflightV2,
    *,
    require_extraction_ready: bool = True,
) -> None:
    """Bind an in-memory tokenizer to its verified artifact and model gate."""
    if not preflight.passed:
        raise ValueError("parser-v2 tokenizer preflight has not passed")
    observed_runtime_sha = _runtime_tokenizer_canonical_sha256(tokenizer)
    if observed_runtime_sha != preflight.runtime_tokenizer_canonical_sha256:
        raise ValueError("runtime tokenizer changed after preflight")
    observed_runtime_semantic_sha = _runtime_tokenizer_semantic_sha256(tokenizer)
    if (
        observed_runtime_semantic_sha
        != preflight.runtime_tokenizer_semantic_sha256
    ):
        raise ValueError("runtime tokenizer semantics changed after preflight")
    if (
        preflight.file_tokenizer_semantic_sha256
        != observed_runtime_semantic_sha
    ):
        raise ValueError("runtime tokenizer is not bound to the verified artifact")
    if not preflight.runtime_tokenizer_matches_file:
        raise ValueError("runtime tokenizer/file binding has not passed")
    padding_disabled, truncation_disabled = _runtime_backend_policy_disabled(tokenizer)
    if (
        not padding_disabled
        or not truncation_disabled
        or not preflight.runtime_padding_disabled
        or not preflight.runtime_truncation_disabled
    ):
        raise ValueError("runtime tokenizer padding/truncation policy changed")
    if int(len(tokenizer)) != int(preflight.vocab_size):
        raise ValueError("runtime tokenizer vocabulary changed after preflight")
    if getattr(tokenizer, "padding_side", None) != preflight.padding_side:
        raise ValueError("runtime tokenizer padding policy changed after preflight")
    if int(getattr(tokenizer, "pad_token_id", -1)) != preflight.pad_token_id:
        raise ValueError("runtime tokenizer pad token changed after preflight")
    if int(getattr(tokenizer, "eos_token_id", -1)) != preflight.eos_token_id:
        raise ValueError("runtime tokenizer eos token changed after preflight")
    if (
        int(tokenizer.convert_tokens_to_ids(LOCKED_SUMMARY_MARKER))
        != preflight.sec_sum_token_id
    ):
        raise ValueError("runtime tokenizer summary marker changed after preflight")
    if require_extraction_ready and not preflight.extraction_ready:
        raise ValueError("model/tokenizer compatibility was not verified")


def validate_model_bound_tokenizer_v2(
    tokenizer: Any,
    preflight: TokenizerPreflightV2,
    model: Any,
) -> TokenizerPreflightV2:
    """Promote a verified tokenizer-only binding to extraction readiness."""

    validate_runtime_tokenizer_binding_v2(
        tokenizer,
        preflight,
        require_extraction_ready=False,
    )
    rows = _embedding_rows(model)
    vocab_size = int(len(tokenizer))
    embedding_rows_match = rows == vocab_size
    if not embedding_rows_match:
        raise ValueError("tokenizer vocab size and model embedding rows differ")
    return replace(
        preflight,
        vocab_size=vocab_size,
        embedding_rows=rows,
        embedding_rows_match=True,
        model_compatibility_checked=True,
        extraction_ready=True,
        passed=True,
    )


def build_local_input_v2(
    body: str,
    tokenizer: Any,
    *,
    max_content_tokens: int = LOCKED_LOCAL_CONTENT_TOKENS,
    marker_token: str = LOCKED_SUMMARY_MARKER,
) -> LocalInputV2:
    if not isinstance(body, str):
        raise TypeError("body must be a string")
    if max_content_tokens != LOCKED_LOCAL_CONTENT_TOKENS:
        raise ValueError("parser-v2 local content budget is locked to 511")
    if marker_token != LOCKED_SUMMARY_MARKER:
        raise ValueError("parser-v2 summary marker is locked to <SEC_SUM>")
    marker_id = tokenizer.convert_tokens_to_ids(marker_token)
    if marker_id is None or marker_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("summary marker is absent from the tokenizer")
    if [int(item) for item in tokenizer.encode(marker_token, add_special_tokens=False)] != [
        int(marker_id)
    ]:
        raise ValueError("summary marker must be exactly one token")

    encoded = tokenizer(
        body,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    content_ids = _flat_int_list(encoded.get("input_ids"), name="input_ids")
    offsets = _flat_offsets(encoded.get("offset_mapping"))
    validate_offsets_v2(offsets, text_length=len(body), expected_count=len(content_ids))

    retained_ids = content_ids[:max_content_tokens]
    retained_offsets = offsets[:max_content_tokens]
    zero_width_mask = [start == end for start, end in offsets]
    retained_zero_width_mask = zero_width_mask[:max_content_tokens]
    input_ids = [*retained_ids, int(marker_id)]
    return LocalInputV2(
        original_content_ids=tuple(content_ids),
        retained_content_ids=tuple(retained_ids),
        dropped_content_count=len(content_ids) - len(retained_ids),
        original_offsets=tuple(offsets),
        retained_offsets=tuple(retained_offsets),
        original_zero_width_mask=tuple(zero_width_mask),
        retained_zero_width_mask=tuple(retained_zero_width_mask),
        input_ids=tuple(input_ids),
        attention_mask=tuple([1] * len(input_ids)),
        sec_sum_token_id=int(marker_id),
    )
