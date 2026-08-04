from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Literal

from notesm_v2.data.section_parser import CANONICAL_SECTIONS, HEADING_MAP, normalize_heading


PARSER_V2_VERSION = "2.0.0"

HeadingKind = Literal["recognized_heading", "unrecognized_heading", "preamble_or_gap"]
OccurrenceOrigin = Literal[
    "recognized_heading",
    "unrecognized_heading",
    "preamble",
    "inter_section_gap",
    "headerless_document",
]
LedgerSpanKind = Literal[
    "recognized_heading",
    "unrecognized_heading",
    "section_body",
    "preamble_or_gap",
]

_ALLOWED_HEADING_RE = re.compile(r"[A-Za-z][A-Za-z0-9 _/()&,'\.\-\[\]]*")
_LEADING_ENUMERATOR_RE = re.compile(
    r"^\s*(?:(?:\d{1,2}|[IVX]{1,6}|[A-Z])[\.\)]|[-*#])\s+",
    flags=re.IGNORECASE,
)

# These aliases are generic clinical-document terms. They are deliberately
# independent of the benchmark corpus and remain subject to the locked human
# parser audit before Parser-v2 can be declared valid.
_PUBLIC_HEADING_ALIASES = {
    "PRESENTING COMPLAINT": "CHIEF_COMPLAINT",
    "HISTORY PRESENT ILLNESS": "HISTORY_OF_PRESENT_ILLNESS",
    "HOSPITAL COURSE AND TREATMENT": "BRIEF_HOSPITAL_COURSE",
    "COURSE IN HOSPITAL": "BRIEF_HOSPITAL_COURSE",
    "PERTINENT LAB RESULTS": "PERTINENT_RESULTS",
    "PERTINENT LABORATORY RESULTS": "PERTINENT_RESULTS",
    "LABORATORY DATA": "PERTINENT_RESULTS",
    "DISCHARGE MEDS": "DISCHARGE_MEDICATIONS",
    "MEDICATIONS AT DISCHARGE": "DISCHARGE_MEDICATIONS",
    "CONDITION ON DISCHARGE": "DISCHARGE_CONDITION",
    "INSTRUCTIONS ON DISCHARGE": "DISCHARGE_INSTRUCTIONS",
    "FOLLOW UP APPOINTMENTS": "FOLLOW_UP",
    "FOLLOWUP": "FOLLOW_UP",
}


@dataclass(frozen=True, slots=True)
class ParserV2Config:
    min_sections: int = 2
    max_content_tokens: int = 511
    max_heading_chars: int = 80
    max_heading_words: int = 12

    def __post_init__(self) -> None:
        if self.min_sections < 1:
            raise ValueError("min_sections must be at least 1")
        if self.max_content_tokens < 1:
            raise ValueError("max_content_tokens must be at least 1")
        if self.max_heading_chars < 2:
            raise ValueError("max_heading_chars must be at least 2")
        if self.max_heading_words < 1:
            raise ValueError("max_heading_words must be at least 1")


@dataclass(frozen=True, slots=True)
class LedgerSpanV2:
    """Raw-free, half-open source span in the conservation ledger."""

    kind: LedgerSpanKind
    start: int
    end: int
    occurrence_index: int | None
    canonical: str | None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("ledger spans must be non-empty half-open intervals")
        if self.canonical is not None and self.canonical not in CANONICAL_SECTIONS:
            raise ValueError("ledger span has an invalid canonical section")


@dataclass(frozen=True, slots=True)
class SectionOccurrenceV2:
    """Occurrence-level section metadata without heading or body strings."""

    occurrence_index: int
    source_order: int
    canonical: str
    heading_kind: HeadingKind
    occurrence_origin: OccurrenceOrigin
    heading_start: int | None
    heading_end: int | None
    body_start: int
    body_end: int
    content_start: int
    content_end: int
    token_length: int
    is_model_slot: bool
    canonical_occurrence_index: int = 1
    canonical_occurrence_count: int = 1
    canonical_model_slot_index: int | None = None
    canonical_model_slot_count: int = 0
    model_slot_index: int | None = None
    repeated_section: bool = False

    def __post_init__(self) -> None:
        if self.occurrence_index < 0 or self.source_order < 0:
            raise ValueError("occurrence indices must be non-negative")
        if self.canonical not in CANONICAL_SECTIONS:
            raise ValueError("occurrence has an invalid canonical section")
        if self.body_start < 0 or self.body_end < self.body_start:
            raise ValueError("invalid body span")
        if not (self.body_start <= self.content_start <= self.content_end <= self.body_end):
            raise ValueError("content span must be within its body span")
        if (self.heading_start is None) != (self.heading_end is None):
            raise ValueError("heading offsets must both be present or absent")
        if self.heading_start is not None:
            if self.heading_start < 0 or self.heading_end is None or self.heading_end <= self.heading_start:
                raise ValueError("invalid heading span")
            if self.heading_end != self.body_start:
                raise ValueError("body_start must immediately follow heading_end")
        if self.heading_kind == "preamble_or_gap" and self.heading_start is not None:
            raise ValueError("preamble and gap occurrences cannot have heading offsets")
        if self.heading_kind != "preamble_or_gap" and self.heading_start is None:
            raise ValueError("heading occurrences require heading offsets")
        if self.token_length < 0:
            raise ValueError("token_length must be non-negative")
        if self.is_model_slot != (self.content_end > self.content_start):
            raise ValueError("is_model_slot must match non-whitespace content")
        if self.is_model_slot != (self.model_slot_index is not None):
            raise ValueError("model_slot_index must exist exactly for model-facing occurrences")
        if self.canonical_occurrence_index < 1 or self.canonical_occurrence_count < 1:
            raise ValueError("canonical occurrence indices are one-based")
        if self.canonical_occurrence_index > self.canonical_occurrence_count:
            raise ValueError("canonical occurrence index exceeds its count")
        if self.canonical_model_slot_count < 0:
            raise ValueError("canonical model-slot count must be non-negative")
        if self.canonical_model_slot_index is not None:
            if not 1 <= self.canonical_model_slot_index <= self.canonical_model_slot_count:
                raise ValueError("invalid canonical model-slot index")

    @property
    def empty_body(self) -> bool:
        return not self.is_model_slot


@dataclass(frozen=True, slots=True)
class SectionParseResultV2:
    """Coverage-preserving parser and tokenizer-length plan for one note."""

    occurrences: tuple[SectionOccurrenceV2, ...]
    ledger: tuple[LedgerSpanV2, ...]
    fallback_required: bool
    structurally_usable: bool
    few_sections: bool
    too_long: bool
    max_section_tokens: int
    evidence_preservation_coverage: float
    recognized_canonical_coverage: float | None
    source_character_count: int
    nonheading_character_count: int
    recognized_canonical_character_count: int
    heading_candidate_count: int
    recognized_heading_count: int
    unrecognized_heading_count: int
    empty_body_occurrence_count: int
    omitted_character_count: int = 0
    duplicated_character_count: int = 0
    overlap_span_count: int = 0
    synthetic_repeat_marker_count: int = 0
    parser_version: str = PARSER_V2_VERSION

    @property
    def model_sections(self) -> tuple[SectionOccurrenceV2, ...]:
        return tuple(item for item in self.occurrences if item.is_model_slot)

    @property
    def nonempty_occurrence_count(self) -> int:
        return len(self.model_sections)


@dataclass(frozen=True, slots=True)
class _HeadingCandidate:
    line_start: int
    line_end: int
    canonical: str
    kind: Literal["recognized_heading", "unrecognized_heading"]


@dataclass(frozen=True, slots=True)
class _OccurrenceDraft:
    canonical: str
    heading_kind: HeadingKind
    occurrence_origin: OccurrenceOrigin
    heading_start: int | None
    heading_end: int | None
    body_start: int
    body_end: int


def normalize_heading_v2(heading: str) -> str:
    value = unicodedata.normalize("NFKC", heading)
    value = (
        value.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2019", "'")
        .strip()
    )
    value = _LEADING_ENUMERATOR_RE.sub("", value)
    value = value.replace("&", " AND ")
    value = value.replace("(", " ").replace(")", " ")
    value = value.replace("[", " ").replace("]", " ")
    value = value.replace("/", " ")
    value = value.replace("-", " ")
    return normalize_heading(value)


def _build_heading_map_v2() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_alias, canonical in [*HEADING_MAP.items(), *_PUBLIC_HEADING_ALIASES.items()]:
        normalized = normalize_heading_v2(raw_alias)
        previous = mapping.get(normalized)
        if previous is not None and previous != canonical:
            raise RuntimeError("normalized public heading aliases collide")
        if canonical not in CANONICAL_SECTIONS or canonical == "OTHER":
            raise RuntimeError("heading alias maps to an invalid canonical section")
        mapping[normalized] = canonical
    return mapping


HEADING_MAP_V2 = _build_heading_map_v2()


def canonicalize_heading_v2(heading: str) -> str | None:
    return HEADING_MAP_V2.get(normalize_heading_v2(heading))


def _line_records(text: str) -> Iterator[tuple[int, int, str]]:
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line_end = offset + len(raw_line)
        without_terminator = raw_line.rstrip("\r\n")
        yield offset, line_end, without_terminator
        offset = line_end
    if offset < len(text):
        # Defensive only: splitlines(keepends=True) normally covers this branch.
        yield offset, len(text), text[offset:]


def _candidate_label(line: str, config: ParserV2Config) -> str | None:
    normalized_line = unicodedata.normalize("NFKC", line)
    stripped = normalized_line.strip(" \t\f\v")
    if not stripped.endswith(":") or stripped.count(":") != 1:
        return None
    label = stripped[:-1].strip()
    deenumerated = _LEADING_ENUMERATOR_RE.sub("", label)
    if not (2 <= len(deenumerated) <= config.max_heading_chars):
        return None
    if not deenumerated or not deenumerated[0].isalpha():
        return None
    if _ALLOWED_HEADING_RE.fullmatch(deenumerated) is None:
        return None
    if len(deenumerated.split()) > config.max_heading_words:
        return None
    return deenumerated


def _find_heading_candidates(text: str, config: ParserV2Config) -> list[_HeadingCandidate]:
    candidates: list[_HeadingCandidate] = []
    for line_start, line_end, line in _line_records(text):
        label = _candidate_label(line, config)
        if label is None:
            continue
        canonical = canonicalize_heading_v2(label)
        candidates.append(
            _HeadingCandidate(
                line_start=line_start,
                line_end=line_end,
                canonical=canonical or "OTHER",
                kind="recognized_heading" if canonical is not None else "unrecognized_heading",
            )
        )
    return candidates


def _content_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = start
    while left < end and text[left].isspace():
        left += 1
    right = end
    while right > left and text[right - 1].isspace():
        right -= 1
    return left, right


def _trailing_whitespace_start(text: str, start: int, end: int) -> int:
    """Return the start of the maximal trailing-whitespace suffix."""
    cursor = end
    while cursor > start and text[cursor - 1].isspace():
        cursor -= 1
    return cursor


def validate_ledger_partition(
    text_length: int,
    ledger: tuple[LedgerSpanV2, ...] | list[LedgerSpanV2],
) -> None:
    if text_length < 0:
        raise ValueError("text_length must be non-negative")
    if text_length == 0:
        if ledger:
            raise ValueError("empty text cannot have ledger spans")
        return
    if not ledger:
        raise ValueError("non-empty text requires ledger spans")
    cursor = 0
    for span in ledger:
        if span.start != cursor:
            relation = "overlap" if span.start < cursor else "gap"
            raise ValueError(f"ledger partition has a {relation}")
        cursor = span.end
    if cursor != text_length:
        raise ValueError("ledger partition does not cover the full text")


def occurrence_body(text: str, occurrence: SectionOccurrenceV2) -> str:
    """Restricted helper: slice exact body text only inside the trusted runtime."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if occurrence.body_end > len(text):
        raise ValueError("occurrence body span exceeds source text")
    return text[occurrence.body_start : occurrence.body_end]


def _build_occurrence_drafts(
    text: str,
    candidates: list[_HeadingCandidate],
) -> list[_OccurrenceDraft]:
    if not candidates:
        return [
            _OccurrenceDraft(
                canonical="OTHER",
                heading_kind="preamble_or_gap",
                occurrence_origin="headerless_document",
                heading_start=None,
                heading_end=None,
                body_start=0,
                body_end=len(text),
            )
        ]

    drafts: list[_OccurrenceDraft] = []
    first_start = candidates[0].line_start
    if first_start > 0:
        drafts.append(
            _OccurrenceDraft(
                canonical="OTHER",
                heading_kind="preamble_or_gap",
                occurrence_origin="preamble",
                heading_start=None,
                heading_end=None,
                body_start=0,
                body_end=first_start,
            )
        )

    for index, candidate in enumerate(candidates):
        next_start = candidates[index + 1].line_start if index + 1 < len(candidates) else len(text)
        body_end = _trailing_whitespace_start(text, candidate.line_end, next_start)
        drafts.append(
            _OccurrenceDraft(
                canonical=candidate.canonical,
                heading_kind=candidate.kind,
                occurrence_origin=candidate.kind,
                heading_start=candidate.line_start,
                heading_end=candidate.line_end,
                body_start=candidate.line_end,
                body_end=body_end,
            )
        )
        if body_end < next_start:
            drafts.append(
                _OccurrenceDraft(
                    canonical="OTHER",
                    heading_kind="preamble_or_gap",
                    occurrence_origin="inter_section_gap",
                    heading_start=None,
                    heading_end=None,
                    body_start=body_end,
                    body_end=next_start,
                )
            )
    return drafts


def _safe_token_count(token_count: Callable[[str], int], body: str) -> int:
    try:
        length = int(token_count(body))
    except Exception:
        # Do not propagate tokenizer exceptions that may embed restricted text.
        raise ValueError("token counting failed") from None
    if length < 0:
        raise ValueError("token_count returned a negative value")
    return length


def parse_sections_v2(
    text: str,
    *,
    token_count: Callable[[str], int],
    config: ParserV2Config = ParserV2Config(),
) -> SectionParseResultV2:
    """Parse an explicit-heading note into ordered, raw-free occurrences.

    ``token_count`` must use the locked augmented tokenizer with
    ``add_special_tokens=False`` in production. The cache builder enforces that
    precondition. Keeping it injectable here permits deterministic unit tests
    without introducing a silent whitespace-token fallback.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not callable(token_count):
        raise TypeError("token_count must be callable")
    if not text:
        return SectionParseResultV2(
            occurrences=(),
            ledger=(),
            fallback_required=True,
            structurally_usable=False,
            few_sections=True,
            too_long=False,
            max_section_tokens=0,
            evidence_preservation_coverage=1.0,
            recognized_canonical_coverage=None,
            source_character_count=0,
            nonheading_character_count=0,
            recognized_canonical_character_count=0,
            heading_candidate_count=0,
            recognized_heading_count=0,
            unrecognized_heading_count=0,
            empty_body_occurrence_count=0,
        )

    candidates = _find_heading_candidates(text, config)
    drafts = _build_occurrence_drafts(text, candidates)
    occurrences: list[SectionOccurrenceV2] = []
    ledger: list[LedgerSpanV2] = []
    model_slot_index = 0

    for occurrence_index, draft in enumerate(drafts):
        content_start, content_end = _content_bounds(text, draft.body_start, draft.body_end)
        is_model_slot = content_end > content_start
        length = (
            _safe_token_count(token_count, text[draft.body_start : draft.body_end])
            if is_model_slot
            else 0
        )
        slot_index = model_slot_index if is_model_slot else None
        if is_model_slot:
            model_slot_index += 1
        occurrence = SectionOccurrenceV2(
            occurrence_index=occurrence_index,
            source_order=occurrence_index,
            canonical=draft.canonical,
            heading_kind=draft.heading_kind,
            occurrence_origin=draft.occurrence_origin,
            heading_start=draft.heading_start,
            heading_end=draft.heading_end,
            body_start=draft.body_start,
            body_end=draft.body_end,
            content_start=content_start,
            content_end=content_end,
            token_length=length,
            is_model_slot=is_model_slot,
            model_slot_index=slot_index,
        )
        occurrences.append(occurrence)
        if draft.heading_start is not None and draft.heading_end is not None:
            ledger.append(
                LedgerSpanV2(
                    kind=draft.heading_kind,
                    start=draft.heading_start,
                    end=draft.heading_end,
                    occurrence_index=occurrence_index,
                    canonical=draft.canonical,
                )
            )
        if draft.body_end > draft.body_start:
            ledger.append(
                LedgerSpanV2(
                    kind=(
                        "preamble_or_gap"
                        if draft.heading_kind == "preamble_or_gap"
                        else "section_body"
                    ),
                    start=draft.body_start,
                    end=draft.body_end,
                    occurrence_index=occurrence_index,
                    canonical=draft.canonical,
                )
            )

    ledger.sort(key=lambda item: (item.start, item.end))
    validate_ledger_partition(len(text), ledger)

    occurrence_counts = Counter(item.canonical for item in occurrences)
    model_counts = Counter(item.canonical for item in occurrences if item.is_model_slot)
    occurrence_seen: Counter[str] = Counter()
    model_seen: Counter[str] = Counter()
    numbered_occurrences: list[SectionOccurrenceV2] = []
    for item in occurrences:
        occurrence_seen[item.canonical] += 1
        canonical_model_slot_index: int | None = None
        if item.is_model_slot:
            model_seen[item.canonical] += 1
            canonical_model_slot_index = int(model_seen[item.canonical])
        numbered_occurrences.append(
            replace(
                item,
                canonical_occurrence_index=int(occurrence_seen[item.canonical]),
                canonical_occurrence_count=int(occurrence_counts[item.canonical]),
                canonical_model_slot_index=canonical_model_slot_index,
                canonical_model_slot_count=int(model_counts[item.canonical]),
                repeated_section=occurrence_counts[item.canonical] > 1,
            )
        )
    occurrences = numbered_occurrences

    model_sections = [item for item in occurrences if item.is_model_slot]
    few_sections = len(model_sections) < config.min_sections
    max_section_tokens = max((item.token_length for item in model_sections), default=0)
    too_long = max_section_tokens > config.max_content_tokens
    heading_characters = sum(
        span.end - span.start
        for span in ledger
        if span.kind in {"recognized_heading", "unrecognized_heading"}
    )
    nonheading_characters = len(text) - heading_characters
    recognized_characters = sum(
        span.end - span.start
        for span in ledger
        if span.kind == "section_body" and span.canonical != "OTHER"
    )
    recognized_coverage = (
        recognized_characters / nonheading_characters if nonheading_characters else None
    )
    recognized_headings = sum(item.kind == "recognized_heading" for item in candidates)
    unrecognized_headings = len(candidates) - recognized_headings
    empty_bodies = sum(
        item.heading_kind != "preamble_or_gap" and not item.is_model_slot
        for item in occurrences
    )

    return SectionParseResultV2(
        occurrences=tuple(occurrences),
        ledger=tuple(ledger),
        fallback_required=few_sections or too_long,
        structurally_usable=not few_sections,
        few_sections=few_sections,
        too_long=too_long,
        max_section_tokens=max_section_tokens,
        evidence_preservation_coverage=1.0,
        recognized_canonical_coverage=(
            None if recognized_coverage is None else float(recognized_coverage)
        ),
        source_character_count=len(text),
        nonheading_character_count=nonheading_characters,
        recognized_canonical_character_count=recognized_characters,
        heading_candidate_count=len(candidates),
        recognized_heading_count=recognized_headings,
        unrecognized_heading_count=unrecognized_headings,
        empty_body_occurrence_count=empty_bodies,
    )
