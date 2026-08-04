from __future__ import annotations

from dataclasses import asdict

import pytest

from notesm_v2.data.section_parser_v2 import (
    PARSER_V2_VERSION,
    ParserV2Config,
    occurrence_body,
    parse_sections_v2,
    validate_ledger_partition,
)


def whitespace_token_count(text: str) -> int:
    return len(text.split())


def heading_occurrences(result):
    return tuple(
        occurrence
        for occurrence in result.occurrences
        if occurrence.heading_kind != "preamble_or_gap"
    )


def assert_dense_ledger(text: str, result) -> None:
    validate_ledger_partition(len(text), result.ledger)
    assert sum(span.end - span.start for span in result.ledger) == len(text)
    assert result.omitted_character_count == 0
    assert result.duplicated_character_count == 0
    assert result.overlap_span_count == 0
    for left, right in zip(result.ledger, result.ledger[1:]):
        assert left.end == right.start


def test_offsets_are_raw_free_and_cover_unicode_crlf_source_exactly() -> None:
    raw_heading = "Unpublished Synthetic Heading"
    raw_body = "PRIVATE_SYNTHETIC_BODY_Ω🙂"
    text = f"front matter\r\n{raw_heading}:\r\n  {raw_body}  \r\n"

    result = parse_sections_v2(text, token_count=whitespace_token_count)

    assert result.parser_version == PARSER_V2_VERSION
    assert_dense_ledger(text, result)

    unknown = next(
        occurrence
        for occurrence in result.occurrences
        if occurrence.heading_kind == "unrecognized_heading"
    )
    heading_start = text.index(raw_heading)
    expected_body_start = text.index("\r\n", heading_start) + 2
    assert unknown.heading_start == heading_start
    assert unknown.heading_end == expected_body_start
    assert unknown.body_start == expected_body_start
    assert occurrence_body(text, unknown) == text[unknown.body_start : unknown.body_end]

    # Parser objects and the restricted offset ledger must not retain raw surfaces.
    serialized = repr(asdict(result))
    assert raw_heading not in serialized
    assert raw_body not in serialized


def test_unknown_standalone_heading_starts_ordered_other_occurrence() -> None:
    text = "Novel Clinical Block:\nSynthetic evidence.\n"

    result = parse_sections_v2(text, token_count=whitespace_token_count)

    assert result.heading_candidate_count == 1
    assert result.recognized_heading_count == 0
    assert result.unrecognized_heading_count == 1
    assert len(result.model_sections) == 1
    occurrence = result.model_sections[0]
    assert occurrence.canonical == "OTHER"
    assert occurrence.heading_kind == "unrecognized_heading"
    assert occurrence.source_order == occurrence.occurrence_index
    assert "Novel Clinical Block" not in occurrence_body(text, occurrence)


def test_inline_colon_text_is_not_promoted_to_heading() -> None:
    text = "Status: stable and ready for discharge.\nNo standalone heading follows."

    result = parse_sections_v2(text, token_count=whitespace_token_count)

    assert result.heading_candidate_count == 0
    assert len(result.occurrences) == 1
    assert result.occurrences[0].canonical == "OTHER"
    assert occurrence_body(text, result.occurrences[0]) == text


def test_numbered_heading_normalizes_to_existing_canonical_map() -> None:
    text = "1. Discharge Dx:\nSynthetic diagnosis.\n"

    result = parse_sections_v2(text, token_count=whitespace_token_count)

    occurrence = heading_occurrences(result)[0]
    assert occurrence.heading_kind == "recognized_heading"
    assert occurrence.canonical == "DISCHARGE_DIAGNOSIS"


def test_preamble_and_inter_section_gap_are_distinct_ordered_other_occurrences() -> None:
    text = (
        "Synthetic preamble.\n"
        "Chief Complaint:\n"
        "Dyspnea."
        "\n\n"
        "Discharge Medications:\n"
        "Synthetic medication.\n"
    )

    result = parse_sections_v2(text, token_count=whitespace_token_count)
    preamble_end = text.index("Chief Complaint:")
    gap_start = text.index("Dyspnea.") + len("Dyspnea.")
    gap_end = text.index("Discharge Medications:")

    preamble = [
        occurrence
        for occurrence in result.occurrences
        if occurrence.heading_kind == "preamble_or_gap"
        and occurrence.body_start == 0
        and occurrence.body_end == preamble_end
    ]
    gaps = [
        occurrence
        for occurrence in result.occurrences
        if occurrence.heading_kind == "preamble_or_gap"
        and occurrence.body_start == gap_start
        and occurrence.body_end == gap_end
    ]
    assert len(preamble) == 1
    assert preamble[0].canonical == "OTHER"
    assert len(gaps) == 1
    assert gaps[0].canonical == "OTHER"
    assert gaps[0].is_model_slot is False

    assert [item.source_order for item in result.occurrences] == list(
        range(len(result.occurrences))
    )
    assert result.occurrences.index(preamble[0]) < result.occurrences.index(gaps[0])
    assert_dense_ledger(text, result)


def test_empty_heading_bodies_remain_in_ledger_but_create_no_model_slots() -> None:
    text = "Chief Complaint:\r\n\r\nDischarge Medications:\r\n"

    result = parse_sections_v2(text, token_count=whitespace_token_count)
    headings = heading_occurrences(result)

    assert len(headings) == 2
    assert all(occurrence.is_model_slot is False for occurrence in headings)
    assert result.empty_body_occurrence_count == 2
    assert result.model_sections == ()
    assert result.few_sections is True
    assert result.fallback_required is True
    assert_dense_ledger(text, result)


def test_repeated_canonical_occurrences_remain_unmerged_in_source_order() -> None:
    text = (
        "Past Medical History:\n"
        "First synthetic history.\n"
        "Brief Hospital Course:\n"
        "Synthetic course.\n"
        "Past Medical History:\n"
        "Second synthetic history.\n"
    )

    result = parse_sections_v2(text, token_count=whitespace_token_count)
    headings = heading_occurrences(result)

    assert [item.canonical for item in headings] == [
        "PAST_MEDICAL_HISTORY",
        "BRIEF_HOSPITAL_COURSE",
        "PAST_MEDICAL_HISTORY",
    ]
    repeated = [
        item for item in headings if item.canonical == "PAST_MEDICAL_HISTORY"
    ]
    assert len(repeated) == 2
    assert all(item.canonical_occurrence_count == 2 for item in repeated)
    assert all(item.repeated_section for item in repeated)
    assert repeated[0].body_start < repeated[1].body_start
    assert all("<SEC_REPEAT>" not in occurrence_body(text, item) for item in headings)


@pytest.mark.parametrize(
    ("long_length", "expected_too_long", "expected_fallback"),
    [
        (511, False, False),
        (512, True, True),
    ],
)
def test_fallback_uses_strict_511_content_token_boundary(
    long_length: int,
    expected_too_long: bool,
    expected_fallback: bool,
) -> None:
    long_body = " ".join(["token"] * long_length)
    text = (
        f"Chief Complaint:\n{long_body}\n"
        "Discharge Medications:\nshort\n"
    )

    result = parse_sections_v2(
        text,
        token_count=whitespace_token_count,
        config=ParserV2Config(max_content_tokens=511),
    )

    assert len(result.model_sections) == 2
    assert result.max_section_tokens == long_length
    assert result.few_sections is False
    assert result.too_long is expected_too_long
    assert result.fallback_required is expected_fallback


def test_repeated_occurrence_lengths_are_not_merged_for_fallback() -> None:
    body = " ".join(["token"] * 300)
    text = (
        f"Past Medical History:\n{body}\n"
        "Brief Hospital Course:\nshort\n"
        f"Past Medical History:\n{body}\n"
    )

    result = parse_sections_v2(text, token_count=whitespace_token_count)

    assert result.max_section_tokens == 300
    assert result.too_long is False
    assert result.fallback_required is False


def test_low_recognized_coverage_is_not_a_fallback_reason() -> None:
    text = (
        "First Unknown Block:\n"
        "Synthetic evidence one.\n"
        "Second Unknown Block:\n"
        "Synthetic evidence two.\n"
    )

    result = parse_sections_v2(text, token_count=whitespace_token_count)

    assert len(result.model_sections) == 2
    assert all(item.canonical == "OTHER" for item in result.model_sections)
    assert result.recognized_canonical_coverage == 0.0
    assert result.few_sections is False
    assert result.too_long is False
    assert result.fallback_required is False


def test_token_counter_receives_exact_untrimmed_source_body() -> None:
    seen: list[str] = []
    text = "Chief Complaint:\n  synthetic body  \n\nDischarge Medications:\nshort"

    def count_exact(body: str) -> int:
        seen.append(body)
        return len(body.split())

    result = parse_sections_v2(text, token_count=count_exact)
    headings = heading_occurrences(result)

    expected = [
        text[item.body_start : item.body_end]
        for item in headings
        if item.content_end > item.content_start
    ]
    assert seen[-len(expected) :] == expected


def test_non_string_and_negative_token_counts_fail_closed_without_raw_text() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        parse_sections_v2(None, token_count=whitespace_token_count)  # type: ignore[arg-type]

    raw = "Chief Complaint:\nPRIVATE_SYNTHETIC_VALUE"
    with pytest.raises(ValueError, match="negative") as exc_info:
        parse_sections_v2(raw, token_count=lambda _body: -1)
    assert "PRIVATE_SYNTHETIC_VALUE" not in str(exc_info.value)
