from __future__ import annotations

from notesm_v2.data.section_parser import fallback_required_for_model, canonicalize_heading, merge_duplicate_sections_for_model, parse_sections
from notesm_v2.evaluation.section_importance import remove_canonical_section_occurrences


def test_heading_variants_map_to_canonical_sections() -> None:
    assert canonicalize_heading("Discharge Dx") == "DISCHARGE_DIAGNOSIS"
    assert canonicalize_heading("Follow-up Instructions") == "FOLLOW_UP"
    assert canonicalize_heading("Medications on Discharge") == "DISCHARGE_MEDICATIONS"


def test_parser_recognizes_common_sections() -> None:
    text = """Chief Complaint:
Shortness of breath.

Brief Hospital Course:
Treated with diuresis.

Discharge Medications:
Medication list.
"""
    result = parse_sections(text)
    names = [section.canonical for section in result.sections]
    assert "CHIEF_COMPLAINT" in names
    assert "BRIEF_HOSPITAL_COURSE" in names
    assert "DISCHARGE_MEDICATIONS" in names
    assert result.fallback_required is False


def test_parser_fallback_when_too_few_sections() -> None:
    result = parse_sections("No recognizable headings here.")
    assert result.fallback_required is True
    assert result.sections[0].canonical == "OTHER"


def test_merge_duplicate_sections_for_model_collapses_repeated_canonical_sections() -> None:
    text = """Past Medical History:
Diabetes.

Brief Hospital Course:
Improved.

Past Medical History:
Hypertension.
"""
    result = parse_sections(text)
    merged = merge_duplicate_sections_for_model(result.sections)
    pmh_sections = [section for section in merged if section.canonical == "PAST_MEDICAL_HISTORY"]
    assert len(pmh_sections) == 1
    assert "<SEC_REPEAT>" in pmh_sections[0].text
    assert "Diabetes." in pmh_sections[0].text
    assert "Hypertension." in pmh_sections[0].text


def test_remove_one_section_ablation_removes_all_repeated_occurrences() -> None:
    text = """Past Medical History:
Diabetes.

Brief Hospital Course:
Improved.

Past Medical History:
Hypertension.
"""
    result = parse_sections(text)
    ablated = remove_canonical_section_occurrences(result.sections, "PAST_MEDICAL_HISTORY")
    assert [section.canonical for section in ablated] == ["BRIEF_HOSPITAL_COURSE"]


def test_fallback_length_check_uses_merged_section_length() -> None:
    text = """Past Medical History:
one two three

Brief Hospital Course:
short

Past Medical History:
four five six
"""
    result = parse_sections(text)
    assert fallback_required_for_model(result, coverage_threshold=0.0, section_max_tokens=6)
    assert not fallback_required_for_model(result, coverage_threshold=0.0, section_max_tokens=7)
