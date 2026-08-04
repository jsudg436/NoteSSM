from __future__ import annotations

import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from collections.abc import Callable


CANONICAL_SECTIONS = [
    "CHIEF_COMPLAINT",
    "HISTORY_OF_PRESENT_ILLNESS",
    "PAST_MEDICAL_HISTORY",
    "PHYSICAL_EXAM",
    "PERTINENT_RESULTS",
    "BRIEF_HOSPITAL_COURSE",
    "HOSPITAL_COURSE_BY_PROBLEM",
    "DISCHARGE_DIAGNOSIS",
    "DISCHARGE_MEDICATIONS",
    "DISCHARGE_DISPOSITION",
    "DISCHARGE_CONDITION",
    "DISCHARGE_INSTRUCTIONS",
    "FOLLOW_UP",
    "OTHER",
]

HEADING_MAP = {
    "CHIEF COMPLAINT": "CHIEF_COMPLAINT",
    "HISTORY OF PRESENT ILLNESS": "HISTORY_OF_PRESENT_ILLNESS",
    "HPI": "HISTORY_OF_PRESENT_ILLNESS",
    "PAST MEDICAL HISTORY": "PAST_MEDICAL_HISTORY",
    "PMH": "PAST_MEDICAL_HISTORY",
    "PHYSICAL EXAM": "PHYSICAL_EXAM",
    "PERTINENT RESULTS": "PERTINENT_RESULTS",
    "LABS": "PERTINENT_RESULTS",
    "BRIEF HOSPITAL COURSE": "BRIEF_HOSPITAL_COURSE",
    "HOSPITAL COURSE": "BRIEF_HOSPITAL_COURSE",
    "HOSPITAL COURSE BY PROBLEM": "HOSPITAL_COURSE_BY_PROBLEM",
    "DISCHARGE DIAGNOSIS": "DISCHARGE_DIAGNOSIS",
    "DISCHARGE DIAGNOSES": "DISCHARGE_DIAGNOSIS",
    "DISCHARGE DX": "DISCHARGE_DIAGNOSIS",
    "DISCHARGE MEDICATIONS": "DISCHARGE_MEDICATIONS",
    "MEDICATIONS ON DISCHARGE": "DISCHARGE_MEDICATIONS",
    "DISCHARGE DISPOSITION": "DISCHARGE_DISPOSITION",
    "DISPOSITION": "DISCHARGE_DISPOSITION",
    "DISCHARGE CONDITION": "DISCHARGE_CONDITION",
    "CONDITION AT DISCHARGE": "DISCHARGE_CONDITION",
    "DISCHARGE INSTRUCTIONS": "DISCHARGE_INSTRUCTIONS",
    "FOLLOWUP INSTRUCTIONS": "FOLLOW_UP",
    "FOLLOW-UP": "FOLLOW_UP",
    "FOLLOW UP": "FOLLOW_UP",
    "FOLLOW-UP INSTRUCTIONS": "FOLLOW_UP",
    "APPOINTMENTS": "FOLLOW_UP",
}

HEADING_RE = re.compile(r"(?m)^(?P<header>[A-Za-z][A-Za-z0-9 /()\-]{1,80})\s*:\s*$")


@dataclass(frozen=True)
class ParsedSection:
    canonical: str
    heading: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class SectionParseResult:
    sections: list[ParsedSection]
    fallback_required: bool
    coverage: float


def normalize_heading(heading: str) -> str:
    return " ".join(heading.upper().replace("_", " ").split())


def canonicalize_heading(heading: str) -> str | None:
    return HEADING_MAP.get(normalize_heading(heading))


def parse_sections(text: str, min_sections: int = 2, coverage_threshold: float = 0.50) -> SectionParseResult:
    """Parse MIMIC-style discharge summary sections without logging note text."""
    if not text:
        return SectionParseResult([], True, 0.0)

    matches = [m for m in HEADING_RE.finditer(text) if canonicalize_heading(m.group("header"))]
    sections: list[ParsedSection] = []
    for index, match in enumerate(matches):
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_heading = match.group("header")
        canonical = canonicalize_heading(raw_heading) or "OTHER"
        content = text[content_start:content_end].strip()
        if content:
            sections.append(
                ParsedSection(
                    canonical=canonical,
                    heading=raw_heading.strip(),
                    text=content,
                    start=content_start,
                    end=content_end,
                )
            )

    covered_chars = sum(section.end - section.start for section in sections)
    coverage = covered_chars / max(len(text.strip()), 1)
    fallback = len(sections) < min_sections or coverage < coverage_threshold
    if not sections and text.strip():
        sections = [ParsedSection("OTHER", "OTHER", text.strip(), 0, len(text))]
    return SectionParseResult(sections, fallback, coverage)


def section_occurrence_counts(sections: list[ParsedSection]) -> Counter[str]:
    return Counter(section.canonical for section in sections)


def repeated_section_names(sections: list[ParsedSection]) -> list[str]:
    counts = section_occurrence_counts(sections)
    return sorted([canonical for canonical, count in counts.items() if count > 1])


def merge_duplicate_sections_for_model(
    sections: list[ParsedSection],
    repeat_separator: str = "<SEC_REPEAT>",
) -> list[ParsedSection]:
    """Collapse repeated canonical sections for section-aware pooling.

    Parser output remains occurrence-level. This helper is the model-facing
    view: each canonical section appears at most once, with repeated content
    concatenated in original order.
    """
    grouped: OrderedDict[str, list[ParsedSection]] = OrderedDict()
    for section in sections:
        grouped.setdefault(section.canonical, []).append(section)

    merged: list[ParsedSection] = []
    for canonical, group in grouped.items():
        text = f"\n{repeat_separator}\n".join(section.text for section in group)
        heading = group[0].heading
        start = min(section.start for section in group)
        end = max(section.end for section in group)
        merged.append(ParsedSection(canonical=canonical, heading=heading, text=text, start=start, end=end))
    return merged


def default_token_count(text: str) -> int:
    return len(text.split())


def fallback_required_for_model(
    parse_result: SectionParseResult,
    min_sections: int = 2,
    coverage_threshold: float = 0.50,
    section_max_tokens: int = 2048,
    token_count: Callable[[str], int] = default_token_count,
) -> bool:
    """Return whether NoteSSM-v2 should use chunk-to-note fallback.

    The section count and length checks use merged canonical sections, not
    repeated occurrence-level sections.
    """
    merged_sections = merge_duplicate_sections_for_model(parse_result.sections)
    if len(merged_sections) < min_sections:
        return True
    if parse_result.coverage < coverage_threshold:
        return True
    max_merged_length = max((token_count(section.text) for section in merged_sections), default=0)
    return max_merged_length > section_max_tokens
