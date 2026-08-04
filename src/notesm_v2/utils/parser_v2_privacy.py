from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCIENTIFIC_FLOAT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+"
    r"(?![A-Za-z0-9_.])"
)
HASH16_PATTERN = re.compile(r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{16}(?![A-Fa-f0-9])")
HASH64_PATTERN = re.compile(r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{64}(?![A-Fa-f0-9])")
RAW_ASSIGNMENT_PATTERN = re.compile(
    r"""(?ix)
    \b(?:raw_?(?:text|heading)|note_text|merged_section_text|chunk_text)
    \s*[:=]\s*['"]
    """
)
RESTRICTED_PATH_PATTERN = re.compile(
    r"""(?ix)
    (?:^|[/\\])workspace(?:[/\\]) |
    benchmark_examples_(?:train|validation|val|test)\.parquet |
    (?:^|[/\\])(?:checkpoints?|embedding_cache|restricted_ledgers?)(?:[/\\])
    """
)

FORBIDDEN_EXACT_COLUMNS = {
    "example_id",
    "patient_id",
    "patient_cluster_id",
    "patient_cluster_key",
    "subject_id",
    "hadm_id",
    "stay_id",
    "text",
    "raw_text",
    "raw_note",
    "raw_heading",
    "heading",
    "merged_section_text",
    "chunk_text",
    "original_content_ids",
    "retained_content_ids",
    "content_token_ids",
    "input_ids",
    "token_ids",
    "source_offset_starts",
    "source_offset_ends",
    "source_offset_zero_width_mask",
    "retained_source_offset_starts",
    "retained_source_offset_ends",
    "retained_source_offset_zero_width_mask",
    "retained_mask",
    "attention_mask",
    "embedding",
    "embeddings",
    "label",
    "labels",
    "prediction",
    "predictions",
    "probability",
    "probabilities",
}
RESTRICTED_FILENAME_TERMS = {
    "restricted",
    "token_ledger",
    "span_ledger",
    "occurrence_ledger",
    "note_index",
    "fallback_chunk_token_cache",
    "embedding",
    "prediction",
}
ALLOWED_SHA_KEY_TERMS = ("sha256", "sha_256", "digest", "manifest_hash")


@dataclass(frozen=True, slots=True)
class PrivacyValidationResult:
    file: str
    passed: bool
    finding_count: int
    findings: tuple[str, ...]


def _standalone_16hex_values(text: str) -> list[str]:
    scientific_spans = [match.span() for match in SCIENTIFIC_FLOAT_PATTERN.finditer(text)]
    return [
        match.group(0)
        for match in HASH16_PATTERN.finditer(text)
        if not any(
            match.start() < scientific_end and scientific_start < match.end()
            for scientific_start, scientific_end in scientific_spans
        )
    ]


def _is_allowed_sha_key(key: str) -> bool:
    normalized = key.lower()
    return any(term in normalized for term in ALLOWED_SHA_KEY_TERMS)


def _json_hash_findings(
    value: Any,
    *,
    key: str = "",
    allowed_hash_context: bool = False,
) -> list[str]:
    findings: list[str] = []
    child_hash_context = allowed_hash_context or _is_allowed_sha_key(key)
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            findings.extend(
                _json_hash_findings(
                    child_value,
                    key=str(child_key),
                    allowed_hash_context=child_hash_context,
                )
            )
    elif isinstance(value, list):
        for child in value:
            findings.extend(
                _json_hash_findings(
                    child,
                    key=key,
                    allowed_hash_context=child_hash_context,
                )
            )
    elif isinstance(value, str):
        if (
            HASH64_PATTERN.fullmatch(value)
            and not child_hash_context
            and not _is_allowed_sha_key(key)
        ):
            findings.append("64-hex value outside an allowlisted SHA/digest field")
    return findings


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return [str(value) for value in next(reader)]
        except StopIteration:
            return []


def _csv_hash_findings(path: Path) -> list[str]:
    findings: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                if value is None:
                    continue
                for match in HASH64_PATTERN.finditer(str(value)):
                    if not _is_allowed_sha_key(str(key)):
                        findings.append(
                            "64-hex value outside an allowlisted SHA/digest column"
                        )
                        return findings
    return findings


def _markdown_hash_findings(text: str) -> list[str]:
    findings: list[str] = []
    previous_normalized = ""
    for line in text.splitlines():
        normalized = line.lower()
        if not HASH64_PATTERN.search(line):
            if normalized.strip():
                previous_normalized = normalized
            continue
        hash_label_context = f"{previous_normalized} {normalized}"
        if not any(
            term in hash_label_context
            for term in ("sha256", "sha-256", "digest", "manifest")
        ):
            findings.append("64-hex value outside a labeled SHA/digest line")
        previous_normalized = normalized
    return findings


def validate_public_artifact(path: str | Path) -> PrivacyValidationResult:
    """Validate a raw-text-free artifact before it can leave trusted storage."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError("privacy-validation target is missing")
    findings: list[str] = []
    normalized_name = artifact.name.lower()
    if any(term in normalized_name for term in RESTRICTED_FILENAME_TERMS):
        findings.append("restricted artifact filename")

    text = artifact.read_text(encoding="utf-8")
    if RAW_ASSIGNMENT_PATTERN.search(text):
        findings.append("raw text or heading assignment")
    if RESTRICTED_PATH_PATTERN.search(text):
        findings.append("restricted source path")
    if _standalone_16hex_values(text):
        findings.append("standalone identifier-like 16-hex value")

    suffix = artifact.suffix.lower()
    if suffix == ".csv":
        columns = [column.strip().lower() for column in _csv_header(artifact)]
        forbidden = sorted(set(columns) & FORBIDDEN_EXACT_COLUMNS)
        if forbidden:
            findings.append("restricted columns: " + ",".join(forbidden))
        findings.extend(_csv_hash_findings(artifact))
    elif suffix == ".json" or suffix == ".complete":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            findings.append("malformed JSON artifact")
        else:
            findings.extend(_json_hash_findings(payload))
    elif "sha256sums" in normalized_name or suffix == ".sha256":
        # Explicit hash manifests and protocol sidecars are allowlisted.
        pass
    else:
        findings.extend(_markdown_hash_findings(text))

    unique_findings = tuple(sorted(set(findings)))
    return PrivacyValidationResult(
        file=artifact.name,
        passed=not unique_findings,
        finding_count=len(unique_findings),
        findings=unique_findings,
    )


def require_public_artifacts_safe(paths: list[str | Path]) -> tuple[PrivacyValidationResult, ...]:
    results = tuple(validate_public_artifact(path) for path in paths)
    if not results or not all(result.passed for result in results):
        raise ValueError("parser-v2 public artifact privacy validation failed")
    return results
