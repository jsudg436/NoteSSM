from __future__ import annotations

from collections.abc import Iterable


def normalize_admission_type(value: object) -> str:
    """Normalize an admission_type value for configurable matching."""
    if value is None:
        return ""
    return " ".join(str(value).strip().upper().split())


def is_planned_admission(admission_type: object, planned_types: Iterable[str]) -> bool:
    planned = {normalize_admission_type(item) for item in planned_types}
    return normalize_admission_type(admission_type) in planned
