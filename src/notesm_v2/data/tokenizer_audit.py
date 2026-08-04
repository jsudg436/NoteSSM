from __future__ import annotations

from collections.abc import Callable


CLINICAL_TERMS = [
    "myocardial infarction",
    "congestive heart failure",
    "hypoxemic respiratory failure",
    "chronic obstructive pulmonary disease",
    "acute kidney injury",
    "creatinine",
    "vancomycin",
    "metoprolol",
    "norepinephrine",
    "anticoagulation",
    "CHF",
    "COPD",
    "AKI",
    "CABG",
    "NSTEMI",
    "STEMI",
]


def term_fragmentation(tokenize: Callable[[str], list[str]], terms: list[str] | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for term in terms or CLINICAL_TERMS:
        tokens = tokenize(term)
        rows.append({"term": term, "num_tokens": len(tokens), "tokens": tokens})
    return rows
