from __future__ import annotations

import pandas as pd

from notesm_v2.data.labels import find_first_unplanned_readmission


def test_planned_readmission_is_skipped_by_default() -> None:
    future = pd.DataFrame(
        [
            {"admittime": "2020-01-05", "admission_type": "ELECTIVE"},
            {"admittime": "2020-01-08", "admission_type": "URGENT"},
        ]
    )
    readmit, reason = find_first_unplanned_readmission("2020-01-01", future, ["ELECTIVE"])
    assert str(readmit) == "2020-01-08 00:00:00"
    assert reason == "unplanned"


def test_planned_first_can_exclude_when_configured() -> None:
    future = pd.DataFrame([{"admittime": "2020-01-05", "admission_type": "ELECTIVE"}])
    readmit, reason = find_first_unplanned_readmission(
        "2020-01-01",
        future,
        ["ELECTIVE"],
        planned_strategy="exclude",
    )
    assert readmit is None
    assert reason == "exclude_planned_first"
