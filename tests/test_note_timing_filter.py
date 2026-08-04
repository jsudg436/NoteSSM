from __future__ import annotations

import pandas as pd

from notesm_v2.data.note_timing import classify_note_timing, derive_note_time


def test_derive_note_time_prefers_storetime() -> None:
    notes = pd.DataFrame(
        {
            "storetime": ["2020-01-03 08:00:00", None],
            "charttime": ["2020-01-02 08:00:00", "2020-01-04 09:00:00"],
        }
    )
    result = derive_note_time(notes)
    assert str(result.iloc[0]) == "2020-01-03 08:00:00"
    assert str(result.iloc[1]) == "2020-01-04 09:00:00"


def test_exclude_if_readmission_before_note_time() -> None:
    status = classify_note_timing(
        dischtime="2020-01-01 12:00:00",
        note_time="2020-01-04 12:00:00",
        event_time="2020-01-03 12:00:00",
    )
    assert status == "exclude_event_before_note"


def test_positive_if_readmission_after_note_within_horizon() -> None:
    status = classify_note_timing(
        dischtime="2020-01-01 12:00:00",
        note_time="2020-01-02 12:00:00",
        event_time="2020-01-20 12:00:00",
    )
    assert status == "positive"


def test_exclude_if_note_too_late() -> None:
    status = classify_note_timing(
        dischtime="2020-01-01 12:00:00",
        note_time="2020-01-09 12:00:01",
        event_time=None,
        max_note_delay_hours=168,
    )
    assert status == "exclude_late_note"
