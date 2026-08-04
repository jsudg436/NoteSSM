from __future__ import annotations

import pandas as pd


def derive_note_time(notes: pd.DataFrame) -> pd.Series:
    """Return storetime when available, otherwise charttime."""
    if "storetime" not in notes.columns or "charttime" not in notes.columns:
        raise ValueError("notes must contain storetime and charttime columns")
    store = pd.to_datetime(notes["storetime"])
    chart = pd.to_datetime(notes["charttime"])
    return store.fillna(chart)


def classify_note_timing(
    dischtime: object,
    note_time: object,
    event_time: object | None,
    horizon_days: int = 30,
    max_note_delay_hours: int = 168,
) -> str:
    """Classify benchmark timing eligibility for a discharge summary."""
    discharge_ts = pd.Timestamp(dischtime)
    note_ts = pd.Timestamp(note_time) if pd.notna(note_time) else pd.NaT
    if pd.isna(note_ts):
        return "exclude_missing_note_time"

    horizon_end = discharge_ts + pd.Timedelta(days=horizon_days)
    late_cutoff = discharge_ts + pd.Timedelta(hours=max_note_delay_hours)
    if note_ts > late_cutoff:
        return "exclude_late_note"
    if note_ts > horizon_end:
        return "exclude_note_after_horizon"

    if event_time is not None and pd.notna(event_time):
        event_ts = pd.Timestamp(event_time)
        if event_ts <= note_ts:
            return "exclude_event_before_note"
        if event_ts <= horizon_end:
            return "positive"

    return "negative"
