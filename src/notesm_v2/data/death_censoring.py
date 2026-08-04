from __future__ import annotations

import pandas as pd


def is_in_hospital_death(row: pd.Series | dict[str, object]) -> bool:
    """Detect in-hospital death using available MIMIC-style signals."""
    if int(row.get("hospital_expire_flag") or 0) == 1:
        return True
    admittime = row.get("admittime")
    dischtime = row.get("dischtime")
    deathtime = row.get("deathtime")
    if pd.notna(admittime) and pd.notna(dischtime) and pd.notna(deathtime):
        death_ts = pd.Timestamp(deathtime)
        if pd.Timestamp(admittime) <= death_ts <= pd.Timestamp(dischtime):
            return True
    discharge_location = str(row.get("discharge_location") or "").upper()
    death_terms = ("DEAD", "DIED", "EXPIRED")
    return any(term in discharge_location for term in death_terms)


def death_censor_status(
    dischtime: object,
    death_time: object | None,
    readmission_time: object | None,
    horizon_days: int = 30,
) -> str:
    """Return censored when death precedes readmission within the horizon."""
    if death_time is None or pd.isna(death_time):
        return "not_censored"

    discharge_ts = pd.Timestamp(dischtime)
    death_ts = pd.Timestamp(death_time)
    if not (discharge_ts < death_ts <= discharge_ts + pd.Timedelta(days=horizon_days)):
        return "not_censored"

    if readmission_time is not None and pd.notna(readmission_time):
        readmit_ts = pd.Timestamp(readmission_time)
        if readmit_ts < death_ts:
            return "readmission_before_death"

    return "censored_death_before_readmission"
