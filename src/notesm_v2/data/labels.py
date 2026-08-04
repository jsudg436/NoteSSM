from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .planned_admission import is_planned_admission


def find_first_unplanned_readmission(
    index_dischtime: object,
    future_admissions: pd.DataFrame,
    planned_types: Iterable[str],
    horizon_days: int = 30,
    planned_strategy: str = "continue_search",
) -> tuple[pd.Timestamp | None, str]:
    """Find the first eligible unplanned readmission after index discharge.

    planned_strategy:
      - continue_search: skip planned admissions and keep looking within horizon.
      - exclude: report excluded if a planned admission occurs before an unplanned one.
    """
    if planned_strategy not in {"continue_search", "exclude"}:
        raise ValueError("planned_strategy must be 'continue_search' or 'exclude'")

    discharge_ts = pd.Timestamp(index_dischtime)
    horizon_end = discharge_ts + pd.Timedelta(days=horizon_days)
    if future_admissions.empty:
        return None, "none"

    df = future_admissions.copy()
    df["admittime"] = pd.to_datetime(df["admittime"])
    df = df[(df["admittime"] > discharge_ts) & (df["admittime"] <= horizon_end)]
    df = df.sort_values("admittime")
    for row in df.to_dict("records"):
        if is_planned_admission(row.get("admission_type"), planned_types):
            if planned_strategy == "exclude":
                return None, "exclude_planned_first"
            continue
        return pd.Timestamp(row["admittime"]), "unplanned"
    return None, "none_after_planned" if not df.empty else "none"
