from __future__ import annotations

import pandas as pd

from .death_censoring import is_in_hospital_death


def filter_alive_index_admissions(admissions: pd.DataFrame) -> pd.DataFrame:
    """Return admissions not marked as in-hospital deaths."""
    if admissions.empty:
        return admissions.copy()
    mask = admissions.apply(lambda row: not is_in_hospital_death(row), axis=1)
    return admissions.loc[mask].copy()
