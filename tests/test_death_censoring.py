from __future__ import annotations

from notesm_v2.data.death_censoring import death_censor_status, is_in_hospital_death


def test_death_before_readmission_within_horizon_is_censored() -> None:
    status = death_censor_status(
        dischtime="2020-01-01",
        death_time="2020-01-10",
        readmission_time="2020-01-12",
    )
    assert status == "censored_death_before_readmission"


def test_readmission_before_death_is_not_censored_negative() -> None:
    status = death_censor_status(
        dischtime="2020-01-01",
        death_time="2020-01-12",
        readmission_time="2020-01-10",
    )
    assert status == "readmission_before_death"


def test_in_hospital_death_flag() -> None:
    assert is_in_hospital_death({"hospital_expire_flag": 1}) is True


def test_in_hospital_death_by_location() -> None:
    assert is_in_hospital_death({"discharge_location": "DIED"}) is True
