from __future__ import annotations

import pandas as pd

from notesm_v2.data.episode_merge import merge_continuous_care_episodes


def test_next_admission_within_24_hours_is_merged() -> None:
    admissions = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 10, "admittime": "2020-01-01", "dischtime": "2020-01-05 08:00"},
            {"subject_id": 1, "hadm_id": 11, "admittime": "2020-01-05 20:00", "dischtime": "2020-01-07"},
        ]
    )
    episodes, counts = merge_continuous_care_episodes(admissions, gap_hours=24)
    assert len(episodes) == 1
    assert episodes.iloc[0]["hadm_ids"] == (10, 11)
    assert episodes.iloc[0]["start_hadm_id"] == 10
    assert episodes.iloc[0]["end_hadm_id"] == 11
    assert counts["merged_admissions"] == 1


def test_next_admission_after_gap_is_separate_episode() -> None:
    admissions = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 10, "admittime": "2020-01-01", "dischtime": "2020-01-05 08:00"},
            {"subject_id": 1, "hadm_id": 11, "admittime": "2020-01-06 09:00", "dischtime": "2020-01-07"},
        ]
    )
    episodes, _ = merge_continuous_care_episodes(admissions, gap_hours=24)
    assert len(episodes) == 2
