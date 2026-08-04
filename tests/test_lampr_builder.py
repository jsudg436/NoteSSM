from __future__ import annotations

import pandas as pd

from notesm_v2.data.lampr_builder import build_lampr


def base_config(train_frac: float = 0.67, val_frac: float = 0.0) -> dict:
    return {
        "cohort": {"horizon_days": 30, "adult_only_if_age_available": True, "require_tail_followup": False},
        "planned_admissions": {"types": ["ELECTIVE"], "strategy": "continue_search"},
        "episode_merge": {"gap_hours": 24},
        "note_timing": {"max_note_delay_hours": 168},
        "splits": {"seed": 1, "train_frac": train_frac, "val_frac": val_frac},
        "dapt": {"max_length": 32},
    }


def sectioned_text(marker: str) -> str:
    return f"Chief Complaint:\n{marker}\n\nBrief Hospital Course:\nRecovered.\n\nDischarge Instructions:\nFollow up."


def repeated_pmh_text() -> str:
    return """Past Medical History:
Diabetes mellitus with chronic hypertension and prior outpatient follow up documented in the history.

Brief Hospital Course:
Improved with observation and routine discharge planning after an uncomplicated short stay.

Past Medical History:
Hypertension and chronic kidney disease listed again in a repeated history section.
"""


def test_lampr_builder_labels_and_splits_synthetic() -> None:
    admissions = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 10, "admittime": "2020-01-01", "dischtime": "2020-01-05", "admission_type": "URGENT", "hospital_expire_flag": 0},
            {"subject_id": 1, "hadm_id": 11, "admittime": "2020-01-20", "dischtime": "2020-01-22", "admission_type": "URGENT", "hospital_expire_flag": 0},
            {"subject_id": 2, "hadm_id": 20, "admittime": "2020-02-01", "dischtime": "2020-02-05", "admission_type": "URGENT", "hospital_expire_flag": 0},
            {"subject_id": 3, "hadm_id": 30, "admittime": "2020-03-01", "dischtime": "2020-03-05", "admission_type": "URGENT", "hospital_expire_flag": 0},
            {"subject_id": 3, "hadm_id": 31, "admittime": "2020-03-20", "dischtime": "2020-03-22", "admission_type": "URGENT", "hospital_expire_flag": 0},
        ]
    )
    patients = pd.DataFrame(
        [
            {"subject_id": 1, "anchor_age": 70},
            {"subject_id": 2, "anchor_age": 60},
            {"subject_id": 3, "anchor_age": 50, "dod": "2020-03-10"},
        ]
    )
    notes = pd.DataFrame(
        [
            {"note_id": "n10", "subject_id": 1, "hadm_id": 10, "charttime": "2020-01-05", "storetime": "2020-01-06", "text": sectioned_text("positive")},
            {"note_id": "n20", "subject_id": 2, "hadm_id": 20, "charttime": "2020-02-05", "storetime": "2020-02-06", "text": sectioned_text("negative")},
            {"note_id": "n30", "subject_id": 3, "hadm_id": 30, "charttime": "2020-03-05", "storetime": "2020-03-06", "text": sectioned_text("death censored")},
        ]
    )

    result = build_lampr(admissions, patients, notes, base_config())

    assert set(result.split_examples["label"]) == {0, 1}
    assert len(result.split_examples) == 2
    assert int(result.filter_counts.loc[result.filter_counts["step"] == "death_censored", "count"].iloc[0]) == 1
    assert int(result.filter_counts.loc[result.filter_counts["step"] == "final_positive", "count"].iloc[0]) == 1
    assert result.split_examples["subject_id"].nunique() == 2
    assert 3 not in set(result.split_examples["subject_id"])


def test_terminal_end_hadm_note_is_used_and_intermediate_transfer_note_is_excluded() -> None:
    admissions = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 10, "admittime": "2020-01-01", "dischtime": "2020-01-05 08:00", "admission_type": "URGENT", "hospital_expire_flag": 0},
            {"subject_id": 1, "hadm_id": 11, "admittime": "2020-01-05 12:00", "dischtime": "2020-01-08 08:00", "admission_type": "URGENT", "hospital_expire_flag": 0},
        ]
    )
    patients = pd.DataFrame([{"subject_id": 1, "anchor_age": 70}])
    notes = pd.DataFrame(
        [
            {"note_id": "intermediate", "subject_id": 1, "hadm_id": 10, "charttime": "2020-01-05", "storetime": "2020-01-05 09:00", "text": sectioned_text("INTERMEDIATE_RAW_MARKER")},
            {"note_id": "terminal", "subject_id": 1, "hadm_id": 11, "charttime": "2020-01-08", "storetime": "2020-01-08 09:00", "text": sectioned_text("TERMINAL_RAW_MARKER")},
        ]
    )

    result = build_lampr(admissions, patients, notes, base_config(train_frac=1.0))

    assert len(result.split_examples) == 1
    row = result.split_examples.iloc[0]
    assert row["hadm_id"] == 11
    assert row["end_hadm_id"] == 11
    assert "TERMINAL_RAW_MARKER" in row["text"]
    assert "INTERMEDIATE_RAW_MARKER" not in result.benchmark_manifest_no_text.to_csv(index=False)


def test_multiple_terminal_discharge_notes_choose_earliest_qualifying_availability_time() -> None:
    admissions = pd.DataFrame(
        [{"subject_id": 1, "hadm_id": 10, "admittime": "2020-01-01", "dischtime": "2020-01-05", "admission_type": "URGENT", "hospital_expire_flag": 0}]
    )
    patients = pd.DataFrame([{"subject_id": 1, "anchor_age": 70}])
    notes = pd.DataFrame(
        [
            {"note_id": "late", "subject_id": 1, "hadm_id": 10, "charttime": "2020-01-05", "storetime": "2020-01-20", "text": sectioned_text("late")},
            {"note_id": "second", "subject_id": 1, "hadm_id": 10, "charttime": "2020-01-05", "storetime": "2020-01-07", "text": sectioned_text("second")},
            {"note_id": "first", "subject_id": 1, "hadm_id": 10, "charttime": "2020-01-05", "storetime": "2020-01-06", "text": sectioned_text("first")},
        ]
    )

    result = build_lampr(admissions, patients, notes, base_config(train_frac=1.0))

    assert len(result.split_examples) == 1
    assert result.split_examples.iloc[0]["note_id_hash"] == result.split_examples["note_id_hash"].iloc[0]
    assert "first" in result.split_examples.iloc[0]["text"]
    assert int(result.filter_counts.loc[result.filter_counts["step"] == "duplicate_terminal_notes_removed", "count"].iloc[0]) == 1
    assert int(result.note_time_filter_counts.loc[result.note_time_filter_counts["metric"] == "terminal_late_note", "count"].iloc[0]) == 1


def test_availability_time_drives_event_before_note_filter() -> None:
    admissions = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 10, "admittime": "2020-01-01", "dischtime": "2020-01-05", "admission_type": "URGENT", "hospital_expire_flag": 0},
            {"subject_id": 1, "hadm_id": 11, "admittime": "2020-01-08", "dischtime": "2020-01-12", "admission_type": "URGENT", "hospital_expire_flag": 0},
        ]
    )
    patients = pd.DataFrame([{"subject_id": 1, "anchor_age": 70}])
    notes = pd.DataFrame(
        [{"note_id": "n10", "subject_id": 1, "hadm_id": 10, "charttime": "2020-01-10", "storetime": "2020-01-10", "text": sectioned_text("availability")}]
    )

    result = build_lampr(admissions, patients, notes, {**base_config(train_frac=1.0), "episode_merge": {"gap_hours": 0}})

    assert result.split_examples.empty
    assert int(result.note_time_filter_counts.loc[result.note_time_filter_counts["metric"] == "timing_exclude_event_before_availability", "count"].iloc[0]) == 1


def test_section_dapt_and_no_text_artifacts_are_built_without_raw_text_reports() -> None:
    admissions = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 10, "admittime": "2020-01-01", "dischtime": "2020-01-05", "admission_type": "URGENT", "hospital_expire_flag": 0},
            {"subject_id": 2, "hadm_id": 20, "admittime": "2020-02-01", "dischtime": "2020-02-05", "admission_type": "URGENT", "hospital_expire_flag": 0},
        ]
    )
    patients = pd.DataFrame([{"subject_id": 1, "anchor_age": 70}, {"subject_id": 2, "anchor_age": 60}])
    notes = pd.DataFrame(
        [
            {"note_id": "n10", "subject_id": 1, "hadm_id": 10, "charttime": "2020-01-05", "storetime": "2020-01-06", "text": sectioned_text("RAW_SECRET_ONE")},
            {"note_id": "n20", "subject_id": 2, "hadm_id": 20, "charttime": "2020-02-05", "storetime": "2020-02-06", "text": sectioned_text("RAW_SECRET_TWO")},
        ]
    )

    result = build_lampr(admissions, patients, notes, base_config(train_frac=0.5))

    assert "num_sections" in result.split_examples.columns
    assert not result.section_parser_stats.empty
    assert "text" not in result.benchmark_manifest_no_text.columns
    assert "subject_id" not in result.benchmark_manifest_no_text.columns
    assert "sequence_text" not in result.dapt_train_manifest.columns
    train_hashes = set(result.benchmark_manifest_no_text.loc[result.benchmark_manifest_no_text["split"] == "train", "subject_id_hash"])
    assert set(result.dapt_train_manifest["subject_id_hash"]).issubset(train_hashes)
    report_payload = "\n".join(
        [
            result.benchmark_manifest_no_text.to_csv(index=False),
            result.section_parser_stats.to_csv(index=False),
            result.note_time_filter_counts.to_csv(index=False),
            result.death_censoring_counts.to_csv(index=False),
            result.planned_admission_counts.to_csv(index=False),
        ]
    )
    assert "RAW_SECRET_ONE" not in report_payload
    assert "RAW_SECRET_TWO" not in report_payload


def test_section_stats_distinguish_occurrences_note_presence_and_repeats() -> None:
    admissions = pd.DataFrame(
        [{"subject_id": 1, "hadm_id": 10, "admittime": "2020-01-01", "dischtime": "2020-01-05", "admission_type": "URGENT", "hospital_expire_flag": 0}]
    )
    patients = pd.DataFrame([{"subject_id": 1, "anchor_age": 70}])
    notes = pd.DataFrame(
        [{"note_id": "n10", "subject_id": 1, "hadm_id": 10, "charttime": "2020-01-05", "storetime": "2020-01-06", "text": repeated_pmh_text()}]
    )

    result = build_lampr(admissions, patients, notes, base_config(train_frac=1.0))
    stats = dict(zip(result.section_parser_stats["metric"], result.section_parser_stats["value"]))

    assert stats["section_occurrence_count.PAST_MEDICAL_HISTORY"] == 2
    assert stats["section_note_count.PAST_MEDICAL_HISTORY"] == 1
    assert stats["section_note_count_parsed.PAST_MEDICAL_HISTORY"] == 1
    assert stats["section_note_prevalence_parsed.PAST_MEDICAL_HISTORY"] == 1
    assert stats["section_repeat_note_count.PAST_MEDICAL_HISTORY"] == 1
    assert stats["section_repeat_rate.PAST_MEDICAL_HISTORY"] == 1
    assert stats["notes_with_repeated_sections"] == 1
    assert "section_frequency.PAST_MEDICAL_HISTORY" not in stats
    row = result.split_examples.iloc[0]
    assert row["num_section_occurrences"] == 3
    assert row["num_unique_sections"] == 2
    assert row["has_repeated_sections"]
    assert row["repeated_section_names"] == "PAST_MEDICAL_HISTORY"
    assert row["max_occurrences_for_any_section"] == 2
