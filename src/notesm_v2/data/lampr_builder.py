from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from notesm_v2.data.dapt_packing import build_patient_sequences, pack_patient_sequences
from notesm_v2.data.death_censoring import death_censor_status, is_in_hospital_death
from notesm_v2.data.episode_merge import merge_continuous_care_episodes
from notesm_v2.data.labels import find_first_unplanned_readmission
from notesm_v2.data.note_timing import derive_note_time
from notesm_v2.data.planned_admission import is_planned_admission
from notesm_v2.data.section_parser import fallback_required_for_model, parse_sections, section_occurrence_counts
from notesm_v2.data.splits import assert_no_patient_overlap, make_patient_splits
from notesm_v2.utils.privacy import stable_hash


REQUIRED_ADMISSION_COLUMNS = {"subject_id", "hadm_id", "admittime", "dischtime"}
REQUIRED_NOTE_COLUMNS = {"subject_id", "hadm_id", "charttime", "storetime", "text"}
REQUIRED_PATIENT_COLUMNS = {"subject_id"}
REPORT_COLUMNS_WITHOUT_TEXT = {
    "text",
    "sequence_text",
    "sections",
    "dod",
    "deathtime",
}


@dataclass(frozen=True)
class LamprBuildResult:
    examples: pd.DataFrame
    split_examples: pd.DataFrame
    filter_counts: pd.DataFrame
    split_counts: pd.DataFrame
    label_counts: pd.DataFrame
    episode_merge_counts: pd.DataFrame
    timing_counts: pd.DataFrame
    benchmark_flow_counts: pd.DataFrame
    benchmark_flow_counts_priority: pd.DataFrame
    note_time_filter_counts: pd.DataFrame
    death_censoring_counts: pd.DataFrame
    planned_admission_counts: pd.DataFrame
    section_parser_stats: pd.DataFrame
    dapt_train_manifest: pd.DataFrame
    dapt_train_sequences: pd.DataFrame
    benchmark_manifest_no_text: pd.DataFrame


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format for {path}; use parquet, csv, or csv.gz")


def require_columns(frame: pd.DataFrame, required: set[str], table_name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{table_name} missing required columns: {sorted(missing)}")


def parse_time_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], format="mixed", errors="coerce")
    return out


def normalize_inputs(admissions: pd.DataFrame, patients: pd.DataFrame, notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_columns(admissions, REQUIRED_ADMISSION_COLUMNS, "admissions")
    require_columns(patients, REQUIRED_PATIENT_COLUMNS, "patients")
    require_columns(notes, REQUIRED_NOTE_COLUMNS, "discharge_notes")

    admissions = parse_time_columns(admissions, ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"])
    patients = parse_time_columns(patients, ["dod"])
    notes = parse_time_columns(notes, ["charttime", "storetime"])
    notes = notes.copy()
    notes["note_time"] = derive_note_time(notes)
    notes["text"] = notes["text"].fillna("")
    return admissions, patients, notes


def explode_episode_map(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for episode_index, row in episodes.reset_index(drop=True).iterrows():
        for hadm_id in row["hadm_ids"]:
            rows.append(
                {
                    "hadm_id": hadm_id,
                    "episode_index": episode_index,
                    "episode_id": row["episode_id"],
                    "episode_admittime": row["admittime"],
                    "episode_dischtime": row["dischtime"],
                    "episode_hadm_ids": row["hadm_ids"],
                    "start_hadm_id": row["start_hadm_id"],
                    "end_hadm_id": row["end_hadm_id"],
                }
            )
    return pd.DataFrame(rows)


def get_death_time(row: pd.Series) -> pd.Timestamp | pd.NaT:
    candidates: list[pd.Timestamp] = []
    for column in ["dod", "deathtime"]:
        if column in row and pd.notna(row[column]):
            candidates.append(pd.Timestamp(row[column]))
    if not candidates:
        return pd.NaT
    return min(candidates)


def make_future_episode_frame(episodes: pd.DataFrame) -> dict[object, pd.DataFrame]:
    future_by_subject: dict[object, pd.DataFrame] = {}
    for subject_id, group in episodes.groupby("subject_id", sort=False):
        cols = ["episode_index", "admittime", "dischtime", "start_hadm_id"]
        optional = [col for col in ["admission_type"] if col in group.columns]
        future_by_subject[subject_id] = group[cols + optional].sort_values("admittime").copy()
    return future_by_subject


def first_readmission_for_row(
    row: pd.Series,
    future_by_subject: dict[object, pd.DataFrame],
    planned_types: Iterable[str],
    horizon_days: int,
    planned_strategy: str,
) -> tuple[pd.Timestamp | None, str]:
    future = future_by_subject.get(row["subject_id"], pd.DataFrame())
    if future.empty:
        return None, "none"
    future = future[future["episode_index"] != row["episode_index"]].copy()
    future = future.rename(columns={"start_hadm_id": "hadm_id"})
    return find_first_unplanned_readmission(
        index_dischtime=row["episode_dischtime"],
        future_admissions=future,
        planned_types=planned_types,
        horizon_days=horizon_days,
        planned_strategy=planned_strategy,
    )


def planned_skip_count_for_row(
    row: pd.Series,
    future_by_subject: dict[object, pd.DataFrame],
    planned_types: Iterable[str],
    horizon_days: int,
) -> int:
    future = future_by_subject.get(row["subject_id"], pd.DataFrame())
    if future.empty:
        return 0
    discharge_ts = pd.Timestamp(row["episode_dischtime"])
    horizon_end = discharge_ts + pd.Timedelta(days=horizon_days)
    future = future[future["episode_index"] != row["episode_index"]].copy()
    future = future[(pd.to_datetime(future["admittime"]) > discharge_ts) & (pd.to_datetime(future["admittime"]) <= horizon_end)]
    future = future.sort_values("admittime")
    skipped = 0
    for future_row in future.to_dict("records"):
        if is_planned_admission(future_row.get("admission_type"), planned_types):
            skipped += 1
            continue
        break
    return skipped


def classify_availability_timing(
    episode_dischtime: object,
    note_time: object,
    availability_time: object,
    readmission_time: object | None,
    horizon_days: int,
    max_note_delay_hours: int,
) -> str:
    discharge_ts = pd.Timestamp(episode_dischtime)
    note_ts = pd.Timestamp(note_time) if pd.notna(note_time) else pd.NaT
    if pd.isna(note_ts):
        return "exclude_missing_note_time"
    if note_ts > discharge_ts + pd.Timedelta(hours=max_note_delay_hours):
        return "exclude_late_note"
    horizon_end = discharge_ts + pd.Timedelta(days=horizon_days)
    if note_ts > horizon_end:
        return "exclude_note_after_horizon"
    if readmission_time is not None and pd.notna(readmission_time):
        readmission_ts = pd.Timestamp(readmission_time)
        if readmission_ts <= pd.Timestamp(availability_time):
            return "exclude_event_before_note"
        if readmission_ts <= horizon_end:
            return "positive"
    return "negative"


def add_section_stats(examples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    examples = examples.copy()
    rows: list[dict[str, object]] = []
    section_occurrence_counter: Counter[str] = Counter()
    section_note_counter: Counter[str] = Counter()
    section_note_counter_parsed: Counter[str] = Counter()
    section_repeat_note_counter: Counter[str] = Counter()
    occurrence_sum_for_notes_with_section: Counter[str] = Counter()
    notes_with_repeated_sections = 0

    for _, row in examples.iterrows():
        result = parse_sections(str(row.get("text", "")))
        occurrence_counts = section_occurrence_counts(result.sections)
        unique_sections = set(occurrence_counts)
        repeated_names = sorted([canonical for canonical, count in occurrence_counts.items() if count > 1])
        has_repeated_sections = bool(repeated_names)
        if has_repeated_sections:
            notes_with_repeated_sections += 1
        used_fallback = fallback_required_for_model(result)
        section_occurrence_counter.update(occurrence_counts)
        section_note_counter.update(unique_sections)
        if not used_fallback:
            section_note_counter_parsed.update(unique_sections)
        section_repeat_note_counter.update(repeated_names)
        for canonical, count in occurrence_counts.items():
            occurrence_sum_for_notes_with_section[canonical] += count
        rows.append(
            {
                "num_sections": len(result.sections),
                "num_section_occurrences": len(result.sections),
                "num_unique_sections": len(unique_sections),
                "has_repeated_sections": has_repeated_sections,
                "repeated_section_names": "|".join(repeated_names),
                "max_occurrences_for_any_section": max(occurrence_counts.values(), default=0),
                "section_coverage": float(result.coverage),
                "section_parser_success": not used_fallback,
                "used_fallback": used_fallback,
            }
        )

    if rows:
        stats_frame = pd.DataFrame(rows, index=examples.index)
        examples = pd.concat([examples, stats_frame], axis=1)
    else:
        typed_columns = {
            "num_sections": "float64",
            "num_section_occurrences": "float64",
            "num_unique_sections": "float64",
            "has_repeated_sections": "bool",
            "repeated_section_names": "object",
            "max_occurrences_for_any_section": "float64",
            "section_coverage": "float64",
            "section_parser_success": "bool",
            "used_fallback": "bool",
        }
        for column, dtype in typed_columns.items():
            examples[column] = pd.Series(dtype=dtype)

    stats_rows: list[dict[str, object]] = []
    n_examples = len(examples)
    success_count = int(examples["section_parser_success"].sum()) if n_examples else 0
    fallback_count = int(examples["used_fallback"].sum()) if n_examples else 0
    parsed_count = success_count
    stats_rows.extend(
        [
            {"metric": "n_examples", "value": n_examples},
            {"metric": "parser_success_count", "value": success_count},
            {"metric": "fallback_count", "value": fallback_count},
            {"metric": "fallback_rate", "value": fallback_count / n_examples if n_examples else 0.0},
            {"metric": "mean_num_sections", "value": float(examples["num_sections"].mean()) if n_examples else 0.0},
            {"metric": "median_num_sections", "value": float(examples["num_sections"].median()) if n_examples else 0.0},
            {"metric": "mean_num_section_occurrences", "value": float(examples["num_section_occurrences"].mean()) if n_examples else 0.0},
            {"metric": "median_num_section_occurrences", "value": float(examples["num_section_occurrences"].median()) if n_examples else 0.0},
            {"metric": "mean_num_unique_sections", "value": float(examples["num_unique_sections"].mean()) if n_examples else 0.0},
            {"metric": "median_num_unique_sections", "value": float(examples["num_unique_sections"].median()) if n_examples else 0.0},
            {"metric": "mean_section_coverage", "value": float(examples["section_coverage"].mean()) if n_examples else 0.0},
            {"metric": "notes_with_repeated_sections", "value": notes_with_repeated_sections},
            {"metric": "notes_with_repeated_sections_rate", "value": notes_with_repeated_sections / n_examples if n_examples else 0.0},
        ]
    )
    all_canonicals = sorted(set(section_occurrence_counter) | set(section_note_counter) | set(section_note_counter_parsed) | set(section_repeat_note_counter))
    for canonical in all_canonicals:
        occurrence_count = int(section_occurrence_counter.get(canonical, 0))
        note_count = int(section_note_counter.get(canonical, 0))
        parsed_note_count = int(section_note_counter_parsed.get(canonical, 0))
        repeat_note_count = int(section_repeat_note_counter.get(canonical, 0))
        stats_rows.extend(
            [
                {"metric": f"section_occurrence_count.{canonical}", "value": occurrence_count},
                {"metric": f"section_note_count.{canonical}", "value": note_count},
                {"metric": f"section_note_count_parsed.{canonical}", "value": parsed_note_count},
                {"metric": f"section_note_prevalence_all.{canonical}", "value": note_count / n_examples if n_examples else 0.0},
                {"metric": f"section_note_prevalence_parsed.{canonical}", "value": parsed_note_count / parsed_count if parsed_count else 0.0},
                {"metric": f"section_repeat_note_count.{canonical}", "value": repeat_note_count},
                {"metric": f"section_repeat_rate.{canonical}", "value": repeat_note_count / note_count if note_count else 0.0},
                {
                    "metric": f"section_mean_occurrences_per_note_with_section.{canonical}",
                    "value": occurrence_sum_for_notes_with_section.get(canonical, 0) / note_count if note_count else 0.0,
                },
            ]
        )
    return examples, pd.DataFrame(stats_rows)


def whitespace_tokenize(text: str) -> list[int]:
    return list(range(len(text.split())))


def build_dapt_artifacts(split_examples: pd.DataFrame, max_length: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_examples.empty or "split" not in split_examples.columns:
        return pd.DataFrame(), pd.DataFrame()

    train_examples = split_examples[split_examples["split"] == "train"].copy()
    val_test_subjects = set(split_examples.loc[split_examples["split"].isin(["val", "test"]), "subject_id"])
    train_subjects = set(train_examples["subject_id"])
    leakage = train_subjects & val_test_subjects
    if leakage:
        raise AssertionError(f"DAPT train subjects overlap val/test subjects: {len(leakage)}")

    sequences = build_patient_sequences(train_examples)
    _, manifest = pack_patient_sequences(sequences, whitespace_tokenize, max_length=max_length)
    if manifest.empty:
        manifest["subject_id_hash"] = pd.Series(dtype="object")
    else:
        manifest["subject_id_hash"] = manifest["subject_id"].map(stable_hash)
        manifest = manifest.drop(columns=["subject_id"])

    sequences = sequences.copy()
    if not sequences.empty:
        sequences["subject_id_hash"] = sequences["subject_id"].map(stable_hash)
        sequences = sequences.drop(columns=["subject_id"])
    return manifest, sequences


def make_no_text_manifest(split_examples: pd.DataFrame) -> pd.DataFrame:
    manifest = split_examples.copy()
    if "subject_id" in manifest.columns:
        manifest["subject_id_hash"] = manifest["subject_id"].map(stable_hash)
        manifest = manifest.drop(columns=["subject_id"])
    if "episode_id" in manifest.columns:
        manifest["episode_id_hash"] = manifest["episode_id"].map(stable_hash)
    for raw_id_col in ["episode_id", "hadm_id", "start_hadm_id", "end_hadm_id", "episode_hadm_ids", "note_id"]:
        if raw_id_col in manifest.columns:
            manifest = manifest.drop(columns=[raw_id_col])
    drop_cols = [col for col in REPORT_COLUMNS_WITHOUT_TEXT if col in manifest.columns]
    manifest = manifest.drop(columns=drop_cols, errors="ignore")
    preferred = [
        "example_id",
        "note_id_hash",
        "subject_id_hash",
        "split",
        "label",
        "readmission_time",
        "label_reason",
        "timing_status",
        "censor_status",
        "availability_time",
        "episode_id_hash",
        "episode_admittime",
        "episode_dischtime",
        "note_time",
        "num_sections",
        "num_section_occurrences",
        "num_unique_sections",
        "has_repeated_sections",
        "repeated_section_names",
        "max_occurrences_for_any_section",
        "section_coverage",
        "section_parser_success",
        "used_fallback",
    ]
    cols = [col for col in preferred if col in manifest.columns]
    extra = [col for col in manifest.columns if col not in cols]
    return manifest[cols + extra]


def flow_frame(rows: list[tuple[str, int]]) -> pd.DataFrame:
    return pd.DataFrame([{"step": step, "count": int(count)} for step, count in rows])


def select_note_columns(examples: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "example_id",
        "note_id_hash",
        "subject_id",
        "hadm_id",
        "episode_id",
        "episode_hadm_ids",
        "start_hadm_id",
        "end_hadm_id",
        "admittime",
        "dischtime",
        "episode_admittime",
        "episode_dischtime",
        "note_time",
        "availability_time",
        "split",
        "label",
        "readmission_time",
        "label_reason",
        "timing_status",
        "censor_status",
        "exclusion_reason_priority",
        "admission_type",
        "discharge_location",
        "anchor_age",
        "num_sections",
        "num_section_occurrences",
        "num_unique_sections",
        "has_repeated_sections",
        "repeated_section_names",
        "max_occurrences_for_any_section",
        "section_coverage",
        "section_parser_success",
        "used_fallback",
        "text",
    ]
    cols = [col for col in preferred if col in examples.columns]
    extra = [col for col in examples.columns if col not in cols and col not in {"dod", "deathtime"}]
    return examples[cols + extra]


def build_lampr(
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    notes: pd.DataFrame,
    config: dict[str, Any],
) -> LamprBuildResult:
    admissions, patients, notes = normalize_inputs(admissions, patients, notes)

    counts: Counter[str] = Counter()
    counts["raw_admissions"] = len(admissions)
    counts["raw_patients"] = len(patients)
    counts["raw_discharge_notes"] = len(notes)
    counts["missing_note_time"] = int(notes["note_time"].isna().sum())
    counts["empty_note_text"] = int(notes["text"].str.strip().eq("").sum())

    horizon_days = int(config.get("cohort", {}).get("horizon_days", 30))
    max_note_delay_hours = int(config.get("note_timing", {}).get("max_note_delay_hours", 168))
    adult_only = bool(config.get("cohort", {}).get("adult_only_if_age_available", True))
    require_tail_followup = bool(config.get("cohort", {}).get("require_tail_followup", True))
    tail_reference = str(config.get("cohort", {}).get("tail_reference", "dataset_max_dischtime"))
    planned_types = config.get("planned_admissions", {}).get("types", [])
    planned_strategy = str(config.get("planned_admissions", {}).get("strategy", "continue_search"))
    gap_hours = float(config.get("episode_merge", {}).get("gap_hours", 24))
    dapt_max_length = int(config.get("dapt", {}).get("max_length", 4096))

    episodes_source = admissions.copy()
    if "admission_type" not in episodes_source.columns:
        episodes_source["admission_type"] = ""
    episodes, episode_counts = merge_continuous_care_episodes(episodes_source, gap_hours=gap_hours)
    episodes = episodes.reset_index(drop=True).copy()
    episodes["episode_index"] = episodes.index
    if "admission_type" not in episodes.columns and "admission_type" in admissions.columns:
        first_adm_type = admissions.sort_values(["subject_id", "admittime"]).drop_duplicates("hadm_id")[["hadm_id", "admission_type"]]
        episodes = episodes.merge(first_adm_type, left_on="start_hadm_id", right_on="hadm_id", how="left").drop(columns=["hadm_id"])

    patient_cols = ["subject_id"] + [col for col in ["anchor_age", "gender", "dod"] if col in patients.columns]
    episode_candidates = episodes.merge(patients[patient_cols], on="subject_id", how="left")

    alive_mask = episode_candidates.apply(lambda row: not is_in_hospital_death(row), axis=1)
    episode_candidates = episode_candidates.loc[alive_mask].copy()
    counts["after_alive_discharge"] = len(episode_candidates)

    if adult_only and "anchor_age" in episode_candidates.columns:
        adult_mask = episode_candidates["anchor_age"].isna() | (episode_candidates["anchor_age"] >= 18)
        episode_candidates = episode_candidates.loc[adult_mask].copy()
    counts["after_adult_filter"] = len(episode_candidates)

    if require_tail_followup and tail_reference == "dataset_max_dischtime":
        dataset_max_dischtime = admissions["dischtime"].max()
        followup_mask = episode_candidates["dischtime"] + pd.Timedelta(days=horizon_days) <= dataset_max_dischtime
        episode_candidates = episode_candidates.loc[followup_mask].copy()
    counts["after_tail_followup"] = len(episode_candidates)

    note_adm = notes.merge(
        admissions,
        on=["subject_id", "hadm_id"],
        how="inner",
        suffixes=("_note", ""),
    )
    counts["notes_linked_to_admission"] = len(note_adm)

    terminal_episode_cols = [
        "subject_id",
        "episode_index",
        "episode_id",
        "start_hadm_id",
        "end_hadm_id",
        "hadm_ids",
        "admittime",
        "dischtime",
        "anchor_age",
        "gender",
        "dod",
    ]
    terminal_episode_cols = [col for col in terminal_episode_cols if col in episode_candidates.columns]
    terminal_episodes = episode_candidates[terminal_episode_cols].rename(
        columns={
            "admittime": "episode_admittime",
            "dischtime": "episode_dischtime",
            "hadm_ids": "episode_hadm_ids",
        }
    )
    terminal_notes = note_adm.merge(
        terminal_episodes,
        left_on=["subject_id", "hadm_id"],
        right_on=["subject_id", "end_hadm_id"],
        how="inner",
        suffixes=("", "_episode"),
    )
    counts["terminal_discharge_notes_linked"] = len(terminal_notes)
    counts["episodes_with_terminal_discharge_note"] = int(terminal_notes["episode_index"].nunique()) if not terminal_notes.empty else 0
    counts["no_terminal_discharge_note"] = int(len(episode_candidates) - counts["episodes_with_terminal_discharge_note"])

    terminal_notes["availability_time"] = terminal_notes[["episode_dischtime", "note_time"]].max(axis=1)
    terminal_notes["note_text_nonempty"] = terminal_notes["text"].str.strip().ne("")
    terminal_notes["note_time_present"] = terminal_notes["note_time"].notna()
    terminal_notes["note_on_time"] = terminal_notes["note_time"] <= (
        terminal_notes["episode_dischtime"] + pd.Timedelta(hours=max_note_delay_hours)
    )

    counts["terminal_missing_note_time"] = int((~terminal_notes["note_time_present"]).sum())
    counts["terminal_empty_note_text"] = int((~terminal_notes["note_text_nonempty"]).sum())
    counts["terminal_late_note"] = int((terminal_notes["note_time_present"] & terminal_notes["note_text_nonempty"] & ~terminal_notes["note_on_time"]).sum())

    qualifying_notes = terminal_notes[
        terminal_notes["note_time_present"] & terminal_notes["note_text_nonempty"] & terminal_notes["note_on_time"]
    ].copy()
    counts["terminal_qualifying_notes"] = len(qualifying_notes)
    counts["episodes_with_qualifying_terminal_note"] = int(qualifying_notes["episode_index"].nunique()) if not qualifying_notes.empty else 0

    sort_cols = ["subject_id", "episode_index", "availability_time", "note_time"]
    if "note_id" in qualifying_notes.columns:
        sort_cols.append("note_id")
    selected_notes = qualifying_notes.sort_values(sort_cols).drop_duplicates(["subject_id", "episode_id"], keep="first").copy()
    counts["selected_episode_examples"] = len(selected_notes)
    counts["duplicate_terminal_notes_removed"] = int(len(qualifying_notes) - len(selected_notes))

    future_by_subject = make_future_episode_frame(episodes)
    readmission_times: list[pd.Timestamp | None] = []
    readmission_reasons: list[str] = []
    timing_statuses: list[str] = []
    censor_statuses: list[str] = []
    labels: list[int | None] = []
    exclusion_reasons: list[str] = []
    planned_skip_counts: list[int] = []

    for _, row in selected_notes.iterrows():
        readmission_time, reason = first_readmission_for_row(
            row=row,
            future_by_subject=future_by_subject,
            planned_types=planned_types,
            horizon_days=horizon_days,
            planned_strategy=planned_strategy,
        )
        planned_skip_counts.append(
            planned_skip_count_for_row(
                row=row,
                future_by_subject=future_by_subject,
                planned_types=planned_types,
                horizon_days=horizon_days,
            )
        )
        timing_status = classify_availability_timing(
            episode_dischtime=row["episode_dischtime"],
            note_time=row["note_time"],
            availability_time=row["availability_time"],
            readmission_time=readmission_time,
            horizon_days=horizon_days,
            max_note_delay_hours=max_note_delay_hours,
        )
        death_time = get_death_time(row)
        censor_status = death_censor_status(
            dischtime=row["episode_dischtime"],
            death_time=death_time,
            readmission_time=readmission_time,
            horizon_days=horizon_days,
        )

        label: int | None
        if timing_status == "exclude_late_note":
            label = None
            exclusion_reason = "late_note"
        elif timing_status == "exclude_event_before_note":
            label = None
            exclusion_reason = "event_before_availability"
        elif censor_status == "censored_death_before_readmission":
            label = None
            exclusion_reason = "death_censoring"
        elif reason == "exclude_planned_first":
            label = None
            exclusion_reason = "planned_first_exclusion"
        elif timing_status == "positive":
            label = 1
            exclusion_reason = "labeled"
        elif timing_status == "negative" and censor_status in {"not_censored", "readmission_before_death"}:
            label = 0
            exclusion_reason = "labeled"
        else:
            label = None
            exclusion_reason = timing_status

        readmission_times.append(readmission_time)
        readmission_reasons.append(reason)
        timing_statuses.append(timing_status)
        censor_statuses.append(censor_status)
        labels.append(label)
        exclusion_reasons.append(exclusion_reason)

    examples = selected_notes.copy()
    examples["readmission_time"] = readmission_times
    examples["label_reason"] = readmission_reasons
    examples["timing_status"] = timing_statuses
    examples["censor_status"] = censor_statuses
    examples["label"] = labels
    examples["exclusion_reason_priority"] = exclusion_reasons
    examples["planned_skip_count"] = planned_skip_counts

    counts["timing_exclude_event_before_note"] = int((examples["timing_status"] == "exclude_event_before_note").sum())
    counts["timing_exclude_late_note"] = int((examples["timing_status"] == "exclude_late_note").sum())
    counts["death_censored"] = int((examples["censor_status"] == "censored_death_before_readmission").sum())
    labeled_examples = examples[examples["label"].notna()].copy()
    labeled_examples["label"] = labeled_examples["label"].astype(int)
    counts["final_labeled_examples"] = len(labeled_examples)
    counts["final_positive"] = int((labeled_examples["label"] == 1).sum())
    counts["final_negative"] = int((labeled_examples["label"] == 0).sum())

    labeled_examples, section_parser_stats = add_section_stats(labeled_examples)

    if labeled_examples.empty:
        split_examples = labeled_examples.copy()
        split_counts = pd.DataFrame(columns=["split", "n_examples", "n_subjects", "positive", "negative"])
    else:
        split_cfg = config.get("splits", {})
        split_examples = make_patient_splits(
            labeled_examples,
            seed=int(split_cfg.get("seed", 17)),
            train_frac=float(split_cfg.get("train_frac", 0.8)),
            val_frac=float(split_cfg.get("val_frac", 0.1)),
        )
        assert_no_patient_overlap(split_examples)
        split_counts = (
            split_examples.groupby("split")
            .agg(
                n_examples=("label", "size"),
                n_subjects=("subject_id", "nunique"),
                positive=("label", "sum"),
            )
            .reset_index()
        )
        split_counts["negative"] = split_counts["n_examples"] - split_counts["positive"]

    note_id_source = split_examples["note_id"] if "note_id" in split_examples.columns else split_examples.index
    split_examples = split_examples.copy()
    split_examples["note_id_hash"] = note_id_source.map(stable_hash)
    split_examples["example_id"] = [f"lampr_{idx:08d}" for idx in range(len(split_examples))]
    split_examples = select_note_columns(split_examples)

    dapt_train_manifest, dapt_train_sequences = build_dapt_artifacts(split_examples, max_length=dapt_max_length)
    benchmark_manifest_no_text = make_no_text_manifest(split_examples)

    filter_counts = pd.DataFrame([{"step": key, "count": int(value)} for key, value in counts.items()])
    label_counts = split_examples["label"].value_counts(dropna=False).rename_axis("label").to_frame("count").reset_index()
    timing_counts = (
        pd.DataFrame({"timing_status": timing_statuses, "censor_status": censor_statuses})
        .value_counts(["timing_status", "censor_status"])
        .rename("count")
        .reset_index()
    )
    episode_merge_counts = pd.DataFrame([{"metric": key, "count": int(value)} for key, value in episode_counts.items()])
    priority_exclusions = pd.Series(exclusion_reasons, dtype="object")
    selected_episode_examples = counts["selected_episode_examples"]
    event_before_availability = int((priority_exclusions == "event_before_availability").sum())
    after_event_before_availability = selected_episode_examples - event_before_availability
    death_censoring_priority = int((priority_exclusions == "death_censoring").sum())
    after_death_censoring = after_event_before_availability - death_censoring_priority
    planned_first_priority = int((priority_exclusions == "planned_first_exclusion").sum())
    after_planned_first_exclusion = after_death_censoring - planned_first_priority
    benchmark_flow_counts = flow_frame(
        [
            ("raw_admissions", len(admissions)),
            ("merged_episodes", len(episodes)),
            ("after_alive_discharge", counts["after_alive_discharge"]),
            ("after_adult_filter", counts["after_adult_filter"]),
            ("after_tail_followup", counts["after_tail_followup"]),
            ("episodes_with_terminal_discharge_note", counts["episodes_with_terminal_discharge_note"]),
            ("after_note_time_text_on_time", counts["episodes_with_qualifying_terminal_note"]),
            ("one_example_per_episode", counts["selected_episode_examples"]),
            ("after_timing_censor_planned_exclusions", counts["final_labeled_examples"]),
        ]
    )
    benchmark_flow_counts_priority = flow_frame(
        [
            ("selected_episode_examples", selected_episode_examples),
            ("after_event_before_availability_exclusion", after_event_before_availability),
            ("after_death_censoring", after_death_censoring),
            ("after_planned_first_exclusion", after_planned_first_exclusion),
            ("final_labeled_examples", counts["final_labeled_examples"]),
        ]
    )
    note_time_filter_counts = pd.DataFrame(
        [
            {"metric": "raw_discharge_notes", "count": len(notes)},
            {"metric": "raw_missing_note_time", "count": counts["missing_note_time"]},
            {"metric": "raw_empty_note_text", "count": counts["empty_note_text"]},
            {"metric": "terminal_missing_note_time", "count": counts["terminal_missing_note_time"]},
            {"metric": "terminal_empty_note_text", "count": counts["terminal_empty_note_text"]},
            {"metric": "terminal_late_note", "count": counts["terminal_late_note"]},
            {"metric": "timing_exclude_event_before_availability", "count": counts["timing_exclude_event_before_note"]},
        ]
    )
    death_censoring_counts = (
        pd.DataFrame({"censor_status": censor_statuses})
        .value_counts("censor_status")
        .rename("count")
        .reset_index()
        if censor_statuses
        else pd.DataFrame(columns=["censor_status", "count"])
    )
    planned_reason_counts = (
        pd.DataFrame({"label_reason": readmission_reasons}).value_counts("label_reason").rename("count").reset_index()
        if readmission_reasons
        else pd.DataFrame(columns=["label_reason", "count"])
    )
    planned_rows = [
        {"metric": "planned_admission_skipped_count", "count": int(sum(planned_skip_counts))},
        {"metric": "rows_with_planned_admission_skipped", "count": int(sum(1 for value in planned_skip_counts if value > 0))},
    ]
    for row in planned_reason_counts.to_dict("records"):
        planned_rows.append({"metric": f"label_reason.{row['label_reason']}", "count": int(row["count"])})
    planned_admission_counts = pd.DataFrame(planned_rows)

    return LamprBuildResult(
        examples=labeled_examples,
        split_examples=split_examples,
        filter_counts=filter_counts,
        split_counts=split_counts,
        label_counts=label_counts,
        episode_merge_counts=episode_merge_counts,
        timing_counts=timing_counts,
        benchmark_flow_counts=benchmark_flow_counts,
        benchmark_flow_counts_priority=benchmark_flow_counts_priority,
        note_time_filter_counts=note_time_filter_counts,
        death_censoring_counts=death_censoring_counts,
        planned_admission_counts=planned_admission_counts,
        section_parser_stats=section_parser_stats,
        dapt_train_manifest=dapt_train_manifest,
        dapt_train_sequences=dapt_train_sequences,
        benchmark_manifest_no_text=benchmark_manifest_no_text,
    )
