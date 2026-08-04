from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.utils.privacy import stable_hash  # noqa: E402


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def read_split(root: Path, split: str) -> pd.DataFrame:
    path = root / f"benchmark_examples_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    frame = pd.read_parquet(path)
    if "split" not in frame.columns:
        frame["split"] = split
    return frame


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"{name} missing required columns: {sorted(missing)}")


def assert_no_split_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    train_s = set(train["subject_id"])
    val_s = set(val["subject_id"])
    test_s = set(test["subject_id"])
    overlaps = {
        "train_val": train_s & val_s,
        "train_test": train_s & test_s,
        "val_test": val_s & test_s,
    }
    bad = {name: len(values) for name, values in overlaps.items() if values}
    if bad:
        raise AssertionError(f"Subject overlap detected: {bad}")


def assert_section_stats(root: Path) -> None:
    path = root / "section_parser_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    stats = pd.read_csv(path)
    require_columns(stats, {"metric", "value"}, "section_parser_stats")
    metrics = set(stats["metric"])
    if any(metric.startswith("section_frequency.") for metric in metrics):
        raise AssertionError("section_parser_stats still contains deprecated section_frequency.* metrics")
    required_prefixes = [
        "section_occurrence_count.",
        "section_note_count.",
        "section_note_count_parsed.",
        "section_note_prevalence_all.",
        "section_note_prevalence_parsed.",
        "section_repeat_note_count.",
        "section_repeat_rate.",
        "section_mean_occurrences_per_note_with_section.",
    ]
    missing_prefixes = [prefix for prefix in required_prefixes if not any(metric.startswith(prefix) for metric in metrics)]
    if missing_prefixes:
        raise AssertionError(f"section_parser_stats missing metric prefixes: {missing_prefixes}")
    prevalence = stats[stats["metric"].str.startswith("section_note_prevalence_")]["value"].astype(float)
    if not prevalence.between(0.0, 1.0).all():
        bad = stats[stats["metric"].str.startswith("section_note_prevalence_") & ~stats["value"].astype(float).between(0.0, 1.0)]
        raise AssertionError(f"Section prevalence outside [0, 1]: {bad.to_dict('records')[:5]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate frozen LAMP-R benchmark artifacts.")
    parser.add_argument("--root", required=True, help="Benchmark output directory, e.g. results/lampr_full_v0_2")
    args = parser.parse_args()

    root = resolve_path(args.root)
    train = read_split(root, "train")
    val = read_split(root, "val")
    test = read_split(root, "test")
    df = pd.concat([train, val, test], ignore_index=True)

    require_columns(
        df,
        {
            "subject_id",
            "episode_id",
            "example_id",
            "label",
            "split",
            "readmission_time",
            "availability_time",
            "censor_status",
            "timing_status",
            "end_hadm_id",
            "hadm_id",
            "num_section_occurrences",
            "num_unique_sections",
            "has_repeated_sections",
            "max_occurrences_for_any_section",
        },
        "benchmark examples",
    )

    assert_no_split_overlap(train, val, test)

    max_per_episode = int(df.groupby(["subject_id", "episode_id"]).size().max()) if not df.empty else 0
    if max_per_episode > 1:
        raise AssertionError(f"More than one example per subject_id + episode_id; max={max_per_episode}")

    if not df["example_id"].is_unique:
        raise AssertionError("example_id is not unique")

    if not (df["hadm_id"] == df["end_hadm_id"]).all():
        raise AssertionError("Representative note hadm_id must equal terminal end_hadm_id")

    pos = df[df["label"] == 1].copy()
    if not pos.empty:
        pos["readmission_time"] = pd.to_datetime(pos["readmission_time"])
        pos["availability_time"] = pd.to_datetime(pos["availability_time"])
        if not (pos["readmission_time"] > pos["availability_time"]).all():
            raise AssertionError("Positive examples must have readmission_time > availability_time")

    if (df["censor_status"] == "censored_death_before_readmission").any():
        raise AssertionError("Death-censored examples should not be labeled examples")

    manifest = pd.read_parquet(root / "benchmark_manifest_no_text.parquet")
    forbidden_manifest_cols = {"text", "sequence_text", "subject_id", "dod", "deathtime"}
    forbidden_present = forbidden_manifest_cols & set(manifest.columns)
    if forbidden_present:
        raise AssertionError(f"No-text manifest contains forbidden columns: {sorted(forbidden_present)}")
    if len(manifest) != len(df):
        raise AssertionError(f"No-text manifest row count {len(manifest)} != labeled examples {len(df)}")

    dapt_manifest = pd.read_parquet(root / "dapt_train_manifest.parquet")
    if "sequence_text" in dapt_manifest.columns or "subject_id" in dapt_manifest.columns:
        raise AssertionError("DAPT manifest should contain subject_id_hash only, not raw subject_id or sequence_text")
    train_hashes = {stable_hash(value) for value in train["subject_id"].unique()}
    dapt_hashes = set(dapt_manifest["subject_id_hash"]) if not dapt_manifest.empty else set()
    if not dapt_hashes.issubset(train_hashes):
        raise AssertionError("DAPT manifest contains subjects outside train split")

    dapt_sequences = pd.read_parquet(root / "dapt_train_sequences.parquet")
    if "subject_id" in dapt_sequences.columns:
        raise AssertionError("DAPT sequences should not contain raw subject_id")
    if set(dapt_sequences["subject_id_hash"]) - train_hashes:
        raise AssertionError("DAPT sequences contain subjects outside train split")

    assert_section_stats(root)

    print("All LAMP-R validation checks passed.")
    print(f"root={root}")
    print(f"n_examples={len(df)}")
    print(f"n_train={len(train)} n_val={len(val)} n_test={len(test)}")
    print(f"n_subjects={df['subject_id'].nunique()}")
    print(f"n_positive={int((df['label'] == 1).sum())} n_negative={int((df['label'] == 0).sum())}")
    print(f"dapt_manifest_rows={len(dapt_manifest)}")
    print(f"dapt_sequence_patients={dapt_sequences['subject_id_hash'].nunique() if not dapt_sequences.empty else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
