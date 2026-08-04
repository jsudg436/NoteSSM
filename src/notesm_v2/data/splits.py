from __future__ import annotations

import random

import pandas as pd


def make_patient_splits(
    examples: pd.DataFrame,
    seed: int = 17,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> pd.DataFrame:
    """Assign train/val/test splits with no subject overlap."""
    if "subject_id" not in examples.columns:
        raise ValueError("examples must contain subject_id")
    subjects = sorted(examples["subject_id"].dropna().unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train : n_train + n_val])

    def split_for(subject_id: object) -> str:
        if subject_id in train_subjects:
            return "train"
        if subject_id in val_subjects:
            return "val"
        return "test"

    out = examples.copy()
    out["split"] = out["subject_id"].map(split_for)
    return out


def assert_no_patient_overlap(split_examples: pd.DataFrame) -> None:
    groups = {
        split: set(group["subject_id"].dropna().unique().tolist())
        for split, group in split_examples.groupby("split")
    }
    for left_name, left_subjects in groups.items():
        for right_name, right_subjects in groups.items():
            if left_name >= right_name:
                continue
            overlap = left_subjects & right_subjects
            if overlap:
                raise AssertionError(f"subject overlap between {left_name} and {right_name}: {overlap}")
