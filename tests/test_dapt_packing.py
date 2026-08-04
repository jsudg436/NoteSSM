from __future__ import annotations

import pandas as pd

from notesm_v2.data.dapt_packing import build_patient_sequences, pack_patient_sequences


def whitespace_tokenize(text: str) -> list[int]:
    return list(range(len(text.split())))


def test_patient_sequences_are_chronological_and_do_not_insert_identifiers() -> None:
    notes = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 20,
                "admittime": "2020-02-01",
                "dischtime": "2020-02-02",
                "note_time": "2020-02-03",
                "text": "second admission text",
            },
            {
                "subject_id": 1,
                "hadm_id": 10,
                "admittime": "2020-01-01",
                "dischtime": "2020-01-02",
                "note_time": "2020-01-03",
                "text": "first admission text",
            },
        ]
    )
    sequences = build_patient_sequences(notes)
    text = sequences.iloc[0]["sequence_text"]
    assert text.index("first admission text") < text.index("second admission text")
    assert "hadm_id" not in text
    assert "subject_id" not in text


def test_packing_never_mixes_patients() -> None:
    notes = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 10,
                "admittime": "2020-01-01",
                "dischtime": "2020-01-02",
                "note_time": "2020-01-03",
                "text": "a " * 20,
            },
            {
                "subject_id": 2,
                "hadm_id": 20,
                "admittime": "2020-01-01",
                "dischtime": "2020-01-02",
                "note_time": "2020-01-03",
                "text": "b " * 20,
            },
        ]
    )
    sequences = build_patient_sequences(notes)
    _, manifest = pack_patient_sequences(sequences, whitespace_tokenize, max_length=10)
    assert all(group["subject_id"].nunique() == 1 for _, group in manifest.groupby(["subject_id", "block_id"]))
    assert set(manifest["subject_id"]) == {1, 2}
