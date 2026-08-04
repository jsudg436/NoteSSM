from __future__ import annotations

from collections.abc import Callable

import pandas as pd


SPECIAL_TOKENS = [
    "<PATIENT_START>",
    "<PATIENT_END>",
    "<ADM_START>",
    "<ADM_END>",
    "<NOTE_TYPE_DISCHARGE>",
    "<NOTE_END>",
    "<TIME_GAP_0D>",
    "<TIME_GAP_1_7D>",
    "<TIME_GAP_8_30D>",
    "<TIME_GAP_31_90D>",
    "<TIME_GAP_90D_PLUS>",
    "<TIME_GAP_UNKNOWN>",
    "<NOTE_SUM>",
    "<SEC_SUM>",
    "<CHUNK_SUM>",
]


def time_gap_token(previous_time: object | None, next_time: object | None) -> str:
    if previous_time is None or next_time is None or pd.isna(previous_time) or pd.isna(next_time):
        return "<TIME_GAP_UNKNOWN>"
    days = (pd.Timestamp(next_time) - pd.Timestamp(previous_time)).days
    if days <= 0:
        return "<TIME_GAP_0D>"
    if days <= 7:
        return "<TIME_GAP_1_7D>"
    if days <= 30:
        return "<TIME_GAP_8_30D>"
    if days <= 90:
        return "<TIME_GAP_31_90D>"
    return "<TIME_GAP_90D_PLUS>"


def build_patient_sequences(notes: pd.DataFrame) -> pd.DataFrame:
    """Build chronological patient text sequences without inserting identifiers."""
    required = {"subject_id", "hadm_id", "text", "admittime", "dischtime", "note_time"}
    missing = required - set(notes.columns)
    if missing:
        raise ValueError(f"notes missing columns: {sorted(missing)}")

    sort_cols = ["subject_id", "admittime", "dischtime", "note_time"]
    if "note_seq" in notes.columns:
        sort_cols.append("note_seq")
    df = notes.copy()
    for col in ["admittime", "dischtime", "note_time"]:
        df[col] = pd.to_datetime(df[col])
    df = df.sort_values(sort_cols)

    rows: list[dict[str, object]] = []
    for subject_id, group in df.groupby("subject_id", sort=False):
        pieces = ["<PATIENT_START>"]
        prev_time = None
        for row in group.to_dict("records"):
            pieces.extend(
                [
                    time_gap_token(prev_time, row["admittime"]),
                    "<ADM_START>",
                    "<NOTE_TYPE_DISCHARGE>",
                    str(row["text"]),
                    "<NOTE_END>",
                    "<ADM_END>",
                ]
            )
            prev_time = row["dischtime"]
        pieces.append("<PATIENT_END>")
        rows.append({"subject_id": subject_id, "sequence_text": "\n".join(pieces), "num_notes": len(group)})
    return pd.DataFrame(rows)


def pack_patient_sequences(
    patient_sequences: pd.DataFrame,
    tokenize: Callable[[str], list[int]],
    max_length: int,
) -> tuple[list[list[int]], pd.DataFrame]:
    """Pack each patient independently into token blocks."""
    blocks: list[list[int]] = []
    manifest_rows: list[dict[str, object]] = []
    for row in patient_sequences.to_dict("records"):
        token_ids = tokenize(str(row["sequence_text"]))
        for block_index, start in enumerate(range(0, len(token_ids), max_length)):
            block = token_ids[start : start + max_length]
            blocks.append(block)
            manifest_rows.append(
                {
                    "subject_id": row["subject_id"],
                    "block_id": block_index,
                    "token_length": len(block),
                    "num_notes_in_patient": row["num_notes"],
                }
            )
    return blocks, pd.DataFrame(manifest_rows)
