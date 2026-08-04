from __future__ import annotations

import pandas as pd


def merge_continuous_care_episodes(
    admissions: pd.DataFrame,
    gap_hours: float = 24.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Merge admissions for the same patient separated by <= gap_hours.

    Input columns required: subject_id, hadm_id, admittime, dischtime.
    The output contains one row per merged episode with hadm_ids as a tuple.
    """
    required = {"subject_id", "hadm_id", "admittime", "dischtime"}
    missing = required - set(admissions.columns)
    if missing:
        raise ValueError(f"admissions missing columns: {sorted(missing)}")

    if admissions.empty:
        return admissions.copy(), {"raw_admissions": 0, "episodes": 0, "merged_admissions": 0}

    df = admissions.copy()
    df["admittime"] = pd.to_datetime(df["admittime"], format="mixed")
    df["dischtime"] = pd.to_datetime(df["dischtime"], format="mixed")
    df = df.sort_values(["subject_id", "admittime", "dischtime", "hadm_id"])

    rows: list[dict[str, object]] = []
    gap = pd.Timedelta(hours=gap_hours)

    for subject_id, group in df.groupby("subject_id", sort=False):
        current: dict[str, object] | None = None
        for row in group.to_dict("records"):
            if current is None:
                current = {
                    "subject_id": subject_id,
                    "episode_id": f"{subject_id}:0",
                    "start_hadm_id": row["hadm_id"],
                    "end_hadm_id": row["hadm_id"],
                    "hadm_ids": [row["hadm_id"]],
                    "admittime": row["admittime"],
                    "dischtime": row["dischtime"],
                }
                for col in ["hospital_expire_flag", "deathtime", "discharge_location"]:
                    if col in row:
                        current[col] = row[col]
                continue

            previous_end = pd.to_datetime(current["dischtime"])
            next_start = pd.to_datetime(row["admittime"])
            if next_start - previous_end <= gap:
                current["hadm_ids"].append(row["hadm_id"])  # type: ignore[index]
                row_dischtime = pd.to_datetime(row["dischtime"])
                if row_dischtime >= previous_end:
                    current["end_hadm_id"] = row["hadm_id"]
                    current["dischtime"] = row_dischtime
                if "hospital_expire_flag" in row:
                    current["hospital_expire_flag"] = max(
                        int(current.get("hospital_expire_flag") or 0),
                        int(row.get("hospital_expire_flag") or 0),
                    )
                if "deathtime" in row and pd.notna(row.get("deathtime")):
                    current["deathtime"] = row["deathtime"]
                if "discharge_location" in row and row.get("discharge_location"):
                    current["discharge_location"] = row["discharge_location"]
            else:
                rows.append(current)
                current = {
                    "subject_id": subject_id,
                    "episode_id": f"{subject_id}:{len(rows)}",
                    "start_hadm_id": row["hadm_id"],
                    "end_hadm_id": row["hadm_id"],
                    "hadm_ids": [row["hadm_id"]],
                    "admittime": row["admittime"],
                    "dischtime": row["dischtime"],
                }
                for col in ["hospital_expire_flag", "deathtime", "discharge_location"]:
                    if col in row:
                        current[col] = row[col]
        if current is not None:
            rows.append(current)

    episodes = pd.DataFrame(rows)
    episodes["hadm_ids"] = episodes["hadm_ids"].apply(tuple)
    counts = {
        "raw_admissions": int(len(df)),
        "episodes": int(len(episodes)),
        "merged_admissions": int(len(df) - len(episodes)),
    }
    return episodes, counts
