from __future__ import annotations

import pandas as pd

from notesm_v2.data.splits import assert_no_patient_overlap, make_patient_splits


def test_patient_split_has_no_subject_overlap() -> None:
    examples = pd.DataFrame({"subject_id": list(range(30)), "label": [0, 1] * 15})
    split = make_patient_splits(examples, seed=3)
    assert set(split["split"]) == {"train", "val", "test"}
    assert_no_patient_overlap(split)
