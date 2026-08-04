from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aggregate_head_seed_results", ROOT / "scripts" / "aggregate_head_seed_results.py")
aggregator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(aggregator)


def test_read_seed_metrics_and_setting_parse() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        run_dir = root / "mamba2_setting"
        run_dir.mkdir()
        pd.DataFrame(
            {
                "split": ["val", "test"],
                "AUPRC": [0.2, 0.3],
                "AUROC": [0.6, 0.7],
                "F1": [0.4, 0.5],
                "threshold": [0.1, 0.1],
            }
        ).to_csv(run_dir / "notesm_v1_threshold_tuned_metrics_seed17.csv", index=False)

        frame = aggregator.read_seed_metrics(run_dir, "notesm_v1", 17, tuned=True)

        assert aggregator.parse_setting(run_dir, "mamba2_") == "setting"
        assert frame["seed"].tolist() == [17, 17]
        assert frame["split"].tolist() == ["val", "test"]
