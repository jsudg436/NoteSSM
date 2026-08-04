from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_dapt_script", ROOT / "scripts" / "run_dapt.py")
run_dapt = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_dapt)


def test_run_training_requires_explicit_model_id() -> None:
    args = Namespace(run_training=True, max_steps=3, debug_sample=32, model_id=None)
    try:
        run_dapt.validate_smoke_training_args(args)
        raise AssertionError("expected missing --model-id to fail")
    except ValueError as exc:
        assert "--model-id is required" in str(exc)
