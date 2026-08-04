from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_dapt_from_shards", ROOT / "scripts" / "train_dapt_from_shards.py")
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


class DummyTokenizer:
    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class FakeConfig:
    use_cache = True


class FakeCheckpointModel:
    def __init__(self) -> None:
        self.config = FakeConfig()
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True


class FakeTrainModeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eval()


def make_args(**overrides):
    defaults = {
        "max_steps": None,
        "allow_full_epoch": False,
        "confirm_full_dapt": "",
        "checkpoint_format": "state_dict",
        "model_id": "debug-model",
        "tokenizer_id": "debug-tokenizer",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_full_epoch_requires_explicit_confirmation() -> None:
    try:
        trainer.validate_full_epoch_gate(make_args())
        raise AssertionError("expected missing max_steps/full confirmation to fail")
    except ValueError as exc:
        assert "Full DAPT is gated" in str(exc)

    try:
        trainer.validate_full_epoch_gate(make_args(allow_full_epoch=True, confirm_full_dapt="WRONG"))
        raise AssertionError("expected wrong confirmation to fail")
    except ValueError as exc:
        assert trainer.CONFIRM_FULL_DAPT in str(exc)


def test_max_steps_pilot_allowed() -> None:
    trainer.validate_full_epoch_gate(make_args(max_steps=3))


def test_effective_batch_and_step_estimation() -> None:
    assert trainer.estimate_steps_per_epoch(total_blocks=100, world_size=2, per_device_batch_size=2, gradient_accumulation_steps=5) == 5


def test_scheduler_total_steps_can_differ_from_target_steps() -> None:
    assert trainer.resolve_scheduler_total_steps(None, target_steps=500) == 500
    assert trainer.resolve_scheduler_total_steps(14764, target_steps=2000) == 14764

    try:
        trainer.resolve_scheduler_total_steps(0, target_steps=2000)
        raise AssertionError("expected non-positive scheduler total steps to fail")
    except ValueError as exc:
        assert "--scheduler-total-steps" in str(exc)


def test_scheduler_uses_explicit_scheduler_total_steps() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = trainer.build_scheduler(optimizer, total_steps=100, warmup_ratio=0.0)

    optimizer.step()
    scheduler.step()

    assert abs(optimizer.param_groups[0]["lr"] - 0.99) < 1e-12


def test_scheduler_total_steps_resume_reuses_checkpoint_value_when_omitted() -> None:
    state = {"args": {"scheduler_total_steps": 14764}}

    resolved, checkpoint_value, resume_match = trainer.resolve_scheduler_total_steps_for_resume(
        current_scheduler_total_steps=None,
        target_steps=2000,
        checkpoint_state=state,
    )

    assert resolved == 14764
    assert checkpoint_value == 14764
    assert resume_match is True


def test_scheduler_total_steps_resume_mismatch_fails() -> None:
    state = {"args": {"scheduler_total_steps": 14764}}

    try:
        trainer.resolve_scheduler_total_steps_for_resume(
            current_scheduler_total_steps=2000,
            target_steps=2000,
            checkpoint_state=state,
        )
        raise AssertionError("expected scheduler_total_steps mismatch to fail")
    except ValueError as exc:
        assert "scheduler_total_steps mismatch" in str(exc)


def test_full_epoch_residual_accumulation_produces_ceiled_step_count() -> None:
    assert trainer.estimate_steps_per_epoch(total_blocks=10, world_size=1, per_device_batch_size=1, gradient_accumulation_steps=8) == 2
    assert trainer.should_step_residual_accumulation(max_steps=None, accum_counter=2, global_step=1, target_steps=2) is True
    assert trainer.should_step_residual_accumulation(max_steps=2000, accum_counter=2, global_step=1, target_steps=2) is False


def test_final_checkpoint_saved_when_final_step_not_divisible_by_save_steps() -> None:
    assert trainer.should_save_final_checkpoint(global_step=14763, save_steps=1000) is True
    assert trainer.should_save_final_checkpoint(global_step=14000, save_steps=1000) is False
    assert trainer.should_save_final_checkpoint(global_step=14763, save_steps=0) is True


def test_checkpoint_roundtrip_state_dict() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        with torch.no_grad():
            model.weight.fill_(3.0)
            model.bias.fill_(1.0)

        checkpoint = trainer.save_checkpoint(
            output_dir=root,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            tokenizer=DummyTokenizer(),
            global_step=7,
            gradient_accumulation_step=0,
            args={"checkpoint_format": "state_dict"},
            checkpoint_format="state_dict",
            rank=0,
        )

        restored = torch.nn.Linear(2, 1)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lr_lambda=lambda _: 1.0)
        state = trainer.load_checkpoint(checkpoint, model=restored, optimizer=restored_optimizer, scheduler=restored_scheduler)

        assert state["global_step"] == 7
        assert torch.allclose(restored.weight, torch.full_like(restored.weight, 3.0))
        assert torch.allclose(restored.bias, torch.full_like(restored.bias, 1.0))


def test_rank0_only_save_logic() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

        checkpoint = trainer.save_checkpoint(
            output_dir=root,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            tokenizer=DummyTokenizer(),
            global_step=1,
            gradient_accumulation_step=0,
            args={},
            checkpoint_format="state_dict",
            rank=1,
        )

        assert checkpoint is None
        assert not list(root.glob("checkpoint-step-*"))


def test_gradient_checkpointing_sets_use_cache_false() -> None:
    model = FakeCheckpointModel()

    trainer.apply_gradient_checkpointing(model, True)

    assert model.config.use_cache is False
    assert model.gradient_checkpointing_enabled is True


def test_model_train_mode_is_explicitly_enabled() -> None:
    model = FakeTrainModeModel()

    assert model.training is False
    returned = trainer.set_model_train_mode(model)

    assert returned is model
    assert model.training is True


def test_token_count_reduction_is_identity_without_ddp() -> None:
    assert trainer.reduce_token_count(123, torch.device("cpu"), distributed=False) == 123


def test_loss_reduction_is_identity_without_ddp() -> None:
    assert trainer.reduce_scalar_mean(1.25, torch.device("cpu"), distributed=False) == 1.25


def test_ddp_no_sync_only_used_on_non_final_accumulation_steps() -> None:
    assert trainer.should_use_ddp_no_sync(distributed=False, accum_counter=0, gradient_accumulation_steps=8) is False
    assert trainer.should_use_ddp_no_sync(distributed=True, accum_counter=0, gradient_accumulation_steps=8) is True
    assert trainer.should_use_ddp_no_sync(distributed=True, accum_counter=6, gradient_accumulation_steps=8) is True
    assert trainer.should_use_ddp_no_sync(distributed=True, accum_counter=7, gradient_accumulation_steps=8) is False
    assert trainer.should_use_ddp_no_sync(distributed=True, accum_counter=0, gradient_accumulation_steps=1) is False


def test_training_summary_has_required_fields() -> None:
    summary = trainer.make_training_summary(model_id="m", tokenizer_id="t")

    assert list(summary.columns) == trainer.SUMMARY_FIELDS
    for field in trainer.SUMMARY_FIELDS:
        assert field in summary.columns

    for field in [
        "target_steps",
        "scheduler_total_steps",
        "scheduler_total_steps_from_checkpoint",
        "scheduler_total_steps_resume_match",
        "lr_at_final_step",
        "source_total_blocks",
        "trainable_blocks",
        "min_train_block_tokens",
        "skipped_short_train_blocks",
        "max_grad_norm",
        "residual_accumulation_step_used",
        "residual_microbatches",
        "ddp_enabled",
        "local_tokens_seen",
        "global_tokens_seen",
        "bf16_full_params",
        "use_autocast_bf16",
    ]:
        assert field in summary.columns
