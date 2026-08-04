from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.data.dapt_shard_dataset import DaptTokenShardDataset, dapt_token_collate_fn  # noqa: E402
from notesm_v2.models.backbone_registry import BackboneSpec, DEFAULT_BACKBONES, load_tokenizer_and_model  # noqa: E402
from notesm_v2.models.special_tokens import add_project_special_tokens  # noqa: E402


CONFIRM_FULL_DAPT = "I_UNDERSTAND_FULL_DAPT"
SUMMARY_FIELDS = [
    "model_id",
    "tokenizer_id",
    "shard_dir",
    "max_steps",
    "target_steps",
    "scheduler_total_steps",
    "scheduler_total_steps_from_checkpoint",
    "scheduler_total_steps_resume_match",
    "steps_completed",
    "world_size",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_blocks",
    "bf16",
    "bf16_full_params",
    "use_autocast_bf16",
    "gradient_checkpointing",
    "ddp_enabled",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "max_grad_norm",
    "final_loss",
    "lr_at_final_step",
    "finite_losses",
    "total_real_tokens_seen",
    "local_tokens_seen",
    "global_tokens_seen",
    "estimated_total_blocks",
    "source_total_blocks",
    "trainable_blocks",
    "min_train_block_tokens",
    "skipped_short_train_blocks",
    "estimated_steps_per_epoch",
    "residual_accumulation_step_used",
    "residual_microbatches",
    "full_dapt_training_launched",
    "pilot_training_only",
    "checkpoint_saved",
    "resume_from_checkpoint",
    "output_dir",
]


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def validate_full_epoch_gate(args: argparse.Namespace) -> None:
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        return
    if not args.allow_full_epoch:
        raise ValueError("Full DAPT is gated: pass --max-steps for a pilot run or explicitly pass --allow-full-epoch with confirmation")
    if args.confirm_full_dapt != CONFIRM_FULL_DAPT:
        raise ValueError(f"Full DAPT requires --confirm-full-dapt {CONFIRM_FULL_DAPT}")


def validate_resume_policy(args: argparse.Namespace) -> None:
    if args.resume_from_checkpoint and args.max_steps is None:
        raise ValueError(
            "Full-epoch DAPT resume is disabled in the public trainer because the dataloader cursor is not restored. "
            "Restart full DAPT from scratch for paper-grade runs, or use --max-steps only for smoke/debug resume checks."
        )


def estimate_steps_per_epoch(total_blocks: int, world_size: int, per_device_batch_size: int, gradient_accumulation_steps: int) -> int:
    if total_blocks <= 0:
        return 0
    effective_micro_batch = max(world_size, 1) * max(per_device_batch_size, 1)
    dataloader_steps = math.ceil(total_blocks / effective_micro_batch)
    return int(math.ceil(dataloader_steps / max(gradient_accumulation_steps, 1)))


def is_distributed_env() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> dict[str, int | bool | str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return {"distributed": distributed, "world_size": world_size, "rank": rank, "local_rank": local_rank, "backend": backend}


def cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return int(rank) == 0


def apply_gradient_checkpointing(model: Any, enabled: bool) -> None:
    if not enabled:
        return
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


def set_model_train_mode(model: torch.nn.Module) -> torch.nn.Module:
    model.train()
    return model


def load_dapt_tokenizer_and_model(args: argparse.Namespace) -> tuple[Any, torch.nn.Module]:
    registered = next(
        (
            spec
            for spec in DEFAULT_BACKBONES.values()
            if spec.model_id == args.model_id and spec.tokenizer_id == args.tokenizer_id
        ),
        None,
    )
    spec = registered or BackboneSpec(
        name="custom_causal_lm",
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
        family="mamba",
        max_length=0,
    )
    return load_tokenizer_and_model(spec)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def causal_lm_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute next-token loss for HF causal LMs and official mamba-ssm models."""
    try:
        output = model(**batch)
        if hasattr(output, "loss") and output.loss is not None:
            return output.loss
        if isinstance(output, dict) and output.get("loss") is not None:
            return output["loss"]
    except TypeError:
        output = model(input_ids=batch["input_ids"])
    logits = output.logits if hasattr(output, "logits") else output[0]
    labels = batch["labels"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def reduce_token_count(token_count: int, device: torch.device, distributed: bool) -> int:
    if not distributed:
        return int(token_count)
    tensor = torch.tensor(int(token_count), dtype=torch.long, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return int(tensor.detach().cpu().item())


def reduce_scalar_mean(value: float, device: torch.device, distributed: bool) -> float:
    if not distributed:
        return float(value)
    tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / float(torch.distributed.get_world_size())
    return float(tensor.detach().cpu().item())


def should_use_ddp_no_sync(distributed: bool, accum_counter: int, gradient_accumulation_steps: int) -> bool:
    if not distributed:
        return False
    final_accumulation_step = (int(accum_counter) + 1) % max(int(gradient_accumulation_steps), 1) == 0
    return not final_accumulation_step


def resolve_scheduler_total_steps(scheduler_total_steps: int | None, target_steps: int) -> int:
    if scheduler_total_steps is None:
        return int(target_steps)
    if scheduler_total_steps <= 0:
        raise ValueError("--scheduler-total-steps must be positive")
    return int(scheduler_total_steps)


def checkpoint_args_from_state(state: dict[str, Any] | None) -> dict[str, Any]:
    if not state:
        return {}
    checkpoint_args = state.get("args") or {}
    if isinstance(checkpoint_args, argparse.Namespace):
        return vars(checkpoint_args)
    if isinstance(checkpoint_args, dict):
        return checkpoint_args
    return {}


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def checkpoint_scheduler_total_steps(state: dict[str, Any] | None) -> int | None:
    return optional_int(checkpoint_args_from_state(state).get("scheduler_total_steps"))


def resolve_scheduler_total_steps_for_resume(
    *,
    current_scheduler_total_steps: int | None,
    target_steps: int,
    checkpoint_state: dict[str, Any] | None,
) -> tuple[int, int | None, bool | None]:
    checkpoint_total_steps = checkpoint_scheduler_total_steps(checkpoint_state)
    if checkpoint_total_steps is None:
        return resolve_scheduler_total_steps(current_scheduler_total_steps, target_steps), None, None
    if current_scheduler_total_steps is None:
        return checkpoint_total_steps, checkpoint_total_steps, True
    current_total_steps = resolve_scheduler_total_steps(current_scheduler_total_steps, target_steps)
    if current_total_steps != checkpoint_total_steps:
        raise ValueError(
            "Resumed DAPT scheduler_total_steps mismatch: "
            f"checkpoint={checkpoint_total_steps} current={current_total_steps}. "
            "Pass the same --scheduler-total-steps as the original run or omit it to reuse the checkpoint value."
        )
    return current_total_steps, checkpoint_total_steps, True


def should_step_residual_accumulation(
    *,
    max_steps: int | None,
    accum_counter: int,
    global_step: int,
    target_steps: int,
) -> bool:
    return max_steps is None and accum_counter > 0 and global_step < target_steps


def should_save_final_checkpoint(global_step: int, save_steps: int) -> bool:
    return int(global_step) > 0 and (int(save_steps) <= 0 or int(global_step) % int(save_steps) != 0)


def scale_gradients(model: torch.nn.Module, scale: float) -> None:
    if scale == 1.0:
        return
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.detach().mul_(scale)


def build_scheduler(optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_ratio: float):
    warmup_steps = int(max(total_steps, 0) * max(warmup_ratio, 0.0))

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        remaining = max(total_steps - step, 0)
        denom = max(total_steps - warmup_steps, 1)
        return max(float(remaining) / float(denom), 0.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    tokenizer: Any,
    global_step: int,
    gradient_accumulation_step: int,
    args: argparse.Namespace | dict[str, Any],
    checkpoint_format: str,
    rank: int,
) -> Path | None:
    if not is_rank0(rank):
        return None
    checkpoint_dir = output_dir / f"checkpoint-step-{int(global_step)}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    arg_dict = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    base_state = {
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "global_step": int(global_step),
        "gradient_accumulation_step": int(gradient_accumulation_step),
        "args": arg_dict,
        "checkpoint_format": checkpoint_format,
    }
    model_to_save = unwrap_model(model)
    if checkpoint_format == "save_pretrained":
        model_to_save.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(base_state, checkpoint_dir / "trainer_state.pt")
    elif checkpoint_format == "state_dict":
        state = dict(base_state)
        state["model_state"] = model_to_save.state_dict()
        torch.save(state, checkpoint_dir / "pytorch_training_state.pt")
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    else:
        raise ValueError(f"Unknown checkpoint format: {checkpoint_format}")
    return checkpoint_dir


def load_checkpoint_state(checkpoint_dir: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    state_path = checkpoint_dir / "pytorch_training_state.pt"
    trainer_state_path = checkpoint_dir / "trainer_state.pt"
    if state_path.exists():
        try:
            return torch.load(state_path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(state_path, map_location=map_location)
    elif trainer_state_path.exists():
        try:
            return torch.load(trainer_state_path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(trainer_state_path, map_location=map_location)
    raise FileNotFoundError(f"No supported DAPT checkpoint state found in {checkpoint_dir}")


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if state is None:
        state = load_checkpoint_state(checkpoint_dir, map_location=map_location)
    if state.get("model_state") is not None:
        unwrap_model(model).load_state_dict(state["model_state"])
    if optimizer is not None and state.get("optimizer_state") is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])
    return state


def make_training_summary(**kwargs: Any) -> pd.DataFrame:
    row = {field: kwargs.get(field) for field in SUMMARY_FIELDS}
    return pd.DataFrame([row], columns=SUMMARY_FIELDS)


def gpu_peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gated full-block DAPT trainer over train-only token shards.")
    parser.add_argument("--shard-dir", default="results/dapt_token_shards/lampr_v0_2_mamba_len4096")
    parser.add_argument("--model-id", default="benchang1110/mamba2-130m-hf")
    parser.add_argument("--tokenizer-id", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--output-dir", default="checkpoints/dapt_pilot_mamba2_len4096")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--scheduler-total-steps", type=int, default=None)
    parser.add_argument("--allow-full-epoch", action="store_true")
    parser.add_argument("--confirm-full-dapt", default="")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--bf16-full-params", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--checkpoint-format", choices=["state_dict", "save_pretrained"], default="state_dict")
    parser.add_argument("--min-train-block-tokens", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_full_epoch_gate(args)
    validate_resume_policy(args)

    dist = setup_distributed()
    rank = int(dist["rank"])
    world_size = int(dist["world_size"])
    local_rank = int(dist["local_rank"])
    rank0 = is_rank0(rank)
    output_dir = resolve_path(args.output_dir)
    if rank0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    torch.manual_seed(args.seed + rank)
    tokenizer, model = load_dapt_tokenizer_and_model(args)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    add_project_special_tokens(tokenizer, model=model, init="mean")
    if hasattr(model, "get_input_embeddings") and model.get_input_embeddings().num_embeddings != len(tokenizer):
        raise ValueError("Model embedding size does not match tokenizer size after adding project special tokens")
    apply_gradient_checkpointing(model, args.gradient_checkpointing)

    if args.device != "auto":
        device = torch.device(args.device)
    else:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.bf16_full_params and device.type == "cuda":
        model.to(dtype=torch.bfloat16)
    if bool(dist["distributed"]):
        ddp_kwargs = {"device_ids": [local_rank]} if device.type == "cuda" else {}
        model = DistributedDataParallel(model, **ddp_kwargs)
    set_model_train_mode(model)
    resume_state = None
    if args.resume_from_checkpoint:
        resume_state = load_checkpoint_state(args.resume_from_checkpoint, map_location=device)

    if args.min_train_block_tokens < 2:
        raise ValueError("--min-train-block-tokens must be >= 2 for causal LM training")
    dataset = DaptTokenShardDataset(
        resolve_path(args.shard_dir),
        shuffle=not bool(dist["distributed"]),
        seed=args.seed,
        min_block_tokens=args.min_train_block_tokens,
    )
    total_blocks = len(dataset)
    estimated_steps_per_epoch = estimate_steps_per_epoch(total_blocks, world_size, args.per_device_batch_size, args.gradient_accumulation_steps)
    target_steps = args.max_steps if args.max_steps is not None else estimated_steps_per_epoch
    scheduler_total_steps, scheduler_total_steps_from_checkpoint, scheduler_total_steps_resume_match = resolve_scheduler_total_steps_for_resume(
        current_scheduler_total_steps=args.scheduler_total_steps,
        target_steps=int(target_steps),
        checkpoint_state=resume_state,
    )
    args.scheduler_total_steps = int(scheduler_total_steps)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if bool(dist["distributed"]) else None
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        collate_fn=partial(dapt_token_collate_fn, pad_token_id=int(tokenizer.pad_token_id)),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, total_steps=int(scheduler_total_steps), warmup_ratio=args.warmup_ratio)

    global_step = 0
    resume_info = ""
    if args.resume_from_checkpoint:
        state = load_checkpoint(args.resume_from_checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, map_location=device, state=resume_state)
        global_step = int(state.get("global_step", 0))
        resume_info = str(args.resume_from_checkpoint)
        if rank0:
            print(f"Resumed DAPT trainer from {resume_info} at global_step={global_step}")

    losses: list[float] = []
    log_rows: list[dict[str, Any]] = []
    local_tokens_seen = 0
    global_tokens_seen = 0
    checkpoint_saved = False
    residual_accumulation_step_used = False
    residual_microbatches = 0
    start_time = time.time()
    accum_counter = 0
    accum_tokens = 0
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while global_step < int(target_steps):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for microbatch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            tokens_in_microbatch = int(batch["attention_mask"].sum().detach().cpu().item())
            is_last_full_epoch_microbatch = args.max_steps is None and microbatch_index == len(loader) - 1
            use_no_sync = should_use_ddp_no_sync(bool(dist["distributed"]), accum_counter, args.gradient_accumulation_steps)
            use_no_sync = use_no_sync and not is_last_full_epoch_microbatch
            sync_context = model.no_sync() if use_no_sync else nullcontext()
            with sync_context:
                if args.bf16 and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        raw_loss = causal_lm_loss(model, batch)
                else:
                    raw_loss = causal_lm_loss(model, batch)
                if not torch.isfinite(raw_loss):
                    raise ValueError(f"Non-finite DAPT loss before optimizer step {global_step + 1}: {raw_loss}")
                loss = raw_loss / max(args.gradient_accumulation_steps, 1)
                loss.backward()
            accum_counter += 1
            accum_tokens += tokens_in_microbatch
            accum_loss += float(raw_loss.detach().cpu().item())
            if accum_counter % args.gradient_accumulation_steps != 0:
                continue

            grad_norm = None
            if args.max_grad_norm and args.max_grad_norm > 0:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                grad_norm = float(grad_norm_tensor.detach().cpu().item()) if hasattr(grad_norm_tensor, "detach") else float(grad_norm_tensor)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            local_accum_tokens = int(accum_tokens)
            global_accum_tokens = reduce_token_count(local_accum_tokens, device, bool(dist["distributed"]))
            local_tokens_seen += local_accum_tokens
            global_tokens_seen += global_accum_tokens
            local_step_loss = accum_loss / max(args.gradient_accumulation_steps, 1)
            global_step_loss = reduce_scalar_mean(local_step_loss, device, bool(dist["distributed"]))
            losses.append(global_step_loss)
            elapsed = max(time.time() - start_time, 1e-9)
            row = {
                "global_step": int(global_step),
                "optimizer_step": int(global_step),
                "loss": float(global_step_loss),
                "local_step_loss": float(local_step_loss),
                "global_step_loss": float(global_step_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": grad_norm if grad_norm is not None else "",
                "tokens_in_batch": int(global_accum_tokens),
                "local_tokens_in_batch": int(local_accum_tokens),
                "global_tokens_in_batch": int(global_accum_tokens),
                "real_tokens_seen": int(global_tokens_seen),
                "local_tokens_seen": int(local_tokens_seen),
                "global_tokens_seen": int(global_tokens_seen),
                "elapsed_seconds": float(elapsed),
                "steps_per_second": float(global_step / elapsed),
                "tokens_per_second": float(global_tokens_seen / elapsed),
                "gpu_peak_memory_mb": gpu_peak_memory_mb(device),
                "residual_accumulation_step": False,
                "residual_microbatches": 0,
            }
            log_rows.append(row)
            if rank0 and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                print(
                    f"step={global_step} loss={global_step_loss:.6f} lr={optimizer.param_groups[0]['lr']:.6g} "
                    f"tokens={global_tokens_seen} tok/s={row['tokens_per_second']:.2f}"
                )
            if rank0 and args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    tokenizer=tokenizer,
                    global_step=global_step,
                    gradient_accumulation_step=0,
                    args=args,
                    checkpoint_format=args.checkpoint_format,
                    rank=rank,
                )
                checkpoint_saved = True
            accum_tokens = 0
            accum_loss = 0.0
            accum_counter = 0
            if global_step >= int(target_steps):
                break
        if should_step_residual_accumulation(
            max_steps=args.max_steps,
            accum_counter=accum_counter,
            global_step=global_step,
            target_steps=int(target_steps),
        ):
            residual_accumulation_step_used = True
            residual_microbatches = int(accum_counter)
            residual_scale = max(args.gradient_accumulation_steps, 1) / max(residual_microbatches, 1)
            scale_gradients(model, float(residual_scale))
            grad_norm = None
            if args.max_grad_norm and args.max_grad_norm > 0:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                grad_norm = float(grad_norm_tensor.detach().cpu().item()) if hasattr(grad_norm_tensor, "detach") else float(grad_norm_tensor)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            local_accum_tokens = int(accum_tokens)
            global_accum_tokens = reduce_token_count(local_accum_tokens, device, bool(dist["distributed"]))
            local_tokens_seen += local_accum_tokens
            global_tokens_seen += global_accum_tokens
            local_step_loss = accum_loss / max(residual_microbatches, 1)
            global_step_loss = reduce_scalar_mean(local_step_loss, device, bool(dist["distributed"]))
            losses.append(global_step_loss)
            elapsed = max(time.time() - start_time, 1e-9)
            row = {
                "global_step": int(global_step),
                "optimizer_step": int(global_step),
                "loss": float(global_step_loss),
                "local_step_loss": float(local_step_loss),
                "global_step_loss": float(global_step_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": grad_norm if grad_norm is not None else "",
                "tokens_in_batch": int(global_accum_tokens),
                "local_tokens_in_batch": int(local_accum_tokens),
                "global_tokens_in_batch": int(global_accum_tokens),
                "real_tokens_seen": int(global_tokens_seen),
                "local_tokens_seen": int(local_tokens_seen),
                "global_tokens_seen": int(global_tokens_seen),
                "elapsed_seconds": float(elapsed),
                "steps_per_second": float(global_step / elapsed),
                "tokens_per_second": float(global_tokens_seen / elapsed),
                "gpu_peak_memory_mb": gpu_peak_memory_mb(device),
                "residual_accumulation_step": True,
                "residual_microbatches": int(residual_microbatches),
            }
            log_rows.append(row)
            if rank0 and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                print(
                    f"step={global_step} loss={global_step_loss:.6f} lr={optimizer.param_groups[0]['lr']:.6g} "
                    f"tokens={global_tokens_seen} tok/s={row['tokens_per_second']:.2f}"
                )
            if rank0 and args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    tokenizer=tokenizer,
                    global_step=global_step,
                    gradient_accumulation_step=0,
                    args=args,
                    checkpoint_format=args.checkpoint_format,
                    rank=rank,
                )
                checkpoint_saved = True
            accum_tokens = 0
            accum_loss = 0.0
            accum_counter = 0
        epoch += 1
        if args.max_steps is None:
            break

    if rank0 and should_save_final_checkpoint(global_step, args.save_steps):
        save_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            global_step=global_step,
            gradient_accumulation_step=accum_counter,
            args=args,
            checkpoint_format=args.checkpoint_format,
            rank=rank,
        )
        checkpoint_saved = True

    if rank0:
        pd.DataFrame(log_rows).to_csv(output_dir / "dapt_training_log.csv", index=False)
        summary = make_training_summary(
            model_id=args.model_id,
            tokenizer_id=args.tokenizer_id,
            shard_dir=str(resolve_path(args.shard_dir)),
            max_steps=args.max_steps if args.max_steps is not None else "",
            target_steps=int(target_steps),
            scheduler_total_steps=int(scheduler_total_steps),
            scheduler_total_steps_from_checkpoint=scheduler_total_steps_from_checkpoint if scheduler_total_steps_from_checkpoint is not None else "",
            scheduler_total_steps_resume_match=scheduler_total_steps_resume_match if scheduler_total_steps_resume_match is not None else "",
            steps_completed=int(global_step),
            world_size=int(world_size),
            per_device_batch_size=int(args.per_device_batch_size),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            effective_batch_blocks=int(world_size * args.per_device_batch_size * args.gradient_accumulation_steps),
            bf16=bool(args.bf16),
            bf16_full_params=bool(args.bf16_full_params),
            use_autocast_bf16=bool(args.bf16 and device.type == "cuda"),
            gradient_checkpointing=bool(args.gradient_checkpointing),
            ddp_enabled=bool(dist["distributed"]),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            warmup_ratio=float(args.warmup_ratio),
            max_grad_norm=float(args.max_grad_norm),
            final_loss=float(losses[-1]) if losses else "",
            lr_at_final_step=float(optimizer.param_groups[0]["lr"]),
            finite_losses=bool(all(math.isfinite(loss) for loss in losses)),
            total_real_tokens_seen=int(global_tokens_seen),
            local_tokens_seen=int(local_tokens_seen),
            global_tokens_seen=int(global_tokens_seen),
            estimated_total_blocks=int(total_blocks),
            source_total_blocks=int(dataset.source_row_count),
            trainable_blocks=int(total_blocks),
            min_train_block_tokens=int(args.min_train_block_tokens),
            skipped_short_train_blocks=int(dataset.skipped_short_block_count),
            estimated_steps_per_epoch=int(estimated_steps_per_epoch),
            residual_accumulation_step_used=bool(residual_accumulation_step_used),
            residual_microbatches=int(residual_microbatches),
            full_dapt_training_launched=bool(args.max_steps is None),
            pilot_training_only=bool(args.max_steps is not None),
            checkpoint_saved=bool(checkpoint_saved),
            resume_from_checkpoint=resume_info,
            output_dir=str(output_dir),
        )
        summary.to_csv(output_dir / "dapt_training_summary.csv", index=False)
        report = [
            "# DAPT Training Report",
            "",
            "This report intentionally excludes raw clinical note text.",
            "",
            f"- model_id: `{args.model_id}`",
            f"- tokenizer_id: `{args.tokenizer_id}`",
            f"- shard_dir: `{resolve_path(args.shard_dir)}`",
            f"- steps_completed: {global_step}",
            f"- max_steps: {args.max_steps}",
            f"- target_steps: {target_steps}",
            f"- scheduler_total_steps: {scheduler_total_steps}",
            f"- scheduler_total_steps_from_checkpoint: {scheduler_total_steps_from_checkpoint if scheduler_total_steps_from_checkpoint is not None else ''}",
            f"- scheduler_total_steps_resume_match: {scheduler_total_steps_resume_match if scheduler_total_steps_resume_match is not None else ''}",
            f"- estimated_steps_per_epoch: {estimated_steps_per_epoch}",
            f"- world_size: {world_size}",
            f"- ddp_enabled: {bool(dist['distributed'])}",
            f"- effective_batch_blocks: {world_size * args.per_device_batch_size * args.gradient_accumulation_steps}",
            f"- max_grad_norm: {args.max_grad_norm}",
            f"- bf16: {bool(args.bf16)}",
            f"- bf16_full_params: {bool(args.bf16_full_params)}",
            f"- use_autocast_bf16: {bool(args.bf16 and device.type == 'cuda')}",
            f"- local_tokens_seen: {local_tokens_seen}",
            f"- global_tokens_seen: {global_tokens_seen}",
            f"- source_total_blocks: {dataset.source_row_count}",
            f"- trainable_blocks: {total_blocks}",
            f"- min_train_block_tokens: {args.min_train_block_tokens}",
            f"- skipped_short_train_blocks: {dataset.skipped_short_block_count}",
            f"- final_loss: {losses[-1] if losses else ''}",
            f"- lr_at_final_step: {optimizer.param_groups[0]['lr']}",
            f"- finite_losses: {bool(all(math.isfinite(loss) for loss in losses))}",
            f"- residual_accumulation_step_used: {bool(residual_accumulation_step_used)}",
            f"- residual_microbatches: {int(residual_microbatches)}",
            f"- full_dapt_training_launched: {bool(args.max_steps is None)}",
            f"- pilot_training_only: {bool(args.max_steps is not None)}",
            f"- checkpoint_saved: {bool(checkpoint_saved)}",
            f"- resume_from_checkpoint: `{resume_info}`",
            "",
            "Resume restores optimizer/scheduler/global_step but does not exactly restore dataloader position; pilot resume is approximate.",
        ]
        (output_dir / "DAPT_TRAINING_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        print(summary.to_string(index=False))

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
