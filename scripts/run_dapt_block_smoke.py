from __future__ import annotations

import argparse
import math
import sys
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.data.dapt_shard_dataset import DaptTokenShardDataset, dapt_token_collate_fn  # noqa: E402
from notesm_v2.models.backbone_registry import BackboneSpec, DEFAULT_BACKBONES, load_tokenizer_and_model  # noqa: E402
from notesm_v2.models.special_tokens import add_project_special_tokens  # noqa: E402


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def validate_smoke_args(max_steps: int, model_id: str | None) -> None:
    if max_steps > 10:
        raise ValueError("DAPT block smoke is gated to --max-steps <= 10")
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if not model_id:
        raise ValueError("--model-id is required for DAPT block smoke training")


def prepare_tokenizer(tokenizer_id: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    return tokenizer


def load_smoke_tokenizer_and_model(model_id: str, tokenizer_id: str):
    registered = next(
        (
            spec
            for spec in DEFAULT_BACKBONES.values()
            if spec.model_id == model_id and spec.tokenizer_id == tokenizer_id
        ),
        None,
    )
    spec = registered or BackboneSpec("custom_causal_lm", model_id, tokenizer_id, "mamba", 0)
    return load_tokenizer_and_model(spec)


def causal_lm_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run gated max-steps DAPT smoke training from full-block token shards.")
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-id", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--output-dir", default="checkpoints/dapt_block_smoke_mamba2")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save-smoke-state", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    validate_smoke_args(args.max_steps, args.model_id)

    torch.manual_seed(args.seed)
    tokenizer, model = load_smoke_tokenizer_and_model(args.model_id, args.tokenizer_id)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    add_project_special_tokens(tokenizer, model=model, init="mean")
    if hasattr(model, "get_input_embeddings") and model.get_input_embeddings().num_embeddings != len(tokenizer):
        old_embeddings = model.get_input_embeddings().weight.detach().clone()
        old_size = old_embeddings.shape[0]
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            model.get_input_embeddings().weight[old_size:].copy_(old_embeddings.mean(dim=0))
    if args.gradient_checkpointing:
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.train()

    dataset = DaptTokenShardDataset(resolve_path(args.shard_dir), shuffle=True, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(dapt_token_collate_fn, pad_token_id=int(tokenizer.pad_token_id)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses: list[float] = []
    for step, batch in enumerate(loader):
        if step >= args.max_steps:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        if args.bf16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = causal_lm_loss(model, batch)
        else:
            loss = causal_lm_loss(model, batch)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite DAPT smoke loss at step {step}: {loss}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    if not losses:
        raise ValueError("No DAPT smoke training steps completed")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_saved = False
    if args.save_smoke_state:
        torch.save({"losses": losses, "steps_completed": len(losses), "model_id": args.model_id}, output_dir / "smoke_state.pt")
        checkpoint_saved = True
    summary = pd.DataFrame(
        [
            {
                "model_id": args.model_id,
                "tokenizer_id": args.tokenizer_id,
                "shard_dir": str(resolve_path(args.shard_dir)),
                "steps_completed": int(len(losses)),
                "batch_size": int(args.batch_size),
                "max_steps": int(args.max_steps),
                "final_loss": float(losses[-1]),
                "losses": " ".join(f"{loss:.6f}" for loss in losses),
                "full_dapt_training_launched": False,
                "smoke_training_only": True,
                "used_full_block_dataset": True,
                "checkpoint_saved": bool(checkpoint_saved),
                "gradient_checkpointing": bool(args.gradient_checkpointing),
                "bf16": bool(args.bf16),
                "finite_losses": bool(all(math.isfinite(loss) for loss in losses)),
            }
        ]
    )
    summary.to_csv(output_dir / "dapt_block_smoke_training_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("DAPT block smoke completed. Full DAPT training remains gated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
