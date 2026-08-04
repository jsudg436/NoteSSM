from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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


def validate_smoke_training_args(args: argparse.Namespace) -> None:
    if not args.run_training:
        return
    if args.max_steps > 10:
        raise ValueError("--run-training is currently gated to smoke runs with --max-steps <= 10")
    if args.debug_sample is None or args.debug_sample > 128:
        raise ValueError("--run-training smoke mode requires --debug-sample <= 128")
    if not args.model_id:
        raise ValueError("--model-id is required with --run-training; do not rely on tokenizer-id as a model checkpoint")


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


def causal_lm_loss(model, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    try:
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if hasattr(output, "loss") and output.loss is not None:
            return output.loss
        if isinstance(output, dict) and output.get("loss") is not None:
            return output["loss"]
    except TypeError:
        output = model(input_ids=input_ids)
    logits = output.logits if hasattr(output, "logits") else output[0]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenizer-aware strict train-only DAPT runner.")
    parser.add_argument("--benchmark-root", default="results/lampr_full_v0_2")
    parser.add_argument("--model-id", default=None, help="Causal LM checkpoint for smoke DAPT. Required with --run-training.")
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--output-dir", default="checkpoints/notesm_v2_dapt_lampr_v0_2")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--debug-sample", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-training", action="store_true", help="Actually launch DAPT training. Omit for tokenizer/packing dry run.")
    parser.add_argument(
        "--save-smoke-state-dict",
        action="store_true",
        help="Optionally save a raw PyTorch state_dict for smoke runs. Avoids transformers save_pretrained/DeepSpeed imports.",
    )
    args = parser.parse_args()
    validate_smoke_training_args(args)

    from transformers import AutoTokenizer

    root = resolve_path(args.benchmark_root)
    sequences_path = root / "dapt_train_sequences.parquet"
    if not sequences_path.exists():
        raise FileNotFoundError(f"Missing {sequences_path}")
    sequences = pd.read_parquet(sequences_path)
    if args.debug_sample is not None:
        sequences = sequences.head(args.debug_sample).copy()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    special_token_result = add_project_special_tokens(tokenizer)
    added_tokens = list(special_token_result["tokens_added"])
    block_count = 0
    token_count = 0
    for text in sequences["sequence_text"].astype(str):
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        token_count += len(ids)
        block_count += max((len(ids) + args.max_length - 1) // args.max_length, 1)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "tokenizer_id": args.tokenizer_id,
                "patients": len(sequences),
                "total_tokens": token_count,
                "estimated_blocks": block_count,
                "max_length": args.max_length,
                "debug_sample": args.debug_sample if args.debug_sample is not None else 0,
                "training_launched": bool(args.run_training),
                "training_mode": "dry_run_token_count" if not args.run_training else "smoke_truncated_patient_sequences",
                "full_dapt_block_packing_implemented": False,
                "smoke_training_uses_truncated_single_block_per_patient": bool(args.run_training),
                "full_epoch_launched": False,
                "num_special_tokens_added": len(added_tokens),
                "special_tokens_added": " ".join(added_tokens),
            }
        ]
    )
    summary.to_csv(output_dir / "dapt_tokenizer_dry_run_summary.csv", index=False)
    print(summary.to_string(index=False))
    if not args.run_training:
        print("Dry run complete. Re-run with --run-training only after debug counts are reviewed.")
        return 0
    # Smoke mode intentionally truncates one patient sequence into one training
    # block. Full DAPT must later split long patient sequences into multiple
    # causal LM blocks without crossing patient boundaries.
    model_id = args.model_id
    tokenizer, model = load_smoke_tokenizer_and_model(model_id, args.tokenizer_id)
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
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    blocks: list[list[int]] = []
    for text in sequences["sequence_text"].astype(str):
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=args.max_length)["input_ids"]
        if not ids:
            continue
        blocks.append(ids)
        if len(blocks) >= args.max_steps * args.batch_size:
            break
    if not blocks:
        raise ValueError("No tokenized DAPT blocks available for smoke training")
    losses: list[float] = []
    pad_id = int(tokenizer.pad_token_id)
    for step in range(args.max_steps):
        batch = blocks[step * args.batch_size : (step + 1) * args.batch_size]
        if not batch:
            break
        max_len = max(len(row) for row in batch)
        input_ids = torch.tensor([row + [pad_id] * (max_len - len(row)) for row in batch], dtype=torch.long, device=device)
        attention_mask = torch.tensor([[1] * len(row) + [0] * (max_len - len(row)) for row in batch], dtype=torch.long, device=device)
        labels = input_ids.masked_fill(attention_mask == 0, -100)
        optimizer.zero_grad()
        loss = causal_lm_loss(model, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    smoke_summary = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "tokenizer_id": args.tokenizer_id,
                "steps_completed": len(losses),
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "final_loss": losses[-1] if losses else None,
                "training_mode": "smoke_max_10_steps",
                "full_dapt_block_packing_implemented": False,
                "smoke_training_uses_truncated_single_block_per_patient": True,
                "full_epoch_launched": False,
                "smoke_state_dict_saved": bool(args.save_smoke_state_dict),
                "transformers_save_pretrained_used": False,
            }
        ]
    )
    smoke_summary.to_csv(output_dir / "dapt_smoke_training_summary.csv", index=False)
    if args.save_smoke_state_dict:
        state_dir = output_dir / "smoke_state_dict"
        state_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), state_dir / "pytorch_model_state_dict.pt")
        tokenizer.save_pretrained(state_dir)
    print(smoke_summary.to_string(index=False))
    print("Smoke DAPT completed. Full 1-epoch DAPT remains gated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
