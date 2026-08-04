from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    model_id: str
    tokenizer_id: str
    family: str
    max_length: int


DEFAULT_BACKBONES = {
    "mamba2_130m": BackboneSpec("mamba2_130m", "benchang1110/mamba2-130m-hf", "state-spaces/mamba-130m-hf", "mamba", 4096),
    "clinical_mamba_130m": BackboneSpec("clinical_mamba_130m", "whaleloops/clinicalmamba-130m-hf", "whaleloops/clinicalmamba-130m-hf", "mamba", 4096),
    "mamba3_siso_187m": BackboneSpec("mamba3_siso_187m", "state-spaces/mamba3-siso-187m", "meta-llama/Llama-3.1-8B", "mamba3", 2048),
    "bioclinicalbert": BackboneSpec("bioclinicalbert", "emilyalsentzer/Bio_ClinicalBERT", "emilyalsentzer/Bio_ClinicalBERT", "encoder", 512),
    "longformer": BackboneSpec("longformer", "allenai/longformer-base-4096", "allenai/longformer-base-4096", "encoder", 4096),
    "modernbert_base": BackboneSpec("modernbert_base", "answerdotai/ModernBERT-base", "answerdotai/ModernBERT-base", "encoder", 8192),
}


def get_backbone_spec(name: str) -> BackboneSpec:
    if name not in DEFAULT_BACKBONES:
        raise KeyError(f"Unknown backbone {name}; available={sorted(DEFAULT_BACKBONES)}")
    return DEFAULT_BACKBONES[name]


def ensure_mamba3_token_embeddings(model: Any, vocab_size: int) -> None:
    """Resize official mamba-ssm Mamba3 token embeddings and LM head.

    Mamba3-SISO checkpoints are loaded through `MambaLMHeadModel`, which does
    not always expose the Hugging Face `resize_token_embeddings()` API. The
    input embedding and output LM head therefore have to be resized together.
    """
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "embedding") or not hasattr(model, "lm_head"):
        raise ValueError("Mamba3 embedding resize requires backbone.embedding and lm_head")
    old_embedding = model.backbone.embedding
    old_head = model.lm_head
    current = int(old_embedding.num_embeddings)
    if current == int(vocab_size) and int(getattr(model.config, "vocab_size", vocab_size)) == int(vocab_size):
        return
    old_weight = old_embedding.weight.detach().clone()
    old_head_weight = old_head.weight.detach().clone()
    hidden = int(old_weight.shape[1])
    new_embedding = torch.nn.Embedding(
        int(vocab_size),
        hidden,
        device=old_weight.device,
        dtype=old_weight.dtype,
    )
    new_head = torch.nn.Linear(
        hidden,
        int(vocab_size),
        bias=False,
        device=old_head_weight.device,
        dtype=old_head_weight.dtype,
    )
    rows_to_copy = min(int(vocab_size), old_weight.shape[0], old_head_weight.shape[0])
    with torch.no_grad():
        init_vector = old_weight[:rows_to_copy].mean(dim=0)
        new_embedding.weight[:rows_to_copy].copy_(old_weight[:rows_to_copy])
        new_head.weight[:rows_to_copy].copy_(old_head_weight[:rows_to_copy])
        if int(vocab_size) > rows_to_copy:
            new_embedding.weight[rows_to_copy:].copy_(init_vector)
            new_head.weight[rows_to_copy:].copy_(init_vector.to(new_head.weight.dtype))
    model.backbone.embedding = new_embedding
    model.lm_head = new_head
    model.config.vocab_size = int(vocab_size)


def ensure_token_embeddings_for_tokenizer(model: Any, tokenizer: Any) -> None:
    """Ensure a model can index every token in the tokenizer vocabulary."""
    if model is None:
        return
    vocab_size = int(len(tokenizer))
    if hasattr(model, "backbone") and hasattr(model.backbone, "embedding") and hasattr(model, "lm_head"):
        ensure_mamba3_token_embeddings(model, vocab_size)
        return
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None and int(embeddings.num_embeddings) == vocab_size:
            if hasattr(model, "config"):
                model.config.vocab_size = vocab_size
            return
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(vocab_size)
        if hasattr(model, "config"):
            model.config.vocab_size = vocab_size
        return
    raise ValueError("Model does not expose a supported token embedding resize path")


def load_tokenizer_and_model(
    spec: BackboneSpec,
    torch_dtype: Any | None = None,
    device_map: str | None = None,
    device: str | None = None,
):
    """Load tokenizer/model with hidden-state access enabled.

    Hugging Face Mamba/Mamba2 causal models use AutoModelForCausalLM.
    Official Mamba3 checkpoints require the mamba-ssm MambaLMHeadModel
    loader, which does not go through Transformers AutoModel.
    """
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer_id)
    kwargs: dict[str, Any] = {"output_hidden_states": True}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        kwargs["device_map"] = device_map
    if spec.family == "mamba3":
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        mamba_kwargs: dict[str, Any] = {}
        if torch_dtype is not None:
            mamba_kwargs["dtype"] = torch_dtype
        if device is not None:
            mamba_kwargs["device"] = device
        model = MambaLMHeadModel.from_pretrained(spec.model_id, **mamba_kwargs)
    elif spec.family == "mamba":
        model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
    else:
        model = AutoModel.from_pretrained(spec.model_id, **kwargs)
    ensure_token_embeddings_for_tokenizer(model, tokenizer)
    return tokenizer, model


def last_hidden_state_from_outputs(outputs):
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if hasattr(outputs, "logits") and outputs.logits is not None:
        raise TypeError("Backbone output only contains logits; call the backbone module for hidden states")
    if isinstance(outputs, tuple):
        return outputs[0]
    raise TypeError("Could not extract hidden states from backbone output")
