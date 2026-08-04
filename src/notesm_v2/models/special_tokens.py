from __future__ import annotations

import torch

from notesm_v2.data.dapt_packing import SPECIAL_TOKENS
from notesm_v2.models.notesm_v2 import SECTION_SPECIAL_TOKENS


def tokens_to_add(existing_vocab: set[str]) -> list[str]:
    ordered = list(dict.fromkeys([*SPECIAL_TOKENS, *SECTION_SPECIAL_TOKENS]))
    return [token for token in ordered if token not in existing_vocab]


def add_project_special_tokens(tokenizer, model=None, init: str = "mean") -> dict[str, object]:
    """Add all NoteSSM/DAPT project tokens to a tokenizer and optional model.

    This helper does not implement row-level gradient masking. During DAPT the
    resized embedding matrix is trainable; downstream frozen-backbone runs
    should freeze the backbone according to the model config.
    """
    existing = set(tokenizer.get_vocab().keys())
    to_add = tokens_to_add(existing)
    if not to_add:
        if model is not None:
            from notesm_v2.models.backbone_registry import ensure_token_embeddings_for_tokenizer

            ensure_token_embeddings_for_tokenizer(model, tokenizer)
        return {"num_added": 0, "tokens_added": []}
    old_embeddings = None
    if model is not None and hasattr(model, "get_input_embeddings"):
        old_embeddings = model.get_input_embeddings().weight.detach().clone()
    added = tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    if model is not None:
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
            if old_embeddings is not None and added > 0:
                new_embeddings = model.get_input_embeddings().weight
                with torch.no_grad():
                    if init == "eos" and getattr(tokenizer, "eos_token_id", None) is not None:
                        init_vector = old_embeddings[int(tokenizer.eos_token_id)]
                    elif init == "mean":
                        init_vector = old_embeddings.mean(dim=0)
                    else:
                        raise ValueError(f"Unknown special token init strategy: {init}")
                    new_embeddings[-added:].copy_(init_vector)
        else:
            from notesm_v2.models.backbone_registry import ensure_token_embeddings_for_tokenizer

            ensure_token_embeddings_for_tokenizer(model, tokenizer)
    return {"num_added": int(added), "tokens_added": to_add}
