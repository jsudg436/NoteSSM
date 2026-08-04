from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from notesm_v2.models.backbone_registry import last_hidden_state_from_outputs


@dataclass(frozen=True)
class NoteSSMV1Config:
    hidden_size: int
    dropout: float = 0.1
    freeze_backbone: bool = True


def last_valid_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.long().sum(dim=1).clamp(min=1)
    indices = lengths - 1
    return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), indices]


class NoteSSMV1LastTokenClassifier(nn.Module):
    """Frozen-backbone last-valid-token LAMP-R baseline."""

    def __init__(self, config: NoteSSMV1Config, backbone: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone
        if self.backbone is not None and config.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.backbone is None:
            raise ValueError("backbone is required when hidden_states are not provided")
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return last_hidden_state_from_outputs(outputs)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if hidden_states is None:
            if input_ids is None or attention_mask is None:
                raise ValueError("provide hidden_states or input_ids + attention_mask")
            hidden_states = self.encode(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)
        pooled = last_valid_token_pool(hidden_states, attention_mask)
        logits = self.classifier(self.dropout(pooled)).squeeze(-1)
        return {"logits": logits, "pooled": pooled}
