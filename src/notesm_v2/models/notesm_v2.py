from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import warnings

import torch
from torch import nn

from notesm_v2.data.section_parser import (
    ParsedSection,
    SectionParseResult,
    fallback_required_for_model,
    merge_duplicate_sections_for_model,
)
from notesm_v2.models.backbone_registry import last_hidden_state_from_outputs


SECTION_SPECIAL_TOKENS = ["<SEC_SUM>", "<CHUNK_SUM>", "<NOTE_SUM>", "<SEC_REPEAT>"]


@dataclass(frozen=True)
class SectionModelInput:
    sections: list[ParsedSection]
    use_chunk_fallback: bool


@dataclass(frozen=True)
class NoteSSMV2Config:
    hidden_size: int
    num_section_types: int = 64
    dropout: float = 0.1
    freeze_backbone: bool = True
    train_new_token_embeddings: bool = False
    head_type: str = "linear"
    mlp_hidden_size: int | None = None
    branch_mode: str = "shared"
    use_section_type_embedding: bool = True
    section_pooling: str = "attention"
    fallback_pooling: str = "attention"


@dataclass(frozen=True)
class NoteSSMV2HybridConfig:
    hidden_size: int
    note_hidden_size: int | None = None
    num_section_types: int = 64
    num_section_order_buckets: int = 8
    num_section_length_buckets: int = 8
    num_occurrence_buckets: int = 8
    num_chunk_order_buckets: int = 8
    num_chunk_length_buckets: int = 8
    dropout: float = 0.1
    head_type: str = "mlp"
    mlp_hidden_size: int | None = None
    hybrid_mode: str = "hard_routed_mlp"
    section_pooling: str = "attention"
    fallback_pooling: str = "attention"
    use_section_type_embedding: bool = True
    use_fallback_indicator: bool = True
    set_transformer_depth: int = 1
    set_transformer_heads: int = 4
    set_transformer_ff_mult: int = 2
    residual_alpha_init: float = 0.1


def build_section_model_input(
    parse_result: SectionParseResult,
    min_sections: int = 2,
    coverage_threshold: float = 0.50,
    section_max_tokens: int = 2048,
    token_count: Callable[[str], int] | None = None,
) -> SectionModelInput:
    merged_sections = merge_duplicate_sections_for_model(parse_result.sections)
    return SectionModelInput(
        sections=merged_sections,
        use_chunk_fallback=fallback_required_for_model(
            parse_result=parse_result,
            min_sections=min_sections,
            coverage_threshold=coverage_threshold,
            section_max_tokens=section_max_tokens,
            token_count=token_count if token_count is not None else lambda text: len(text.split()),
        ),
    )


def add_section_special_tokens(tokenizer, model: nn.Module | None = None, train_new_token_embeddings: bool = False) -> int:
    """Add section tokens and initialize new rows from the old embedding mean.

    This helper does not implement row-level gradient masking. During DAPT the
    full input embedding matrix should remain trainable; during downstream the
    backbone/embeddings are frozen by the model freeze policy.
    """
    warnings.warn(
        "add_section_special_tokens() is deprecated; use notesm_v2.models.special_tokens.add_project_special_tokens() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    existing = set(tokenizer.get_vocab().keys())
    to_add = [token for token in SECTION_SPECIAL_TOKENS if token not in existing]
    if not to_add:
        return 0
    added = tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    if model is not None and hasattr(model, "resize_token_embeddings"):
        old_embeddings = model.get_input_embeddings().weight.data.clone()
        model.resize_token_embeddings(len(tokenizer))
        new_embeddings = model.get_input_embeddings().weight
        with torch.no_grad():
            mean_vector = old_embeddings.mean(dim=0)
            new_embeddings[-added:].copy_(mean_vector)
    return int(added)


class SectionAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.scorer = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, section_embeddings: torch.Tensor, section_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.scorer(section_embeddings).squeeze(-1)
        scores = scores.masked_fill(section_mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(section_mask.bool(), weights, torch.zeros_like(weights))
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = weights / denom
        pooled = torch.bmm(weights.unsqueeze(1), section_embeddings).squeeze(1)
        return pooled, weights


class MaskedMeanPooling(nn.Module):
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = mask.to(dtype=embeddings.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        weights = weights / denom
        pooled = torch.bmm(weights.unsqueeze(1), embeddings).squeeze(1)
        return pooled, weights


class MaskedMaxPooling(nn.Module):
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = mask.bool().unsqueeze(-1)
        masked = embeddings.masked_fill(~valid, torch.finfo(embeddings.dtype).min)
        pooled = masked.max(dim=1).values
        pooled = torch.where(mask.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))
        weights = torch.zeros(mask.shape, dtype=embeddings.dtype, device=embeddings.device)
        if mask.any():
            scores = embeddings.norm(dim=-1).masked_fill(mask == 0, torch.finfo(embeddings.dtype).min)
            positions = scores.argmax(dim=-1)
            weights.scatter_(1, positions.unsqueeze(1), 1.0)
            weights = torch.where(mask.bool(), weights, torch.zeros_like(weights))
        return pooled, weights


def make_pooler(pooling: str, hidden_size: int, dropout: float) -> nn.Module:
    if pooling == "attention":
        return SectionAttentionPooling(hidden_size, dropout=dropout)
    if pooling == "mean":
        return MaskedMeanPooling()
    if pooling == "max":
        return MaskedMaxPooling()
    raise ValueError(f"Unknown pooling mode: {pooling}")


class HierarchicalChunkToNotePooler(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1, pooling: str = "attention") -> None:
        super().__init__()
        self.pooler = make_pooler(pooling, hidden_size, dropout=dropout)

    def forward(self, chunk_embeddings: torch.Tensor, chunk_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pooler(chunk_embeddings, chunk_mask)


class SetTransformerPooling(nn.Module):
    """Small masked set encoder with a learned pooling token."""

    def __init__(self, hidden_size: int, depth: int = 1, heads: int = 4, ff_mult: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError("hidden_size must be divisible by set transformer heads")
        self.pool_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=max(hidden_size, hidden_size * ff_mult),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = embeddings.size(0)
        pool = self.pool_token.expand(batch, -1, -1)
        sequence = torch.cat([pool, embeddings], dim=1)
        pool_mask = torch.ones((batch, 1), dtype=mask.dtype, device=mask.device)
        full_mask = torch.cat([pool_mask, mask], dim=1).bool()
        encoded = self.encoder(sequence, src_key_padding_mask=~full_mask)
        weights = torch.zeros(mask.shape, dtype=embeddings.dtype, device=embeddings.device)
        return encoded[:, 0], weights


def extract_marker_hidden_states(hidden_states: torch.Tensor, input_ids: torch.Tensor, marker_token_id: int) -> torch.Tensor:
    """Extract the last marker-token hidden state for each sequence."""
    marker_mask = input_ids.eq(marker_token_id)
    if not marker_mask.any(dim=1).all():
        raise ValueError("Every sequence must contain the requested marker token")
    positions = marker_mask.long().cumsum(dim=1).argmax(dim=1)
    return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), positions]


class NoteSSMV2SectionAwareClassifier(nn.Module):
    """Section-aware summary-token pooling head for LAMP-R."""

    def __init__(self, config: NoteSSMV2Config, backbone: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self._validate_config(config)
        self.backbone = backbone
        if self.backbone is not None and config.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.section_type_embedding = (
            nn.Embedding(config.num_section_types, config.hidden_size) if config.use_section_type_embedding else None
        )
        self.section_attention = make_pooler(config.section_pooling, config.hidden_size, dropout=config.dropout)
        self.chunk_pooler = HierarchicalChunkToNotePooler(
            config.hidden_size,
            dropout=config.dropout,
            pooling=config.fallback_pooling,
        )
        self.dropout = nn.Dropout(config.dropout)
        head_input_size = config.hidden_size + (1 if config.branch_mode == "branch_indicator" else 0)
        if config.branch_mode == "branch_specific_heads":
            self.section_classifier = self._make_risk_head(head_input_size)
            self.fallback_classifier = self._make_risk_head(head_input_size)
            self.classifier = None
        else:
            self.classifier = self._make_risk_head(head_input_size)
            self.section_classifier = None
            self.fallback_classifier = None
        self.branch_bias = nn.Embedding(2, 1) if config.branch_mode == "branch_specific_bias" else None

    @staticmethod
    def _validate_config(config: NoteSSMV2Config) -> None:
        if config.head_type not in {"linear", "mlp"}:
            raise ValueError(f"Unknown NoteSSM-v2 head_type: {config.head_type}")
        if config.branch_mode not in {"shared", "branch_indicator", "branch_specific_bias", "branch_specific_heads"}:
            raise ValueError(f"Unknown NoteSSM-v2 branch_mode: {config.branch_mode}")
        for label, pooling in {"section_pooling": config.section_pooling, "fallback_pooling": config.fallback_pooling}.items():
            if pooling not in {"attention", "mean", "max"}:
                raise ValueError(f"Unknown {label}: {pooling}")

    def _make_risk_head(self, input_size: int) -> nn.Module:
        if self.config.head_type == "linear":
            return nn.Linear(input_size, 1)
        hidden = self.config.mlp_hidden_size or self.config.hidden_size
        return nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.backbone is None:
            raise ValueError("backbone is required for token-level encoding")
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return last_hidden_state_from_outputs(outputs)

    def forward(
        self,
        section_embeddings: torch.Tensor,
        section_mask: torch.Tensor,
        section_type_ids: torch.Tensor | None = None,
        fallback_chunk_embeddings: torch.Tensor | None = None,
        fallback_chunk_mask: torch.Tensor | None = None,
        use_chunk_fallback: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.section_type_embedding is not None and section_type_ids is not None:
            section_embeddings = section_embeddings + self.section_type_embedding(section_type_ids)
        section_pooled, section_weights = self.section_attention(section_embeddings, section_mask)
        pooled = section_pooled
        fallback_weights = None
        branch_ids = torch.zeros(section_embeddings.size(0), dtype=torch.long, device=section_embeddings.device)
        if fallback_chunk_embeddings is not None and fallback_chunk_mask is not None and use_chunk_fallback is not None:
            fallback_pooled, fallback_weights = self.chunk_pooler(fallback_chunk_embeddings, fallback_chunk_mask)
            branch_ids = use_chunk_fallback.to(device=section_embeddings.device).long().clamp(min=0, max=1)
            selector = branch_ids.bool().unsqueeze(-1)
            pooled = torch.where(selector, fallback_pooled, section_pooled)
        if self.config.branch_mode == "branch_indicator":
            indicator = branch_ids.to(dtype=pooled.dtype).unsqueeze(-1)
            pooled_for_head = torch.cat([pooled, indicator], dim=-1)
        else:
            pooled_for_head = pooled
        head_input = self.dropout(pooled_for_head)
        if self.config.branch_mode == "branch_specific_heads":
            section_logits = self.section_classifier(head_input).squeeze(-1)
            fallback_logits = self.fallback_classifier(head_input).squeeze(-1)
            logits = torch.where(branch_ids.bool(), fallback_logits, section_logits)
        else:
            logits = self.classifier(head_input).squeeze(-1)
            if self.branch_bias is not None:
                logits = logits + self.branch_bias(branch_ids).squeeze(-1)
        out = {"logits": logits, "pooled": pooled, "section_attention": section_weights}
        if fallback_weights is not None:
            out["fallback_attention"] = fallback_weights
        out["branch_ids"] = branch_ids
        return out


class NoteSSMV2HybridClassifier(nn.Module):
    """Hybrid cached-embedding classifier for section/chunk and note vectors."""

    V2_1_MODES = {"three_view_concat", "three_view_residual", "set_transformer_residual"}

    def __init__(self, config: NoteSSMV2HybridConfig) -> None:
        super().__init__()
        self.config = config
        self._validate_config(config)
        self.note_hidden_size = config.note_hidden_size or config.hidden_size
        self.section_type_embedding = (
            nn.Embedding(config.num_section_types, config.hidden_size) if config.use_section_type_embedding else None
        )
        self.section_pooler = make_pooler(config.section_pooling, config.hidden_size, dropout=config.dropout)
        self.chunk_pooler = HierarchicalChunkToNotePooler(
            config.hidden_size,
            dropout=config.dropout,
            pooling=config.fallback_pooling,
        )
        self.note_projection = (
            nn.Linear(self.note_hidden_size, config.hidden_size) if self.note_hidden_size != config.hidden_size else nn.Identity()
        )
        self.note_view_projection = nn.Linear(self.note_hidden_size, config.hidden_size)
        self.section_view_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.chunk_view_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.note_view_norm = nn.LayerNorm(config.hidden_size)
        self.section_view_norm = nn.LayerNorm(config.hidden_size)
        self.chunk_view_norm = nn.LayerNorm(config.hidden_size)
        self.section_order_embedding = nn.Embedding(config.num_section_order_buckets, config.hidden_size)
        self.section_length_embedding = nn.Embedding(config.num_section_length_buckets, config.hidden_size)
        self.occurrence_embedding = nn.Embedding(config.num_occurrence_buckets, config.hidden_size)
        self.repeated_section_embedding = nn.Embedding(2, config.hidden_size)
        self.chunk_order_embedding = nn.Embedding(config.num_chunk_order_buckets, config.hidden_size)
        self.chunk_length_embedding = nn.Embedding(config.num_chunk_length_buckets, config.hidden_size)
        self.missing_chunk_embedding = nn.Parameter(torch.zeros(config.hidden_size))
        if self.note_hidden_size == config.hidden_size:
            with torch.no_grad():
                self.note_view_projection.weight.copy_(torch.eye(config.hidden_size))
                self.note_view_projection.bias.zero_()
        with torch.no_grad():
            self.section_view_projection.weight.copy_(torch.eye(config.hidden_size))
            self.section_view_projection.bias.zero_()
            self.chunk_view_projection.weight.copy_(torch.eye(config.hidden_size))
            self.chunk_view_projection.bias.zero_()
        if config.hybrid_mode == "set_transformer_residual":
            self.v2_1_section_pooler = SetTransformerPooling(
                config.hidden_size,
                depth=config.set_transformer_depth,
                heads=config.set_transformer_heads,
                ff_mult=config.set_transformer_ff_mult,
                dropout=config.dropout,
            )
            self.v2_1_chunk_pooler = SetTransformerPooling(
                config.hidden_size,
                depth=config.set_transformer_depth,
                heads=config.set_transformer_heads,
                ff_mult=config.set_transformer_ff_mult,
                dropout=config.dropout,
            )
        else:
            self.v2_1_section_pooler = make_pooler(config.section_pooling, config.hidden_size, dropout=config.dropout)
            self.v2_1_chunk_pooler = make_pooler(config.fallback_pooling, config.hidden_size, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        if config.hybrid_mode == "concat_mlp":
            input_size = config.hidden_size * 2 + (1 if config.use_fallback_indicator else 0)
            self.gate = None
            self.classifier = self._make_risk_head(input_size)
            self.fallback_classifier = None
            self.note_classifier = None
        elif config.hybrid_mode == "learned_gate_mlp":
            gate_input_size = config.hidden_size * 2 + (1 if config.use_fallback_indicator else 0)
            hidden = config.mlp_hidden_size or config.hidden_size
            self.gate = nn.Sequential(
                nn.Linear(gate_input_size, hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, 1),
            )
            self.classifier = self._make_risk_head(config.hidden_size)
            self.fallback_classifier = None
            self.note_classifier = None
        elif config.hybrid_mode == "branch_specific_heads_hybrid":
            self.gate = None
            self.classifier = None
            self.fallback_classifier = self._make_risk_head(config.hidden_size)
            self.note_classifier = self._make_risk_head(config.hidden_size)
        elif config.hybrid_mode == "three_view_concat":
            self.gate = None
            self.classifier = self._make_risk_head(config.hidden_size * 3 + 3)
            self.fallback_classifier = None
            self.note_classifier = self._make_risk_head(config.hidden_size)
            self.section_aux_classifier = self._make_risk_head(config.hidden_size)
            self.chunk_aux_classifier = self._make_risk_head(config.hidden_size)
            self.residual_head = None
            self.residual_alpha = None
        elif config.hybrid_mode in {"three_view_residual", "set_transformer_residual"}:
            self.gate = None
            self.classifier = None
            self.fallback_classifier = None
            self.note_classifier = self._make_risk_head(config.hidden_size)
            self.section_aux_classifier = self._make_risk_head(config.hidden_size)
            self.chunk_aux_classifier = self._make_risk_head(config.hidden_size)
            self.residual_head = self._make_risk_head(config.hidden_size * 5 + 3)
            self.residual_alpha = nn.Parameter(torch.tensor(float(config.residual_alpha_init)))
        else:
            self.gate = None
            self.classifier = self._make_risk_head(config.hidden_size)
            self.fallback_classifier = None
            self.note_classifier = None
            self.section_aux_classifier = None
            self.chunk_aux_classifier = None
            self.residual_head = None
            self.residual_alpha = None
        if config.hybrid_mode in {"concat_mlp", "learned_gate_mlp", "branch_specific_heads_hybrid"}:
            self.section_aux_classifier = None
            self.chunk_aux_classifier = None
            self.residual_head = None
            self.residual_alpha = None

    @staticmethod
    def _validate_config(config: NoteSSMV2HybridConfig) -> None:
        if config.head_type not in {"linear", "mlp"}:
            raise ValueError(f"Unknown hybrid head_type: {config.head_type}")
        if config.hybrid_mode not in {
            "hard_routed_mlp",
            "concat_mlp",
            "learned_gate_mlp",
            "branch_specific_heads_hybrid",
            "three_view_concat",
            "three_view_residual",
            "set_transformer_residual",
        }:
            raise ValueError(f"Unknown hybrid_mode: {config.hybrid_mode}")
        for label, pooling in {"section_pooling": config.section_pooling, "fallback_pooling": config.fallback_pooling}.items():
            if pooling not in {"attention", "mean", "max"}:
                raise ValueError(f"Unknown {label}: {pooling}")
        if config.set_transformer_depth < 1:
            raise ValueError("set_transformer_depth must be >= 1")
        if config.set_transformer_heads < 1:
            raise ValueError("set_transformer_heads must be >= 1")

    def _make_risk_head(self, input_size: int) -> nn.Module:
        if self.config.head_type == "linear":
            return nn.Linear(input_size, 1)
        hidden = self.config.mlp_hidden_size or self.config.hidden_size
        return nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _zeros_like_ids(reference: torch.Tensor, num_rows: int) -> torch.Tensor:
        return torch.zeros(reference.size(0), num_rows, dtype=torch.long, device=reference.device)

    def _encode_three_views(
        self,
        *,
        section_embeddings: torch.Tensor,
        section_mask: torch.Tensor,
        note_embedding: torch.Tensor,
        section_type_ids: torch.Tensor | None,
        fallback_chunk_embeddings: torch.Tensor | None,
        fallback_chunk_mask: torch.Tensor | None,
        use_chunk_fallback: torch.Tensor,
        section_order_bucket_ids: torch.Tensor | None,
        section_length_bucket_ids: torch.Tensor | None,
        occurrence_bucket_ids: torch.Tensor | None,
        repeated_section_ids: torch.Tensor | None,
        chunk_order_bucket_ids: torch.Tensor | None,
        chunk_length_bucket_ids: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        batch = section_embeddings.size(0)
        device = section_embeddings.device
        section_available = section_mask.any(dim=1).to(dtype=section_embeddings.dtype)
        if fallback_chunk_embeddings is None or fallback_chunk_mask is None:
            fallback_chunk_embeddings = torch.zeros(
                (batch, 1, self.config.hidden_size),
                dtype=section_embeddings.dtype,
                device=device,
            )
            fallback_chunk_mask = torch.zeros((batch, 1), dtype=torch.long, device=device)
        chunk_available = fallback_chunk_mask.any(dim=1).to(dtype=section_embeddings.dtype)
        branch_ids = use_chunk_fallback.to(device=device).long().clamp(min=0, max=1)

        h_sec = self.section_view_projection(section_embeddings)
        if self.section_type_embedding is not None and section_type_ids is not None:
            h_sec = h_sec + self.section_type_embedding(section_type_ids.to(device=device))
        if section_order_bucket_ids is None:
            section_order_bucket_ids = self._zeros_like_ids(section_embeddings, section_embeddings.size(1))
        if section_length_bucket_ids is None:
            section_length_bucket_ids = self._zeros_like_ids(section_embeddings, section_embeddings.size(1))
        if occurrence_bucket_ids is None:
            occurrence_bucket_ids = self._zeros_like_ids(section_embeddings, section_embeddings.size(1))
        if repeated_section_ids is None:
            repeated_section_ids = self._zeros_like_ids(section_embeddings, section_embeddings.size(1))
        h_sec = (
            h_sec
            + self.section_order_embedding(section_order_bucket_ids.to(device=device).clamp(0, self.config.num_section_order_buckets - 1))
            + self.section_length_embedding(section_length_bucket_ids.to(device=device).clamp(0, self.config.num_section_length_buckets - 1))
            + self.occurrence_embedding(occurrence_bucket_ids.to(device=device).clamp(0, self.config.num_occurrence_buckets - 1))
            + self.repeated_section_embedding(repeated_section_ids.to(device=device).clamp(0, 1))
        )
        h_sec = self.section_view_norm(h_sec)
        z_sec, section_weights = self.v2_1_section_pooler(h_sec, section_mask)

        h_chunk = self.chunk_view_projection(fallback_chunk_embeddings)
        if chunk_order_bucket_ids is None:
            chunk_order_bucket_ids = self._zeros_like_ids(fallback_chunk_embeddings, fallback_chunk_embeddings.size(1))
        if chunk_length_bucket_ids is None:
            chunk_length_bucket_ids = self._zeros_like_ids(fallback_chunk_embeddings, fallback_chunk_embeddings.size(1))
        h_chunk = (
            h_chunk
            + self.chunk_order_embedding(chunk_order_bucket_ids.to(device=device).clamp(0, self.config.num_chunk_order_buckets - 1))
            + self.chunk_length_embedding(chunk_length_bucket_ids.to(device=device).clamp(0, self.config.num_chunk_length_buckets - 1))
        )
        h_chunk = self.chunk_view_norm(h_chunk)
        z_chunk, chunk_weights = self.v2_1_chunk_pooler(h_chunk, fallback_chunk_mask)
        missing = self.missing_chunk_embedding.unsqueeze(0).expand(batch, -1)
        z_chunk = torch.where(chunk_available.bool().unsqueeze(-1), z_chunk, missing)

        z_note = self.note_view_norm(self.note_view_projection(note_embedding))
        indicators = torch.stack(
            [
                branch_ids.to(dtype=section_embeddings.dtype),
                section_available,
                chunk_available,
            ],
            dim=-1,
        )
        return {
            "z_note": z_note,
            "z_sec": z_sec,
            "z_chunk": z_chunk,
            "section_attention": section_weights,
            "fallback_attention": chunk_weights,
            "branch_ids": branch_ids,
            "section_available": section_available,
            "chunk_available": chunk_available,
            "indicators": indicators,
        }

    def load_note_head_checkpoint(self, checkpoint_path: str | bytes | "PathLike[str]") -> None:
        if self.note_classifier is None:
            raise ValueError("Current hybrid mode does not expose a note_classifier")
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        try:
            self.note_classifier.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Could not load note head checkpoint into residual note head: {checkpoint_path}") from exc

    def _section_or_chunk_representation(
        self,
        section_embeddings: torch.Tensor,
        section_mask: torch.Tensor,
        section_type_ids: torch.Tensor | None,
        fallback_chunk_embeddings: torch.Tensor | None,
        fallback_chunk_mask: torch.Tensor | None,
        use_chunk_fallback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if self.section_type_embedding is not None and section_type_ids is not None:
            section_embeddings = section_embeddings + self.section_type_embedding(section_type_ids)
        section_pooled, section_weights = self.section_pooler(section_embeddings, section_mask)
        branch_ids = use_chunk_fallback.to(device=section_embeddings.device).long().clamp(min=0, max=1)
        fallback_weights = None
        pooled = section_pooled
        if fallback_chunk_embeddings is not None and fallback_chunk_mask is not None:
            fallback_pooled, fallback_weights = self.chunk_pooler(fallback_chunk_embeddings, fallback_chunk_mask)
            pooled = torch.where(branch_ids.bool().unsqueeze(-1), fallback_pooled, section_pooled)
        elif branch_ids.any():
            raise ValueError("Fallback examples require fallback_chunk_embeddings and fallback_chunk_mask")
        return pooled, section_weights, fallback_weights, branch_ids

    def forward(
        self,
        section_embeddings: torch.Tensor,
        section_mask: torch.Tensor,
        note_embedding: torch.Tensor,
        section_type_ids: torch.Tensor | None = None,
        fallback_chunk_embeddings: torch.Tensor | None = None,
        fallback_chunk_mask: torch.Tensor | None = None,
        use_chunk_fallback: torch.Tensor | None = None,
        section_order_bucket_ids: torch.Tensor | None = None,
        section_length_bucket_ids: torch.Tensor | None = None,
        occurrence_bucket_ids: torch.Tensor | None = None,
        repeated_section_ids: torch.Tensor | None = None,
        chunk_order_bucket_ids: torch.Tensor | None = None,
        chunk_length_bucket_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if use_chunk_fallback is None:
            use_chunk_fallback = torch.zeros(section_embeddings.size(0), dtype=torch.long, device=section_embeddings.device)
        if self.config.hybrid_mode in self.V2_1_MODES:
            views = self._encode_three_views(
                section_embeddings=section_embeddings,
                section_mask=section_mask,
                note_embedding=note_embedding,
                section_type_ids=section_type_ids,
                fallback_chunk_embeddings=fallback_chunk_embeddings,
                fallback_chunk_mask=fallback_chunk_mask,
                use_chunk_fallback=use_chunk_fallback,
                section_order_bucket_ids=section_order_bucket_ids,
                section_length_bucket_ids=section_length_bucket_ids,
                occurrence_bucket_ids=occurrence_bucket_ids,
                repeated_section_ids=repeated_section_ids,
                chunk_order_bucket_ids=chunk_order_bucket_ids,
                chunk_length_bucket_ids=chunk_length_bucket_ids,
            )
            if self.config.hybrid_mode == "three_view_concat":
                representation = torch.cat([views["z_note"], views["z_sec"], views["z_chunk"], views["indicators"]], dim=-1)
                logits = self.classifier(self.dropout(representation)).squeeze(-1)
                note_logits = self.note_classifier(self.dropout(views["z_note"])).squeeze(-1) if self.note_classifier is not None else None
            else:
                note_logits = self.note_classifier(self.dropout(views["z_note"])).squeeze(-1)
                residual_input = torch.cat(
                    [
                        views["z_note"],
                        views["z_sec"],
                        views["z_chunk"],
                        views["z_note"] * views["z_sec"],
                        views["z_note"] * views["z_chunk"],
                        views["indicators"],
                    ],
                    dim=-1,
                )
                residual = self.residual_head(self.dropout(residual_input)).squeeze(-1)
                logits = note_logits + self.residual_alpha * residual
                representation = residual_input
            out = {
                "logits": logits,
                "pooled": representation,
                "section_attention": views["section_attention"],
                "fallback_attention": views["fallback_attention"],
                "branch_ids": views["branch_ids"],
                "z_note": views["z_note"],
                "z_sec": views["z_sec"],
                "z_chunk": views["z_chunk"],
                "section_available": views["section_available"],
                "chunk_available": views["chunk_available"],
                "section_logits": self.section_aux_classifier(self.dropout(views["z_sec"])).squeeze(-1),
                "chunk_logits": self.chunk_aux_classifier(self.dropout(views["z_chunk"])).squeeze(-1),
            }
            if note_logits is not None:
                out["note_logits"] = note_logits
            if self.residual_alpha is not None:
                out["residual_alpha"] = self.residual_alpha.detach()
            return out
        section_or_chunk, section_weights, fallback_weights, branch_ids = self._section_or_chunk_representation(
            section_embeddings=section_embeddings,
            section_mask=section_mask,
            section_type_ids=section_type_ids,
            fallback_chunk_embeddings=fallback_chunk_embeddings,
            fallback_chunk_mask=fallback_chunk_mask,
            use_chunk_fallback=use_chunk_fallback,
        )
        note_pooled = self.note_projection(note_embedding)
        indicator = branch_ids.to(dtype=section_or_chunk.dtype).unsqueeze(-1)
        gate_values = None
        if self.config.hybrid_mode == "concat_mlp":
            pieces = [section_or_chunk, note_pooled]
            if self.config.use_fallback_indicator:
                pieces.append(indicator)
            representation = torch.cat(pieces, dim=-1)
            logits = self.classifier(self.dropout(representation)).squeeze(-1)
        elif self.config.hybrid_mode == "learned_gate_mlp":
            pieces = [section_or_chunk, note_pooled]
            if self.config.use_fallback_indicator:
                pieces.append(indicator)
            gate_values = torch.sigmoid(self.gate(torch.cat(pieces, dim=-1)))
            representation = gate_values * section_or_chunk + (1.0 - gate_values) * note_pooled
            logits = self.classifier(self.dropout(representation)).squeeze(-1)
        elif self.config.hybrid_mode == "branch_specific_heads_hybrid":
            fallback_logits = self.fallback_classifier(self.dropout(section_or_chunk)).squeeze(-1)
            note_logits = self.note_classifier(self.dropout(note_pooled)).squeeze(-1)
            representation = torch.where(branch_ids.bool().unsqueeze(-1), section_or_chunk, note_pooled)
            logits = torch.where(branch_ids.bool(), fallback_logits, note_logits)
        else:
            representation = torch.where(branch_ids.bool().unsqueeze(-1), section_or_chunk, note_pooled)
            logits = self.classifier(self.dropout(representation)).squeeze(-1)
        out = {
            "logits": logits,
            "pooled": representation,
            "section_attention": section_weights,
            "branch_ids": branch_ids,
        }
        if fallback_weights is not None:
            out["fallback_attention"] = fallback_weights
        if gate_values is not None:
            out["gate"] = gate_values.squeeze(-1)
        return out
