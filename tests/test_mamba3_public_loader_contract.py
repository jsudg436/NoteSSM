from __future__ import annotations

from types import SimpleNamespace

import torch

from notesm_v2.models.backbone_registry import ensure_mamba3_token_embeddings, last_hidden_state_from_outputs


class DummyBackbone(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)


class DummyMamba3(torch.nn.Module):
    def __init__(self, vocab_size: int = 4, hidden_size: int = 3) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.backbone = DummyBackbone(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)


def test_mamba3_resize_updates_embedding_head_and_config() -> None:
    model = DummyMamba3(vocab_size=4, hidden_size=3)
    with torch.no_grad():
        model.backbone.embedding.weight.copy_(torch.arange(12, dtype=torch.float32).view(4, 3))
        model.lm_head.weight.copy_(torch.arange(12, dtype=torch.float32).view(4, 3) + 100)

    ensure_mamba3_token_embeddings(model, 6)

    assert model.config.vocab_size == 6
    assert model.backbone.embedding.num_embeddings == 6
    assert model.lm_head.out_features == 6
    torch.testing.assert_close(model.backbone.embedding.weight[:4], torch.arange(12, dtype=torch.float32).view(4, 3))
    torch.testing.assert_close(model.lm_head.weight[:4], torch.arange(12, dtype=torch.float32).view(4, 3) + 100)
    expected_init = torch.arange(12, dtype=torch.float32).view(4, 3).mean(dim=0)
    torch.testing.assert_close(model.backbone.embedding.weight[4], expected_init)
    torch.testing.assert_close(model.backbone.embedding.weight[5], expected_init)


def test_last_hidden_state_accepts_tensor_outputs() -> None:
    hidden = torch.randn(2, 5, 3)
    assert last_hidden_state_from_outputs(hidden) is hidden
