import torch.nn as nn
from .config import ModelConfig


class TokenEmbedding(nn.Module):
    """Token embedding + dropout. Reasoning-ready: no positional embedding (RoPE does it)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, ids):
        return self.drop(self.wte(ids))
