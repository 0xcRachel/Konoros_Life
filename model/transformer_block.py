import torch
import torch.nn as nn
from .attention import CausalSelfAttention
from .mlp import SwiGLU
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm block: RMSNorm -> Attn -> Residual -> RMSNorm -> SwiGLU -> Residual."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, past_kv=None, start_pos: int = 0, return_hidden: bool = False):
        # Hook return_hidden để sau này gắn critic / process-reward cho reasoning
        h = self.ln1(x)
        a, present = self.attn(h, past_kv=past_kv, start_pos=start_pos)
        x = x + a
        h2 = self.ln2(x)
        x = x + self.mlp(h2)
        if return_hidden:
            return x, {"attn_in": h, "mlp_in": h2}, present
        return x, present
