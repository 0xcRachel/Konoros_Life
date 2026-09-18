import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RoPE


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, Hkv, T, D) -> (B, Hkv*n_rep, T, D). No-op when n_rep == 1 (MHA)."""
    if n_rep == 1:
        return x
    B, Hkv, T, D = x.shape
    x = x[:, :, None, :, :].expand(B, Hkv, n_rep, T, D)
    return x.reshape(B, Hkv * n_rep, T, D)


class CausalSelfAttention(nn.Module):
    """MHA when n_kv_heads == n_heads, GQA/MQA otherwise. KV-cache ready."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = getattr(cfg, "n_kv_heads", 0) or cfg.n_heads
        assert cfg.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RoPE(self.head_dim, cfg.context_length, cfg.rope_theta)
        self.dropout = cfg.dropout

    def forward(self, x, past_kv=None, start_pos: int = 0):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, start_pos=start_pos)
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present = (k.detach() if not self.training else k, v.detach() if not self.training else v)
        y = F.scaled_dot_product_attention(
            q, repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep),
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), present
