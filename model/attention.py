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

        q = (
            self.q_proj(x)
            .view(B, T, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        k = (
            self.k_proj(x)
            .view(B, T, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        v = (
            self.v_proj(x)
            .view(B, T, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply RoPE using the absolute position of the current tokens.
        q, k = self.rope(q, k, start_pos=start_pos)

        # Append new K/V to the cache.
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        present = (
            k.detach() if not self.training else k,
            v.detach() if not self.training else v,
        )

        # GQA: expand KV heads to match Q heads.
        k_attn = repeat_kv(k, self.n_rep)
        v_attn = repeat_kv(v, self.n_rep)

        # ---------------------------------------------------------
        # Causal mask
        # ---------------------------------------------------------
        #
        # Full forward:
        #   Q = T
        #   K = T
        #   start_pos = 0
        #
        # Cached decode:
        #   Q = T
        #   K = past_len + T
        #   start_pos = past_len
        #
        # In the cached case, `is_causal=True` is not sufficient
        # because SDPA only sees the local Q/K dimensions and does
        # not know the absolute position of Q.
        # ---------------------------------------------------------

        total_len = k_attn.size(2)

        if past_kv is None and start_pos == 0:
            # Normal full-sequence causal attention.
            y = F.scaled_dot_product_attention(
                q,
                k_attn,
                v_attn,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Explicit causal mask for cached decoding.
            query_positions = (
                torch.arange(
                    start_pos,
                    start_pos + T,
                    device=x.device,
                )
                .unsqueeze(1)
            )

            key_positions = (
                torch.arange(
                    total_len,
                    device=x.device,
                )
                .unsqueeze(0)
            )

            causal_mask = key_positions <= query_positions

            # SDPA expects:
            # (B, H, Tq, Tk) or broadcastable shape.
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            y = F.scaled_dot_product_attention(
                q,
                k_attn,
                v_attn,
                attn_mask=causal_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        return self.proj(y), present
