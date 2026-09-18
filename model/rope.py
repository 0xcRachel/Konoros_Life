import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len, theta=10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, q, k):
        # q, k: (B, H, T, D)
        T = q.shape[-2]
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)
        return self._apply(q, cos, sin), self._apply(k, cos, sin)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply(self, x, cos, sin):
        return x * cos + self._rotate_half(x) * sin