import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, D)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        self._cached_len = seq_len

    def _maybe_extend(self, seq_len: int, device):
        if seq_len > self._cached_len:
            self.inv_freq = self.inv_freq.to(device)
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        elif self.cos.device != device:
            self.cos = self.cos.to(device)
            self.sin = self.sin.to(device)

    def forward(self, q, k, start_pos: int = 0):
        # q, k: (B, H, T, D). start_pos>0 when decoding with KV-cache.
        T = q.shape[-2]
        self._maybe_extend(start_pos + T, q.device)
        cos = self.cos[start_pos:start_pos + T].unsqueeze(0).unsqueeze(0).to(q.dtype)
        sin = self.sin[start_pos:start_pos + T].unsqueeze(0).unsqueeze(0).to(q.dtype)
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rotary(self, x, cos, sin):
        # NOTE: must NOT be named `_apply` — nn.Module._apply() is called by .to()/.cuda()
        return x * cos + self._rotate_half(x) * sin
