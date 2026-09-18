import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .embeddings import TokenEmbedding
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .initialization import init_weights


class MyAI(nn.Module):
    """Konoros-Sec base LM: decoder-only Transformer (RMSNorm + RoPE + SwiGLU + GQA + KV-cache)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self._grad_ckpt = False
        self.tok_emb = TokenEmbedding(cfg)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.wte.weight
        self.apply(lambda m: init_weights(m, n_layers=cfg.n_layers))

    def gradient_checkpointing_enable(self):
        self._grad_ckpt = True

    def forward(self, ids: torch.Tensor, past=None, start_pos: int = 0,
                use_cache: bool = False, return_hidden: bool = False):
        # ids: (B, T)
        x = self.tok_emb(ids)
        presents = [] if use_cache else None
        aux = [] if return_hidden else None
        for i, blk in enumerate(self.layers):
            pkv = past[i] if past is not None else None
            if self._grad_ckpt and self.training and pkv is None and not return_hidden:
                import torch.utils.checkpoint as ckpt
                x, present = ckpt.checkpoint(blk, x, None, start_pos, False, use_reentrant=False)
            else:
                if return_hidden:
                    x, h, present = blk(x, past_kv=pkv, start_pos=start_pos, return_hidden=True)
                    aux.append(h)
                else:
                    x, present = blk(x, past_kv=pkv, start_pos=start_pos)
            if use_cache:
                presents.append(present)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if use_cache and return_hidden:
            return logits, presents, aux
        if use_cache:
            return logits, presents
        if return_hidden:
            return logits, aux
        return logits

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float):
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / max(temperature, 1e-6)
        if top_k and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits[logits < v[:, [-1]]] = -float("inf")
        if top_p and 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            mask = cum - probs > top_p
            sorted_logits[mask] = -float("inf")
            logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0,
                 eos_id: int | None = None) -> torch.Tensor:
        """Autoregressive generation with KV-cache (prefill once, decode 1 token/step)."""
        self.eval()
        out = ids
        logits, past = self.forward(ids[:, -self.cfg.context_length:], use_cache=True)
        pos = min(out.shape[1], self.cfg.context_length)
        for _ in range(max_new_tokens):
            nxt = self._sample(logits[:, -1, :], temperature, top_k, top_p)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
            if pos >= self.cfg.context_length:
                # Context full: re-prefill from sliding window (correct RoPE positions)
                logits, past = self.forward(out[:, -self.cfg.context_length:], use_cache=True)
                pos = self.cfg.context_length
            else:
                logits, past = self.forward(nxt, past=past, start_pos=pos, use_cache=True)
                pos += 1
        return out
