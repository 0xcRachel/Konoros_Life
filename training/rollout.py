"""Rollout helpers: build prompt prefix, generate G completions, parse think/answer."""
import torch

from training.rewards import parse_reasoning, composite_reward

SYSTEM = ("You are Konoros-Sec, a defensive security assistant. "
          "Think step by step inside <think>...</think>, then give the final "
          "user-facing reply inside <answer>...</answer>. Only help with "
          "authorized, lawful security work. Refuse exploit, persistence, "
          "credential-theft, evasion, and unauthorized-access requests.")


def build_prefix(tok, prompt: str) -> list[int]:
    return tok.encode(f"<system>{SYSTEM}</system><user>{prompt}</user>", add_bos=True)


@torch.no_grad()
def rollout_group(model, tok, prompt: str, G: int = 4, max_new_tokens: int = 128,
                  temperature: float = 0.8, top_p: float = 0.9,
                  device: str = "cpu") -> list[dict]:
    """Sample G completions for one prompt. Returns [{ids, text, think, answer}]."""
    prefix = build_prefix(tok, prompt)
    x = torch.tensor([prefix], dtype=torch.long, device=device)
    outs = []
    model.eval()
    for _ in range(G):
        gen = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature,
                             top_k=0, top_p=top_p, eos_id=tok.eos_id)
        new_ids = gen[0, len(prefix):].tolist()
        text = tok.decode(new_ids)
        p = parse_reasoning(text)
        outs.append({"ids": prefix + new_ids, "new_ids": new_ids,
                     "text": text, "think": p["think"], "answer": p["answer"]})
    return outs


def score_group(outs: list[dict], label: str, max_think_chars: int = 2000) -> list[dict]:
    for o in outs:
        o["reward"] = composite_reward(o["text"], label, max_think_chars)
    return outs
