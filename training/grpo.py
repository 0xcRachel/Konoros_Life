import json
import os
import torch
import torch.nn.functional as F

from model import ModelConfig, MyAI
from tokenizer import ByteTokenizer, load_tokenizer
from training.checkpoint import save_checkpoint, load_checkpoint, check_vocab
from training.optimizer import build_optimizer
from training.rollout import rollout_group, score_group, build_prefix


def load_rows(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows.append({"prompt": r.get("prompt", ""),
                             "label": r.get("label", "benign")})
    return [r for r in rows if r["prompt"]]


def seq_logprobs(model, ids: torch.Tensor) -> torch.Tensor:
    """Log-probs of each token given prefix: (B, T-1)."""
    logits = model(ids)[:, :-1, :]
    logp = F.log_softmax(logits, dim=-1)
    tgt = ids[:, 1:].unsqueeze(-1)
    return logp.gather(-1, tgt).squeeze(-1)


def grpo_step(policy, ref, opt, tok, prompt: str, label: str, G: int,
              max_new: int, temp: float, top_p: float, beta: float,
              eps: float, device: str) -> dict:
    outs = rollout_group(policy, tok, prompt, G, max_new, temp, top_p, device)
    outs = score_group(outs, label)
    rs = torch.tensor([o["reward"]["total"] for o in outs])
    adv = (rs - rs.mean()) / (rs.std() + 1e-6) if G > 1 else rs * 0.0

    prefix_len = len(build_prefix(tok, prompt))
    policy.train()
    tot_loss, tot_kl, tot_r = 0.0, 0.0, 0.0
    opt.zero_grad(set_to_none=True)
    for o, a in zip(outs, adv):
        ids = torch.tensor([o["ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            ref_lp = seq_logprobs(ref, ids)
            old_lp = seq_logprobs(policy, ids)
        new_lp = seq_logprobs(policy, ids)
        # only train on generated tokens (mask prompt)
        T = new_lp.shape[1]
        mask = torch.zeros(T, device=device)
        mask[max(0, prefix_len - 1):] = 1.0
        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
        pg = -torch.min(ratio * a.item(), clipped * a.item()) * mask
        # k3 KL to reference: exp(d) - d - 1, d = ref - new
        d = ref_lp - new_lp
        kl = (torch.exp(d) - d - 1.0) * mask
        denom = mask.sum().clamp_min(1)
        loss = (pg.sum() + beta * kl.sum()) / denom
        (loss / G).backward()
        tot_loss += loss.item() / G
        tot_kl += (kl.sum() / denom).item() / G
        tot_r += o["reward"]["total"] / G
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    return {"loss": tot_loss, "kl": tot_kl, "reward": tot_r}


def main(config_path: str, data_path: str, base_ckpt: str | None, out: str,
         steps: int, G: int, lr: float, beta: float, eps: float,
         max_new: int, temp: float):
    import yaml
    with open(config_path, encoding="utf-8") as f:
        full = yaml.safe_load(f)
    g = full.get("grpo", {})
    data_path = data_path or g.get("data_path", "data/sft/security_reasoning.jsonl")
    steps = steps or int(g.get("steps", 200))
    G = G or int(g.get("group", 4))
    lr = lr or float(g.get("lr", 1e-6))
    beta = float(g.get("beta", 0.02))
    eps = float(g.get("eps", 0.2))
    max_new = max_new or int(g.get("max_new_tokens", 128))

    mcfg = ModelConfig.from_yaml(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _tok_cfg = full.get("tokenizer", {})
    tok = load_tokenizer(_tok_cfg.get("type", "byte"), _tok_cfg.get("path"))
    check_vocab(tok, mcfg)
    policy = MyAI(mcfg).to(device)
    if base_ckpt and os.path.exists(base_ckpt):
        load_checkpoint(base_ckpt, policy, map_location=device)
        print(f"loaded base {base_ckpt}")
    import copy
    ref = copy.deepcopy(policy).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    rows = load_rows(data_path)
    assert rows, f"no rows in {data_path}"
    print(f"GRPO rows={len(rows)} G={G} steps={steps} device={device}")
    opt = build_optimizer(policy, lr=lr, weight_decay=0.0)
    for step in range(steps):
        r = rows[step % len(rows)]
        m = grpo_step(policy, ref, opt, tok, r["prompt"], r["label"],
                      G, max_new, temp, top_p=0.9, beta=beta, eps=eps, device=device)
        if step % 10 == 0:
            print(f"grpo {step:4d} loss {m['loss']:.4f} kl {m['kl']:.4f} "
                  f"reward {m['reward']:.3f} :: {r['prompt'][:60]}")
        if (step + 1) % 50 == 0:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            save_checkpoint(out.replace(".pt", f"_step{step+1}.pt"), policy, opt,
                            step=step + 1, config=full,
                            tokenizer_meta={"vocab_size": tok.vocab_size})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_checkpoint(out, policy, opt, step=steps, config=full,
                    tokenizer_meta={"vocab_size": tok.vocab_size})
    print(f"saved GRPO -> {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/model/tiny.yaml")
    p.add_argument("--data", default="data/sft/security_reasoning.jsonl")
    p.add_argument("--base-ckpt", default="")
    p.add_argument("--out", default="experiments/v1.3/grpo.pt")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--G", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--max-new", type=int, default=0)
    p.add_argument("--temp", type=float, default=0.8)
    a = p.parse_args()
    main(a.config, a.data, a.base_ckpt or None, a.out,
         a.steps, a.G, a.lr, a.beta, a.eps, a.max_new, a.temp)
