"""DPO (v1.2 alignment): learn from chosen/rejected pairs, no reward model needed.

Loss: -log sigmoid(beta * ((lp_c - lp_r) - (ref_c - ref_r)))
Prompt tokens masked (identical in both arms, they cancel anyway).
Pairs file: {"prompt","chosen","rejected"} where chosen/rejected are full
assistant texts (may contain <think>/<answer>).
"""
import json
import os

from tokenizer import ByteTokenizer
from training.chat_template import render_pref, prompt_prefix


def load_pairs(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("prompt") and r.get("chosen") and r.get("rejected"):
                    rows.append(r)
    return rows


def build_ids(tok: ByteTokenizer, prompt: str, completion: str,
              context_length: int = 512):
    """Full SFT string with prompt part masked (-100). Pure list ops (testable w/o torch)."""
    prefix = prompt_prefix(prompt)
    full = render_pref(prompt, completion)
    pre_ids = tok.encode(prefix, add_bos=True)[:context_length]
    ids = tok.encode(full, add_bos=True, add_eos=True)[:context_length]
    labels = list(ids)
    # mask everything up to the length of the shared prefix
    for i in range(min(len(pre_ids), len(labels))):
        labels[i] = -100
    return ids, labels


def seq_logprob_masked(model, ids, labels) -> float:
    import torch
    import torch.nn.functional as F
    logits = model(ids)[:, :-1, :]
    logp = F.log_softmax(logits, dim=-1)
    tgt = ids[:, 1:].unsqueeze(-1)
    tok_lp = logp.gather(-1, tgt).squeeze(-1)
    mask = (labels[:, 1:] != -100).float()
    return (tok_lp * mask).sum() / mask.sum().clamp_min(1)


def dpo_loss(policy, ref, tok, prompt, chosen, rejected, beta, context_length, device):
    import torch
    import torch.nn.functional as F
    ids_c, lab_c = build_ids(tok, prompt, chosen, context_length)
    ids_r, lab_r = build_ids(tok, prompt, rejected, context_length)
    xc = torch.tensor([ids_c], dtype=torch.long, device=device)
    lc = torch.tensor([lab_c], dtype=torch.long, device=device)
    xr = torch.tensor([ids_r], dtype=torch.long, device=device)
    lr_ = torch.tensor([lab_r], dtype=torch.long, device=device)
    with torch.no_grad():
        ref_c = seq_logprob_masked(ref, xc, lc)
        ref_r = seq_logprob_masked(ref, xr, lr_)
    lp_c = seq_logprob_masked(policy, xc, lc)
    lp_r = seq_logprob_masked(policy, xr, lr_)
    margin = (lp_c - lp_r) - (ref_c - ref_r)
    loss = -F.logsigmoid(beta * margin)
    acc = (margin > 0).float()
    return loss, acc, margin.detach()


def main(config_path: str, data_path: str, base_ckpt: str | None, out: str,
         steps: int, lr: float, beta: float):
    import yaml
    import torch
    from model import ModelConfig, MyAI
    from training.checkpoint import save_checkpoint, load_checkpoint
    from training.optimizer import build_optimizer
    with open(config_path, encoding="utf-8") as f:
        full = yaml.safe_load(f)
    d = full.get("dpo", {})
    data_path = data_path or d.get("data_path", "data/sft/security_prefs.jsonl")
    steps = steps or int(d.get("steps", 200))
    lr = lr or float(d.get("lr", 5e-6))
    beta = beta or float(d.get("beta", 0.1))

    mcfg = ModelConfig.from_yaml(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = ByteTokenizer()
    policy = MyAI(mcfg).to(device)
    if base_ckpt and os.path.exists(base_ckpt):
        load_checkpoint(base_ckpt, policy, map_location=device)
        print(f"loaded base {base_ckpt}")
    import copy
    ref = copy.deepcopy(policy).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    rows = load_pairs(data_path)
    assert rows, f"no pairs in {data_path}"
    print(f"DPO pairs={len(rows)} steps={steps} device={device}")
    opt = build_optimizer(policy, lr=lr, weight_decay=0.0)
    policy.train()
    for step in range(steps):
        r = rows[step % len(rows)]
        loss, acc, margin = dpo_loss(policy, ref, tok, r["prompt"], r["chosen"],
                                     r["rejected"], beta, mcfg.context_length, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            print(f"dpo {step:4d} loss {loss.item():.4f} acc {acc.item():.0f} "
                  f"margin {margin.item():+.3f} :: {r['prompt'][:60]}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_checkpoint(out, policy, opt, step=steps, config=full,
                    tokenizer_meta={"vocab_size": tok.vocab_size})
    print(f"saved DPO -> {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/model/tiny.yaml")
    p.add_argument("--data", default="data/sft/security_prefs.jsonl")
    p.add_argument("--base-ckpt", default="")
    p.add_argument("--out", default="experiments/v1.2/dpo.pt")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=0.0)
    a = p.parse_args()
    main(a.config, a.data, a.base_ckpt or None, a.out, a.steps, a.lr, a.beta)
