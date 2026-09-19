"""Preference eval: fraction of pairs where policy prefers chosen over rejected (torch, Colab)."""
import torch

from model import ModelConfig, MyAI
from tokenizer import ByteTokenizer, load_tokenizer
from training.checkpoint import load_checkpoint, resolve_ckpt
from training.dpo import load_pairs, seq_logprob_masked, build_ids


def main(config="config/model/tiny.yaml", ckpt=None, data="data/sft/security_prefs.jsonl"):
    import yaml
    with open(config, encoding="utf-8") as f:
        _tok_cfg = yaml.safe_load(f).get("tokenizer", {})
    mcfg = ModelConfig.from_yaml(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyAI(mcfg).to(device).eval()
    if ckpt:
        ckpt = resolve_ckpt(ckpt)
        load_checkpoint(ckpt, model, map_location=device)
    tok = load_tokenizer(_tok_cfg.get("type", "byte"), _tok_cfg.get("path"))
    rows = load_pairs(data)
    win, tot = 0, 0
    with torch.no_grad():
        for r in rows:
            ids_c, lab_c = build_ids(tok, r["prompt"], r["chosen"], mcfg.context_length)
            ids_r, lab_r = build_ids(tok, r["prompt"], r["rejected"], mcfg.context_length)
            xc = torch.tensor([ids_c], dtype=torch.long, device=device)
            lc = torch.tensor([lab_c], dtype=torch.long, device=device)
            xr = torch.tensor([ids_r], dtype=torch.long, device=device)
            lr_ = torch.tensor([lab_r], dtype=torch.long, device=device)
            sc = seq_logprob_masked(model, xc, lc).item()
            sr = seq_logprob_masked(model, xr, lr_).item()
            ok = sc > sr
            win += ok
            tot += 1
            print(f"[{'PASS' if ok else 'FAIL'}] ({r.get('label','?')}) {r['prompt'][:60]} "
                  f"chosen={sc:.2f} rejected={sr:.2f}")
    print(f"preference accuracy: {win}/{tot}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/model/tiny.yaml")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--data", default="data/sft/security_prefs.jsonl")
    a = p.parse_args()
    main(a.config, a.ckpt, a.data)
