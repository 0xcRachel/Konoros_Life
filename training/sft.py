"""v1.1 SFT: python -m training.sft --config config/model/tiny.yaml --data data/sft/security_sft.jsonl"""
import argparse
import json
import os
import yaml
import torch

from model import ModelConfig, MyAI
from tokenizer import ByteTokenizer, load_tokenizer
from training.checkpoint import save_checkpoint, load_checkpoint, check_vocab
from training.chat_template import encode_sft
from training.optimizer import build_optimizer


def load_rows(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(config_path: str, data_path: str, base_ckpt: str | None,
         out: str, max_steps: int, lr: float):
    import yaml as _yaml
    with open(config_path, encoding="utf-8") as f:
        full = _yaml.safe_load(f)
    sft_cfg = full.get("sft", {})
    data_path = data_path or sft_cfg.get("data_path", "data/sft/security_sft.jsonl")
    base_ckpt = base_ckpt or sft_cfg.get("base_ckpt", "")
    max_steps = max_steps or int(sft_cfg.get("max_steps", 1000))
    lr = lr or float(sft_cfg.get("lr", 1e-5))

    mcfg = ModelConfig.from_yaml(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _tok_cfg = full.get("tokenizer", {})
    tok = load_tokenizer(_tok_cfg.get("type", "byte"), _tok_cfg.get("path"))
    check_vocab(tok, mcfg)
    model = MyAI(mcfg).to(device)
    if base_ckpt and os.path.exists(base_ckpt):
        load_checkpoint(base_ckpt, model, map_location=device)
        print(f"loaded base {base_ckpt}")
    rows = load_rows(data_path)
    assert rows, f"no SFT rows in {data_path}"
    print(f"SFT rows={len(rows)} device={device}")
    opt = build_optimizer(model, lr=lr, weight_decay=0.0)
    model.train()
    import torch.nn.functional as F
    step = 0
    while step < max_steps:
        for r in rows:
            if step >= max_steps:
                break
            ids, labels = encode_sft(tok, r.get("prompt", ""), r.get("think", ""),
                                     r.get("answer", ""), mcfg.context_length)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            lab = torch.tensor([labels + [-100] * (len(ids) - len(labels))][:len(ids)],
                               dtype=torch.long, device=device)
            # shift for causal LM
            logits = model(x)[:, :-1, :]
            tgt = lab[:, 1:] if lab.shape[1] > 1 else lab
            tgt_in = x[:, 1:] if x.shape[1] > 1 else x
            # use masked labels where available
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                   ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 20 == 0:
                print(f"sft step {step} loss {loss.item():.4f}")
            step += 1
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_checkpoint(out, model, opt, step=step, config=full,
                    tokenizer_meta={"vocab_size": tok.vocab_size})
    print(f"saved SFT -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/model/tiny.yaml")
    p.add_argument("--data", default="data/sft/security_sft.jsonl")
    p.add_argument("--base-ckpt", default="")
    p.add_argument("--out", default="experiments/v1.1/sft.pt")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.0)
    a = p.parse_args()
    main(a.config, a.data, a.base_ckpt or None, a.out, a.max_steps, a.lr)
