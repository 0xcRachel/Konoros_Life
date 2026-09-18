"""Eval: perplexity + fixed generation samples. python -m evaluation.evaluate --config ... --ckpt ..."""
import argparse
import itertools
import math
import torch

from model import ModelConfig, MyAI
from tokenizer import ByteTokenizer
from training.dataset import get_loaders
from training.loss import lm_loss
from training.checkpoint import load_checkpoint

PROMPTS = [
    "The cat is",
    "The model is learning",
    "def hello():",
    "<user>How do I secure my home WiFi?</user>",
]


def main(config_path: str, ckpt_path: str | None, max_batches: int = 20):
    import yaml
    with open(config_path, encoding="utf-8") as f:
        full = yaml.safe_load(f)
    mcfg = ModelConfig.from_yaml(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyAI(mcfg).to(device)
    if ckpt_path:
        load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()
    tok = ByteTokenizer()

    dcfg = full.get("data", {})
    try:
        _, val_loader = get_loaders(dcfg["train_path"], dcfg["val_path"],
                                    mcfg.context_length, full["training"].get("batch_size", 32))
        tot, n = 0.0, 0
        with torch.no_grad():
            for x, y in itertools.islice(val_loader, max_batches):
                x, y = x.to(device), y.to(device)
                tot += lm_loss(model(x), y, mcfg.pad_id)["loss"].item()
                n += 1
        vl = tot / max(1, n)
        print(f"val_loss={vl:.4f} val_ppl={math.exp(min(vl, 20)):.2f}")
    except Exception as e:
        print(f"[eval] skip ppl (missing data): {e}")

    print("---- samples ----")
    with torch.no_grad():
        for pr in PROMPTS:
            ids = tok.encode(pr) or [tok.bos_id]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(x, max_new_tokens=60, temperature=0.8, top_k=50, eos_id=tok.eos_id)
            print(f"> {pr}\n{tok.decode(out[0].tolist())}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/model/tiny.yaml")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--max-batches", type=int, default=20)
    a = p.parse_args()
    main(a.config, a.ckpt, a.max_batches)
