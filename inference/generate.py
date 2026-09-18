"""Generate text: python -m inference.generate --ckpt ... --prompt "Hello" --config config/model/tiny.yaml"""
import argparse
import torch

from model import ModelConfig, MyAI
from tokenizer import ByteTokenizer
from training.checkpoint import load_checkpoint, check_tokenizer_compat


def main(ckpt_path, prompt, max_new_tokens, temperature, top_k, top_p, config_path):
    mcfg = ModelConfig.from_yaml(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyAI(mcfg).to(device)
    ckpt = load_checkpoint(ckpt_path, model, map_location=device)
    tok = ByteTokenizer()
    check_tokenizer_compat(ckpt.get("tokenizer_meta", {}), tok)
    model.eval()

    ids = tok.encode(prompt)
    if not ids:
        ids = [tok.bos_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_k=top_k, top_p=top_p, eos_id=tok.eos_id)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.0)
    p.add_argument("--config", default="config/model/tiny.yaml")
    args = p.parse_args()
    main(args.ckpt, args.prompt, args.max_new_tokens, args.temperature,
         args.top_k, args.top_p, args.config)
