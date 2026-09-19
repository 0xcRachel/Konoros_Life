"""Generate text: python -m inference.generate --ckpt ... --prompt "Hello" --config config/model/tiny.yaml
Phân nhánh output: --best-of N sample N nhánh, chấm điểm (format + ít lặp + đủ dài), trả nhánh tốt nhất.
"""
import argparse
import torch

from model import ModelConfig, MyAI
from tokenizer import ByteTokenizer
from training.checkpoint import load_checkpoint, check_tokenizer_compat
from training.rewards import parse_reasoning


def branch_score(text: str, new_ids: list[int]) -> float:
    p = parse_reasoning(text if "<think>" in text or "<answer>" in text else f"<answer>{text}</answer>")
    s = 0.0
    if p["has_answer"]:
        s += 2.0
    if p["has_think"]:
        s += 1.0
    if len(p["answer"]) >= 20:
        s += 0.5
    s += len(set(new_ids)) / max(1, len(new_ids))  # chống lặp token
    return s


def main(ckpt_path, prompt, max_new_tokens, temperature, top_k, top_p, config_path, best_of: int = 1):
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
    best_of = max(1, best_of)
    cands = []
    for _ in range(best_of):
        out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature,
                             top_k=top_k, top_p=top_p, eos_id=tok.eos_id)
        new_ids = out[0, len(ids):].tolist()
        text = tok.decode(out[0].tolist()[len(ids):])
        cands.append((branch_score(text, new_ids), text))
    cands.sort(key=lambda t: -t[0])
    if best_of > 1:
        print(f"[best-of-{best_of}] scores: {[round(s, 2) for s, _ in cands]}")
    print(cands[0][1])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.0)
    p.add_argument("--best-of", type=int, default=1)
    p.add_argument("--config", default="config/model/tiny.yaml")
    args = p.parse_args()
    main(args.ckpt, args.prompt, args.max_new_tokens, args.temperature,
         args.top_k, args.top_p, args.config, args.best_of)
