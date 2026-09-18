import argparse, torch, yaml
from model.config import ModelConfig
from model.model import MyAI
from tokenizer.byte_tokenizer import ByteTokenizer
from training.checkpoint import load_checkpoint

def main(ckpt_path, prompt, max_new_tokens, temperature, top_k, config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    mcfg = ModelConfig(**cfg["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MyAI(mcfg).to(device)
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()

    tok = ByteTokenizer()
    ids = tok.encode(prompt)
    if not ids:
        ids = [0]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens, temperature, top_k)
    print(tok.decode(out[0].tolist()))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--config", default="configs/model/tiny.yaml")
    args = p.parse_args()
    main(args.ckpt, args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.config)