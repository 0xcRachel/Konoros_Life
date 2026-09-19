"""Chat CLI đa lượt: python -m chat.chat --ckpt <ckpt> --config config/model/tiny.yaml
Lệnh trong chat: /reset (xoá lịch sử) | /think on|off (hiện suy nghĩ) | /temp 0.8 | /quit
"""
import argparse
import torch

from model import ModelConfig, MyAI
from tokenizer import ByteTokenizer
from training.checkpoint import load_checkpoint
from chat.session import build_history, add_turn, render_history, parse_reply


def main(ckpt, config, max_new, temperature, top_p, show_think):
    mcfg = ModelConfig.from_yaml(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyAI(mcfg).to(device).eval()
    if ckpt:
        load_checkpoint(ckpt, model, map_location=device)
        print(f"loaded {ckpt} [{device}]")
    tok = ByteTokenizer()
    hist = build_history()
    print("Konoros-Sec chat (defensive only). /quit để thoát.")
    while True:
        try:
            u = input("\nYOU> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not u:
            continue
        if u.startswith("/"):
            cmd, *rest = u[1:].split()
            if cmd == "quit":
                break
            elif cmd == "reset":
                hist = build_history();
                print("(đã xoá lịch sử)");
                continue
            elif cmd == "think":
                show_think = (rest + ["on"])[0].lower() != "off"
                print(f"show_think={show_think}");
                continue
            elif cmd == "temp":
                try:
                    temperature = float(rest[0]);
                    print(f"temperature={temperature}")
                except Exception:
                    print("dùng: /temp 0.8")
                continue
            else:
                print("lệnh: /reset /think on|off /temp X /quit");
                continue
        prompt = render_history(add_turn(hist, u))
        ids = tok.encode(prompt, add_bos=False)
        ids = ids[-mcfg.context_length:]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=max_new, temperature=temperature,
                                 top_k=0, top_p=top_p, eos_id=tok.eos_id)
        text = tok.decode(out[0, len(ids):].tolist())
        rep = parse_reply(text)
        add_turn(hist, u, text)
        if show_think and rep["think"]:
            print(f"\n[think] {rep['think']}")
        print(f"\nSEC> {rep['answer'] or text[:500]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="")
    p.add_argument("--config", default="config/model/tiny.yaml")
    p.add_argument("--max-new", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--show-think", action="store_true")
    a = p.parse_args()
    main(a.ckpt, a.config, a.max_new, a.temperature, a.top_p, a.show_think)
