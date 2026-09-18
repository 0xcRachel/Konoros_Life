"""Prepare .bin from raw txt/jsonl. --input accepts comma-separated files:
python scripts/prepare_data.py --input data/raw/so_python.jsonl,data/raw/so_security.jsonl,data/raw/train.txt --train-out data/processed/train.bin --val-out data/processed/val.bin
"""
import argparse
import json
import os
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import ByteTokenizer


def load_texts(path: str) -> str:
    if path.endswith(".jsonl"):
        parts = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "text" in obj:
                    parts.append(obj["text"])
                else:
                    parts.append(
                        f"<user>{obj.get('prompt','')}</user>"
                        f"<think>{obj.get('think','')}</think>"
                        f"<answer>{obj.get('answer', obj.get('completion',''))}</answer>"
                    )
        return "\n".join(parts) + "\n"
    with open(path, encoding="utf-8") as f:
        return f.read()


def main(inp, train_out, val_out, tok_out, val_ratio=0.05, add_eos=True, repeat=200):
    os.makedirs(os.path.dirname(train_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(val_out) or ".", exist_ok=True)
    if tok_out:
        os.makedirs(os.path.dirname(tok_out) or ".", exist_ok=True)
    tok = ByteTokenizer()
    if tok_out:
        tok.save(tok_out)
    ids: list[int] = []
    for one in [s.strip() for s in inp.split(",") if s.strip()]:
        text = load_texts(one)
        part = tok.encode(text, add_eos=add_eos)
        print(f"[prepare] {one}: {len(part)} tokens")
        ids.extend(part)
    if len(ids) < 4096:
        print(f"[prepare] only {len(ids)} tokens, repeating x{repeat} for smoke training")
        ids = (ids * repeat)[: 20000]
    n_val = max(512, int(len(ids) * val_ratio))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    np.array(train_ids, dtype=np.uint16).tofile(train_out)
    np.array(val_ids, dtype=np.uint16).tofile(val_out)
    print(f"train {len(train_ids)} -> {train_out} | val {len(val_ids)} -> {val_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/train.txt")
    p.add_argument("--train-out", default="data/processed/train.bin")
    p.add_argument("--val-out", default="data/processed/val.bin")
    p.add_argument("--tok-out", default="data/tokenizer/byte.json")
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--repeat", type=int, default=200)
    a = p.parse_args()
    main(a.input, a.train_out, a.val_out, a.tok_out, a.val_ratio, True, a.repeat)
