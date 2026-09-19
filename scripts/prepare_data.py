"""Prepare .bin from raw txt/jsonl. --input accepts comma-separated files:
python scripts/prepare_data.py --input data/raw/so_python.jsonl,data/raw/so_security.jsonl,data/raw/train.txt --train-out data/processed/train.bin --val-out data/processed/val.bin
"""
import argparse
import json
import os
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import ByteTokenizer, BPETokenizer


def get_tokenizer(tok_type: str, tok_out: str | None):
    if tok_type == "bpe":
        assert tok_out and os.path.exists(tok_out), f"BPE cần file có sẵn: {tok_out} (train bằng scripts/train_bpe.py)"
        return BPETokenizer.load(tok_out)
    tok = ByteTokenizer()
    if tok_out:
        tok.save(tok_out)
    return tok


def load_texts(path: str) -> list[str]:
    """Return list of DOCS (packing: mỗi doc encode riêng + EOS, nối liền, zero padding)."""
    if path.endswith(".jsonl"):
        docs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "text" in obj:
                    docs.append(obj["text"])
                else:
                    docs.append(
                        f"<user>{obj.get('prompt','')}</user>"
                        f"<think>{obj.get('think','')}</think>"
                        f"<answer>{obj.get('answer', obj.get('completion',''))}</answer>"
                    )
        return docs
    with open(path, encoding="utf-8") as f:
        return [f.read()]


def main(inp, train_out, val_out, tok_out, tok_type="byte", val_ratio=0.05, add_eos=True, repeat=200):
    os.makedirs(os.path.dirname(train_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(val_out) or ".", exist_ok=True)
    if tok_out:
        os.makedirs(os.path.dirname(tok_out) or ".", exist_ok=True)
    tok = get_tokenizer(tok_type, tok_out)
    ids: list[int] = []
    n_docs = 0
    for one in [s.strip() for s in inp.split(",") if s.strip()]:
        for doc in load_texts(one):
            part = tok.encode(doc, add_eos=add_eos)  # EOS phân cách doc (packing, không padding)
            if part:
                ids.extend(part)
                n_docs += 1
    print(f"[prepare] {n_docs} docs packed: {len(ids)} tokens")
    if len(ids) < 4096:
        print(f"[prepare] only {len(ids)} tokens, repeating x{repeat} for smoke training")
        ids = (ids * repeat)[: 20000]
    n_val = max(512, int(len(ids) * val_ratio))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    np.array(train_ids, dtype=np.uint16).tofile(train_out)
    np.array(val_ids, dtype=np.uint16).tofile(val_out)
    print(f"train {len(train_ids)} -> {train_out} | val {len(val_ids)} -> {val_out}")
    # token budget: ước lượng số steps/epoch cho Colab
    for name, bs, ctx in [("tiny/bs64x512", 64, 512), ("small/eff64x1024", 64, 1024)]:
        print(f"[budget] {name}: {len(train_ids) // (bs * ctx)} steps/epoch")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/prime_all.jsonl")
    p.add_argument("--train-out", default="data/processed/train.bin")
    p.add_argument("--val-out", default="data/processed/val.bin")
    p.add_argument("--tok-out", default="data/tokenizer/byte.json")
    p.add_argument("--tok-type", default="byte", choices=["byte", "bpe"])
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--repeat", type=int, default=200)
    a = p.parse_args()
    main(a.input, a.train_out, a.val_out, a.tok_out, a.tok_type, a.val_ratio, True, a.repeat)
