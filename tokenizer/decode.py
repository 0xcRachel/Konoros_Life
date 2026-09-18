"""Decode .bin -> text for debugging. Usage: python -m tokenizer.decode --input data/processed/train.bin --max 500"""
import argparse
import numpy as np
from .byte_tokenizer import ByteTokenizer


def main(inp: str, tok_path: str | None, max_tokens: int):
    tok = ByteTokenizer.load(tok_path) if tok_path else ByteTokenizer()
    arr = np.fromfile(inp, dtype=np.uint16)[:max_tokens]
    print(tok.decode(arr.tolist()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--tok", default=None)
    p.add_argument("--max", type=int, default=500, dest="max_tokens")
    a = p.parse_args()
    main(a.input, a.tok, a.max_tokens)
