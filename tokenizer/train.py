"""Build byte tokenizer vocab file. Usage: python -m tokenizer.train --out data/tokenizer/byte.json"""
import argparse
import os
from .byte_tokenizer import ByteTokenizer


def main(out: str):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tok = ByteTokenizer()
    tok.save(out)
    print(f"saved tokenizer vocab_size={tok.vocab_size} -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/tokenizer/byte.json")
    args = p.parse_args()
    main(args.out)
