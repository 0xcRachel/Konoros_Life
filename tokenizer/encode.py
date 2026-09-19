"""Encode text/txt/jsonl -> .bin (uint16). Usage:
python -m tokenizer.encode --input data/raw/train.txt --output data/processed/train.bin --add-eos
"""
import argparse
import json
import numpy as np
from .byte_tokenizer import ByteTokenizer


def iter_texts(path: str):
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # support pretrain {"text"} and sft {"prompt","think","answer"}
                if "text" in obj:
                    yield obj["text"]
                else:
                    p = obj.get("prompt", "")
                    th = obj.get("think", "")
                    a = obj.get("answer", obj.get("completion", ""))
                    yield f"<user>{p}</user><think>{th}</think><answer>{a}</answer>"
    else:
        with open(path, encoding="utf-8") as f:
            yield f.read()


def main(inp: str, out: str, tok_path: str | None, add_bos: bool, add_eos: bool):
    import os
    from .bpe import BPETokenizer
    if tok_path and os.path.exists(tok_path):
        with open(tok_path, encoding="utf-8") as f:
            head = f.read(500)
        tok = BPETokenizer.load(tok_path) if '"model"' in head else ByteTokenizer.load(tok_path)
    else:
        tok = ByteTokenizer()
    ids: list[int] = []
    for t in iter_texts(inp):
        ids.extend(tok.encode(t, add_bos=add_bos, add_eos=add_eos))
    arr = np.array(ids, dtype=np.uint16)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    arr.tofile(out)
    print(f"encoded {len(ids)} tokens -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tok", default=None)
    p.add_argument("--add-bos", action="store_true")
    p.add_argument("--add-eos", action="store_true", default=True)
    p.add_argument("--no-eos", dest="add_eos", action="store_false")
    a = p.parse_args()
    main(a.input, a.output, a.tok, a.add_bos, a.add_eos)
