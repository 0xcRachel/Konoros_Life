"""Train BPE ByteLevel trên data prime (chạy Colab, cần RAM lớn để đọc corpus).

  pip install tokenizers
  python scripts/train_bpe.py --input data/raw/prime_all.jsonl --out data/tokenizer/bpe.json --vocab 16000
ByteLevel (như GPT-2): không bao giờ OOV, hợp code + đa ngôn ngữ en/vi.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def iter_texts(path: str):
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                yield o.get("text") or f"{o.get('prompt','')}\n{o.get('answer', o.get('completion',''))}"
    else:
        with open(path, encoding="utf-8") as f:
            yield f.read()


def main(inp: str, out: str, vocab: int):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizer.bpe import BPETokenizer

    files = [s.strip() for s in inp.split(",") if s.strip()]
    texts, n = [], 0
    for fp in files:
        for t in iter_texts(fp):
            if t.strip():
                texts.append(t)
                n += 1
    print(f"corpus docs: {n}")
    tmp = out + ".corpus.txt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab, min_frequency=2,
                         special_tokens=BPETokenizer.special_list())
    tok.train([tmp], trainer)
    tok.save(out)
    os.remove(tmp)
    meta = BPETokenizer(out)
    print(f"BPE vocab={meta.vocab_size} -> {out}")
    print("sample:", meta.encode("def hello(): print('hi')")[:12])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/prime_all.jsonl")
    p.add_argument("--out", default="data/tokenizer/bpe.json")
    p.add_argument("--vocab", type=int, default=16000)
    a = p.parse_args()
    main(a.input, a.out, a.vocab)
