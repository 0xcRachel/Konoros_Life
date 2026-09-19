"""BPE tokenizer (ByteLevel, GPT-2 style) — thay byte-level khi data/code lớn.

Train trên Colab: pip install tokenizers
  python scripts/train_bpe.py --input data/raw/prime_all.jsonl --out data/tokenizer/bpe.json --vocab 16000
Wrapper có API giống ByteTokenizer (encode/decode/encode_batch/save/load) để
prepare_data/train/generate dùng chung, chỉ khác --tok-type bpe.
"""
import json
import os

from .special_tokens import SPECIAL_TOKENS


class BPETokenizer:
    def __init__(self, tok_file: str):
        from tokenizers import Tokenizer  # pip install tokenizers (Colab)
        self.tok = Tokenizer.from_file(tok_file)
        self.tok_file = tok_file
        vs = self.tok.get_vocab()
        self.pad_id = vs.get("<pad>", 0)
        self.unk_id = vs.get("<unk>", 1)
        self.bos_id = vs.get("<bos>", 2)
        self.eos_id = vs.get("<eos>", 3)
        self.vocab_size = self.tok.get_vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tok.encode(text).ids
        out = ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])
        return out

    def decode(self, ids: list[int] | tuple, skip_special: bool = False) -> str:
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=skip_special)

    def encode_batch(self, texts: list[str], add_bos=False, add_eos=True, pad_to: int | None = None):
        seqs = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]
        maxlen = pad_to or max(len(s) for s in seqs)
        padded, mask = [], []
        for s in seqs:
            s = s[:maxlen]
            m = [1] * len(s) + [0] * (maxlen - len(s))
            s = s + [self.pad_id] * (maxlen - len(s))
            padded.append(s)
            mask.append(m)
        return padded, mask

    def save(self, path: str):
        import shutil
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.abspath(path) != os.path.abspath(self.tok_file):
            shutil.copyfile(self.tok_file, path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        return cls(path)

    @staticmethod
    def special_list() -> list[str]:
        return list(SPECIAL_TOKENS) + [f"<reserved_{i:02d}>" for i in range(50)]


def dump_meta(tok_file: str, out: str):
    tok = BPETokenizer(tok_file)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"vocab_size": tok.vocab_size, "bos": tok.bos_id,
                   "eos": tok.eos_id, "pad": tok.pad_id, "unk": tok.unk_id}, f)
