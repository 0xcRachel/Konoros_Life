"""Byte-level tokenizer, reasoning-ready.

- IDs 0..255: raw UTF-8 bytes
- IDs 256+: special tokens (pad/unk/bos/eos/think/answer/tool/roles...)
- Never splits special tokens inside text: longest-match scan for <...>.
"""
import json
import re

from .special_tokens import build_vocab, vocab_size, PAD, UNK, BOS, EOS

_SPECIAL_RE = re.compile(r"<[a-zA-Z/_]+(?:_\d+)?>")


class ByteTokenizer:
    def __init__(self, vocab: dict[str, int] | None = None):
        self.vocab: dict[str, int] = vocab or build_vocab()
        self.inv_vocab: dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.vocab_size: int = vocab_size()
        self.pad_id = self.vocab[PAD]
        self.unk_id = self.vocab[UNK]
        self.bos_id = self.vocab[BOS]
        self.eos_id = self.vocab[EOS]

    # ---- core ----
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        pos = 0
        for m in _SPECIAL_RE.finditer(text):
            # bytes before the special token
            ids.extend(text[pos:m.start()].encode("utf-8"))
            tok = m.group(0)
            ids.append(self.vocab.get(tok, self.unk_id))
            pos = m.end()
        ids.extend(text[pos:].encode("utf-8"))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int] | tuple, skip_special: bool = False) -> str:
        out_bytes = bytearray()
        pieces: list[str] = []
        for i in ids:
            i = int(i)
            if i < 256:
                out_bytes.append(i)
            elif i in self.inv_vocab:
                tok = self.inv_vocab[i]
                if skip_special:
                    continue
                # flush pending bytes first
                if out_bytes:
                    pieces.append(out_bytes.decode("utf-8", errors="replace"))
                    out_bytes = bytearray()
                pieces.append(tok)
            else:
                if not skip_special:
                    pieces.append(self.inv_vocab.get(self.unk_id, "<unk>"))
        if out_bytes:
            pieces.append(out_bytes.decode("utf-8", errors="replace"))
        # If skip_special, pieces only has text chunks; join directly
        return "".join(pieces)

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

    # ---- IO ----
    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "vocab_size": self.vocab_size}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ByteTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"])
