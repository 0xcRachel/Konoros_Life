import os

from .byte_tokenizer import ByteTokenizer
from .bpe import BPETokenizer
from .special_tokens import (
    PAD, UNK, BOS, EOS, THINK_BOS, THINK_EOS, ANSWER_BOS, ANSWER_EOS,
    TOOL_BOS, TOOL_EOS, USER, ASSISTANT, SYSTEM, SCRATCHPAD,
    SPECIAL_TOKENS, build_vocab, vocab_size,
)

__all__ = [
    "ByteTokenizer", "BPETokenizer", "load_tokenizer",
    "PAD", "UNK", "BOS", "EOS",
    "THINK_BOS", "THINK_EOS", "ANSWER_BOS", "ANSWER_EOS",
    "TOOL_BOS", "TOOL_EOS", "USER", "ASSISTANT", "SYSTEM", "SCRATCHPAD",
    "SPECIAL_TOKENS", "build_vocab", "vocab_size",
]


def load_tokenizer(tok_type: str = "byte", path: str | None = None):
    """Đọc tokenizer theo yaml: type byte|bpe. BPE cần file đã train (scripts/train_bpe.py)."""
    if (tok_type or "byte") == "bpe":
        assert path, "bpe tokenizer needs tokenizer.path in yaml"
        return BPETokenizer.load(path)
    if path and os.path.exists(path):
        try:
            return ByteTokenizer.load(path)
        except Exception:
            pass
    return ByteTokenizer()
