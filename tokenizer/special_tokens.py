# Konoros-Sec special tokens (reserved 64 IDs: 256..319, vocab_size=320)
# Byte 0..255 = raw UTF-8 bytes. Special tokens never collide with bytes.
PAD = "<pad>"            # 256
UNK = "<unk>"            # 257
BOS = "<bos>"            # 258
EOS = "<eos>"            # 259
THINK_BOS = "<think>"    # 260
THINK_EOS = "</think>"   # 261
ANSWER_BOS = "<answer>"  # 262
ANSWER_EOS = "</answer>" # 263
TOOL_BOS = "<tool>"      # 264
TOOL_EOS = "</tool>"     # 265
USER = "<user>"          # 266
ASSISTANT = "<assistant>"  # 267
SYSTEM = "<system>"      # 268
SCRATCHPAD = "<scratchpad>"  # 269

SPECIAL_TOKENS: list[str] = [
    PAD, UNK, BOS, EOS,
    THINK_BOS, THINK_EOS, ANSWER_BOS, ANSWER_EOS,
    TOOL_BOS, TOOL_EOS, USER, ASSISTANT, SYSTEM, SCRATCHPAD,
]

# Reserved for future (reasoning v1.3+, tool-use v1.5+): <reserved_00..49>
RESERVED_N = 50

BASE_VOCAB = 256


def build_vocab() -> dict[str, int]:
    vocab: dict[str, int] = {}
    # bytes as single-char keys are handled numerically; store specials only
    idx = BASE_VOCAB
    for tok in SPECIAL_TOKENS:
        vocab[tok] = idx
        idx += 1
    for i in range(RESERVED_N):
        vocab[f"<reserved_{i:02d}>"] = idx
        idx += 1
    return vocab


def vocab_size() -> int:
    return BASE_VOCAB + len(SPECIAL_TOKENS) + RESERVED_N  # 320
