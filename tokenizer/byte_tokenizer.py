import torch

class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 260

    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes([i for i in ids if i < 256]).decode("utf-8", errors="replace")

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]