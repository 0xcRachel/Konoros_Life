from dataclasses import dataclass, field
import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 320
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 0  # 0 = MHA (n_kv_heads == n_heads); e.g. 2 = GQA, 1 = MQA
    context_length: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_weights: bool = True
    norm: str = "rmsnorm"
    mlp: str = "swiglu"
    bos_id: int = 258
    eos_id: int = 259
    pad_id: int = 256
    unk_id: int = 257

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert (self.d_model // self.n_heads) % 2 == 0, "head_dim must be even for RoPE"
        assert self.vocab_size > 260, "vocab_size must reserve special tokens (>=261)"
        kv = self.n_kv_heads or self.n_heads
        assert self.n_heads % kv == 0, "n_heads must be divisible by n_kv_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_cfg = dict(cfg.get("model", {}))
        # Backward compat: fix old typos vocal_size / drop_out
        if "vocal_size" in model_cfg and "vocab_size" not in model_cfg:
            model_cfg["vocab_size"] = model_cfg.pop("vocal_size")
        if "drop_out" in model_cfg and "dropout" not in model_cfg:
            model_cfg["dropout"] = model_cfg.pop("drop_out")
        # Drop unknown keys gracefully (e.g. reasoning block lives elsewhere)
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        model_cfg = {k: v for k, v in model_cfg.items() if k in allowed}
        obj = cls(**model_cfg)
        obj._raw = cfg  # keep full yaml (training/data/tokenizer/reasoning)
        return obj

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__dataclass_fields__ if not f.startswith("_")}
