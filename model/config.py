from dataclasses import dataclass
import yaml

@dataclass
class ModelConfig:
    vocal_size : int = 260
    d_model : int = 256
    n_layers : int = 4
    n_heads : int = 4
    context_length : int = 512
    drop_out : float = 0.0
    rope_theta: float = 10000.0
    tie_weights: bool = True


    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg["model"])