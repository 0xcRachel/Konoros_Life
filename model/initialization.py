import math
import torch.nn as nn


def init_weights(module: nn.Module, n_layers: int = 4):
    """GPT-2 style init, scaled by depth."""
    if isinstance(module, nn.Linear):
        std = 0.02 / math.sqrt(2 * n_layers)
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
