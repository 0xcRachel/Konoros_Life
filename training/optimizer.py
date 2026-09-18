import math
import torch


def build_optimizer(model, lr: float = 3e-4, weight_decay: float = 0.1,
                    betas=(0.9, 0.95)):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=tuple(betas),
    )


def cosine_schedule(step: int, max_steps: int, warmup_steps: int,
                    base_lr: float, min_lr_ratio: float = 0.1):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)


def set_lr(optimizer, lr: float):
    for g in optimizer.param_groups:
        # group 0 is decay group carrying base lr; scale others proportionally
        g["lr"] = lr if g.get("weight_decay", 0) else lr
