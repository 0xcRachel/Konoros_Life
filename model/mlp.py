import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 * cfg.d_model / 3)
        hidden = ((hidden + 63) // 64) * 64  # make multiple of 64
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))