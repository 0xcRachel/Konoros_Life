import torch
import torch.nn.functional as F


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int | None = None):
    """Cross-entropy over (B,T,V). Returns dict with loss/ppl/acc."""
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T),
        ignore_index=pad_id if pad_id is not None else -100,
    )
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        mask = torch.ones_like(targets, dtype=torch.bool) if pad_id is None else (targets != pad_id)
        acc = ((pred == targets) & mask).float().sum() / mask.float().sum().clamp_min(1)
    return {"loss": loss, "ppl": torch.exp(loss.detach()), "acc": acc}
