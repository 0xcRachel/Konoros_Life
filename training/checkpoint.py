"""Checkpoint save/load with tokenizer-hash guard (torch import lazy để tooling nhẹ dùng được)."""
import glob
import hashlib
import os


def _tok_hash(vocab_size: int, bos: int, eos: int, pad: int) -> str:
    return hashlib.md5(f"{vocab_size}-{bos}-{eos}-{pad}".encode()).hexdigest()[:12]


def save_checkpoint(path: str, model, optimizer=None, scheduler_state=None,
                    step: int = 0, config: dict | None = None,
                    tokenizer_meta: dict | None = None):
    import torch
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "config": config or {},
        "tokenizer_meta": tokenizer_meta or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler_state is not None:
        payload["scheduler"] = scheduler_state
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer=None, map_location="cpu", strict: bool = True):
    import torch
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[ckpt] optimizer load skipped: {e}")
    return ckpt


def check_tokenizer_compat(tokenizer_meta: dict, tok) -> bool:
    """Warn if vocab changed since checkpoint (would break embeddings)."""
    if not tokenizer_meta:
        return True
    expect = tokenizer_meta.get("vocab_size")
    if expect is not None and expect != tok.vocab_size:
        print(f"[ckpt][WARN] vocab mismatch: ckpt={expect} vs tok={tok.vocab_size}. "
              "Embedding shape may fail — rebuild tokenizer or retrain head.")
        return False
    return True


def resolve_ckpt(path: str) -> str:
    """Trả về path nếu tồn tại; nếu không, fallback checkpoint mới nhất dưới experiments/.

    Không bao giờ crash FileNotFoundError câm — luôn in rõ đang dùng file nào.
    """
    if path and os.path.exists(path):
        return path
    cands = sorted(
        glob.glob("experiments/**/step_*.pt", recursive=True)
        + glob.glob("experiments/**/last.pt", recursive=True),
        key=os.path.getmtime,
    )
    if not cands:
        raise FileNotFoundError(
            f"checkpoint '{path}' not found and no fallback under experiments/. "
            "Train trước (training.train) hoặc kiểm tra lại đường dẫn."
        )
    best = cands[-1]
    print(f"[ckpt] '{path}' not found -> fallback newest: {best}")
    return best
