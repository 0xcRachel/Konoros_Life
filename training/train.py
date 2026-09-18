"""v0.1 pretrain loop: python -m training.train --config config/model/tiny.yaml"""
import argparse
import itertools
import os
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

from model import ModelConfig, MyAI, count_params
from tokenizer import ByteTokenizer
from .dataset import get_loaders
from .loss import lm_loss
from .optimizer import build_optimizer, cosine_schedule, set_lr
from .checkpoint import save_checkpoint, load_checkpoint, check_tokenizer_compat


def load_full_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(model, loader, device, pad_id, max_batches=20):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y in itertools.islice(loader, max_batches):
            x, y = x.to(device), y.to(device)
            out = lm_loss(model(x), y, pad_id)
            tot += out["loss"].item()
            n += 1
    model.train()
    return tot / max(1, n)


def main(config_path: str, resume: str | None = None):
    full = load_full_config(config_path)
    mcfg = ModelConfig.from_yaml(config_path)
    tcfg = full.get("training", {})
    dcfg = full.get("data", {})
    ccfg = full.get("checkpoint", {})

    batch_size = int(tcfg.get("batch_size", 32))
    base_lr = float(tcfg.get("lr", 3e-4))
    max_steps = int(tcfg.get("max_steps", 5000))
    warmup = int(tcfg.get("warmup_steps", 200))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    grad_accum = max(1, int(tcfg.get("grad_accum_steps", 1)))
    use_ckpt = bool(tcfg.get("grad_checkpoint", False))
    eval_interval = int(tcfg.get("eval_interval", 500))
    save_interval = int(tcfg.get("save_interval", 1000))
    seed = int(tcfg.get("seed", 42))
    want = str(tcfg.get("device", "cuda"))
    dtype_s = str(tcfg.get("dtype", "bfloat16"))

    torch.manual_seed(seed)
    device = "cuda" if (want == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"device={device} params_cfg={mcfg.to_dict()}")

    tok = ByteTokenizer()
    model = MyAI(mcfg).to(device)
    if use_ckpt:
        model.gradient_checkpointing_enable()
        print("gradient checkpointing: ON")
    print(f"grad_accum_steps={grad_accum} (~{batch_size * grad_accum} toks-batch effective x{mcfg.context_length})")
    print(f"params={count_params(model):,}")
    opt = build_optimizer(model, lr=base_lr,
                          weight_decay=float(tcfg.get("weight_decay", 0.1)),
                          betas=tuple(tcfg.get("betas", [0.9, 0.95])))

    start_step = 0
    if resume and os.path.exists(resume):
        ckpt = load_checkpoint(resume, model, opt, map_location=device)
        check_tokenizer_compat(ckpt.get("tokenizer_meta", {}), tok)
        start_step = int(ckpt.get("step", 0)) + 1
        print(f"resumed from {resume} at step {start_step}")

    train_loader, val_loader = get_loaders(
        dcfg["train_path"], dcfg["val_path"], mcfg.context_length, batch_size, seed)
    train_iter = itertools.cycle(train_loader)

    ckpt_dir = ccfg.get("dir", "experiments/v0.1/checkpoints")
    log_dir = ccfg.get("log_dir", "experiments/v0.1/logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if dtype_s == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    model.train()
    try:
        for step in range(start_step, max_steps):
            lr = cosine_schedule(step, max_steps, warmup, base_lr)
            set_lr(opt, lr)
            opt.zero_grad(set_to_none=True)
            acc_loss, acc_ppl, acc_acc = 0.0, 0.0, 0.0
            for _ in range(grad_accum):
                x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(x)
                    out = lm_loss(logits, y, mcfg.pad_id)
                    loss = out["loss"] / grad_accum
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                acc_loss += out["loss"].item() / grad_accum
                acc_ppl += out["ppl"].item() / grad_accum
                acc_acc += out["acc"].item() / grad_accum
            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            if step % 10 == 0:
                print(f"step {step:5d} loss {acc_loss:.4f} ppl {acc_ppl:.2f} "
                      f"acc {acc_acc:.3f} lr {lr:.2e}")
                writer.add_scalar("train/loss", acc_loss, step)
                writer.add_scalar("train/lr", lr, step)

            if (step + 1) % eval_interval == 0:
                vl = evaluate(model, val_loader, device, mcfg.pad_id)
                print(f"[eval] step {step+1} val_loss {vl:.4f} val_ppl {float(__import__('math').exp(min(vl, 20))):.2f}")
                writer.add_scalar("val/loss", vl, step + 1)

            if (step + 1) % save_interval == 0:
                path = os.path.join(ckpt_dir, f"step_{step+1:06d}.pt")
                save_checkpoint(path, model, opt, step=step + 1,
                                config=full,
                                tokenizer_meta={"vocab_size": tok.vocab_size,
                                                "bos": tok.bos_id, "eos": tok.eos_id, "pad": tok.pad_id})
                print(f"saved {path}")
    except KeyboardInterrupt:
        print("interrupted — saving last.pt (kill-switch safe)")
        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, opt,
                        step=step, config=full,
                        tokenizer_meta={"vocab_size": tok.vocab_size,
                                        "bos": tok.bos_id, "eos": tok.eos_id, "pad": tok.pad_id})
    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/model/tiny.yaml")
    p.add_argument("--resume", default=None)
    a = p.parse_args()
    main(a.config, a.resume)
