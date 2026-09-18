import os
import torch

from model import ModelConfig, MyAI
from tokenizer import ByteTokenizer
from training.checkpoint import save_checkpoint, load_checkpoint
from training.loss import lm_loss


def test_forward_shape():
    cfg = ModelConfig(vocab_size=320, d_model=64, n_layers=2, n_heads=4, context_length=32)
    m = MyAI(cfg)
    x = torch.randint(0, 320, (2, 16))
    logits = m(x)
    assert logits.shape == (2, 16, 320), logits.shape


def test_overfit_one_batch():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=320, d_model=64, n_layers=2, n_heads=4, context_length=32)
    m = MyAI(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (4, 32))
    y = x.roll(-1, dims=1)
    l0 = lm_loss(m(x), y)["loss"].item()
    for _ in range(30):
        opt.zero_grad()
        loss = lm_loss(m(x), y)["loss"]
        loss.backward()
        opt.step()
    l1 = lm_loss(m(x), y)["loss"].item()
    assert l1 < l0, f"loss did not decrease: {l0} -> {l1}"


def test_save_load(tmp_path=None):
    import tempfile
    cfg = ModelConfig(vocab_size=320, d_model=64, n_layers=2, n_heads=4, context_length=32)
    m = MyAI(cfg)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.pt")
        save_checkpoint(p, m, step=1, tokenizer_meta={"vocab_size": 320})
        m2 = MyAI(cfg)
        load_checkpoint(p, m2, map_location="cpu")
        for a, b in zip(m.parameters(), m2.parameters()):
            assert torch.allclose(a, b)


def test_tokenizer_roundtrip():
    tok = ByteTokenizer()
    s = "hello <think>reason</think> world"
    ids = tok.encode(s, add_eos=True)
    assert tok.eos_id in ids and 260 in ids  # <think>
    back = tok.decode(ids)
    assert "hello" in back and "<think>" in back


def test_generate_valid():
    cfg = ModelConfig(vocab_size=320, d_model=64, n_layers=2, n_heads=4, context_length=32)
    m = MyAI(cfg).eval()
    tok = ByteTokenizer()
    x = torch.tensor([tok.encode("hi")])
    out = m.generate(x, max_new_tokens=10, temperature=0)
    assert out.shape[1] == x.shape[1] + 10
    assert out.max().item() < 320


def test_gqa_shapes_and_cache():
    # GQA: 4 query heads, 2 kv heads
    cfg = ModelConfig(vocab_size=320, d_model=64, n_layers=2, n_heads=4,
                      n_kv_heads=2, context_length=32)
    m = MyAI(cfg).eval()
    x = torch.randint(0, 320, (2, 16))
    logits, past = m(x, use_cache=True)
    assert logits.shape == (2, 16, 320)
    assert len(past) == 2 and past[0][0].shape == (2, 2, 16, 16)  # (B, Hkv, T, D)
    # decode 1 step with cache == full forward slice (exact match)
    nxt = torch.randint(0, 320, (2, 1))
    logits_c, _ = m(nxt, past=past, start_pos=16, use_cache=True)
    logits_full = m(torch.cat([x, nxt], dim=1))
    assert torch.allclose(logits_c, logits_full[:, -1:, :], atol=1e-5)
