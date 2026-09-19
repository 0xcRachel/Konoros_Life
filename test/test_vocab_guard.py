from model.config import ModelConfig
from tokenizer import ByteTokenizer
from training.checkpoint import check_vocab


def test_check_vocab_ok(tmp_path):
    import numpy as np
    cfg = ModelConfig(vocab_size=320, d_model=64, n_layers=2, n_heads=4, context_length=32)
    tok = ByteTokenizer()
    assert tok.vocab_size == 320
    check_vocab(tok, cfg)  # no bin -> only tok/model check
    bp = tmp_path / "t.bin"
    np.array([0, 10, 319], dtype=np.uint16).tofile(str(bp))
    check_vocab(tok, cfg, str(bp))  # max_id 319 < 320 ok


def test_check_vocab_bin_overflow(tmp_path):
    import numpy as np
    cfg = ModelConfig(vocab_size=320, d_model=64, n_layers=2, n_heads=4, context_length=32)
    tok = ByteTokenizer()
    bp = tmp_path / "bad.bin"
    # mô phỏng bins BPE (id tới 11381) đem train model vocab 320 -> phải FATAL sớm
    np.array([0, 500, 11381], dtype=np.uint16).tofile(str(bp))
    try:
        check_vocab(tok, cfg, str(bp))
        assert False, "should SystemExit"
    except SystemExit as e:
        assert "max id" in str(e)


def test_check_vocab_tok_mismatch():
    cfg = ModelConfig(vocab_size=320, d_model=64, n_layers=2, n_heads=4, context_length=32)

    class FakeTok:
        vocab_size = 11381

    try:
        check_vocab(FakeTok(), cfg)
        assert False, "should SystemExit"
    except SystemExit as e:
        assert "tokenizer vocab" in str(e)
