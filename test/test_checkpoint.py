import os
import time

from training.checkpoint import resolve_ckpt


def test_resolve_existing(tmp_path):
    p = tmp_path / "c.pt"
    p.write_bytes(b"x")
    assert resolve_ckpt(str(p)) == str(p)


def test_resolve_fallback_newest(tmp_path, monkeypatch):
    d = tmp_path / "experiments" / "v0.1" / "checkpoints"
    d.mkdir(parents=True)
    a = d / "step_000001.pt"
    b = d / "step_000002.pt"
    a.write_bytes(b"a")
    time.sleep(0.02)
    b.write_bytes(b"b")
    monkeypatch.chdir(tmp_path)
    got = resolve_ckpt("experiments/v1.1/sft.pt")
    assert os.path.abspath(got) == str(b)


def test_resolve_no_candidate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    try:
        resolve_ckpt("nope.pt")
        assert False, "should raise"
    except FileNotFoundError as e:
        assert "experiments" in str(e)


def test_resolve_empty_string_fallback(tmp_path, monkeypatch):
    d = tmp_path / "experiments" / "x"
    d.mkdir(parents=True)
    (d / "last.pt").write_bytes(b"z")
    monkeypatch.chdir(tmp_path)
    assert resolve_ckpt("").endswith("last.pt")
    assert os.path.exists(resolve_ckpt(""))
