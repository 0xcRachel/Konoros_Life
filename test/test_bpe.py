import pytest

from tokenizer.bpe import BPETokenizer

tokenizers = pytest.importorskip("tokenizers", reason="pip install tokenizers (Colab)")


def test_bpe_roundtrip_tmp(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    corpus = tmp_path / "c.txt"
    corpus.write_text("def hello(): print('hi')\n" * 20, encoding="utf-8")
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    tok.train([str(corpus)], BpeTrainer(vocab_size=200, min_frequency=1,
                                        special_tokens=BPETokenizer.special_list()))
    fp = str(tmp_path / "bpe.json")
    tok.save(fp)
    b = BPETokenizer.load(fp)
    ids = b.encode("def hello():", add_bos=True, add_eos=True)
    assert ids[0] == b.bos_id and ids[-1] == b.eos_id
    assert "def" in b.decode(ids)
    padded, mask = b.encode_batch(["hi", "hello world"], add_eos=True)
    assert len(padded[0]) == len(padded[1]) and sum(mask[0]) < sum(mask[1])


def test_special_list_reserved():
    assert len(BPETokenizer.special_list()) == 64  # 14 specials + 50 reserved
