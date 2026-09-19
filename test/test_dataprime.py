from data_prime.common.clean_text import clean_text, repeat_line_ratio, looks_boilerplate
from data_prime.common.dedup import sha_key, shingles, jaccard, NearDupFilter


def test_clean_controls_and_dup_lines():
    s = clean_text("hello\x00  world\nhello world\n\n\nnext")
    assert "\x00" not in s and "hello world" in s
    assert s.count("hello world") == 1  # dòng lặp liên tiếp bị gộp


def test_repeat_ratio():
    assert repeat_line_ratio("a\na\na\nb") > 0.3
    assert looks_boilerplate("a\na\na\nb")
    assert not looks_boilerplate("a\nb\nc\nd")


def test_dedup_exact_and_near():
    f = NearDupFilter(threshold=0.85)
    a = "the quick brown fox jumps over the lazy dog today"
    assert not f.is_dup(a)
    assert f.is_dup(a)  # exact
    b = "the quick brown fox jumps over the lazy dog today!"  # near
    assert f.is_dup(b)
    assert not f.is_dup("completely different content about wifi security hardening guide")


def test_sha_and_jaccard():
    assert sha_key("x") != sha_key("y")
    s = shingles("a b c d e f")
    assert jaccard(s, s) == 1.0
    assert 0.0 <= jaccard(s, {"zzz"}) <= 1.0
