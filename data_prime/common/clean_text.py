"""Làm sạch text chung cho mọi nguồn elite (pure Python)."""
import re
import unicodedata

CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTIBLANK_RE = re.compile(r"\n{3,}")
SPACE_RE = re.compile(r"[ \t]+")


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = CTRL_RE.sub("", s)
    s = SPACE_RE.sub(" ", s)
    # bỏ dòng lặp liên tiếp (boilerplate)
    lines = [ln.strip() for ln in s.split("\n")]
    kept: list[str] = []
    for ln in lines:
        if kept and ln and ln == kept[-1]:
            continue
        kept.append(ln)
    s = "\n".join(kept)
    s = MULTIBLANK_RE.sub("\n\n", s)
    return s.strip()


def repeat_line_ratio(s: str) -> float:
    lines = [ln for ln in s.split("\n") if ln.strip()]
    if not lines:
        return 1.0
    return 1.0 - len(set(lines)) / len(lines)


def looks_boilerplate(s: str, max_repeat_ratio: float = 0.3) -> bool:
    return repeat_line_ratio(s) > max_repeat_ratio


def within_chars(s: str, lo: int, hi: int) -> bool:
    return lo <= len(s) <= hi
