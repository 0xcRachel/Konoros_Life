"""Dedup + license/manifest helpers (pure Python, không cần lib ngoài)."""
import hashlib
import json
import os
import re

WORD_RE = re.compile(r"\w+", re.U)


def sha_key(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def shingles(s: str, k: int = 5) -> set:
    w = WORD_RE.findall(s.lower())
    if len(w) < k:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class NearDupFilter:
    """Giữ tối đa N shingle-set gần nhất trong RAM; đủ cho ~100MB."""

    def __init__(self, threshold: float = 0.85, keep: int = 20000):
        self.threshold = threshold
        self.keep = keep
        self.seen: list[set] = []

    def is_dup(self, s: str) -> bool:
        sh = shingles(s)
        for prev in self.seen[-self.keep:]:
            if jaccard(sh, prev) >= self.threshold:
                return True
        self.seen.append(sh)
        if len(self.seen) > self.keep:
            self.seen = self.seen[-self.keep:]
        return False


def write_record(f, text: str, source: str, license: str, lang: str, extra: dict | None = None):
    rec = {"text": text, "source": source, "license": license, "lang": lang}
    if extra:
        rec.update(extra)
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_seen_links(path: str) -> set:
    seen = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line).get("source", ""))
                except Exception:
                    pass
    return seen
