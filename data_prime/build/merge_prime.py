"""Gộp SO elite + Wiki elite -> prime_all.jsonl + stats.

Chạy: python data_prime/build/merge_prime.py --out data/raw/prime_all.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_prime.common.dedup import NearDupFilter


def main(inputs: list[str], out: str):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    dup = NearDupFilter(threshold=0.9)
    n_in, n_out = 0, 0
    kinds: dict[str, int] = {}
    langs: dict[str, int] = {}
    with open(out, "w", encoding="utf-8") as fo:
        for path in inputs:
            if not os.path.exists(path):
                print(f"skip (missing): {path}");
                continue
            with open(path, encoding="utf-8") as fi:
                for line in fi:
                    line = line.strip()
                    if not line:
                        continue
                    n_in += 1
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    # chuẩn hoá 2 schema: {"text",...} hoặc {"prompt","answer",...}
                    if "text" not in r:
                        parts = [r.get("prompt", ""), r.get("answer", r.get("completion", ""))]
                        r["text"] = "\n".join(p for p in parts if p)
                    if len(r.get("text", "")) < 50:
                        continue
                    if dup.is_dup(r["text"]):
                        continue
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_out += 1
                    kinds[r.get("kind", "?")] = kinds.get(r.get("kind", "?"), 0) + 1
                    langs[r.get("lang", "?")] = langs.get(r.get("lang", "?"), 0) + 1
    size_mb = os.path.getsize(out) / 1e6 if os.path.exists(out) else 0
    print(f"in={n_in} out={n_out} size={size_mb:.1f}MB -> {out}")
    print("kinds:", kinds)
    print("langs:", langs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", default="data_prime/raw/so_elite.jsonl,data_prime/raw/wiki_en.jsonl,data_prime/raw/wiki_vi.jsonl")
    p.add_argument("--out", default="data/raw/prime_all.jsonl")
    a = p.parse_args()
    main([s.strip() for s in a.inputs.split(",") if s.strip()], a.out)
