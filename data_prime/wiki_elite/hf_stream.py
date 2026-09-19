"""Wiki elite qua HuggingFace datasets (streaming, không tải full).

Chạy: pip install datasets
  python data_prime/wiki_elite/hf_stream.py --out-en data_prime/raw/wiki_en.jsonl --max-en 8000
EN: giữ bài dài + có References (proxy featured/good, rẻ hơn parse category).
VI: giữ bài dài + có section (bỏ stub bot).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_prime.common.clean_text import clean_text, within_chars, looks_boilerplate
from data_prime.common.dedup import NearDupFilter, write_record, load_seen_links

try:
    import yaml
    _CFG = yaml.safe_load(open("data_prime/configs/elite_filters.yaml", encoding="utf-8"))["wikipedia"]
except Exception:
    _CFG = {"en_min_chars": 2000, "vi_min_chars": 800}

DROP = ("references", "see also", "external links", "notes",
        "chú thích", "tham khảo", "xem thêm", "liên kết ngoài")


def clean_wiki(text: str) -> str:
    # cắt từ section tham khảo trở đi
    lines = text.split("\n")
    keep = []
    for ln in lines:
        if ln.strip().lower().rstrip("=") .strip() in DROP and len(ln.strip()) < 40:
            break
        keep.append(ln)
    text = "\n".join(keep)
    text = re.sub(r"\{\{[^}]*\}\}", " ", text)  # infobox/templates
    text = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", text)  # wikilinks
    text = re.sub(r"={2,}([^=]+)={2,}", r"\1", text)  # headings
    return clean_text(text)


def stream_cfg(cfg: str, max_n: int, min_chars: int, out: str, lang: str):
    from datasets import load_dataset  # pip install datasets
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    seen = load_seen_links(out)
    dup = NearDupFilter()
    added = 0
    ds = load_dataset("wikipedia", cfg, split="train", streaming=True, trust_remote_code=False)
    with open(out, "a", encoding="utf-8") as f:
        for row in ds:
            if added >= max_n:
                break
            title = (row.get("title") or "").strip()
            body = clean_wiki(row.get("text") or "")
            if not within_chars(body, min_chars, 20000) or looks_boilerplate(body):
                continue
            link = row.get("url") or f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
            if link in seen:
                continue
            if dup.is_dup(body):
                continue
            text = f"{title}\n{body}"
            write_record(f, text, link, "cc-by-sa", lang, {"title": title, "kind": "wiki_elite"})
            seen.add(link)
            added += 1
            if added % 500 == 0:
                print(f"  [{lang}] ...{added}")
    print(f"[{lang}] added {added} -> {out} (total {len(seen)})")


def main(out_en, max_en, out_vi, max_vi):
    stream_cfg("20230301.en", max_en, _CFG["en_min_chars"], out_en, "en")
    stream_cfg("20230301.vi", max_vi, _CFG["vi_min_chars"], out_vi, "vi")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-en", default="data_prime/raw/wiki_en.jsonl")
    p.add_argument("--max-en", type=int, default=8000)
    p.add_argument("--out-vi", default="data_prime/raw/wiki_vi.jsonl")
    p.add_argument("--max-vi", type=int, default=4000)
    a = p.parse_args()
    main(a.out_en, a.max_en, a.out_vi, a.max_vi)
