"""SO dump full GĐ2 (chạy trên Colab): stream Posts.xml.7z, chỉ giữ elite.

Chạy trên Colab:
  !apt -qq install -y p7zip-full
  !wget -O /tmp/so_posts.7z https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z
  !python data_prime/stackoverflow_elite/dump_prime.py --in /tmp/so_posts.7z --out data_prime/raw/so_dump_elite.jsonl --max 200000
Yêu cầu: pip install py7zr lxml (script dùng py7zr để stream, lxml parse nhanh).
"""
import argparse
import html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_collectors.common.guardrails import redact_pii
from data_prime.common.clean_text import clean_text, within_chars
from data_prime.common.dedup import NearDupFilter, write_record, load_seen_links

CODE_RE = re.compile(r"<code>(.*?)</code>", re.S)
TAG_RE = re.compile(r"<[^>]+>")


def strip(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s or "")
    s = re.sub(r"</p>", "\n", s)
    return html.unescape(TAG_RE.sub("", s)).strip()


def code_lines(body: str) -> int:
    n = 0
    for m in CODE_RE.findall(body or ""):
        code = html.unescape(TAG_RE.sub("", m)).strip()
        if "\n" in code or len(code) > 40:
            n += code.count("\n") + 1
    return n


def main(inp, out, max_records, min_q, min_a, min_code):
    from py7zr import SevenZipFile  # pip install py7zr
    from lxml import etree  # pip install lxml
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    seen = load_seen_links(out)
    dup = NearDupFilter()
    added = 0
    by_id: dict[str, dict] = {}
    with SevenZipFile(inp, "r") as z, open(out, "a", encoding="utf-8") as f:
        names = z.getnames()
        xml_name = next(n for n in names if n.endswith(".xml"))
        with z.read([xml_name])[xml_name] as fh:
            ctx = etree.iterparse(fh, events=("end",), tag="row", huge_tree=True)
            for _, el in ctx:
                if added >= max_records:
                    break
                try:
                    pt = el.get("PostTypeId")  # 1=question, 2=answer
                    if pt == "1":
                        if int(el.get("Score", "0")) >= min_q:
                            by_id[el.get("Id")] = {
                                "title": html.unescape(el.get("Title", "")),
                                "body": el.get("Body", ""),
                                "aid": el.get("AcceptedAnswerId"),
                                "score": int(el.get("Score", "0")),
                                "tags": el.get("Tags", ""),
                            }
                    elif pt == "2":
                        q = by_id.get(el.get("ParentId"))
                        if (q and q["aid"] == el.get("Id")
                                and int(el.get("Score", "0")) >= min_a
                                and code_lines(el.get("Body", "")) >= min_code):
                            link = f"https://stackoverflow.com/questions/{el.get('ParentId')}"
                            if link not in seen:
                                text = clean_text(
                                    f"{q['title']}\n{strip(q['body'])[:2500]}\n---\n"
                                    f"{strip(el.get('Body', ''))[:4500]}")
                                text = redact_pii(text)
                                if within_chars(text, 200, 6000) and not dup.is_dup(text):
                                    write_record(f, text, link, "cc-by-sa", "en",
                                                 {"q_score": q["score"], "kind": "so_dump_elite"})
                                    seen.add(link)
                                    added += 1
                                    if added % 1000 == 0:
                                        print(f"  ...{added}")
                            by_id.pop(el.get("ParentId"), None)
                finally:
                    el.clear()
    print(f"added {added} -> {out} (total {len(seen)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", default="data_prime/raw/so_dump_elite.jsonl")
    p.add_argument("--max", type=int, default=200000)
    p.add_argument("--min-q", type=int, default=5)
    p.add_argument("--min-a", type=int, default=3)
    p.add_argument("--min-code", type=int, default=3)
    a = p.parse_args()
    main(a.inp, a.out, a.max, a.min_q, a.min_a, a.min_code)
