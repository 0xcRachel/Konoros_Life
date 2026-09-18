"""StackOverflow collector — ONLY via official Stack Exchange API / data dump.

Respects API quota, backoff, CC-BY-SA license. No scraping of user profiles,
no private data. Output: JSONL {"prompt","think","answer","source","license"}.
"""
import argparse
import html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_collectors.common.guardrails import RateLimiter, redact_pii

try:
    import requests
except ImportError:
    requests = None

API = "https://api.stackexchange.com/2.3/questions"
CODE_RE = re.compile(r"<code>(.*?)</code>", re.S)


def clean_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"</p>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()


def fetch_questions(tag="python", pagesize=20, pages=1, min_score=2):
    assert requests is not None, "pip install requests"
    lim = RateLimiter(1.0)
    out = []
    quota = None
    for page in range(1, pages + 1):
        lim.wait()
        r = requests.get(API, params={
            "order": "desc", "sort": "votes", "tagged": tag, "site": "stackoverflow",
            "pagesize": pagesize, "page": page, "filter": "withbody",
        }, timeout=30)
        r.raise_for_status()
        data = r.json()
        quota = data.get("quota_remaining", quota)
        for q in data.get("items", []):
            if q.get("score", 0) < min_score or q.get("is_answered") is not True:
                continue
            out.append(q)
        if data.get("backoff"):
            import time
            time.sleep(data["backoff"] + 1)
        if quota is not None and quota < 20:
            print(f"[quota] remaining={quota}, stopping early")
            break
    return out, quota


def fetch_answer(qid: int):
    lim = getattr(fetch_answer, "_lim", None) or RateLimiter(1.0)
    fetch_answer._lim = lim
    lim.wait()
    r = requests.get(f"https://api.stackexchange.com/2.3/questions/{qid}/answers",
                     params={"order": "desc", "sort": "votes", "site": "stackoverflow",
                             "pagesize": 1, "filter": "withbody"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    items = data.get("items", [])
    return (items[0] if items else None), data.get("quota_remaining")


def to_record(q, a) -> dict:
    title = html.unescape(q.get("title", ""))
    qtext = redact_pii(clean_html(q.get("body", ""))[:2000])
    atext = redact_pii(clean_html((a or {}).get("body", ""))[:4000])
    return {
        "prompt": f"{title}\n{qtext}",
        "think": "",
        "answer": atext,
        "source": q.get("link", ""),
        "license": "cc-by-sa",
        "tags": q.get("tags", []),
    }


def main(tag="python", pages=2, pagesize=20, min_score=2, out="data/raw/so_python.jsonl"):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    qs, quota = fetch_questions(tag, pagesize, pages, min_score)
    print(f"quota_remaining after questions: {quota}")
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for q in qs:
            try:
                a, quota = fetch_answer(q["question_id"])
            except Exception as e:
                print(f"skip {q['question_id']}: {e}")
                continue
            if not a:
                continue
            f.write(json.dumps(to_record(q, a), ensure_ascii=False) + "\n")
            n += 1
            if quota is not None and quota < 20:
                print(f"[quota] remaining={quota}, stopping early")
                break
    print(f"saved {n} records -> {out} (CC-BY-SA, attribution in 'source')")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="python")
    p.add_argument("--pages", type=int, default=2)
    p.add_argument("--pagesize", type=int, default=20)
    p.add_argument("--min-score", type=int, default=2)
    p.add_argument("--out", default="data/raw/so_python.jsonl")
    a = p.parse_args()
    main(a.tag, a.pages, a.pagesize, a.min_score, a.out)
