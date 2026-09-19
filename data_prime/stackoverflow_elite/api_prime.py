"""SO elite qua SE API: chỉ accepted + score cao + code xịn (GĐ1).

Chạy: python data_prime/stackoverflow_elite/api_prime.py --out data_prime/raw/so_elite.jsonl --max 300
Hỗ trợ SE_API_KEY (env) để lên 10k req/ngày.
"""
import argparse
import html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_collectors.common.guardrails import RateLimiter, redact_pii
from data_prime.common.clean_text import clean_text, within_chars
from data_prime.common.dedup import NearDupFilter, write_record, load_seen_links

try:
    import requests
except ImportError:
    requests = None

try:
    import yaml
    _CFG = yaml.safe_load(open("data_prime/configs/elite_filters.yaml", encoding="utf-8"))["stackoverflow"]
except Exception:
    _CFG = {"min_q_score": 5, "min_a_score": 3, "require_accepted": True,
            "min_code_lines": 3, "min_chars": 200, "max_chars": 6000}

API = "https://api.stackexchange.com/2.3/questions"
CODE_RE = re.compile(r"<code>(.*?)</code>", re.S)
TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"</p>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s)


def code_lines(body_html: str) -> int:
    n = 0
    for m in CODE_RE.findall(body_html or ""):
        code = html.unescape(TAG_RE.sub("", m)).strip()
        if "\n" in code or len(code) > 40:
            n += code.count("\n") + 1
    return n


def api_key() -> str | None:
    return os.environ.get("SE_API_KEY") or None


def fetch(tag, pages, pagesize, key):
    lim = RateLimiter(0.5 if key else 1.0)
    out, quota = [], None
    for page in range(1, pages + 1):
        lim.wait()
        params = {"order": "desc", "sort": "votes", "tagged": tag, "site": "stackoverflow",
                  "pagesize": pagesize, "page": page, "filter": "withbody"}
        if key:
            params["key"] = key
        r = requests.get(API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        quota = data.get("quota_remaining", quota)
        for q in data.get("items", []):
            if q.get("score", 0) < _CFG["min_q_score"] or q.get("is_answered") is not True:
                continue
            if _CFG["require_accepted"] and not q.get("accepted_answer_id"):
                continue
            out.append(q)
        if data.get("backoff"):
            import time
            time.sleep(data["backoff"] + 1)
        if quota is not None and quota < 20:
            print(f"[quota] {quota}, dừng sớm");
            break
    return out, quota


def fetch_accepted(qid, aid, key):
    lim = getattr(fetch_accepted, "_lim", None) or RateLimiter(0.5 if key else 1.0)
    fetch_accepted._lim = lim
    lim.wait()
    params = {"order": "desc", "sort": "votes", "site": "stackoverflow",
              "pagesize": 10, "filter": "withbody"}
    if key:
        params["key"] = key
    r = requests.get(f"https://api.stackexchange.com/2.3/questions/{qid}/answers",
                     params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    for a in data.get("items", []):
        if a.get("answer_id") == aid:
            return a, data.get("quota_remaining")
    return None, data.get("quota_remaining")


def main(out, max_records, tags, pages, pagesize):
    assert requests is not None, "pip install requests"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    key = api_key()
    print("SE_API_KEY:", "có (10k/ngày)" if key else "không (quota ẩn danh)")
    seen = load_seen_links(out)
    dup = NearDupFilter()
    added, stop = 0, False
    with open(out, "a", encoding="utf-8") as f:
        for tag in tags:
            if stop or added >= max_records:
                break
            try:
                qs, quota = fetch(tag, pages, pagesize, key)
            except Exception as e:
                print(f"[{tag}] lỗi: {e}");
                continue
            print(f"[{tag}] candidates={len(qs)} quota={quota}")
            for q in qs:
                if added >= max_records:
                    break
                link = q.get("link", "")
                if link in seen:
                    continue
                try:
                    a, quota = fetch_accepted(q["question_id"], q.get("accepted_answer_id"), key)
                except Exception as e:
                    print(f"  skip {q['question_id']}: {e}")
                    if "400" in str(e):
                        stop = True;
                        break
                    continue
                if not a or a.get("score", 0) < _CFG["min_a_score"]:
                    continue
                if code_lines(a.get("body", "")) < _CFG["min_code_lines"]:
                    continue
                title = html.unescape(q.get("title", ""))
                qtext = clean_text(html_to_text(q.get("body", ""))[:2500])
                atext = clean_text(html_to_text(a.get("body", ""))[:4500])
                text = f"{title}\n{qtext}\n---\n{atext}"
                text = redact_pii(text)
                if not within_chars(text, _CFG["min_chars"], _CFG["max_chars"]):
                    continue
                if dup.is_dup(text):
                    continue
                write_record(f, text, link, "cc-by-sa", "en",
                             {"tags": q.get("tags", []), "q_score": q.get("score", 0),
                              "kind": "so_elite"})
                seen.add(link)
                added += 1
                if quota is not None and quota < 20:
                    print(f"[quota] {quota}, dừng sớm");
                    stop = True;
                    break
    print(f"added {added} -> {out} (total {len(seen)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data_prime/raw/so_elite.jsonl")
    p.add_argument("--max", type=int, default=300)
    p.add_argument("--tags", default="python,javascript,linux,security,networking")
    p.add_argument("--pages", type=int, default=3)
    p.add_argument("--pagesize", type=int, default=30)
    a = p.parse_args()
    main(a.out, a.max, [t.strip() for t in a.tags.split(",") if t.strip()], a.pages, a.pagesize)
