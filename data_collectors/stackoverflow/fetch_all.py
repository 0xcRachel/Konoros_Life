"""Batch fetch nhiều tag Stack Overflow -> 1 file JSONL, dedup theo link.

Chạy: python data_collectors/stackoverflow/fetch_all.py --out data/raw/so_all.jsonl --max-answers 150
- Chỉ dùng Stack Exchange API chính thức (CC-BY-SA, giữ attribution).
- Quota-aware: API ẩn danh ~300 req/ngày/IP -> tự dừng khi quota < 20.
- Muốn lấy hàng chục nghìn bản ghi: đăng ký key tại stackapps.com (miễn phí,
  quota 10k/ngày) rồi chạy lại trên Colab với --pages lớn hơn.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_collectors.stackoverflow.collect_so import fetch_questions, fetch_answer, to_record

PLAN = [
    # (tag, pages, pagesize, min_score)
    ("python", 3, 30, 3),
    ("javascript", 2, 20, 3),
    ("linux", 2, 20, 2),
    ("security", 2, 20, 2),
    ("networking", 1, 20, 2),
    ("encryption", 1, 20, 2),
]


def load_existing(out: str) -> set:
    seen = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line).get("source", ""))
                except Exception:
                    pass
    return seen


def main(out: str, max_answers: int, only_tags: str | None = None):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    seen = load_existing(out)
    print(f"existing records: {len(seen)}")
    plan = PLAN
    if only_tags:
        want = {t.strip() for t in only_tags.split(",")}
        plan = [row for row in PLAN if row[0] in want]
    added, stop = 0, False
    with open(out, "a", encoding="utf-8") as f:
        for tag, pages, pagesize, min_score in plan:
            if stop or added >= max_answers:
                break
            try:
                qs, quota = fetch_questions(tag, pagesize, pages, min_score)
            except Exception as e:
                print(f"[{tag}] question fetch failed: {e}")
                continue
            print(f"[{tag}] candidates={len(qs)} quota={quota}")
            for q in qs:
                if added >= max_answers:
                    break
                link = q.get("link", "")
                if link in seen:
                    continue
                try:
                    a, quota = fetch_answer(q["question_id"])
                except Exception as e:
                    print(f"  skip {q['question_id']}: {e}")
                    if "400" in str(e) or "quota" in str(e).lower():
                        stop = True
                        break
                    continue
                if not a:
                    continue
                rec = to_record(q, a)
                rec["fetch_tag"] = tag
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                seen.add(link)
                added += 1
                if quota is not None and quota < 20:
                    print(f"[quota] remaining={quota}, stopping")
                    stop = True
                    break
    print(f"added {added} records -> {out} (total now {len(seen)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/so_all.jsonl")
    p.add_argument("--max-answers", type=int, default=150)
    p.add_argument("--tags", default=None, help="comma-separated subset, e.g. security,networking")
    a = p.parse_args()
    main(a.out, a.max_answers, a.tags)
