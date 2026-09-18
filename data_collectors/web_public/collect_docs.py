"""Public-web collector stub — allowlist only, robots.txt enforced.

v1.1: only fetches official docs (python.org, pytorch.org) explicitly allowed.
No general crawling, no user data.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_collectors.common.guardrails import can_fetch, redact_pii, RateLimiter

ALLOWLIST = ["docs.python.org", "pytorch.org", "developer.mozilla.org"]

SEED_URLS = [
    "https://docs.python.org/3/tutorial/index.html",
    "https://docs.python.org/3/library/secrets.html",
    "https://docs.python.org/3/library/hashlib.html",
    "https://developer.mozilla.org/en-US/docs/Web/Security",
]


def main(out="data/raw/web_docs.txt"):
    try:
        import requests
    except ImportError:
        print("pip install requests"); return
    from urllib.parse import urlparse
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    lim = RateLimiter(2.0)
    kept = []
    for url in SEED_URLS:
        host = urlparse(url).netloc
        if host not in ALLOWLIST:
            print(f"skip {url}: not allowlisted"); continue
        if not can_fetch(url):
            print(f"skip {url}: robots.txt disallows"); continue
        lim.wait()
        r = requests.get(url, timeout=30, headers={"User-Agent": "Konoros-Sec-DataBot/1.0"})
        if r.ok:
            import re
            text = re.sub(r"<[^>]+>", " ", r.text)
            kept.append(redact_pii(text[:20000]))
            print(f"ok {url} ({len(text)} chars)")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(kept))
    print(f"saved -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/web_docs.txt")
    main(p.parse_args().out)
