"""Public OSINT only: allowlisted docs sources, robots.txt respected, no personal data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_collectors.common.guardrails import can_fetch, redact_pii


def public_fetch(url: str, allowlist=("docs.python.org", "pytorch.org")) -> dict:
    from urllib.parse import urlparse
    if urlparse(url).netloc not in allowlist:
        return {"ok": False, "error": "domain not allowlisted"}
    if not can_fetch(url):
        return {"ok": False, "error": "robots.txt disallows"}
    try:
        import requests
        r = requests.get(url, timeout=20, headers={"User-Agent": "Konoros-Sec-DataBot/1.0"})
        return {"ok": r.ok, "text": redact_pii(r.text[:8000]) if r.ok else ""}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
