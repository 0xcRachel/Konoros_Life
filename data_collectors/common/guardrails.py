"""Shared HTTP + guardrails. All collectors MUST respect robots.txt / ToS / licenses."""
import re
import time
import hashlib
import urllib.robotparser as robotparser
from urllib.parse import urlparse

ALLOWED_LICENSES = {"cc-by-sa", "mit", "apache-2.0", "bsd", "public-domain", "cc0", "cc-by"}

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s\-.]?){7,15}(?!\d)")
SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|secret|passwd|password|aws_|github_pat|ghp_|xox[bap]-)[\s:=]+[^\s'\"]{4,}"
)


def can_fetch(url: str, user_agent: str = "Konoros-Sec-DataBot/1.0") -> bool:
    """Check robots.txt before any web fetch. Fail-closed."""
    try:
        parts = urlparse(url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return False


def redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = SECRET_RE.sub("[REDACTED_SECRET]", text)
    return text


def license_ok(lic: str | None) -> bool:
    if not lic:
        return False
    return lic.strip().lower() in ALLOWED_LICENSES


def sha_dedup_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RateLimiter:
    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self):
        dt = time.time() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.time()
