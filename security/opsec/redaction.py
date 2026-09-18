"""Defensive OPSEC: redact secrets/PII before logging or storing."""
import re

EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
SECRET = re.compile(r"(?i)(api[_-]?key|secret|password|passwd|token)[\s:=]+[^\s'\"]+")


def redact(text: str) -> str:
    text = EMAIL.sub("[REDACTED_EMAIL]", text)
    text = SECRET.sub("[REDACTED_SECRET]", text)
    return text
