"""Append-only audit log (JSONL). Every intrusive action must call audit()."""
import json
import os
import datetime

LOG_PATH = os.environ.get("KONOROS_AUDIT_LOG", "experiments/audit.jsonl")


def audit(action: str, details: dict) -> dict:
    rec = {"ts": datetime.datetime.utcnow().isoformat() + "Z",
           "action": action, "details": details}
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
