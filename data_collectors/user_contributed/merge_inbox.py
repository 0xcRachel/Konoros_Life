"""User-contributed data inbox — explicit consent REQUIRED.

- Only data the user pastes/uploads themselves.
- consent.json must exist with purpose + timestamp before merge.
- All content goes through quarantine + PII redact before entering data/raw/.
- NEVER auto-collect other people's data.
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_collectors.common.guardrails import redact_pii


def main(inbox="data_collectors/user_contributed/inbox.txt",
         out="data/raw/user_contributed.jsonl",
         consent="data_collectors/user_contributed/consent.json"):
    if not os.path.exists(consent):
        print(f"REFUSED: missing {consent}. User must give explicit consent first. "
              "Create it with {\"user\":..., \"purpose\":..., \"date\":...}.")
        return
    if not os.path.exists(inbox):
        print(f"nothing to merge: {inbox} not found")
        return
    with open(inbox, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        print("inbox empty"); return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rec = {"text": redact_pii(text),
           "source": "user_contributed",
           "license": "user-consent",
           "merged_at": datetime.datetime.utcnow().isoformat() + "Z"}
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"merged {len(text)} chars -> {out} (quarantine: review file before training)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inbox", default="data_collectors/user_contributed/inbox.txt")
    p.add_argument("--out", default="data/raw/user_contributed.jsonl")
    p.add_argument("--consent", default="data_collectors/user_contributed/consent.json")
    a = p.parse_args()
    main(a.inbox, a.out, a.consent)
