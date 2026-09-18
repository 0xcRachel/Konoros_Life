"""Authorization gate: intrusive tools require proof + human approval. No auto-exploit."""
import datetime


ALLOWED_PROOF_TYPES = {"written-consent", "ownership-ticket", "bug-bounty-scope-url"}


class AuthorizationError(PermissionError):
    pass


def require_authorization(proof: dict | None, human_approved: bool) -> dict:
    """proof example: {"type": "written-consent", "scope": ["127.0.0.1"], "ref": "ticket-123"}"""
    if not proof or proof.get("type") not in ALLOWED_PROOF_TYPES:
        raise AuthorizationError(f"missing/invalid authorization proof. Need one of {ALLOWED_PROOF_TYPES}")
    if not proof.get("scope"):
        raise AuthorizationError("authorization proof has empty scope")
    if not human_approved:
        raise AuthorizationError("human approval required for intrusive action")
    return {"ok": True, "checked_at": datetime.datetime.utcnow().isoformat() + "Z",
            "type": proof["type"], "scope": proof["scope"]}
