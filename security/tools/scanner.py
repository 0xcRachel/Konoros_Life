"""Authorized defensive scans only. Refuses exploit flags and out-of-scope targets."""
from security.policy.scope_validator import validate_target
from security.policy.authorization import require_authorization
from security.tools.sandbox import run_tool
from security.audit.logger import audit

SAFE_NMAP_FLAGS = {"-sV", "-oA", "-F", "--top-ports"}


def safe_port_audit(target: str, proof: dict | None, human_approved: bool,
                    allowed_scope: list[str]) -> dict:
    """Example: audit open ports on your OWN host. No exploit scripts (--script) allowed."""
    validate_target(target, allowed_scope)
    require_authorization(proof, human_approved)
    audit("safe_port_audit", {"target": target, "proof": (proof or {}).get("type")})
    return run_tool("nmap", ["-sV", "-F", target], timeout=120)
