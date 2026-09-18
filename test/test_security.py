import pytest
from security.policy.scope_validator import validate_target, ScopeError
from security.policy.authorization import require_authorization, AuthorizationError
from security.opsec.redaction import redact


def test_scope_allows_own_host():
    assert validate_target("127.0.0.1", ["127.0.0.1"])


def test_scope_denies_other():
    try:
        validate_target("8.8.8.8", ["127.0.0.1"])
        assert False, "should have raised"
    except ScopeError:
        pass


def test_auth_requires_human():
    proof = {"type": "written-consent", "scope": ["127.0.0.1"]}
    try:
        require_authorization(proof, False)
        assert False
    except AuthorizationError:
        pass
    assert require_authorization(proof, True)["ok"]


def test_redaction():
    assert "[REDACTED_EMAIL]" in redact("contact me at a@b.com, password= secret123")
