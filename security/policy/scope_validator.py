"""Scope validator: every intrusive action needs explicit authorized scope."""
import ipaddress
import re


class ScopeError(ValueError):
    pass


def validate_target(target: str, allowed: list[str]) -> bool:
    """allowed: list of hostnames / IPs / CIDRs the operator proved ownership/consent for.
    Raises ScopeError if target not in scope. Fail-closed."""
    t = target.strip().lower()
    for scope in allowed:
        s = scope.strip().lower()
        if t == s:
            return True
        # CIDR match
        try:
            if "/" in s and _ip_in_cidr(t, s):
                return True
        except ValueError:
            pass
        # subdomain match: *.example.com
        if s.startswith("*."):
            if t.endswith(s[1:]):
                return True
    raise ScopeError(f"target '{target}' not in authorized scope {allowed}")


def _ip_in_cidr(ip: str, cidr: str) -> bool:
    return ipaddress.ip_address(ip) in ipaddress.ip_network(cidr, strict=False)


DOMAIN_RE = re.compile(r"^(?=.{1,253}$)(?!-)[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}$")


def looks_like_target(s: str) -> bool:
    try:
        ipaddress.ip_address(s)
        return True
    except ValueError:
        return bool(DOMAIN_RE.match(s.lower()))
