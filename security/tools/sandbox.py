"""Sandbox: whitelist + no-shell + timeout. Tools run isolated, fail-closed."""
import subprocess

ALLOWED_BINARIES = {"nmap", "nuclei", "trivy"}


def run_tool(binary: str, args: list[str], timeout: int = 60) -> dict:
    if binary not in ALLOWED_BINARIES:
        return {"ok": False, "error": f"binary '{binary}' not whitelisted"}
    if any(a.strip().startswith("-") is False and ";" in a or "&&" in a or "|" in a for a in args):
        return {"ok": False, "error": "shell metacharacters denied"}
    try:
        p = subprocess.run([binary, *args], capture_output=True, text=True, timeout=timeout, shell=False)
        out = (p.stdout or "")[-8000:]
        return {"ok": p.returncode == 0, "code": p.returncode, "output": out,
                "stderr": (p.stderr or "")[-2000:]}
    except FileNotFoundError:
        return {"ok": False, "error": f"{binary} not installed"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout (kill-switch)"}
