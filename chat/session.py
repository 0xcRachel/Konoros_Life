"""Multi-turn session builder (pure Python, test được không cần torch)."""

SYSTEM = ("You are Konoros-Sec, a defensive security assistant. "
          "Think step by step inside <think>...</think>, then reply inside "
          "<answer>...</answer>. Only help with authorized, lawful security work.")


def build_history(system: str = SYSTEM) -> list[dict]:
    return [{"role": "system", "content": system}]


def add_turn(history: list[dict], user_text: str, assistant_text: str = "") -> list[dict]:
    history.append({"role": "user", "content": user_text})
    if assistant_text:
        history.append({"role": "assistant", "content": assistant_text})
    return history


def render_history(history: list[dict], last_n: int = 6) -> str:
    """Nối các turn gần nhất thành 1 prompt (giữ ngữ cảnh mà không tràn context)."""
    turns = [t for t in history if t["role"] in ("user", "assistant")][-last_n:]
    parts = [f"<system>{history[0]['content']}</system>"] if history else []
    for t in turns:
        tag = "<user>" if t["role"] == "user" else "<assistant>"
        close = "</user>" if t["role"] == "user" else "</assistant>"
        parts.append(f"{tag}{t['content']}{close}")
    return "".join(parts)


def parse_reply(text: str) -> dict:
    import re
    th = re.search(r"<think>(.*?)</think>", text, re.S)
    an = re.search(r"<answer>(.*?)</answer>", text, re.S)
    return {"think": th.group(1).strip() if th else "",
            "answer": an.group(1).strip() if an else text.strip()}
