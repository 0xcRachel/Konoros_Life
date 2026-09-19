"""Chat template for Konoros-Sec SFT / reasoning.

Format: <system>...</system><user>...</user><think>...</think><answer>...</answer>
v0.1-v1.0: think may be empty. v1.1 SFT trains on defensive security pairs.
"""
from tokenizer import ByteTokenizer

SYSTEM = ("You are Konoros-Sec, a defensive security assistant. "
          "Only help with authorized, lawful security work. "
          "Refuse exploit/persistence/credential-theft/evasion requests.")


def render_sft(prompt: str, think: str = "", answer: str = "",
               system: str = SYSTEM) -> str:
    return (f"<system>{system}</system>"
            f"<user>{prompt}</user>"
            f"<think>{think}</think>"
            f"<answer>{answer}</answer>")


def render_pref(prompt: str, completion: str, system: str = SYSTEM) -> str:
    """Preference format: completion carries its own <think>/<answer> tags."""
    return (f"<system>{system}</system>"
            f"<user>{prompt}</user>"
            f"{completion}")


def prompt_prefix(prompt: str, system: str = SYSTEM) -> str:
    return f"<system>{system}</system><user>{prompt}</user>"


def encode_sft(tok: ByteTokenizer, prompt: str, think: str, answer: str,
               context_length: int = 512):
    full = render_sft(prompt, think, answer)
    ids = tok.encode(full, add_bos=True, add_eos=True)[:context_length]
    # Mask prompt part: find where <answer> starts, set labels -100 before it
    answer_ids = tok.encode("<answer>", add_bos=False)
    labels = list(ids)
    # locate answer start index
    start = len(ids)
    for i in range(len(ids) - len(answer_ids) + 1):
        if ids[i:i + len(answer_ids)] == answer_ids:
            start = i + len(answer_ids)
            break
    for i in range(min(start, len(labels))):
        labels[i] = -100
    return ids, labels
