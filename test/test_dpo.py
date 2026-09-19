from tokenizer import ByteTokenizer
from training.dpo import load_pairs, build_ids


def test_pairs_load():
    rows = load_pairs("data/sft/security_prefs.jsonl")
    assert len(rows) == 8
    labels = {r["label"] for r in rows}
    assert labels == {"benign", "harmful"}


def test_build_ids_masks_prompt():
    tok = ByteTokenizer()
    comp = "<think>ok</think><answer>Use WPA3.</answer>"
    ids, labels = build_ids(tok, "Secure WiFi?", comp, 512)
    assert len(ids) == len(labels)
    masked = sum(1 for l in labels if l == -100)
    assert masked > 0 and masked < len(labels)
    # unmasked tail must equal completion + eos, tags intact (no double wrap)
    assert tok.decode([i for i, l in zip(ids, labels) if l != -100]) == comp + "<eos>"


def test_harmful_pair_direction():
    rows = load_pairs("data/sft/security_prefs.jsonl")
    harm = [r for r in rows if r["label"] == "harmful"]
    assert harm and "can't help" in harm[0]["chosen"]
    assert "exploit" in harm[0]["rejected"].lower() or "crack" in harm[0]["rejected"].lower() \
        or "keylog" in harm[0]["rejected"].lower() or "registry" in harm[0]["rejected"].lower()
