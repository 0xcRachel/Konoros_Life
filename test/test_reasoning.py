from training.rewards import (
    parse_reasoning, is_refusal, format_reward, safety_reward,
    think_shape_reward, fluency_reward, composite_reward,
)


def test_parse_ok():
    p = parse_reasoning("<think>step 1</think><answer>done</answer>")
    assert p["has_think"] and p["has_answer"]
    assert p["think"] == "step 1"


def test_parse_missing():
    p = parse_reasoning("just plain text")
    assert not p["has_think"] and not p["has_answer"]
    assert format_reward(p) == 0.0


def test_harmful_refusal_rewarded():
    r = composite_reward("<think>denied request</think><answer>I can't help with that.</answer>", "harmful")
    assert r["safety"] == 1.0 and r["total"] > 0.8


def test_harmful_compliance_penalized():
    r = composite_reward("<think>ok</think><answer>Here is the exploit code...</answer>", "harmful")
    assert r["safety"] == 0.0


def test_benign_helpful_rewarded():
    r = composite_reward(
        "<think>defensive question</think><answer>Change the default admin password and use WPA3 with a long passphrase.</answer>",
        "benign")
    assert r["safety"] == 1.0 and r["format"] == 1.0


def test_benign_refusal_penalized():
    r = composite_reward("<think>hmm</think><answer>I can't help with that.</answer>", "benign")
    assert r["safety"] == 0.0


def test_think_shape():
    assert think_shape_reward({"think": ""}) == 0.0
    assert think_shape_reward({"think": "a" * 100}) == 1.0
    assert is_refusal("I cannot help with that request")
    assert not is_refusal("Here is how to enable MFA")


def test_fluency_natural_vs_gibberish():
    assert fluency_reward("Change the default admin password and use WPA3.") == 1.0
    assert fluency_reward("I can't help with that.") == 1.0
    assert fluency_reward("tly: trljsstjsT foljsl-ps.") == 0.0
    # chữ rác từ log Colab thật (vowel-ratio cũ chấm nhầm 1.0/0.5)
    assert fluency_reward("nesstUnx tillbs:dllindlex f. rejs rp") == 0.0
    assert fluency_reward("till-p-pip, repjlintex lly-pt-plist1 ex lotstplbex") == 0.0
    assert fluency_reward("") == 0.0
    # chữ rác của model học vẹt không được PASS benign
    r = composite_reward("<answer>nesstUnx tillbs:dllindlex f. rejs rp</answer>", "benign")
    assert not (r["safety"] == 1.0 and r["fluency"] >= 0.5)
