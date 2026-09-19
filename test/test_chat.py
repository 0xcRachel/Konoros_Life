from chat.session import build_history, add_turn, render_history, parse_reply


def test_history_roundtrip():
    h = build_history()
    add_turn(h, "Secure WiFi?", "<think>t</think><answer>Use WPA3.</answer>")
    add_turn(h, "And camera?")
    s = render_history(h)
    assert "<system>" in s and "<user>Secure WiFi?</user>" in s
    assert s.count("<user>") == 2  # giữ ngữ cảnh lượt trước


def test_history_window():
    h = build_history()
    for i in range(10):
        add_turn(h, f"q{i}", f"a{i}")
    s = render_history(h, last_n=4)
    assert "q0" not in s and "q9" in s  # chỉ giữ 4 turn gần nhất


def test_parse_reply():
    r = parse_reply("<think>step</think><answer>done</answer>")
    assert r == {"think": "step", "answer": "done"}
    r2 = parse_reply("plain text")
    assert r2["answer"] == "plain text" and r2["think"] == ""
