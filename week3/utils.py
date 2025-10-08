"""
Utility functions for grading week 3 machine learning exercises.
"""
import torch
import torch.nn.functional as F

def show_result(name: str, res: dict):
    status = "PASS" if res.get("passed") else "FAIL"
    msg = res.get("message", "")
    print(f"[{status}] {name}" + (f" | {msg}" if msg else ""))

def test_exercise_2(student_function):
    x = torch.randn(3,)
    w1 = torch.randn(2, 3)
    b1 = torch.randn(2, )
    w2 = torch.randn(4, 2)
    b2 = torch.randn(4,)
    z1 = w1 @ x + b1
    h1 = F.relu(z1)
    z2 = w2 @ h1 + b2
    y = F.relu(z2)
    try:
        answer = y
        out = student_function(x, w1, w2, b1, b2)
    except Exception as e:
        return {"passed": False, "message": f"runtime error: {e}"}
    ok = torch.allclose(out, answer, atol=1e-6, rtol=0.0)
    if not ok:
        return {"passed": False, "message": "mismatch between student answer and right answer"}
    return {"passed": True}

