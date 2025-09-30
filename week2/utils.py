"""
Utility functions for grading week 2 machine learning exercises.
"""
import torch
import torch.nn.functional as F

def show_result(name: str, res: dict):
    status = "✅ PASS" if res.get("passed") else "❌ FAIL"
    msg = res.get("message", "")
    print(f"[{status}] {name}" + (f" | {msg}" if msg else ""))

def grade_sigmoid(student_fn):
    """Compare student sigmoid against torch.sigmoid on a range of logits."""
    try:
        z = torch.linspace(-8, 8, steps=401)
        ref = torch.sigmoid(z)
        out = student_fn(z.clone())
    except Exception as e:
        return {"passed": False, "message": f"runtime error: {e}"}

    if out.shape != ref.shape:
        return {"passed": False, "message": f"shape mismatch: expected {tuple(ref.shape)}, got {tuple(out.shape)}"}

    if not torch.isfinite(out).all():
        return {"passed": False, "message": "non-finite values detected"}

    ok = torch.allclose(out, ref, atol=1e-6, rtol=0.0)
    if not ok:
        diff = (out - ref).abs().max().item()
        return {"passed": False, "message": f"max abs diff {diff:.3e} exceeds tolerance"}
    return {"passed": True, "message": ""}


def grade_bce(student_fn):
    """Compare student BCE(probs, y) against F.binary_cross_entropy."""
    try:
        torch.manual_seed(0)
        y = torch.randint(0, 2, (256, 1)).float()
        logits = torch.randn(256, 1)
        probs = torch.sigmoid(logits)
        loss = student_fn(probs, y)
    except Exception as e:
        return {"passed": False, "message": f"runtime error: {e}"}

    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        return {"passed": False, "message": "loss must be a scalar tensor"}

    ref = F.binary_cross_entropy(probs, y)
    ok = torch.allclose(loss, ref, atol=1e-6, rtol=0.0)
    if not ok:
        return {"passed": False, "message": f"mismatch vs reference"}
    return {"passed": True, "message": ""}

def _row_set(X: torch.Tensor):
    """Create a set of tuples for simple disjointness checks."""
    return set(map(tuple, X.detach().cpu().numpy().round(6)))  # rounding keeps it stable enough for this synthetic data

def grade_split(split_fn, X, y, ratios=(0.7, 0.15, 0.15), seed=123):
    """Checks sizes, reproducibility, basic disjointness, and shapes."""
    try:
        (Xtr, ytr), (Xva, yva), (Xte, yte) = split_fn(X, y, ratios=ratios, seed=seed)
    except Exception as e:
        return {"passed": False, "message": f"runtime error: {e}"}

    n = len(X)
    if len(Xtr) + len(Xva) + len(Xte) != n:
        return {"passed": False, "message": "sizes do not sum to N"}

    # Shapes and contiguity
    if Xtr.shape[1] != X.shape[1] or ytr.shape != (len(Xtr), y.shape[1]):
        return {"passed": False, "message": "shape mismatch in train split"}
    if not (Xtr.is_contiguous() and Xva.is_contiguous() and Xte.is_contiguous()):
        return {"passed": False, "message": "tensors must be contiguous"}

    # Reproducibility with same seed
    (Xtr2, ytr2), (Xva2, yva2), (Xte2, yte2) = split_fn(X, y, ratios=ratios, seed=seed)
    same = torch.allclose(Xtr, Xtr2) and torch.allclose(Xva, Xva2) and torch.allclose(Xte, Xte2)
    if not same:
        return {"passed": False, "message": "same seed did not reproduce identical splits"}

    # Disjointness check using row sets (good enough for synthetic blobs)
    S_tr, S_va, S_te = _row_set(Xtr), _row_set(Xva), _row_set(Xte)
    if S_tr & S_va or S_tr & S_te or S_va & S_te:
        return {"passed": False, "message": "splits overlap"}

    return {"passed": True, "message": ""}

def grade_training_progress(losses, min_epochs=50, improve_ratio=0.90):
    """Late loss should be lower than early loss by a margin."""
    if len(losses) < min_epochs:
        return {"passed": False, "message": f"need at least {min_epochs} epochs"}
    import numpy as np
    early = float(np.mean(losses[:20]))
    late = float(np.mean(losses[-20:]))
    if not (late < improve_ratio * early):
        return {"passed": False, "message": f"loss did not improve enough: early={early:.4f}, late={late:.4f}"}
    return {"passed": True, "message": ""}

def grade_validation_accuracy(accs, threshold=0.85):
    """Last validation accuracy should exceed a threshold on the blob dataset."""
    if not accs:
        return {"passed": False, "message": "no accuracy values recorded"}
    ok = accs[-1] >= threshold
    return {"passed": ok, "message": f"val_acc_last={accs[-1]:.3f}, threshold={threshold:.2f}"}
