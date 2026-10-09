"""Multi-label EC metrics."""
import warnings
warnings.filterwarnings("ignore", message="No positive class")
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve)


def pr_at_k(y_true, scores, ks=(1, 3, 5)):
    order = np.argsort(-scores, axis=1)
    out = {}
    for k in ks:
        topk = order[:, :k]
        hits = np.take_along_axis(y_true, topk, axis=1).sum(axis=1)
        npos = y_true.sum(axis=1).clip(min=1)
        out[f"P@{k}"] = float((hits / k).mean())
        out[f"R@{k}"] = float((hits / npos).mean())
    return out


def full_report(y_true, scores, thr=0.5):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= thr).astype(int)
    rep = {
        "macro-F1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "micro-F1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "weighted-F1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "mAP": float(np.mean([average_precision_score(y_true[:, c], scores[:, c])
                              for c in range(y_true.shape[1])])),
    }
    rep.update(pr_at_k(y_true, scores))
    return rep


def tune_threshold(y_true, scores, grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    best = max(((float(f1_score(y_true, scores >= t, average="macro",
                                zero_division=0)), t) for t in grid))
    return {"thr": best[1], "val-macro-F1": best[0]}