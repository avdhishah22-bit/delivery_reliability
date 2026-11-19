"""Shared metrics for classification, regression and prediction intervals."""
import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, r2_score, recall_score, roc_auc_score,
)


def break_ties(p, seed: int = 7919, eps: float = 1e-6):
    """Random tie-breaking. A rule or lookup table gives many orders the SAME score; PR-AUC and
    precision-at-recall then look worse than they are, because a real deployment could flag any
    subset of a tied group. Adding tiny noise orders ties at random. ROC-AUC is unaffected."""
    p = np.asarray(p, float)
    return p + np.random.default_rng(seed).uniform(0, eps, len(p))


def pr_auc(y, p) -> float:
    return float(average_precision_score(y, break_ties(p)))


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[np.argmax(f1)])


def threshold_for_recall(y, p, target_recall: float = 0.8) -> float:
    """Highest threshold that still catches `target_recall` of the positives (tune on validation)."""
    _, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target_recall)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def precision_at_recall(y, p, target_recall: float = 0.8) -> float:
    """Best precision achievable while catching at least `target_recall` of positives."""
    prec, rec, _ = precision_recall_curve(y, break_ties(p))
    mask = rec >= target_recall
    return float(prec[mask].max()) if mask.any() else float("nan")


def lift_at_top(y, p, frac: float = 0.2) -> float:
    """Late rate among the top `frac` highest-risk orders divided by the overall late rate."""
    y, p = np.asarray(y), break_ties(p)
    k = max(1, int(len(y) * frac))
    order = np.argsort(-p, kind="stable")
    return float(y[order[:k]].mean() / y.mean())


def evaluate_binary(y, p, thr: float) -> dict:
    pred = (np.asarray(p) >= thr).astype(int)
    return {
        "threshold": round(float(thr), 3),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(pr_auc(y, p), 4),
        "f1": round(f1_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred), 4),
        "flagged_share": round(float(pred.mean()), 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def evaluate_regression(y, pred) -> dict:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return {
        "mae": round(float(np.mean(np.abs(y - pred))), 4),
        "rmse": round(float(np.sqrt(np.mean((y - pred) ** 2))), 4),
        "r2": round(float(r2_score(y, pred)), 4),
    }


def interval_metrics(y, lo, hi, groups=None) -> dict:
    """Coverage (share of true values inside [lo, hi]) and mean width, overall and per group."""
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    inside = (y >= lo) & (y <= hi)
    out = {"coverage": round(float(inside.mean()), 3), "mean_width": round(float((hi - lo).mean()), 3)}
    if groups is not None:
        groups = np.asarray(groups)
        for g in np.unique(groups):
            m = groups == g
            out[f"cov_{g}"] = round(float(inside[m].mean()), 3)
            out[f"width_{g}"] = round(float((hi[m] - lo[m]).mean()), 3)
    return out
