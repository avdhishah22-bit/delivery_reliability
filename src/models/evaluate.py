"""Shared metrics for classification and regression tasks."""
import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, r2_score, recall_score, roc_auc_score,
)


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[np.argmax(f1)])


def evaluate_binary(y, p, thr: float) -> dict:
    pred = (np.asarray(p) >= thr).astype(int)
    return {
        "threshold": round(float(thr), 3),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "f1": round(f1_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred), 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def evaluate_regression(y, pred) -> dict:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return {
        "mae": round(float(np.mean(np.abs(y - pred))), 4),
        "rmse": round(float(np.sqrt(np.mean((y - pred) ** 2))), 4),
        "r2": round(float(r2_score(y, pred)), 4),
    }
