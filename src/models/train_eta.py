"""Task 2 helpers: prediction intervals for the ETA (days for shipping) on DataCo.

Three ways to build an 80% interval (alpha = 0.2), compared in notebooks/04_eta_models.ipynb:
  1. empirical per-mode quantiles of the training data
  2. LightGBM quantile regression (alpha/2 and 1 - alpha/2)
  3. split conformal prediction: point model + per-mode residual quantile from the validation split
"""
import numpy as np
import pandas as pd


def empirical_group_intervals(train: pd.DataFrame, target: str, group: str, alpha: float = 0.2) -> pd.DataFrame:
    q = train.groupby(group)[target].quantile([alpha / 2, 1 - alpha / 2]).unstack()
    q.columns = ["lo", "hi"]
    return q


def apply_group_intervals(table: pd.DataFrame, groups: pd.Series, default: tuple):
    lo = groups.map(table["lo"]).fillna(default[0]).to_numpy(dtype=float)
    hi = groups.map(table["hi"]).fillna(default[1]).to_numpy(dtype=float)
    return lo, hi


def _conformal_q(resid: np.ndarray, alpha: float) -> float:
    n = len(resid)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(resid)[k - 1])


def conformal_halfwidths(y_cal, pred_cal, groups_cal, alpha: float = 0.2) -> dict:
    """Per-group half-width = the (1 - alpha) quantile of |y - pred| on the calibration split."""
    r = np.abs(np.asarray(y_cal, float) - np.asarray(pred_cal, float))
    g = np.asarray(groups_cal)
    out = {k: _conformal_q(r[g == k], alpha) for k in np.unique(g)}
    out["__all__"] = _conformal_q(r, alpha)
    return out


def conformal_interval(pred, groups, halfwidths: dict, lower_clip: float = 0.0):
    pred = np.asarray(pred, float)
    hw = np.array([halfwidths.get(g, halfwidths["__all__"]) for g in np.asarray(groups)])
    return np.clip(pred - hw, lower_clip, None), pred + hw
