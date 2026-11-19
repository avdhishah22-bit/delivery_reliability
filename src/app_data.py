"""All tables the Streamlit app shows, computed once from the shipped DataCo orders.

Pure pandas / numpy / sklearn (no lightgbm, no streamlit), so it can be unit-tested.
`build_aggregates` is called by src/models/export_artifacts.py and the result is stored in
models/app_artifacts.joblib.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

MODE, REGION, MARKET, DEPT = "Shipping Mode", "Order Region", "Market", "Department Name"
PROFIT_CANDIDATES = ["Order Profit Per Order", "Benefit per order"]
MAX_DAYS = 6


def _expected_from_mode_mix(df: pd.DataFrame, group: str, y: str, mode_rate: pd.Series) -> pd.DataFrame:
    """Late rate of a group vs the rate expected from its shipping-mode mix alone.

    Without this control, a region that happens to use Second Class looks 'unreliable'
    when really it is the promise that is short."""
    g = df.assign(_exp=df[MODE].map(mode_rate)).groupby(group)
    t = pd.DataFrame({"orders": g.size(), "late_rate": g[y].mean(), "expected": g["_exp"].mean()})
    t["gap"] = t["late_rate"] - t["expected"]
    se = np.sqrt(t["expected"].clip(0.01, 0.99) * (1 - t["expected"].clip(0.01, 0.99)) / t["orders"])
    t["z"] = t["gap"] / se
    return t.sort_values("orders", ascending=False).reset_index()


def downsample_roc(y, score, points: int = 250):
    fpr, tpr, _ = roc_curve(y, score)
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(points, len(fpr))).astype(int))
    return {"fpr": fpr[idx], "tpr": tpr[idx], "auc": float(roc_auc_score(y, score))}


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    p = np.asarray(p, float)
    cut = pd.cut(p, np.linspace(0, 1, bins + 1), include_lowest=True)
    t = pd.DataFrame({"p": p, "y": np.asarray(y, float), "bin": cut}).groupby("bin", observed=True)
    out = pd.DataFrame({"predicted": t["p"].mean(), "observed": t["y"].mean(), "orders": t.size()})
    return out.reset_index(drop=True)


def build_aggregates(data: pd.DataFrame, scored: pd.DataFrame, *, y: str, yr: str, promise: str,
                     date_col: str, option_fn, split_info: dict) -> dict:
    """data   : every shipped order (all splits) with features, target y, real days yr, order dates.
    scored : the TEST split with p_late (calibrated model), p_lookup, p_mode columns.
    option_fn(df) -> array of lookup keys ('Same Day' split at noon)."""
    out = {}
    mode_rate = data.groupby(MODE)[y].mean()
    opt = pd.Series(option_fn(data), index=data.index, name="option")

    # --- scorecard per shipping mode
    g = data.groupby(MODE)
    out["scorecard"] = pd.DataFrame({
        "orders": g.size(), "promised_days": g[promise].mean(), "mean_real_days": g[yr].mean(),
        "std_real_days": g[yr].std(), "late_rate": g[y].mean()}).round(3)

    # --- real delivery-day distribution per lookup option, and each option's promise
    days = data[yr].astype(int).clip(0, MAX_DAYS)
    dist = pd.crosstab(opt, days, normalize="index").reindex(columns=range(MAX_DAYS + 1), fill_value=0.0)
    out["real_days_dist"] = dist
    out["option_orders"] = opt.value_counts().rename("orders").to_frame()
    out["option_promise"] = data.groupby(opt)[promise].agg(lambda s: int(s.mode().iloc[0]))
    out["within"] = pd.DataFrame({f"within {d} days": dist.loc[:, : d].sum(axis=1) for d in range(MAX_DAYS + 1)})
    out["within"].insert(0, "orders", out["option_orders"]["orders"])

    # --- time and place
    month = data[date_col].dt.to_period("M").dt.to_timestamp()
    out["monthly"] = (data.assign(month=month).groupby(["month", MODE])
                      .agg(orders=(y, "size"), late_rate=(y, "mean"), mean_real_days=(yr, "mean")).reset_index())
    out["hour_weekday"] = (data.groupby([MODE, "order_weekday", "order_hour"])
                           .agg(orders=(y, "size"), late_rate=(y, "mean")).reset_index())
    out["region_mode"] = (data.groupby([REGION, MODE]).agg(orders=(y, "size"), late_rate=(y, "mean")).reset_index())
    out["region_table"] = _expected_from_mode_mix(data, REGION, y, mode_rate)
    out["market_table"] = _expected_from_mode_mix(data, MARKET, y, mode_rate)
    out["dept_table"] = _expected_from_mode_mix(data, DEPT, y, mode_rate).head(15)

    # --- money (if the column survived cleaning)
    pcol = next((c for c in PROFIT_CANDIDATES if c in data.columns), None)
    if pcol:
        out["profit_mode"] = (data.groupby([MODE, y])[pcol].agg(["mean", "size"]).reset_index()
                              .rename(columns={"mean": "profit_per_order", "size": "orders"}))
    out["profit_col"] = pcol

    # --- model quality on the held-out test split
    yt = scored[y].to_numpy()
    out["roc"] = {"Calibrated LightGBM": downsample_roc(yt, scored["p_late"]),
                  "5-row lookup table": downsample_roc(yt, scored["p_lookup"]),
                  "Shipping mode only": downsample_roc(yt, scored["p_mode"])}
    out["calibration"] = calibration_table(yt, scored["p_late"])
    out["split_info"] = split_info
    out["kpis"] = {"orders": int(len(data)), "late_rate": float(data[y].mean()),
                   "test_orders": int(len(scored)), "test_late_rate": float(scored[y].mean()),
                   "date_min": data[date_col].min(), "date_max": data[date_col].max()}
    return out


def threshold_metrics(df: pd.DataFrame, thr: float, y: str, score: str = "p_late") -> dict:
    """What a 'flag every order with P(late) >= thr' rule would do on these orders."""
    flag = df[score].to_numpy() >= thr
    yv = df[y].to_numpy().astype(float)
    n, flagged, late = len(df), int(flag.sum()), float(yv.sum())
    tp = float(yv[flag].sum())
    return {"orders": n, "flagged": flagged, "flagged_share": flagged / n if n else np.nan,
            "precision": tp / flagged if flagged else np.nan,
            "recall": tp / late if late else np.nan,
            "false_alarms": flagged - tp,
            "false_per_catch": (flagged - tp) / tp if tp else np.nan,
            "tp": tp, "late": late}


def threshold_sweep(df: pd.DataFrame, y: str, score: str = "p_late", steps: int = 91) -> pd.DataFrame:
    rows = [{"threshold": t, **threshold_metrics(df, t, y, score)} for t in np.linspace(0.05, 0.95, steps)]
    return pd.DataFrame(rows)


def promise_late_rate(dist_row: pd.Series, promised_days: int) -> float:
    """Share of orders whose real delivery takes MORE days than the promise."""
    return float(dist_row.loc[dist_row.index > promised_days].sum())


def promise_for_target(dist_row: pd.Series, target_late: float) -> int:
    """Shortest promise (in days) that keeps the late rate at or below the target."""
    for p in range(MAX_DAYS + 1):
        if promise_late_rate(dist_row, p) <= target_late + 1e-9:
            return p
    return MAX_DAYS
