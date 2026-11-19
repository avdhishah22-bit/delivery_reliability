"""Logic behind the Streamlit app (kept out of the UI file so it can be tested)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_all(models_dir):
    d = Path(models_dir)
    art = joblib.load(d / "app_artifacts.joblib")
    model = joblib.load(d / "late_delivery_lgbm.joblib")["model"]
    eta = joblib.load(d / "eta_lgbm.joblib")
    return art, model, eta


def lookup_key(mode: str, hour: int) -> str:
    if mode == "Same Day":
        return "Same Day (from noon)" if hour >= 12 else "Same Day (before noon)"
    return mode


def make_order_frame(art, *, mode, order_dt, type_, market, region, department, category, segment,
                     sales, price, quantity, discount_rate) -> pd.DataFrame:
    """One-row frame with exactly the model's pre-shipment features (no post-delivery columns)."""
    row = {
        "Type": type_, "Shipping Mode": mode, "Market": market, "Order Region": region,
        "Customer Segment": segment, "Department Name": department, "Category Name": category,
        "Days for shipment (scheduled)": art["scheduled_days"][mode],
        "Order Item Discount Rate": discount_rate, "Order Item Quantity": quantity,
        "Product Price": price, "Sales": sales,
        "order_hour": order_dt.hour, "order_weekday": order_dt.weekday(), "order_month": order_dt.month,
        "same_country": art["same_country_default"], "discount_ratio": discount_rate,
    }
    return pd.DataFrame([row])[art["features"]]


def score_order(art, model, eta, frame) -> dict:
    p_raw = float(model.predict_proba(frame)[:, 1][0])
    p_late = float(art["calibrator"].predict([p_raw])[0])
    mode, hour = frame["Shipping Mode"].iloc[0], int(frame["order_hour"].iloc[0])
    key = lookup_key(mode, hour)
    hw = art["conformal_halfwidths"]; half = hw.get(mode, hw["__all__"])
    point = float(eta["point"].predict(frame)[0])
    return {"p_late": p_late, "p_lookup": float(art["lookup"].get(key, np.nan)), "lookup_key": key,
            "eta_point": point, "eta_lo": max(0.0, point - half), "eta_hi": point + half, "interval_days": 2 * half}


def explain(model, frame, top: int = 6) -> pd.Series:
    """TreeSHAP contributions (log-odds) of the biggest drivers of this order's late score."""
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    X = prep.transform(frame); X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    sv = clf.booster_.predict(X, pred_contrib=True)[0][:-1]
    s = pd.Series(sv, index=[clean_feature_name(n) for n in prep.get_feature_names_out()])
    s = s.groupby(level=0).sum()
    return s.reindex(s.abs().sort_values(ascending=False).index).head(top).rename("contribution")


def clean_feature_name(name: str) -> str:
    """'cat__Shipping Mode_Second Class' -> 'Shipping Mode = Second Class'."""
    name = name.split("__", 1)[-1]
    for col in ("Shipping Mode", "Order Region", "Customer Segment", "Department Name", "Category Name",
                "Market", "Type"):
        if name.startswith(col + "_"):
            return f"{col} = {name[len(col) + 1:]}"
    return name


def score_all_modes(art, model, eta, **order) -> pd.DataFrame:
    """The same order scored under every shipping mode (what-if comparison)."""
    rows = []
    for mode in art["scheduled_days"]:
        frame = make_order_frame(art, mode=mode, **order)
        r = score_order(art, model, eta, frame)
        rows.append({"Shipping mode": mode, "Promised (days)": art["scheduled_days"][mode],
                     "P(late) model": r["p_late"], "P(late) lookup": r["p_lookup"],
                     "Expected days": r["eta_point"], "Earliest": r["eta_lo"], "Latest": r["eta_hi"],
                     "Interval (days)": r["interval_days"]})
    return pd.DataFrame(rows)


def compare_modes(art, model, eta, base: dict) -> pd.DataFrame:
    """Score the SAME order under every shipping mode (base = make_order_frame kwargs except mode)."""
    rows = []
    for m in art["scheduled_days"]:
        r = score_order(art, model, eta, make_order_frame(art, mode=m, **base))
        rows.append({"Shipping mode": m, "Promised days": art["scheduled_days"][m],
                     "Chance late (model)": r["p_late"], "Chance late (lookup)": r["p_lookup"],
                     "ETA low": r["eta_lo"], "ETA high": r["eta_hi"], "lookup_key": r["lookup_key"]})
    return pd.DataFrame(rows)
