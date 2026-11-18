"""core.py - all the logic of the project, in one file, in the order the project happens.

    1. Settings        what the columns are, which ones are forbidden, the thresholds
    2. Loading         read the CSV, repair the dates, add features
    3. Splitting       oldest 70% train, next 15% validation, newest 15% test
    4. Metrics         how we score models (tie-aware) and intervals
    5. Models          LightGBM for "will it be late?" and "how many days?"
    6. Early warning   calibration, risk tiers, the decision rules
    7. Analysis        scorecard, region fairness, promise designer, threshold table
    8. Order scoring   what the Streamlit app calls for one order (incl. SHAP)
    9. App data        everything the app needs, saved once from notebook 3
   10. Deeper checks   bootstrap intervals, rolling validation, decision curve, calibration error, drift (notebook 4)

The notebooks and app.py import from here; test_core.py tests it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =====================================================================================
# 1. SETTINGS
# =====================================================================================
SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "DataCoSupplyChainDataset.csv"
MODELS_FILE = ROOT / "models" / "models.joblib"      # written by notebook 2
APP_FILE = ROOT / "models" / "app_data.joblib"       # written by notebook 3, read by app.py
EVIDENCE_FILE = ROOT / "models" / "evidence.joblib"   # written by notebook 4, read by the app's Evidence page

DATE = "order date (DateOrders)"
TARGET = "Late_delivery_risk"               # 1 = late. What we predict.
TARGET_DAYS = "Days for shipping (real)"    # real delivery days. A target, never a feature.
MODE = "Shipping Mode"
PROMISE = "Days for shipment (scheduled)"   # days promised to the customer

# Columns that are only known AFTER the order ships or arrives. Using one as a feature would be cheating.
LEAKY = ["Days for shipping (real)", "Delivery Status", "Order Status", "shipping date (DateOrders)"]

# Personal data and free text: removed on loading.
PRIVATE_COLS = ["Customer Email", "Customer Password", "Customer Fname", "Customer Lname", "Customer Street",
                "Customer Zipcode", "Order Zipcode", "Product Image", "Product Description", "Product Status"]

CAT = ["Type", "Shipping Mode", "Market", "Order Region", "Customer Segment", "Department Name", "Category Name"]
NUM = ["Days for shipment (scheduled)", "Order Item Discount Rate", "Order Item Quantity", "Product Price", "Sales",
       "order_hour", "order_weekday", "order_month", "same_country", "discount_ratio"]
FEATURES = CAT + NUM
assert not set(FEATURES) & set(LEAKY), "a leaky column is in the feature list"

# Early-warning thresholds (on CALIBRATED probabilities). Chosen after looking at the score clusters.
TIER_HIGH = 0.70            # at or above: High  (Second Class sits near 0.80)
TIER_LOW = 0.20             # below: Low         (Same Day before noon); in between: Medium (Standard ~0.40)
HIGH_VALUE_ORDER = 200.0    # a High-risk order worth at least this gets RE-PROMISE instead of ALERT
WIDE_INTERVAL_DAYS = 3.0    # an 80% delivery window at least this wide triggers WARN CUSTOMER
DISAGREE = 0.25             # if model and lookup table differ by more than this, the app trusts the table


# =====================================================================================
# 2. LOADING
# =====================================================================================
def parse_dates(s: pd.Series) -> pd.Series:
    """DataCo dates come in two formats because the CSV was once saved by Excel.

    '1/31/2018 22:56'  original month/day/year (Excel could not misread these).
    '01-12-2018 23:51' Excel re-read the original 1/12/2018 (12 Jan) as day/month and rewrote it day-first,
                       so the FIRST number is really the month.
    Without this, 39% of dates fail to parse."""
    s = s.astype(str).str.strip()
    original = pd.to_datetime(s, format="%m/%d/%Y %H:%M", errors="coerce")
    swapped = pd.to_datetime(s, format="%m-%d-%Y %H:%M", errors="coerce")
    return original.fillna(swapped)


def load_dataco(path=DATA_FILE) -> pd.DataFrame:
    """Read DataCo (latin-1), drop personal columns and duplicates, parse dates, sort by time, add features."""
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in PRIVATE_COLS if c in df.columns]).drop_duplicates()
    df[DATE] = parse_dates(df[DATE])
    bad = int(df[DATE].isna().sum())
    if bad:
        print(f"WARNING: {bad} order dates could not be parsed and were dropped.")
    df = df.dropna(subset=[DATE, TARGET]).sort_values(DATE, kind="stable").reset_index(drop=True)
    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that use only information available when the order is placed."""
    out = df.copy()
    out["order_hour"] = out[DATE].dt.hour
    out["order_weekday"] = out[DATE].dt.weekday
    out["order_month"] = out[DATE].dt.month
    out["same_country"] = (out["Customer Country"] == out["Order Country"]).astype(int)
    out["discount_ratio"] = out["Order Item Discount"] / out["Order Item Product Price"].replace(0, np.nan)
    return out


def shipped_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cancelled / suspected-fraud orders. They never ship, yet DataCo labels them 'not late',
    which makes payment type TRANSFER look reliable. All lateness modelling uses shipped orders only."""
    return df[df["Delivery Status"] != "Shipping canceled"].reset_index(drop=True)


# =====================================================================================
# 3. SPLITTING AND PREPROCESSING
# =====================================================================================
def time_split(df: pd.DataFrame, train: float = 0.70, val: float = 0.15):
    """Chronological split of a time-sorted frame: train (oldest), validation, test (newest).
    Never a random split: 89% of rows belong to multi-item orders, so a random split leaks siblings."""
    n = len(df)
    i, j = int(n * train), int(n * (train + val))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def make_preprocessor(cat=CAT, num=NUM, scale: bool = False) -> ColumnTransformer:
    """Median-impute numbers, one-hot encode categories (rare ones grouped). Put it inside a Pipeline
    so that it is fit on the training split only."""
    steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    parts = [("num", Pipeline(steps), list(num))]
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), list(cat)))
    return ColumnTransformer(parts)


# =====================================================================================
# 4. METRICS
# =====================================================================================
def break_ties(p, seed: int = 7919, eps: float = 1e-6):
    """Add tiny random noise so tied scores are ordered at random. A rule or lookup table gives many orders
    the SAME score; without this, PR-AUC and precision-at-recall wrongly punish it. ROC-AUC is unaffected."""
    p = np.asarray(p, float)
    return p + np.random.default_rng(seed).uniform(0, eps, len(p))


def pr_auc(y, p) -> float:
    return float(average_precision_score(y, break_ties(p)))


def threshold_for_recall(y, p, target_recall: float = 0.8) -> float:
    """Highest threshold that still catches `target_recall` of the late orders (tune on validation)."""
    _, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target_recall)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def precision_at_recall(y, p, target_recall: float = 0.8) -> float:
    """Best precision while still catching at least `target_recall` of the late orders."""
    prec, rec, _ = precision_recall_curve(y, break_ties(p))
    mask = rec >= target_recall
    return float(prec[mask].max()) if mask.any() else float("nan")


def lift_at_top(y, p, frac: float = 0.2) -> float:
    """Late rate among the top `frac` highest-risk orders, divided by the overall late rate."""
    y, p = np.asarray(y), break_ties(p)
    k = max(1, int(len(y) * frac))
    return float(y[np.argsort(-p, kind="stable")[:k]].mean() / y.mean())


def score_row(y, p) -> dict:
    """The four numbers every classifier is judged on. (F1 is not used: flagging every order scores 0.71.)"""
    return {"roc_auc": round(float(roc_auc_score(y, p)), 3), "pr_auc": round(pr_auc(y, p), 3),
            "precision@80%recall": round(precision_at_recall(y, p, 0.8), 3),
            "lift@top20%": round(lift_at_top(y, p, 0.2), 2)}


def operating_point(y, p, thr: float) -> dict:
    """What flagging every order with score >= thr does."""
    y, flag = np.asarray(y), np.asarray(p) >= thr
    tp = int((flag & (y == 1)).sum())
    return {"threshold": round(float(thr), 3), "flagged_share": round(float(flag.mean()), 3),
            "precision": round(tp / max(int(flag.sum()), 1), 3), "recall": round(tp / max(int(y.sum()), 1), 3)}


def regression_row(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return {"mae": round(float(np.mean(np.abs(y - pred))), 3), "rmse": round(float(np.sqrt(np.mean((y - pred) ** 2))), 3),
            "r2": round(float(r2_score(y, pred)), 3)}


def interval_row(y, lo, hi, groups=None) -> dict:
    """Coverage (share of true values inside [lo, hi]) and mean width, overall and per group."""
    y, lo, hi = (np.asarray(v, float) for v in (y, lo, hi))
    inside = (y >= lo) & (y <= hi)
    out = {"coverage": round(float(inside.mean()), 3), "mean_width": round(float((hi - lo).mean()), 3)}
    if groups is not None:
        g = np.asarray(groups)
        for k in np.unique(g):
            m = g == k
            out[f"cov_{k}"], out[f"width_{k}"] = round(float(inside[m].mean()), 3), round(float((hi - lo)[m].mean()), 3)
    return out


def wilson(k, n, z: float = 1.96):
    """95% confidence interval for a rate k/n (better than +/- 1.96 se for small n or rates near 0 or 1)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p, d = k / n, 1 + z * z / n
    centre, half = p + z * z / (2 * n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / d, (centre + half) / d


# =====================================================================================
# 5. MODELS
# =====================================================================================
LATE_SEARCH = {"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.05, 0.1],
               "n_estimators": [200, 400], "min_child_samples": [20, 50, 100]}
LATE_FIXED = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                  random_state=SEED, verbose=-1)


def fit_late_model(train: pd.DataFrame, val: pd.DataFrame, n_search: int = 8):
    """LightGBM for P(late). A small random search picks the settings with the best VALIDATION ROC-AUC;
    the test split is not touched. Returns (pipeline fit on train, best settings, best validation AUC)."""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import ParameterSampler
    prep = make_preprocessor().fit(train[FEATURES])
    Xtr, Xva = prep.transform(train[FEATURES]), prep.transform(val[FEATURES])
    best, best_auc = None, -1.0
    for params in ParameterSampler(LATE_SEARCH, n_iter=n_search, random_state=SEED):
        auc = roc_auc_score(val[TARGET], LGBMClassifier(**params, **LATE_FIXED).fit(Xtr, train[TARGET]).predict_proba(Xva)[:, 1])
        if auc > best_auc:
            best, best_auc = params, auc
    model = Pipeline([("prep", make_preprocessor()), ("clf", LGBMClassifier(**best, **LATE_FIXED))])
    return model.fit(train[FEATURES], train[TARGET]), best, float(best_auc)


def eta_pipeline(**extra) -> Pipeline:
    """LightGBM regressor for real delivery days. extra: objective='quantile', alpha=0.1 for quantile regression."""
    from lightgbm import LGBMRegressor
    reg = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8,
                        subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, **extra)
    return Pipeline([("prep", make_preprocessor()), ("reg", reg)])


# ---- three ways to turn a delivery-time guess into an 80% window ----------------------------------
def empirical_intervals(train: pd.DataFrame, group: str = MODE, alpha: float = 0.2) -> pd.DataFrame:
    """1) per-group quantiles of the real days in the training data."""
    q = train.groupby(group)[TARGET_DAYS].quantile([alpha / 2, 1 - alpha / 2]).unstack()
    q.columns = ["lo", "hi"]
    return q


def apply_intervals(table: pd.DataFrame, groups: pd.Series, default: tuple):
    lo = groups.map(table["lo"]).fillna(default[0]).to_numpy(float)
    hi = groups.map(table["hi"]).fillna(default[1]).to_numpy(float)
    return lo, hi


def _conformal_q(resid: np.ndarray, alpha: float) -> float:
    n = len(resid)
    return float(np.sort(resid)[min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1])


def conformal_halfwidths(y_cal, pred_cal, groups_cal, alpha: float = 0.2) -> dict:
    """3) split conformal: per-group half-width = the (1 - alpha) quantile of |real - predicted| on the
    validation split. Gives about 80% coverage without assuming a distribution."""
    r = np.abs(np.asarray(y_cal, float) - np.asarray(pred_cal, float))
    g = np.asarray(groups_cal)
    out = {k: _conformal_q(r[g == k], alpha) for k in np.unique(g)}
    out["__all__"] = _conformal_q(r, alpha)
    return out


def conformal_interval(pred, groups, halfwidths: dict):
    pred = np.asarray(pred, float)
    hw = np.array([halfwidths.get(g, halfwidths["__all__"]) for g in np.asarray(groups)])
    return np.clip(pred - hw, 0, None), pred + hw


# =====================================================================================
# 6. EARLY WARNING: calibration, tiers, lookup table, decision rules
# =====================================================================================
def fit_calibrator(val_scores, val_y) -> IsotonicRegression:
    """Isotonic regression fit on VALIDATION scores, so a score of 0.8 means about 80% of such orders are late."""
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        np.asarray(val_scores, float), np.asarray(val_y, float))


def tier_of(p, high: float = TIER_HIGH, low: float = TIER_LOW):
    """'High' / 'Medium' / 'Low' for one probability or an array."""
    p = np.asarray(p, float)
    out = np.where(p >= high, "High", np.where(p < low, "Low", "Medium"))
    return str(out) if out.ndim == 0 else out


def rule_key(df: pd.DataFrame):
    """Key of the 5-row lookup table: shipping mode, with Same Day split at the noon order cut-off."""
    mode, hour = df[MODE].to_numpy(), df["order_hour"].to_numpy()
    return np.where((mode == "Same Day") & (hour >= 12), "Same Day (from noon)",
                    np.where(mode == "Same Day", "Same Day (before noon)", mode))


def lookup_key(mode: str, hour: int) -> str:
    """rule_key for a single order."""
    if mode == "Same Day":
        return "Same Day (from noon)" if hour >= 12 else "Same Day (before noon)"
    return mode


def recommend(p_late: float, order_value: float, interval_days: float) -> list:
    """Actions for one order. p_late must be CALIBRATED.
    'Expedite' is not an option: nothing is faster than First Class. The useful actions are to correct
    the promise and to tell the customer."""
    actions, tier = [], tier_of(p_late)
    if tier == "High" and order_value >= HIGH_VALUE_ORDER:
        actions.append("RE-PROMISE: very likely late; update the promised date or offer a shipping option with a realistic promise")
    elif tier == "High":
        actions.append("ALERT: high chance of late delivery")
    if interval_days >= WIDE_INTERVAL_DAYS:
        actions.append("WARN CUSTOMER: delivery date is uncertain")
    return actions or ["NO ACTION"]


# =====================================================================================
# 7. ANALYSIS (notebooks and app)
# =====================================================================================
def scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """One row per shipping mode: promise, reality, late rate with a 95% interval, and profit per order."""
    g = df.groupby(MODE)
    sc = pd.DataFrame({"orders": g.size(), "promised_days": g[PROMISE].mean(), "mean_real_days": g[TARGET_DAYS].mean(),
                       "std_real_days": g[TARGET_DAYS].std(), "late_rate": g[TARGET].mean()})
    sc["late_low"], sc["late_high"] = wilson(g[TARGET].sum().to_numpy(), sc["orders"].to_numpy())
    if "Order Profit Per Order" in df.columns:
        sc["profit_per_order"] = g["Order Profit Per Order"].mean()
    return sc.sort_values("late_rate").round(3)


def real_days_distribution(df: pd.DataFrame, max_day: int = 6) -> pd.DataFrame:
    """Share of orders delivered on each real day (0..max_day, longer capped), per lookup option."""
    days = df[TARGET_DAYS].astype(int).clip(0, max_day)
    ct = pd.crosstab(pd.Series(rule_key(df), index=df.index, name="option"), days, normalize="index")
    return ct.reindex(columns=range(max_day + 1), fill_value=0.0)


def promise_late_rate(dist_row: pd.Series, promised_days: int) -> float:
    """Share of orders whose real delivery takes MORE days than the promise."""
    return float(dist_row.loc[dist_row.index > promised_days].sum())


def promise_for_target(dist_row: pd.Series, target_late: float) -> int:
    """Shortest promise (days) that keeps the late rate at or below the target."""
    for p in range(int(dist_row.index.max()) + 1):
        if promise_late_rate(dist_row, p) <= target_late + 1e-9:
            return p
    return int(dist_row.index.max())


def mode_mix_gap(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Late rate of each region / market / department versus the rate its shipping-mode mix alone would give.

    A region that uses more Second Class looks unreliable only because the promise is short. So we compare
    observed with expected; z measures the gap in standard errors (|z| > 1.96 is unusual, but with many groups
    a few exceed it by chance)."""
    mode_rate = df.groupby(MODE)[TARGET].mean()
    e = df[MODE].map(mode_rate)
    d = df.assign(_e=e, _v=e * (1 - e))
    r = d.groupby(group).agg(orders=(TARGET, "size"), observed=(TARGET, "mean"), expected=("_e", "mean"), _v=("_v", "sum"))
    r["gap"] = r["observed"] - r["expected"]
    r["z"] = r["gap"] * r["orders"] / np.sqrt(r["_v"])
    return r.drop(columns="_v").reset_index()


def threshold_table(scored: pd.DataFrame, thr: float, score: str = "p_late") -> dict:
    """What 'flag every order with P(late) >= thr' does on these orders."""
    y, flag = scored[TARGET].to_numpy(float), scored[score].to_numpy() >= thr
    n, flagged, late = len(scored), int(flag.sum()), float(y.sum())
    tp = float(y[flag].sum())
    return {"flagged": flagged, "flagged_share": flagged / n, "precision": tp / flagged if flagged else np.nan,
            "recall": tp / late if late else np.nan, "false_alarms": flagged - tp,
            "false_per_catch": (flagged - tp) / tp if tp else np.nan}


# =====================================================================================
# 8. ORDER SCORING (called by app.py)
# =====================================================================================
def clean_feature_name(name: str) -> str:
    """'cat__Shipping Mode_Second Class' -> 'Shipping Mode = Second Class'."""
    name = name.split("__", 1)[-1]
    for col in CAT:
        if name.startswith(col + "_"):
            return f"{col} = {name[len(col) + 1:]}"
    return name


def shap_values(model: Pipeline, frame: pd.DataFrame):
    """TreeSHAP contributions (log-odds) straight from LightGBM. Returns (matrix, feature names)."""
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    X = prep.transform(frame[FEATURES])
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    sv = clf.booster_.predict(X, pred_contrib=True)[:, :-1]      # last column is the base value
    return sv, np.array([clean_feature_name(n) for n in prep.get_feature_names_out()]), X


def explain_order(model: Pipeline, frame: pd.DataFrame, top: int = 6) -> pd.Series:
    """The biggest drivers of one order's late score (positive = pushes towards late).

    SHAP gives one value per one-hot column ('Shipping Mode = First Class' = 'is it First Class?'). For a reader that
    is confusing, so the columns of one feature are added up and labelled with the order's own value."""
    sv, names, _ = shap_values(model, frame)
    by_feature = pd.Series(sv[0], index=[n.split(" = ")[0] for n in names]).groupby(level=0).sum()
    row = frame.iloc[0]

    def label(f):
        v = row[f] if f in row.index else None
        return f"{f} = {v:.4g}" if isinstance(v, (int, float, np.number)) else f"{f} = {v}"
    by_feature.index = [label(f) for f in by_feature.index]
    return by_feature.reindex(by_feature.abs().sort_values(ascending=False).index).head(top)


def make_order_frame(app: dict, *, mode, order_dt, type_, market, region, department, category, segment,
                     sales, price, quantity, discount_rate) -> pd.DataFrame:
    """One-row frame with exactly the model's features (and none of the post-delivery columns)."""
    row = {"Type": type_, "Shipping Mode": mode, "Market": market, "Order Region": region,
           "Customer Segment": segment, "Department Name": department, "Category Name": category,
           "Days for shipment (scheduled)": app["scheduled_days"][mode], "Order Item Discount Rate": discount_rate,
           "Order Item Quantity": quantity, "Product Price": price, "Sales": sales, "order_hour": order_dt.hour,
           "order_weekday": order_dt.weekday(), "order_month": order_dt.month,
           "same_country": app["same_country_default"], "discount_ratio": discount_rate}
    return pd.DataFrame([row])[FEATURES]


def score_order(app: dict, models: dict, frame: pd.DataFrame) -> dict:
    """Calibrated P(late), expected days and an 80% delivery window for one order.

    The 5-row lookup table is as accurate as the model (notebook 2), so when the two disagree by more than DISAGREE
    (an unusual or rare input the model has overfitted), the table's answer is used and `disagree` is True."""
    p_model = float(app["calibrator"].predict([models["late"].predict_proba(frame)[:, 1][0]])[0])
    mode, hour = frame[MODE].iloc[0], int(frame["order_hour"].iloc[0])
    key = lookup_key(mode, hour)
    p_lookup = float(app["lookup"].get(key, np.nan))
    disagree = bool(not np.isnan(p_lookup) and abs(p_model - p_lookup) > DISAGREE)
    hw = app["halfwidths"]
    half = hw.get(mode, hw["__all__"])
    days = float(models["eta"].predict(frame)[0])
    return {"p_late": p_lookup if disagree else p_model, "p_model": p_model, "p_lookup": p_lookup, "disagree": disagree,
            "key": key, "days": days, "earliest": max(0.0, days - half), "latest": days + half, "window": 2 * half}


def score_all_modes(app: dict, models: dict, **order) -> pd.DataFrame:
    """The same order under every shipping mode (what-if comparison)."""
    rows = []
    for m in app["scheduled_days"]:
        r = score_order(app, models, make_order_frame(app, mode=m, **order))
        rows.append({"Shipping mode": m, "Promised days": app["scheduled_days"][m], "Chance late": r["p_late"],
                     "Model alone": r["p_model"], "Lookup table": r["p_lookup"], "Expected days": r["days"],
                     "Earliest": r["earliest"], "Latest": r["latest"]})
    return pd.DataFrame(rows)


# =====================================================================================
# 9. APP DATA (run from notebook 3, read by app.py)
# =====================================================================================
def build_app_data(data: pd.DataFrame, models: dict, save: bool = True) -> dict:
    """Score the held-out test split and collect everything the app needs into one file.

    data   : shipped orders with features (the output of shipped_only(load_dataco())).
    models : {'late': pipeline, 'eta': pipeline, 'halfwidths': dict} from notebook 2.
    """
    import joblib
    train, val, test = time_split(data)
    cal = fit_calibrator(models["late"].predict_proba(val[FEATURES])[:, 1], val[TARGET])     # validation only
    lookup = train.assign(key=rule_key(train)).groupby("key")[TARGET].mean().to_dict()       # the 5-row table
    mode_rate = train.groupby(MODE)[TARGET].mean()
    hw = models["halfwidths"]

    orders = data.copy()
    orders["split"] = ["Train"] * len(train) + ["Validation"] * len(val) + ["Test"] * len(test)
    orders["option"] = rule_key(orders)
    for c in ("p_late", "p_lookup", "p_mode", "interval_days"):
        orders[c] = np.nan
    t = orders["split"] == "Test"                                   # model scores exist for the test split only
    orders.loc[t, "p_late"] = cal.predict(models["late"].predict_proba(test[FEATURES])[:, 1])
    orders.loc[t, "p_lookup"] = orders.loc[t, "option"].map(lookup).to_numpy()
    orders.loc[t, "p_mode"] = orders.loc[t, MODE].map(mode_rate).to_numpy()
    orders.loc[t, "interval_days"] = [2 * hw.get(m, hw["__all__"]) for m in orders.loc[t, MODE]]

    keep = [DATE, MODE, "option", "Type", "Market", "Order Region", "Department Name", "Sales", "order_hour",
            "order_weekday", TARGET_DAYS, TARGET, PROMISE, "split", "p_late", "p_lookup", "p_mode", "interval_days"]
    if "Order Profit Per Order" in orders.columns:
        keep.append("Order Profit Per Order")
    slim = orders[keep].reset_index(drop=True)      # plain strings; joblib compression keeps the file small

    app = {
        "orders": slim, "calibrator": cal, "lookup": lookup, "halfwidths": hw,
        "options": {c: sorted(data[c].dropna().unique().tolist()) for c in CAT},
        "category_by_dept": {d: sorted(g["Category Name"].dropna().unique().tolist()) for d, g in data.groupby("Department Name")},
        "num_defaults": data[NUM].median().to_dict(),
        "scheduled_days": {m: int(s.mode()[0]) for m, s in data.groupby(MODE)[PROMISE]},
        "same_country_default": int(data["same_country"].mode()[0]),
        "split_info": {n: {"orders": len(d), "start": d[DATE].min(), "end": d[DATE].max()}
                       for n, d in (("Train", train), ("Validation", val), ("Test", test))},
    }
    if save:
        APP_FILE.parent.mkdir(exist_ok=True)
        joblib.dump(app, APP_FILE, compress=3)
        print("saved", APP_FILE, "| orders:", len(slim), "| test orders scored:", int(t.sum()))
    return app


# =====================================================================================
# 10. DEEPER CHECKS (notebook 4)
# =====================================================================================
def cluster_bootstrap_auc(y, scores: dict, groups=None, n_boot: int = 500, seed: int = SEED) -> dict:
    """ROC-AUC of several score vectors with 95% bootstrap intervals, and the paired AUC difference of the first
    two. Orders are resampled as a whole (groups = Order Id) because the items of one order share their outcome;
    resampling single rows would make the intervals too narrow. Implemented with row weights, so it is fast."""
    y = np.asarray(y)
    names = list(scores)
    s = {k: np.asarray(v, float) for k, v in scores.items()}
    codes = np.arange(len(y)) if groups is None else pd.factorize(np.asarray(groups))[0]
    n_groups = int(codes.max()) + 1
    rng = np.random.default_rng(seed)
    aucs = {k: [] for k in names}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)[codes]
        for k in names:
            aucs[k].append(roc_auc_score(y, s[k], sample_weight=w))
    out = {k: {"auc": float(roc_auc_score(y, s[k])), "lo": float(np.percentile(aucs[k], 2.5)),
               "hi": float(np.percentile(aucs[k], 97.5))} for k in names}
    if len(names) >= 2:
        d = np.array(aucs[names[0]]) - np.array(aucs[names[1]])
        out["difference"] = {"label": f"{names[0]} minus {names[1]}", "auc": out[names[0]]["auc"] - out[names[1]]["auc"],
                             "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
    return out


def rolling_origin_folds(df: pd.DataFrame, n_folds: int = 5, first_train: float = 0.4):
    """Expanding-window validation for a time-sorted frame: train on everything before a block, test on that block.
    The first `first_train` share is only ever training data; the rest is cut into n_folds equal test blocks."""
    n = len(df)
    start = int(n * first_train)
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    return [(df.iloc[:edges[i]], df.iloc[edges[i]:edges[i + 1]]) for i in range(n_folds)]


def decision_curve(y, scores: dict, thresholds=None) -> pd.DataFrame:
    """Net benefit of acting on orders whose score is >= t (Vickers and Elkin, 2006):
    TP/n - FP/n * t/(1-t). Compared with 'act on everything' and 'act on nothing' (0). Higher is better."""
    y = np.asarray(y)
    n = len(y)
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        row = {"threshold": float(t), "act on all": float(y.mean() - (1 - y.mean()) * t / (1 - t))}
        for k, p in scores.items():
            flag = np.asarray(p) >= t
            tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
            row[k] = tp / n - fp / n * t / (1 - t)
        rows.append(row)
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Average gap between the predicted chance and the observed late rate, in equal-width bins, weighted by size."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def psi(reference, current, bins: int = 10) -> float:
    """Population stability index between two samples of one column. Numbers use the reference's quantile bins;
    categories use their shares. Below 0.1 stable, 0.1 to 0.25 moderate shift, above 0.25 large shift."""
    ref, cur = pd.Series(reference).dropna(), pd.Series(current).dropna()
    if pd.api.types.is_numeric_dtype(ref):
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        a = np.histogram(ref, edges)[0] / len(ref)
        b = np.histogram(cur, edges)[0] / len(cur)
    else:
        cats = sorted(set(ref) | set(cur))
        a = np.array([(ref == c).mean() for c in cats])
        b = np.array([(cur == c).mean() for c in cats])
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))
