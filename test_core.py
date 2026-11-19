"""Tests for core.py. Run from the project root:  python -m pytest -q

They use a small fake DataCo-shaped table, so they run in seconds and need no real data.
Tests that need LightGBM are skipped if it is not installed."""
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import core
from core import (FEATURES, LEAKY, MODE, PROMISE, TARGET, TARGET_DAYS, break_ties, conformal_halfwidths,
                  conformal_interval, lookup_key, make_order_frame, mode_mix_gap, parse_dates, promise_for_target,
                  promise_late_rate, recommend, rule_key, score_row, shipped_only, threshold_table, tier_of,
                  time_split, wilson, cluster_bootstrap_auc, rolling_origin_folds, decision_curve,
                  expected_calibration_error, psi)

PROMISES = {"Same Day": 0, "First Class": 1, "Second Class": 2, "Standard Class": 4}


def fake_dataco(n=6000, seed=0) -> pd.DataFrame:
    """Raw-looking DataCo: lateness is set by the promise, as in the real data. Some dates use the Excel-damaged format."""
    r = np.random.default_rng(seed)
    mode = r.choice(list(PROMISES), n, p=[0.05, 0.15, 0.2, 0.6])
    when = pd.Timestamp("2016-01-01") + pd.to_timedelta(np.sort(r.integers(0, 24 * 60 * 700, n)), unit="m")
    hour = when.hour.to_numpy()
    real = np.select([mode == "Same Day", mode == "First Class"],
                     [np.where(hour >= 12, 1, 0), np.full(n, 2)], default=r.integers(2, 7, n))
    promise = np.array([PROMISES[m] for m in mode])
    cancelled = r.random(n) < 0.04
    late = (real > promise).astype(int)
    late[cancelled] = 0
    damaged = (r.random(n) < 0.4) & (when.day.to_numpy() <= 12)
    text = np.where(damaged, when.strftime("%m-%d-%Y %H:%M"), when.strftime("%m/%d/%Y %H:%M"))
    return pd.DataFrame({
        "Order Id": np.arange(n) // 2, core.DATE: text, TARGET: late, "Delivery Status": np.where(cancelled, "Shipping canceled", "Advance shipping"),
        "Order Status": "COMPLETE", "Days for shipping (real)": real, "Days for shipment (scheduled)": promise,
        "shipping date (DateOrders)": text, MODE: mode, "Type": np.where(cancelled, "TRANSFER", r.choice(["DEBIT", "CASH", "PAYMENT"], n)),
        "Market": r.choice(["LATAM", "Europe", "Pacific Asia"], n), "Order Region": r.choice(list("ABCDEFGH"), n),
        "Customer Segment": r.choice(["Consumer", "Corporate"], n), "Department Name": r.choice(["Golf", "Fitness"], n),
        "Category Name": r.choice(["Cleats", "Shoes", "Balls"], n), "Order Item Discount Rate": r.random(n) * 0.2,
        "Order Item Quantity": r.integers(1, 5, n), "Product Price": r.integers(20, 300, n).astype(float),
        "Sales": r.integers(20, 500, n).astype(float), "Customer Country": "US", "Order Country": np.where(r.random(n) < 0.3, "FR", "US"),
        "Order Item Discount": r.random(n) * 20, "Order Item Product Price": r.integers(20, 300, n).astype(float),
        "Order Profit Per Order": r.normal(20, 40, n), "Customer Email": "x@y.z", "Customer Password": "XXXX"})


@pytest.fixture(scope="module")
def raw():
    return fake_dataco()


@pytest.fixture(scope="module")
def data(raw):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "dataco.csv"
        raw.to_csv(path, index=False, encoding="latin-1")
        return shipped_only(core.load_dataco(path))


# ---------------------------------------------------------------- loading
def test_dates_in_both_formats_parse_to_the_same_day():
    s = pd.Series(["1/12/2018 23:51", "01-12-2018 23:51", "1/31/2018 22:56"])
    out = parse_dates(s)
    assert out.notna().all() and out[0] == out[1] == pd.Timestamp("2018-01-12 23:51")


def test_load_removes_private_columns_sorts_and_parses_everything(raw):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "dataco.csv"
        raw.to_csv(path, index=False, encoding="latin-1")
        df = core.load_dataco(path)
    assert "Customer Email" not in df.columns and "order_hour" in df.columns
    assert df[core.DATE].is_monotonic_increasing and len(df) == len(raw)


def test_shipped_only_drops_cancelled(data):
    assert (data["Delivery Status"] != "Shipping canceled").all()


def test_no_leaky_column_is_a_feature():
    assert not set(FEATURES) & set(LEAKY)


# ---------------------------------------------------------------- splitting
def test_time_split_is_chronological_and_complete():
    df = pd.DataFrame({"t": range(100)})
    tr, va, te = time_split(df)
    assert len(tr) + len(va) + len(te) == 100 and tr["t"].max() < va["t"].min() < te["t"].min()


# ---------------------------------------------------------------- metrics
def test_break_ties_keeps_the_order_of_distinct_scores():
    p = np.array([0.1, 0.5, 0.9, 0.3])
    assert (np.argsort(break_ties(p)) == np.argsort(p)).all()


def test_score_row_perfect_and_random():
    y = np.random.default_rng(0).integers(0, 2, 4000)
    assert score_row(y, y.astype(float))["roc_auc"] == 1.0
    assert abs(score_row(y, np.random.default_rng(1).random(4000))["roc_auc"] - 0.5) < 0.05


def test_tied_scores_are_not_penalised_for_precision_at_recall():
    y = np.random.default_rng(0).integers(0, 2, 5000)
    tied = np.full(5000, 0.5)                       # no information at all
    assert abs(score_row(y, tied)["precision@80%recall"] - y.mean()) < 0.05


def test_conformal_interval_covers_about_80_percent():
    r = np.random.default_rng(0)
    g = r.choice(["a", "b"], 6000)
    truth = 10 + r.normal(0, np.where(g == "a", 1, 3))             # prediction is 10; the noise is the error
    hw = conformal_halfwidths(truth[:3000], np.full(3000, 10.0), g[:3000])
    lo, hi = conformal_interval(np.full(3000, 10.0), g[3000:], hw)
    covered = (truth[3000:] >= lo) & (truth[3000:] <= hi)
    assert 0.77 < covered.mean() < 0.83
    assert hw["b"] > 2 * hw["a"]                                   # the noisier group gets a wider window


def test_wilson_interval_contains_the_rate():
    lo, hi = wilson(8, 100)
    assert lo < 0.08 < hi


# ---------------------------------------------------------------- early warning
def test_tiers_and_calibrator():
    assert list(tier_of([0.9, 0.5, 0.1])) == ["High", "Medium", "Low"] and tier_of(0.8) == "High"
    p = np.random.default_rng(0).random(2000)
    y = (np.random.default_rng(1).random(2000) < p).astype(int)
    cal = core.fit_calibrator(p, y)
    out = cal.predict(np.linspace(0, 1, 50))
    assert (np.diff(out) >= -1e-12).all() and 0 <= out.min() and out.max() <= 1


def test_recommend_rules():
    assert recommend(0.9, 500, 1.0)[0].startswith("RE-PROMISE")
    assert recommend(0.9, 50, 1.0)[0].startswith("ALERT")
    assert recommend(0.1, 500, 4.0) == ["WARN CUSTOMER: delivery date is uncertain"]
    assert recommend(0.1, 50, 1.0) == ["NO ACTION"]


def test_same_day_noon_cutoff_in_both_key_functions():
    df = pd.DataFrame({MODE: ["Same Day", "Same Day", "First Class"], "order_hour": [11, 12, 3]})
    assert list(rule_key(df)) == ["Same Day (before noon)", "Same Day (from noon)", "First Class"]
    assert [lookup_key("Same Day", 11), lookup_key("Same Day", 12), lookup_key("First Class", 3)] == list(rule_key(df))


# ---------------------------------------------------------------- analysis
def test_scorecard_reproduces_the_promise_story(data):
    sc = core.scorecard(data)
    assert sc.loc["Second Class", "late_rate"] > sc.loc["Standard Class", "late_rate"] + 0.3
    assert sc.loc["First Class", "late_rate"] == 1.0


def test_mode_mix_gap_is_small_for_random_regions(data):
    r = mode_mix_gap(data, "Order Region")
    assert (r["z"].abs() < 4.5).all() and {"observed", "expected", "gap", "z"} <= set(r.columns)


def test_promise_designer(data):
    dist = core.real_days_distribution(data).loc["Standard Class"]
    assert promise_late_rate(dist, 4) == pytest.approx(0.4, abs=0.03)
    assert promise_for_target(dist, 0.41) <= 5 and promise_for_target(dist, 0.0) >= 6


def test_threshold_table_counts():
    df = pd.DataFrame({TARGET: [1, 1, 0, 0], "p_late": [0.9, 0.4, 0.8, 0.1]})
    t = threshold_table(df, 0.5)
    assert t["flagged"] == 2 and t["precision"] == 0.5 and t["recall"] == 0.5 and t["false_per_catch"] == 1.0


# ---------------------------------------------------------------- order scoring and app data
def test_order_frame_has_exactly_the_model_features():
    app = {"scheduled_days": {"Standard Class": 4}, "same_country_default": 1}
    f = make_order_frame(app, mode="Standard Class", order_dt=datetime(2017, 6, 20, 14, 5), type_="DEBIT", market="LATAM",
                         region="Western Europe", department="Golf", category="Cleats", segment="Consumer",
                         sales=120.0, price=60.0, quantity=2, discount_rate=0.1)
    assert list(f.columns) == FEATURES and f["order_hour"].iloc[0] == 14 and f[PROMISE].iloc[0] == 4


class _FakeLate:                      # stands in for the LightGBM pipeline
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


class _FakeEta:
    def predict(self, X):
        return np.full(len(X), 4.0)


def test_model_and_table_disagreement_falls_back_to_the_table():
    from sklearn.isotonic import IsotonicRegression
    cal = IsotonicRegression(out_of_bounds="clip").fit([0, 1], [0, 1])             # identity calibrator
    app = {"calibrator": cal, "lookup": {"Standard Class": 0.4}, "halfwidths": {"Standard Class": 2.0, "__all__": 2.0}}
    frame = pd.DataFrame({MODE: ["Standard Class"], "order_hour": [14]})
    far = core.score_order(app, {"late": _FakeLate(0.02), "eta": _FakeEta()}, frame)
    assert far["disagree"] and far["p_late"] == 0.4 and far["p_model"] == pytest.approx(0.02)
    near = core.score_order(app, {"late": _FakeLate(0.45), "eta": _FakeEta()}, frame)
    assert not near["disagree"] and near["p_late"] == pytest.approx(0.45)


def test_full_pipeline_with_lightgbm(data):
    pytest.importorskip("lightgbm")
    train, val, test = time_split(data)
    late, params, auc = core.fit_late_model(train, val, n_search=2)
    assert auc > 0.7
    eta = core.eta_pipeline().fit(train[FEATURES], train[TARGET_DAYS])
    hw = conformal_halfwidths(val[TARGET_DAYS], eta.predict(val[FEATURES]), val[MODE])
    app = core.build_app_data(data, {"late": late, "eta": eta, "halfwidths": hw}, save=False)
    assert set(app["orders"]["split"].unique()) == {"Train", "Validation", "Test"}
    assert app["orders"].loc[app["orders"]["split"] == "Test", "p_late"].between(0, 1).all()
    assert app["orders"].loc[app["orders"]["split"] == "Train", "p_late"].isna().all()
    models = {"late": late, "eta": eta}
    f = make_order_frame(app, mode="Second Class", order_dt=datetime(2017, 6, 20, 9), type_="DEBIT", market="LATAM",
                         region="A", department="Golf", category="Cleats", segment="Consumer", sales=100.0,
                         price=50.0, quantity=2, discount_rate=0.1)
    r = core.score_order(app, models, f)
    assert 0 <= r["p_late"] <= 1 and r["earliest"] <= r["days"] <= r["latest"]
    assert len(core.score_all_modes(app, models, order_dt=datetime(2017, 6, 20, 9), type_="DEBIT", market="LATAM",
                                    region="A", department="Golf", category="Cleats", segment="Consumer",
                                    sales=100.0, price=50.0, quantity=2, discount_rate=0.1)) == 4
    drivers = core.explain_order(late, f)
    assert len(drivers) <= 6 and drivers.index.is_unique and all(" = " in lab for lab in drivers.index)


# ---------------------------------------------------------------- deeper checks
def test_cluster_bootstrap_difference_is_zero_for_identical_scores_and_covers_a_real_gap():
    r = np.random.default_rng(1)
    y = r.integers(0, 2, 2000)
    good = y + r.normal(0, 1, 2000)
    same = core.cluster_bootstrap_auc(y, {"a": good, "b": good}, groups=np.arange(2000) // 2, n_boot=100)
    assert same["difference"]["auc"] == 0 and same["difference"]["lo"] == 0 and same["difference"]["hi"] == 0
    gap = core.cluster_bootstrap_auc(y, {"good": good, "noise": r.normal(size=2000)}, n_boot=100)
    assert gap["difference"]["lo"] > 0.1                      # clearly better than noise
    assert gap["good"]["lo"] < gap["good"]["auc"] < gap["good"]["hi"]


def test_rolling_origin_folds_expand_and_never_overlap():
    df = pd.DataFrame({"x": range(1000)})
    folds = rolling_origin_folds(df, n_folds=5, first_train=0.4)
    assert len(folds) == 5 and len(folds[0][0]) == 400
    for (tr, te), (tr2, te2) in zip(folds, folds[1:]):
        assert tr.index.max() < te.index.min() and len(tr2) > len(tr) and te.index.max() < te2.index.min()
    assert folds[-1][1].index.max() == 999


def test_decision_curve_known_values():
    y = np.array([1, 1, 0, 0])
    dc = decision_curve(y, {"perfect": np.array([1.0, 1.0, 0.0, 0.0])}, thresholds=[0.5])
    assert dc.loc[0, "perfect"] == 0.5                         # TP 2 of 4, no FP
    assert dc.loc[0, "act on all"] == 0.0                      # 0.5 - 0.5 * 1


def test_calibration_error_and_psi():
    y = np.array([0, 1] * 500)
    assert expected_calibration_error(y, np.full(1000, 0.5)) < 0.01
    assert expected_calibration_error(y, np.full(1000, 0.9)) > 0.35
    r = np.random.default_rng(2)
    a = r.normal(size=5000)
    assert psi(a, r.normal(size=5000)) < 0.05
    assert psi(a, r.normal(2, 1, 5000)) > 0.25
    assert psi(pd.Series(["x"] * 90 + ["y"] * 10), pd.Series(["x"] * 50 + ["y"] * 50)) > 0.25


# ---------------------------------------------------------------- customer impact (notebook 5)
def _orders(effect=0.0, n=20000, seed=3):
    """Simulated orders: real days random 2..6; customers reorder with probability 0.5 + effect * (days - 2)."""
    r = np.random.default_rng(seed)
    days = r.integers(2, 7, n)
    cust = r.integers(0, n // 3, n)
    p = np.clip(0.5 + effect * (days - 2), 0, 1)
    return pd.DataFrame({core.CUSTOMER: cust, "days": days.astype(float), "reorder_90": (r.random(n) < p).astype(float),
                         core.MODE: "Standard Class", "month": r.choice(["a", "b", "c"], n)})


def test_ols_cluster_matches_lstsq_and_hc1_for_singleton_clusters():
    r = np.random.default_rng(0)
    X = np.column_stack([np.ones(300), r.normal(size=300)])
    y = 2 + 3 * X[:, 1] + r.normal(size=300)
    beta, cov = core.ols_cluster(y, X, np.arange(300))
    assert np.allclose(beta, np.linalg.lstsq(X, y, rcond=None)[0])
    e = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    hc1 = 300 / 298 * bread @ (X.T * e ** 2) @ X @ bread
    assert np.allclose(cov, hc1, rtol=1e-6)


def test_effect_of_delay_recovers_a_known_effect_and_a_null():
    hit = core.effect_of_delay(_orders(effect=-0.03), "reorder_90", "days", fe=(core.MODE, "month"))
    assert hit["lo"] < -0.03 < hit["hi"] and hit["p"] < 0.001
    null = core.effect_of_delay(_orders(effect=0.0), "reorder_90", "days", fe=(core.MODE, "month"))
    assert abs(null["coef"]) < 3 * null["se"] and null["mde"] == 2.8 * null["se"]


def test_dose_response_levels_and_reference():
    out = core.dose_response(_orders(effect=-0.05), "reorder_90", "days", fe=("month",))
    assert list(out["real days"]) == [2.0, 3.0, 4.0, 5.0, 6.0] and out.loc[0, "difference"] == 0
    assert out["difference"].iloc[-1] < -0.1                 # 4 extra days at -0.05 each


def test_balance_test_passes_when_random_and_fails_when_confounded():
    d = _orders()
    d["x"] = np.random.default_rng(9).normal(size=len(d))
    assert core.balance_test(d, "days", numeric=["x"], fe=(core.MODE, "month"))["p"] > 0.01
    d["x"] = d["days"] + np.random.default_rng(9).normal(size=len(d))      # x now predicts delay
    assert core.balance_test(d, "days", numeric=["x"], fe=(core.MODE, "month"))["p"] < 0.001


def test_add_followup_counts_only_orders_after_arrival():
    t0 = pd.Timestamp("2016-01-01")
    items = pd.DataFrame({
        core.ORDER: [1, 2, 3, 4], core.CUSTOMER: [7, 7, 7, 8],
        core.DATE: [t0, t0 + pd.Timedelta(days=1), t0 + pd.Timedelta(days=10), t0 + pd.Timedelta(days=300)],
        core.MODE: "Standard Class", core.PROMISE: 4, core.TARGET_DAYS: [5, 2, 3, 2],
        "Delivery Status": "Late delivery", "Sales": 10.0, "Customer Segment": "Consumer", "Market": "Europe", "Department Name": "Golf"})
    o = core.add_followup(core.order_table(items), windows=(30, 90))
    first = o[o[core.ORDER] == 1].iloc[0]           # arrives day 5; order 2 (day 1) came before arrival; order 3 (day 10) is the next
    assert first["days_to_next"] == 5 and first["reorder_30"] == 1
    assert first["prior_orders"] == 0 and o[o[core.ORDER] == 3].iloc[0]["prior_orders"] == 2
    assert o[o[core.ORDER] == 3].iloc[0]["reorder_30"] == 0         # no later order
    assert bool(o[o[core.ORDER] == 4].iloc[0]["followup_90"]) is False     # data ends before 90 days after arrival
    assert bool(first["followup_90"]) is True
    assert core.order_consistency(items).max() == 0
