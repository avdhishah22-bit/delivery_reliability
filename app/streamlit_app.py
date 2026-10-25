"""Delivery Reliability Studio (Streamlit). Run from the project root:

    streamlit run app/streamlit_app.py

Needs notebooks 04-06 run once, then:  python -m src.models.export_artifacts
"""
import itertools
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import app_charts as ch  # noqa: E402
from src.app_data import promise_for_target, promise_late_rate, threshold_metrics, threshold_sweep  # noqa: E402
from src.app_support import explain, load_all, make_order_frame, score_all_modes, score_order  # noqa: E402
from src.decision.rules import recommend, risk_tier  # noqa: E402

st.set_page_config(page_title="Delivery Reliability Studio", page_icon="🚚", layout="wide")
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
RULES = CFG["rules"]
Y, YR = "Late_delivery_risk", "Days for shipping (real)"

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
.hero {background: linear-gradient(135deg,#4f46e5 0%,#0ea5e9 100%); color: #fff; padding: 26px 30px;
       border-radius: 18px; margin-bottom: 18px;}
.hero h1 {margin: 0 0 6px 0; font-size: 2.0rem; color: #fff;}
.hero p {margin: 0; font-size: 1.02rem; opacity: .93;}
.kpi {border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.08); border-radius: 14px;
      padding: 14px 16px; border-left: 6px solid var(--c, #6366f1); height: 100%;}
.kpi .l {font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; opacity: .7;}
.kpi .v {font-size: 1.85rem; font-weight: 700; line-height: 1.15;}
.kpi .s {font-size: .82rem; opacity: .75;}
.note {border-radius: 12px; padding: 12px 16px; margin: 6px 0 10px 0; border-left: 5px solid var(--c, #6366f1);
       background: rgba(128,128,128,.09);}
.note b {display: block; margin-bottom: 2px;}
.badge {display: inline-block; padding: 4px 16px; border-radius: 999px; color: #fff; font-weight: 700; font-size: 1.1rem;}
</style>
""", unsafe_allow_html=True)

_keys = itertools.count()


def chart(fig):
    st.plotly_chart(fig, key=f"chart{next(_keys)}")


def kpi(label, value, sub="", color="#6366f1"):
    st.markdown(f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div><div class="v">{value}</div>'
                f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def note(title, body, color="#6366f1"):
    st.markdown(f'<div class="note" style="--c:{color}"><b>{title}</b>{body}</div>', unsafe_allow_html=True)


def pct(x, d=0):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}%}"


@st.cache_resource
def get_models():
    return load_all(ROOT / CFG["paths"]["models"])


try:
    art, model, eta = get_models()
except FileNotFoundError:
    st.error("Model files not found. Run notebooks 04-06, then `python -m src.models.export_artifacts`.")
    st.stop()
if "scored" not in art:
    st.error("`models/app_artifacts.joblib` is from an older version. Re-run `python -m src.models.export_artifacts`.")
    st.stop()

SC, DIST, SCORED = art["scorecard"], art["real_days_dist"], art["scored"]
OPT_PROMISE = art["option_promise"].to_dict()
HIGH, LOW = float(RULES["tier_high_prob"]), float(RULES["tier_low_prob"])


# =============================================================== 1. overview
def page_overview():
    k, roc = art["kpis"], art["roc"]
    st.markdown(f"""<div class="hero"><h1>🚚 Delivery Reliability Studio</h1>
    <p>{k['orders']:,} shipped orders from {k['date_min']:%b %Y} to {k['date_max']:%b %Y}. Which shipping option can you trust,
    which orders will arrive late, and what should you do about it?</p></div>""", unsafe_allow_html=True)

    tm = threshold_metrics(SCORED, HIGH, Y)
    c = st.columns(4)
    with c[0]: kpi("Orders analysed", f"{k['orders']:,}", "cancelled orders excluded", "#6366f1")
    with c[1]: kpi("Late deliveries", pct(k["late_rate"], 1), "delivered after the promised date", "#ef4444")
    with c[2]: kpi("Model vs 5-row table", f"{roc['Calibrated LightGBM']['auc']:.3f} / {roc['5-row lookup table']['auc']:.3f}",
                   "ROC-AUC on the held-out test period", "#f59e0b")
    with c[3]: kpi("High-risk alert precision", pct(tm["precision"]),
                   f"flags {pct(tm['flagged_share'])} of orders, catches {pct(tm['recall'])} of late ones", "#22c55e")

    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        chart(ch.late_rate_bars(SC))
    with right:
        st.subheader("The story in four findings")
        s2, st4 = SC.loc["Second Class"], SC.loc["Standard Class"]
        note("1. Lateness is set by the promise, not by slow delivery",
             f"Second Class promises {s2['promised_days']:.0f} days but takes {s2['mean_real_days']:.2f}; Standard promises "
             f"{st4['promised_days']:.0f} and takes {st4['mean_real_days']:.2f}. Same speed, so Second Class is late "
             f"{pct(s2['late_rate'])} of the time against {pct(st4['late_rate'])}.", "#f59e0b")
        note("2. First Class is the only genuinely faster option",
             f"It always takes {SC.loc['First Class', 'mean_real_days']:.0f} days with no variation, against a 1-day promise, "
             f"so every First Class order is late.", "#0ea5e9")
        note("3. Same Day has a noon cut-off",
             "Ordered before 12:00 it arrives the same day. From 12:00 it arrives the next day and counts as late.", "#8b5cf6")
        note("4. A 5-row table is as good as the ML model",
             f"ROC-AUC {roc['5-row lookup table']['auc']:.3f} vs {roc['Calibrated LightGBM']['auc']:.3f}. The rest of the "
             "lateness in Second and Standard Class is random, so about 0.78 is the ceiling on this data.", "#6366f1")

    st.subheader("Where to go next")
    g = st.columns(5)
    guide = [("📦 Shipping Lab", "Compare options, test promises, find a promise you can keep."),
             ("🔎 Order Checker", "Score one order and compare every shipping option for it."),
             ("🚨 Early Warning", "Tune the alert, see false alarms, estimate its money value."),
             ("🗺️ Where & When", "Regions, markets, months and hours, corrected for shipping mode."),
             ("🧪 Model Lab", "ROC, calibration, the lookup table, leakage guard and limits.")]
    for col, (t, d) in zip(g, guide):
        with col:
            note(t, d)


# ============================================================ 2. shipping lab
def page_shipping():
    st.title("📦 Shipping Lab")
    t1, t2, t3, t4 = st.tabs(["Scorecard", "Promise vs reality", "Promise designer", "Profit"])

    with t1:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            chart(ch.late_rate_bars(SC))
        with c2:
            show = SC.copy()
            show["late_rate"] = (show["late_rate"] * 100).round(1).astype(str) + "%"
            show.columns = ["Orders", "Promised (d)", "Real mean (d)", "Real std (d)", "Late"]
            st.dataframe(show)
            note("Reading the table", "Compare 'Promised' with 'Real mean'. The gap between them, not the real speed, "
                 "decides how often an option is late.")
        d = st.slider("I need my order within (days)", 0, 6, 4, key="deadline")
        chart(ch.within_days_bars(art["within"], d))
        st.caption("Same Day is split by the hour the order is placed: before noon vs from noon.")

    with t2:
        opt = st.selectbox("Shipping option", list(DIST.index), index=list(DIST.index).index("Second Class"))
        cur = int(OPT_PROMISE[opt])
        promise = st.slider("What if we promised (days)?", 0, 6, cur, key=f"promise_{opt}")
        c = st.columns(3)
        with c[0]: kpi("Late rate at today's promise", pct(promise_late_rate(DIST.loc[opt], cur)), f"{cur} day(s) promised")
        with c[1]: kpi("Late rate at your promise", pct(promise_late_rate(DIST.loc[opt], promise)), f"{promise} day(s) promised", "#0ea5e9")
        with c[2]:
            real = float((DIST.loc[opt] * DIST.columns).sum())
            kpi("Average real delivery", f"{real:.2f} days", f"{int(art['option_orders'].loc[opt, 'orders']):,} orders", "#22c55e")
        chart(ch.promise_vs_reality(DIST.loc[opt], promise, opt))
        st.caption("Green bars arrive within the promise, red bars are late. Drag the slider and watch the late share move.")

    with t3:
        target = st.slider("Largest late rate I can accept", 0.0, 0.5, 0.10, 0.01, format="%.2f")
        rows = []
        for opt in DIST.index:
            cur = int(OPT_PROMISE[opt]); rec = promise_for_target(DIST.loc[opt], target)
            rows.append({"Option": opt, "Promise today (d)": cur, "Late today": pct(promise_late_rate(DIST.loc[opt], cur)),
                         "Promise needed (d)": rec, "Late at that promise": pct(promise_late_rate(DIST.loc[opt], rec)),
                         "Change (d)": rec - cur})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        chart(ch.promise_curves(DIST, OPT_PROMISE))
        note("What this means", "Second Class would need to promise about as long as Standard to be as reliable, so the "
             "extra cost of Second Class buys no real speed. First Class can never reach a low late rate with a 1-day promise.",
             "#f59e0b")

    with t4:
        if art.get("profit_mode") is not None:
            chart(ch.profit_fig(art["profit_mode"], Y))
            st.caption(f"Column used: {art['profit_col']}. Late and on-time orders earn similar profit per order, "
                       "so late delivery is a service problem here, not a margin problem.")
        else:
            st.info("No profit column was found in the cleaned data.")


# ============================================================ 3. order checker
def apply_preset(**kw):
    for key, val in kw.items():
        st.session_state[key] = val


def page_checker():
    st.title("🔎 Order Checker")
    opts, nd = art["options"], art["num_defaults"]
    test_start = pd.Timestamp(art["split_info"]["Test"]["start"]).date()
    for k, v in {"o_mode": "Standard Class", "o_date": test_start, "o_hour": 10, "o_sales": float(nd["Sales"]),
                 "o_price": float(nd["Product Price"]), "o_qty": int(nd["Order Item Quantity"]),
                 "o_disc": float(np.clip(round(nd["Order Item Discount Rate"], 2), 0.0, 0.3))}.items():
        st.session_state.setdefault(k, v)

    p = st.columns(4)
    presets = [("Typical Standard order", dict(o_mode="Standard Class", o_hour=10)),
               ("Evening Same Day", dict(o_mode="Same Day", o_hour=17)),
               ("Big Second Class basket", dict(o_mode="Second Class", o_hour=14, o_sales=450.0, o_qty=4)),
               ("Premium First Class", dict(o_mode="First Class", o_hour=9, o_sales=300.0))]
    for col, (label, kw) in zip(p, presets):
        col.button(label, on_click=apply_preset, kwargs=kw, key=f"preset_{label}")

    c1, c2, c3 = st.columns(3)
    modes = list(art["scheduled_days"].keys())
    mode = c1.selectbox("Shipping mode", modes, key="o_mode")
    day = c1.date_input("Order date", key="o_date")
    hour = c1.slider("Order hour", 0, 23, key="o_hour")
    type_ = c2.selectbox("Payment type", opts["Type"], key="o_type")
    market = c2.selectbox("Market", opts["Market"], key="o_market")
    region = c2.selectbox("Order region", opts["Order Region"], key="o_region")
    segment = c2.selectbox("Customer segment", opts["Customer Segment"], key="o_segment")
    dept = c3.selectbox("Department", opts["Department Name"], key="o_dept")
    cats = art.get("category_by_dept", {}).get(dept) or opts["Category Name"]
    if st.session_state.get("o_cat") not in cats:
        st.session_state["o_cat"] = cats[0]
    category = c3.selectbox("Category", cats, key="o_cat")
    sales = c3.number_input("Order value (Sales)", min_value=0.0, key="o_sales")
    price = c3.number_input("Product price", min_value=0.0, key="o_price")
    quantity = c3.number_input("Quantity", min_value=1, step=1, key="o_qty")
    discount = c3.slider("Discount rate", 0.0, 0.3, step=0.01, key="o_disc")

    order_dt = datetime.combine(day, datetime.min.time()) + timedelta(hours=int(hour))
    order = dict(order_dt=order_dt, type_=type_, market=market, region=region, department=dept, category=category,
                 segment=segment, sales=float(sales), price=float(price), quantity=int(quantity), discount_rate=float(discount))
    frame = make_order_frame(art, mode=mode, **order)
    r = score_order(art, model, eta, frame)
    tier = risk_tier(r["p_late"], RULES)
    promised = art["scheduled_days"][mode]
    lo, hi = math.floor(r["eta_lo"] + 1e-9), math.ceil(r["eta_hi"] - 1e-9)

    st.divider()
    g, right = st.columns([1, 1.25])
    with g:
        chart(ch.gauge(r["p_late"], LOW, HIGH))
        st.markdown(f'<div style="text-align:center"><span class="badge" style="background:{ch.TIER_COLORS[tier]}">'
                    f'{tier} risk</span></div>', unsafe_allow_html=True)
    with right:
        k = st.columns(2)
        with k[0]: kpi("Model says", pct(r["p_late"]), "calibrated chance of being late", ch.TIER_COLORS[tier])
        with k[1]: kpi("Lookup table says", pct(r["p_lookup"]), f"row: {r['lookup_key']}", "#94a3b8")
        st.write("")
        win = f"{lo} to {hi} days" if lo != hi else f"{lo} days"
        note("Expected delivery",
             f"<b style='display:inline'>{win}</b> after ordering: {(order_dt + timedelta(days=lo)):%a %d %b} to "
             f"{(order_dt + timedelta(days=hi)):%a %d %b}. Promised: {promised} day(s).", "#0ea5e9")
        acts = recommend(r["p_late"], float(sales), r["interval_days"], RULES)
        note("Recommended action", "<br>".join("• " + a for a in acts), ch.TIER_COLORS[tier])
    chart(ch.delivery_timeline(r["eta_lo"], r["eta_hi"], r["eta_point"], promised))

    t1, t2 = st.tabs(["Compare every shipping option", "Why this score"])
    with t1:
        table = score_all_modes(art, model, eta, **order)
        chart(ch.compare_modes_fig(table))
        safest = table.sort_values(["P(late) model", "Expected days"]).iloc[0]
        fastest = table.sort_values("Expected days").iloc[0]
        c = st.columns(2)
        with c[0]: kpi("Most reliable option", safest["Shipping mode"], f"{pct(safest['P(late) model'])} chance of late", "#22c55e")
        with c[1]: kpi("Fastest option", fastest["Shipping mode"], f"about {fastest['Expected days']:.1f} days", "#0ea5e9")
        show = pd.DataFrame({"Shipping mode": table["Shipping mode"], "Promised (d)": table["Promised (days)"],
                             "P(late) model": table["P(late) model"].map(pct), "P(late) lookup": table["P(late) lookup"].map(pct),
                             "Expected (d)": table["Expected days"].round(1),
                             "Window (d)": [f"{math.floor(a + 1e-9)} to {math.ceil(b - 1e-9)}" for a, b in zip(table["Earliest"], table["Latest"])]})
        st.dataframe(show, hide_index=True)
    with t2:
        try:
            chart(ch.driver_bars(explain(model, frame, top=8)))
            st.caption("Bars are TreeSHAP contributions in log-odds. Shipping mode and the promised days dominate, as expected.")
        except Exception as e:  # SHAP needs the real LightGBM pipeline
            st.info(f"Explanation not available: {e}")


# ========================================================== 4. early warning
def page_warning():
    st.title("🚨 Early-Warning Center")
    f1, f2, f3 = st.columns(3)
    modes = f1.multiselect("Shipping modes", sorted(SCORED["Shipping Mode"].unique()), default=sorted(SCORED["Shipping Mode"].unique()))
    markets = f2.multiselect("Markets", sorted(SCORED["Market"].unique()), default=sorted(SCORED["Market"].unique()))
    regions = f3.multiselect("Regions (empty = all)", sorted(SCORED["Order Region"].unique()))
    mask = SCORED["Shipping Mode"].isin(modes) & SCORED["Market"].isin(markets)
    if regions:
        mask &= SCORED["Order Region"].isin(regions)
    f = SCORED[mask]
    if f.empty:
        st.warning("No orders match these filters."); return

    thr = st.slider("Alert threshold: flag orders whose calibrated P(late) is at least", 0.05, 0.95, HIGH, 0.01)
    m = threshold_metrics(f, thr, Y)
    c = st.columns(4)
    with c[0]: kpi("Orders flagged", pct(m["flagged_share"], 1), f"{m['flagged']:,} of {m['orders']:,} test orders", "#6366f1")
    with c[1]: kpi("Precision", pct(m["precision"]), "flagged orders that really are late", "#22c55e")
    with c[2]: kpi("Recall", pct(m["recall"]), "late orders the alert catches", "#f59e0b")
    with c[3]: kpi("False alarms per catch", "-" if math.isnan(m["false_per_catch"]) else f"{m['false_per_catch']:.2f}",
                   "wrongly flagged orders per late order caught", "#ef4444")

    t1, t2, t3, t4 = st.tabs(["Trade-off", "Business value", "Watchlist", "Risk tiers"])
    with t1:
        a, b = st.columns([1.4, 1])
        with a: chart(ch.sweep_fig(threshold_sweep(f, Y), thr))
        with b: chart(ch.outcome_bars(m))
        st.caption("The curves step because the calibrated scores form a few clusters (one per shipping option), "
                   "so moving the threshold between clusters changes nothing.")
    with t2:
        st.caption("Illustrative: replace these assumptions with your own numbers. The model supplies precision and recall.")
        a, b, c2, d = st.columns(4)
        vol = a.number_input("Orders per month", 1000, 10_000_000, 20_000, 1000)
        late_cost = b.number_input("Cost of one late order", 0.0, 1000.0, 12.0, 1.0)
        act_cost = c2.number_input("Cost of acting on one alert", 0.0, 1000.0, 1.5, 0.5)
        fix = d.slider("Alerts that prevent the damage", 0.0, 1.0, 0.30, 0.05)
        flagged_n = m["flagged_share"] * vol
        tp_n = m["tp"] / m["orders"] * vol
        gross, spent = tp_n * fix * late_cost, flagged_n * act_cost
        k = st.columns(3)
        with k[0]: kpi("Net value per month", f"{gross - spent:,.0f}", "cost avoided minus cost of acting", "#6366f1")
        with k[1]: kpi("Alerts per month", f"{flagged_n:,.0f}", f"{tp_n:,.0f} of them truly late", "#f59e0b")
        be = act_cost / (m["precision"] * late_cost) if m["precision"] and late_cost else float("nan")
        with k[2]: kpi("Break-even prevention rate", pct(be), "alerts must fix at least this share", "#22c55e")
        chart(ch.cost_waterfall(gross, spent))
    with t3:
        w = f[f["p_late"] >= thr].copy()
        w["priority"] = w["p_late"] * w["Sales"]
        w = w.sort_values("priority", ascending=False).head(300)
        view = w[["order date (DateOrders)", "Shipping Mode", "Order Region", "Market", "Sales", "p_late", "priority"]]
        view = view.rename(columns={"order date (DateOrders)": "Order date", "p_late": "P(late)", "priority": "Priority (P x value)"})
        st.caption("Sorted by priority = chance of being late x order value, so the costliest problems come first.")
        st.dataframe(view, hide_index=True, column_config={
            "P(late)": st.column_config.ProgressColumn("P(late)", min_value=0.0, max_value=1.0, format="%.2f"),
            "Sales": st.column_config.NumberColumn("Sales", format="%.0f"),
            "Priority (P x value)": st.column_config.NumberColumn(format="%.0f")})
        st.download_button("Download watchlist (CSV)", view.to_csv(index=False).encode(), "watchlist.csv", "text/csv")
    with t4:
        low = st.slider("Low tier below", 0.01, 0.5, LOW, 0.01)
        tier = np.where(f["p_late"] >= thr, "High", np.where(f["p_late"] < low, "Low", "Medium"))
        s = f.assign(tier=tier).groupby("tier")[Y].agg(orders="size", late_rate="mean").reindex(["High", "Medium", "Low"]).dropna()
        s["share"] = s["orders"] / s["orders"].sum()
        chart(ch.tier_bars(s))
        st.dataframe(s.round(3))


# ============================================================ 5. where / when
def page_where_when():
    st.title("🗺️ Where & When")
    t1, t2, t3, t4 = st.tabs(["Region × shipping mode", "Markets", "Calendar", "Departments"])
    explain_txt = ("Raw late rates mislead: a region using more Second Class looks unreliable only because the promise is "
                   "short. Bars show the gap AFTER controlling for each place's shipping-mode mix. Near zero means no real difference.")
    with t1:
        rm = art["region_mode"]
        vals = rm.pivot(index="Order Region", columns="Shipping Mode", values="late_rate")
        cnt = rm.pivot(index="Order Region", columns="Shipping Mode", values="orders")
        vals = vals.where(cnt >= 30)
        vals = vals.reindex(art["region_table"]["Order Region"])
        chart(ch.heatmap(vals, "Late rate (%) by region and shipping mode (cells with fewer than 30 orders hidden)"))
        chart(ch.gap_bars(art["region_table"], "Order Region"))
        note("Read this correctly", explain_txt)
    with t2:
        chart(ch.gap_bars(art["market_table"], "Market", min_orders=100))
        mt = art["market_table"].copy()
        for c_ in ("late_rate", "expected", "gap"):
            mt[c_] = (mt[c_] * 100).round(1)
        st.dataframe(mt.round(2), hide_index=True)
        note("Read this correctly", explain_txt)
    with t3:
        modes = list(SC.index)
        pick = st.multiselect("Shipping modes", modes, default=modes)
        if pick:
            v = st.radio("Show", ["late_rate", "mean_real_days"], horizontal=True,
                         format_func=lambda x: "Late rate" if x == "late_rate" else "Average real delivery days")
            chart(ch.monthly_lines(art["monthly"], pick, v))
        hm = st.selectbox("Hour × weekday heatmap for", modes, index=modes.index("Same Day") if "Same Day" in modes else 0)
        chart(ch.hour_weekday_heatmap(art["hour_weekday"][art["hour_weekday"]["Shipping Mode"] == hm],
                                      f"{hm}: late rate by order time"))
        note("What to look for", "Same Day lights up from 12:00 because of the noon cut-off. Standard and Second Class stay "
             "flat across hours and weekdays: their lateness is random with respect to when you order.", "#8b5cf6")
    with t4:
        chart(ch.gap_bars(art["dept_table"], "Department Name", min_orders=100))
        note("Read this correctly", explain_txt)


# ============================================================== 6. model lab
def page_model_lab():
    st.title("🧪 Model Lab")
    roc = art["roc"]
    c = st.columns(3)
    for col, (name, color) in zip(c, [("Calibrated LightGBM", "#6366f1"), ("5-row lookup table", "#f59e0b"),
                                      ("Shipping mode only", "#94a3b8")]):
        with col:
            kpi(name, f"{roc[name]['auc']:.3f}", "ROC-AUC, held-out test period", color)
    st.write("")
    a, b = st.columns(2)
    with a: chart(ch.roc_fig(roc))
    with b: chart(ch.calibration_fig(art["calibration"]))
    chart(ch.split_timeline(art["split_info"]))

    t1, t2, t3 = st.tabs(["The 5-row lookup table", "Leakage guard", "Limitations"])
    with t1:
        rows = []
        for opt in DIST.index:
            rows.append({"Option": opt, "Promised (d)": int(OPT_PROMISE[opt]),
                         "Average real (d)": round(float((DIST.loc[opt] * DIST.columns).sum()), 2),
                         "P(late)": pct(art["lookup"].get(opt, float("nan")), 1),
                         "Orders": int(art["option_orders"].loc[opt, "orders"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        note("Why this matters", "A model that cannot beat five numbers has not found hidden structure. It has learned the "
             "promise plus the noon cut-off. Reporting that honestly is the result.")
    with t2:
        st.markdown("""
**Columns never used as features** (known only after shipping): `Days for shipping (real)`, `Delivery Status`,
`Order Status`, `shipping date (DateOrders)`. A unit test fails if any of them reaches the model.

**Time-based split**: train on the oldest 70%, validate on the next 15%, test on the newest 15%.

| Check (notebook runs) | ROC-AUC |
|---|---|
| Time split | 0.779 |
| Random split | 0.801 (inflated: 89% of rows belong to multi-item orders) |
| Time split without order-time features | 0.743 |
""")
    with t3:
        st.markdown("""
- DataCo is a synthetic dataset: lateness is generated from the promise and a Same Day noon cut-off plus random noise.
- There are no supplier IDs and no shipping-cost column, so supplier ranking and cost optimisation are out of scope.
- The two logistics files showed no honest predictive signal and are used only as leakage audits.
- Cancelled and suspected-fraud orders are excluded because the data labels them 'on time'.
- The money page uses assumptions you set yourself; it is a way to reason, not a forecast.
""")


# ================================================================== router
PAGES = {"🏠 Overview": page_overview, "📦 Shipping Lab": page_shipping, "🔎 Order Checker": page_checker,
         "🚨 Early Warning": page_warning, "🗺️ Where & When": page_where_when, "🧪 Model Lab": page_model_lab}
st.sidebar.title("Delivery Reliability")
choice = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
st.sidebar.caption(f"Test period: {pd.Timestamp(art['split_info']['Test']['start']):%b %Y} to "
                   f"{pd.Timestamp(art['split_info']['Test']['end']):%b %Y}")
PAGES[choice]()
