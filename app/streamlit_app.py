"""Delivery Reliability (Streamlit). Run from the project root:

    streamlit run app/streamlit_app.py

Needs notebooks 04-06 run once, then:  python -m src.models.export_artifacts
"""
import itertools
import math
import sys
from datetime import datetime, timedelta
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

st.set_page_config(page_title="Delivery Reliability", layout="wide")
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
RULES = CFG["rules"]
Y = "Late_delivery_risk"

# --------------------------------------------------------------------- style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root {--ink:#1B2A3A; --page:#F3F5F7; --signal:#FFC633; --signal-soft:#FFF1BF; --muted:#5B6773; --line:#D5DCE3;
       --late:#C8372D; --ok:#2E7D5B;}
.stApp {background: var(--page);}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp th, .stApp td,
.stApp [data-baseweb="tab"] {font-family: 'IBM Plex Sans', system-ui, sans-serif;}
.stApp h1, .stApp h2, .stApp h3, .headline {font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; color: var(--ink);}
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu {display: none !important;}
.block-container {max-width: 1100px; padding: 1.1rem 1.5rem 3rem;}

/* top bar */
.st-key-topnav {background: var(--ink); border-radius: 6px; padding: .5rem 1.1rem; margin-bottom: 2rem;}
.brand {font-family: 'Barlow Condensed', sans-serif; font-size: 1.65rem; font-weight: 700; color: #fff; line-height: 2.2rem; white-space: nowrap;}
.brand i {display: inline-block; width: .72rem; height: .72rem; background: var(--signal); margin-right: .55rem;}
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"] {gap: .2rem; justify-content: flex-end; flex-wrap: wrap;}
.st-key-topnav [data-testid="stRadio"] label {padding: .38rem .85rem; border-radius: 4px; cursor: pointer; margin: 0;}
.st-key-topnav [data-testid="stRadio"] label > div:first-child {display: none;}
.st-key-topnav [data-testid="stRadio"] label p {color: #D3DBE4; font-size: .95rem; font-weight: 500; margin: 0; white-space: nowrap;}
.st-key-topnav [data-testid="stRadio"] label:hover p {color: #fff;}
.st-key-topnav [data-testid="stRadio"] label:has(input:checked) {background: var(--signal);}
.st-key-topnav [data-testid="stRadio"] label:has(input:checked) p {color: var(--ink); font-weight: 600;}

/* page headings */
.stApp .pagehead h1 {font-size: 3.3rem !important; line-height: 1.0 !important; font-weight: 700; margin: 0 0 .7rem !important; max-width: 22ch; padding: 0 !important;}
.pagehead .lede {font-size: 1.12rem; line-height: 1.6; color: #34424F; max-width: 62ch; margin: 0 0 1.4rem;}
.stApp h2.sec {font-size: 1.9rem !important; font-weight: 600; margin: 2.4rem 0 .6rem !important; padding: 0 !important;}
.secsub {color: var(--muted); max-width: 62ch; margin: 0 0 .9rem; line-height: 1.5;}
.stApp h3.fh {font-size: 1.4rem !important; font-weight: 600; margin: 0 0 .3rem !important; padding: 0 !important;}
p.fp {margin: 0 0 1.5rem; color: #34424F; line-height: 1.6; max-width: 46ch;}
.colhead {font-weight: 600; margin: .2rem 0 .3rem; color: var(--ink);}

/* figures and notes */
.kpi {padding: .1rem 0 .9rem;}
.kpi .v {font-family: 'Barlow Condensed', sans-serif; font-size: 2.6rem; font-weight: 600; line-height: 1; color: var(--ink);}
.kpi .l {font-weight: 500; margin-top: .3rem;}
.kpi .s {font-size: .85rem; color: var(--muted); line-height: 1.35;}
.aside {background: var(--signal-soft); padding: .7rem 1rem; border-radius: 3px; margin: .7rem 0; color: #2A2F36; line-height: 1.55;}
.badge {display: inline-block; padding: .3rem 1rem; border-radius: 3px; color: #fff; font-weight: 600;}
.foot {margin-top: 3.5rem; border-top: 1px solid var(--line); padding-top: .9rem; color: var(--muted); font-size: .85rem; max-width: 80ch;}

/* controls */
.stApp .stButton > button, .stApp .stDownloadButton > button {background: var(--ink); color: #fff; border: 1px solid var(--ink);
    border-radius: 4px; font-weight: 600; padding: .45rem 1.15rem;}
.stApp .stButton > button p, .stApp .stDownloadButton > button p {color: inherit; font-weight: 600;}
.stApp .stButton > button:hover, .stApp .stDownloadButton > button:hover {background: var(--signal); border-color: var(--signal); color: var(--ink);}
.st-key-presets .stButton > button {background: transparent; color: var(--ink);}
.st-key-presets .stButton > button:hover {background: var(--signal);}
.st-key-links .stButton > button {background: none; border: none; color: var(--ink); text-decoration: underline;
    text-underline-offset: 4px; text-decoration-color: var(--signal); text-decoration-thickness: 3px; padding: 0; justify-content: flex-start;}
.st-key-links .stButton > button:hover {background: none; color: var(--ink);}
[data-baseweb="tab-highlight"] {background: var(--signal) !important; height: 3px !important;}
button[data-baseweb="tab"] p {font-weight: 600; font-size: 1rem;}
@media (max-width: 760px) {.stApp .pagehead h1 {font-size: 2.4rem !important;} .brand {font-size: 1.35rem;}}
</style>
""", unsafe_allow_html=True)

_keys = itertools.count()


def chart(fig):
    st.plotly_chart(fig, key=f"chart{next(_keys)}", config={"displayModeBar": False})


def box(key):
    try:
        return st.container(key=key)
    except TypeError:          # older Streamlit without container keys: styling is lost, nothing else
        return st.container()


def page_header(title, lede=""):
    st.markdown(f'<div class="pagehead"><h1>{title}</h1>' + (f'<p class="lede">{lede}</p>' if lede else "") + "</div>",
                unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<h2 class="sec">{title}</h2>' + (f'<p class="secsub">{sub}</p>' if sub else ""), unsafe_allow_html=True)


def kpi(label, value, sub="", tone=None):
    style = f' style="color:{tone}"' if tone else ""
    st.markdown(f'<div class="kpi"><div class="v"{style}>{value}</div><div class="l">{label}</div>'
                f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def aside(text, lead=""):
    st.markdown(f'<div class="aside">{"<b>" + lead + "</b> " if lead else ""}{text}</div>', unsafe_allow_html=True)


def pct(x, d=0):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}%}"


@st.cache_resource
def get_models():
    return load_all(ROOT / CFG["paths"]["models"])


try:
    art, model, eta = get_models()
except FileNotFoundError:
    st.error("Model files not found. Run notebooks 04 to 06, then run `python -m src.models.export_artifacts`.")
    st.stop()
if "scored" not in art:
    st.error("`models/app_artifacts.joblib` was built by an older version. Run `python -m src.models.export_artifacts` again.")
    st.stop()

SC, DIST, SCORED = art["scorecard"], art["real_days_dist"], art["scored"]
OPT_PROMISE = art["option_promise"].to_dict()
HIGH, LOW = float(RULES["tier_high_prob"]), float(RULES["tier_low_prob"])
NAV = ["Overview", "Shipping options", "Check an order", "Early warnings", "Where and when", "How it works"]


def goto(page):
    st.session_state["nav"] = page


# =============================================================== overview
def page_overview():
    k, roc = art["kpis"], art["roc"]
    tm = threshold_metrics(SCORED, HIGH, Y)
    s0, s1, s2, s4 = (SC.loc[m] for m in ("Same Day", "First Class", "Second Class", "Standard Class"))
    page_header("Late deliveries are decided by the promise, not the courier.",
                f"We went through {k['orders']:,} shipped DataCo orders to see which shipping options keep their word, and "
                "whether a model can warn us before an order runs late. Two of the four options deliver at the same speed. "
                "What differs is how much each one promised.")
    with box("cta"):
        a, b, _ = st.columns([1, 1.2, 4])
        a.button("Check an order", on_click=goto, args=("Check an order",), key="cta1")
        b.button("Compare shipping options", on_click=goto, args=("Shipping options",), key="cta2")
    chart(ch.promise_gap_chart(SC))
    aside("Hollow circle: the promised delivery time. Filled circle: how long it actually takes on average. "
          "A red bar means the option is slower than it promised.")

    section("What the orders show")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<h3 class="fh">Second Class is Standard with a shorter promise</h3><p class="fp">Both take about '
                    f'{s4["mean_real_days"]:.1f} days. Second Class promises {s2["promised_days"]:.0f} and Standard '
                    f'{s4["promised_days"]:.0f}, so Second Class is late {pct(s2["late_rate"])} of the time and Standard '
                    f'{pct(s4["late_rate"])}.</p>', unsafe_allow_html=True)
        st.markdown(f'<h3 class="fh">Same Day stops at noon</h3><p class="fp">Ordered before 12:00, it arrives the same day. '
                    f'Ordered after, it arrives the next day and counts as late. Overall {pct(s0["late_rate"])} of Same Day '
                    f'orders are late.</p>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<h3 class="fh">First Class is faster, and still late</h3><p class="fp">Every First Class order arrives in '
                    f'{s1["mean_real_days"]:.0f} days. The promise is {s1["promised_days"]:.0f}, so {pct(s1["late_rate"])} are '
                    f'late.</p>', unsafe_allow_html=True)
        st.markdown(f'<h3 class="fh">Five numbers do the model\'s job</h3><p class="fp">A table with one late rate per option '
                    f'scores {roc["5-row lookup table"]["auc"]:.3f} on ROC-AUC, against {roc["Calibrated LightGBM"]["auc"]:.3f} '
                    f'for the model. Beyond that, lateness in Second and Standard Class is random.</p>', unsafe_allow_html=True)

    section("Where to look next")
    rows = [("Shipping options", "Compare reliability, try a different promise, find one an option can keep."),
            ("Check an order", "Score a single order and see every shipping option for it side by side."),
            ("Early warnings", f"Choose how sure the model must be before it raises an alert. At today's setting "
                               f"{pct(tm['precision'])} of alerts are right and {pct(tm['recall'])} of late orders are caught."),
            ("Where and when", "Regions, markets, months and hours, corrected for shipping option."),
            ("How it works", "Accuracy, calibration, and what the model cannot do.")]
    with box("links"):
        for name, desc in rows:
            a, b = st.columns([1, 3])
            a.button(name, on_click=goto, args=(name,), key=f"go_{name}")
            b.markdown(f'<div style="color:#34424F;padding-top:.3rem">{desc}</div>', unsafe_allow_html=True)


# ========================================================= shipping options
def page_shipping():
    page_header("Which shipping option keeps its promise?",
                "Late means the order arrived after the date the customer was promised. The tabs show how often each option "
                "misses, what happens if the promise changes, and whether being late costs money.")
    t1, t2, t3, t4 = st.tabs(["How often each is late", "Test a promise", "Find a promise it can keep", "Profit"])

    with t1:
        c1, c2 = st.columns([1.15, 1])
        with c1:
            chart(ch.late_rate_bars(SC))
        with c2:
            show = SC.copy()
            show["late_rate"] = (show["late_rate"] * 100).round(1).astype(str) + "%"
            show.columns = ["Orders", "Promised (days)", "Takes (days)", "Spread (days)", "Late"]
            st.dataframe(show)
            aside("Compare what is promised with what is delivered. That gap decides how often an option is late, "
                  "more than its speed does.")
        section("Need it by a deadline?")
        d = st.slider("I need my order within this many days", 0, 6, 4, key="deadline")
        chart(ch.within_days_bars(art["within"], d))
        st.caption("Same Day is split by the hour the order is placed: before noon and from noon.")

    with t2:
        opt = st.selectbox("Shipping option", list(DIST.index), index=list(DIST.index).index("Second Class"))
        cur = int(OPT_PROMISE[opt])
        promise = st.slider("If we promised this many days", 0, 6, cur, key=f"promise_{opt}")
        c = st.columns(3)
        with c[0]: kpi("Late at today's promise", pct(promise_late_rate(DIST.loc[opt], cur)), f"{cur} day(s) promised")
        with c[1]: kpi("Late at this promise", pct(promise_late_rate(DIST.loc[opt], promise)), f"{promise} day(s) promised")
        with c[2]:
            real = float((DIST.loc[opt] * DIST.columns).sum())
            kpi("Average delivery time", f"{real:.2f} days", f"{int(art['option_orders'].loc[opt, 'orders']):,} orders")
        chart(ch.promise_vs_reality(DIST.loc[opt], promise, opt))
        st.caption("Green bars arrive within the promise. Red bars are late.")

    with t3:
        target = st.slider("Highest late rate we can accept", 0.0, 0.5, 0.10, 0.01, format="%.2f")
        rows = []
        for opt in DIST.index:
            cur = int(OPT_PROMISE[opt]); rec = promise_for_target(DIST.loc[opt], target)
            rows.append({"Option": opt, "Promise today (days)": cur, "Late today": pct(promise_late_rate(DIST.loc[opt], cur)),
                         "Promise needed (days)": rec, "Late at that promise": pct(promise_late_rate(DIST.loc[opt], rec)),
                         "Change (days)": rec - cur})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        chart(ch.promise_curves(DIST, OPT_PROMISE))
        aside("Second Class would need to promise about as long as Standard to be as reliable. It delivers just as slowly; "
              "it only promises less. First Class cannot reach a low late rate with a one-day promise.")

    with t4:
        if art.get("profit_mode") is not None:
            chart(ch.profit_fig(art["profit_mode"], Y))
            st.caption(f"Profit column used: {art['profit_col']}. Late and on-time orders earn about the same profit, so "
                       "lateness here is a service problem, not a margin problem.")
        else:
            st.info("The cleaned data has no profit column, so this tab is empty.")


# ============================================================ check an order
def apply_preset(**kw):
    for key, val in kw.items():
        st.session_state[key] = val


def page_checker():
    page_header("Check one order",
                "Enter an order the way a customer would place it. You get the chance it arrives late, the window it should "
                "arrive in, and how the other shipping options compare.")
    opts, nd = art["options"], art["num_defaults"]
    test_start = pd.Timestamp(art["split_info"]["Test"]["start"]).date()
    for k, v in {"o_mode": "Standard Class", "o_date": test_start, "o_hour": 10, "o_sales": float(nd["Sales"]),
                 "o_price": float(nd["Product Price"]), "o_qty": int(nd["Order Item Quantity"]),
                 "o_disc": float(np.clip(round(nd["Order Item Discount Rate"], 2), 0.0, 0.3))}.items():
        st.session_state.setdefault(k, v)

    st.caption("Start from an example, then change anything.")
    with box("presets"):
        p = st.columns(4)
        presets = [("Standard order", dict(o_mode="Standard Class", o_hour=10)),
                   ("Same Day, evening", dict(o_mode="Same Day", o_hour=17)),
                   ("Large Second Class order", dict(o_mode="Second Class", o_hour=14, o_sales=450.0, o_qty=4)),
                   ("First Class order", dict(o_mode="First Class", o_hour=9, o_sales=300.0))]
        for col, (label, kw) in zip(p, presets):
            col.button(label, on_click=apply_preset, kwargs=kw, key=f"preset_{label}")

    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="colhead">Shipping and timing</div>', unsafe_allow_html=True)
    mode = c1.selectbox("Shipping mode", list(art["scheduled_days"].keys()), key="o_mode")
    day = c1.date_input("Order date", key="o_date")
    hour = c1.slider("Hour of the day", 0, 23, key="o_hour")
    c2.markdown('<div class="colhead">Customer</div>', unsafe_allow_html=True)
    type_ = c2.selectbox("Payment type", opts["Type"], key="o_type")
    market = c2.selectbox("Market", opts["Market"], key="o_market")
    region = c2.selectbox("Order region", opts["Order Region"], key="o_region")
    segment = c2.selectbox("Customer segment", opts["Customer Segment"], key="o_segment")
    c3.markdown('<div class="colhead">Product</div>', unsafe_allow_html=True)
    dept = c3.selectbox("Department", opts["Department Name"], key="o_dept")
    cats = art.get("category_by_dept", {}).get(dept) or opts["Category Name"]
    if st.session_state.get("o_cat") not in cats:
        st.session_state["o_cat"] = cats[0]
    category = c3.selectbox("Category", cats, key="o_cat")
    sales = c3.number_input("Order value", min_value=0.0, key="o_sales")
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

    section("What to expect")
    g, right = st.columns([1, 1.3])
    with g:
        chart(ch.gauge(r["p_late"], LOW, HIGH))
        st.markdown(f'<span class="badge" style="background:{ch.TIER_COLORS[tier]}">{tier} risk</span>', unsafe_allow_html=True)
    with right:
        k = st.columns(2)
        with k[0]: kpi("Chance of being late", pct(r["p_late"]), "model, calibrated", ch.TIER_COLORS[tier])
        with k[1]: kpi("Lookup table says", pct(r["p_lookup"]), f"row: {r['lookup_key']}")
        win = f"{lo} to {hi} days" if lo != hi else f"{lo} days"
        aside(f"{win} after ordering, between {(order_dt + timedelta(days=lo)):%a %d %b} and "
              f"{(order_dt + timedelta(days=hi)):%a %d %b}. The promise was {promised} day(s).", "Expected delivery:")
        acts = recommend(r["p_late"], float(sales), r["interval_days"], RULES)
        aside("<br>".join(acts), "Suggested action:")
    chart(ch.delivery_timeline(r["eta_lo"], r["eta_hi"], r["eta_point"], promised))

    t1, t2 = st.tabs(["Compare every shipping option", "Why this score"])
    with t1:
        table = score_all_modes(art, model, eta, **order)
        chart(ch.compare_modes_fig(table))
        safest = table.sort_values(["P(late) model", "Expected days"]).iloc[0]
        fastest = table.sort_values("Expected days").iloc[0]
        c = st.columns(2)
        with c[0]: kpi("Most reliable option", safest["Shipping mode"], f"{pct(safest['P(late) model'])} chance of being late")
        with c[1]: kpi("Fastest option", fastest["Shipping mode"], f"about {fastest['Expected days']:.1f} days")
        show = pd.DataFrame({"Shipping mode": table["Shipping mode"], "Promised (days)": table["Promised (days)"],
                             "Late, model": table["P(late) model"].map(pct), "Late, lookup table": table["P(late) lookup"].map(pct),
                             "Expected (days)": table["Expected days"].round(1),
                             "Window (days)": [f"{math.floor(a + 1e-9)} to {math.ceil(b - 1e-9)}" for a, b in zip(table["Earliest"], table["Latest"])]})
        st.dataframe(show, hide_index=True)
    with t2:
        try:
            chart(ch.driver_bars(explain(model, frame, top=8)))
            st.caption("Bars show each factor's push on the late score (TreeSHAP, in log-odds).")
        except Exception as e:  # needs the real LightGBM pipeline
            st.info(f"The explanation is not available: {e}")


# ============================================================ early warnings
def page_warning():
    page_header("Who needs a call before the package is late?",
                "Choose how sure the model must be before it raises an alert. Below, you can see what that choice costs in "
                "false alarms and how many late orders it catches.")
    with st.expander("Filter the orders"):
        f1, f2, f3 = st.columns(3)
        modes = f1.multiselect("Shipping modes", sorted(SCORED["Shipping Mode"].unique()), default=sorted(SCORED["Shipping Mode"].unique()))
        markets = f2.multiselect("Markets", sorted(SCORED["Market"].unique()), default=sorted(SCORED["Market"].unique()))
        regions = f3.multiselect("Regions (none selected means all)", sorted(SCORED["Order Region"].unique()))
    mask = SCORED["Shipping Mode"].isin(modes) & SCORED["Market"].isin(markets)
    if regions:
        mask &= SCORED["Order Region"].isin(regions)
    f = SCORED[mask]
    if f.empty:
        st.warning("No orders match these filters. Select at least one shipping mode and one market."); return

    thr = st.slider("Raise an alert when the chance of being late is at least", 0.05, 0.95, HIGH, 0.01)
    m = threshold_metrics(f, thr, Y)
    c = st.columns(4)
    with c[0]: kpi("Orders flagged", pct(m["flagged_share"], 1), f"{m['flagged']:,} of {m['orders']:,} test orders")
    with c[1]: kpi("Alerts that are right", pct(m["precision"]), "flagged orders that really are late")
    with c[2]: kpi("Late orders caught", pct(m["recall"]), "share of all late orders")
    with c[3]: kpi("False alarms per catch", "-" if math.isnan(m["false_per_catch"]) else f"{m['false_per_catch']:.2f}",
                   "wrong alerts for each late order caught")

    t1, t2, t3, t4 = st.tabs(["Trade-off", "What it is worth", "Watchlist", "Risk tiers"])
    with t1:
        a, b = st.columns([1.4, 1])
        with a: chart(ch.sweep_fig(threshold_sweep(f, Y), thr))
        with b: chart(ch.outcome_bars(m))
        st.caption("The curves move in steps because calibrated scores fall in a few clusters, one per shipping option. "
                   "Moving the threshold between two clusters changes nothing.")
    with t2:
        st.caption("This is an illustration. Replace the assumptions with your own figures; the model only supplies precision and recall.")
        a, b, c2, d = st.columns(4)
        vol = a.number_input("Orders per month", 1000, 10_000_000, 20_000, 1000)
        late_cost = b.number_input("Cost of one late order", 0.0, 1000.0, 12.0, 1.0)
        act_cost = c2.number_input("Cost of acting on one alert", 0.0, 1000.0, 1.5, 0.5)
        fix = d.slider("Share of alerts that prevent the damage", 0.0, 1.0, 0.30, 0.05)
        flagged_n = m["flagged_share"] * vol
        tp_n = m["tp"] / m["orders"] * vol
        gross, spent = tp_n * fix * late_cost, flagged_n * act_cost
        k = st.columns(3)
        with k[0]: kpi("Net value per month", f"{gross - spent:,.0f}", "cost avoided minus cost of acting")
        with k[1]: kpi("Alerts per month", f"{flagged_n:,.0f}", f"{tp_n:,.0f} of them truly late")
        be = act_cost / (m["precision"] * late_cost) if m["precision"] and late_cost else float("nan")
        with k[2]: kpi("Break-even prevention rate", pct(be), "alerts must fix at least this share")
        chart(ch.cost_waterfall(gross, spent))
    with t3:
        w = f[f["p_late"] >= thr].copy()
        w["priority"] = w["p_late"] * w["Sales"]
        w = w.sort_values("priority", ascending=False).head(300)
        view = w[["order date (DateOrders)", "Shipping Mode", "Order Region", "Market", "Sales", "p_late", "priority"]]
        view = view.rename(columns={"order date (DateOrders)": "Order date", "p_late": "Chance late", "priority": "Priority"})
        st.caption("Sorted by priority: chance of being late times order value, so the costliest problems come first.")
        st.dataframe(view, hide_index=True, column_config={
            "Chance late": st.column_config.ProgressColumn("Chance late", min_value=0.0, max_value=1.0, format="%.2f"),
            "Sales": st.column_config.NumberColumn("Sales", format="%.0f"),
            "Priority": st.column_config.NumberColumn("Priority", format="%.0f")})
        st.download_button("Download the watchlist (CSV)", view.to_csv(index=False).encode(), "watchlist.csv", "text/csv")
    with t4:
        low = st.slider("Low risk below", 0.01, 0.5, LOW, 0.01)
        tier = np.where(f["p_late"] >= thr, "High", np.where(f["p_late"] < low, "Low", "Medium"))
        s = f.assign(tier=tier).groupby("tier")[Y].agg(orders="size", late_rate="mean").reindex(["High", "Medium", "Low"]).dropna()
        s["share"] = s["orders"] / s["orders"].sum()
        chart(ch.tier_bars(s))
        st.dataframe(s.round(3))


# ============================================================ where and when
def page_where_when():
    page_header("Does it matter where or when you order?",
                "Late rates by region and market, corrected for which shipping options each place uses. Without that "
                "correction, a region that favours Second Class looks unreliable only because of its shorter promise.")
    t1, t2, t3, t4 = st.tabs(["Region and shipping option", "Markets", "Calendar", "Departments"])
    how = ("Each bar is the gap between a place's real late rate and the rate its shipping mix would predict. "
           "A bar near zero means the place is no better or worse than expected.")
    with t1:
        rm = art["region_mode"]
        vals = rm.pivot(index="Order Region", columns="Shipping Mode", values="late_rate")
        cnt = rm.pivot(index="Order Region", columns="Shipping Mode", values="orders")
        vals = vals.where(cnt >= 30).reindex(art["region_table"]["Order Region"])
        chart(ch.heatmap(vals, "Late rate (%) by region and shipping option"))
        st.caption("Cells with fewer than 30 orders are left blank.")
        chart(ch.gap_bars(art["region_table"], "Order Region"))
        aside(how)
    with t2:
        chart(ch.gap_bars(art["market_table"], "Market", min_orders=100))
        mt = art["market_table"].copy()
        for c_ in ("late_rate", "expected", "gap"):
            mt[c_] = (mt[c_] * 100).round(1)
        st.dataframe(mt.round(2), hide_index=True)
        aside(how)
    with t3:
        modes = list(SC.index)
        pick = st.multiselect("Shipping modes", modes, default=modes)
        if pick:
            v = st.radio("Show", ["late_rate", "mean_real_days"], horizontal=True,
                         format_func=lambda x: "Late rate" if x == "late_rate" else "Average delivery days")
            chart(ch.monthly_lines(art["monthly"], pick, v))
        hm = st.selectbox("Late rate by hour and weekday for", modes, index=modes.index("Same Day") if "Same Day" in modes else 0)
        chart(ch.hour_weekday_heatmap(art["hour_weekday"][art["hour_weekday"]["Shipping Mode"] == hm], f"{hm}: when the order was placed"))
        aside("Same Day turns darker from 12:00 because of the noon cut-off. Look for the same pattern, or its absence, in the other options.")
    with t4:
        chart(ch.gap_bars(art["dept_table"], "Department Name", min_orders=100))
        aside(how)


# ================================================================= how it works
def page_model_lab():
    page_header("How far can the model be trusted?",
                "Results on orders the model never saw, how well its probabilities match reality, the five-number table it "
                "is measured against, and what the data cannot tell us.")
    roc = art["roc"]
    c = st.columns(3)
    for col, name in zip(c, ["Calibrated LightGBM", "5-row lookup table", "Shipping mode only"]):
        with col:
            kpi(name, f"{roc[name]['auc']:.3f}", "ROC-AUC on the test period")
    a, b = st.columns(2)
    with a: chart(ch.roc_fig(roc))
    with b: chart(ch.calibration_fig(art["calibration"]))
    chart(ch.split_timeline(art["split_info"]))

    t1, t2, t3 = st.tabs(["The five-number table", "Avoiding leakage", "Limits"])
    with t1:
        rows = []
        for opt in DIST.index:
            rows.append({"Option": opt, "Promised (days)": int(OPT_PROMISE[opt]),
                         "Average delivery (days)": round(float((DIST.loc[opt] * DIST.columns).sum()), 2),
                         "Chance of late": pct(art["lookup"].get(opt, float("nan")), 1),
                         "Orders": int(art["option_orders"].loc[opt, "orders"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        aside("A model that cannot beat five numbers has not found hidden structure. It has learned the promise and the noon "
              "cut-off. Saying so is part of the result.")
    with t2:
        st.markdown("""
Four columns are only known after an order ships, so the model never sees them: `Days for shipping (real)`,
`Delivery Status`, `Order Status` and `shipping date`. A unit test fails if one of them reaches the model.

The oldest 70% of orders trains the model, the next 15% tunes it, and the newest 15% tests it. Results from notebook runs:

| Check | ROC-AUC |
|---|---|
| Time-based split | 0.779 |
| Random split | 0.801 (too high: 89% of rows belong to multi-item orders) |
| Time-based, without order-time features | 0.743 |
""")
    with t3:
        st.markdown("""
- DataCo is a synthetic dataset. Lateness comes from the promise and the Same Day cut-off, plus random noise.
- It has no supplier IDs and no shipping-cost column, so ranking suppliers or optimising cost is out of scope.
- Two other logistics datasets showed no real predictive signal and are used only as leakage examples.
- Cancelled and suspected-fraud orders are left out because the data labels them "on time".
- The money estimate on the Early warnings page uses assumptions you enter. It is a way to reason, not a forecast.
""")


# =================================================================== router
PAGES = {"Overview": page_overview, "Shipping options": page_shipping, "Check an order": page_checker,
         "Early warnings": page_warning, "Where and when": page_where_when, "How it works": page_model_lab}
with box("topnav"):
    left, right = st.columns([1.25, 3.4])
    left.markdown('<div class="brand"><i></i>Delivery Reliability</div>', unsafe_allow_html=True)
    choice = right.radio("Navigate", NAV, horizontal=True, key="nav", label_visibility="collapsed")
PAGES[choice]()

kp = art["kpis"]; ts = art["split_info"]["Test"]
st.markdown(f'<div class="foot">Data: DataCo Smart Supply Chain (Kaggle), orders from {kp["date_min"]:%B %Y} to '
            f'{kp["date_max"]:%B %Y}. Models learn from earlier orders and are tested on {pd.Timestamp(ts["start"]):%B %Y} to '
            f'{pd.Timestamp(ts["end"]):%B %Y}. Course project.</div>', unsafe_allow_html=True)
