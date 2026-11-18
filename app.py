"""Delivery Reliability: the Streamlit app. Run from the project root:   streamlit run app.py

Needs notebooks 2 and 3 to have been run once (they create models/models.joblib and models/app_data.joblib).
Six pages: Overview, Shipping options, Check an order, Early warnings, Where and when, Evidence.
The Evidence page also needs notebook 4 (it creates models/evidence.joblib); without it the page shows a short note.
All logic is in core.py; this file only draws it.
"""
import itertools
import math
from datetime import datetime, time, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score

import core
from core import DATE, MODE, PROMISE, TARGET, TARGET_DAYS

st.set_page_config(page_title="Delivery Reliability", layout="wide")

INK, SIGNAL, LINE, MUTED = "#1B2A3A", "#FFC633", "#D5DCE3", "#5B6773"
LATE, OK = "#C8372D", "#2E7D5B"
MODE_COLORS = {"Same Day": "#6D5A9C", "First Class": "#1F7A8C", "Second Class": "#D08A1E", "Standard Class": "#4F6D8F"}
TIER_COLORS = {"High": LATE, "Medium": "#E0A52A", "Low": OK}

# ------------------------------------------------------------------------------------------ style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root {--ink:#1B2A3A; --page:#F3F5F7; --signal:#FFC633; --signal-soft:#FFF1BF; --muted:#5B6773; --line:#D5DCE3;}
.stApp {background: var(--page);}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp th, .stApp td {font-family: 'IBM Plex Sans', system-ui, sans-serif;}
.stApp h1, .stApp h2, .stApp h3 {font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; color: var(--ink);}
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu {display: none !important;}
.block-container {max-width: 1100px; padding: 1.1rem 1.5rem 3rem;}
.st-key-topnav {background: var(--ink); border-radius: 6px; padding: .5rem 1.1rem; margin-bottom: 2rem;}
.brand {font-family: 'Barlow Condensed', sans-serif; font-size: 1.65rem; font-weight: 700; color: #fff; line-height: 2.2rem; white-space: nowrap;}
.brand i {display: inline-block; width: .72rem; height: .72rem; background: var(--signal); margin-right: .55rem;}
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"] {gap: .2rem; justify-content: flex-end; flex-wrap: wrap;}
.st-key-topnav [data-testid="stRadio"] label {padding: .38rem .85rem; border-radius: 4px; cursor: pointer; margin: 0;}
.st-key-topnav [data-testid="stRadio"] label > div:first-child {display: none;}
.st-key-topnav [data-testid="stRadio"] label p {color: #D3DBE4; font-size: .95rem; font-weight: 500; margin: 0; white-space: nowrap;}
.st-key-topnav [data-testid="stRadio"] label:has(input:checked) {background: var(--signal);}
.st-key-topnav [data-testid="stRadio"] label:has(input:checked) p {color: var(--ink); font-weight: 600;}
.stApp .pagehead h1 {font-size: 3.2rem !important; line-height: 1.0 !important; font-weight: 700; margin: 0 0 .7rem !important; max-width: 22ch; padding: 0 !important;}
.pagehead .lede {font-size: 1.12rem; line-height: 1.6; color: #34424F; max-width: 62ch; margin: 0 0 1.4rem;}
.stApp h2.sec {font-size: 1.9rem !important; font-weight: 600; margin: 2.2rem 0 .5rem !important; padding: 0 !important;}
.secsub {color: var(--muted); max-width: 66ch; margin: 0 0 .9rem; line-height: 1.5;}
.kpi .v {font-family: 'Barlow Condensed', sans-serif; font-size: 2.6rem; font-weight: 600; line-height: 1; color: var(--ink);}
.kpi .l {font-weight: 500; margin-top: .3rem;} .kpi .s {font-size: .85rem; color: var(--muted); line-height: 1.35;}
.aside {background: var(--signal-soft); padding: .7rem 1rem; border-radius: 3px; margin: .7rem 0; color: #2A2F36; line-height: 1.55;}
.badge {display: inline-block; padding: .3rem 1rem; border-radius: 3px; color: #fff; font-weight: 600;}
.foot {margin-top: 3.5rem; border-top: 1px solid var(--line); padding-top: .9rem; color: var(--muted); font-size: .85rem; max-width: 80ch;}
.stApp .stButton > button, .stApp [data-testid="stFormSubmitButton"] > button {background: var(--ink); color: #fff; border: 1px solid var(--ink); border-radius: 4px; font-weight: 600; padding: .45rem 1.15rem;}
.stApp .stButton > button:hover, .stApp [data-testid="stFormSubmitButton"] > button:hover {background: var(--signal); border-color: var(--signal); color: var(--ink);}
[data-baseweb="tab-highlight"] {background: var(--signal) !important; height: 3px !important;}
@media (max-width: 760px) {.stApp .pagehead h1 {font-size: 2.3rem !important;} .brand {font-size: 1.35rem;}}
</style>
""", unsafe_allow_html=True)

_keys = itertools.count()


def chart(fig):
    st.plotly_chart(fig, key=f"chart{next(_keys)}", config={"displayModeBar": False})


KEYED = True


def box(key):
    """Container with a CSS hook. Streamlit older than 1.40 has no `key`; the page then loses its dark top bar."""
    global KEYED
    try:
        return st.container(key=key)
    except TypeError:
        KEYED = False
        return st.container()


def header(title, lede):
    st.markdown(f'<div class="pagehead"><h1>{title}</h1><p class="lede">{lede}</p></div>', unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<h2 class="sec">{title}</h2>' + (f'<p class="secsub">{sub}</p>' if sub else ""), unsafe_allow_html=True)


def kpi(label, value, sub=""):
    st.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div><div class="s">{sub}</div></div>',
                unsafe_allow_html=True)


def aside(text):
    st.markdown(f'<div class="aside">{text}</div>', unsafe_allow_html=True)


def pct(x, d=0):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}%}"


def style(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="IBM Plex Sans", color=INK, size=13),
                      legend=dict(orientation="h", y=-0.18))
    fig.update_xaxes(gridcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig


# ------------------------------------------------------------------------------------------ data
@st.cache_resource
def load():
    return joblib.load(core.APP_FILE), joblib.load(core.MODELS_FILE)


try:
    app, models = load()
except FileNotFoundError:
    st.error("Model files not found. Run notebooks 2 and 3 first (they create the two files in the models folder).")
    st.stop()

try:
    evidence = joblib.load(core.EVIDENCE_FILE)
except Exception:                      # notebook 4 not run yet: the Evidence page then explains what to do
    evidence = None

orders = app["orders"]
test = orders[orders["split"] == "Test"].reset_index(drop=True)


@st.cache_data
def tables():
    """Everything below is computed from the orders table that notebook 3 saved."""
    t = {"scorecard": core.scorecard(orders), "dist": core.real_days_distribution(orders),
         "promise": orders.groupby("option")[PROMISE].agg(lambda s: int(s.mode().iloc[0])),
         "auc": {"5-row lookup table": roc_auc_score(test[TARGET], test["p_lookup"]),
                 "Calibrated LightGBM": roc_auc_score(test[TARGET], test["p_late"]),
                 "Shipping mode only": roc_auc_score(test[TARGET], test["p_mode"])}}
    if "Order Profit Per Order" in orders.columns:
        g = orders.groupby(MODE)["Order Profit Per Order"]
        t["profit"] = pd.DataFrame({"mean": g.mean(), "ci": 1.96 * g.std() / np.sqrt(g.size())})
    t["regions"] = {m: sorted(s.unique()) for m, s in orders.groupby("Market")["Order Region"]}
    t["top_region"] = orders.groupby("Market")["Order Region"].agg(lambda s: s.value_counts().index[0]).to_dict()
    t["region_n"] = orders["Order Region"].value_counts().to_dict()
    t["top_market"] = orders["Market"].value_counts().index[0]
    t["top_dept"] = orders["Department Name"].value_counts().index[0]
    t["top_type"] = orders["Type"].value_counts().index[0]
    steps = np.round(np.linspace(0.05, 0.95, 91), 2)
    t["sweep"] = pd.DataFrame([{"threshold": s, **core.threshold_table(test, s)} for s in steps])
    return t


T = tables()
sc, dist = T["scorecard"], T["dist"]
late_total = float(test[TARGET].sum())

NAV = ["Overview", "Shipping options", "Check an order", "Early warnings", "Where and when", "Evidence"]
with box("topnav"):
    left, right = st.columns([1.0, 4.2])
    left.markdown('<div class="brand"><i></i>Delivery Reliability</div>', unsafe_allow_html=True)
    page = right.radio("Page", NAV, horizontal=True, label_visibility="collapsed", key="nav")
if not KEYED:
    st.markdown("<style>.brand {color: #1B2A3A !important;}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------ pages
def page_overview():
    header("Can I trust this delivery?", "Every order comes with a promised delivery time. This app shows how often each shipping option "
           "keeps that promise, why it fails, and which orders deserve a warning before they ship.")
    c = st.columns(4)
    with c[0]:
        kpi("Shipped orders", f"{len(orders):,}", f"{orders[DATE].min():%b %Y} to {orders[DATE].max():%b %Y}")
    with c[1]:
        kpi("Delivered late", pct(orders[TARGET].mean()), "slower than the promise")
    with c[2]:
        raw = None if evidence is None else float(evidence["bootstrap"].iloc[0]["auc"])
        kpi("Model ROC-AUC", f"{T['auc']['Calibrated LightGBM']:.3f}",
            "newest 15% of orders" + ("" if raw is None else f"; {raw:.3f} before calibration"))
    with c[3]:
        kpi("5-row table ROC-AUC", f"{T['auc']['5-row lookup table']:.3f}", "the simple rule it has to beat")

    section("Promise against reality", "Late means slower than the promise. Compare what each option promises with how long it really takes.")
    fig = go.Figure()
    fig.add_bar(x=sc.index, y=sc["promised_days"], name="Promised (days)", marker_color=INK)
    fig.add_bar(x=sc.index, y=sc["mean_real_days"], name="Real, on average (days)", marker_color=SIGNAL,
                text=[f"{r:.0%} late" for r in sc["late_rate"]], textposition="outside")
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, float(sc[["promised_days", "mean_real_days"]].to_numpy().max()) * 1.3])
    chart(style(fig, 380))

    s, q = sc.loc["Second Class"], sc.loc["Standard Class"]
    section("What the data says")
    aside(f"<b>Lateness is set by the promise.</b> Second Class takes {s['mean_real_days']:.1f} days on average, the same as Standard Class "
          f"({q['mean_real_days']:.1f}), but promises {s['promised_days']:.0f} days instead of {q['promised_days']:.0f}. "
          f"It is late {pct(s['late_rate'])} of the time against {pct(q['late_rate'])}.")
    aside(f"<b>First Class is the only truly faster option</b>, but it always takes 2 days against a 1-day promise "
          f"({pct(sc.loc['First Class', 'late_rate'])} late). <b>Same Day</b> has a noon cut-off: before noon it arrives the same day, from noon the next day.")
    aside(f"<b>A 5-row table matches the machine-learning model</b> (ROC-AUC {T['auc']['5-row lookup table']:.3f} against "
          f"{T['auc']['Calibrated LightGBM']:.3f}). The model has learned the promise and the cut-off, and nothing more.")
    section("What this app cannot tell you", "The data is synthetic: lateness comes from the promise plus random noise, so about 0.78 ROC-AUC is the ceiling. "
            "There are no supplier IDs and no shipping costs, so it cannot rank suppliers or optimise cost. Scores on the other pages come from "
            "the newest 15% of orders, which the model never saw in training.")


def page_options():
    header("Shipping options", "How long each option really takes, how often it breaks its promise, and what promise it should make.")
    view = sc.reset_index()[[MODE, "orders", "promised_days", "mean_real_days", "std_real_days", "late_rate", "late_low", "late_high"]]
    view.columns = ["Shipping mode", "Orders", "Promised days", "Real days (mean)", "Real days (std)", "Late rate", "Late, low", "Late, high"]
    st.dataframe(view.style.format({"Orders": "{:,.0f}", "Promised days": "{:.0f}", "Real days (mean)": "{:.2f}", "Real days (std)": "{:.2f}",
                                    "Late rate": "{:.1%}", "Late, low": "{:.1%}", "Late, high": "{:.1%}"}), hide_index=True)

    section("When does each option really arrive?", "Green days are on time, red days are later than the promise.")
    option = st.selectbox("Option", list(dist.index), index=list(dist.index).index("Second Class"))
    row, promise = dist.loc[option], int(T["promise"][option])
    fig = go.Figure(go.Bar(x=[f"{d}" for d in row.index], y=row.values, marker_color=[OK if d <= promise else LATE for d in row.index],
                           text=[f"{v:.0%}" for v in row.values], textposition="outside"))
    fig.update_layout(xaxis_title="real delivery days (6 = six or more)", yaxis_title="share of orders", yaxis_tickformat=".0%")
    fig.update_yaxes(range=[0, float(row.max()) * 1.2 + 0.001])
    chart(style(fig, 340))
    aside(f"{option} promises {promise} day{'s' if promise != 1 else ''}; {pct(core.promise_late_rate(row, promise))} of its orders take longer.")

    section("Promise designer", "Pick a late rate you can live with; see the shortest promise that achieves it.")
    target = st.slider("Highest acceptable late rate", 0, 60, 10, step=5, format="%d%%") / 100
    rows = [{"Option": o, "Promise now (days)": int(T["promise"][o]), "Late rate now": core.promise_late_rate(dist.loc[o], int(T["promise"][o])),
             "Promise needed (days)": core.promise_for_target(dist.loc[o], target)} for o in dist.index]
    st.dataframe(pd.DataFrame(rows).style.format({"Late rate now": "{:.0%}"}), hide_index=True)

    if "profit" in T:
        pp = T["profit"]
        section("Is lateness a margin problem?", "Profit per order by shipping mode, with 95% error bars.")
        fig = go.Figure(go.Bar(x=pp.index, y=pp["mean"], error_y=dict(type="data", array=pp["ci"].to_numpy(), color=INK),
                               marker_color=[MODE_COLORS.get(m, INK) for m in pp.index],
                               text=[f"{v:.1f}" for v in pp["mean"]], textposition="outside"))
        fig.update_layout(yaxis_title="profit per order")
        fig.update_yaxes(range=[0, float((pp["mean"] + pp["ci"]).max()) * 1.25])
        chart(style(fig, 320))
        if (pp["mean"] - pp["ci"]).max() <= (pp["mean"] + pp["ci"]).min():
            aside("All the error bars overlap, so there is no evidence that any option earns more or less per order. "
                  "Late delivery here is a service problem, not a margin problem.")
        else:
            aside("Some options differ by more than the noise, so profit per order is not the same across shipping modes.")


def page_order():
    header("Check an order", "Describe an order as it is placed. The model scores it with only what is known at that moment.")
    o, d = app["options"], app["num_defaults"]
    modes, depts, markets = list(app["scheduled_days"]), sorted(app["category_by_dept"]), o["Market"]
    a, b, c = st.columns(3)
    mode = a.selectbox("Shipping mode", modes, index=modes.index("Standard Class"))
    day = b.date_input("Order date", value=datetime(2017, 6, 20))
    hour = c.time_input("Order time", value=time(14, 30))
    a, b, c = st.columns(3)
    dept = a.selectbox("Department", depts, index=depts.index(T["top_dept"]))
    category = b.selectbox("Category", app["category_by_dept"][dept])
    segment = c.selectbox("Customer segment", o["Customer Segment"])
    a, b, c = st.columns(3)
    market = a.selectbox("Market", markets, index=markets.index(T["top_market"]))
    regions = T["regions"][market]
    region = b.selectbox("Region", regions, index=regions.index(T["top_region"][market]))
    type_ = c.selectbox("Payment type", o["Type"], index=o["Type"].index(T["top_type"]))
    a, b, c = st.columns(3)
    price = a.number_input("Product price", 1.0, 2000.0, float(round(d["Product Price"], 0)))
    qty = b.number_input("Quantity", 1, 20, int(d["Order Item Quantity"]))
    disc = c.slider("Discount", 0, 30, 10, format="%d%%") / 100
    sales = price * qty
    st.caption(f"Order value: {sales:,.0f} (price times quantity). The result updates as you change any field.")

    order = dict(order_dt=datetime.combine(day, hour), type_=type_, market=market, region=region, department=dept,
                 category=category, segment=segment, sales=sales, price=price, quantity=int(qty), discount_rate=disc)
    frame = core.make_order_frame(app, mode=mode, **order)
    r = core.score_order(app, models, frame)
    tier = core.tier_of(r["p_late"])

    section("Result")
    a, b, c = st.columns(3)
    with a:
        sub = (f"model alone said {pct(r['p_model'])}; the 5-row table ({r['key']}) says {pct(r['p_lookup'])} and is used"
               if r["disagree"] else f"the 5-row table says {pct(r['p_lookup'])} ({r['key']})")
        kpi("Chance of arriving late", pct(r["p_late"]), sub)
        st.markdown(f'<span class="badge" style="background:{TIER_COLORS[tier]}">{tier} risk</span>', unsafe_allow_html=True)
    with b:
        kpi("Expected delivery time", f"{r['days']:.1f} days", f"80% window: {r['earliest']:.1f} to {r['latest']:.1f} days")
    with c:
        arrival = datetime.combine(day, hour) + timedelta(days=r["days"])
        kpi("Expected arrival", f"{arrival:%d %b}", f"promised: {app['scheduled_days'][mode]} day(s)")
    n_region = T["region_n"].get(region, 0)
    if r["disagree"] or n_region < 1000:
        aside(f"<b>Treat the model score with care.</b> {region} has only {n_region:,} orders in the data, and region carries almost no real signal. "
              "When the model and the 5-row table disagree by more than 25 points, the app uses the table, which was just as accurate when tested.")
    st.markdown("**Recommended action**")
    for act in core.recommend(r["p_late"], sales, r["window"]):
        aside(act)

    section("Why this score?", "Each bar adds up everything the model read from one field. Right of zero pushes towards late, left towards on time.")
    contrib = core.explain_order(models["late"], frame).iloc[::-1]
    fig = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h", marker_color=[LATE if v > 0 else OK for v in contrib.values]))
    fig.update_layout(xaxis_title="push on the late score (log-odds)")
    chart(style(fig, 320))

    section("Same order, every shipping option")
    cmp_ = core.score_all_modes(app, models, **order)
    fig = go.Figure(go.Bar(x=cmp_["Shipping mode"], y=cmp_["Chance late"], marker_color=[MODE_COLORS.get(m, INK) for m in cmp_["Shipping mode"]],
                           text=[pct(v) for v in cmp_["Chance late"]], textposition="outside"))
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.15], yaxis_title="chance of arriving late")
    chart(style(fig, 320))
    st.dataframe(cmp_.style.format({"Chance late": "{:.0%}", "Model alone": "{:.0%}", "Lookup table": "{:.0%}",
                                    "Expected days": "{:.1f}", "Earliest": "{:.1f}", "Latest": "{:.1f}"}), hide_index=True)


def page_warnings():
    header("Early warnings", "Turn a risk score into a flag, and see what that flag catches and what it costs. Everything here is measured on orders the model never saw.")
    section("Choose the alert level", "Flag every order whose calibrated chance of being late is at least this.")
    thr = st.slider("Alert threshold", 5, 95, int(core.TIER_HIGH * 100), format="%d%%") / 100
    m, ml = core.threshold_table(test, thr), core.threshold_table(test, thr, "p_lookup")
    def column(r):
        return [pct(r["flagged_share"]), pct(r["precision"]), pct(r["recall"]), "-" if math.isnan(r["false_per_catch"]) else f"{r['false_per_catch']:.2f}"]
    cmp_ = pd.DataFrame({"Calibrated LightGBM": column(m), "5-row lookup table": column(ml)},
                        index=["Orders flagged", "Precision (flagged that are late)", "Recall (late orders caught)", "False alarms per late order caught"])
    c = st.columns(4)
    with c[0]:
        kpi("Orders flagged", pct(m["flagged_share"]), f"{m['flagged']:,} of {len(test):,}")
    with c[1]:
        kpi("Precision", pct(m["precision"]), "of flagged orders are late")
    with c[2]:
        kpi("Recall", pct(m["recall"]), "of late orders are caught")
    with c[3]:
        kpi("False alarms per catch", "-" if math.isnan(m["false_per_catch"]) else f"{m['false_per_catch']:.2f}", "on-time orders flagged")
    st.dataframe(cmp_)
    aside("If the two columns are close, the model is doing what a 5-row table does.")

    sw = T["sweep"]
    fig = go.Figure()
    fig.add_scatter(x=sw["threshold"], y=sw["precision"], name="Precision", line=dict(color=INK, width=3))
    fig.add_scatter(x=sw["threshold"], y=sw["recall"], name="Recall", line=dict(color=SIGNAL, width=3))
    fig.add_scatter(x=sw["threshold"], y=sw["flagged_share"], name="Orders flagged", line=dict(color=MUTED, dash="dot"))
    fig.add_vline(x=thr, line_color=LATE)
    fig.update_layout(xaxis_title="alert threshold", yaxis_tickformat=".0%")
    chart(style(fig, 340))
    st.caption("The curves move in steps because calibrated scores form a few clusters, one per shipping option.")

    section("What does it cost?", "Set your own costs. These are assumptions, not data.")
    a, b = st.columns(2)
    c_fp = a.number_input("Cost of a false alarm", 0.0, 1000.0, 1.0)
    c_fn = b.number_input("Cost of a missed late order", 0.0, 1000.0, 5.0)
    cost = (sw["false_alarms"] * c_fp + late_total * (1 - sw["recall"]) * c_fn) / len(test) * 1000
    fig = go.Figure(go.Scatter(x=sw["threshold"], y=cost, line=dict(color=INK, width=3)))
    t_star = c_fp / (c_fp + c_fn) if c_fp + c_fn > 0 else None
    if t_star is not None:
        fig.add_vline(x=min(max(t_star, 0.05), 0.95), line_color=OK)
    fig.update_layout(xaxis_title="alert threshold", yaxis_title="cost per 1,000 orders")
    chart(style(fig, 300))
    if t_star is not None:
        share = core.threshold_table(test, t_star)["flagged_share"]
        aside(f"<b>Break-even rule:</b> flag an order when its chance of being late is above {t_star:.0%} "
              f"(cost of a false alarm divided by the cost of a false alarm plus a missed late order). The green line marks it. "
              f"With your costs that flags {share:.0%} of orders. When a missed late order costs much more than a false alarm, "
              "the cheapest policy flags almost everything, and an alert that fires on almost every order is no longer an alert.")

    section("Risk tiers", f"High: at least {core.TIER_HIGH:.0%}. Low: below {core.TIER_LOW:.0%}. Medium in between.")
    tiers = test.assign(tier=core.tier_of(test["p_late"]))
    g = tiers.groupby("tier")[TARGET].agg(orders="size", late_rate="mean").reindex(["High", "Medium", "Low"]).fillna(0)
    g["share_of_orders"] = g["orders"] / len(test)
    st.dataframe(g.reset_index().style.format({"late_rate": "{:.0%}", "share_of_orders": "{:.0%}"}), hide_index=True)

    section("Is the probability honest?", "Dots on the diagonal mean a score of 0.8 really is late 80% of the time.")
    bins = pd.cut(test["p_late"], np.linspace(0, 1, 11), include_lowest=True)
    cal = test.groupby(bins, observed=True).agg(predicted=("p_late", "mean"), observed=(TARGET, "mean"), orders=(TARGET, "size"))
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=MUTED, dash="dash"), name="perfect")
    fig.add_scatter(x=cal["predicted"], y=cal["observed"], mode="markers+lines", marker=dict(size=cal["orders"] / cal["orders"].max() * 30 + 6, color=INK), name="model")
    fig.update_layout(xaxis_title="predicted chance of late", yaxis_title="observed late rate", xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    chart(style(fig, 340))

    section("Priority watchlist", "The highest-value High-tier orders in the held-out test period. In real use you would score each new order as it is placed.")
    watch = tiers[tiers["tier"] == "High"].sort_values("Sales", ascending=False).head(15)
    st.dataframe(watch[[DATE, MODE, "Order Region", "Department Name", "Sales", "p_late"]].rename(columns={DATE: "Order date", "p_late": "Chance late"})
                 .style.format({"Sales": "{:.0f}", "Chance late": "{:.0%}"}), hide_index=True)
    everything = tiers[tiers["tier"] == "High"].sort_values("Sales", ascending=False)[[DATE, MODE, "Order Region", "Department Name", "Sales", "p_late"]]
    st.download_button(f"Download all {len(everything):,} High-tier orders (CSV)", everything.to_csv(index=False).encode("utf-8"),
                       "high_risk_orders.csv", "text/csv")


def page_where():
    header("Where and when", "Do regions, markets or departments differ, and does the time of the order matter?")
    section("Regions, markets and departments", "A region that uses more Second Class looks unreliable only because of its short promise. "
            "So each bar compares the late rate with what its shipping-mode mix alone would give. Dark bars are unusual (|z| above 1.96); with many groups a few appear by chance.")
    group = st.radio("Group by", ["Order Region", "Market", "Department Name"], horizontal=True)
    g = core.mode_mix_gap(orders, group)
    g = g[g["orders"] >= 1000].sort_values("gap")
    fig = go.Figure(go.Bar(x=g["gap"], y=g[group], orientation="h", marker_color=[INK if abs(z) > 1.96 else "#9AA7B4" for z in g["z"]]))
    fig.update_layout(xaxis_title="late rate minus rate expected from the shipping-mode mix", xaxis_tickformat=".1%")
    chart(style(fig, max(260, 26 * len(g) + 80)))
    st.caption(f"{int((g['z'].abs() > 1.96).sum())} of {len(g)} groups are unusual; about {0.05 * len(g):.1f} would be by chance alone.")

    section("Time of order", "Late rate by weekday and hour for one shipping option. Same Day shows the noon cut-off; the others are flat.")
    pick = st.selectbox("Shipping mode", list(app["scheduled_days"]), index=list(app["scheduled_days"]).index("Same Day"))
    sub = orders[orders[MODE] == pick]
    hw = sub.pivot_table(index="order_weekday", columns="order_hour", values=TARGET, aggfunc="mean")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure(go.Heatmap(z=hw.values, x=hw.columns, y=[days[i] for i in hw.index], zmin=0, zmax=1,
                               colorscale=[[0, "#FFF8E1"], [0.5, SIGNAL], [1, LATE]], colorbar=dict(title="late rate", tickformat=".0%")))
    fig.update_layout(xaxis_title="order hour", yaxis_autorange="reversed")
    chart(style(fig, 320))

    section("Over time", "Monthly late rate by shipping mode.")
    month = orders[DATE].dt.to_period("M").dt.to_timestamp()
    mt = orders.assign(month=month).groupby(["month", MODE])[TARGET].agg(rate="mean", n="size").reset_index()
    mt = mt[mt["n"] >= 30]
    fig = go.Figure()
    for m in sorted(mt[MODE].unique()):
        s = mt[mt[MODE] == m]
        fig.add_scatter(x=s["month"], y=s["rate"], name=m, line=dict(color=MODE_COLORS.get(m, INK), width=3))
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.05], yaxis_title="late rate")
    chart(style(fig, 340))


def page_evidence():
    header("Evidence", "How sure are we that a 5-row table is as good as the model? The checks from notebook 4, measured on the newest 15% of orders.")
    if evidence is None:
        aside("This page needs <b>notebook 4</b>. Run it to the end (its last cell saves <code>models/evidence.joblib</code>), then reload the app.")
        return
    boot, roll, dc = evidence["bootstrap"], evidence["rolling"], evidence["decision"]
    m, tb, diff = boot.iloc[0], boot.iloc[1], boot.iloc[2]
    ahead = int((roll["difference"] > 0).sum())
    c = st.columns(4)
    with c[0]:
        kpi("AUC gap, model minus table", f"{diff['auc']:+.4f}", f"95% interval {diff['lo']:+.4f} to {diff['hi']:+.4f}")
    with c[1]:
        kpi("Folds where the model wins", f"{ahead} of {len(roll)}", "rolling time windows")
    with c[2]:
        kpi("Best net-benefit gap", f"{(dc['LightGBM (calibrated)'] - dc['5-row table']).abs().max():.4f}", "model against table, any threshold")
    inside = evidence["within_mode"]
    std = inside[inside["shipping mode"] == "Standard Class"]
    with c[3]:
        kpi("Model AUC inside Standard Class", "-" if std.empty else f"{std.iloc[0, 3]:.3f}", "0.5 means no better than chance")

    section("1. Is the gap noise?", "The test orders were resampled 500 times, a whole order at a time, and both AUCs recomputed on each resample.")
    fig = go.Figure(go.Scatter(x=[m["auc"], tb["auc"]], y=["LightGBM", "5-row table"], mode="markers",
                               marker=dict(size=12, color=INK),
                               error_x=dict(type="data", symmetric=False, array=[m["hi"] - m["auc"], tb["hi"] - tb["auc"]],
                                            arrayminus=[m["auc"] - m["lo"], tb["auc"] - tb["lo"]], color=INK)))
    fig.update_layout(xaxis_title="ROC-AUC with 95% interval", showlegend=False)
    chart(style(fig, 220))
    verdict = ("The interval of the difference includes zero, so the model is not distinguishably better than the table."
               if diff["lo"] <= 0 <= diff["hi"] else "The interval excludes zero, but check whether the size of the gap could change any decision.")
    aside(f"<b>Difference {diff['auc']:+.4f}</b> (interval {diff['lo']:+.4f} to {diff['hi']:+.4f}). {verdict}")

    section("2. Does it depend on where the timeline is cut?", "The model and table were refitted on five expanding windows; each point is a later test block.")
    fig = go.Figure()
    fig.add_scatter(x=roll["fold"], y=roll["LightGBM"], name="LightGBM", mode="lines+markers", line=dict(color=INK, width=3))
    fig.add_scatter(x=roll["fold"], y=roll["5-row table"], name="5-row table", mode="lines+markers", line=dict(color=SIGNAL, width=3, dash="dash"))
    fig.update_layout(xaxis_title="fold (later = more recent)", yaxis_title="ROC-AUC", xaxis=dict(dtick=1))
    chart(style(fig, 300))
    aside(f"Mean difference <b>{roll['difference'].mean():+.4f}</b>, standard deviation {roll['difference'].std():.4f}. "
          f"The model is ahead in {ahead} of {len(roll)} folds, so neither one is consistently better.")

    section("3. Does acting on the model pay more?", "Net benefit counts each caught late order as 1 and each false alarm as t/(1-t). Higher is better.")
    fig = go.Figure()
    fig.add_scatter(x=dc["threshold"], y=dc["LightGBM (calibrated)"], name="LightGBM (calibrated)", line=dict(color=INK, width=3))
    fig.add_scatter(x=dc["threshold"], y=dc["5-row table"], name="5-row table", line=dict(color=SIGNAL, width=3, dash="dash"))
    fig.add_scatter(x=dc["threshold"], y=dc["act on all"], name="flag every order", line=dict(color=MUTED, dash="dot"))
    fig.add_hline(y=0, line_color=MUTED)
    top = float(dc[["LightGBM (calibrated)", "5-row table"]].max().max())
    fig.update_layout(xaxis_title="alert threshold", yaxis_title="net benefit per order", yaxis_range=[-0.1, top * 1.15], xaxis_tickformat=".0%")
    chart(style(fig, 340))
    aside("The two lines lie on top of each other: using the model instead of the table changes no decision. "
          "Flagging every order is worse than doing nothing once the threshold passes the overall late rate.")

    section("4. Is it equally good for every group?")
    st.markdown("**Inside one shipping option.** The table gives every order in an option the same score (AUC 0.5), except Same Day, which it splits at noon. "
                "If the model found anything beyond the promise, its AUC would rise above 0.5.")
    st.dataframe(inside.style.format({"late rate": "{:.0%}", "LightGBM AUC inside the option": "{:.3f}", "table AUC inside the option": "{:.3f}"}, na_rep="-"),
                 hide_index=True)
    st.markdown("**By region.** Groups with at least 300 test orders. Calibration error is the average gap between predicted and observed late rate.")
    reg = evidence["region"]
    fmt = {c: "{:.3f}" for c in reg.columns if c not in ("orders", "Order Region", "note", "late rate")}
    if "late rate" in reg.columns:
        fmt["late rate"] = "{:.0%}"
    st.dataframe(reg.style.format(fmt, na_rep="-"), hide_index=True)
    aside("Small groups are where the model is weakest: it overfits them, which is why the Check an order page uses the table whenever the two disagree strongly.")

    section("5. Would it stay valid in use?", "Population stability index (PSI): below 0.1 stable, 0.1 to 0.25 moderate shift, above 0.25 large shift.")
    dr = evidence["drift"]
    st.dataframe(dr.style.format({dr.columns[1]: "{:.3f}"}), hide_index=True)
    qd = evidence["quarterly"]
    fig = go.Figure(go.Scatter(x=qd["quarter"], y=qd["late_rate"], mode="lines+markers", line=dict(color=INK, width=3)))
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], yaxis_title="late rate", showlegend=False)
    chart(style(fig, 280))
    aside("Some inputs (market and department mix) shifted a lot, yet the model score is stable and the late rate barely moves: lateness depends on the shipping option, "
          "and that mix is stable. So watch the score PSI, the shipping-option mix and the observed late rate, not every input.")

    section("Limits")
    aside("The data is synthetic, from one company and period. The newest orders come almost entirely from two markets, so the test numbers say little about the others. "
          "Costs on the Early warnings page are assumptions. A different promise (for example Second Class at 3 days) changes every late rate, so retrain after any promise change.")


{"Overview": page_overview, "Shipping options": page_options, "Check an order": page_order,
 "Early warnings": page_warnings, "Where and when": page_where, "Evidence": page_evidence}[page]()

st.markdown('<div class="foot">Built on the DataCo supply-chain dataset (synthetic, 2015 to 2018). Late means slower than the promised shipping time. '
            'Model scores are measured on the newest 15% of orders, which the model never saw during training.</div>', unsafe_allow_html=True)
