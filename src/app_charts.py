"""Plotly figure builders for the Streamlit app. Each function returns a go.Figure."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

MODE_COLORS = {"Same Day": "#8b5cf6", "First Class": "#0ea5e9", "Second Class": "#f59e0b", "Standard Class": "#10b981"}
TIER_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
GOOD, BAD, NEUTRAL = "#22c55e", "#ef4444", "#64748b"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def mode_color(name: str) -> str:
    return MODE_COLORS.get(str(name).split(" (")[0], NEUTRAL)


def _style(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), showlegend=legend,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=13), legend=dict(orientation="h", y=-0.2),
                      hoverlabel=dict(font_size=13))
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.18)", zerolinecolor="rgba(128,128,128,0.3)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zerolinecolor="rgba(128,128,128,0.3)")
    return fig


# ------------------------------------------------------------ shipping modes
def late_rate_bars(sc: pd.DataFrame) -> go.Figure:
    sc = sc.sort_values("late_rate")
    fig = go.Figure(go.Bar(
        x=sc["late_rate"] * 100, y=list(sc.index), orientation="h",
        marker_color=[mode_color(m) for m in sc.index],
        text=[f"{v:.0%}" for v in sc["late_rate"]], textposition="outside",
        customdata=np.c_[sc["promised_days"], sc["mean_real_days"], sc["orders"]],
        hovertemplate="<b>%{y}</b><br>Late: %{x:.1f}%<br>Promised: %{customdata[0]:.1f} days"
                      "<br>Actually takes: %{customdata[1]:.2f} days<br>Orders: %{customdata[2]:,.0f}<extra></extra>"))
    fig.update_xaxes(title="Late deliveries (%)", range=[0, 115])
    fig.update_layout(title="How often is each option late?")
    return _style(fig, 300, legend=False)


def promise_vs_reality(dist_row: pd.Series, promised: int, name: str) -> go.Figure:
    days = list(dist_row.index)
    on_time = [d <= promised for d in days]
    fig = go.Figure(go.Bar(
        x=days, y=dist_row.values * 100, marker_color=[GOOD if o else BAD for o in on_time],
        text=[f"{v:.0%}" if v > 0.005 else "" for v in dist_row.values], textposition="outside",
        hovertemplate="Arrives on day %{x}<br>%{y:.1f}% of orders<extra></extra>"))
    fig.add_vline(x=promised + 0.5, line_dash="dash", line_color=NEUTRAL,
                  annotation_text=f"promise: {promised} d", annotation_position="top")
    late = float(dist_row.loc[dist_row.index > promised].sum())
    fig.update_layout(title=f"{name}: {late:.0%} late if we promise {promised} day(s)")
    fig.update_xaxes(title="Days from order to delivery", dtick=1)
    fig.update_yaxes(title="Share of orders (%)", range=[0, max(100 * dist_row.max() * 1.25, 10)])
    return _style(fig, 340, legend=False)


def within_days_bars(within: pd.DataFrame, d: int) -> go.Figure:
    col = f"within {d} days"
    t = (within[col] * 100).sort_values()
    fig = go.Figure(go.Bar(x=t.values, y=list(t.index), orientation="h",
                           marker_color=[mode_color(i) for i in t.index],
                           text=[f"{v:.0f}%" for v in t.values], textposition="outside"))
    fig.update_xaxes(range=[0, 118], title=f"% of orders delivered within {d} day(s)")
    fig.update_layout(title=f"Chance of arriving within {d} day(s)")
    return _style(fig, 320, legend=False)


def profit_fig(profit: pd.DataFrame, y: str) -> go.Figure:
    fig = go.Figure()
    for val, label, color in ((0, "Delivered on time", GOOD), (1, "Delivered late", BAD)):
        t = profit[profit[y] == val]
        fig.add_bar(x=t["Shipping Mode"], y=t["profit_per_order"], name=label, marker_color=color,
                    customdata=t["orders"], hovertemplate="%{x}<br>%{y:.1f} per order<br>%{customdata:,} orders<extra></extra>")
    fig.update_layout(barmode="group", title="Average profit per order: on time vs late")
    fig.update_yaxes(title="Profit per order")
    return _style(fig, 340)


# --------------------------------------------------------------- one order
def gauge(p: float, low: float, high: float) -> go.Figure:
    tier = "High" if p >= high else ("Low" if p < low else "Medium")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 44}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": TIER_COLORS[tier], "thickness": 0.35},
               "steps": [{"range": [0, low * 100], "color": "rgba(34,197,94,0.18)"},
                         {"range": [low * 100, high * 100], "color": "rgba(245,158,11,0.18)"},
                         {"range": [high * 100, 100], "color": "rgba(239,68,68,0.18)"}]}))
    fig.update_layout(title="Chance this order is late")
    return _style(fig, 270, legend=False)


def delivery_timeline(lo: float, hi: float, point: float, promised: int) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=lo - 0.4, x1=hi + 0.4, y0=0.25, y1=0.75,
                  fillcolor="rgba(14,165,233,0.35)", line=dict(color="#0ea5e9", width=2))
    fig.add_trace(go.Scatter(x=[point], y=[0.5], mode="markers", name="Expected",
                             marker=dict(size=16, color="#0ea5e9", symbol="diamond"),
                             hovertemplate="Expected: %{x:.1f} days<extra></extra>"))
    fig.add_vline(x=promised + 0.5, line_dash="dash", line_color=BAD,
                  annotation_text=f"promised: {promised} d", annotation_position="top")
    fig.update_xaxes(range=[-0.5, 7.5], dtick=1, title="Days after the order")
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(title="Expected arrival window vs the promise (right of the red line = late)")
    return _style(fig, 230, legend=False)


def driver_bars(contrib: pd.Series) -> go.Figure:
    s = contrib.iloc[::-1]
    fig = go.Figure(go.Bar(x=s.values, y=list(s.index), orientation="h",
                           marker_color=[BAD if v > 0 else "#38bdf8" for v in s.values],
                           hovertemplate="%{y}<br>%{x:+.2f} log-odds<extra></extra>"))
    fig.update_xaxes(title="Pushes risk UP (red) / DOWN (blue)")
    fig.update_layout(title="What drives this order's score")
    return _style(fig, 330, legend=False)


def compare_modes_fig(t: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=t["Shipping mode"], y=t["P(late) model"] * 100, name="Model",
                marker_color=[mode_color(m) for m in t["Shipping mode"]],
                text=[f"{v:.0%}" for v in t["P(late) model"]], textposition="outside")
    fig.add_trace(go.Scatter(x=t["Shipping mode"], y=t["P(late) lookup"] * 100, mode="markers", name="Lookup table",
                             marker=dict(symbol="line-ew", size=26, line=dict(width=3, color="#111827"))))
    fig.update_yaxes(title="Chance of being late (%)", range=[0, 118])
    fig.update_layout(title="Same order, every shipping option")
    return _style(fig, 340)


# --------------------------------------------------------- early warning
def sweep_fig(sweep: pd.DataFrame, thr: float) -> go.Figure:
    fig = go.Figure()
    for col, name, color in (("precision", "Precision (flagged orders that are late)", "#6366f1"),
                             ("recall", "Recall (late orders caught)", "#f59e0b"),
                             ("flagged_share", "Share of orders flagged", "#94a3b8")):
        fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep[col] * 100, name=name, mode="lines",
                                 line=dict(width=3, color=color, shape="hv")))
    fig.add_vline(x=thr, line_dash="dash", line_color=BAD, annotation_text=f"threshold {thr:.2f}")
    fig.update_xaxes(title="Flag orders with P(late) at or above")
    fig.update_yaxes(title="%", range=[0, 105])
    fig.update_layout(title="The alert trade-off")
    return _style(fig, 360)


def outcome_bars(m: dict) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=["Late orders caught", "Late orders missed", "False alarms"],
        y=[m["tp"], m["late"] - m["tp"], m["false_alarms"]],
        marker_color=[GOOD, "#f59e0b", BAD], text=[f"{v:,.0f}" for v in (m["tp"], m["late"] - m["tp"], m["false_alarms"])],
        textposition="outside"))
    fig.update_layout(title="What the alert does to these orders")
    return _style(fig, 300, legend=False)


def cost_waterfall(gross: float, action_cost: float) -> go.Figure:
    fig = go.Figure(go.Waterfall(
        x=["Late-delivery cost avoided", "Cost of acting on alerts", "Net per month"],
        y=[gross, -action_cost, 0], measure=["relative", "relative", "total"],
        increasing=dict(marker=dict(color=GOOD)), decreasing=dict(marker=dict(color=BAD)),
        totals=dict(marker=dict(color="#6366f1")), connector=dict(line=dict(color=NEUTRAL))))
    fig.update_layout(title="Illustrative monthly value of the alert")
    return _style(fig, 320, legend=False)


# ----------------------------------------------------------- where / when
def heatmap(pivot: pd.DataFrame, title: str, as_percent: bool = True, text: bool = True) -> go.Figure:
    z = pivot.values * (100 if as_percent else 1)
    labels = [["" if np.isnan(v) else f"{v:.0f}" for v in row] for row in z] if text else None
    fig = go.Figure(go.Heatmap(z=z, x=list(pivot.columns), y=list(pivot.index), colorscale="RdYlGn_r",
                               zmin=0, zmax=100 if as_percent else None, text=labels,
                               texttemplate="%{text}" if text else None, colorbar=dict(title="% late"),
                               hovertemplate="%{y} / %{x}<br>%{z:.1f}% late<extra></extra>"))
    fig.update_layout(title=title)
    return _style(fig, max(320, 34 * len(pivot.index) + 120), legend=False)


def hour_weekday_heatmap(t: pd.DataFrame, title: str) -> go.Figure:
    z = t.pivot_table(index="order_weekday", columns="order_hour", values="late_rate")
    z = z.reindex(index=range(7), columns=range(24))
    fig = go.Figure(go.Heatmap(z=z.values * 100, x=list(range(24)), y=WEEKDAYS, colorscale="RdYlGn_r",
                               zmin=0, zmax=100, colorbar=dict(title="% late"),
                               hovertemplate="%{y} %{x}:00<br>%{z:.1f}% late<extra></extra>"))
    fig.update_xaxes(title="Hour the order was placed", dtick=2)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title)
    return _style(fig, 320, legend=False)


def monthly_lines(monthly: pd.DataFrame, modes: list, value: str = "late_rate") -> go.Figure:
    fig = go.Figure()
    for m in modes:
        t = monthly[monthly["Shipping Mode"] == m]
        y = t[value] * 100 if value == "late_rate" else t[value]
        fig.add_trace(go.Scatter(x=t["month"], y=y, name=m, mode="lines+markers",
                                 line=dict(width=3, color=mode_color(m)), marker=dict(size=5)))
    fig.update_yaxes(title="Late deliveries (%)" if value == "late_rate" else "Mean real days")
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(title="Month by month")
    return _style(fig, 420)


def gap_bars(t: pd.DataFrame, name: str, min_orders: int = 300) -> go.Figure:
    t = t[t["orders"] >= min_orders].sort_values("gap")
    colors = [BAD if g > 0 else "#38bdf8" for g in t["gap"]]
    fig = go.Figure(go.Bar(
        x=t["gap"] * 100, y=t[name].astype(str), orientation="h", marker_color=colors,
        customdata=np.c_[t["late_rate"] * 100, t["expected"] * 100, t["orders"], t["z"]],
        hovertemplate="<b>%{y}</b><br>Actual late: %{customdata[0]:.1f}%<br>Expected from its mode mix: %{customdata[1]:.1f}%"
                      "<br>Orders: %{customdata[2]:,.0f}<br>z = %{customdata[3]:.1f}<extra></extra>"))
    fig.update_xaxes(title="Percentage points later (+) or earlier (-) than its shipping-mode mix predicts")
    fig.update_layout(title=f"Is a {name.lower()} really worse, or just using riskier shipping modes?")
    return _style(fig, max(300, 26 * len(t) + 110), legend=False)


# ------------------------------------------------------------- model lab
def roc_fig(roc: dict) -> go.Figure:
    colors = {"Calibrated LightGBM": "#6366f1", "5-row lookup table": "#f59e0b", "Shipping mode only": "#94a3b8"}
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Coin flip",
                               line=dict(dash="dot", color=NEUTRAL)))
    for name, r in roc.items():
        fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"{name} (AUC {r['auc']:.3f})",
                                 line=dict(width=3, color=colors.get(name, NEUTRAL))))
    fig.update_xaxes(title="False positive rate"); fig.update_yaxes(title="True positive rate")
    fig.update_layout(title="ROC curves on the held-out test period")
    return _style(fig, 400)


def calibration_fig(cal: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dot", color=NEUTRAL)))
    fig.add_trace(go.Scatter(x=cal["predicted"], y=cal["observed"], mode="markers+lines", name="Model",
                             marker=dict(size=np.clip(np.sqrt(cal["orders"]) / 4, 8, 40), color="#6366f1"),
                             customdata=cal["orders"], hovertemplate="Said %{x:.0%}, actually %{y:.0%}<br>%{customdata:,} orders<extra></extra>"))
    fig.update_xaxes(title="Predicted chance of late", tickformat=".0%")
    fig.update_yaxes(title="Observed late rate", tickformat=".0%")
    fig.update_layout(title="When the model says 70%, is it 70%?")
    return _style(fig, 400)


def split_timeline(info: dict) -> go.Figure:
    colors = {"Train": "#6366f1", "Validation": "#f59e0b", "Test": "#22c55e"}
    fig = go.Figure()
    for name, d in info.items():
        start, end = pd.Timestamp(d["start"]), pd.Timestamp(d["end"])
        fig.add_trace(go.Bar(y=["Time split"], x=[(end - start).total_seconds() * 1000], base=[start.strftime("%Y-%m-%d")], orientation="h",
                             name=f"{name} ({d['orders']:,} orders)", marker_color=colors.get(name, NEUTRAL),
                             hovertemplate=f"{name}<br>{start:%d %b %Y} to {end:%d %b %Y}<extra></extra>"))
    fig.update_layout(barmode="overlay", title="Train on the past, test on the future")
    fig.update_xaxes(type="date")
    return _style(fig, 200)


def tier_bars(summary: pd.DataFrame) -> go.Figure:
    """summary index = High/Medium/Low with columns orders, late_rate, share."""
    fig = go.Figure(go.Bar(
        x=list(summary.index), y=summary["late_rate"] * 100, marker_color=[TIER_COLORS[t] for t in summary.index],
        text=[f"{v:.0%} late<br>{n:,.0f} orders" for v, n in zip(summary["late_rate"], summary["orders"])],
        textposition="outside"))
    fig.update_yaxes(title="Late rate (%)", range=[0, 118])
    fig.update_layout(title="Does each risk tier mean what it says?")
    return _style(fig, 330, legend=False)


def promise_curves(dist: pd.DataFrame, promises: dict) -> go.Figure:
    """Late rate as a function of the promised days, one line per shipping option."""
    fig = go.Figure()
    days = list(dist.columns)
    for opt, row in dist.iterrows():
        late = [float(row.loc[row.index > d].sum()) * 100 for d in days]
        fig.add_trace(go.Scatter(x=days, y=late, name=opt, mode="lines+markers", line=dict(width=3, color=mode_color(opt),
                                 dash="dash" if "noon" in opt else "solid"), marker=dict(size=7)))
        cur = promises.get(opt)
        if cur is not None:
            fig.add_trace(go.Scatter(x=[cur], y=[late[days.index(cur)]], mode="markers", showlegend=False,
                                     marker=dict(size=16, color=mode_color(opt), line=dict(width=3, color="#111827")),
                                     hovertemplate=f"{opt}: current promise {cur} d<extra></extra>"))
    fig.update_xaxes(title="Days promised to the customer", dtick=1)
    fig.update_yaxes(title="Late deliveries (%)", range=[-3, 105])
    fig.update_layout(title="Late rate vs the promise (circled = what is promised today)")
    return _style(fig, 400)
