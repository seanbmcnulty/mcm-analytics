"""Figures for the five Ecoinometrics-inspired charts added 2026-11-23.

Maths and data live in ``lib.eco_more``. These builders only draw, in the
page's Ecoinometrics look (``lib.eco_style``), so they match the other tabs
on screen and in a Telegram export (``fx_style.fig_to_png``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib import eco_style as es

GOLD = "#C9A227"
NAVY = "#1B4F72"
SCENARIO_COLOURS = {"upside": es.RED, "continuation": es.YELLOW, "downside": es.TEAL}
SCENARIO_LABELS = {"upside": "Upside", "continuation": "Continuation", "downside": "Downside"}


def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _empty(fig: go.Figure, text: str) -> go.Figure:
    fig.add_annotation(text=text, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(color=es.INK, size=15))
    return fig


def figure_yield_curve(cc, source_label: str) -> go.Figure:
    """Treasury curve on the latest date vs the full curve one year earlier."""
    fig = go.Figure()
    notes = (f"Notes: constant-maturity Treasury yields. Latest full curve vs the last full curve a year "
             f"earlier; nothing interpolated. Source: {source_label}.")
    if cc is None:
        es.style(fig, "Rates Are Higher Than A Year Ago", "Rates", "No complete curve for both dates.", notes)
        return _empty(fig, "Yield curve unavailable")
    t = cc.table
    fig.add_trace(go.Scatter(x=t["tenor"], y=t["ago"], mode="lines+markers", name=f"{cc.ago_date:%d %b %Y}",
                             line=dict(color=es.TEAL, width=3, dash="dash"), marker=dict(size=8),
                             hovertemplate="%{x}: %{y:.2f}%<extra>" + f"{cc.ago_date:%d %b %Y}" + "</extra>"))
    fig.add_trace(go.Scatter(x=t["tenor"], y=t["now"], mode="lines+markers", name=f"{cc.now_date:%d %b %Y}",
                             line=dict(color=es.ORANGE, width=3.5), marker=dict(size=9),
                             customdata=t["change"],
                             hovertemplate="%{x}: %{y:.2f}% (%{customdata:+.2f} pp)<extra>"
                                           + f"{cc.now_date:%d %b %Y}" + "</extra>"))
    lo = float(min(t["now"].min(), t["ago"].min()))
    hi = float(max(t["now"].max(), t["ago"].max()))
    for _, row in t.iterrows():
        fig.add_annotation(x=row["tenor"], y=lo - 0.12 * (hi - lo + 0.1), text=f"{row['change']:+.2f}",
                           showarrow=False, font=dict(family=es.SERIF, size=12,
                                                      color=es.RED if row["change"] > 0 else es.TEAL))
    word = "higher" if cc.avg_change >= 0 else "lower"
    es.note(fig, t["tenor"].iloc[3], float(t["now"].iloc[3]),
            f"Average change {cc.avg_change:+.2f} pp", color=es.INK, ay=-55, ax=40)
    es.style(fig, "Rates Are Higher Than A Year Ago" if cc.avg_change >= 0 else "Rates Are Lower Than A Year Ago",
             "Rates",
             f"The Treasury curve is on average {abs(cc.avg_change):.2f} percentage points {word} than a year ago "
             f"({cc.now_date:%d %b %Y} vs {cc.ago_date:%d %b %Y}). Row under the curve: change per tenor, pp.",
             notes, yfmt=".2f")
    fig.update_yaxes(ticksuffix="%", title_text="Yield", range=[lo - 0.25 * (hi - lo + 0.1), hi + 0.1 * (hi - lo + 0.1)])
    fig.update_xaxes(title_text="Maturity", type="category")
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.02, x=0.62, bgcolor="rgba(0,0,0,0)"))
    return fig


def figure_flow_dots(dots: pd.DataFrame, stats: dict) -> go.Figure:
    """One dot per Farside session (BTC); the current month is circled and totalled."""
    fig = go.Figure()
    notes = ("Notes: Farside daily US$ total converted to BTC at that day's Deribit close, not ETF holdings "
             "changes. Source: Farside, Deribit.")
    if dots is None or dots.empty or not stats:
        es.style(fig, "Bitcoin ETF Flows Day By Day", "Bitcoin", "No aligned sessions.", notes)
        return _empty(fig, "No ETF flow data")
    rest, cur = dots[~dots["in_month"]], dots[dots["in_month"]]
    colours = [es.RED if v > 0 else es.TEAL for v in rest["flow"]]
    fig.add_trace(go.Scatter(x=rest["date"], y=rest["flow"], mode="markers", name="Earlier sessions",
                             marker=dict(size=6, color=colours, opacity=0.55),
                             hovertemplate="%{x|%d %b %Y}: %{y:,.0f} BTC<extra></extra>"))
    m = stats["month"]
    fig.add_trace(go.Scatter(x=cur["date"], y=cur["flow"], mode="markers", name=f"{m:%B %Y}",
                             marker=dict(size=12, color=[es.RED if v > 0 else es.TEAL for v in cur["flow"]],
                                         line=dict(color=es.INK, width=2)),
                             hovertemplate="%{x|%d %b %Y}: %{y:,.0f} BTC<extra>" + f"{m:%b %Y}" + "</extra>"))
    fig.add_hline(y=0, line_color=es.INK, line_width=1.2)
    if len(cur):
        fig.add_vrect(x0=m, x1=stats["last_date"] + pd.Timedelta(days=1), fillcolor=es.ORANGE, opacity=0.08,
                      line_width=0)
        lowest = cur.loc[cur["flow"].idxmin()]
        es.note(fig, lowest["date"], float(lowest["flow"]),
                f"{m:%B} so far: {stats['total']:+,.0f} BTC<br>{stats['inflow_days']} inflow / "
                f"{stats['outflow_days']} outflow days",
                color=es.INK, ax=-120, ay=10)
    word = "Only" if stats["inflow_days"] <= 3 else ""
    title = (f"{word} {stats['inflow_days']} Days Of Inflows For The Bitcoin ETFs In {m:%B}".strip()
             if stats["sessions"] else "Bitcoin ETF Flows Day By Day")
    es.style(fig, title, "Bitcoin ETFs" if stats["sessions"] else "Bitcoin",
             f"Daily net flow into US spot Bitcoin ETFs. {m:%B %Y} to {stats['last_date']:%d %b}: "
             f"{stats['total']:+,.0f} BTC over {stats['sessions']} sessions.",
             notes, yfmt=",.0f")
    fig.update_yaxes(title_text="Daily net flow (BTC)")
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.02, x=0.62, bgcolor="rgba(0,0,0,0)"))
    return fig


YTD_COLOURS = {"Bitcoin": es.ORANGE, "Gold (GLD)": GOLD, "Nasdaq-100": NAVY}


def figure_ytd_returns(df: pd.DataFrame, info: dict, year: int) -> go.Figure:
    fig = go.Figure()
    notes = ("Notes: price returns, not total returns. Gold is the GLD share price. "
             "Source: Deribit BTC-PERP, SPDR Gold Shares (GLD), FRED (NASDAQ100).")
    if df is None or df.empty:
        es.style(fig, f"Year-To-Date Returns {year}", "Year-To-Date", "No series available.", notes)
        return _empty(fig, "No YTD data")
    for name, g in df.groupby("name", sort=False):
        fig.add_trace(go.Scatter(x=g["date"], y=g["ret"], mode="lines", name=name,
                                 line=dict(color=YTD_COLOURS.get(name, es.INK), width=3),
                                 hovertemplate="%{x|%d %b %Y}: %{y:+.1f}%<extra>" + name + "</extra>"))
        last = info[name]
        fig.add_annotation(x=last["last_date"], y=last["ret"], text=f"<b>{name} {last['ret']:+.1f}%</b>",
                           xanchor="left", xshift=6, showarrow=False,
                           font=dict(family=es.SERIF, size=13, color=YTD_COLOURS.get(name, es.INK)))
    fig.add_hline(y=0, line_color=es.INK, line_width=1.2)
    down = [n for n, v in info.items() if v["ret"] < 0]
    title = "Bitcoin Isn't Alone In A Downtrend" if "Bitcoin" in down and len(down) > 1 else f"Year-To-Date Returns {year}"
    parts = ", ".join(f"{n} {v['ret']:+.1f}%" for n, v in info.items())
    es.style(fig, title, "Downtrend" if "Downtrend" in title else "Year-To-Date",
             f"{year} price return to the latest close: {parts}.", notes, yfmt="+.0f")
    fig.update_yaxes(ticksuffix="%", title_text="Year-to-date price return")
    fig.update_xaxes(range=[df["date"].min(), df["date"].max() + pd.Timedelta(days=3)])
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.02, x=0.6, bgcolor="rgba(0,0,0,0)"),
                      margin=dict(r=170))
    return fig


def figure_btc_gold_flows(cum: pd.DataFrame, gold_note: str | None) -> go.Figure:
    fig = go.Figure()
    notes = ("Notes: from 11 Jan 2024. Gold is GLD only (change in shares outstanding x NAV), an estimate "
             "that leaves out IAU and other gold ETFs. Source: Farside, SPDR Gold Shares.")
    if gold_note:
        notes += f" {gold_note}"
    if cum is None or cum.empty:
        es.style(fig, "Bitcoin And Gold ETF Flows", "Gold", "No flow data.", notes)
        return _empty(fig, "No flow data")
    names = {"btc": ("Spot Bitcoin ETFs", es.ORANGE), "gold": ("GLD (gold, estimate)", GOLD)}
    lasts = {}
    for key, g in cum.groupby("series", sort=False):
        label, colour = names[key]
        fig.add_trace(go.Scatter(x=g["date"], y=g["cum_bn"], mode="lines", name=label,
                                 line=dict(color=colour, width=3),
                                 hovertemplate="%{x|%d %b %Y}: $%{y:,.1f}B<extra>" + label + "</extra>"))
        last = g.iloc[-1]
        lasts[key] = last
        fig.add_annotation(x=last["date"], y=last["cum_bn"], text=f"<b>${last['cum_bn']:,.1f}B</b>",
                           xanchor="left", xshift=6, showarrow=False,
                           font=dict(family=es.SERIF, size=13, color=colour))
    fig.add_hline(y=0, line_color=es.INK, line_width=1.2)
    bits = []
    if "btc" in lasts:
        bits.append(f"spot Bitcoin ETFs &#36;{lasts['btc']['cum_bn']:,.1f}B to {lasts['btc']['date']:%d %b %Y}")
    if "gold" in lasts:
        bits.append(f"GLD &#36;{lasts['gold']['cum_bn']:,.1f}B to {lasts['gold']['date']:%d %b %Y}")
    es.style(fig, "Bitcoin And Gold ETF Flows", "Gold",
             "Cumulative net flows since the Bitcoin ETF launch: " + "; ".join(bits) + ".", notes, yfmt=",.0f")
    fig.update_xaxes(range=[cum["date"].min(), cum["date"].max() + pd.Timedelta(days=45)])
    fig.update_yaxes(tickprefix="$", ticksuffix="B", title_text="Cumulative net flow (US$ billions)")
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.02, x=0.55, bgcolor="rgba(0,0,0,0)"),
                      margin=dict(r=110))
    return fig


def figure_flow_scenarios(sc, display_from=None) -> go.Figure:
    fig = go.Figure()
    notes = (f"Notes: ARIMA{sc.order} on {sc.fit_sessions} daily flows, {sc.n_paths:,} paths, thirds by outcome; "
             "median and 5th-95th band. Will not match Ecoinometrics' 1M-path numbers. Source: Farside, Deribit.")
    act = sc.actual.dropna(subset=["roll"])
    if display_from is not None:
        act = act[act["date"] >= pd.Timestamp(display_from)]
    fig.add_trace(go.Scatter(x=act["date"], y=act["roll"], mode="lines", name="Actual",
                             line=dict(color=es.INK, width=3),
                             hovertemplate="%{x|%d %b %Y}: %{y:,.0f} BTC<extra>Actual</extra>"))
    anchor_x = act["date"].iloc[-1] if len(act) else None
    anchor_y = float(act["roll"].iloc[-1]) if len(act) else None
    for name in ("upside", "continuation", "downside"):
        b = sc.bands[name]
        colour = SCENARIO_COLOURS[name]
        xs = ([anchor_x] if anchor_x is not None else []) + list(b["date"])
        def _with_anchor(v):
            return ([anchor_y] if anchor_y is not None else []) + list(v)
        fig.add_trace(go.Scatter(x=xs, y=_with_anchor(b["p95"]), mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=_with_anchor(b["p5"]), mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=_rgba(colour, 0.18), hoverinfo="skip", showlegend=False))
        label = SCENARIO_LABELS[name]
        fig.add_trace(go.Scatter(x=xs, y=_with_anchor(b["median"]), mode="lines", name=label,
                                 line=dict(color=colour, width=3, dash="dot"),
                                 hovertemplate="%{x|%d %b %Y}: %{y:,.0f} BTC<extra>" + label + " median</extra>"))
        fig.add_annotation(x=b["date"].iloc[-1], y=float(b["median"].iloc[-1]),
                           text=f"<b>{label} {sc.finals[name] / 1e3:+,.1f}K BTC</b>",
                           xanchor="left", xshift=6, showarrow=False,
                           font=dict(family=es.SERIF, size=13, color=colour))
    fig.add_hline(y=0, line_color=es.INK, line_width=1.2)
    last = float(act["roll"].iloc[-1]) if len(act) else float("nan")
    es.style(fig, "Bitcoin ETF Flow Scenarios", "Scenarios",
             f"30-session net flow is {last:+,.0f} BTC. Median 30-session flow 30 weekdays ahead: "
             f"upside {sc.finals['upside']:+,.0f}, continuation {sc.finals['continuation']:+,.0f}, "
             f"downside {sc.finals['downside']:+,.0f} BTC.", notes, yfmt=",.0f")
    fig.update_yaxes(title_text="30-session net flow (BTC)")
    x0 = act["date"].min() if len(act) else sc.future_dates[0]
    fig.update_xaxes(range=[x0, pd.Timestamp(sc.future_dates[-1]) + pd.Timedelta(days=4)])
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.02, x=0.45, bgcolor="rgba(0,0,0,0)"),
                      margin=dict(r=170))
    return fig
