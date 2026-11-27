"""Figures for the Daily Shot-style chart book (data and maths in ``lib.dailyshot``).

One builder per chart ``kind``; each takes (chart, Ctx) and returns
(figure, info) where info carries ``as_of`` (last data date) and ``source``
(human-readable data source with live / committed-CSV status).
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib import dailyshot as ds

BG, INK, MUTED, GRID = "#FFFFFF", "#1A1A1A", "#6B6B6B", "#E6E6E6"
PALETTE = ["#1F3A93", "#C0392B", "#16A085", "#D4A017", "#8E44AD", "#7F8C8D"]
POS, NEG = "#1F3A93", "#C0392B"
FONT = "'Helvetica Neue', Arial, sans-serif"
HOW = {"live": "live", "csv": "committed CSV", "live+csv": "live + CSV", "none": "unavailable"}


@dataclass
class Ctx:
    prices: "ds.Dataset"
    fred: "ds.Dataset"
    yields: "ds.Dataset"
    dvol: "ds.Dataset"


class NoData(ValueError):
    pass


def style(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    fig.update_layout(
        template="plotly_white", paper_bgcolor=BG, plot_bgcolor=BG, height=height,
        font=dict(family=FONT, color=INK, size=12),
        title=dict(text=f"<b>{title}</b>", x=0, xanchor="left", font=dict(size=17, color=INK)),
        margin=dict(l=60, r=60, t=70, b=40),
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=MUTED)
    fig.add_annotation(text="MCM Analytics", xref="paper", yref="paper", x=1.0, y=1.0, xanchor="right",
                       yanchor="bottom", yshift=28, showarrow=False, font=dict(size=11, color=MUTED))
    return fig


def _how(dset: "ds.Dataset", cols) -> str:
    return ", ".join(sorted({HOW[dset.source.get(c, "none")] for c in cols}))


def _yahoo(ctx, cols):
    return f"Yahoo Finance daily closes ({_how(ctx.prices, cols)})"


def _px(ctx, sym, years) -> pd.Series:
    if sym not in ctx.prices.df:
        raise NoData(f"{sym}: no data")
    s = ds.window(ctx.prices.df[sym], years)
    if s.empty:
        raise NoData(f"{sym}: no data")
    return s


def _info(as_of, source):
    return {"as_of": pd.Timestamp(as_of), "source": source}


# ------------------------------------------------------------------- kinds
def k_lines(c, ctx):
    a = c.args
    fig = go.Figure()
    last = []
    series = {lab: _px(ctx, sym, a["years"] + (1 if a.get("ma") else 0)) for sym, lab in a["syms"]}
    start = max(s.index[-1] for s in series.values()) - pd.DateOffset(days=int(365.25 * a["years"]))
    if a.get("rebase"):
        df = pd.DataFrame(series)
        df = df[df.index > start].ffill(limit=3)
        series = ds.rebase(df).to_dict("series")
        fig.add_hline(y=100, line=dict(color=MUTED, dash="dot", width=1))
    for i, (lab, s) in enumerate(series.items()):
        s = s.dropna()
        full = s
        s = s[s.index > start]
        fig.add_scatter(x=s.index, y=s, name=lab, line=dict(color=PALETTE[i % 6], width=2))
        for j, n in enumerate(a.get("ma") or []):
            m = full.rolling(n).mean()
            m = m[m.index > start]
            fig.add_scatter(x=m.index, y=m, name=f"{n}-day avg", line=dict(color=PALETTE[(j + 1) % 6], width=1.3,
                                                                        dash="dash"))
        last.append(s.index[-1])
    style(fig, c.title)
    if a.get("log"):
        fig.update_yaxes(type="log")
    return fig, _info(max(last), _yahoo(ctx, [s for s, _ in a["syms"]]))


def k_dual(c, ctx):
    a = c.args
    (ls, ll), (rs, rl) = a["left"], a["right"]
    l, r = _px(ctx, ls, a["years"]), _px(ctx, rs, a["years"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=l.index, y=l, name=ll, line=dict(color=PALETTE[0], width=2))
    fig.add_scatter(x=r.index, y=r, name=rl + (" (right, axis inverted)" if a.get("invert_right") else " (right)"),
                    line=dict(color=PALETTE[1], width=1.6), secondary_y=True)
    style(fig, c.title)
    fig.update_yaxes(showgrid=False, secondary_y=True)
    if a.get("invert_right"):   # up = stronger local currency, without inverting the quote itself
        fig.update_yaxes(autorange="reversed", secondary_y=True)
    return fig, _info(max(l.index[-1], r.index[-1]), _yahoo(ctx, [ls, rs]))


def k_ratio(c, ctx):
    a = c.args
    num, den = _px(ctx, a["num"], a["years"] + 1), _px(ctx, a["den"], a["years"] + 1)
    ratio = (num / den).dropna()
    if ratio.empty:
        raise NoData("no overlapping dates")
    start = ratio.index[-1] - pd.DateOffset(days=int(365.25 * a["years"]))
    fig = go.Figure()
    s = ratio[ratio.index > start]
    fig.add_scatter(x=s.index, y=s, name=f"{a['num']} / {a['den']}", line=dict(color=PALETTE[0], width=2))
    if a.get("ma"):
        m = ratio.rolling(a["ma"]).mean()
        m = m[m.index > start]
        fig.add_scatter(x=m.index, y=m, name=f"{a['ma']}-day avg", line=dict(color=PALETTE[1], width=1.3, dash="dash"))
    style(fig, c.title)
    return fig, _info(s.index[-1], _yahoo(ctx, [a["num"], a["den"]]))


def k_returns(c, ctx):
    a = c.args
    rows, last = [], []
    for sym, lab in a["syms"]:
        if sym in ctx.prices.df and ctx.prices.df[sym].notna().any():
            s = ctx.prices.df[sym].dropna()
            v = ds.period_return(s, a["period"])
            if np.isfinite(v):
                rows.append((lab, v))
                last.append(s.index[-1])
    if not rows:
        raise NoData("no data")
    rows.sort(key=lambda t: t[1])
    labs, vals = zip(*rows)
    fig = go.Figure(go.Bar(x=list(vals), y=list(labs), orientation="h",
                           marker_color=[POS if v >= 0 else NEG for v in vals],
                           text=[f"{v:+.1f}%" for v in vals], textposition="outside", cliponaxis=False))
    style(fig, c.title, height=max(330, 34 * len(rows) + 120))
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(ticksuffix="%", zeroline=True, zerolinecolor=MUTED)
    return fig, _info(max(last), _yahoo(ctx, [s for s, _ in a["syms"]]))


def k_fred(c, ctx):
    a = c.args
    fig = go.Figure()
    last = []
    for i, (sid, lab) in enumerate(a["series"]):
        if sid not in ctx.fred.df or ctx.fred.df[sid].notna().sum() == 0:
            raise NoData(f"{sid}: no data")
        s = ctx.fred.df[sid].dropna()
        t = a.get("transform", "level")
        if t == "yoy":
            s = ds.yoy(s)
        elif t == "diff":
            s = s.diff().dropna()
        s = s * a.get("scale", 1)
        s = ds.window(s, a["years"])
        if a.get("bar"):
            lo, hi = a.get("clip", (None, None))
            fig.add_bar(x=s.index, y=s, name=lab, marker_color=[POS if v >= 0 else NEG for v in s])
            if lo is not None:
                fig.update_yaxes(range=[max(lo, s.min() * 1.15), min(hi, s.max() * 1.15)])
        else:
            fig.add_scatter(x=s.index, y=s, name=lab, line=dict(color=PALETTE[i % 6], width=2,
                                                                shape="hv" if a.get("step") else "linear"))
        if a.get("ma"):
            m = s.rolling(a["ma"]).mean()
            fig.add_scatter(x=m.index, y=m, name=f"{a['ma']}-period avg", line=dict(color=PALETTE[1], width=1.5))
        last.append(s.index[-1])
    style(fig, c.title)
    if a.get("suffix"):
        fig.update_yaxes(ticksuffix=a["suffix"])
    ids = [sid for sid, _ in a["series"]]
    return fig, _info(max(last), f"FRED {', '.join(ids)} ({_how(ctx.fred, ids)})")


def _ysrc(ctx, cols):
    return f"US Treasury daily par yields ({_how(ctx.yields, cols)})"


def k_curve(c, ctx):
    df = ctx.yields.df.dropna(how="all")
    if df.empty:
        raise NoData("no yields")
    cols = [t for t in ds.TENORS.values() if t in df]
    end = df.index[-1]
    fig = go.Figure()
    for i, (lab, d) in enumerate((("Latest", end), ("1 month ago", end - pd.DateOffset(months=1)),
                                  ("1 year ago", end - pd.DateOffset(years=1)))):
        row = df[df.index <= d]
        if row.empty:
            continue
        r = row.iloc[-1][cols].dropna()
        fig.add_scatter(x=list(r.index), y=r.values, name=f"{lab} ({row.index[-1]:%d %b %Y})", mode="lines+markers",
                        line=dict(color=PALETTE[i], width=2.5 if i == 0 else 1.6, dash=None if i == 0 else "dot"))
    style(fig, c.title)
    fig.update_layout(hovermode="x")
    fig.update_xaxes(type="category")
    fig.update_yaxes(ticksuffix="%")
    return fig, _info(end, _ysrc(ctx, cols))


def k_yields(c, ctx):
    a = c.args
    fig = go.Figure()
    last = []
    for i, (t, lab) in enumerate(a["tenors"]):
        s = ds.window(ctx.yields.df[t], a["years"])
        if s.empty:
            raise NoData(f"{t}: no data")
        fig.add_scatter(x=s.index, y=s, name=lab, line=dict(color=PALETTE[i], width=2))
        last.append(s.index[-1])
    style(fig, c.title)
    fig.update_yaxes(ticksuffix="%")
    return fig, _info(max(last), _ysrc(ctx, [t for t, _ in a["tenors"]]))


def k_yspread(c, ctx):
    a = c.args
    s = ds.window((ctx.yields.df[a["a"]] - ctx.yields.df[a["b"]]) * 100, a["years"])
    if s.empty:
        raise NoData("no yields")
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s, name=f"{a['a']} - {a['b']}", line=dict(color=PALETTE[0], width=2),
                    fill="tozeroy", fillcolor="rgba(31,58,147,0.08)")
    fig.add_hline(y=0, line=dict(color=NEG, width=1))
    style(fig, c.title)
    return fig, _info(s.index[-1], _ysrc(ctx, [a["a"], a["b"]]))


def k_spread(c, ctx):
    a = c.args
    s = (_px(ctx, a["a"], a["years"]) - _px(ctx, a["b"], a["years"])).dropna()
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s, name=f"{a['a']} - {a['b']}", line=dict(color=PALETTE[0], width=2))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    style(fig, c.title)
    return fig, _info(s.index[-1], _yahoo(ctx, [a["a"], a["b"]]))


def k_drawdown(c, ctx):
    a = c.args
    full = ctx.prices.df[a["sym"]].dropna()
    if full.empty:
        raise NoData("no data")
    s = ds.window(ds.drawdown(full), a["years"])
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s, name="Drawdown", line=dict(color=NEG, width=1.6), fill="tozeroy",
                    fillcolor="rgba(192,57,43,0.12)")
    style(fig, c.title)
    fig.update_yaxes(ticksuffix="%")
    return fig, _info(s.index[-1], _yahoo(ctx, [a["sym"]]) + "; peak within the 5-year Yahoo window")


def k_rvol_dvol(c, ctx):
    a = c.args
    rv = ds.window(ds.realised_vol(ctx.prices.df[a["sym"]]), a["years"])
    dv = ds.window(ctx.dvol.df[a["coin"]], a["years"]) if a["coin"] in ctx.dvol.df else pd.Series(dtype=float)
    if rv.empty and dv.empty:
        raise NoData("no data")
    fig = go.Figure()
    if len(dv):
        fig.add_scatter(x=dv.index, y=dv, name="DVOL (implied)", line=dict(color=PALETTE[0], width=2))
    fig.add_scatter(x=rv.index, y=rv, name="30-day realised", line=dict(color=PALETTE[1], width=1.6))
    style(fig, c.title)
    fig.update_yaxes(ticksuffix="%")
    src = f"Deribit DVOL ({_how(ctx.dvol, [a['coin']])}); " + _yahoo(ctx, [a["sym"]])
    return fig, _info(max(x.index[-1] for x in (rv, dv) if len(x)), src)


def k_corr(c, ctx):
    a = c.args
    s = ds.window(ds.rolling_corr(ctx.prices.df[a["a"]], ctx.prices.df[a["b"]], a["window"]), a["years"])
    if s.empty:
        raise NoData("no overlapping data")
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s, name=f"{a['window']}-day correlation", line=dict(color=PALETTE[0], width=2))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    style(fig, c.title)
    fig.update_yaxes(range=[-1, 1])
    return fig, _info(s.index[-1], _yahoo(ctx, [a["a"], a["b"]]))


KINDS = {"lines": k_lines, "dual": k_dual, "ratio": k_ratio, "returns": k_returns, "fred": k_fred,
         "curve": k_curve, "yields": k_yields, "yspread": k_yspread, "spread": k_spread,
         "drawdown": k_drawdown, "rvol_dvol": k_rvol_dvol, "corr": k_corr}


def build(chart, ctx):
    return KINDS[chart.kind](chart, ctx)
