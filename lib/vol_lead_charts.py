"""Plotly figures for pages/14_VIX_vs_DVOL.py, in the fx_style look."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib import fx_style as fx

H = 430
ASSET_COLOR = {"BTC": "#F7931A", "ETH": "#627EEA"}


def _base(fig: go.Figure, title: str, height: int = H) -> go.Figure:
    fig.update_layout(template=fx.TEMPLATE, title=title, height=height, hovermode="x unified",
                      margin=dict(l=55, r=55, t=70, b=40), paper_bgcolor="white", plot_bgcolor="white")
    return fig


def _done(fig, note=None):
    return fx.finalize(fig, note=note)


def levels(df: pd.DataFrame, asset: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=df.index, y=df["DVOL"], name=f"{asset} DVOL", line=dict(color=ASSET_COLOR.get(asset, fx.NAVY), width=2))
    fig.add_scatter(x=df.index, y=df["VIX"], name="VIX", line=dict(color=fx.NAVY, width=1.6))
    if "MOVE" in df:
        fig.add_scatter(x=df.index, y=df["MOVE"], name="MOVE (right)", line=dict(color=fx.GREY, width=1.2, dash="dot"),
                        secondary_y=True)
    _base(fig, f"{asset} DVOL vs VIX and MOVE")
    fig.update_yaxes(title_text="Vol (%)", secondary_y=False)
    fig.update_yaxes(title_text="MOVE (bp)", secondary_y=True, showgrid=False)
    return _done(fig, "DVOL at the 16:00 New York close; TradFi trading days only.")


def spread_ratio(df: pd.DataFrame, asset: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=(f"{asset} DVOL − VIX (vol pts)", f"{asset} DVOL / VIX", f"{asset} DVOL / MOVE"))
    for r, col, c in ((1, "SPREAD_VIX", fx.BLUE), (2, "RATIO_VIX", fx.NAVY), (3, "RATIO_MOVE", fx.GREY)):
        if col not in df:
            continue
        s = df[col]
        fig.add_scatter(x=df.index, y=s, name=col, line=dict(color=c, width=1.5), row=r, col=1, showlegend=False)
        m = s.rolling(252, min_periods=126).mean()
        fig.add_scatter(x=df.index, y=m, name="1y mean", line=dict(color=fx.AMBER, width=1, dash="dash"),
                        row=r, col=1, showlegend=(r == 1))
    _base(fig, f"{asset} DVOL spreads and ratios to TradFi vol", height=620)
    return _done(fig)


def cross_corr(cc: pd.DataFrame, asset: str, xname: str = "VIX") -> go.Figure:
    sig = cc["corr"].abs() > cc["band"]
    colors = [fx.RED if (s and l > 0) else (fx.BLUE if s else "#B8C4D2") for s, l in zip(sig, cc["lag"])]
    fig = go.Figure(go.Bar(x=cc["lag"], y=cc["corr"], marker_color=colors, name="corr",
                           hovertemplate="lag %{x}d: %{y:.3f}<extra></extra>"))
    b = float(cc["band"].median())
    for y in (b, -b):
        fig.add_hline(y=y, line=dict(color=fx.GREY, dash="dash", width=1))
    _base(fig, f"Lead-lag: daily Δ{xname} (t−k) vs Δ{asset} DVOL (t)")
    fig.update_xaxes(title_text=f"k (days). k > 0: {xname} leads DVOL", dtick=1)
    fig.update_yaxes(title_text="Correlation")
    return _done(fig, f"Dashed: ±1.96/√n noise band (±{b:.3f}). Red = significant with {xname} leading.")


def event_paths(paths: dict, title: str, uncond: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    palette = [fx.RED, fx.NAVY, fx.GREEN, fx.AMBER]
    for (name, p), c in zip(paths.items(), palette):
        if p is None or p.empty or p["n"].max() == 0:
            continue
        rgba = "rgba({},{},{},0.15)".format(*[int(c[i:i + 2], 16) for i in (1, 3, 5)])
        fig.add_scatter(x=list(p["offset"]) + list(p["offset"][::-1]), y=list(p["hi"]) + list(p["lo"][::-1]),
                        fill="toself", fillcolor=rgba, line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=p["offset"], y=p["mean"], name=f"{name} (n={int(p['n'].max())})",
                        line=dict(color=c, width=2))
    if uncond is not None and not uncond.empty:
        fig.add_scatter(x=uncond["offset"], y=uncond["mean"], name="All days (drift)",
                        line=dict(color=fx.GREY, width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color=fx.GREY, width=1))
    fig.add_hline(y=0, line=dict(color=fx.GREY, width=1))
    _base(fig, title)
    fig.update_xaxes(title_text="Trading days from the TradFi spike (day 0)")
    fig.update_yaxes(title_text="DVOL change vs day −1 (vol pts)")
    return _done(fig, "Shaded: 95% confidence band across events.")


def rolling(rb: pd.DataFrame, asset: str, xname: str = "VIX") -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=rb.index, y=rb["corr"], name="Rolling corr", line=dict(color=fx.NAVY, width=1.8))
    fig.add_scatter(x=rb.index, y=rb["beta"], name="Rolling beta (right)", line=dict(color=fx.AMBER, width=1.4),
                    secondary_y=True)
    _base(fig, f"Rolling 63d correlation and beta: Δ{asset} DVOL on Δ{xname} (same day)")
    fig.update_yaxes(title_text="Correlation", secondary_y=False)
    fig.update_yaxes(title_text="Beta (DVOL pts per VIX pt)", secondary_y=True, showgrid=False)
    return _done(fig)


def rich_cheap(df: pd.DataFrame, rc: pd.DataFrame, asset: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.55, 0.45],
                        subplot_titles=(f"{asset} DVOL vs TradFi fair value (rolling 1y fit on VIX, MOVE)",
                                        "Composite rich/cheap score (sd; + = crypto vol rich)"))
    fig.add_scatter(x=df.index, y=df["DVOL"], name=f"{asset} DVOL", line=dict(color=ASSET_COLOR.get(asset, fx.NAVY), width=2),
                    row=1, col=1)
    fig.add_scatter(x=rc.index, y=rc["fair"], name="Fair value", line=dict(color=fx.NAVY, width=1.4, dash="dash"),
                    row=1, col=1)
    s = rc["score"]
    fig.add_bar(x=rc.index, y=s.where(s > 0), name="Rich", marker_color=fx.RED, row=2, col=1)
    fig.add_bar(x=rc.index, y=s.where(s <= 0), name="Cheap", marker_color=fx.GREEN, row=2, col=1)
    for y in (1, -1):
        fig.add_hline(y=y, line=dict(color=fx.GREY, dash="dot", width=1), row=2, col=1)
    _base(fig, f"Is {asset} vol rich or cheap vs TradFi vol?", height=620)
    fig.update_layout(bargap=0)
    return _done(fig, "Score = mean of the fair-value residual z and the 1y z-scores of DVOL/VIX and DVOL/MOVE.")


def quintile_bars(q: pd.DataFrame, asset: str) -> go.Figure:
    fig = go.Figure()
    for h, c in zip(sorted(q["h"].unique()), (fx.BLUE, fx.NAVY, fx.AMBER)):
        d = q[q["h"] == h]
        fig.add_bar(x=d["quintile"], y=d["fwd_dvol_mean"], name=f"{h}d forward", marker_color=c)
    _base(fig, f"{asset}: forward DVOL change by rich/cheap quintile")
    fig.update_xaxes(title_text="Score quintile (Q1 cheapest → Q5 richest)")
    fig.update_yaxes(title_text="Avg DVOL change (vol pts)")
    return _done(fig, "Full-sample quintile edges (descriptive).")


def ranking_heatmap(rank: pd.DataFrame, asset: str) -> go.Figure:
    piv = rank.pivot_table(index="label", columns="h", values="oos_r2_gain_pct")
    pv = rank.pivot_table(index="label", columns="h", values="cw_p")
    order = piv.mean(axis=1).sort_values().index
    piv, pv = piv.loc[order], pv.loc[order]
    text = [[f"{v:+.1f}%" + ("*" if p < 0.05 else "") for v, p in zip(rv, rp)] for rv, rp in zip(piv.values, pv.values)]
    lim = float(np.nanmax(np.abs(piv.values.clip(-10, 10)))) or 1.0
    fig = go.Figure(go.Heatmap(z=piv.values.clip(-lim, lim), x=[f"{h}d" for h in piv.columns], y=list(piv.index),
                               text=text, texttemplate="%{text}", colorscale=[[0, fx.RED], [0.5, "#FFFFFF"], [1, fx.GREEN]],
                               zmid=0, zmin=-lim, zmax=lim, colorbar=dict(title="OOS R² gain %")))
    _base(fig, f"Which TradFi gauge adds out-of-sample power for {asset} DVOL?", height=420)
    fig.update_layout(hovermode="closest")
    return _done(fig, "Gain in out-of-sample R² over a DVOL-only model (expanding window). * = Clark-West p < 0.05.")
