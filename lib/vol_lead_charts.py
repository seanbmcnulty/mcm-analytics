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


def _done(fig, note=None, subplot_titles: bool = False):
    fig = fx.finalize(fig, note=note)
    if subplot_titles:
        # lift the legend above the first subplot's title
        t = (fig.layout.margin.t or 70) + 26
        fig.update_layout(margin=dict(t=t), legend=dict(y=1.07))
    return fig


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
    return _done(fig, subplot_titles=True)


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
    fig.add_scatter(x=rc.index, y=s.clip(lower=0), name="Rich", fill="tozeroy", mode="lines",
                    line=dict(color=fx.RED, width=0.8), fillcolor="rgba(192,57,43,0.45)", row=2, col=1)
    fig.add_scatter(x=rc.index, y=s.clip(upper=0), name="Cheap", fill="tozeroy", mode="lines",
                    line=dict(color=fx.GREEN, width=0.8), fillcolor="rgba(14,122,56,0.45)", row=2, col=1)
    for y in (1, -1):
        fig.add_hline(y=y, line=dict(color=fx.GREY, dash="dot", width=1), row=2, col=1)
    _base(fig, f"Is {asset} vol rich or cheap vs TradFi vol?", height=620)
    return _done(fig, "Score = mean of the fair-value residual z and the 1y z-scores of DVOL/VIX and DVOL/MOVE.",
                 subplot_titles=True)


def quintile_bars(q: pd.DataFrame, asset: str) -> go.Figure:
    fig = go.Figure()
    for h, c in zip(sorted(q["h"].unique()), (fx.BLUE, fx.NAVY, fx.AMBER)):
        d = q[q["h"] == h]
        fig.add_bar(x=d["quintile"], y=d["fwd_dvol_mean"], name=f"{h}d forward", marker_color=c)
    _base(fig, f"{asset}: forward DVOL change by rich/cheap quintile")
    fig.update_xaxes(title_text="Score quintile (Q1 cheapest → Q5 richest)")
    fig.update_yaxes(title_text="Avg DVOL change (vol pts)")
    return _done(fig, "Full-sample quintile edges (descriptive).")


def ranking_heatmap(rank: pd.DataFrame, asset: str, spec_label: str = "daily change") -> go.Figure:
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
    return _done(fig, f"Gain in out-of-sample R² over a DVOL-only model from adding the predictor's {spec_label} "
                      "(expanding window). * = Clark-West p < 0.05.")


# ================================================================ intraday
def _sgt(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx).tz_convert(fx.DISPLAY_TZ).tz_localize(None)


def _gap_breaks(x: pd.DatetimeIndex, bar: pd.Timedelta) -> list:
    """Plotly rangebreaks that hide the overnight / weekend gaps between sessions."""
    out = []
    if len(x) < 2:
        return out
    xs = x.sort_values()
    gaps = xs[1:] - xs[:-1]
    for a, g in zip(xs[:-1], gaps):
        if g > bar * 3:
            out.append(dict(values=[(a + bar).strftime("%Y-%m-%d %H:%M:%S")], dvalue=int((g - bar * 2) / pd.Timedelta(milliseconds=1))))
    return out


def intraday_overlay(df: pd.DataFrame, asset: str, bar: pd.Timedelta, title: str) -> go.Figure:
    x = _sgt(df.index)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=x, y=df[f"DVOL_{asset}"], name=f"{asset} DVOL", line=dict(color=ASSET_COLOR.get(asset, fx.NAVY), width=1.8))
    fig.add_scatter(x=x, y=df["VIX"], name="VIX (right)", line=dict(color=fx.NAVY, width=1.4), secondary_y=True)
    _base(fig, title)
    fig.update_xaxes(title_text=fx.XAXIS_TIME, rangebreaks=_gap_breaks(x, bar), tickformat=fx.TIME_TICKFORMAT)
    fig.update_yaxes(title_text=f"{asset} DVOL", secondary_y=False)
    fig.update_yaxes(title_text="VIX", secondary_y=True, showgrid=False)
    return _done(fig, "US cash session only (09:30-16:00 New York); overnight gaps hidden.")


def norm_overlay(z: pd.DataFrame, title: str, bar: pd.Timedelta) -> go.Figure:
    x = _sgt(z.index)
    fig = go.Figure()
    colors = {"VIX": fx.NAVY, "DVOL_BTC": ASSET_COLOR["BTC"], "DVOL_ETH": ASSET_COLOR["ETH"]}
    for c in z.columns:
        fig.add_scatter(x=x, y=z[c], name=c.replace("DVOL_", "") + (" DVOL" if c.startswith("DVOL") else ""),
                        line=dict(color=colors.get(c, fx.GREY), width=1.5))
    _base(fig, title)
    fig.update_xaxes(title_text=fx.XAXIS_TIME, rangebreaks=_gap_breaks(x, bar))
    fig.update_yaxes(title_text="z-score within the window")
    return _done(fig, "Each series standardised over the window shown, so moves are comparable.")


def change_scatter(dx: pd.Series, dy: pd.Series, fit: dict, asset: str, label: str) -> go.Figure:
    d = pd.concat([dx.rename("x"), dy.rename("y")], axis=1).dropna()
    fig = go.Figure()
    fig.add_scatter(x=d["x"], y=d["y"], mode="markers", name=f"{label} changes",
                    marker=dict(color=fx.BLUE, size=4, opacity=0.45))
    if np.isfinite(fit.get("beta", np.nan)) and len(d):
        xs = np.linspace(d["x"].min(), d["x"].max(), 20)
        fig.add_scatter(x=xs, y=fit["alpha"] + fit["beta"] * xs, mode="lines", name=f"beta {fit['beta']:.2f}",
                        line=dict(color=fx.RED, width=2))
    _base(fig, f"{label} ΔVIX vs Δ{asset} DVOL (same bar)")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="ΔVIX (pts)")
    fig.update_yaxes(title_text=f"Δ{asset} DVOL (pts)")
    return _done(fig, f"corr {fit.get('corr', np.nan):+.2f}, beta {fit.get('beta', np.nan):.2f} "
                      f"(HAC t {fit.get('t', np.nan):.1f}), n={fit.get('n', 0)}.")


def rolling_intraday(rbs: dict, asset: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Rolling correlation", "Rolling beta (DVOL pts per VIX pt)"))
    for (name, rb), c in zip(rbs.items(), (fx.AMBER, fx.NAVY)):
        x = _sgt(rb.index)
        fig.add_scatter(x=x, y=rb["corr"], name=name, line=dict(color=c, width=1.5), row=1, col=1)
        fig.add_scatter(x=x, y=rb["beta"], name=name, line=dict(color=c, width=1.5), row=2, col=1, showlegend=False)
    _base(fig, f"Hourly ΔVIX vs Δ{asset} DVOL: rolling 1-week and 1-month", height=560)
    return _done(fig, "Within-session hourly changes only.", subplot_titles=True)


def intraday_cross_corr(cc: pd.DataFrame, asset: str, unit: str) -> go.Figure:
    sig = cc["corr"].abs() > cc["band"]
    colors = [fx.RED if (s and l > 0) else (fx.BLUE if s else "#B8C4D2") for s, l in zip(sig, cc["lag"])]
    xv = cc["minutes"] if unit == "min" else cc["minutes"] / 60
    fig = go.Figure(go.Bar(x=xv, y=cc["corr"], marker_color=colors, name="corr",
                           hovertemplate="%{x}" + unit + ": %{y:.3f}<extra></extra>"))
    b = float(cc["band"].median())
    for y in (b, -b):
        fig.add_hline(y=y, line=dict(color=fx.GREY, dash="dash", width=1))
    step = "5-min" if unit == "min" else "hourly"
    _base(fig, f"Intraday lead-lag ({step}): ΔVIX (t−k) vs Δ{asset} DVOL (t)")
    fig.update_xaxes(title_text=f"k ({unit}). k > 0: VIX leads DVOL")
    fig.update_yaxes(title_text="Correlation")
    return _done(fig, f"Within-session pairs only. Dashed: ±1.96/√n (±{b:.3f}). Red = significant with VIX leading.")


def intraday_events(paths: dict, asset: str) -> go.Figure:
    fig = go.Figure()
    for (name, p), c in zip(paths.items(), (fx.RED, fx.GREEN)):
        if p is None or p.empty or p["n"].max() == 0:
            continue
        rgba = "rgba({},{},{},0.15)".format(*[int(c[i:i + 2], 16) for i in (1, 3, 5)])
        fig.add_scatter(x=list(p["minutes"]) + list(p["minutes"][::-1]), y=list(p["hi"]) + list(p["lo"][::-1]),
                        fill="toself", fillcolor=rgba, line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=p["minutes"], y=p["mean"], name=f"{name} (n={int(p['n'].max())})", line=dict(color=c, width=2))
    fig.add_vline(x=0, line=dict(color=fx.GREY, width=1))
    fig.add_hline(y=0, line=dict(color=fx.GREY, width=1))
    _base(fig, f"{asset} DVOL around large 5-min VIX moves")
    fig.update_xaxes(title_text="Minutes from the VIX move (bar end)")
    fig.update_yaxes(title_text="DVOL change vs the bar before (vol pts)")
    return _done(fig, "Move = 5-min ΔVIX beyond 3 trailing sd; same session only. Shaded: 95% band.")


def today_fair(fv: pd.DataFrame, asset: str, day_label: str) -> go.Figure:
    x = _sgt(fv.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35],
                        subplot_titles=(f"{asset} DVOL vs TradFi fair value (5-min)", "Rich (+) / cheap (−), vol pts"))
    fig.add_scatter(x=x, y=fv["DVOL"], name=f"{asset} DVOL", line=dict(color=ASSET_COLOR.get(asset, fx.NAVY), width=2), row=1, col=1)
    fig.add_scatter(x=x, y=fv["fair"], name="Fair value", line=dict(color=fx.NAVY, width=1.4, dash="dash"), row=1, col=1)
    r = fv["resid"]
    fig.add_scatter(x=x, y=r.clip(lower=0), fill="tozeroy", mode="lines", name="Rich", line=dict(color=fx.RED, width=0.8),
                    fillcolor="rgba(192,57,43,0.45)", row=2, col=1)
    fig.add_scatter(x=x, y=r.clip(upper=0), fill="tozeroy", mode="lines", name="Cheap", line=dict(color=fx.GREEN, width=0.8),
                    fillcolor="rgba(14,122,56,0.45)", row=2, col=1)
    _base(fig, f"Today: {asset} DVOL vs fair value intraday ({day_label})", height=520)
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_xaxes(title_text=fx.XAXIS_TIME, row=2, col=1)
    return _done(fig, "Fair value = latest daily fit (window chosen in the sidebar) applied to live VIX; MOVE held at its last close.",
                 subplot_titles=True)
