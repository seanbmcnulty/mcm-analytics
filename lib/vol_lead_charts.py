"""Plotly figures for pages/14_VIX_vs_DVOL.py, in the fx_style look.

Page-wide colours: BTC orange, ETH blue-purple, VIX slate grey, MOVE green,
fair value navy dashed; rich = red, cheap = green. Lead-lag bars: coloured =
outside the 95% noise band, grey = noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib import fx_style as fx

H = 440
ASSET_COLOR = {"BTC": "#F7931A", "ETH": "#627EEA"}
VIX_C = "#5D6D7E"
MOVE_C = "#27AE60"
VVIX_C = "#8E44AD"
FAIR_C = fx.NAVY
RICH_C, CHEAP_C = fx.RED, fx.GREEN
RICH_FILL, CHEAP_FILL = "rgba(192,57,43,0.30)", "rgba(14,122,56,0.30)"
NOISE_C = "#D5D8DC"
REV_C = "#8E44AD"
PRED_COLOR = {"VIX": VIX_C, "MOVE": MOVE_C, "VVIX": VVIX_C}


def acol(asset: str) -> str:
    return ASSET_COLOR.get(asset, fx.NAVY)


def _rgba(c: str, a: float) -> str:
    return "rgba({},{},{},{})".format(*[int(c[i:i + 2], 16) for i in (1, 3, 5)], a)


def _base(fig: go.Figure, title: str, height: int = H) -> go.Figure:
    fig.update_layout(template=fx.TEMPLATE, title=title, height=height, hovermode="x unified",
                      margin=dict(l=60, r=60, t=70, b=50), paper_bgcolor="white", plot_bgcolor="white",
                      hoverlabel=dict(namelength=-1))
    return fig


def _done(fig, note=None, subplot_titles: bool = False):
    fig = fx.finalize(fig, note=note)
    if subplot_titles:
        # lift the legend above the first subplot's title
        t = (fig.layout.margin.t or 70) + 26
        fig.update_layout(margin=dict(t=t), legend=dict(y=1.07))
    return fig


def _fill_between(fig, x, actual, fair, row=None, col=None, names=("DVOL above fair (rich)", "DVOL below fair (cheap)")):
    """Shade red where actual > fair and green where actual < fair."""
    kw = dict(row=row, col=col) if row else {}
    a, f = pd.Series(np.asarray(actual, float)), pd.Series(np.asarray(fair, float))
    hi, lo = np.fmax(a, f), np.fmin(a, f)
    fig.add_scatter(x=x, y=f, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False, **kw)
    fig.add_scatter(x=x, y=hi, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=RICH_FILL,
                    name=names[0], hoverinfo="skip", **kw)
    fig.add_scatter(x=x, y=f, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False, **kw)
    fig.add_scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=CHEAP_FILL,
                    name=names[1], hoverinfo="skip", **kw)


def _last(s, fmt="{:.1f}") -> str:
    s = pd.Series(s).dropna()
    return fmt.format(float(s.iloc[-1])) if len(s) else "n/a"


def _end_label(fig, x, y, text, color, row=None, col=None, ax=40, ay=0, xref=None, yref=None):
    kw = {}
    if row:
        kw = dict(row=row, col=col)
    fig.add_annotation(x=x, y=y, text=text, showarrow=True, arrowhead=0, ax=ax, ay=ay, font=dict(color=color, size=11),
                       bgcolor="rgba(255,255,255,0.8)", **kw)


# ================================================================ rich / cheap
def rich_cheap_gauge(pct: float, score: float, resid: float, asset: str, span: str) -> go.Figure:
    """Cheap ← Fair → Rich scale (window percentile) with the current marker."""
    from lib.vol_lead_explain import ZONES, zone
    fig = go.Figure()
    fills = ["rgba(14,122,56,0.55)", "rgba(14,122,56,0.25)", "rgba(136,136,136,0.22)", "rgba(192,57,43,0.25)",
             "rgba(192,57,43,0.55)"]
    for (lo, hi, name), c in zip(ZONES, fills):
        fig.add_shape(type="rect", x0=lo, x1=hi, y0=0, y1=1, fillcolor=c, line=dict(width=1, color="white"))
        fig.add_annotation(x=(lo + hi) / 2, y=0.5, text=name, showarrow=False, font=dict(size=12, color="#1F2A37"))
    p = float(np.clip(pct, 0, 100)) if np.isfinite(pct) else 50.0
    word = "rich" if resid > 0 else "cheap"
    fig.add_scatter(x=[p], y=[1.18], mode="markers+text", marker=dict(symbol="triangle-down", size=22, color=fx.NAVY),
                    text=[f"<b>Now: {pct:.0f}th pct</b>"], textposition="top center", showlegend=False,
                    hovertemplate=f"Percentile {pct:.0f}<br>Score {score:+.2f} sd<br>{abs(resid):.1f} vols {word}<extra></extra>")
    fig.add_shape(type="line", x0=p, x1=p, y0=0, y1=1.1, line=dict(color=fx.NAVY, width=3))
    fig.update_layout(template=fx.TEMPLATE, height=230, margin=dict(l=20, r=20, t=50, b=50), paper_bgcolor="white",
                      plot_bgcolor="white", hovermode="closest",
                      title=(f"{asset} vol vs TradFi: {zone(pct).lower()} ({abs(resid):.1f} vols {word}, score {score:+.2f} sd, "
                             f"{pct:.0f}th percentile of {span})"))
    fig.update_xaxes(range=[0, 100], tickvals=[0, 20, 40, 60, 80, 100],
                     ticktext=["0<br>cheapest", "20", "40", "60", "80", "100<br>richest"], showgrid=False,
                     title_text=f"Percentile of the composite rich/cheap score over {span}   (Cheap ← Fair → Rich)")
    fig.update_yaxes(range=[0, 2.0], visible=False)
    fx.add_watermark(fig)
    fig.update_layout(title=dict(x=0.01, xanchor="left", font=dict(size=15)))
    return fig


def rich_cheap(df: pd.DataFrame, rc: pd.DataFrame, asset: str, fv_label: str = "6 months") -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10, row_heights=[0.58, 0.42],
                        subplot_titles=(f"{asset} DVOL vs TradFi fair value (rolling {fv_label} fit on VIX and MOVE)",
                                        "Composite rich/cheap score (sd; above 0 = crypto vol rich)"))
    x = df.index
    fair = rc["fair"].reindex(x)
    _fill_between(fig, x, df["DVOL"], fair, row=1, col=1)
    fig.add_scatter(x=x, y=fair, name=f"TradFi fair value (now {_last(fair)})", line=dict(color=FAIR_C, width=1.5, dash="dash"), row=1, col=1,
                    hovertemplate="Fair value %{y:.1f}<extra></extra>")
    fig.add_scatter(x=x, y=df["DVOL"], name=f"{asset} DVOL (now {_last(df['DVOL'])})", line=dict(color=acol(asset), width=2), row=1, col=1,
                    hovertemplate=f"{asset} DVOL " + "%{y:.1f}<extra></extra>")
    s = rc["score"].reindex(x)
    fig.add_hrect(y0=1, y1=max(3.0, float(np.nanmax(s.values)) + 0.2) if s.notna().any() else 3, row=2, col=1,
                  fillcolor="rgba(192,57,43,0.07)", line_width=0)
    fig.add_hrect(y0=min(-3.0, float(np.nanmin(s.values)) - 0.2) if s.notna().any() else -3, y1=-1, row=2, col=1,
                  fillcolor="rgba(14,122,56,0.07)", line_width=0)
    fig.add_scatter(x=x, y=s.clip(lower=0), fill="tozeroy", mode="lines", name="Score: rich",
                    line=dict(color=RICH_C, width=0.8), fillcolor="rgba(192,57,43,0.45)", row=2, col=1,
                    hovertemplate="Score %{y:+.2f} sd<extra></extra>", showlegend=False)
    fig.add_scatter(x=x, y=s.clip(upper=0), fill="tozeroy", mode="lines", name="Score: cheap",
                    line=dict(color=CHEAP_C, width=0.8), fillcolor="rgba(14,122,56,0.45)", row=2, col=1,
                    hovertemplate="Score %{y:+.2f} sd<extra></extra>", showlegend=False)
    for y, t in ((1, "rich zone (> +1 sd)"), (-1, "cheap zone (< −1 sd)")):
        fig.add_hline(y=y, line=dict(color=fx.GREY, dash="dot", width=1), row=2, col=1)
        fig.add_annotation(x=0.995, xref="x2 domain", y=y, yref="y2", text=t, showarrow=False, xanchor="right",
                           yanchor="bottom" if y > 0 else "top", font=dict(size=10, color=RICH_C if y > 0 else CHEAP_C))
    _base(fig, f"Is {asset} vol rich or cheap vs TradFi vol? (red = rich, green = cheap)", height=640)
    fig.update_yaxes(title_text="Vol points", row=1, col=1)
    fig.update_yaxes(title_text="Score (sd)", row=2, col=1)
    return _done(fig, f"Score = average of three z-scores over {fv_label}: fair-value gap, DVOL/VIX and DVOL/MOVE.",
                 subplot_titles=True)


def quintile_bars(q: pd.DataFrame, asset: str) -> go.Figure:
    fig = go.Figure()
    lab = {"Q1": "Q1<br>cheapest", "Q2": "Q2", "Q3": "Q3<br>fair", "Q4": "Q4", "Q5": "Q5<br>richest"}
    for h, c in zip(sorted(q["h"].unique()), ("#AFC6DD", "#5A8AB8", fx.NAVY)):
        d = q[q["h"] == h]
        fig.add_bar(x=[lab.get(v, v) for v in d["quintile"]], y=d["fwd_dvol_mean"], name=f"next {h} days",
                    marker_color=c, text=[f"{v:+.1f}" for v in d["fwd_dvol_mean"]], textposition="outside",
                    customdata=d["n"], hovertemplate="%{x}: %{y:+.2f} vols (n=%{customdata})<extra>" + f"{h}d</extra>")
    fig.add_hline(y=0, line=dict(color=fx.GREY, width=1))
    _base(fig, f"After a cheap vs rich reading, what did {asset} DVOL do next?")
    fig.update_layout(hovermode="closest", bargap=0.25)
    fig.update_xaxes(title_text="Rich/cheap score bucket (fifths of the window)")
    fig.update_yaxes(title_text="Average DVOL change (vol pts)")
    return _done(fig, "Mean reversion = bars fall from cheap (left) to rich (right). Buckets use the whole window (descriptive).")


def today_fair(fv: pd.DataFrame, asset: str, day_label: str) -> go.Figure:
    x = _sgt(fv.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10, row_heights=[0.65, 0.35],
                        subplot_titles=(f"{asset} DVOL vs TradFi fair value (5-min)", "Gap to fair value (vol pts; + = rich)"))
    _fill_between(fig, x, fv["DVOL"], fv["fair"], row=1, col=1)
    fig.add_scatter(x=x, y=fv["fair"], name=f"TradFi fair value (now {_last(fv['fair'])})", line=dict(color=FAIR_C, width=1.5, dash="dash"),
                    row=1, col=1, hovertemplate="Fair %{y:.1f}<extra></extra>")
    fig.add_scatter(x=x, y=fv["DVOL"], name=f"{asset} DVOL (now {_last(fv['DVOL'])})", line=dict(color=acol(asset), width=2), row=1, col=1,
                    hovertemplate=f"{asset} DVOL " + "%{y:.1f}<extra></extra>")
    r = fv["resid"]
    fig.add_scatter(x=x, y=r.clip(lower=0), fill="tozeroy", mode="lines", name="rich", showlegend=False,
                    line=dict(color=RICH_C, width=0.8), fillcolor="rgba(192,57,43,0.45)", row=2, col=1,
                    hovertemplate="Gap %{y:+.1f}<extra></extra>")
    fig.add_scatter(x=x, y=r.clip(upper=0), fill="tozeroy", mode="lines", name="cheap", showlegend=False,
                    line=dict(color=CHEAP_C, width=0.8), fillcolor="rgba(14,122,56,0.45)", row=2, col=1,
                    hovertemplate="Gap %{y:+.1f}<extra></extra>")
    _base(fig, f"Today: is {asset} DVOL rich or cheap vs TradFi? ({day_label})", height=540)
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_xaxes(title_text=fx.XAXIS_TIME, row=2, col=1)
    fig.update_yaxes(title_text="Vol points", row=1, col=1)
    return _done(fig, "Fair value = latest daily fit (sidebar window) applied to live VIX; MOVE held at its last close.",
                 subplot_titles=True)


# ================================================================ lead-lag
def _leadlag(x, corr, band, asset: str, pred: str, unit_word: str, title: str, note: str, peak_txt=None) -> go.Figure:
    x = np.asarray(x, float)
    corr = np.asarray(corr, float)
    band = np.asarray(band, float)
    sig = np.abs(corr) > band
    cats = [("VIX leads (significant)".replace("VIX", pred), sig & (x > 0), acol(asset)),
            ("Same time (significant)", sig & (x == 0), fx.NAVY),
            ("DVOL leads (significant)", sig & (x < 0), REV_C),
            ("Noise (inside band)", ~sig, NOISE_C)]
    fig = go.Figure()
    step = float(np.nanmin(np.diff(np.sort(x)))) if len(x) > 1 else 1.0
    for name, m, c in cats:
        if m.any():
            fig.add_bar(x=x[m], y=corr[m], name=name, marker_color=c, width=0.8 * step,
                        hovertemplate=f"%{{x}} {unit_word}: corr %{{y:+.3f}}<extra>{name}</extra>")
    b = float(np.nanmedian(band))
    fig.add_hrect(y0=-b, y1=b, fillcolor="rgba(136,136,136,0.15)", line_width=0)
    fig.add_annotation(x=1.0, xref="paper", y=b, text=f"95% noise band ±{b:.2f}", showarrow=False, xanchor="right",
                       yanchor="bottom", font=dict(size=10, color=fx.GREY))
    fig.add_hline(y=0, line=dict(color=fx.GREY, width=1))
    # peak of the unbroken run of significant positive leads starting at the first lag
    order = np.argsort(x)
    lead = np.zeros(len(x), bool)
    for j in order:
        if x[j] <= 0:
            continue
        if sig[j] and corr[j] > 0:
            lead[j] = True
        else:
            break
    if lead.any():
        i = int(np.argmax(np.where(lead, corr, -np.inf)))
        fig.add_annotation(x=x[i], y=corr[i], text=peak_txt or f"peak: {pred} leads by {x[i]:g} {unit_word}<br>corr {corr[i]:+.2f}",
                           showarrow=True, arrowhead=2, ax=45, ay=-35, font=dict(size=11, color=fx.NAVY),
                           bgcolor="rgba(255,255,255,0.85)")
    fig.add_annotation(x=0.01, xref="paper", y=1.0, yref="paper", text=f"← DVOL moves first", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(size=11, color=REV_C))
    fig.add_annotation(x=0.99, xref="paper", y=1.0, yref="paper", text=f"{pred} moves first →", showarrow=False,
                       xanchor="right", yanchor="top", font=dict(size=11, color=acol(asset)))
    _base(fig, title)
    fig.update_layout(hovermode="closest", barmode="overlay", bargap=0.15)
    fig.update_xaxes(title_text=f"Lag: {pred} leads DVOL by N {unit_word} →  (negative = DVOL leads)", zeroline=False)
    fig.update_yaxes(title_text="Correlation")
    return _done(fig, note)


def cross_corr(cc: pd.DataFrame, asset: str, xname: str = "VIX") -> go.Figure:
    f = _leadlag(cc["lag"], cc["corr"], cc["band"], asset, xname, "days",
                 f"Does {xname} lead {asset} DVOL? (daily changes)",
                 f"Correlation of the daily {xname} change with the {asset} DVOL change N days later.")
    f.update_xaxes(dtick=1)
    return f


def intraday_cross_corr(cc: pd.DataFrame, asset: str, unit: str) -> go.Figure:
    xv = cc["minutes"] if unit == "min" else cc["minutes"] / 60
    word = "min" if unit == "min" else "hours"
    step = "5-min bars" if unit == "min" else "hourly bars"
    f = _leadlag(xv, cc["corr"], cc["band"], asset, "VIX", word, f"Does VIX lead {asset} DVOL? ({step})",
                 "US cash hours only; pairs never span the overnight gap.")
    if unit == "min":
        f.update_xaxes(dtick=10)
    else:
        f.update_xaxes(dtick=1)
    return f


# ================================================================ event studies
def _event_fig(paths: dict, key: str, colors, asset: str, title: str, xtitle: str, event_txt: str, end_unit: str,
               uncond: pd.DataFrame | None = None, note: str = "") -> go.Figure:
    fig = go.Figure()
    for (name, p), c in zip(paths.items(), colors):
        if p is None or p.empty or p["n"].max() == 0:
            continue
        fig.add_scatter(x=list(p[key]) + list(p[key][::-1]), y=list(p["hi"]) + list(p["lo"][::-1]),
                        fill="toself", fillcolor=_rgba(c, 0.15), line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=p[key], y=p["mean"], name=f"{name} (n={int(p['n'].max())})", line=dict(color=c, width=2.2),
                        hovertemplate="%{x}: %{y:+.2f} vols<extra>" + name + "</extra>")
        last = p.dropna(subset=["mean"]).iloc[-1]
        lx = last[key]
        when = (f"{int(lx)}d" if end_unit == "d" else f"{int(lx) // 60}h" if lx >= 60 and lx % 60 == 0 else f"{int(lx)} min")
        fig.add_annotation(x=lx, y=last["mean"], text=f"{last['mean']:+.1f} vols after {when}", showarrow=True,
                           arrowhead=0, ax=-10, ay=-22 if last["mean"] >= 0 else 22, xanchor="right",
                           font=dict(size=11, color=c), bgcolor="rgba(255,255,255,0.85)")
    if uncond is not None and not uncond.empty:
        fig.add_scatter(x=uncond["offset"], y=uncond["mean"], name="All days (normal drift)",
                        line=dict(color=fx.GREY, width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color=fx.NAVY, width=1.2, dash="dash"))
    fig.add_annotation(x=0, y=1.0, yref="paper", text=event_txt, showarrow=False, xanchor="left", yanchor="top",
                       font=dict(size=11, color=fx.NAVY), xshift=4)
    fig.add_hline(y=0, line=dict(color=fx.GREY, width=1))
    _base(fig, title)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text=xtitle)
    fig.update_yaxes(title_text="Cumulative DVOL change (vol pts)")
    return _done(fig, note)


def event_paths(paths: dict, title: str, uncond: pd.DataFrame | None = None, asset: str = "BTC") -> go.Figure:
    return _event_fig(paths, "offset", (acol(asset), fx.NAVY, fx.GREEN, fx.AMBER), asset, title,
                      "Trading days from the TradFi vol spike (day 0)", "spike day", "d", uncond,
                      "Change vs the day before the spike. Shaded: 95% confidence band across events.")


def intraday_events(paths: dict, asset: str) -> go.Figure:
    return _event_fig(paths, "minutes", (fx.RED, fx.GREEN), asset,
                      f"What does {asset} DVOL do after a big 5-min VIX move?",
                      "Minutes from the VIX move (bar end)", "VIX move", "min", None,
                      "Move = 5-min VIX change beyond 3 trailing sd; same session only. Change vs the bar before. "
                      "Shaded: 95% band.")


# ================================================================ daily views
def levels(df: pd.DataFrame, asset: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=df.index, y=df["DVOL"], name=f"{asset} DVOL", line=dict(color=acol(asset), width=2),
                    hovertemplate="%{y:.1f}")
    fig.add_scatter(x=df.index, y=df["VIX"], name="VIX", line=dict(color=VIX_C, width=1.6), hovertemplate="%{y:.1f}")
    if "MOVE" in df:
        fig.add_scatter(x=df.index, y=df["MOVE"], name="MOVE (right axis)", line=dict(color=MOVE_C, width=1.3, dash="dot"),
                        secondary_y=True, hovertemplate="%{y:.0f}")
    _base(fig, f"{asset} DVOL vs VIX and MOVE (levels)")
    fig.update_yaxes(title_text="Implied vol (%)", secondary_y=False)
    fig.update_yaxes(title_text="MOVE (bp)", secondary_y=True, showgrid=False)
    return _done(fig, "DVOL at the 16:00 New York close; TradFi trading days only.")


def spread_ratio(df: pd.DataFrame, asset: str) -> go.Figure:
    specs = [("SPREAD_VIX", f"{asset} DVOL minus VIX (vol points)", acol(asset), "{:+.1f}"),
             ("RATIO_VIX", f"{asset} DVOL ÷ VIX (x)", VIX_C, "{:.2f}x"),
             ("RATIO_MOVE", f"{asset} DVOL ÷ MOVE (x)", MOVE_C, "{:.3f}x")]
    specs = [s for s in specs if s[0] in df]
    fig = make_subplots(rows=len(specs), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[s[1] for s in specs])
    for r, (col, lab, c, fmt) in enumerate(specs, start=1):
        s = df[col]
        fig.add_scatter(x=df.index, y=s, name=lab, line=dict(color=c, width=1.6), row=r, col=1, showlegend=False,
                        hovertemplate=lab.split(" (")[0] + " %{y:.2f}<extra></extra>")
        m = float(s.mean())
        fig.add_scatter(x=[df.index[0], df.index[-1]], y=[m, m], mode="lines", name="Window average",
                        line=dict(color=fx.GREY, width=1, dash="dash"), row=r, col=1, showlegend=(r == 1),
                        hovertemplate="average %{y:.2f}<extra></extra>")
        if s.notna().any():
            _end_label(fig, s.dropna().index[-1], float(s.dropna().iloc[-1]), "now " + fmt.format(s.dropna().iloc[-1]), c,
                       row=r, col=1, ax=-40, ay=-14)
    _base(fig, f"How big is {asset}'s vol premium over TradFi vol?", height=660)
    return _done(fig, subplot_titles=True)


def rolling(rb: pd.DataFrame, asset: str, xname: str = "VIX", window: int = 63) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                        subplot_titles=("Correlation (same day)", f"Beta (DVOL pts per 1-pt {xname} move)"))
    fig.add_scatter(x=rb.index, y=rb["corr"], name="Correlation", line=dict(color=fx.NAVY, width=1.8), row=1, col=1,
                    hovertemplate="corr %{y:+.2f}<extra></extra>", showlegend=False)
    fig.add_scatter(x=rb.index, y=rb["beta"], name="Beta", line=dict(color=acol(asset), width=1.6), row=2, col=1,
                    hovertemplate="beta %{y:+.2f}<extra></extra>", showlegend=False)
    for r in (1, 2):
        fig.add_hline(y=0, line=dict(color=fx.GREY, width=1), row=r, col=1)
    _base(fig, f"Is the same-day {xname} link to {asset} DVOL stable? (rolling {window} trading days)", height=520)
    return _done(fig, subplot_titles=True)


def ranking_heatmap(rank: pd.DataFrame, asset: str, spec_label: str = "daily change") -> go.Figure:
    piv = rank.pivot_table(index="label", columns="h", values="oos_r2_gain_pct")
    pv = rank.pivot_table(index="label", columns="h", values="cw_p")
    order = piv.max(axis=1).sort_values().index  # best at the top
    piv, pv = piv.loc[order], pv.loc[order]
    text = [[f"{v:+.1f}%" + (" ✅" if (p < 0.05 and v > 0) else "") for v, p in zip(rv, rp)] for rv, rp in zip(piv.values, pv.values)]
    lim = float(np.nanmax(np.abs(piv.values.clip(-5, 5)))) or 1.0
    fig = go.Figure(go.Heatmap(z=piv.values.clip(-lim, lim), x=[f"{h} day{'s' if h > 1 else ''} ahead" for h in piv.columns],
                               y=list(piv.index), text=text, texttemplate="%{text}",
                               customdata=piv.values, hovertemplate="%{y}, %{x}: %{customdata:+.2f}% gain<extra></extra>",
                               colorscale=[[0, fx.RED], [0.5, "#FFFFFF"], [1, fx.GREEN]], zmid=0, zmin=-lim, zmax=lim,
                               colorbar=dict(title="Forecast<br>gain %")))
    _base(fig, f"Which TradFi gauge improves {asset} DVOL forecasts? (green = helps)", height=440)
    fig.update_layout(hovermode="closest", margin=dict(l=200))
    fig.update_yaxes(automargin=True)
    return _done(fig, f"Out-of-sample forecast gain vs a DVOL-only model from adding the gauge's {spec_label}. "
                      "✅ = significant (Clark-West p < 0.05). Best gauges at the top.")


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
    fig.add_scatter(x=x, y=df[f"DVOL_{asset}"], name=f"{asset} DVOL (left)", line=dict(color=acol(asset), width=1.9),
                    hovertemplate="%{y:.1f}")
    fig.add_scatter(x=x, y=df["VIX"], name="VIX (right)", line=dict(color=VIX_C, width=1.5), secondary_y=True,
                    hovertemplate="%{y:.2f}")
    _base(fig, title)
    fig.update_xaxes(title_text=fx.XAXIS_TIME, rangebreaks=_gap_breaks(x, bar), tickformat=fx.TIME_TICKFORMAT)
    fig.update_yaxes(title_text=f"{asset} DVOL (vol pts)", secondary_y=False)
    fig.update_yaxes(title_text="VIX (vol pts)", secondary_y=True, showgrid=False)
    return _done(fig, "US cash session only (09:30-16:00 New York); overnight gaps hidden.")


def norm_overlay(z: pd.DataFrame, title: str, bar: pd.Timedelta) -> go.Figure:
    x = _sgt(z.index)
    fig = go.Figure()
    colors = {"VIX": VIX_C, "DVOL_BTC": ASSET_COLOR["BTC"], "DVOL_ETH": ASSET_COLOR["ETH"]}
    for c in z.columns:
        nm = c.replace("DVOL_", "") + (" DVOL" if c.startswith("DVOL") else "")
        fig.add_scatter(x=x, y=z[c], name=nm, line=dict(color=colors.get(c, fx.GREY), width=1.6),
                        hovertemplate="%{y:+.2f} sd")
    fig.add_hline(y=0, line=dict(color=fx.GREY, width=1))
    _base(fig, title)
    fig.update_xaxes(title_text=fx.XAXIS_TIME, rangebreaks=_gap_breaks(x, bar))
    fig.update_yaxes(title_text="Standard deviations from window average")
    return _done(fig, "Each series standardised over the window shown, so the shapes are comparable.")


def change_scatter(dx: pd.Series, dy: pd.Series, fit: dict, asset: str, label: str) -> go.Figure:
    d = pd.concat([dx.rename("x"), dy.rename("y")], axis=1).dropna()
    fig = go.Figure()
    fig.add_scatter(x=d["x"], y=d["y"], mode="markers", name=f"one {label.lower()} bar",
                    marker=dict(color=acol(asset), size=4, opacity=0.4),
                    hovertemplate="VIX %{x:+.2f}, DVOL %{y:+.2f}<extra></extra>")
    if np.isfinite(fit.get("beta", np.nan)) and len(d):
        lo, hi = d["x"].quantile([0.001, 0.999])
        xs = np.linspace(lo, hi, 20)
        fig.add_scatter(x=xs, y=fit["alpha"] + fit["beta"] * xs, mode="lines",
                        name=f"fit: DVOL = {fit['beta']:+.2f} × VIX move", line=dict(color=fx.NAVY, width=2.5))
        fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
                           align="left", bgcolor="rgba(255,255,255,0.9)", bordercolor="#CCC",
                           text=(f"beta {fit['beta']:+.2f} vol per VIX pt<br>corr {fit.get('corr', np.nan):+.2f} · "
                                 f"t {fit.get('t', np.nan):.1f} · n {fit.get('n', 0):,}"))
    fig.add_hline(y=0, line=dict(color=fx.GREY, width=1))
    fig.add_vline(x=0, line=dict(color=fx.GREY, width=1))
    _base(fig, f"How much does {asset} DVOL move per 1-pt VIX move? ({label} bars)")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="VIX change over the bar (pts)")
    fig.update_yaxes(title_text=f"{asset} DVOL change over the same bar (vol pts)")
    return _done(fig, "Same-bar moves within US cash hours. HAC t-stat.")


def rolling_intraday(rbs: dict, asset: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                        subplot_titles=("Correlation of hourly moves", "Beta (DVOL pts per 1-pt VIX move)"))
    for (name, rb), c in zip(rbs.items(), (acol(asset), fx.NAVY)):
        x = _sgt(rb.index)
        fig.add_scatter(x=x, y=rb["corr"], name=name, line=dict(color=c, width=1.5), row=1, col=1,
                        hovertemplate="corr %{y:+.2f}")
        fig.add_scatter(x=x, y=rb["beta"], name=name, line=dict(color=c, width=1.5), row=2, col=1, showlegend=False,
                        hovertemplate="beta %{y:+.2f}")
    for r in (1, 2):
        fig.add_hline(y=0, line=dict(color=fx.GREY, width=1), row=r, col=1)
    _base(fig, f"Is the hourly VIX link to {asset} DVOL getting stronger or weaker?", height=560)
    return _done(fig, "Within-session hourly changes only.", subplot_titles=True)
