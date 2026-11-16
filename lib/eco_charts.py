"""Figures for the two Ecoinometrics-inspired views added 2026-11-09.

Math lives in ``lib.ecoinometrics``. These builders only draw it, and they
go through ``fx_style.apply_theme(..., "light")`` and ``fx_style.finalize``
so axis text stays dark on a white card (on-screen and in a kaleido export).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib import ecoinometrics as eco
from lib import fx_style

# Sequential close scale: navy (low) -> blue -> amber -> red (high).
_PRICE_STOPS = (
    (0.00, (0x13, 0x31, 0x4F)),
    (0.45, (0x37, 0x90, 0xC7)),
    (0.72, (0xEF, 0x8A, 0x00)),
    (1.00, (0xC0, 0x39, 0x2B)),
)
PRICE_COLORSCALE = [
    [t, "#%02X%02X%02X" % rgb] for t, rgb in _PRICE_STOPS
]

# Discrete regime colours. Unclassified is a visible grey, not white,
# so a warmup bar still shows on the white export background.
REGIME_COLOURS = {
    "strong_inflow": fx_style.GREEN,
    "inflow": fx_style.BLUE,
    "neutral": fx_style.GREY,
    "outflow": fx_style.ORANGE,
    "strong_outflow": fx_style.RED,
    "unclassified": "#B7C0C8",
}
REGIME_LABELS = {
    "strong_inflow": "Strong inflow",
    "inflow": "Inflow",
    "neutral": "Neutral",
    "outflow": "Outflow",
    "strong_outflow": "Strong outflow",
    "unclassified": "Unclassified",
}

_AXIS = "#1F2A37"
_RATIO_NOTE = (
    "Deribit BTC-PERP daily closes. The average is the mean of 200 calendar "
    "days and is blank until every day in that window has a real close. "
    "Missing days are left blank. Colour is that day's close, scaled across "
    "the closes shown here."
)
_FLOW_NOTE = (
    "Farside daily total in US$ millions, times 1,000,000, divided by that "
    "day's Deribit BTC-PERP close. Colour is the MCM trailing-flow regime, "
    "an independent approximation, not the Ecoinometrics model."
)


def _sgt_axis(dates: pd.Series) -> pd.Series:
    """Naive UTC session clocks -> naive SGT wall time for the x axis.

    Dates here are UTC calendar dates (midnight). A Deribit daily bar is
    stamped at 08:00 UTC, which is 16:00 SGT on that same calendar date,
    so the date label matches both clocks.
    """
    converted = fx_style.to_local(pd.DatetimeIndex(pd.to_datetime(dates)))
    return pd.Series(converted, index=getattr(dates, "index", None))


def _price_hex(t: float) -> str:
    t = float(np.clip(t, 0.0, 1.0))
    for (t0, c0), (t1, c1) in zip(_PRICE_STOPS, _PRICE_STOPS[1:]):
        if t <= t1 or t1 == _PRICE_STOPS[-1][0]:
            span = t1 - t0
            f = 0.0 if span == 0 else (t - t0) / span
            f = float(np.clip(f, 0.0, 1.0))
            rgb = tuple(int(round(a + (b - a) * f)) for a, b in zip(c0, c1))
            return "#%02X%02X%02X" % rgb
    return "#C0392B"


def _from_first_valid(df: pd.DataFrame, col: str) -> pd.DataFrame:
    idx = np.flatnonzero(df[col].notna().to_numpy())
    if len(idx) == 0:
        return df.iloc[0:0]
    return df.iloc[int(idx[0]):].reset_index(drop=True)


def _theme(fig: go.Figure, note: str) -> go.Figure:
    """Light card, dark axes, shared header layout. Safe to export."""
    fig.update_layout(
        height=620,
        hovermode="closest",
        margin=dict(l=72, r=88, b=56),
    )
    fig.update_xaxes(title_text="Date (SGT)", tickformat="%d %b %Y")
    fx_style.apply_theme(fig, "light")
    # apply_theme sets the axis font; repeat the titles so they stay explicit.
    fig.update_xaxes(
        title_text="Date (SGT)",
        title_font=dict(color=_AXIS, size=13),
        tickfont=dict(color=_AXIS, size=12),
        color=_AXIS,
    )
    fig.update_yaxes(
        title_font=dict(color=_AXIS, size=13),
        tickfont=dict(color=_AXIS, size=12),
        color=_AXIS,
    )
    return fx_style.finalize(fig, note=note, legend_rows=2)


def figure_price_sma_ratio(df: pd.DataFrame) -> go.Figure:
    """Close / 200-calendar-day SMA, coloured by the BTC close.

    ``df`` is a ``price_sma_ratio`` frame, already sliced to the display
    window. Leading warmup rows are omitted from the axis. Interior rows
    with a null ratio are kept so the line breaks at a gap instead of
    bridging it. The colour scale is the min-to-max close in this frame.
    """
    fig = go.Figure()
    fig.update_layout(title="BTC close / 200-day average")
    fig.update_yaxes(title_text="Close / 200-day SMA", tickformat=".3f")
    view = _from_first_valid(df, "ratio") if df is not None and len(df) else df
    if view is None or len(view) == 0 or not view["ratio"].notna().any():
        fig.add_annotation(
            text="No 200-calendar-day ratio in this window",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color=_AXIS, size=14),
        )
        return _theme(fig, _RATIO_NOTE)

    x = _sgt_axis(view["date"])
    y = view["ratio"].to_numpy(dtype=float)
    close = view["close"].to_numpy(dtype=float)
    finite = np.isfinite(y) & np.isfinite(close)
    lo = float(np.min(close[finite]))
    hi = float(np.max(close[finite]))
    span = hi - lo
    n_bins = 12
    bins = np.zeros(len(y), dtype=int)
    if span <= 0:
        cmin, cmax = lo - 1.0, hi + 1.0
    else:
        raw = np.floor((close[finite] - lo) / span * n_bins)
        bins[np.flatnonzero(finite)] = np.clip(raw, 0, n_bins - 1).astype(int)
        cmin, cmax = lo, hi

    # One run per stretch of adjacent sessions in the same colour bin.
    # A new run starts on the previous point, so a colour change meets the
    # last segment instead of leaving a gap. A missing close, or a jump of
    # more than one calendar day, does not bridge.
    runs: list[tuple[int, list, list]] = []
    prev_i = None
    for i in range(len(y)):
        if not finite[i]:
            prev_i = None
            continue
        if prev_i is None:
            prev_i = i
            continue
        d0 = pd.Timestamp(x.iloc[prev_i]).normalize()
        d1 = pd.Timestamp(x.iloc[i]).normalize()
        if d1 - d0 != pd.Timedelta(days=1):
            prev_i = i
            continue
        b = int(bins[i])
        same = (
            runs and runs[-1][0] == b and runs[-1][1]
            and pd.Timestamp(runs[-1][1][-1]).normalize() == d0
        )
        if same:
            runs[-1][1].append(x.iloc[i])
            runs[-1][2].append(float(y[i]))
        else:
            runs.append((b, [x.iloc[prev_i], x.iloc[i]], [float(y[prev_i]), float(y[i])]))
        prev_i = i
    for b, rx, ry in runs:
        t = (b + 0.5) / n_bins if span > 0 else 0.5
        fig.add_trace(go.Scatter(
            x=rx, y=ry, mode="lines",
            line=dict(color=_price_hex(t), width=2.6),
            hoverinfo="skip", showlegend=False,
        ))

    pts = finite
    custom = np.column_stack([
        close[pts],
        view["sma"].to_numpy(dtype=float)[pts],
    ])
    fig.add_trace(go.Scatter(
        x=x.iloc[np.flatnonzero(pts)],
        y=y[pts],
        mode="markers",
        name="Close / 200-day SMA",
        showlegend=False,
        marker=dict(
            size=6,
            color=close[pts],
            colorscale=PRICE_COLORSCALE,
            cmin=cmin, cmax=cmax,
            colorbar=dict(
                title=dict(text="BTC close (USD)", font=dict(color=_AXIS, size=12)),
                tickfont=dict(color=_AXIS, size=11),
                tickformat="$,.0f",
            ),
            line=dict(width=0),
        ),
        customdata=custom,
        hovertemplate=(
            "%{x|%d %b %Y}<br>Ratio %{y:.3f}"
            "<br>Close %{customdata[0]:,.0f} USD"
            "<br>200-day average %{customdata[1]:,.0f} USD"
            "<extra></extra>"
        ),
    ))
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="#888888", line_width=1,
    )
    return _theme(fig, _RATIO_NOTE)


def figure_etf_flow_regime(df: pd.DataFrame) -> go.Figure:
    """Daily BTC ETF flow bars coloured by the MCM regime.

    ``df`` is an ``etf_flow_regime`` frame, already sliced for display.
    One overlay trace per regime so a quiet regime does not stretch the
    bar width of the sessions that do print. Bar width is fixed at 18 hours.
    """
    fig = go.Figure()
    fig.update_layout(title="US spot BTC ETF net flow", barmode="overlay")
    fig.update_yaxes(title_text="Net flow (BTC)", tickformat=",.0f")
    if df is None or len(df) == 0:
        fig.add_annotation(
            text="No aligned ETF flow in this window",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color=_AXIS, size=14),
        )
        return _theme(fig, _FLOW_NOTE)

    view = df.copy()
    view["regime"] = view["regime"].where(view["regime"].notna(), "unclassified")
    x_all = _sgt_axis(view["date"])
    # 18 hours, in milliseconds, so neighbouring daily bars do not merge
    # when a regime trace is sparse.
    bar_width = 18 * 3600 * 1000
    present = set(view["regime"])
    order = [r for r in eco.FLOW_REGIMES if r != "unclassified"]
    if "unclassified" in present:
        order.append("unclassified")
    for regime in order:
        mask = (view["regime"] == regime).to_numpy()
        if not mask.any() and regime == "unclassified":
            continue
        sub = view.loc[mask]
        xs = x_all.iloc[np.flatnonzero(mask)]
        sums = sub["sum_window"].to_numpy(dtype=float) if len(sub) else np.array([])
        scales = sub["scale_median"].to_numpy(dtype=float) if len(sub) else np.array([])
        labels = [REGIME_LABELS.get(regime, regime)] * len(sub)
        sum_txt = ["n/a" if not np.isfinite(v) else f"{v:,.1f}" for v in sums]
        scale_txt = ["n/a" if not np.isfinite(v) else f"{v:,.1f}" for v in scales]
        fig.add_trace(go.Bar(
            x=list(xs),
            y=sub["flow"].to_numpy(dtype=float) if len(sub) else [],
            name=REGIME_LABELS.get(regime, regime),
            marker_color=REGIME_COLOURS.get(regime, fx_style.GREY),
            marker_line_width=0,
            width=bar_width,
            customdata=list(zip(labels, sum_txt, scale_txt)),
            hovertemplate=(
                "%{x|%d %b %Y}<br>Daily flow %{y:,.1f} BTC"
                "<br>%{customdata[0]}"
                f"<br>{eco.FLOW_REGIME_WINDOW}-session sum %{{customdata[1]}} BTC"
                "<br>Scale median %{customdata[2]} BTC"
                "<extra></extra>"
            ),
        ))
    fig.add_hline(y=0, line_color="#888888", line_width=1)
    return _theme(fig, _FLOW_NOTE)


# ----------------------------------------------------------------------------
# Added 2026-11-16, inspired by the 9/11/13 Nov 2026 Ecoinometrics issues.
# These copy the newsletter look (lib/eco_style.style), like the original
# replicas on the page, rather than the white fx_style card above.
from lib import eco_style as es  # noqa: E402


def _empty_note(fig: go.Figure, text: str) -> go.Figure:
    fig.add_annotation(text=text, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(color=es.INK, size=15))
    return fig


def figure_nasdaq_sma(nq: pd.DataFrame, start=None, end=None) -> go.Figure:
    """Nasdaq Composite with its 200-trading-day SMA. ``nq`` is ``eco.sma_rows`` output."""
    fig = go.Figure()
    notes = ("Notes: FRED NASDAQCOM daily closes. The average is 200 trading days (FRED prints), "
             "while the Bitcoin panel above uses 200 daily bars including weekends. Source: FRED.")
    view = nq if start is None else nq[nq["date"] >= pd.Timestamp(start)]
    valid = view.dropna(subset=["sma"])
    if valid.empty:
        es.style(fig, "The Nasdaq vs Its 200-Day Average", "Nasdaq", "No Nasdaq data in this window.", notes, height=520)
        return _empty_note(fig, "No Nasdaq data")
    last = valid.iloc[-1]
    fig.add_trace(go.Scatter(x=view["date"], y=view["sma"], line=dict(color=es.ORANGE, width=3), name="200-day average"))
    fig.add_trace(go.Scatter(x=view["date"], y=view["value"], line=dict(color=es.INK, width=2), name="Nasdaq Composite",
                             hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra></extra>"))
    es.style(fig, "The Nasdaq vs Its 200-Day Average", "Nasdaq",
             f"The Nasdaq trades {last['gap']:+.2%} versus its 200-day average "
             f"({last['value']:,.0f} vs {last['sma']:,.0f}) as of {pd.Timestamp(last['date']):%d %b %Y}.",
             notes, height=520, yfmt=",.0f")
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.04, x=0.55))
    if start is not None:
        fig.update_xaxes(range=[pd.Timestamp(start), pd.Timestamp(end) if end is not None else view["date"].max()])
    return fig


def figure_rolling_flow_price(df: pd.DataFrame, window: int = 30) -> go.Figure:
    """30-session rolling ETF flow (BTC) as a line coloured by the BTC close."""
    fig = go.Figure()
    notes = (f"Notes: sum of the last {window} Farside sessions, each converted from US$ to BTC at that UTC "
             "day's Deribit close. Ecoinometrics uses the change in BTC held by the ETFs, so levels differ. "
             "Source: Farside, Deribit.")
    view = df.dropna(subset=["roll", "close"]) if df is not None and len(df) else pd.DataFrame()
    if view.empty:
        es.style(fig, "Bitcoin ETF Flows Over 30 Days", "Bitcoin", "Not enough aligned sessions yet.", notes)
        return _empty_note(fig, f"Needs {window} aligned sessions")
    x = view["date"].tolist()
    y = view["roll"].to_numpy(dtype=float)
    close = view["close"].to_numpy(dtype=float)
    lo, hi = float(close.min()), float(close.max())
    span = hi - lo
    n_bins = 12
    bins = np.zeros(len(y), dtype=int) if span <= 0 else np.clip(
        np.floor((close - lo) / span * n_bins), 0, n_bins - 1).astype(int)
    runs: list[tuple[int, list, list]] = []
    for i in range(1, len(y)):
        b = int(bins[i])
        if runs and runs[-1][0] == b:
            runs[-1][1].append(x[i]); runs[-1][2].append(float(y[i]))
        else:
            runs.append((b, [x[i - 1], x[i]], [float(y[i - 1]), float(y[i])]))
    for b, rx, ry in runs:
        fig.add_trace(go.Scatter(x=rx, y=ry, mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=_price_hex((b + 0.5) / n_bins if span > 0 else 0.5), width=3)))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", showlegend=False,
        marker=dict(size=5, color=close, colorscale=PRICE_COLORSCALE, cmin=lo, cmax=hi if span > 0 else lo + 1,
                    colorbar=dict(title=dict(text="BTC close", font=dict(color=es.INK, size=12)),
                                  tickfont=dict(color=es.INK, size=11), tickformat="$,.0f", thickness=12)),
        customdata=close,
        hovertemplate="%{x|%d %b %Y}<br>30-session flow %{y:,.0f} BTC<br>Close %{customdata:,.0f} USD<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=es.INK, line_width=1.5)
    last = view.iloc[-1]
    word = "inflow" if last["roll"] >= 0 else "outflow"
    es.style(fig, "Bitcoin ETF Flows Over 30 Days", "Bitcoin",
             f"Net {word} of {abs(last['roll']):,.0f} BTC over the last {window} sessions "
             f"to {pd.Timestamp(last['date']):%d %b %Y}. Line colour is the Bitcoin price.",
             notes, yfmt=",.0f")
    fig.update_yaxes(title_text=f"{window}-session net flow (BTC)")
    fig.update_layout(margin=dict(r=110))
    return fig


def figure_nasdaq_down_days(df: pd.DataFrame, stats: pd.DataFrame) -> go.Figure:
    """Box plots of BTC same-date returns per Nasdaq-100 down-day bucket."""
    fig = go.Figure()
    notes = ("Notes: Nasdaq-100 (FRED NASDAQ100) close-to-close return on its trading date D; each bucket includes its lower bound. "
             "BTC return is the daily bar stamped D over the bar stamped D−1 (Bitstamp 00:00 UTC bars to Jul 2018, "
             "Deribit BTC-PERP 08:00 UTC bars after); the 13:30-20:00 UTC Nasdaq session sits inside both. "
             "Since 2014. Source: FRED, Bitstamp, Deribit.")
    if df is None or len(df) == 0 or stats is None or stats.empty:
        es.style(fig, "Bitcoin on Nasdaq Down Days", "Bitcoin", "No overlapping data.", notes)
        return _empty_note(fig, "No overlapping data")
    order = stats["bucket"].tolist()
    colours = [es.RED, "#B3123F", es.ORANGE, es.YELLOW, es.TEAL]
    cmap = {lab: colours[i % len(colours)] for i, lab in enumerate(eco.NASDAQ_BUCKET_LABELS)}
    for lab in order:
        g = df[df["bucket"] == lab]
        fig.add_trace(go.Box(
            x=[lab] * len(g), y=g["btc_ret"], name=lab, marker_color=cmap[lab],
            boxpoints="all" if len(g) < 10 else "outliers", pointpos=0 if len(g) < 10 else None,
            line=dict(width=2), fillcolor=cmap[lab], opacity=0.75,
            customdata=np.column_stack([g["date"].dt.strftime("%d %b %Y"), g["ndx_ret"]]),
            hovertemplate="%{customdata[0]}<br>Nasdaq-100 %{customdata[1]:.1f}%<br>BTC %{y:.1f}%<extra></extra>",
        ))
    def _pct(v):
        return f"{0.0 if abs(v) < 0.005 else v:+.2f}%"
    ticktext = [f"<b>{r['bucket']}</b><br>median {_pct(r['median'])}<br>n = {int(r['n'])}"
                for _, r in stats.iterrows()]
    worst = stats.iloc[0] if len(stats) else None
    mild = stats[stats["bucket"] == "−2% to 0%"]
    sub = "Bitcoin's median daily return by size of the Nasdaq-100's down day"
    if worst is not None and len(mild):
        sub += f" ({_pct(worst['median'])} in the {worst['bucket']} bucket vs {_pct(mild.iloc[0]['median'])} on −2% to 0% days)."
    else:
        sub += "."
    es.style(fig, "Bitcoin on Nasdaq Down Days", "Bitcoin", sub, notes)
    fig.update_yaxes(ticksuffix="%", title_text="BTC return on the same date")
    fig.update_xaxes(title_text="", categoryorder="array", categoryarray=order,
                     tickmode="array", tickvals=order, ticktext=ticktext)
    fig.update_layout(margin=dict(b=175))
    return fig


def figure_drawdowns(dd: pd.DataFrame, trend: tuple[float, float] | None, min_depth: float) -> go.Figure:
    """Duration (log x) vs max drawdown for each BTC drawdown; ongoing one highlighted."""
    fig = go.Figure()
    notes = (f"Notes: daily closes (Bitstamp to Jul 2018, Deribit BTC-PERP after). A drawdown runs from an all-time-high "
             f"close to the first close back at or above it; only drawdowns of {min_depth:.0%} or more that were still "
             "open on or after 1 Jan 2014. Trend: least squares of depth on log duration, finished drawdowns only. "
             "Source: Bitstamp, Deribit.")
    if dd is None or dd.empty:
        es.style(fig, "Bitcoin Drawdowns: Duration vs Depth", "Bitcoin", "No drawdowns at this threshold.", notes)
        return _empty_note(fig, "No drawdowns")
    done = dd[~dd["ongoing"]]
    cur = dd[dd["ongoing"]]
    fig.add_trace(go.Scatter(
        x=done["days"], y=done["depth"] * 100, mode="markers+text", name="Finished",
        text=pd.to_datetime(done["peak_date"]).dt.strftime("%b %Y"), textposition="top center",
        textfont=dict(family="Arial", size=11, color=es.INK),
        marker=dict(size=13, color=es.TEAL, line=dict(color=es.INK, width=0.6)),
        customdata=np.column_stack([pd.to_datetime(done["peak_date"]).dt.strftime("%d %b %Y"),
                                    pd.to_datetime(done["trough_date"]).dt.strftime("%d %b %Y"),
                                    pd.to_datetime(done["recovered"]).dt.strftime("%d %b %Y")]),
        hovertemplate="Peak %{customdata[0]}<br>Trough %{customdata[1]}<br>Recovered %{customdata[2]}"
                      "<br>%{x} days, %{y:.1f}%<extra></extra>",
    ))
    if trend is not None:
        a, b = trend
        xs = np.logspace(np.log10(max(1, dd["days"].min() * 0.8)), np.log10(dd["days"].max() * 1.3), 50)
        fig.add_trace(go.Scatter(x=xs, y=np.clip(a + b * np.log10(xs), -100, 0), mode="lines", name="Trend",
                                 line=dict(color=es.MUTED, width=2, dash="dash"), hoverinfo="skip"))
    sub = f"{len(done)} finished drawdowns of {min_depth:.0%} or more since 2014."
    if len(cur):
        c = cur.iloc[0]
        fig.add_trace(go.Scatter(
            x=[c["days"]], y=[c["depth"] * 100], mode="markers", name="Current (ongoing)",
            marker=dict(size=20, color=es.ORANGE, symbol="star", line=dict(color=es.INK, width=1)),
            hovertemplate=f"Current: peak {pd.Timestamp(c['peak_date']):%d %b %Y} ({c['peak']:,.0f})"
                          f"<br>Low {pd.Timestamp(c['trough_date']):%d %b %Y} ({c['trough']:,.0f})"
                          "<br>%{x} days so far, max %{y:.1f}%<extra></extra>",
        ))
        es.note(fig, np.log10(c["days"]), c["depth"] * 100,
                f"Current: {int(c['days'])} days,<br>{c['depth']:.0%} max", color=es.ORANGE, ax=-70, ay=40,
                xref="x", yref="y")
        sub += (f" The current one is {int(c['days'])} days old with a {c['depth']:.0%} maximum drawdown "
                f"(peak {pd.Timestamp(c['peak_date']):%d %b %Y}).")
        if trend is not None:
            sub += f" The trend line implies {trend[0] + trend[1] * np.log10(c['days']):.0f}% at that duration."
    es.style(fig, "Bitcoin Drawdowns: Duration vs Depth", "Bitcoin", sub, notes)
    fig.update_xaxes(type="log", title_text="Duration, peak to recovery (calendar days, log scale)",
                     tickvals=[5, 10, 30, 60, 100, 200, 365, 730, 1000, 1500],
                     ticktext=["5", "10", "30", "60", "100", "200", "1y", "2y", "1,000", "1,500"])
    fig.update_yaxes(ticksuffix="%", title_text="Maximum drawdown from peak", range=[-100, 0])
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.04, x=0.55))
    return fig


def figure_strategy_funding(f) -> go.Figure:
    """Horizontal bars: where Strategy's money came from and went in its latest weekly 8-K."""
    period = ""
    if f.period_start is not None and f.period_end is not None:
        period = f"{f.period_start:%d %b} to {f.period_end:%d %b %Y}"
    notes = (f"Notes: parsed from Strategy's 8-K filed {f.filed} (SEC EDGAR, accession {f.accession}). "
             "Sources are ATM net proceeds. Uses come from the 8-K footnotes, BTC table and repurchase table; "
             "a buyback can be paid from existing USD Cash, so uses need not equal sources. Source: SEC EDGAR.")
    rows = [("Raised", k, v) for k, v in sorted(f.sources.items(), key=lambda kv: kv[1])]
    rows += [("Used", k, v) for k, v in sorted(f.uses.items(), key=lambda kv: kv[1])]
    fig = go.Figure()
    if not rows:
        es.style(fig, "Strategy's Weekly Funding", "Strategy", "Nothing raised or spent this week.", notes, height=520)
        return _empty_note(fig, "No amounts in this 8-K")
    colour = {"Raised": es.INK}
    use_col = {"BTC purchases": es.ORANGE, "USD Cash": es.TEAL, "USD Reserve": "#1B4F72"}
    labels = [f"{k}" for _, k, _ in rows]
    cols = [colour.get(g) or use_col.get(k, es.RED) for g, k, _ in rows]
    fig.add_trace(go.Bar(
        y=[[g for g, _, _ in rows], labels], x=[v for _, _, v in rows], orientation="h",
        marker_color=cols, text=[f"${v:,.1f}M" for _, _, v in rows], textposition="outside",
        textfont=dict(family=es.SERIF, size=13, color=es.INK),
        hovertemplate="%{y}: $%{x:,.1f}M<extra></extra>",
    ))
    raised = f.total_raised
    parts = []
    for k in ("BTC purchases",) + tuple(k for k in f.uses if k.endswith("buyback")):
        if k in f.uses and raised:
            parts.append(f"{f.uses[k] / raised:.0%} to {k.replace('BTC purchases', 'bitcoin')}")
    # "&#36;" not "$": two dollar signs in one string switch Plotly to LaTeX.
    sub = f"Week {period}: raised &#36;{raised:,.1f}M" + (f" ({', '.join(parts)} of that)" if parts else "") + "."
    if f.btc_bought:
        sub += f" Bought {f.btc_bought:,.0f} BTC at &#36;{f.btc_avg_price:,.0f}; holds {f.btc_holdings:,.0f} BTC."
    es.style(fig, "Strategy's Weekly Funding", "Strategy", sub, notes, height=560)
    fig.update_xaxes(title_text="US$ millions", tickprefix="$", range=[0, max(v for _, _, v in rows) * 1.25])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", zeroline=False)
    fig.update_layout(margin=dict(l=190))
    return fig
