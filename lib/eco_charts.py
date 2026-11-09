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
