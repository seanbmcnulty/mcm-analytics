"""Figures for the 10x Research chart collection (maths in ``lib.tenx``).

Dark card with white / violet lines, in the spirit of the 10x exhibits, but
branded ours ("MCM Analytics") and drawn from our own data.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib import tenx

BG, INK, MUTED, GRID = "#000000", "#FFFFFF", "#A6A6A6", "#2A2A2A"
VIOLET, GREY, LILAC = "#7B61FF", "#BFBFBF", "#8E7CC3"
FONT = "Poppins, 'Helvetica Neue', Arial, sans-serif"


def style(fig: go.Figure, title: str, subtitle: str = "", notes: str = "", height: int = 620) -> go.Figure:
    sub = "<br>".join(textwrap.wrap(subtitle, 110)) if subtitle else ""
    top = 60 + 22 * (sub.count("<br>") + 1 if sub else 0)
    fig.update_layout(
        template=None, paper_bgcolor=BG, plot_bgcolor=BG, height=height,
        font=dict(family=FONT, color=INK, size=14),
        title=dict(text=f"<b>{title}</b>", x=0, xref="paper", xanchor="left", y=0.99, yanchor="top",
                   font=dict(size=22, color=INK)),
        margin=dict(l=70, r=70, t=top + 30, b=110),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", bgcolor="rgba(0,0,0,0)",
                    font=dict(color=INK, size=13)),
        hovermode="x unified", hoverlabel=dict(bgcolor="#111111", font_color=INK),
        xaxis=dict(showgrid=False, linecolor=MUTED, tickfont=dict(color=INK), tickformat="%b-%y"),
        yaxis=dict(showgrid=False, linecolor=MUTED, tickfont=dict(color=INK), zeroline=False),
    )
    if sub:
        fig.add_annotation(text=sub, xref="paper", yref="paper", x=0, y=1.0, xanchor="left", yanchor="bottom",
                           yshift=34, showarrow=False, align="left", font=dict(size=15, color=INK))
    if notes:
        fig.add_annotation(text="<br>".join(textwrap.wrap(notes, 120)), xref="paper", yref="paper", x=0, y=-0.12,
                           xanchor="left", yanchor="top", showarrow=False, align="left",
                           font=dict(size=11, color=MUTED))
    fig.add_annotation(text="MCM Analytics", xref="paper", yref="paper", x=1.0, y=-0.12, xanchor="right",
                       yanchor="top", showarrow=False, font=dict(size=16, color=MUTED))
    return fig


def _empty(title: str, text: str) -> go.Figure:
    fig = go.Figure()
    style(fig, title)
    fig.add_annotation(text=text, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                       font=dict(color=INK, size=15))
    return fig


def figure_btc_rsi(df: pd.DataFrame, stats: dict, source: str) -> go.Figure:
    title = "Bitcoin daily RSI"
    if df is None or df.empty:
        return _empty(title, "No Bitcoin closes available")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Bitcoin (LHS)", line=dict(color=INK, width=2.2),
                             hovertemplate="$%{y:,.0f}"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name=f"RSI ({stats['n']})", line=dict(color=VIOLET, width=1.6),
                             hovertemplate="%{y:.1f}"), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df["ma"], name=f"MA ({stats['ma_n']}d)", line=dict(color=GREY, width=1.2),
                             hovertemplate="%{y:.1f}"), secondary_y=True)
    for lvl in (stats["lo"], stats["hi"]):
        fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[lvl, lvl], mode="lines", name=str(lvl),
                                 line=dict(color=GREY, width=1.5), hoverinfo="skip", showlegend=True),
                      secondary_y=True)
    sig = df[df["signal"]]
    if len(sig):
        fig.add_trace(go.Scatter(x=sig.index, y=[8] * len(sig), mode="markers", name="Buy reversal",
                                 marker=dict(symbol="arrow-up", size=22, color=VIOLET, line=dict(color=INK, width=1)),
                                 customdata=sig["rsi"], hovertemplate="RSI %{customdata:.1f}"), secondary_y=True)
    st = stats
    sub = (f"Level {st['level']:,.0f} · RSI {st['rsi']:.1f} · Avg. {st['ma']:.1f} · Dif. {st['diff']:+.1f} · "
           f"View: {st['view']}  (as of {st['date']:%d %b %Y})")
    notes = (f"RSI: Wilder {st['n']}-day on daily closes; MA: {st['ma_n']}-day average of the RSI. Arrow = buy reversal: "
             f"RSI at or below {st['lo']:g} crosses up through its MA (one per 20 days). Source: {source}.")
    style(fig, title, sub, notes)
    fig.update_yaxes(tickformat=",.0f", secondary_y=False, showgrid=False, tickfont=dict(color=INK))
    fig.update_yaxes(range=[5, 95], tickmode="linear", tick0=10, dtick=10, secondary_y=True,
                     showgrid=False, tickfont=dict(color=INK))
    return fig


def figure_gold_channel(df: pd.DataFrame, stats: dict, source: str) -> go.Figure:
    title = "Gold (GC)"
    if df is None or df.empty:
        return _empty(title, "No gold closes available")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df["gold"], name="Gold", line=dict(color=INK, width=2.4),
                             hovertemplate="$%{y:,.0f}"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_slow"], name=f"{stats['ma_slow_n']}d MA",
                             line=dict(color=VIOLET, width=2.4), hovertemplate="$%{y:,.0f}"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_fast"], name=f"{stats['ma_fast_n']}d MA",
                             line=dict(color=GREY, width=2), hovertemplate="$%{y:,.0f}"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["relative"], name="Relative to BTC (RHS)",
                             line=dict(color=LILAC, width=1.3), hovertemplate="%{y:+.1f}%"), secondary_y=True)
    ch = stats.get("channel")
    if ch is not None:
        x0 = max(ch.start, df.index[0])
        x1 = df.index[-1] + pd.Timedelta(days=10)
        for which in ("upper", "lower"):
            fig.add_trace(go.Scatter(x=[x0, x1], y=[ch.at(x0, which), ch.at(x1, which)], mode="lines",
                                     name="Channel" if which == "upper" else None, showlegend=which == "upper",
                                     line=dict(color=GREY, width=1.5), hoverinfo="skip"), secondary_y=False)
    st = stats
    sub = f"Gold {st['gold']:,.0f} · {st['ma_fast_n']}d MA {st['ma_fast']:,.0f} · {st['ma_slow_n']}d MA {st['ma_slow']:,.0f}"
    if ch is not None:
        sub += (f" · channel {st['chan_lower']:,.0f}–{st['chan_upper']:,.0f} "
                f"({st['chan_pos'] * 100:.0f}% up from the lower line)")
    sub += f" · gold vs BTC 1y: {st['relative']:+.0f} pp  (as of {st['date']:%d %b %Y})"
    notes = ("Channel: least-squares trend of the daily close from the highest close of the last 400 days, with "
             "parallel lines through the largest move above and below it. Relative to BTC: gold's trailing "
             f"{st['rel_days']}-day return minus Bitcoin's, percentage points. Source: {source}.")
    style(fig, title, sub, notes)
    fig.update_yaxes(tickformat=",.0f", secondary_y=False, showgrid=False, tickfont=dict(color=INK))
    fig.update_yaxes(ticksuffix="%", tickmode="auto", nticks=10, secondary_y=True, showgrid=False,
                     tickfont=dict(color=INK))
    return fig


def figure_btc_coin(df: pd.DataFrame, stats: dict, source: str) -> go.Figure:
    title = "Bitcoin vs. Coinbase"
    if df is None or df.empty:
        return _empty(title, "No Bitcoin / Coinbase closes available")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df["btc"], name="Bitcoin (LHS)", line=dict(color=INK, width=2),
                             hovertemplate="$%{y:,.0f}"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["other"], name="Coinbase (RHS)", line=dict(color=VIOLET, width=2),
                             hovertemplate="$%{y:,.2f}"), secondary_y=True)
    st = stats
    sub = f"BTC {st['btc']:,.0f} · COIN ${st['other']:,.2f}"
    if "btc_chg" in st:
        sub += f" · last {st['chg_days']}d: BTC {st['btc_chg']:+.0f}%, COIN {st['other_chg']:+.0f}%"
    if np.isfinite(st.get("corr", np.nan)):
        sub += f" · {st['corr_days']}d return correlation {st['corr']:.2f}"
    sub += f"  (as of {st['date']:%d %b %Y})"
    notes = ("BTC shown on Coinbase trading days only (no weekend points). Correlation of daily log returns. "
             f"Source: {source}.")
    style(fig, title, sub, notes)
    fig.update_yaxes(tickformat=",.0f", secondary_y=False, showgrid=False, tickfont=dict(color=INK))
    fig.update_yaxes(tickformat=",.0f", tickmode="auto", nticks=10, secondary_y=True, showgrid=False,
                     tickfont=dict(color=INK))
    return fig


def _source(prices: "tenx.Prices", cols) -> str:
    parts = []
    for c in cols:
        how = prices.source.get(c, "none")
        how = {"live": "live", "csv": "committed CSV of Yahoo closes", "live+csv": "live + committed CSV",
               "none": "unavailable"}[how]
        parts.append(f"{tenx.SYMBOL_LABELS[c]} [{how}]")
    return "; ".join(parts)


def build_btc_rsi(prices):
    df, st = tenx.rsi_frame(prices.df["BTC"], start="2024-07-01")
    return figure_btc_rsi(df, st, _source(prices, ["BTC"])), st


def build_gold_channel(prices):
    df, st = tenx.gold_frame(prices.df["GOLD"], prices.df["BTC"], start="2025-10-01")
    return figure_gold_channel(df, st, _source(prices, ["GOLD", "BTC"])), st


def build_btc_coin(prices):
    df, st = tenx.pair_frame(prices.df["BTC"], prices.df["COIN"], start="2023-12-01")
    return figure_btc_coin(df, st, _source(prices, ["BTC", "COIN"])), st


BUILDERS = {"btc_rsi": build_btc_rsi, "gold_channel": build_gold_channel, "btc_coin": build_btc_coin}
