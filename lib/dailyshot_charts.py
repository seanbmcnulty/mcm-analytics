"""Figures for The Daily Shot chart collection (maths in ``lib.dailyshot``).

Clean white card in the spirit of the newsletter's charts, branded ours
("MCM Analytics") and drawn from our own public data. Every builder takes a
``Ctx`` (Yahoo prices, FRED series, Treasury curve) and returns (figure, stats).
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
NAVY, RED, TEAL, GOLD, GREY = "#1F3A93", "#C0392B", "#16A085", "#D4A017", "#8C8C8C"
FONT = "'Helvetica Neue', Arial, sans-serif"
SRC = {"live": "live", "csv": "committed CSV", "live+csv": "live + committed CSV", "none": "unavailable"}


@dataclass
class Ctx:
    prices: "ds.Prices"
    fred: "ds.Fred"
    yields: pd.DataFrame | None        # date column + 2Y / 10Y ..., percent
    yields_source: str = ""


def style(fig: go.Figure, title: str, subtitle: str = "", notes: str = "", height: int = 560) -> go.Figure:
    sub = "<br>".join(textwrap.wrap(subtitle, 120)) if subtitle else ""
    top = 60 + 20 * (sub.count("<br>") + 1 if sub else 0)
    fig.update_layout(
        template="plotly_white", paper_bgcolor=BG, plot_bgcolor=BG, height=height,
        font=dict(family=FONT, color=INK, size=13),
        title=dict(text=f"<b>{title}</b>", x=0, xref="paper", xanchor="left", y=0.99, yanchor="top",
                   font=dict(size=20, color=INK)),
        margin=dict(l=70, r=70, t=top + 30, b=100),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=MUTED)
    if sub:
        fig.add_annotation(text=sub, xref="paper", yref="paper", x=0, y=1.0, xanchor="left", yanchor="bottom",
                           yshift=30, showarrow=False, align="left", font=dict(size=14, color=MUTED))
    if notes:
        fig.add_annotation(text="<br>".join(textwrap.wrap(notes, 130)), xref="paper", yref="paper", x=0,
                           y=-0.13, xanchor="left", yanchor="top", showarrow=False, align="left",
                           font=dict(size=11, color=MUTED))
    fig.add_annotation(text="MCM Analytics", xref="paper", yref="paper", x=1.0, y=-0.13, xanchor="right",
                       yanchor="top", showarrow=False, font=dict(size=14, color=MUTED))
    return fig


def _src(ctx: Ctx, cols=(), fred=(), yields=False) -> str:
    parts = []
    if cols:
        how = sorted({SRC[ctx.prices.source.get(c, "none")] for c in cols})
        parts.append(f"Yahoo Finance daily closes ({', '.join(how)})")
    if fred:
        how = sorted({SRC[ctx.fred.source.get(c, "none")] for c in fred})
        parts.append(f"FRED {', '.join(fred)} ({', '.join(how)})")
    if yields:
        parts.append(ctx.yields_source or "US Treasury par yields")
    return "Source: " + "; ".join(parts) + "."


WEEKENDS = [dict(bounds=["sat", "mon"])]


def _px(ctx: Ctx, cols, start=None) -> pd.DataFrame:
    df = ctx.prices.df[list(cols)]
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    return df.dropna(how="all")


def _yld(ctx: Ctx, col: str, start=None) -> pd.Series:
    if ctx.yields is None or ctx.yields.empty or col not in ctx.yields:
        return pd.Series(dtype=float)
    s = ctx.yields.set_index("date")[col].dropna()
    s.index = pd.to_datetime(s.index)
    return s[s.index >= pd.Timestamp(start)] if start is not None else s


# ------------------------------------------------------------------ builders
def build_nvda_rebound(ctx: Ctx):
    df = _px(ctx, ["NVDA", "SPX"], "2026-09-01").dropna()
    rb = ds.rebase(df)
    last = df.pct_change().iloc[-1] * 100
    stats = {"last_date": df.index[-1], "nvda_day": last["NVDA"], "spx_day": last["SPX"],
             "nvda_since": rb["NVDA"].iloc[-1] - 100, "spx_since": rb["SPX"].iloc[-1] - 100}
    fig = go.Figure()
    fig.add_scatter(x=rb.index, y=rb["NVDA"], name="Nvidia", line=dict(color=TEAL, width=2.5))
    fig.add_scatter(x=rb.index, y=rb["SPX"], name="S&P 500", line=dict(color=NAVY, width=2.5))
    fig.add_hline(y=100, line=dict(color=GREY, dash="dot"))
    style(fig, "Nvidia and the S&P 500",
          f"Rebased to 100 on 1 Sep 2026. Latest session ({stats['last_date']:%d %b}): Nvidia "
          f"{stats['nvda_day']:+.1f}%, S&P 500 {stats['spx_day']:+.1f}%.", _src(ctx, ["NVDA", "SPX"]))
    fig.update_xaxes(tickformat="%d %b")
    fig.update_xaxes(rangebreaks=WEEKENDS)
    return fig, stats


def build_nasdaq_ytd(ctx: Ctx):
    year = ctx.prices.df["IXIC"].dropna().index[-1].year
    a, b = ds.ytd(ctx.prices.df["IXIC"], year), ds.ytd(ctx.prices.df["NDX"], year)
    stats = {"year": year, "ixic": a.iloc[-1], "ndx": b.iloc[-1], "ixic_peak": a.max(),
             "ixic_peak_date": a.idxmax()}
    fig = go.Figure()
    fig.add_scatter(x=a.index, y=a, name="Nasdaq Composite", line=dict(color=NAVY, width=2.5))
    fig.add_scatter(x=b.index, y=b, name="Nasdaq 100", line=dict(color=TEAL, width=2))
    fig.add_hline(y=0, line=dict(color=RED, width=1.5))
    style(fig, f"Nasdaq year-to-date return, {year} (%)",
          f"Composite {stats['ixic']:+.1f}% YTD, from a {stats['ixic_peak']:+.1f}% peak on "
          f"{stats['ixic_peak_date']:%d %b}. Nasdaq 100 {stats['ndx']:+.1f}%.", _src(ctx, ["IXIC", "NDX"]))
    fig.update_xaxes(tickformat="%b")
    fig.update_yaxes(ticksuffix="%")
    return fig, stats


def build_umich(ctx: Ctx):
    s = ctx.fred.df["UMCSENT"].dropna()
    stats = {"last": s.iloc[-1], "last_date": s.index[-1], "min": s.min(), "min_date": s.idxmin()}
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s, name="UMich sentiment", line=dict(color=NAVY, width=2), showlegend=False)
    fig.add_scatter(x=[s.index[-1]], y=[s.iloc[-1]], mode="markers+text", marker=dict(color=RED, size=9),
                    text=[f"{s.iloc[-1]:.1f}"], textposition="middle right", showlegend=False)
    style(fig, "University of Michigan consumer sentiment",
          f"{stats['last']:.1f} in {stats['last_date']:%b %Y} (latest on FRED); record low "
          f"{stats['min']:.1f} in {stats['min_date']:%b %Y}.", _src(ctx, fred=["UMCSENT"]))
    fig.update_xaxes(tickformat="%Y")
    return fig, stats


def build_anthropic_ipo(ctx: Ctx):
    ant = ctx.prices.df["ANT"].dropna()
    first = ant.index[0]
    base = ctx.prices.df["SPX"].dropna()
    base_day = base.index[base.index < first][-1]
    df = _px(ctx, ["ANT", "SPX", "NDX"], base_day)
    idx = ds.rebase(df[["SPX", "NDX"]], base_day)
    stats = {"ipo_date": first, "ant_first": ant.iloc[0], "ant_last": ant.iloc[-1],
             "ant_chg": (ant.iloc[-1] / ant.iloc[0] - 1) * 100, "spx": idx["SPX"].iloc[-1] - 100,
             "ndx": idx["NDX"].iloc[-1] - 100, "last_date": ant.index[-1]}
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=ant.index, y=ant, name="Anthropic (ANT), US&#36;", mode="lines+markers",
                    line=dict(color=GOLD, width=3), secondary_y=True)
    fig.add_scatter(x=idx.index, y=idx["SPX"], name="S&P 500 (rebased)", line=dict(color=NAVY, width=2))
    fig.add_scatter(x=idx.index, y=idx["NDX"], name="Nasdaq 100 (rebased)", line=dict(color=TEAL, width=2))
    fig.add_vline(x=first, line=dict(color=GREY, dash="dot"))
    style(fig, "Anthropic since its debut vs the market",
          f"ANT first close {stats['ipo_date']:%d %b} US&#36;{stats['ant_first']:.2f}, now US&#36;{stats['ant_last']:.2f} "
          f"({stats['ant_chg']:+.1f}%). Since the day before the listing: S&P 500 {stats['spx']:+.1f}%, "
          f"Nasdaq 100 {stats['ndx']:+.1f}%.", _src(ctx, ["ANT", "SPX", "NDX"]))
    fig.update_yaxes(title_text="Index, 100 = day before listing", secondary_y=False)
    fig.update_yaxes(title_text="ANT, US&#36;", showgrid=False, secondary_y=True)
    fig.update_xaxes(tickformat="%d %b")
    fig.update_xaxes(rangebreaks=WEEKENDS)
    return fig, stats


def build_defensive_rotation(ctx: Ctx):
    d = ds.defensive_ratio(ctx.prices.df)
    chg = d["chg"].dropna()
    hs = ds.highest_since(chg, "2026-10-01", "2026-11-14")   # into the issue (session before 17 Nov)
    stats = {"last": chg.iloc[-1], "last_date": chg.index[-1], **hs}
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.06)
    fig.add_scatter(x=d.index, y=d["ratio"], name="Defensives / cyclicals", line=dict(color=NAVY, width=1.8),
                    row=1, col=1)
    fig.add_scatter(x=chg.index, y=chg, name="1-month change, %", line=dict(color=TEAL, width=1.2),
                    row=2, col=1)
    fig.add_hline(y=0, line=dict(color=GREY, width=1), row=2, col=1)
    style(fig, "Defensive vs cyclical sectors",
          f"Peak 21-session change into the issue: {stats['peak']:+.1f}% on {stats['peak_date']:%d %b %Y}, "
          + (f"the biggest since {stats['since']:%b %Y}" if stats["since"] is not None else "a record")
          + f". Now {stats['last']:+.1f}% ({stats['last_date']:%d %b}).",
          "Equal-weight XLP, XLU, XLV over XLK, XLY, XLF, XLI, price only (Yahoo closes, dividends "
          "excluded). " + _src(ctx, ds.DEFENSIVES + ds.CYCLICALS), height=640)
    fig.update_xaxes(tickformat="%Y")
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    return fig, stats


def build_cpi_2y(ctx: Ctx):
    head, core = ds.yoy(ctx.fred.df["CPIAUCSL"]), ds.yoy(ctx.fred.df["CPILFESL"])
    head, core = head[head.index >= "2021-01-01"], core[core.index >= "2021-01-01"]
    two = _yld(ctx, "2Y")
    stats = {"cpi": head.iloc[-1], "core": core.iloc[-1], "cpi_date": head.index[-1],
             "two": two.iloc[-1] if len(two) else np.nan, "two_date": two.index[-1] if len(two) else None}
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=head.index, y=head, name="CPI, % y/y", line=dict(color=NAVY, width=2.5))
    fig.add_scatter(x=core.index, y=core, name="Core CPI, % y/y", line=dict(color=TEAL, width=2.5))
    if len(two):
        fig.add_scatter(x=two.index, y=two, name="2-year Treasury yield, % (right)",
                        line=dict(color=RED, width=1.5), secondary_y=True)
    two_txt = (f"; 2-year yield {stats['two']:.2f}% ({stats['two_date']:%d %b})" if len(two)
               else "; 2-year yield unavailable")
    style(fig, "US inflation and the 2-year Treasury yield",
          f"CPI {stats['cpi']:.1f}% y/y, core {stats['core']:.1f}% in {stats['cpi_date']:%b %Y}{two_txt}.",
          _src(ctx, fred=["CPIAUCSL", "CPILFESL"], yields=True))
    fig.update_yaxes(ticksuffix="%", secondary_y=False)
    fig.update_yaxes(ticksuffix="%", showgrid=False, secondary_y=True)
    fig.update_xaxes(tickformat="%b-%y")
    return fig, stats


def build_dow_midterm(ctx: Ctx):
    r = ds.post_event_returns(ctx.prices.df["DJI"], ds.MIDTERMS)
    stats = {"rows": len(r), "latest": r["pct"].iloc[-1] if len(r) else np.nan,
             "best_prior": r["pct"].iloc[:-1].max() if len(r) > 1 else np.nan}
    colors = [GOLD if i == len(r) - 1 else NAVY for i in range(len(r))]
    fig = go.Figure(go.Bar(x=[f"{d:%Y}" for d in r["election"]], y=r["pct"], marker_color=colors,
                           text=[f"{v:+.2f}%" for v in r["pct"]], textposition="outside",
                           customdata=[f"{d:%d %b %Y}" for d in r["session"]],
                           hovertemplate="%{x}: %{y:+.2f}% on %{customdata}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=GREY))
    style(fig, "The Dow on the session after each midterm election",
          f"2026: {stats['latest']:+.2f}%; best earlier session in this sample {stats['best_prior']:+.2f}%.",
          "Close vs prior close on the first session after election day. " + _src(ctx, ["DJI"]))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(ticksuffix="%")
    return fig, stats


def build_sector_week(ctx: Ctx):
    r = ds.period_returns(ctx.prices.df[list(ds.SECTORS)], "2026-10-23", "2026-10-30")
    labels = [ds.SYMBOL_LABELS[c] for c in r.index]
    colors = [TEAL if c in ds.DEFENSIVES else (RED if v < 0 else NAVY) for c, v in r.items()]
    stats = {"best": r.idxmax(), "worst": r.idxmin(), "xlk": r.get("XLK", np.nan)}
    fig = go.Figure(go.Bar(x=r.values, y=labels, orientation="h", marker_color=colors,
                           text=[f"{v:+.1f}%" for v in r.values], textposition="outside"))
    fig.add_vline(x=0, line=dict(color=GREY))
    style(fig, "S&P 500 sectors, week of 26 to 30 Oct 2026",
          f"Best {ds.SYMBOL_LABELS[stats['best']]}, worst {ds.SYMBOL_LABELS[stats['worst']]}; defensives in "
          f"teal. Technology {stats['xlk']:+.1f}%.",
          "SPDR sector ETFs, 23 Oct to 30 Oct 2026 closes, price only. " + _src(ctx, ds.SECTORS))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(ticksuffix="%")
    return fig, stats


def build_gdp(ctx: Ctx):
    s = ctx.fred.df["A191RL1Q225SBEA"].dropna()
    s = s[s.index >= "2019-01-01"]
    stats = {"last": s.iloc[-1], "last_q": s.index[-1]}
    q = [f"{d.year} Q{(d.month - 1) // 3 + 1}" for d in s.index]
    fig = go.Figure(go.Bar(x=q, y=s, marker_color=[RED if v < 0 else NAVY for v in s],
                           text=[f"{v:.1f}" for v in s], textposition="outside"))
    fig.add_hline(y=0, line=dict(color=GREY))
    style(fig, "US real GDP growth (% q/q, annualised)",
          f"{q[-1]}: {stats['last']:.1f}%.", _src(ctx, fred=["A191RL1Q225SBEA"]))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(ticksuffix="%", range=[max(-10, s.min() - 2), min(15, s.max() + 3)])
    return fig, stats


def build_brazil(ctx: Ctx):
    df = _px(ctx, ["BVSP", "EWZ", "USDBRL"], "2026-10-01").copy()
    df["BRL"] = 1 / df["USDBRL"]
    base_day = pd.Timestamp("2026-10-23")   # last session before the 25 Oct run-off
    rb = ds.rebase(df[["BVSP", "EWZ", "BRL"]], base_day)
    at = rb[rb.index <= pd.Timestamp("2026-10-27")].ffill().iloc[-1]   # two sessions after the vote
    stats = {k: at[k] - 100 for k in ("BVSP", "EWZ", "BRL")}
    stats.update({f"{k}_now": rb[k].dropna().iloc[-1] - 100 for k in ("BVSP", "EWZ", "BRL")})
    fig = go.Figure()
    for k, name, col in (("BVSP", "Ibovespa", NAVY), ("EWZ", "EWZ (USD)", TEAL), ("BRL", "Real vs USD", GOLD)):
        s = rb[k].dropna()
        fig.add_scatter(x=s.index, y=s, name=name, line=dict(color=col, width=2.2))
    fig.add_hline(y=100, line=dict(color=GREY, dash="dot"))
    fig.add_vline(x=pd.Timestamp("2026-10-25"), line=dict(color=RED, dash="dash"))
    style(fig, "Brazil's markets around the presidential run-off",
          f"23 to 27 Oct closes: Ibovespa {stats['BVSP']:+.1f}%, EWZ {stats['EWZ']:+.1f}%, real "
          f"{stats['BRL']:+.1f}%. Since 23 Oct, latest: Ibovespa {stats['BVSP_now']:+.1f}%, EWZ "
          f"{stats['EWZ_now']:+.1f}%, real {stats['BRL_now']:+.1f}%.",
          "Rebased to 100 at the 23 Oct 2026 close; dashed line = run-off (25 Oct). "
          + _src(ctx, ["BVSP", "EWZ", "USDBRL"]))
    fig.update_xaxes(tickformat="%d %b")
    fig.update_xaxes(rangebreaks=WEEKENDS)
    return fig, stats


def build_tech_treasuries(ctx: Ctx):
    ndx = _px(ctx, ["NDX"], "2026-09-01")["NDX"].dropna()
    ten = _yld(ctx, "10Y", "2026-09-01")
    stats = {"ndx_last": ndx.iloc[-1], "ten_last": ten.iloc[-1] if len(ten) else np.nan}
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=ndx.index, y=ndx, name="Nasdaq 100", line=dict(color=NAVY, width=2.5))
    if len(ten):
        fig.add_scatter(x=ten.index, y=ten, name="10-year Treasury yield, % (right)",
                        line=dict(color=RED, width=2), secondary_y=True)
    fig.add_vline(x=pd.Timestamp("2026-10-22"), line=dict(color=GREY, dash="dot"))
    style(fig, "Tech rout and the 10-year Treasury yield",
          f"Nasdaq 100 {stats['ndx_last']:,.0f}; 10-year yield {stats['ten_last']:.2f}%. Dotted line: "
          "22 Oct 2026, the session the issue covers.", _src(ctx, ["NDX"], yields=True))
    fig.update_yaxes(ticksuffix="%", showgrid=False, secondary_y=True)
    fig.update_xaxes(tickformat="%d %b")
    fig.update_xaxes(rangebreaks=WEEKENDS)
    return fig, stats


BUILDERS = {
    "nvda_rebound": build_nvda_rebound, "nasdaq_ytd": build_nasdaq_ytd, "umich": build_umich,
    "anthropic_ipo": build_anthropic_ipo, "defensive_rotation": build_defensive_rotation,
    "cpi_2y": build_cpi_2y, "dow_midterm": build_dow_midterm, "sector_week": build_sector_week,
    "gdp": build_gdp, "brazil": build_brazil, "tech_treasuries": build_tech_treasuries,
}
