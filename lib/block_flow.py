"""
Shared block-trade analytics + Streamlit page renderer for non-Deribit venues.

``pages/03_Block_Trades_-_Derive.py`` and ``pages/04_Block_Trades_-_Paradex.py``
are thin wrappers around :func:`render_page`.  The layout, charts and Telegram
sending mirror ``pages/02_Block_Trades_-_Deribit.py``; that page is left
untouched.

A venue object (see ``lib.derive.VENUE`` / ``lib.paradex.VENUE``) supplies:

    key, title, icon, assets, default_min_sizes, has_mark, perp_label, definition
    fetch_all(start_ms, end_ms, assets) -> {"frames": {asset: df}, "meta": {...}}
    fetch_spot(asset) -> float
    fetch_hist_spot(asset, start_ms, end_ms) -> DataFrame[timestamp (SGT), close]
    feed_status(meta) -> Optional[str]

Trade frames use one schema (all USD-quoted, USDC-settled, linear options):
    timestamp (SGT tz-aware), instrument_name, strike, expiry, option_type ('C'/'P'),
    direction ('buy'/'sell' = taker side), abs_amount, amount (signed), price,
    mark_price (NaN if the venue has none), index_price (NaN allowed), iv (% or NaN),
    block_id, is_block

Differences from the Deribit page (all because these venues are USD-quoted):
* premium = amount * price (no * spot — Deribit BTC/ETH are coin-margined)
* Greeks use each trade's own IV (backed out of price when the venue gives none)
  instead of one flat per-asset vol
* bubble/marker sizes scale with premium, since block sizes differ wildly by asset
"""

from __future__ import annotations

import concurrent.futures
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from lib import cache as cache_lib
from lib import deribit
from lib import fx_style
from lib import telegram
from lib.constants import TTL_MEDIUM, TTL_SHORT
from lib.telegram_caption import caption_from_title

SGT = timezone(timedelta(hours=8))
BG = "#FFFFFF"
TEXT = "#000000"
SECONDARY = "#3790C7"
YEAR = 365.25 * 86400
DEFAULT_IV = {"BTC": 0.60, "ETH": 0.65, "SOL": 0.75, "HYPE": 0.90}
TENOR_EDGES = [0, 7, 30, 90, 365, float("inf")]
TENOR_LABELS = ["0d-1w", "1w-1m", "1m-3m", "3m-1y", "1y+"]


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------
def implied_vol(price, S, K, T, is_call, lo=1e-3, hi=8.0, iters=60):
    """Vectorised Black-Scholes implied vol (r=0, USD price per 1 underlying) in
    decimal.  NaN where the price is outside no-arbitrage bounds or the solve
    pins to the search bounds."""
    price, S, K, T = (np.asarray(x, dtype=float) for x in np.broadcast_arrays(price, S, K, T))
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), price.shape)
    with np.errstate(all="ignore"):
        intrinsic = np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
        upper = np.where(is_call, S, K)
        a = np.full(price.shape, lo)
        b = np.full(price.shape, hi)

        def bs(sig):
            sq = sig * np.sqrt(T)
            d1 = (np.log(S / K) + 0.5 * sig ** 2 * T) / sq
            d2 = d1 - sq
            call = S * norm.cdf(d1) - K * norm.cdf(d2)
            return np.where(is_call, call, call - S + K)

        for _ in range(iters):
            mid = 0.5 * (a + b)
            below = bs(mid) < price
            a = np.where(below, mid, a)
            b = np.where(below, b, mid)
        iv = 0.5 * (a + b)
        bad = (~np.isfinite(price) | ~np.isfinite(S) | ~np.isfinite(K) | ~np.isfinite(T)
               | (S <= 0) | (K <= 0) | (T <= 0) | (price <= intrinsic + 1e-9)
               | (price >= upper) | (iv >= hi - 1e-4) | (iv <= lo + 1e-4))
        iv = np.where(bad, np.nan, iv)
    return iv


def enrich(df: pd.DataFrame, spot: float, asset: str, has_mark: bool) -> pd.DataFrame:
    """Add IV, Greeks, dollar Greeks, premium, aggression and bucket columns."""
    if df.empty:
        return df
    df = df.copy()
    now = pd.Timestamp.now(tz="UTC")
    exp_utc = (pd.to_datetime(df["expiry"]).dt.tz_localize("UTC") + pd.Timedelta(hours=8))
    ts_utc = df["timestamp"].dt.tz_convert("UTC")
    tte_now = ((exp_utc - now).dt.total_seconds() / YEAR).clip(lower=0.001)
    tte_trade = ((exp_utc - ts_utc).dt.total_seconds() / YEAR)

    px = df["index_price"].where(df["index_price"] > 0)
    px = px.fillna(spot if spot and spot > 0 else np.nan)
    df["index_price"] = px

    if "iv" not in df.columns:
        df["iv"] = np.nan
    missing = df["iv"].isna()
    if missing.any():
        solved = implied_vol(df.loc[missing, "price"].values, px[missing].values,
                             df.loc[missing, "strike"].values, tte_trade[missing].values,
                             (df.loc[missing, "option_type"] == "C").values)
        df.loc[missing, "iv"] = solved * 100.0

    # Greeks are the risk the block *transferred at execution*: trade-time spot
    # and time-to-expiry (not "now" — that zeroes out every option that has since
    # expired inside the lookback window).  Floor T at one hour.
    sigma = (df["iv"] / 100.0).where(df["iv"] > 0).fillna(DEFAULT_IV.get(asset, 0.80)).values
    fallback = spot if spot and spot > 0 else (float(np.nanmedian(px)) if px.notna().any() else np.nan)
    S = px.fillna(fallback).values
    K = df["strike"].values
    T = tte_trade.clip(lower=1 / (365.25 * 24)).values
    sqrt_T = np.sqrt(T)
    with np.errstate(all="ignore"):
        d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    is_call = (df["option_type"] == "C").values
    amt = df["amount"].values

    df["tte"] = T
    df["delta"] = np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1)
    df["gamma"] = norm.pdf(d1) / (S * sigma * sqrt_T)
    df["vega"] = S * norm.pdf(d1) * sqrt_T * 0.01
    df["dollar_delta"] = df["delta"] * amt * S
    df["dollar_gamma_1pct"] = df["gamma"] * amt * S ** 2 * 0.01
    df["dollar_vega"] = df["vega"] * amt
    df["gross_notional"] = df["abs_amount"] * df["strike"]

    df["minute"] = df["timestamp"].dt.floor("min")
    df["direction_str"] = np.where(df["direction"] == "buy", "Buy", "Sell")
    df["expiry_str"] = df["expiry"].dt.strftime("%d%b%y").str.upper()
    df["vega_usd"] = df["dollar_vega"]
    df["dollar_gamma"] = df["dollar_gamma_1pct"]
    df["vega_usd_30d"] = df["vega_usd"] * np.sqrt((30 / 365.25) / df["tte"])

    # USD-quoted linear options: premium needs no * spot.  Signed: buyer pays (+).
    df["premium_usd"] = df["amount"] * df["price"]
    df["abs_premium_usd"] = df["premium_usd"].abs()
    if has_mark:
        df["aggression_usd"] = (df["price"] - df["mark_price"]) * df["amount"]
        df["edge"] = ((df["mark_price"] - df["price"]) * df["amount"]).abs()
    else:
        df["aggression_usd"] = np.nan
        df["edge"] = np.nan
    abs_vega = df["vega_usd"].abs()
    df["edge_vol"] = np.where(abs_vega > 0, df["aggression_usd"] / abs_vega, np.nan)

    a = df["delta"].abs()
    call_side = df["delta"] > 0
    df["delta_bucket"] = np.select(
        [(a >= 0.35) & (a <= 0.65),
         call_side & (a >= 0.15) & (a < 0.35), call_side & (a > 0) & (a < 0.15),
         ~call_side & (a >= 0.15) & (a < 0.35), ~call_side & (a > 0) & (a < 0.15)],
        ["ATM", "25D Call", "10D Call", "25D Put", "10D Put"], default="Unknown")
    return df


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def create_expiry_table(df: pd.DataFrame, has_mark: bool) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    d = df.assign(net_puts=np.where(df["option_type"] == "P", df["amount"], 0.0),
                  net_calls=np.where(df["option_type"] == "C", df["amount"], 0.0))
    agg = {"dollar_delta": "sum", "dollar_gamma_1pct": "sum", "dollar_vega": "sum",
           "net_puts": "sum", "net_calls": "sum", "gross_notional": "sum", "abs_premium_usd": "sum"}
    if has_mark:
        agg["edge"] = "sum"
    g = d.groupby("expiry").agg(agg).reset_index().sort_values("expiry")
    g["Expiry"] = g["expiry"].dt.strftime("%d-%b-%Y")
    g = g.rename(columns={"dollar_delta": "Delta", "dollar_gamma_1pct": "Gamma (1%)",
                          "dollar_vega": "Vega", "net_puts": "Net Puts", "net_calls": "Net Calls",
                          "gross_notional": "Gross Notional", "abs_premium_usd": "Premium", "edge": "Edge"})
    cols = ["Expiry", "Delta", "Vega", "Gamma (1%)", "Net Puts", "Net Calls", "Gross Notional", "Premium"]
    if has_mark:
        cols.append("Edge")
    total = {c: g[c].sum() for c in cols if c != "Expiry"}
    total["Expiry"] = "Total"
    return pd.concat([g[cols], pd.DataFrame([total])[cols]], ignore_index=True)


def style_statistics_table(df: pd.DataFrame):
    if df.empty:
        return df
    out = df.copy()
    for c in ("Delta", "Vega", "Gamma (1%)", "Edge", "Premium"):
        if c in out.columns:
            out[c] = out[c].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "$0")
    for c in ("Net Puts", "Net Calls", "Gross Notional"):
        if c in out.columns:
            out[c] = out[c].apply(lambda x: f"{x:,.2f}" if c.startswith("Net") else f"{x:,.0f}")

    def colour(v):
        if isinstance(v, str):
            try:
                n = float(v.replace("$", "").replace(",", ""))
                return "color: red" if n < 0 else "color: green" if n > 0 else ""
            except ValueError:
                return ""
        return ""

    cols = [c for c in ("Delta", "Vega", "Gamma (1%)", "Net Puts", "Net Calls") if c in out.columns]
    sty = out.style
    return sty.map(colour, subset=cols) if hasattr(sty, "map") else sty.applymap(colour, subset=cols)


def build_packages(df: pd.DataFrame) -> pd.DataFrame:
    """One row per block package (legs sharing a block_id; unlabelled legs stand alone)."""
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["_pkg"] = d["block_id"].where(d["block_id"].notna(), d["trade_id"] if "trade_id" in d else d.index.astype(str))
    rows = []
    for pkg, g in d.groupby("_pkg", sort=False):
        g = g.sort_values(["expiry", "strike", "option_type"])
        legs = " / ".join(
            f"{'+' if r.direction == 'buy' else '-'}{r.abs_amount:g} {r.expiry_str} {r.strike:g}{r.option_type}"
            for r in g.itertuples())
        rows.append({
            "Time (SGT)": g["timestamp"].min().strftime("%m-%d %H:%M:%S"),
            "Legs": len(g),
            "Structure (taker side)": legs,
            "Net premium $": g["premium_usd"].sum(),
            "Net delta $": g["dollar_delta"].sum(),
            "Net vega $": g["dollar_vega"].sum(),
            "Gross premium $": g["abs_premium_usd"].sum(),
            "_t": g["timestamp"].min(),
        })
    out = pd.DataFrame(rows).sort_values("_t", ascending=False).drop(columns="_t")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _layout(fig, title, h=400, **kw):
    base = dict(title=title, paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXT),
                height=h, margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    base.update(kw)
    fig.update_layout(**base)
    fx_style.add_watermark(fig)
    return fx_style.apply_theme(fig)


def _empty(title):
    return _layout(go.Figure(), title)


def _sizes(prem, lo=7.0, hi=46.0):
    p = pd.Series(prem, dtype=float).abs().fillna(0.0)
    m = p.max()
    if not m or m <= 0:
        return pd.Series(lo, index=p.index)
    return lo + (hi - lo) * np.sqrt(p / m)


def plot_scatter(data, hist_spot, dvol, asset, perp_label):
    has_dvol = dvol is not None and not dvol.empty
    title = f"{asset} Block Trades Over Time ({perp_label}{' & DVOL' if has_dvol else ''})"
    if data.empty:
        return _empty(f"{asset} Block Trades Over Time: No Data")
    df = data.copy()
    df["minute_n"] = df["timestamp"].dt.floor("min").dt.tz_localize(None)
    g = df.groupby(["minute_n", "instrument_name", "strike", "expiry_str", "option_type"]).agg(
        amount=("amount", "sum"), price=("price", "mean"), prem=("abs_premium_usd", "sum")).reset_index()
    fig = go.Figure()
    for sign in (1, -1):
        for opt in ("C", "P"):
            t = g[(np.sign(g["amount"]) == sign) & (g["option_type"] == opt)]
            if t.empty:
                continue
            hover = [f"{r.minute_n:%Y-%m-%d %H:%M} SGT<br>{r.instrument_name}<br>Amount: {r.amount:,.3f}"
                     f"<br>Avg price: {r.price:,.2f}<br>Premium: ${r.prem:,.0f}" for r in t.itertuples()]
            fig.add_trace(go.Scatter(
                x=t["minute_n"], y=t["strike"], mode="markers", text=hover, hoverinfo="text",
                marker=dict(size=_sizes(t["prem"]), color="green" if sign == 1 else "red",
                            symbol="square" if opt == "P" else "circle", opacity=0.75,
                            line=dict(width=1, color="white")),
                name=f'{"Buy" if sign == 1 else "Sell"} {opt}'))
    if hist_spot is not None and not hist_spot.empty:
        fig.add_trace(go.Scatter(x=hist_spot["timestamp"].dt.tz_localize(None), y=hist_spot["close"],
                                 mode="lines", name=f"{asset} {perp_label}", yaxis="y2",
                                 line=dict(color=SECONDARY, width=1.5)))
    extra = dict(xaxis_title="Timestamp (SGT)", yaxis_title="Option Strike",
                 yaxis2=dict(title=f"{asset} {perp_label}", overlaying="y", side="right", showgrid=False))
    if dvol is not None and not dvol.empty:
        fig.add_trace(go.Scatter(x=dvol.index.tz_localize(None), y=dvol.values, mode="lines",
                                 name="DVOL (Deribit)", yaxis="y3", line=dict(color="orange", width=1.5)))
        extra["yaxis3"] = dict(title="DVOL", overlaying="y", side="right", position=0.95, showgrid=False)
    return _layout(fig, title, h=600, **extra)


def plot_strike_vs_expiry(data, asset):
    if data.empty:
        return _empty(f"{asset} Aggregated Trades by Expiry and Strike: No Data")
    agg = data.groupby(["strike", "expiry", "expiry_str", "option_type"]).agg(
        amount=("amount", "sum"), prem=("abs_premium_usd", "sum")).reset_index()
    agg["direction"] = np.where(agg["amount"] >= 0, "Buy", "Sell")
    sizeref = 2.0 * max(float(agg["prem"].max()), 1.0) / (40 ** 2)
    order = agg[["expiry_str", "expiry"]].drop_duplicates().sort_values("expiry")["expiry_str"].tolist()
    fig = go.Figure()
    for d, colr in (("Buy", "green"), ("Sell", "red")):
        for opt, sym in (("C", "circle"), ("P", "square")):
            s = agg[(agg["direction"] == d) & (agg["option_type"] == opt)]
            if s.empty:
                continue
            fig.add_trace(go.Scatter(
                x=s["expiry_str"], y=s["strike"], mode="markers", hoverinfo="text",
                text=[f"Expiry: {r.expiry_str}<br>Strike: {r.strike:,.0f}<br>{d} {opt}<br>"
                      f"Premium: ${r.prem:,.0f}<br>Net amount: {r.amount:,.3f}" for r in s.itertuples()],
                marker=dict(size=s["prem"], sizemode="area", sizeref=max(sizeref, 1e-12), sizemin=4,
                            color=colr, symbol=sym, opacity=0.75, line=dict(width=1, color="white")),
                name=f"{d} {opt}"))
    lo, hi = max(agg["strike"].min(), 1e-6), max(agg["strike"].max(), agg["strike"].min() + 1e-6)
    ticks = np.unique(np.round(np.geomspace(lo, hi, 10), 4))
    return _layout(fig, f"{asset} Aggregated Trades by Expiry and Strike (bubble = premium)", h=500,
                   xaxis_title="Expiry Date", yaxis_title="Strike (Log Scale)",
                   xaxis=dict(type="category", categoryorder="array", categoryarray=order),
                   yaxis=dict(type="log", tickvals=ticks, ticktext=[f"{t:,.4g}" for t in ticks],
                              ticks="outside", showline=True, linewidth=1, linecolor="black", mirror=True))


def _heatmap(data, asset, value, title, scale, zsym, cbar):
    if data.empty:
        return _empty(f"{asset} {title}: No Data")
    p = data.groupby(["strike", "expiry"])[value].sum().reset_index().pivot(
        index="expiry", columns="strike", values=value).fillna(0).sort_index()
    p = p.reindex(sorted(p.columns), axis=1)
    m = float(np.max(np.abs(p.values))) or 1.0
    fig = go.Figure(go.Heatmap(
        z=p.values, x=list(p.columns), y=list(p.index.strftime("%Y-%m-%d")), colorscale=scale,
        zmin=-m if zsym else 0, zmax=m,
        colorbar=dict(title=cbar, bgcolor=BG, bordercolor=TEXT, tickfont=dict(color=TEXT)),
        hovertemplate="Strike: %{x}<br>Expiry: %{y}<br>%{z}<extra></extra>"))
    return _layout(fig, f"{asset} {title}",
                   xaxis=dict(title="Strike", type="category", categoryorder="array", categoryarray=list(p.columns)),
                   yaxis=dict(title="Expiry", type="category", categoryorder="array",
                              categoryarray=list(p.index.strftime("%Y-%m-%d"))))


def plot_net_heatmap(data, asset):
    return _heatmap(data, asset, "amount", "Option Strike vs Expiry Net Flow Heatmap",
                    [[0, "red"], [0.5, "white"], [1, "green"]], True, "Net Contracts")


def plot_gross_heatmap(data, asset):
    return _heatmap(data, asset, "abs_amount", "Gross Volume Heatmap",
                    [[0, "white"], [0.5, "#66b2ff"], [1, "#0000ff"]], False, "Gross Contracts")


def plot_cumulative_flow(data, asset):
    if data.empty:
        return _empty(f"{asset} Cumulative Flow: No Data")
    fig = go.Figure()
    for lab, mask, colr, w in (
            ("Total Volume", data["abs_amount"].notna(), SECONDARY, 2), ("Cumulative Buys", data["direction"] == "buy", "green", 1.5),
            ("Cumulative Sells", data["direction"] == "sell", "red", 1.5), ("Cumulative Calls", data["option_type"] == "C", "blue", 1.5),
            ("Cumulative Puts", data["option_type"] == "P", "purple", 1.5)):
        s = data[mask].groupby("minute")["abs_amount"].sum().cumsum()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=lab, line=dict(color=colr, width=w)))
    return _layout(fig, f"{asset} Cumulative Volume Over Time", xaxis_title="Date (SGT)", yaxis_title="Contracts")


def plot_iv_surface(data, asset, with_note):
    f = data[(data["strike"] > 0) & data["iv"].notna() & (data["iv"] > 0)] if not data.empty else data
    if f.empty:
        return _empty(f"{asset} Traded IV Surface: No IV Data")
    f = f.assign(_w=f["abs_amount"], _ivw=f["iv"] * f["abs_amount"])
    exps = f[["expiry", "expiry_str"]].drop_duplicates().sort_values("expiry")
    n = len(exps)
    cols = (pcolors.sample_colorscale("Turbo", [i / (n - 1) for i in range(n)]) if n > 1
            else pcolors.sample_colorscale("Turbo", [0.5]))
    fig = go.Figure()
    for (_, r), colr in zip(exps.iterrows(), cols):
        e = f[f["expiry"] == r["expiry"]]
        sums = e.groupby("strike")[["_ivw", "_w"]].sum()
        line = (sums["_ivw"] / sums["_w"]).rename("v").reset_index().sort_values("strike")
        if len(line) > 1:
            fig.add_trace(go.Scatter(x=line["strike"], y=line["v"], mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color=colr, width=2, dash="dot"), opacity=0.6, legendgroup=r["expiry_str"]))
        for side, sym in (("Buy", "triangle-up"), ("Sell", "triangle-down")):
            s = e[e["direction_str"] == side]
            if s.empty:
                continue
            fig.add_trace(go.Scatter(
                x=s["strike"], y=s["iv"], mode="markers", hoverinfo="text", legendgroup=r["expiry_str"],
                name=f"{r['expiry_str']} ({side})", showlegend=(side == "Buy"),
                text=[f"{side} {x.option_type}<br>{r['expiry_str']}<br>Strike: {x.strike:,.0f}<br>"
                      f"Size: {x.abs_amount:,.3f}<br>IV: {x.iv:.1f}%" for x in s.itertuples()],
                marker=dict(size=_sizes(s["abs_premium_usd"], 6, 30), color=colr, symbol=sym, opacity=0.8,
                            line=dict(width=1, color="white"))))
    return _layout(fig, f"{asset} Traded IV Surface (Coloured by Tenor)", xaxis_title="Strike (Log Scale)",
                   yaxis_title="Implied Volatility (%)", xaxis=dict(type="log"),
                   legend=dict(x=1.02, y=1, bordercolor="rgba(0,0,0,0.2)", borderwidth=1),
                   margin=dict(l=40, r=120, t=40, b=40))


def plot_term_structure_flow(data, asset):
    if data.empty:
        return _empty(f"{asset} Term Structure Flow: No Data")
    agg = data.groupby(["expiry", "expiry_str", "option_type"])["abs_amount"].sum().reset_index().sort_values("expiry")
    fig = go.Figure()
    for opt, colr, nm in (("C", "blue", "Calls"), ("P", "purple", "Puts")):
        s = agg[agg["option_type"] == opt]
        if not s.empty:
            fig.add_trace(go.Bar(x=s["expiry_str"], y=s["abs_amount"], name=nm, marker_color=colr))
    return _layout(fig, f"{asset} Term Structure Flow (Total Traded Volume)", xaxis_title="Expiry",
                   yaxis_title="Absolute Volume", barmode="stack",
                   xaxis=dict(categoryorder="array", categoryarray=list(agg["expiry_str"].unique())))


def plot_put_call(data, asset):
    if data.empty:
        return _empty(f"{asset} Flow & Put/Call Ratio: No Data")
    d = data.assign(call_vol=np.where(data["option_type"] == "C", data["abs_amount"], 0.0),
                    put_vol=np.where(data["option_type"] == "P", data["abs_amount"], 0.0))
    g = d.groupby("minute")[["call_vol", "put_vol"]].sum().cumsum()
    g["pc"] = np.where(g["call_vol"] > 0, g["put_vol"] / g["call_vol"], 0.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.index, y=g["call_vol"], mode="lines", name="Cumul. Calls", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=g.index, y=g["put_vol"], mode="lines", name="Cumul. Puts", line=dict(color="purple")))
    fig.add_trace(go.Scatter(x=g.index, y=g["pc"], mode="lines", name="Put/Call Ratio", yaxis="y2",
                             line=dict(color="#CC9900", dash="dot")))
    return _layout(fig, f"{asset} Flow & Put/Call Ratio", xaxis_title="Time (SGT)", yaxis_title="Cumulative Volume",
                   yaxis2=dict(title="Put/Call Ratio", overlaying="y", side="right", range=[0, max(3, g["pc"].max() * 1.2)]))


def plot_delta_term(data, asset):
    f = data.dropna(subset=["iv", "expiry"]) if not data.empty else data
    f = f[(f["delta_bucket"] != "Unknown") & (f["iv"] > 0)] if not f.empty else f
    if f.empty:
        return _empty(f"{asset} Traded Term Structure by Delta: No Data")
    f = f.assign(_w=f["abs_amount"], _ivw=f["iv"] * f["abs_amount"])
    a = f.groupby(["expiry", "expiry_str", "delta_bucket"])[["_ivw", "_w"]].sum().reset_index()
    a["v"] = a["_ivw"] / a["_w"]
    a = a.sort_values("expiry")
    styles = {"10D Call": ("#33CCFF", "dot"), "25D Call": ("#3366FF", "dash"), "ATM": ("#CC9900", "solid"),
              "25D Put": ("#CC66FF", "dash"), "10D Put": ("#9933FF", "dot")}
    fig = go.Figure()
    for b, (colr, dash) in styles.items():
        s = a[a["delta_bucket"] == b]
        if not s.empty:
            fig.add_trace(go.Scatter(x=s["expiry_str"], y=s["v"], mode="lines+markers", name=b,
                                     line=dict(color=colr, width=2, dash=dash), marker=dict(size=8, color=colr)))
    return _layout(fig, f"{asset} Traded Term Structure by Delta", xaxis_title="Expiry Date",
                   yaxis_title="VWAP Implied Volatility (%)",
                   xaxis=dict(categoryorder="array", categoryarray=list(a["expiry_str"].unique())),
                   legend=dict(x=1.02, y=1, bordercolor="rgba(0,0,0,0.2)", borderwidth=1),
                   margin=dict(l=40, r=120, t=40, b=40))


def _vega_bars(data, asset, group, col, title, ytitle, spot=0.0):
    f = data[data[col].notna()] if not data.empty else data
    if f.empty:
        return _empty(f"{asset} {title}: No Data")
    by_strike = group == "strike"
    keys = ["strike", "option_type"] if by_strike else ["expiry", "expiry_str", "option_type"]
    a = f.groupby(keys, as_index=False).agg(v=(col, "sum"), gross=("abs_amount", "sum"), net=("amount", "sum"))
    if by_strike:
        cat = sorted(a["strike"].unique())
        xa = dict(title="Strike", type="category", categoryorder="array", categoryarray=cat)
    else:
        cat = a[["expiry", "expiry_str"]].drop_duplicates().sort_values("expiry")["expiry_str"].tolist()
        xa = dict(title="Expiry", categoryorder="array", categoryarray=cat)
    fig = go.Figure()
    xcol = "strike" if by_strike else "expiry_str"
    for opt, colr, nm in (("C", "blue", "Calls"), ("P", "purple", "Puts")):
        s = a[a["option_type"] == opt]
        if s.empty:
            continue
        fig.add_trace(go.Bar(x=s[xcol], y=s["v"], name=nm, marker_color=colr,
                             customdata=np.stack([s["net"], s["gross"]], axis=-1),
                             hovertemplate="%{x}<br>$%{y:,.0f}<br>Net: %{customdata[0]:,.3f}<br>Gross: %{customdata[1]:,.3f}"
                                           f"<extra>{nm}</extra>"))
    fig.add_hline(y=0, line_color="rgba(0,0,0,0.3)")
    if by_strike and spot and spot > 0 and cat:
        near = min(cat, key=lambda k: abs(k - spot))
        # category axis of numeric labels: add_vline needs the index (plotly.py#3013)
        fig.add_vline(x=cat.index(near), line_dash="dot", line_color="rgba(0,0,0,0.6)",
                      annotation_text="spot", annotation_font=dict(color=TEXT, size=10))
    return _layout(fig, f"{asset} {title}", barmode="relative", xaxis=xa,
                   yaxis=dict(title=ytitle, tickprefix="$", tickformat="~s"))


def plot_cumulative_aggression(data, asset):
    f = data[data["aggression_usd"].notna()] if not data.empty else data
    if f.empty:
        return _empty(f"{asset} Aggression: No Data")
    f = f.copy()
    tot = f.groupby("minute")["aggression_usd"].sum().cumsum()
    fig = go.Figure(go.Scatter(x=tot.index, y=tot.values, mode="lines", name="Total", line=dict(color=TEXT, width=2)))
    f["tb"] = pd.cut(f["tte"] * 365.25, bins=TENOR_EDGES, labels=TENOR_LABELS, right=False)
    cols = dict(zip(TENOR_LABELS, pcolors.sample_colorscale("Turbo", [i / 4 for i in range(5)])))
    for b in TENOR_LABELS:
        if (f["tb"] == b).any():
            s = f[f["tb"] == b].groupby("minute")["aggression_usd"].sum().cumsum()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=b, line=dict(color=cols[b], width=1.5)))
    fig.add_hline(y=0, line_color="rgba(0,0,0,0.3)")
    return _layout(fig, f"{asset} Aggression - Cumulative Premium Through Mark (>0 = takers paying up)",
                   xaxis_title="Time (SGT)", yaxis=dict(title="Cum. $ through mark", tickprefix="$", tickformat="~s"))


def build_figs(data, hist_spot, dvol, spot, asset, venue):
    """The ordered chart list for one asset (also what is sent to Telegram)."""
    figs = [
        plot_scatter(data, hist_spot, dvol, asset, venue.perp_label),
        plot_strike_vs_expiry(data, asset),
        plot_net_heatmap(data, asset),
        plot_gross_heatmap(data, asset),
        plot_cumulative_flow(data, asset),
        plot_iv_surface(data, asset, False),
        plot_term_structure_flow(data, asset),
        plot_put_call(data, asset),
        plot_delta_term(data, asset),
        _vega_bars(data, asset, "expiry", "vega_usd", "Net Vega Flow by Expiry (buy +, sell -)", "Net Vega (USD)"),
        _vega_bars(data, asset, "strike", "vega_usd", "Net Vega by Strike (session, buy +, sell -)", "Net Vega (USD)", spot),
        _vega_bars(data, asset, "expiry", "vega_usd_30d", "Weighted Vega Flow by Expiry (30d-equiv, buy +, sell -)",
                   "30d-Weighted Vega (USD)"),
        _vega_bars(data, asset, "strike", "vega_usd_30d", "Weighted Vega by Strike (30d-equiv, session, buy +, sell -)",
                   "30d-Weighted Vega (USD)", spot),
    ]
    if venue.has_mark:
        figs.append(plot_cumulative_aggression(data, asset))
    return figs


# ---------------------------------------------------------------------------
# Telegram (same render path as pages/02_Block_Trades_-_Deribit.py)
# ---------------------------------------------------------------------------
def _render_png(fig):
    try:
        return fig.to_image(format="png", width=1200, height=800), None
    except Exception as e:  # kaleido / chromium problems surface here
        return None, str(e)


def _send_chart(fig, base):
    if fig is None:
        return False, f"{base}: no chart to send"
    t = getattr(getattr(fig, "layout", None), "title", None)
    cap = caption_from_title(getattr(t, "text", None) if t else None, base)
    themed = fx_style.apply_theme(fig, "light")
    png, err = _render_png(themed)
    if png is None:
        png, err = _render_png(themed)
    if png is None:
        return False, f"{base}: failed to **render**" + (f" ({err})" if err else "")
    return (True, None) if telegram.send_photo(png, cap) else (False, f"{base}: failed to **send**")


def _send_many(pairs, label, venue_title):
    if not pairs:
        st.warning(f"Nothing to send for {label}.")
        return
    ok_n = bad_n = 0
    reasons = []
    bar = st.progress(0.0) if len(pairs) > 1 else None
    with st.spinner(f"Sending {label} to Telegram..."):
        for i, (name, fig) in enumerate(pairs, 1):
            ok, why = _send_chart(fig, f"{name} {venue_title} block trades")
            ok_n += ok
            bad_n += (not ok)
            if why:
                reasons.append(why)
            if bar:
                bar.progress(i / len(pairs))
    if bar:
        bar.empty()
    if ok_n:
        st.success(f"Sent {ok_n} chart(s) to Telegram." + (f" ({bad_n} failed.)" if bad_n else ""))
    else:
        st.error("Failed to send charts to Telegram.")
    if reasons:
        with st.expander(f"⚠️ {len(reasons)} chart(s) had trouble"):
            for r in reasons:
                st.markdown(f"- {r}")


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------
_VENUES: dict = {}


@st.cache_data(ttl=TTL_MEDIUM, show_spinner=False)
def _cached_all(venue_key: str, start_ms: int):
    v = _VENUES[venue_key]
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return v.fetch_all(start_ms, end_ms, list(v.assets))


@st.cache_data(ttl=TTL_SHORT, show_spinner=False)
def _cached_spot(venue_key: str, asset: str) -> float:
    return float(_VENUES[venue_key].fetch_spot(asset) or 0.0)


@st.cache_data(ttl=TTL_MEDIUM, show_spinner=False)
def _cached_hist(venue_key: str, asset: str, start_ms: int):
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return _VENUES[venue_key].fetch_hist_spot(asset, start_ms, end_ms)


@st.cache_data(ttl=TTL_MEDIUM, show_spinner=False)
def _cached_dvol(asset: str, start_ms: int) -> pd.Series:
    """Deribit DVOL (BTC/ETH only) — reference vol overlay, not venue data."""
    if asset not in ("BTC", "ETH"):
        return pd.Series(dtype=float)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    df = deribit.get_dvol(asset, resolution="60", start_ms=start_ms, end_ms=end_ms)
    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(SGT)
    return out.set_index("timestamp")["close"].sort_index()


def _anchor(now):
    am = now.replace(hour=7, minute=0, second=0, microsecond=0)
    pm = now.replace(hour=19, minute=0, second=0, microsecond=0)
    if now >= pm:
        return pm
    if now >= am:
        return am
    return (now - timedelta(days=1)).replace(hour=19, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render_page(venue) -> None:
    _VENUES[venue.key] = venue
    assets = list(venue.assets)

    st.set_page_config(page_title=f"Block Trades — {venue.title}", page_icon="📊", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown("""
<style>
.main-header {background: linear-gradient(90deg,#1E4D7A 0%,#3790C7 100%); padding:1rem; border-radius:.5rem;
  color:#fff; text-align:center; font-weight:bold; margin-bottom:1rem;}
.asset-header {background-color:#1E4D7A; padding:.5rem; border-radius:.5rem; color:#fff; text-align:center;
  font-weight:bold; margin-bottom:1rem;}
</style>""", unsafe_allow_html=True)
    st.markdown(f'<div class="main-header"><h1>📊 BLOCK TRADES - {venue.title.upper()}</h1></div>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Dashboard Controls")
        mode = st.radio("Time Window", ["Last 24 Hours", "Last 12 Hours", "Last 7 Days", "Custom Start Time"], index=0,
                        key=f"{venue.key}_mode")
        now_sg = datetime.now(SGT)
        if mode == "Last 24 Hours":
            start_sgt = now_sg - timedelta(hours=24)
        elif mode == "Last 12 Hours":
            start_sgt = now_sg - timedelta(hours=12)
        elif mode == "Last 7 Days":
            start_sgt = now_sg - timedelta(days=7)
        else:
            anc = _anchor(now_sg)
            d = st.date_input("Start Date", value=anc.date(), key=f"{venue.key}_d")
            t = st.time_input("Start Time (SGT)", value=anc.time(), key=f"{venue.key}_t")
            start_sgt = datetime.combine(d, t).replace(tzinfo=SGT)
        # minute-rounded so the cache key is stable within a minute
        start_ms = int(start_sgt.astimezone(timezone.utc).replace(second=0, microsecond=0).timestamp() * 1000)

        st.subheader("Minimum Block Sizes")
        min_sizes = {}
        for a in assets:
            dv = float(venue.default_min_sizes[a])
            min_sizes[a] = st.number_input(f"{a} Min Size", min_value=0.0, value=dv,
                                           step=max(dv / 10, 0.001), format="%g", key=f"{venue.key}_min_{a}")
        st.divider()
        if st.checkbox("Auto-refresh (60s)", value=False, key=f"{venue.key}_auto"):
            time.sleep(60)
            st.rerun()
        cache_lib.render_refresh_button()
        st.divider()
        st.subheader("Telegram")
        if telegram.is_configured():
            st.success("Telegram ready")
        else:
            st.warning("Telegram not configured")
            st.caption(telegram.config_status())

    with st.spinner(f"Fetching {venue.title} block trades, spot and DVOL..."):
        try:
            payload = _cached_all(venue.key, start_ms)
        except Exception as e:
            st.error(f"{venue.title} fetch failed: {e}")
            st.stop()
        frames, meta = payload["frames"], payload["meta"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            spot_f = {a: ex.submit(_cached_spot, venue.key, a) for a in assets}
            hist_f = {a: ex.submit(_cached_hist, venue.key, a, start_ms) for a in assets}
            dvol_f = {a: ex.submit(_cached_dvol, a, start_ms) for a in assets}
            spots = {a: spot_f[a].result() for a in assets}
            hists = {a: hist_f[a].result() for a in assets}
            dvols = {a: dvol_f[a].result() for a in assets}

    data = {}
    for a in assets:
        raw = frames.get(a, pd.DataFrame())
        if raw is None or raw.empty:
            data[a] = pd.DataFrame()
            continue
        f = raw[raw["abs_amount"] >= min_sizes[a]].copy()
        data[a] = enrich(f, spots[a], a, venue.has_mark) if not f.empty else pd.DataFrame()

    st.caption(venue.definition)
    note = venue.feed_status(meta)
    if note:
        st.info(note)

    cols = st.columns(len(assets))
    for c, a in zip(cols, assets):
        n = len(data[a])
        prem = data[a]["abs_premium_usd"].sum() if n else 0.0
        c.metric(f"{a} Blocks", f"{n}", f"${prem:,.0f} premium" if n else None, delta_color="off")

    # --- build charts once; reused by screen + Telegram
    figs = {a: build_figs(data[a], hists[a], dvols[a], spots[a], a, venue) for a in assets}
    n_charts = len(next(iter(figs.values())))

    tg_ok = telegram.is_configured()
    st.caption("Send charts to Telegram (as images):" if tg_ok else f"Telegram not configured: {telegram.config_status()}")
    bcols = st.columns(len(assets) + 2)
    clicked = {}
    for a, c in zip(assets, bcols):
        with c:
            clicked[a] = st.button(a, key=f"{venue.key}_tg_{a}", width="stretch", disabled=not tg_ok,
                                   help=f"Send {a}'s {n_charts} charts to Telegram.")
    btc_eth = [a for a in assets if a in ("BTC", "ETH")]
    with bcols[len(assets)]:
        c_pair = st.button("BTC+ETH", key=f"{venue.key}_tg_pair", width="stretch",
                           disabled=not tg_ok or not btc_eth)
    with bcols[len(assets) + 1]:
        c_all = st.button("📤 All", key=f"{venue.key}_tg_all", width="stretch", disabled=not tg_ok)

    tabs = st.tabs([f"📈 {a}" for a in assets] + ["📊 ALL (2x2 Grid)", "📋 Block Trade Statistics"])
    for i, a in enumerate(assets):
        with tabs[i]:
            st.markdown(f'<div class="asset-header">{a} Option Block Flow Analysis — {venue.title}</div>',
                        unsafe_allow_html=True)
            df = data[a]
            if df.empty:
                n_raw = len(frames.get(a, pd.DataFrame()))
                st.info(f"No {a} blocks at/above the {min_sizes[a]:g} minimum size in this window"
                        + (f" ({n_raw} below the minimum)." if n_raw else "."))
            fs = figs[a]
            k = f"{venue.key}_{a}"
            st.plotly_chart(fs[0], width="stretch", key=f"{k}_scatter")
            st.plotly_chart(fs[1], width="stretch", key=f"{k}_strike_expiry")
            c1, c2 = st.columns(2)
            c1.plotly_chart(fs[2], width="stretch", key=f"{k}_net_heat")
            c2.plotly_chart(fs[3], width="stretch", key=f"{k}_gross_heat")
            st.plotly_chart(fs[4], width="stretch", key=f"{k}_cum")
            st.divider()
            st.subheader(f"{a} Flow Analytics")
            st.plotly_chart(fs[5], width="stretch", key=f"{k}_iv")
            c3, c4 = st.columns(2)
            c3.plotly_chart(fs[6], width="stretch", key=f"{k}_term")
            c4.plotly_chart(fs[7], width="stretch", key=f"{k}_pc")
            st.plotly_chart(fs[8], width="stretch", key=f"{k}_delta")
            c5, c6 = st.columns(2)
            c5.plotly_chart(fs[9], width="stretch", key=f"{k}_vega_exp")
            c6.plotly_chart(fs[10], width="stretch", key=f"{k}_vega_strike")
            c7, c8 = st.columns(2)
            c7.plotly_chart(fs[11], width="stretch", key=f"{k}_wvega_exp")
            c8.plotly_chart(fs[12], width="stretch", key=f"{k}_wvega_strike")
            if venue.has_mark:
                st.plotly_chart(fs[13], width="stretch", key=f"{k}_aggr")
            if not df.empty:
                st.divider()
                st.subheader(f"{a} Block Packages")
                pk = build_packages(df)
                st.dataframe(pk.style.format({"Net premium $": "${:,.0f}", "Net delta $": "${:,.0f}",
                                              "Net vega $": "${:,.0f}", "Gross premium $": "${:,.0f}"}),
                             width="stretch", hide_index=True)

    for a in assets:
        if clicked.get(a):
            _send_many([(a, f) for f in figs[a]], a, venue.title)
    if c_pair and btc_eth:
        _send_many([(a, f) for a in btc_eth for f in figs[a]], "BTC+ETH", venue.title)
    if c_all:
        _send_many([(a, f) for a in assets for f in figs[a]], "all assets", venue.title)

    with tabs[len(assets)]:
        st.markdown(f'<div class="asset-header">📊 ALL {venue.title} Block Trades Overview</div>', unsafe_allow_html=True)
        for r in range(0, len(assets), 2):
            cc = st.columns(2)
            for c, a in zip(cc, assets[r:r + 2]):
                c.plotly_chart(plot_scatter(data[a], hists[a], dvols[a], a, venue.perp_label),
                               width="stretch", key=f"{venue.key}_grid_{a}")

    with tabs[len(assets) + 1]:
        st.markdown('<div class="asset-header">📋 Block Trade Statistics (Greeks & Expiry Breakdown)</div>',
                    unsafe_allow_html=True)
        sub = st.tabs(assets)
        for s, a in zip(sub, assets):
            with s:
                if data[a].empty:
                    st.info(f"No {a} block trades in this window.")
                    continue
                st.subheader(f"{a} Greeks & Volume by Expiry")
                st.dataframe(style_statistics_table(create_expiry_table(data[a], venue.has_mark)), width="stretch")
                st.caption("Greeks are dollar Greeks at current spot using each trade's own (price-implied) IV; "
                           "Delta/Vega/Gamma are signed from the taker's side (buy +, sell -).")
