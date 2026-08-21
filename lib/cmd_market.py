"""
MCM Analytics — market-structure commands (basis, RV, funding, flow, moon).

Deribit-only ports of the corresponding FalconX bot commands.  Where FalconX
used Binance klines or a multi-exchange funding wrapper, these use Deribit's
own perpetual candles and funding history.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

from lib import deribit, history, surface
from lib.fx_style import (BLUE, GREEN, GREY, NAVY, ORANGE, RED, TEMPLATE,
                          TIME_TICKFORMAT, XAXIS_TIME, add_watermark,
                          finalize, to_local, to_local_ts)

_finite = surface.finite

LUNAR_CYCLE_DAYS = 29.530588853
MOON_EPOCH = datetime(2000, 1, 6, 18, 14, 0, tzinfo=timezone.utc)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ===========================================================================
# Basis
# ===========================================================================

def _future_instrument(asset: str, dte: int) -> str | None:
    """Deribit instrument name for the dated future expiring in ``dte`` days."""
    cfg = surface.acfg(asset)
    chain = deribit.get_option_chain(cfg["deribit_ccy"], "future")
    if not chain:
        return None
    today = _now().date()
    prefix = cfg["deribit_prefix"] + "-"
    best, best_gap = None, None
    for row in chain:
        instr = row.get("instrument_name") or ""
        up = instr.upper()
        if "PERPETUAL" in up or "-PERP" in up:
            continue
        if cfg["style"] == "linear" and not instr.startswith(prefix):
            continue
        parts = instr.split("-")
        if len(parts) < 2:
            continue
        exp = surface.parse_expiry_date(parts[1])
        if exp is None or exp < today:
            continue
        gap = abs((exp - today).days - int(dte))
        if best_gap is None or gap < best_gap:
            best, best_gap = instr, gap
    return best


def _basis_apr_history(asset: str, dte: int, days: int) -> pd.Series | None:
    """
    Annualised basis history for the dated future nearest ``dte``.

    Computed as ``((1 + (F - S)/S) ** (365 / dte_at_bar) - 1) * 100`` from
    Deribit candles, so the DTE decays through the series exactly as the real
    instrument does.
    """
    instr = _future_instrument(asset, dte)
    if not instr:
        return None
    cfg = surface.acfg(asset)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(days * 24 * 3600 * 1000)
    fut = deribit.get_tradingview_ohlc(instr, "1D", start_ms, end_ms)
    perp = deribit.get_tradingview_ohlc(cfg["perp"], "1D", start_ms, end_ms)
    if fut is None or perp is None or fut.empty or perp.empty:
        return None

    parts = instr.split("-")
    exp = surface.parse_expiry_date(parts[1]) if len(parts) >= 2 else None
    if exp is None:
        return None

    fut = fut.set_index(pd.to_datetime(fut["timestamp"], utc=True))["close"]
    perp = perp.set_index(pd.to_datetime(perp["timestamp"], utc=True))["close"]
    idx = fut.index.intersection(perp.index)
    if len(idx) < 2:
        return None
    f, s = fut.reindex(idx), perp.reindex(idx)
    basis_dec = (f - s) / s.replace(0, np.nan)
    out = {}
    for ts, bd in basis_dec.items():
        if not _finite(bd):
            continue
        d = (exp - ts.date()).days
        if d <= 0:
            continue
        base = 1.0 + float(bd)
        if base <= 0:
            continue
        out[ts] = (base ** (365.0 / d) - 1.0) * 100.0
    s_out = pd.Series(out).sort_index()
    return s_out if len(s_out) >= 2 else None


def cmd_basis_run(asset: str, lookback_days: int = 90, **kwargs):
    """Futures basis term structure (1Y APR) with high/low/percentile context."""
    cfg = surface.acfg(asset)
    spot = deribit.get_index_price(cfg["index"])
    if not spot:
        return None, None, "No basis data available."
    perp = surface.perp_mark(asset)
    futures = surface.future_prices_by_dte(asset)

    today = _now().date()
    rows = []
    if perp is not None:
        rows.append({
            "Token": asset, "Expiry": "PERPETUAL", "DTE": 0,
            "Basis Low %": np.nan,
            "Basis % (1Y APR)": round((perp - spot) / spot * 100, 3),
            "Basis High %": np.nan, "P": np.nan,
            "Basis $": round(perp - spot, 2), "Mid Price": round(perp, 2),
        })

    for dte in sorted(futures):
        if dte <= 0:
            continue
        fut = futures[dte]
        basis_dec = (fut - spot) / spot
        apr = ((1.0 + basis_dec) ** (365.0 / dte) - 1.0) * 100.0
        hist = _basis_apr_history(asset, dte, int(lookback_days))
        lo = hi = pct = np.nan
        if hist is not None and len(hist) >= 2:
            lo, hi = float(hist.min()), float(hist.max())
            pct = float(100.0 * (hist <= apr).mean())
        rows.append({
            "Token": asset,
            "Expiry": (today + timedelta(days=int(dte))).strftime("%d-%b-%Y"),
            "DTE": int(dte),
            "Basis Low %": round(lo, 3) if _finite(lo) else np.nan,
            "Basis % (1Y APR)": round(apr, 3),
            "Basis High %": round(hi, 3) if _finite(hi) else np.nan,
            "P": round(pct) if _finite(pct) else np.nan,
            "Basis $": round(fut - spot, 2),
            "Mid Price": round(fut, 2),
        })

    if not rows:
        return None, None, "No basis data."
    df = pd.DataFrame(rows)

    lead = [f"**{asset} SPOT** ${spot:,.{cfg['price_dp']}f}"]
    if perp is not None:
        lead.append(f"**PERP** {((perp - spot) / spot * 100):+.3f}%")
    dated = df[df["DTE"] > 0]
    if not dated.empty:
        longest = dated.loc[dated["DTE"].idxmax()]
        lead.append(f"**{int(longest['DTE'])}d APR** "
                    f"{float(longest['Basis % (1Y APR)']):+.2f}%")
    header = "   |   ".join(lead)

    y_vals = df["Basis % (1Y APR)"].astype(float).values
    fig = go.Figure(go.Bar(
        x=df["Expiry"].astype(str).tolist(), y=y_vals,
        name="Basis % (1Y APR)",
        marker_color=[GREEN if v >= 0 else RED for v in y_vals],
        text=[f"{v:+.2f}%" for v in y_vals], textposition="outside",
        textfont=dict(size=11)))
    fig.update_layout(
        title=f"{asset} Basis (1Y APR) by Expiry",
        xaxis_title="Expiry", yaxis_title="Basis % (1Y APR)",
        xaxis=dict(type="category", tickangle=-45),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="rgba(0,0,0,0.3)"),
        showlegend=False, height=400, template=TEMPLATE, margin=dict(b=120))
    finalize(fig, legend_rows=0)
    return fig, df, header


def cmd_intraday_basis(asset: str, dte: int | None = None, **kwargs):
    """Today's USD basis for one dated future, with the perp overlaid."""
    target = int(dte) if dte is not None else 30
    resolved = surface.resolve_dte(asset, target)
    instr = _future_instrument(asset, resolved)
    if not instr:
        return None, None, "No basis data for this expiry."

    cfg = surface.acfg(asset)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - 24 * 3600 * 1000
    fut = deribit.get_tradingview_ohlc(instr, "15", start_ms, end_ms)
    perp = deribit.get_tradingview_ohlc(cfg["perp"], "15", start_ms, end_ms)
    if fut is None or perp is None or fut.empty or perp.empty:
        return None, None, "No intraday basis data."

    fut = fut.set_index(pd.to_datetime(fut["timestamp"], utc=True))["close"]
    perp_s = perp.set_index(pd.to_datetime(perp["timestamp"], utc=True))["close"]
    idx = fut.index.intersection(perp_s.index)
    if len(idx) < 2:
        return None, None, "No intraday basis data."
    basis = (fut.reindex(idx) - perp_s.reindex(idx)).dropna()
    if basis.empty:
        return None, None, "No basis."

    expiry_str = (_now().date() + timedelta(days=resolved)).strftime("%d-%b-%Y")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=to_local(basis.index), y=basis.values,
                            name="USD Basis", mode="lines+markers",
                            line=dict(color=NAVY, shape="spline")))
    fig.add_trace(go.Scatter(x=to_local(idx), y=perp_s.reindex(idx).values,
                            name="Perp", mode="lines", yaxis="y2",
                            line=dict(color=ORANGE, width=1.5, dash="dot",
                                      shape="spline"), connectgaps=False))
    fig.update_layout(
        title=f"{asset} {expiry_str} Basis", xaxis_title=XAXIS_TIME,
        yaxis=dict(title="USD Basis", tickprefix="$"),
        yaxis2=dict(overlaying="y", side="right", title="Perp",
                    showgrid=False, tickformat=",.0f"),
        height=450, template=TEMPLATE,
        xaxis=dict(type="date", tickformat=TIME_TICKFORMAT, tickangle=-90),
    )
    finalize(fig, legend_rows=1)
    return fig, None, None


# ===========================================================================
# Realized vol
# ===========================================================================

def cmd_rv_plot(asset: str, **kwargs):
    """Parkinson realized vol (3/7/30/90d) with spot bars behind."""
    ohlc = history.perp_ohlc(asset, days=140, resolution="1D")
    if ohlc is None or ohlc.empty:
        return None, None, "OHLC data not available from Deribit."
    if "high" not in ohlc or "low" not in ohlc:
        return None, None, "OHLC data missing high/low."

    fig = go.Figure()
    if "close" in ohlc:
        closes = pd.to_numeric(ohlc["close"], errors="coerce").dropna()
        if not closes.empty:
            fig.add_trace(go.Bar(
                x=to_local(ohlc.index), y=ohlc["close"].values, name="Spot",
                marker=dict(color="rgba(130,140,155,0.30)", line=dict(width=0)),
                yaxis="y2", hovertemplate="Spot %{y:$,.0f}<extra></extra>"))
            fig.update_layout(yaxis2=dict(
                title=dict(text="Spot (USD)", font=dict(color="#5A6675")),
                overlaying="y", side="right", showgrid=False,
                range=[float(closes.min()) * 0.90, float(closes.max()) * 1.05],
                tickformat="$,.0f", tickfont=dict(color="#5A6675")))

    for win, name, color in ((3, "3d RV", GREEN), (7, "7d RV", NAVY),
                             (30, "30d RV", BLUE), (90, "90d RV", "#888888")):
        s = history.parkinson_rv(ohlc, win).dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=to_local(s.index), y=s.values, name=name,
                                line=dict(color=color, shape="spline")))

    fig.update_layout(
        title=f"{asset} Realized Volatility (Parkinson)",
        xaxis_title=XAXIS_TIME, yaxis_title="Realized Volatility (RV)",
        xaxis=dict(type="date", tickformat=TIME_TICKFORMAT, tickangle=-90),
        height=450, template=TEMPLATE, barmode="overlay",
    )
    finalize(fig, legend_rows=2)
    return fig, None, None


# ===========================================================================
# Funding
# ===========================================================================

def cmd_plot_funding_rates(asset: str, days: int = 30, **kwargs):
    """Deribit perpetual funding history, annualised to APR."""
    cfg = surface.acfg(asset)
    end_ms = int(time.time() * 1000)
    frames = []
    # The endpoint caps the window, so walk it in ~7-day chunks.
    chunk_ms = 7 * 24 * 3600 * 1000
    cursor = end_ms - int(days * 24 * 3600 * 1000)
    while cursor < end_ms:
        stop = min(cursor + chunk_ms, end_ms)
        df = deribit.get_funding_history(cfg["perp"], start_ms=cursor, end_ms=stop)
        if df is not None and not df.empty:
            frames.append(df)
        cursor = stop
    if not frames:
        return None, None, "No funding rate data from Deribit."

    df = pd.concat(frames, ignore_index=True)
    if "timestamp" not in df.columns:
        return None, None, "No funding rate data from Deribit."
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()

    if "interest_8h" in df.columns:
        apr = pd.to_numeric(df["interest_8h"], errors="coerce") * 3 * 365 * 100.0
    elif "interest_1h" in df.columns:
        apr = pd.to_numeric(df["interest_1h"], errors="coerce") * 24 * 365 * 100.0
    else:
        return None, None, "No funding rate data from Deribit."
    apr = apr.dropna()
    if apr.empty:
        return None, None, "No funding rate data from Deribit."

    fig = go.Figure(go.Scatter(x=to_local(apr.index), y=apr.values, name="Deribit",
                              mode="lines",
                              line=dict(color="#7B68EE", shape="spline")))
    last_x, last_y = to_local(apr.index)[-1], float(apr.iloc[-1])
    fig.update_layout(
        title=f"{asset} Historical Funding Rate (Deribit Perpetual)",
        xaxis_title=XAXIS_TIME, yaxis_title="Funding Rate (Yearly APR %)",
        xaxis=dict(type="date", tickformat=TIME_TICKFORMAT, tickangle=-90),
        yaxis=dict(ticksuffix="%"), height=450, template=TEMPLATE,
        annotations=[dict(x=last_x, y=last_y, text=f"Deribit<br>{last_y:+.2f}%",
                          showarrow=True, arrowhead=2, arrowsize=0.8, ax=40, ay=0,
                          xref="x", yref="y", font=dict(size=10, color="#7B68EE"),
                          bgcolor="rgba(255,255,255,0.8)", borderpad=4)])
    finalize(fig, legend_rows=1)
    return fig, None, None


# ===========================================================================
# Block trades
# ===========================================================================

def _fetch_trades_paginated(currency: str, start_ms: int, end_ms: int,
                            max_pages: int = 25) -> list[dict]:
    trades: list[dict] = []
    cursor = start_ms
    for _ in range(max_pages):
        result = deribit._request("get_last_trades_by_currency_and_time", {
            "currency": currency, "kind": "option",
            "start_timestamp": cursor, "end_timestamp": end_ms,
            "count": 1000, "sorting": "asc",
        }, ttl=60)
        if not result:
            break
        page = result.get("trades", []) if isinstance(result, dict) else result
        if not page:
            break
        trades.extend(page)
        if isinstance(result, dict) and not result.get("has_more"):
            break
        try:
            cursor = int(page[-1]["timestamp"]) + 1
        except (KeyError, TypeError, ValueError):
            break
        if cursor >= end_ms:
            break
    return trades


def cmd_block_trades_summary(asset: str, **kwargs):
    """24h option flow aggregated by expiry, with dollar Greeks."""
    cfg = surface.acfg(asset)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - 24 * 3600 * 1000
    cols = ["Expiry", "Delta", "Vega", "Gamma (1%)", "Net Puts", "Net Calls",
            "Gross Notional"]
    try:
        trades = _fetch_trades_paginated(cfg["deribit_ccy"], start_ms, end_ms)
    except Exception as exc:
        return None, None, f"Failed to fetch trades: {exc}"
    if not trades:
        return None, pd.DataFrame(columns=cols), None

    fallback_spot = deribit.get_index_price(cfg["index"]) or 0.0
    min_size = float(cfg["min_block"])
    prefix = cfg["deribit_prefix"] + "-"
    contract_size = float(cfg.get("contract_size", 1.0))

    recs = []
    for t in trades:
        # Skip trades Deribit tags as forced liquidations ("liquidation":
        # "M"/"T"/"MT") - these aren't genuine negotiated block trades, and
        # a cascading liquidation can fire off many of them in a burst,
        # which inflates Net Puts/Calls and Gross Notional with noise
        # rather than real desk flow. Same fix as the ALL-tab scatter grid
        # on the Block Trades page (see CLAUDE.md session log).
        if t.get("liquidation"):
            continue
        instr = t.get("instrument_name") or ""
        if not instr or "-" not in instr:
            continue
        if cfg["style"] == "linear" and not instr.startswith(prefix):
            continue
        try:
            amount_raw = float(t.get("amount") or 0.0)
        except (TypeError, ValueError):
            continue
        if abs(amount_raw) < min_size:
            continue
        parts = instr.split("-")
        if len(parts) < 4:
            continue
        exp = surface.parse_expiry_date(parts[1])
        strike = surface.parse_strike(parts[2])
        if exp is None or strike is None or strike <= 0:
            continue
        is_call = not parts[-1].upper().startswith("P")
        direction = str(t.get("direction") or t.get("side") or "buy").lower()
        amount = amount_raw * (1 if direction == "buy" else -1)

        iv = t.get("mark_iv", t.get("iv"))
        try:
            iv = float(iv) / 100.0 if iv is not None else 0.65
        except (TypeError, ValueError):
            iv = 0.65
        try:
            spot = float(t.get("index_price")) if t.get("index_price") else fallback_spot
        except (TypeError, ValueError):
            spot = fallback_spot
        if not spot:
            continue

        try:
            trade_ts = pd.to_datetime(int(t["timestamp"]), unit="ms", utc=True)
        except (KeyError, TypeError, ValueError):
            trade_ts = pd.Timestamp(_now())
        expiry_ts = pd.Timestamp(year=exp.year, month=exp.month, day=exp.day,
                                 hour=8, tz="UTC")
        sec_per_year = 365.25 * 24 * 3600
        T = max((expiry_ts - trade_ts).total_seconds() / sec_per_year,
                1.0 / sec_per_year)

        recs.append({"expiry": exp, "strike": strike, "is_call": is_call,
                     "amount": amount, "amount_raw": amount_raw, "iv": iv,
                     "spot": spot, "T": T})

    if not recs:
        return None, pd.DataFrame(columns=cols), None

    raw = pd.DataFrame(recs)
    safe_iv = np.where(raw["iv"] > 1e-8, raw["iv"], 1e-8)
    sqrt_T = np.sqrt(raw["T"].values)
    log_ratio = np.log(raw["spot"].values / raw["strike"].values)
    d1 = (log_ratio + 0.5 * safe_iv * safe_iv * raw["T"].values) / (safe_iv * sqrt_T)
    d1 = np.where(np.isfinite(d1), d1, 0.0)

    delta_per = np.where(raw["is_call"], norm.cdf(d1), norm.cdf(d1) - 1.0)
    gamma_per = norm.pdf(d1) / (raw["spot"].values * safe_iv * sqrt_T)
    vega_per = raw["spot"].values * norm.pdf(d1) * sqrt_T * 0.01

    raw["dollar_delta"] = delta_per * raw["amount"] * raw["spot"] * contract_size
    raw["dollar_gamma_1pct"] = (gamma_per * raw["amount"] *
                                (raw["spot"] ** 2) * contract_size * 0.01)
    raw["dollar_vega"] = vega_per * raw["amount"] * contract_size
    raw["net_puts"] = np.where(raw["is_call"], 0, raw["amount"])
    raw["net_calls"] = np.where(raw["is_call"], raw["amount"], 0)
    raw["gross_notional"] = raw["amount_raw"].abs() * raw["strike"] * contract_size

    agg = raw.groupby("expiry").agg({
        "dollar_delta": "sum", "dollar_vega": "sum", "dollar_gamma_1pct": "sum",
        "net_puts": "sum", "net_calls": "sum", "gross_notional": "sum",
    }).reset_index().sort_values("expiry")

    out = pd.DataFrame({
        "Expiry": [d.strftime("%d-%b-%Y") for d in agg["expiry"]],
        "Delta": agg["dollar_delta"].round(0).values,
        "Vega": agg["dollar_vega"].round(0).values,
        "Gamma (1%)": agg["dollar_gamma_1pct"].round(0).values,
        "Net Puts": agg["net_puts"].round(0).astype(int).values,
        "Net Calls": agg["net_calls"].round(0).astype(int).values,
        "Gross Notional": agg["gross_notional"].round(0).astype(int).values,
    })
    total = pd.DataFrame([{
        "Expiry": "Total",
        "Delta": out["Delta"].sum(), "Vega": out["Vega"].sum(),
        "Gamma (1%)": out["Gamma (1%)"].sum(),
        "Net Puts": int(out["Net Puts"].sum()),
        "Net Calls": int(out["Net Calls"].sum()),
        "Gross Notional": int(out["Gross Notional"].sum()),
    }])
    out = pd.concat([out, total], ignore_index=True)

    spot = fallback_spot
    header = (f"**{asset} 24h OPTION FLOW**   |   "
              f"**SPOT** ${spot:,.{cfg['price_dp']}f}   |   "
              f"**MIN SIZE** {min_size:g}") if spot else None
    return None, out, header


# ===========================================================================
# Moon phases
# ===========================================================================

def _moon_phase_events(days_back: int = 31, days_forward: int = 0):
    now = _now()
    days_since = (now - MOON_EPOCH).total_seconds() / 86400.0
    lunation = days_since % LUNAR_CYCLE_DAYS
    last_new = now - timedelta(days=lunation)
    offsets = [(0, "New Moon"), (7.38, "First Quarter"),
               (14.76, "Full Moon"), (22.14, "Last Quarter")]
    start, end = now - timedelta(days=days_back), now + timedelta(days=days_forward)
    back = max(2, int(days_back // int(LUNAR_CYCLE_DAYS)) + 1)
    fwd = max(2, int(days_forward // int(LUNAR_CYCLE_DAYS)) + 2)
    events = []
    for cycle in range(-back, fwd):
        base = last_new + timedelta(days=cycle * LUNAR_CYCLE_DAYS)
        for off, name in offsets:
            t = base + timedelta(days=off)
            if start <= t <= end:
                events.append((t, name))
    return sorted(events)


def cmd_moonphase(asset: str, **kwargs):
    """Perp price over 3 months with lunar phase bands and full-moon stats."""
    days_back = 90
    ohlc = history.perp_ohlc(asset, days=days_back + 10, resolution="1D")
    if ohlc is None or ohlc.empty or "close" not in ohlc:
        return None, None, "Could not load price data for moon phase chart."
    px = pd.to_numeric(ohlc["close"], errors="coerce").dropna()
    cutoff = _now() - timedelta(days=days_back)
    px = px[px.index >= cutoff]
    if px.empty:
        return None, None, "No price data in range."

    now = _now()
    days_since = (now - MOON_EPOCH).total_seconds() / 86400.0
    last_new = now - timedelta(days=days_since % LUNAR_CYCLE_DAYS)
    next_full = last_new + timedelta(days=14.76)
    while next_full <= now:
        next_full += timedelta(days=LUNAR_CYCLE_DAYS)
    days_forward = max(0, int(math.ceil((next_full - now).total_seconds() / 86400.0)))
    events = _moon_phase_events(days_back=days_back, days_forward=days_forward)

    y_min, y_max = float(px.min()), float(px.max())
    y_span = max(1e-9, y_max - y_min)

    fig = go.Figure(go.Scatter(
        x=to_local(px.index), y=px.values, name="Perp close",
        line=dict(color=NAVY, width=2, shape="spline")))

    phase_colors = {"New Moon": "#1a1a1a", "First Quarter": "#4a90d9",
                    "Full Moon": "#f5d76e", "Last Quarter": "#e67e22"}
    phase_emojis = {"New Moon": "🌑", "First Quarter": "🌓",
                    "Full Moon": "🌕", "Last Quarter": "🌗"}
    shapes, annotations = [], []

    def _x(ts):
        return to_local_ts(ts).isoformat()

    for t, name in events:
        band = 2.0 if name == "Full Moon" else 1.5
        color = phase_colors.get(name, GREY)
        opacity = 0.2 if name == "Full Moon" else 0.12
        rgb = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        shapes.append(dict(type="rect", xref="x", yref="y",
                           x0=_x(t - timedelta(days=band)),
                           x1=_x(t + timedelta(days=band)),
                           y0=y_min, y1=y_max,
                           fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{opacity})",
                           line=dict(width=0)))
        shapes.append(dict(type="line", xref="x", yref="y",
                           x0=_x(t), x1=_x(t), y0=y_min, y1=y_max,
                           line=dict(color=color, width=1.5, dash="dot")))
        annotations.append(dict(
            x=_x(t), y=y_max + 0.02 * y_span, xref="x", yref="y",
            text=f"{phase_emojis.get(name, '')} {name}", showarrow=False,
            font=dict(size=12 if name == "Full Moon" else 10, color=color),
            xanchor="center"))

    # Full-moon statistics.
    N = 3
    full_moons = [t for t, n in events if n == "Full Moon"]
    daily = px.copy()

    def _px_near(ts, max_gap_days=2.5):
        try:
            pos = daily.index.get_indexer([pd.Timestamp(ts)], method="nearest")[0]
        except Exception:
            return np.nan
        pos = max(0, min(int(pos), len(daily) - 1))
        if abs((daily.index[pos] - pd.Timestamp(ts)).total_seconds()) > max_gap_days * 86400:
            return np.nan
        return float(daily.iloc[pos])

    win_returns, fm_to_fm = [], []
    for fm in full_moons:
        p0, p1 = _px_near(fm - timedelta(days=N)), _px_near(fm + timedelta(days=N))
        if _finite(p0) and _finite(p1) and p0 > 0:
            win_returns.append((p1 / p0 - 1.0) * 100.0)
    for a, b in zip(full_moons, full_moons[1:]):
        pa, pb = _px_near(a), _px_near(b)
        if _finite(pa) and _finite(pb) and pa > 0:
            fm_to_fm.append((pb / pa - 1.0) * 100.0)

    abs_rets = daily.pct_change().abs() * 100.0
    near = pd.Series(False, index=daily.index)
    for fm in full_moons:
        tsp = pd.Timestamp(fm)
        near |= ((daily.index >= tsp - timedelta(days=N)) &
                 (daily.index <= tsp + timedelta(days=N)))
    mag_near = float(abs_rets[near].mean()) if near.any() else np.nan
    mag_rest = float(abs_rets[~near].mean()) if (~near).any() else np.nan

    W = 2 * N + 1
    rmax = daily.rolling(W, center=True, min_periods=3).max()
    rmin = daily.rolling(W, center=True, min_periods=3).min()
    is_ext = (daily >= rmax - 1e-9) | (daily <= rmin + 1e-9)
    ext_idx = daily.index[is_ext]
    hits = n_base = 0
    for fm in full_moons:
        tsp = pd.Timestamp(fm)
        lo, hi = tsp - timedelta(days=N), tsp + timedelta(days=N)
        if daily.index.min() <= tsp <= daily.index.max():
            n_base += 1
            if ((ext_idx >= lo) & (ext_idx <= hi)).any():
                hits += 1

    lines = [f"<b>Full-moon stats</b>  <span style='color:#888'>"
             f"(±{N}d, n={len(win_returns)})</span>"]
    if win_returns:
        lines.append(f"Avg ±{N}d return: <b>{np.mean(win_returns):+.2f}%</b>  "
                     f"(med {np.median(win_returns):+.2f}%)")
    if fm_to_fm:
        lines.append(f"Full→next-full: <b>{np.mean(fm_to_fm):+.2f}%</b>")
    if _finite(mag_near) and _finite(mag_rest):
        lines.append(f"|daily move| near FM <b>{mag_near:.2f}%</b> "
                     f"vs rest {mag_rest:.2f}%")
    if n_base:
        lines.append(f"Local extreme near FM: <b>{hits}/{n_base}</b>")
    lines.append("<span style='color:#999'>small sample — indicative only</span>")
    if len(lines) > 2:
        annotations.append(dict(
            xref="paper", yref="paper", x=0.008, y=0.02, xanchor="left",
            yanchor="bottom", text="<br>".join(lines), showarrow=False,
            align="left", font=dict(size=10.5, color="#13314F"),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#f5d76e",
            borderwidth=1, borderpad=6))
    annotations.append(dict(
        xref="paper", yref="paper", x=0.5, y=0.96, showarrow=False,
        text="Full Moon band (wider): often coincides with local price extremes",
        font=dict(size=11, color="#666")))

    fig.update_layout(
        title=dict(text=f"{asset} price vs moon phases (past 3 months)",
                   x=0.5, xanchor="center"),
        xaxis=dict(title=XAXIS_TIME, type="date", tickformat="%b %d", tickangle=-90),
        yaxis=dict(title="Price (USD)", side="left"),
        shapes=shapes, annotations=annotations, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=420, margin=dict(t=100, b=50, l=60, r=40),
        template=TEMPLATE, hovermode="x unified")
    finalize(fig, legend_rows=1, keep_margin=True)
    return fig, None, None
