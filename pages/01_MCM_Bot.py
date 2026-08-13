import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
MCM Bot — Command-based options analytics interface.

Deribit-only implementation providing vol term structure, skew, smile,
block trades, basis, realized vol, funding rates, and DVOL snapshots.
"""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.deribit import (
    get_index_price,
    get_option_chain,
    get_tradingview_ohlc,
    get_dvol,
    get_trades,
    get_funding_history,
    get_expirations,
)
from lib.constants import ASSET_CONFIG, ASSETS, ASSET_COLORS, PLOTLY_LAYOUT
from lib.vol_math import (
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
    compute_rv_matrix,
    close_to_close_vol,
    parkinson_vol,
    garman_klass_vol,
    yang_zhang_vol,
    rogers_satchell_vol,
)
from lib.instruments import parse_instrument
from lib.telegram import send_message, send_photo, is_configured

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="MCM Bot", page_icon="🤖", layout="wide")

COMMANDS = [
    "Dashboard",
    "Vol Term Structure",
    "Skew Term Structure",
    "Vol Smile",
    "Block Trades",
    "Basis Run",
    "Realized Vol",
    "Funding Rate",
    "DVol Snapshot",
]


# ---------------------------------------------------------------------------
# Cached data fetchers (defined before sidebar so they can be called there)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def _fetch_option_chain(currency: str) -> list[dict] | None:
    return get_option_chain(currency, "option")


@st.cache_data(ttl=120)
def _fetch_future_chain(currency: str) -> list[dict] | None:
    return get_option_chain(currency, "future")


@st.cache_data(ttl=30)
def _fetch_spot(index_name: str) -> float | None:
    return get_index_price(index_name)


@st.cache_data(ttl=120)
def _fetch_expirations(asset_key: str) -> list[int]:
    c = ASSET_CONFIG[asset_key]
    result = get_expirations(c["deribit_ccy"], "option")
    if not result:
        return []
    # Filter to only options with matching prefix
    prefix = c["deribit_prefix"]
    instruments = get_option_chain(c["deribit_ccy"], "option")
    if instruments:
        valid_expiries = set()
        for inst in instruments:
            if inst["instrument_name"].startswith(prefix + "-"):
                valid_expiries.add(inst.get("expiration_timestamp", 0))
        result = sorted(ts for ts in result if ts in valid_expiries)
    return result or []


@st.cache_data(ttl=60)
def _fetch_ohlc(instrument: str, resolution: str, days: int) -> pd.DataFrame | None:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    return get_tradingview_ohlc(instrument, resolution, start_ms, end_ms)


@st.cache_data(ttl=300)
def _fetch_dvol(currency: str, days: int = 90) -> pd.DataFrame | None:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    return get_dvol(currency, "1D", start_ms, end_ms)


@st.cache_data(ttl=60)
def _fetch_trades(currency: str, kind: str = "option") -> list[dict] | None:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - 24 * 3600 * 1000
    return get_trades(currency, kind, start_ms, end_ms, count=1000)


@st.cache_data(ttl=300)
def _fetch_funding(instrument: str, days: int = 30) -> pd.DataFrame | None:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    return get_funding_history(instrument, start_ms, end_ms)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🤖 MCM Bot")
    asset = st.selectbox("Asset", ASSETS, index=0)
    command = st.selectbox("Command", COMMANDS, index=0)

    cfg = ASSET_CONFIG[asset]
    color = ASSET_COLORS[asset]

    # Expiry picker for commands that need it
    selected_expiry_ts = None
    if command in ("Vol Smile",):
        expiries = _fetch_expirations(asset)
        if expiries:
            expiry_labels = [
                datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%d%b%y").upper()
                for ts in expiries
            ]
            sel_idx = st.selectbox(
                "Expiry",
                range(len(expiry_labels)),
                format_func=lambda i: expiry_labels[i],
                index=min(2, len(expiry_labels) - 1),
            )
            selected_expiry_ts = expiries[sel_idx]

    st.divider()
    st.caption(f"Data: Deribit Public API | Asset: {asset}")
    if is_configured():
        st.caption("📡 Telegram: Connected")
    else:
        st.caption("📡 Telegram: Not configured")


# ---------------------------------------------------------------------------
# Helper: create figure with standard layout
# ---------------------------------------------------------------------------

def _make_fig(title: str = "", height: int = 450) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=height)
    return fig


def _fig_to_bytes(fig: go.Figure) -> bytes:
    """Convert plotly figure to PNG bytes for Telegram."""
    return fig.to_image(format="png", width=1200, height=600, scale=2)


def _telegram_button(key: str, text: str = "", fig: go.Figure | None = None,
                     caption: str = ""):
    """Render a Telegram blast button."""
    if not is_configured():
        return
    if st.button("📡 Send to Telegram", key=f"tg_{key}"):
        success = False
        if fig is not None:
            try:
                img = _fig_to_bytes(fig)
                success = send_photo(img, caption=caption or key)
            except Exception:
                success = send_message(text or caption or key)
        elif text:
            success = send_message(text)
        if success:
            st.success("Sent to Telegram!")
        else:
            st.error("Failed to send.")


# ---------------------------------------------------------------------------
# Command: Vol Term Structure
# ---------------------------------------------------------------------------

def cmd_vol_term_structure(asset: str, cfg: dict, color: str) -> go.Figure | None:
    st.subheader(f"📈 {asset} — Vol Term Structure")

    spot = _fetch_spot(cfg["index"])
    chain = _fetch_option_chain(cfg["deribit_ccy"])

    if not spot or not chain:
        st.error("Failed to fetch option chain or spot price.")
        return None

    prefix = cfg["deribit_prefix"]
    now_ms = time.time() * 1000

    # Group by expiry, find ATM IV
    expiry_data = {}
    for inst in chain:
        name = inst.get("instrument_name", "")
        if not name.startswith(prefix + "-"):
            continue
        mark_iv = inst.get("mark_iv")
        if not mark_iv or mark_iv == 0:
            continue
        parts = name.split("-")
        if len(parts) < 4:
            continue
        try:
            strike = float(parts[2].replace("d", "."))
        except ValueError:
            continue

        expiry_ts = inst.get("expiration_timestamp", 0)
        moneyness = abs(strike - spot) / spot

        if expiry_ts not in expiry_data or moneyness < expiry_data[expiry_ts]["moneyness"]:
            expiry_data[expiry_ts] = {
                "atm_iv": mark_iv,
                "moneyness": moneyness,
                "expiry_ts": expiry_ts,
            }

    if not expiry_data:
        st.warning("No ATM options found in the chain.")
        return None

    rows = []
    for ts, d in sorted(expiry_data.items()):
        dte = max((ts - now_ms) / (1000 * 86400), 0.01)
        expiry_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%d%b%y").upper()
        rows.append({"expiry": expiry_str, "dte": round(dte, 1), "atm_iv": round(d["atm_iv"], 2)})

    df = pd.DataFrame(rows)

    # Chart
    fig = _make_fig(f"{asset} ATM IV Term Structure")
    fig.add_trace(go.Scatter(
        x=df["dte"], y=df["atm_iv"],
        mode="lines+markers",
        name="ATM IV",
        line=dict(color=color, width=2),
        marker=dict(size=8),
    ))
    fig.update_xaxes(title_text="Days to Expiry")
    fig.update_yaxes(title_text="Implied Volatility (%)")
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"Spot: ${spot:,.{cfg['price_dp']}f}",
        showarrow=False, font=dict(size=12, color=color),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    with st.expander("Data Table"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    _telegram_button("vol_ts", fig=fig, caption=f"{asset} Vol Term Structure | Spot ${spot:,.0f}")
    return fig


# ---------------------------------------------------------------------------
# Command: Skew Term Structure
# ---------------------------------------------------------------------------

def cmd_skew_term_structure(asset: str, cfg: dict, color: str) -> go.Figure | None:
    st.subheader(f"📊 {asset} — Skew Term Structure (25Δ Risk Reversal)")

    spot = _fetch_spot(cfg["index"])
    chain = _fetch_option_chain(cfg["deribit_ccy"])

    if not spot or not chain:
        st.error("Failed to fetch data.")
        return None

    prefix = cfg["deribit_prefix"]
    now_ms = time.time() * 1000

    # Group instruments by expiry
    by_expiry: dict[int, list[dict]] = {}
    for inst in chain:
        name = inst.get("instrument_name", "")
        if not name.startswith(prefix + "-"):
            continue
        mark_iv = inst.get("mark_iv")
        if not mark_iv or mark_iv == 0:
            continue
        parts = name.split("-")
        if len(parts) < 4:
            continue
        try:
            strike = float(parts[2].replace("d", "."))
        except ValueError:
            continue

        expiry_ts = inst.get("expiration_timestamp", 0)
        dte = max((expiry_ts - now_ms) / (1000 * 86400), 0.01)
        tte_years = dte / 365.0
        kind = parts[3]  # C or P
        iv_dec = mark_iv / 100.0

        delta = bs_delta(spot, strike, tte_years, iv_dec, kind)

        entry = {
            "strike": strike,
            "kind": kind,
            "mark_iv": mark_iv,
            "delta": delta,
            "expiry_ts": expiry_ts,
            "dte": dte,
        }
        by_expiry.setdefault(expiry_ts, []).append(entry)

    # For each expiry find 25-delta call and put
    rows = []
    for expiry_ts, instruments in sorted(by_expiry.items()):
        dte = instruments[0]["dte"]
        if dte < 1:
            continue  # Skip near-expiry

        calls = [i for i in instruments if i["kind"] == "C"]
        puts = [i for i in instruments if i["kind"] == "P"]

        if not calls or not puts:
            continue

        # Find call closest to +0.25 delta
        call_25 = min(calls, key=lambda x: abs(x["delta"] - 0.25))
        # Find put closest to -0.25 delta
        put_25 = min(puts, key=lambda x: abs(x["delta"] + 0.25))

        # Risk reversal = 25d call IV - 25d put IV
        rr = call_25["mark_iv"] - put_25["mark_iv"]
        expiry_str = datetime.fromtimestamp(expiry_ts / 1000, tz=timezone.utc).strftime("%d%b%y").upper()

        rows.append({
            "expiry": expiry_str,
            "dte": round(dte, 1),
            "25d_call_iv": round(call_25["mark_iv"], 2),
            "25d_put_iv": round(put_25["mark_iv"], 2),
            "risk_reversal": round(rr, 2),
        })

    if not rows:
        st.warning("Could not compute 25-delta skew for any expiry.")
        return None

    df = pd.DataFrame(rows)

    # Bar chart
    fig = _make_fig(f"{asset} 25Δ Risk Reversal by Expiry")
    colors = [color if v >= 0 else "#ef5350" for v in df["risk_reversal"]]
    fig.add_trace(go.Bar(
        x=df["expiry"], y=df["risk_reversal"],
        marker_color=colors,
        name="25Δ RR",
    ))
    fig.update_xaxes(title_text="Expiry")
    fig.update_yaxes(title_text="Risk Reversal (vol pts)")
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Data Table"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    _telegram_button("skew_ts", fig=fig, caption=f"{asset} 25Δ Risk Reversal")
    return fig


# ---------------------------------------------------------------------------
# Command: Vol Smile
# ---------------------------------------------------------------------------

def cmd_vol_smile(asset: str, cfg: dict, color: str,
                  expiry_ts: int | None = None) -> go.Figure | None:
    st.subheader(f"😊 {asset} — Volatility Smile")

    spot = _fetch_spot(cfg["index"])
    chain = _fetch_option_chain(cfg["deribit_ccy"])

    if not spot or not chain:
        st.error("Failed to fetch data.")
        return None

    prefix = cfg["deribit_prefix"]
    now_ms = time.time() * 1000

    # If no expiry selected, pick the one nearest 30 DTE
    if expiry_ts is None:
        all_expiries = set()
        for inst in chain:
            if inst["instrument_name"].startswith(prefix + "-"):
                all_expiries.add(inst.get("expiration_timestamp", 0))
        if not all_expiries:
            st.warning("No expiries found.")
            return None
        expiry_ts = min(all_expiries, key=lambda ts: abs((ts - now_ms) / 86400000 - 30))

    dte = max((expiry_ts - now_ms) / (1000 * 86400), 0.01)
    expiry_str = datetime.fromtimestamp(expiry_ts / 1000, tz=timezone.utc).strftime("%d%b%y").upper()

    calls = []
    puts = []
    for inst in chain:
        name = inst.get("instrument_name", "")
        if not name.startswith(prefix + "-"):
            continue
        mark_iv = inst.get("mark_iv")
        if not mark_iv or mark_iv == 0:
            continue
        if inst.get("expiration_timestamp") != expiry_ts:
            continue
        parts = name.split("-")
        if len(parts) < 4:
            continue
        try:
            strike = float(parts[2].replace("d", "."))
        except ValueError:
            continue
        kind = parts[3]
        entry = {"strike": strike, "iv": mark_iv}
        if kind == "C":
            calls.append(entry)
        else:
            puts.append(entry)

    if not calls and not puts:
        st.warning(f"No options data for expiry {expiry_str}.")
        return None

    fig = _make_fig(f"{asset} Vol Smile — {expiry_str} ({dte:.0f} DTE)")

    if calls:
        df_c = pd.DataFrame(calls).sort_values("strike")
        fig.add_trace(go.Scatter(
            x=df_c["strike"], y=df_c["iv"],
            mode="lines+markers", name="Calls",
            line=dict(color="#4caf50", width=2),
            marker=dict(size=5),
        ))
    if puts:
        df_p = pd.DataFrame(puts).sort_values("strike")
        fig.add_trace(go.Scatter(
            x=df_p["strike"], y=df_p["iv"],
            mode="lines+markers", name="Puts",
            line=dict(color="#ef5350", width=2),
            marker=dict(size=5),
        ))

    # Spot vertical line
    fig.add_vline(x=spot, line_dash="dash", line_color=color, opacity=0.7,
                  annotation_text=f"Spot ${spot:,.{cfg['price_dp']}f}")
    fig.update_xaxes(title_text="Strike")
    fig.update_yaxes(title_text="Implied Volatility (%)")
    st.plotly_chart(fig, use_container_width=True)

    # Combined table
    with st.expander("Data Table"):
        all_data = []
        for c in (calls or []):
            all_data.append({**c, "type": "Call"})
        for p in (puts or []):
            all_data.append({**p, "type": "Put"})
        st.dataframe(pd.DataFrame(all_data).sort_values("strike"), use_container_width=True,
                     hide_index=True)

    _telegram_button("smile", fig=fig, caption=f"{asset} Smile {expiry_str}")
    return fig


# ---------------------------------------------------------------------------
# Command: Block Trades
# ---------------------------------------------------------------------------

def cmd_block_trades(asset: str, cfg: dict, color: str) -> go.Figure | None:
    st.subheader(f"🧱 {asset} — Block Trades (24h)")

    spot = _fetch_spot(cfg["index"])
    trades = _fetch_trades(cfg["deribit_ccy"], "option")

    if not trades:
        st.info("No trades found in the last 24 hours.")
        return None
    if not spot:
        st.error("Could not fetch spot price.")
        return None

    prefix = cfg["deribit_prefix"]
    min_block = cfg["min_block"]
    now_ms = time.time() * 1000

    blocks = []
    for t in trades:
        name = t.get("instrument_name", "")
        if not name.startswith(prefix + "-"):
            continue
        amount = abs(t.get("amount", 0))
        if amount < min_block:
            continue

        parsed = parse_instrument(name)
        if not parsed:
            continue

        expiry_ts = t.get("expiration_timestamp")
        if not expiry_ts:
            # Derive from parsed instrument
            expiry_ts = int(parsed.expiry_date.timestamp() * 1000)

        dte = max((expiry_ts - now_ms) / (1000 * 86400), 0.01)
        tte_years = dte / 365.0
        strike = parsed.strike
        kind = parsed.kind
        iv = t.get("iv", 0)
        iv_dec = iv / 100.0 if iv else 0.3

        # Compute Greeks
        delta = bs_delta(spot, strike, tte_years, iv_dec, kind)
        gamma = bs_gamma(spot, strike, tte_years, iv_dec)
        vega = bs_vega(spot, strike, tte_years, iv_dec)
        theta = bs_theta(spot, strike, tte_years, iv_dec, kind)

        direction = t.get("direction", "unknown")
        ts_str = datetime.fromtimestamp(
            t.get("timestamp", 0) / 1000, tz=timezone.utc
        ).strftime("%H:%M:%S")

        blocks.append({
            "time": ts_str,
            "instrument": name,
            "direction": direction,
            "amount": amount,
            "price": t.get("price", 0),
            "iv": round(iv, 1),
            "strike": strike,
            "dte": round(dte, 1),
            "delta": round(delta, 3),
            "gamma": round(gamma, 6),
            "vega": round(vega, 2),
            "theta": round(theta, 2),
            "kind": kind,
        })

    if not blocks:
        st.info(f"No block trades (>= {min_block} contracts) in the last 24h.")
        return None

    df = pd.DataFrame(blocks)
    st.metric("Block Count", len(df))

    # Scatter: strike vs DTE, colored by direction
    fig = _make_fig(f"{asset} Blocks — Strike vs DTE")
    for direction, d_color in [("buy", "#4caf50"), ("sell", "#ef5350")]:
        mask = df["direction"] == direction
        if mask.any():
            subset = df[mask]
            fig.add_trace(go.Scatter(
                x=subset["dte"], y=subset["strike"],
                mode="markers",
                name=direction.capitalize(),
                marker=dict(
                    color=d_color,
                    size=np.clip(subset["amount"] / min_block * 6, 6, 30),
                    opacity=0.7,
                ),
                text=subset["instrument"],
                hovertemplate="%{text}<br>DTE: %{x:.0f}<br>Strike: %{y:,.0f}<extra></extra>",
            ))

    fig.add_hline(y=spot, line_dash="dash", line_color=color, opacity=0.5,
                  annotation_text=f"Spot ${spot:,.0f}")
    fig.update_xaxes(title_text="Days to Expiry")
    fig.update_yaxes(title_text="Strike")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Block Trades Table"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    _telegram_button("blocks", fig=fig,
                     caption=f"{asset} Blocks: {len(df)} trades | Spot ${spot:,.0f}")
    return fig


# ---------------------------------------------------------------------------
# Command: Basis Run
# ---------------------------------------------------------------------------

def cmd_basis_run(asset: str, cfg: dict, color: str) -> go.Figure | None:
    st.subheader(f"💰 {asset} — Futures Basis Term Structure")

    spot = _fetch_spot(cfg["index"])
    futures = _fetch_future_chain(cfg["deribit_ccy"])

    if not spot or not futures:
        st.error("Failed to fetch futures chain or spot price.")
        return None

    prefix = cfg["deribit_prefix"]
    now_ms = time.time() * 1000

    rows = []
    for fut in futures:
        name = fut.get("instrument_name", "")
        if not name.startswith(prefix + "-"):
            continue
        # Skip perpetuals
        if "PERPETUAL" in name:
            continue

        mark_price = fut.get("mark_price")
        if not mark_price or mark_price == 0:
            continue

        # For inverse futures, mark_price is in USD. For linear, in USDC.
        # The book_summary mark_price for futures is the actual USD price.
        future_price = mark_price

        expiry_ts = fut.get("expiration_timestamp", 0)
        dte = max((expiry_ts - now_ms) / (1000 * 86400), 0.5)

        # Annualized basis
        basis_pct = ((future_price - spot) / spot) * (365 / dte) * 100

        expiry_str = datetime.fromtimestamp(expiry_ts / 1000, tz=timezone.utc).strftime("%d%b%y").upper()
        rows.append({
            "expiry": expiry_str,
            "dte": round(dte, 1),
            "future_price": round(future_price, cfg["price_dp"]),
            "spot": round(spot, cfg["price_dp"]),
            "premium_pct": round((future_price - spot) / spot * 100, 3),
            "annualized_basis": round(basis_pct, 2),
        })

    if not rows:
        st.warning("No active futures found.")
        return None

    df = pd.DataFrame(rows).sort_values("dte")

    fig = _make_fig(f"{asset} Annualized Futures Basis")
    fig.add_trace(go.Bar(
        x=df["expiry"], y=df["annualized_basis"],
        marker_color=[color if v >= 0 else "#ef5350" for v in df["annualized_basis"]],
        name="Ann. Basis %",
    ))
    fig.update_xaxes(title_text="Expiry")
    fig.update_yaxes(title_text="Annualized Basis (%)")
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Data Table"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    avg_basis = df["annualized_basis"].mean()
    _telegram_button("basis", fig=fig,
                     caption=f"{asset} Basis | Avg Ann: {avg_basis:.1f}% | Spot ${spot:,.0f}")
    return fig


# ---------------------------------------------------------------------------
# Command: Realized Vol
# ---------------------------------------------------------------------------

def cmd_realized_vol(asset: str, cfg: dict, color: str) -> go.Figure | None:
    st.subheader(f"📉 {asset} — Realized Volatility")

    df_ohlc = _fetch_ohlc(cfg["perp"], "1D", 180)
    if df_ohlc is None or len(df_ohlc) < 30:
        st.error("Insufficient OHLC data for RV computation.")
        return None

    # Compute multiple estimators at 30-day window for time series
    windows = [7, 14, 30, 60, 90]

    # Time series chart (30-day window, all estimators)
    fig = _make_fig(f"{asset} Realized Volatility (30d window)")

    estimator_colors = ["#4caf50", "#2196f3", color, "#ff9800", "#9c27b0"]
    estimators = [
        ("Close-to-Close", close_to_close_vol),
        ("Parkinson", parkinson_vol),
        ("Garman-Klass", garman_klass_vol),
        ("Yang-Zhang", yang_zhang_vol),
        ("Rogers-Satchell", rogers_satchell_vol),
    ]

    for i, (name, func) in enumerate(estimators):
        if name == "Close-to-Close":
            series = func(df_ohlc["close"], window=30)
        elif name == "Parkinson":
            series = func(df_ohlc["high"], df_ohlc["low"], window=30)
        else:
            series = func(df_ohlc["open"], df_ohlc["high"], df_ohlc["low"],
                         df_ohlc["close"], window=30)

        fig.add_trace(go.Scatter(
            x=df_ohlc["timestamp"], y=series * 100,
            mode="lines", name=name,
            line=dict(color=estimator_colors[i], width=1.5),
        ))

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Annualized Vol (%)")
    st.plotly_chart(fig, use_container_width=True)

    # RV Matrix
    st.markdown("**Current RV Matrix**")
    rv_matrix = compute_rv_matrix(df_ohlc, windows=windows)
    pivot = rv_matrix.pivot(index="estimator", columns="window", values="value")
    pivot.columns = [f"{w}d" for w in pivot.columns]
    st.dataframe(pivot.round(1), use_container_width=True)

    _telegram_button("rv", fig=fig, caption=f"{asset} Realized Vol (30d)")
    return fig


# ---------------------------------------------------------------------------
# Command: Funding Rate
# ---------------------------------------------------------------------------

def cmd_funding_rate(asset: str, cfg: dict, color: str) -> go.Figure | None:
    st.subheader(f"💸 {asset} — Perpetual Funding Rate")

    df_fund = _fetch_funding(cfg["perp"], 30)
    if df_fund is None or df_fund.empty:
        st.error("No funding data available.")
        return None

    # Ensure proper columns
    if "interest_1h" not in df_fund.columns:
        # Try alternative column name
        if "interest_8h" in df_fund.columns:
            df_fund["interest_1h"] = df_fund["interest_8h"] / 8
        else:
            st.error("Unexpected funding data format.")
            return None

    df_fund = df_fund.sort_values("timestamp").reset_index(drop=True)

    # Annualized rate
    df_fund["annualized"] = df_fund["interest_1h"] * 24 * 365 * 100

    # Rolling averages
    df_fund["rolling_8h"] = df_fund["annualized"].rolling(8, min_periods=1).mean()
    df_fund["rolling_7d"] = df_fund["annualized"].rolling(24 * 7, min_periods=1).mean()

    # Current stats
    current_rate = df_fund["annualized"].iloc[-1]
    avg_7d = df_fund["rolling_7d"].iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Current (ann.)", f"{current_rate:.1f}%")
    col2.metric("7d Avg (ann.)", f"{avg_7d:.1f}%")
    col3.metric("Hourly Rate", f"{df_fund['interest_1h'].iloc[-1] * 100:.4f}%")

    fig = _make_fig(f"{asset} Perpetual Funding Rate (Annualized)")
    fig.add_trace(go.Scatter(
        x=df_fund["timestamp"], y=df_fund["annualized"],
        mode="lines", name="Hourly (ann.)",
        line=dict(color=color, width=1),
        opacity=0.4,
    ))
    fig.add_trace(go.Scatter(
        x=df_fund["timestamp"], y=df_fund["rolling_8h"],
        mode="lines", name="8h Avg",
        line=dict(color="#4caf50", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df_fund["timestamp"], y=df_fund["rolling_7d"],
        mode="lines", name="7d Avg",
        line=dict(color="#ff9800", width=2),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Annualized Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Data Table (last 48h)"):
        recent = df_fund.tail(48)[["timestamp", "interest_1h", "annualized"]].copy()
        recent["interest_1h"] = recent["interest_1h"].map(lambda x: f"{x*100:.4f}%")
        recent["annualized"] = recent["annualized"].map(lambda x: f"{x:.1f}%")
        st.dataframe(recent, use_container_width=True, hide_index=True)

    _telegram_button("funding", fig=fig,
                     caption=f"{asset} Funding | Current: {current_rate:.1f}% ann. | 7d: {avg_7d:.1f}%")
    return fig


# ---------------------------------------------------------------------------
# Command: DVol Snapshot
# ---------------------------------------------------------------------------

def cmd_dvol_snapshot(asset: str, cfg: dict, color: str) -> go.Figure | None:
    st.subheader(f"🌊 {asset} — DVOL (Deribit Volatility Index)")

    if not cfg["has_dvol"]:
        st.info(f"DVOL is not available for {asset}. Only BTC and ETH have DVOL.")
        return None

    df_dvol = _fetch_dvol(cfg["deribit_ccy"], 90)
    if df_dvol is None or df_dvol.empty:
        st.error("Failed to fetch DVOL data.")
        return None

    current = df_dvol["close"].iloc[-1]
    high_90d = df_dvol["high"].max()
    low_90d = df_dvol["low"].min()

    col1, col2, col3 = st.columns(3)
    col1.metric("Current DVOL", f"{current:.1f}")
    col2.metric("90d High", f"{high_90d:.1f}")
    col3.metric("90d Low", f"{low_90d:.1f}")

    # OHLC candlestick chart
    fig = _make_fig(f"{asset} DVOL — 90 Day History")
    fig.add_trace(go.Candlestick(
        x=df_dvol["timestamp"],
        open=df_dvol["open"],
        high=df_dvol["high"],
        low=df_dvol["low"],
        close=df_dvol["close"],
        increasing_line_color=color,
        decreasing_line_color="#ef5350",
        name="DVOL",
    ))
    fig.update_xaxes(title_text="Date", rangeslider_visible=False)
    fig.update_yaxes(title_text="DVOL")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Data Table (last 14 days)"):
        recent = df_dvol.tail(14).copy()
        recent["timestamp"] = recent["timestamp"].dt.strftime("%Y-%m-%d")
        for col in ["open", "high", "low", "close"]:
            recent[col] = recent[col].round(1)
        st.dataframe(recent, use_container_width=True, hide_index=True)

    _telegram_button("dvol", fig=fig,
                     caption=f"{asset} DVOL: {current:.1f} | 90d Range: {low_90d:.0f}-{high_90d:.0f}")
    return fig


# ---------------------------------------------------------------------------
# Command: Dashboard (Run All)
# ---------------------------------------------------------------------------

def cmd_dashboard(asset: str, cfg: dict, color: str):
    st.subheader(f"📋 {asset} — Full Dashboard")

    spot = _fetch_spot(cfg["index"])
    if spot:
        st.markdown(f"**{asset} Spot:** ${spot:,.{cfg['price_dp']}f}")
    st.divider()

    with st.expander("📈 Vol Term Structure", expanded=True):
        cmd_vol_term_structure(asset, cfg, color)

    with st.expander("📊 Skew Term Structure", expanded=True):
        cmd_skew_term_structure(asset, cfg, color)

    with st.expander("😊 Vol Smile (nearest 30 DTE)", expanded=False):
        cmd_vol_smile(asset, cfg, color)

    with st.expander("🧱 Block Trades (24h)", expanded=True):
        cmd_block_trades(asset, cfg, color)

    with st.expander("💰 Futures Basis", expanded=True):
        cmd_basis_run(asset, cfg, color)

    with st.expander("📉 Realized Vol", expanded=False):
        cmd_realized_vol(asset, cfg, color)

    with st.expander("💸 Funding Rate", expanded=True):
        cmd_funding_rate(asset, cfg, color)

    if cfg["has_dvol"]:
        with st.expander("🌊 DVOL Snapshot", expanded=False):
            cmd_dvol_snapshot(asset, cfg, color)


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

st.title(f"🤖 MCM Bot — {asset}")

if command == "Dashboard":
    cmd_dashboard(asset, cfg, color)
elif command == "Vol Term Structure":
    cmd_vol_term_structure(asset, cfg, color)
elif command == "Skew Term Structure":
    cmd_skew_term_structure(asset, cfg, color)
elif command == "Vol Smile":
    cmd_vol_smile(asset, cfg, color, expiry_ts=selected_expiry_ts)
elif command == "Block Trades":
    cmd_block_trades(asset, cfg, color)
elif command == "Basis Run":
    cmd_basis_run(asset, cfg, color)
elif command == "Realized Vol":
    cmd_realized_vol(asset, cfg, color)
elif command == "Funding Rate":
    cmd_funding_rate(asset, cfg, color)
elif command == "DVol Snapshot":
    cmd_dvol_snapshot(asset, cfg, color)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    f"MCM Bot | Data: Deribit Public API | "
    f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
)
