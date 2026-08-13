"""
Block Trades — Deribit public block trade analysis.

Fetches recent option trades, filters to block-size, computes Greeks,
and visualizes flow by strike/DTE. Public API only (no authentication).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.deribit import get_trades, get_index_price, get_book_summary_by_instrument
from lib.constants import ASSET_CONFIG, ASSETS, ASSET_COLORS, PLOTLY_LAYOUT
from lib.instruments import parse_instrument, dte_from_expiry, format_strike
from lib.vol_math import bs_delta, bs_gamma, bs_vega
from lib.telegram import send_message, send_photo, is_configured

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Block Trades — Deribit",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Deribit Block Trades")
st.caption("Public option block trades — last 24h")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    selected_asset = st.selectbox("Currency", options=ASSETS, index=0)
    cfg = ASSET_CONFIG[selected_asset]
    min_block = cfg["min_block"]

    st.metric("Min Block Size", f"{min_block:,.0f} contracts")

    custom_min = st.number_input(
        "Override Min Size",
        min_value=0.0,
        value=float(min_block),
        step=float(min_block) / 10,
        help="Custom minimum trade size filter",
    )
    if custom_min > 0:
        min_block = custom_min

    show_calls = st.checkbox("Show Calls", value=True)
    show_puts = st.checkbox("Show Puts", value=True)

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    if auto_refresh:
        time.sleep(0.1)  # Prevent tight loop
        st.rerun()


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Fetching block trades...", ttl=60)
def fetch_block_trades(asset: str, min_size: float) -> pd.DataFrame:
    """Fetch last 24h option trades and filter to block size."""
    cfg = ASSET_CONFIG[asset]
    currency = cfg["deribit_ccy"]

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - 24 * 3600 * 1000

    # Fetch trades — may need multiple calls for high-volume assets
    all_trades = []
    trades = get_trades(currency, "option", start_ms, end_ms, count=1000)
    if trades:
        all_trades.extend(trades)

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)

    # Filter to the correct asset prefix for USDC-settled (SOL, HYPE share currency)
    prefix = cfg["deribit_prefix"]
    if "instrument_name" in df.columns:
        df = df[df["instrument_name"].str.startswith(prefix)].copy()

    # Filter to block size
    if "amount" in df.columns:
        df = df[df["amount"].abs() >= min_size].copy()

    if df.empty:
        return pd.DataFrame()

    # Parse timestamps
    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=30)
def fetch_spot_price(asset: str) -> float | None:
    """Get current index/spot price."""
    cfg = ASSET_CONFIG[asset]
    return get_index_price(cfg["index"])


# ---------------------------------------------------------------------------
# Greek computation
# ---------------------------------------------------------------------------

def compute_greeks(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Add delta, gamma, vega columns to trades DataFrame."""
    if df.empty or spot is None:
        return df

    deltas, gammas, vegas = [], [], []

    for _, row in df.iterrows():
        parsed = parse_instrument(row.get("instrument_name", ""))
        if parsed is None:
            deltas.append(np.nan)
            gammas.append(np.nan)
            vegas.append(np.nan)
            continue

        # Time to expiry in years
        now = datetime.now(timezone.utc)
        tte = max((parsed.expiry_date - now).total_seconds() / (365.25 * 86400), 0.001)

        # Use mark_iv if available, else fallback IV
        iv = row.get("iv")
        if iv is None or iv == 0 or pd.isna(iv):
            iv = row.get("mark_iv", 0)
        if iv is None or iv == 0 or pd.isna(iv):
            iv = 80.0  # Fallback 80% for crypto

        # Convert IV from percentage to decimal
        vol = iv / 100.0

        strike = parsed.strike
        opt_type = parsed.kind  # "C" or "P"

        d = bs_delta(spot, strike, tte, vol, opt_type)
        g = bs_gamma(spot, strike, tte, vol)
        v = bs_vega(spot, strike, tte, vol)

        deltas.append(d)
        gammas.append(g)
        vegas.append(v)

    df = df.copy()
    df["delta"] = deltas
    df["gamma"] = gammas
    df["vega"] = vegas

    # Dollar Greeks (amount-weighted)
    amount = df["amount"].abs()
    df["dollar_delta"] = df["delta"] * amount * spot
    df["dollar_gamma_1pct"] = df["gamma"] * amount * spot ** 2 * 0.01
    df["dollar_vega"] = df["vega"] * amount

    return df


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# Fetch data
trades_df = fetch_block_trades(selected_asset, min_block)
spot = fetch_spot_price(selected_asset)

if trades_df.empty:
    st.warning(
        f"No block trades found for {selected_asset} in the last 24h "
        f"with minimum size {min_block:,.0f}."
    )
    st.stop()

# Enrich with parsed fields
parsed_data = []
for _, row in trades_df.iterrows():
    p = parse_instrument(row.get("instrument_name", ""))
    if p:
        parsed_data.append({
            "strike": p.strike,
            "expiry_date": p.expiry_date,
            "option_type": p.kind,
            "base_currency": p.base_currency,
        })
    else:
        parsed_data.append({
            "strike": np.nan,
            "expiry_date": pd.NaT,
            "option_type": None,
            "base_currency": None,
        })

parsed_df = pd.DataFrame(parsed_data, index=trades_df.index)
trades_df = pd.concat([trades_df, parsed_df], axis=1)

# Filter by call/put selection
if not show_calls:
    trades_df = trades_df[trades_df["option_type"] != "C"]
if not show_puts:
    trades_df = trades_df[trades_df["option_type"] != "P"]

if trades_df.empty:
    st.info("No trades match current filters.")
    st.stop()

# Compute Greeks
trades_df = compute_greeks(trades_df, spot)

# Compute DTE
now = datetime.now(timezone.utc)
trades_df["dte"] = trades_df["expiry_date"].apply(
    lambda x: max((x - now).total_seconds() / 86400, 0) if pd.notna(x) else np.nan
)

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

st.subheader("Summary")

col1, col2, col3, col4, col5 = st.columns(5)

total_volume = trades_df["amount"].abs().sum()
num_blocks = len(trades_df)
buy_vol = trades_df[trades_df["direction"] == "buy"]["amount"].abs().sum() if "direction" in trades_df.columns else 0
sell_vol = trades_df[trades_df["direction"] == "sell"]["amount"].abs().sum() if "direction" in trades_df.columns else 0
net_direction = "BUY" if buy_vol > sell_vol else "SELL"

col1.metric("Total Block Volume", f"{total_volume:,.0f}")
col2.metric("Number of Blocks", f"{num_blocks}")
col3.metric("Buy Volume", f"{buy_vol:,.0f}")
col4.metric("Sell Volume", f"{sell_vol:,.0f}")
col5.metric("Net Direction", net_direction)

# Top strikes
st.markdown("**Top Strikes by Volume:**")
if "strike" in trades_df.columns:
    top_strikes = (
        trades_df.groupby("strike")["amount"]
        .apply(lambda x: x.abs().sum())
        .sort_values(ascending=False)
        .head(5)
    )
    strike_str = " | ".join(
        f"{format_strike(s, ASSET_CONFIG[selected_asset]['price_dp'])}: {v:,.0f}"
        for s, v in top_strikes.items()
    )
    st.text(strike_str)

# ---------------------------------------------------------------------------
# Scatter chart: strike x DTE, sized by amount, colored by direction
# ---------------------------------------------------------------------------

st.subheader("Block Flow Map")

fig_scatter = go.Figure()

for direction, color, symbol in [("buy", "#26a69a", "circle"), ("sell", "#ef5350", "diamond")]:
    if "direction" not in trades_df.columns:
        break
    mask = trades_df["direction"] == direction
    subset = trades_df[mask]
    if subset.empty:
        continue

    sizes = subset["amount"].abs()
    # Normalize marker size (5-50 range)
    max_size = sizes.max() if sizes.max() > 0 else 1
    marker_sizes = 5 + (sizes / max_size) * 45

    fig_scatter.add_trace(go.Scatter(
        x=subset["dte"],
        y=subset["strike"],
        mode="markers",
        name=direction.capitalize(),
        marker=dict(
            size=marker_sizes,
            color=color,
            symbol=symbol,
            opacity=0.7,
            line=dict(width=1, color="white"),
        ),
        text=subset["instrument_name"],
        customdata=np.stack([
            subset["amount"].abs().values,
            subset.get("iv", pd.Series(0, index=subset.index)).values,
            subset.get("delta", pd.Series(0, index=subset.index)).values,
        ], axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "DTE: %{x:.0f}d<br>"
            "Strike: %{y:,.0f}<br>"
            "Size: %{customdata[0]:,.0f}<br>"
            "IV: %{customdata[1]:.1f}%<br>"
            "Delta: %{customdata[2]:.2f}<br>"
            "<extra></extra>"
        ),
    ))

# Add spot price reference line
if spot:
    fig_scatter.add_hline(
        y=spot, line_dash="dash", line_color="yellow",
        annotation_text=f"Spot: {spot:,.0f}",
    )

fig_scatter.update_layout(
    **PLOTLY_LAYOUT,
    height=500,
    title=f"{selected_asset} Block Trades — Strike vs DTE",
    xaxis_title="Days to Expiry",
    yaxis_title="Strike",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------------------
# Trades table
# ---------------------------------------------------------------------------

st.subheader("Trade Log")

display_cols = ["time", "instrument_name", "direction", "amount", "price"]
if "iv" in trades_df.columns:
    display_cols.append("iv")
if "mark_iv" in trades_df.columns:
    display_cols.append("mark_iv")
display_cols.extend(["delta", "dollar_delta", "dollar_vega"])

# Filter to existing columns
display_cols = [c for c in display_cols if c in trades_df.columns]

display_df = trades_df[display_cols].copy()
if "time" in display_df.columns:
    display_df["time"] = display_df["time"].dt.strftime("%H:%M:%S")

# Format numeric columns
format_dict = {}
for c in ["amount", "dollar_delta", "dollar_gamma_1pct", "dollar_vega"]:
    if c in display_df.columns:
        format_dict[c] = "{:,.0f}"
for c in ["price", "delta", "gamma", "vega"]:
    if c in display_df.columns:
        format_dict[c] = "{:.4f}"
for c in ["iv", "mark_iv"]:
    if c in display_df.columns:
        format_dict[c] = "{:.1f}"

st.dataframe(
    display_df.style.format(format_dict, na_rep="—"),
    use_container_width=True,
    height=400,
)

# ---------------------------------------------------------------------------
# Greeks summary
# ---------------------------------------------------------------------------

st.subheader("Aggregate Greeks")

greek_cols = st.columns(3)
if "dollar_delta" in trades_df.columns:
    net_delta = trades_df["dollar_delta"].sum()
    greek_cols[0].metric("Net Dollar Delta", f"${net_delta:,.0f}")
if "dollar_gamma_1pct" in trades_df.columns:
    net_gamma = trades_df["dollar_gamma_1pct"].sum()
    greek_cols[1].metric("Net Dollar Gamma (1%)", f"${net_gamma:,.0f}")
if "dollar_vega" in trades_df.columns:
    net_vega = trades_df["dollar_vega"].sum()
    greek_cols[2].metric("Net Dollar Vega", f"${net_vega:,.0f}")

# ---------------------------------------------------------------------------
# Telegram blast
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Telegram Report")

if not is_configured():
    st.info("Telegram not configured. Set credentials in secrets.toml or environment.")
else:
    if st.button("Send Block Summary to Telegram", type="primary"):
        lines = [f"<b>{selected_asset} Block Trades — Last 24h</b>"]
        lines.append(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        lines.append("")
        lines.append(f"Total Volume: {total_volume:,.0f} contracts")
        lines.append(f"Blocks: {num_blocks}")
        lines.append(f"Net Direction: {net_direction}")
        lines.append(f"Buy: {buy_vol:,.0f} | Sell: {sell_vol:,.0f}")
        lines.append("")

        if "dollar_delta" in trades_df.columns:
            lines.append(f"Net $Delta: ${trades_df['dollar_delta'].sum():,.0f}")
        if "dollar_vega" in trades_df.columns:
            lines.append(f"Net $Vega: ${trades_df['dollar_vega'].sum():,.0f}")
        lines.append("")

        lines.append("<b>Top Strikes:</b>")
        if "strike" in trades_df.columns:
            for s, v in top_strikes.head(5).items():
                lines.append(
                    f"  {format_strike(s, ASSET_CONFIG[selected_asset]['price_dp'])}: "
                    f"{v:,.0f} contracts"
                )

        msg = "\n".join(lines)
        success = send_message(msg)
        if success:
            st.success("Block summary sent to Telegram")
        else:
            st.error("Failed to send to Telegram")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.caption(
    f"Spot: {spot:,.{ASSET_CONFIG[selected_asset]['price_dp']}f} | "
    f"Min block: {min_block:,.0f} | "
    f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
)
