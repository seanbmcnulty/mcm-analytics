"""
Term Structure — ATM implied volatility term structure across expiries.

Shows ATM IV vs days-to-expiry for Deribit-listed assets, with optional
multi-asset overlay and DVOL context.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.deribit import get_option_chain, get_index_price, get_dvol
from lib.constants import ASSET_CONFIG, ASSETS, ASSET_COLORS, PLOTLY_LAYOUT
from lib.instruments import parse_instrument, dte_from_expiry

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Term Structure",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120, show_spinner=False)
def fetch_term_structure(asset: str) -> pd.DataFrame | None:
    """
    Fetch option chain and compute ATM IV term structure for an asset.
    Returns DataFrame with: expiry_date, dte, atm_strike, atm_iv, bid_iv, ask_iv, atm_oi
    """
    cfg = ASSET_CONFIG.get(asset)
    if not cfg:
        return None

    # Get spot price
    spot = get_index_price(cfg["index"])
    if not spot:
        return None

    # Get option chain
    chain = get_option_chain(cfg["deribit_ccy"], "option")
    if not chain:
        return None

    # Filter to this asset's prefix
    prefix = cfg["deribit_prefix"]
    instruments = [i for i in chain if i["instrument_name"].startswith(prefix)]

    if not instruments:
        return None

    now_ms = int(time.time() * 1000)

    # Group by expiry, find ATM strike for each
    expiry_groups: dict[int, list[dict]] = {}
    for inst in instruments:
        parsed = parse_instrument(inst["instrument_name"])
        if not parsed:
            continue
        # Only use calls for ATM determination (avoid double-counting)
        if parsed.kind != "C":
            continue
        # Skip if no valid IV
        mark_iv = inst.get("mark_iv")
        if mark_iv is None or mark_iv == 0:
            continue

        expiry_ts = inst.get("expiration_timestamp", 0)
        if expiry_ts not in expiry_groups:
            expiry_groups[expiry_ts] = []
        expiry_groups[expiry_ts].append({
            "instrument": inst,
            "parsed": parsed,
        })

    rows = []
    for expiry_ts, group in expiry_groups.items():
        dte = (expiry_ts - now_ms) / (1000 * 86400)

        # Filter: skip expiries < 2 DTE
        if dte < 2:
            continue

        # Find ATM strike (closest to spot)
        best = None
        best_dist = float("inf")
        for item in group:
            dist = abs(item["parsed"].strike - spot)
            if dist < best_dist:
                best_dist = dist
                best = item

        if best is None:
            continue

        inst = best["instrument"]
        atm_strike = best["parsed"].strike

        # Filter: skip very low OI expiries (less than 10 contracts total)
        oi = inst.get("open_interest", 0) or 0
        if oi < 10:
            continue

        # mark_iv is in decimal (0.55 = 55%) — convert to percentage for display
        mark_iv = inst.get("mark_iv", 0) or 0
        bid_iv = inst.get("bid_iv", 0) or 0
        ask_iv = inst.get("ask_iv", 0) or 0

        expiry_date = datetime.fromtimestamp(expiry_ts / 1000, tz=timezone.utc)

        rows.append({
            "expiry_date": expiry_date,
            "dte": round(dte, 1),
            "atm_strike": atm_strike,
            "atm_iv": round(mark_iv * 100, 2),
            "bid_iv": round(bid_iv * 100, 2),
            "ask_iv": round(ask_iv * 100, 2),
            "atm_oi": int(oi),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("dte").reset_index(drop=True)
    return df


@st.cache_data(ttl=120, show_spinner=False)
def fetch_spot_price(asset: str) -> float | None:
    """Fetch current spot price for an asset."""
    cfg = ASSET_CONFIG.get(asset)
    if not cfg:
        return None
    return get_index_price(cfg["index"])


@st.cache_data(ttl=300, show_spinner=False)
def fetch_dvol_latest(asset: str) -> float | None:
    """Fetch latest DVOL reading (BTC/ETH only)."""
    cfg = ASSET_CONFIG.get(asset)
    if not cfg or not cfg.get("has_dvol"):
        return None

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - 2 * 24 * 3600 * 1000  # last 2 days
    df = get_dvol(cfg["deribit_ccy"], resolution="1D", start_ms=start_ms, end_ms=end_ms)
    if df is not None and not df.empty:
        return float(df["close"].iloc[-1])
    return None


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_term_structure_chart(df: pd.DataFrame, asset: str,
                               compare_df: pd.DataFrame | None = None,
                               compare_asset: str | None = None) -> go.Figure:
    """Build ATM IV term structure line chart."""
    fig = go.Figure()

    color = ASSET_COLORS.get(asset, "#ffffff")

    # Main asset line
    fig.add_trace(go.Scatter(
        x=df["dte"],
        y=df["atm_iv"],
        mode="lines+markers",
        name=f"{asset} ATM IV",
        line=dict(color=color, width=2.5),
        marker=dict(size=7),
        hovertemplate=(
            f"<b>{asset}</b><br>"
            "DTE: %{x:.0f}<br>"
            "ATM IV: %{y:.1f}%<br>"
            "<extra></extra>"
        ),
    ))

    # Bid/Ask IV band
    fig.add_trace(go.Scatter(
        x=df["dte"],
        y=df["ask_iv"],
        mode="lines",
        name="Ask IV",
        line=dict(color=color, width=0.5, dash="dot"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=df["dte"],
        y=df["bid_iv"],
        mode="lines",
        name="Bid IV",
        line=dict(color=color, width=0.5, dash="dot"),
        fill="tonexty",
        fillcolor=f"rgba({_hex_to_rgb(color)}, 0.1)",
        showlegend=False,
    ))

    # Comparison overlay
    if compare_df is not None and compare_asset:
        comp_color = ASSET_COLORS.get(compare_asset, "#aaaaaa")
        fig.add_trace(go.Scatter(
            x=compare_df["dte"],
            y=compare_df["atm_iv"],
            mode="lines+markers",
            name=f"{compare_asset} ATM IV",
            line=dict(color=comp_color, width=2.5, dash="dash"),
            marker=dict(size=6),
            hovertemplate=(
                f"<b>{compare_asset}</b><br>"
                "DTE: %{x:.0f}<br>"
                "ATM IV: %{y:.1f}%<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="ATM Implied Volatility Term Structure",
        xaxis_title="Days to Expiry",
        yaxis_title="ATM IV (%)",
        height=500,
        legend=dict(x=0, y=-0.2, orientation="h"),
        hovermode="x unified",
    )

    return fig


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to 'r, g, b' string for rgba()."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r}, {g}, {b}"


# ---------------------------------------------------------------------------
# Metrics and analysis
# ---------------------------------------------------------------------------

def compute_slope_metrics(df: pd.DataFrame) -> dict:
    """Compute term structure slope metrics."""
    if len(df) < 2:
        return {}

    # Short-term: expiries <= 30 DTE
    # Long-term: expiries > 60 DTE
    short_term = df[df["dte"] <= 30]
    long_term = df[df["dte"] > 60]

    metrics = {}

    if not short_term.empty and not long_term.empty:
        short_iv = short_term["atm_iv"].mean()
        long_iv = long_term["atm_iv"].mean()
        metrics["short_term_iv"] = round(short_iv, 2)
        metrics["long_term_iv"] = round(long_iv, 2)
        metrics["slope_ratio"] = round(short_iv / long_iv, 3) if long_iv > 0 else None
        metrics["spread"] = round(short_iv - long_iv, 2)
    elif not short_term.empty:
        metrics["short_term_iv"] = round(short_term["atm_iv"].mean(), 2)
    elif not long_term.empty:
        metrics["long_term_iv"] = round(long_term["atm_iv"].mean(), 2)

    # Front vs back
    metrics["front_iv"] = round(df["atm_iv"].iloc[0], 2)
    metrics["back_iv"] = round(df["atm_iv"].iloc[-1], 2)

    return metrics


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title("Term Structure")
st.caption("ATM implied volatility term structure across expiries.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    asset = st.selectbox("Asset", ASSETS, index=0)
    compare_enabled = st.checkbox("Compare with another asset", value=False)
    compare_asset = None
    if compare_enabled:
        other_assets = [a for a in ASSETS if a != asset]
        compare_asset = st.selectbox("Compare Asset", other_assets, index=0)

    min_oi = st.number_input("Min Open Interest filter", min_value=0, value=10, step=10)

# Fetch data
with st.spinner(f"Loading {asset} term structure..."):
    df = fetch_term_structure(asset)

if df is None or df.empty:
    st.warning(f"No term structure data available for {asset}. Check that Deribit is reachable.")
    st.stop()

# Apply OI filter
df = df[df["atm_oi"] >= min_oi].reset_index(drop=True)

if df.empty:
    st.warning("No expiries pass the minimum OI filter. Try lowering the threshold.")
    st.stop()

# Fetch comparison data if enabled
compare_df = None
if compare_enabled and compare_asset:
    compare_df = fetch_term_structure(compare_asset)
    if compare_df is not None:
        compare_df = compare_df[compare_df["atm_oi"] >= min_oi].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------

spot = fetch_spot_price(asset)
slope_metrics = compute_slope_metrics(df)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if spot:
        dp = ASSET_CONFIG[asset]["price_dp"]
        st.metric(f"{asset} Spot", f"${spot:,.{dp}f}")
    else:
        st.metric(f"{asset} Spot", "N/A")

with col2:
    st.metric("Front IV", f"{slope_metrics.get('front_iv', 'N/A')}%")

with col3:
    st.metric("Back IV", f"{slope_metrics.get('back_iv', 'N/A')}%")

with col4:
    ratio = slope_metrics.get("slope_ratio")
    if ratio is not None:
        # Ratio > 1 = backwardation (short > long), < 1 = contango (short < long)
        shape = "Backwardation" if ratio > 1 else "Contango"
        st.metric("Short/Long Ratio", f"{ratio:.3f}", delta=shape)
    else:
        st.metric("Short/Long Ratio", "N/A")

# ---------------------------------------------------------------------------
# DVOL context
# ---------------------------------------------------------------------------

dvol_value = fetch_dvol_latest(asset)
if dvol_value is not None:
    front_iv = slope_metrics.get("front_iv")
    st.divider()
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        st.metric(f"{asset} DVOL (30d index)", f"{dvol_value:.1f}%")
    with dcol2:
        if front_iv:
            diff = front_iv - dvol_value
            st.metric("Front IV vs DVOL", f"{diff:+.1f}%",
                      delta="Rich" if diff > 0 else "Cheap")
    with dcol3:
        long_iv = slope_metrics.get("long_term_iv")
        if long_iv:
            diff_long = long_iv - dvol_value
            st.metric("Long IV vs DVOL", f"{diff_long:+.1f}%",
                      delta="Rich" if diff_long > 0 else "Cheap")

# ---------------------------------------------------------------------------
# Term structure chart
# ---------------------------------------------------------------------------

st.divider()

fig = build_term_structure_chart(df, asset, compare_df=compare_df, compare_asset=compare_asset)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Slope analysis
# ---------------------------------------------------------------------------

st.subheader("Term Structure Slope")

if slope_metrics.get("slope_ratio") is not None:
    slope_col1, slope_col2 = st.columns(2)
    with slope_col1:
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Short-term IV (<=30 DTE) | {slope_metrics.get('short_term_iv', 'N/A')}% |
        | Long-term IV (>60 DTE) | {slope_metrics.get('long_term_iv', 'N/A')}% |
        | Spread (Short - Long) | {slope_metrics.get('spread', 'N/A')}% |
        | Ratio (Short / Long) | {slope_metrics.get('slope_ratio', 'N/A')} |
        """)
    with slope_col2:
        ratio = slope_metrics["slope_ratio"]
        if ratio > 1.1:
            st.warning("Steep backwardation — market pricing near-term risk/event premium.")
        elif ratio > 1.0:
            st.info("Mild backwardation — slightly elevated short-term vol expectations.")
        elif ratio > 0.9:
            st.info("Normal contango — longer-dated vol carries time premium.")
        else:
            st.success("Steep contango — low near-term vol expectations relative to long-term.")
else:
    st.info("Need expiries in both short (<= 30 DTE) and long (> 60 DTE) buckets for slope analysis.")

# ---------------------------------------------------------------------------
# Data table
# ---------------------------------------------------------------------------

st.subheader("Term Structure Data")

display_df = df[["expiry_date", "dte", "atm_strike", "atm_iv", "bid_iv", "ask_iv", "atm_oi"]].copy()
display_df["expiry_date"] = display_df["expiry_date"].dt.strftime("%Y-%m-%d")
display_df = display_df.rename(columns={
    "expiry_date": "Expiry",
    "dte": "DTE",
    "atm_strike": "ATM Strike",
    "atm_iv": "ATM IV (%)",
    "bid_iv": "Bid IV (%)",
    "ask_iv": "Ask IV (%)",
    "atm_oi": "ATM OI",
})

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Multi-asset comparison table (if enabled)
# ---------------------------------------------------------------------------

if compare_enabled and compare_df is not None and not compare_df.empty:
    st.subheader(f"{asset} vs {compare_asset} Comparison")

    # Align on nearest DTE buckets
    merged_rows = []
    for _, row in df.iterrows():
        dte_val = row["dte"]
        # Find closest DTE in compare_df
        comp_match = compare_df.iloc[(compare_df["dte"] - dte_val).abs().argsort()[:1]]
        if not comp_match.empty:
            comp_row = comp_match.iloc[0]
            # Only match if within 5 DTE
            if abs(comp_row["dte"] - dte_val) <= 5:
                merged_rows.append({
                    "DTE": dte_val,
                    f"{asset} IV (%)": row["atm_iv"],
                    f"{compare_asset} IV (%)": comp_row["atm_iv"],
                    "Spread (%)": round(row["atm_iv"] - comp_row["atm_iv"], 2),
                })

    if merged_rows:
        merged_df = pd.DataFrame(merged_rows)
        st.dataframe(merged_df, use_container_width=True, hide_index=True)
    else:
        st.info("No overlapping DTE buckets found for comparison.")
