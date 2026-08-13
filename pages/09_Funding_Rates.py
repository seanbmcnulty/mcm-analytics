"""
Funding Rates — Deribit perpetual funding rate analysis for BTC, ETH, SOL, HYPE.

Shows current rates, historical charts, rolling averages, annualized rates,
and percentile distributions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.deribit import get_funding_history, get_ticker, get_index_price
from lib.constants import ASSET_CONFIG, ASSETS, ASSET_COLORS, PLOTLY_LAYOUT

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Funding Rates",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Funding Rates")
st.caption("Deribit perpetual funding rate analysis across all listed assets.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ordinal(n: int) -> str:
    """Convert number to ordinal string."""
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return str(n) + suffix


@st.cache_data(ttl=120, show_spinner=False)
def fetch_funding_data(instrument_name: str, days: int) -> pd.DataFrame | None:
    """Fetch Deribit funding rate history and compute annualized rate."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000

    df = get_funding_history(instrument_name, start_ms=start_ms, end_ms=end_ms)
    if df is None or df.empty:
        return None

    # Deribit returns interest_1h (hourly rate)
    if "interest_1h" in df.columns:
        df["funding_rate"] = df["interest_1h"].astype(float)
    elif "funding_rate" not in df.columns:
        return None

    df["annualized_rate"] = df["funding_rate"] * (365 * 24)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    selected_assets = st.multiselect(
        "Assets",
        options=ASSETS,
        default=ASSETS,
    )
    lookback_days = st.selectbox(
        "Lookback Period",
        options=[30, 60, 90, 180, 365],
        index=2,
        format_func=lambda x: f"{x} days",
    )
    ma_periods = st.multiselect(
        "Moving Averages (hours)",
        options=[8, 24, 72, 168, 336, 720],
        default=[8, 24, 168],
        format_func=lambda x: (
            f"{x}h" if x < 24 else
            f"{x // 24}d" if x % 24 == 0 else
            f"{x}h"
        ),
    )


# ---------------------------------------------------------------------------
# Current rates overview
# ---------------------------------------------------------------------------

st.subheader("Current Funding Rates")

rate_cols = st.columns(len(selected_assets))
for i, asset in enumerate(selected_assets):
    cfg = ASSET_CONFIG[asset]
    with rate_cols[i]:
        ticker = get_ticker(cfg["perp"])
        if ticker:
            current_funding = ticker.get("current_funding", 0) or 0
            annualized = current_funding * 365 * 24
            price = get_index_price(cfg["index"])

            st.metric(
                label=f"{asset}",
                value=f"{current_funding * 100:.4f}%/h",
                delta=f"Ann: {annualized * 100:.1f}%",
            )
            if price:
                st.caption(f"Index: ${price:,.{cfg['price_dp']}f}")
        else:
            st.metric(label=asset, value="--")

st.divider()

# ---------------------------------------------------------------------------
# Historical funding rate charts
# ---------------------------------------------------------------------------

st.subheader("Historical Funding Rates")

# Fetch all data
funding_data: dict[str, pd.DataFrame] = {}
for asset in selected_assets:
    cfg = ASSET_CONFIG[asset]
    # Fetch extra days for MA warmup
    max_ma = max(ma_periods) if ma_periods else 168
    fetch_days = lookback_days + (max_ma // 24) + 5
    df = fetch_funding_data(cfg["perp"], fetch_days)
    if df is not None and not df.empty:
        funding_data[asset] = df

if not funding_data:
    st.warning("No funding rate data available from Deribit.")
    st.stop()

# Combined time-series chart
fig_ts = go.Figure()
for asset, df in funding_data.items():
    # Resample to hourly for cleaner chart
    df_plot = df.set_index("timestamp").resample("1h")["annualized_rate"].mean().reset_index()
    # Filter to display window
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
    df_plot = df_plot[df_plot["timestamp"] >= cutoff]

    fig_ts.add_trace(go.Scatter(
        x=df_plot["timestamp"],
        y=df_plot["annualized_rate"] * 100,
        name=asset,
        line=dict(color=ASSET_COLORS.get(asset, "#999999"), width=1.5),
    ))

fig_ts.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig_ts.update_layout(
    **PLOTLY_LAYOUT,
    title="Annualized Funding Rate (%)",
    xaxis_title="Date",
    yaxis_title="Annualized Rate (%)",
    legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
    height=500,
)
st.plotly_chart(fig_ts, use_container_width=True)

# ---------------------------------------------------------------------------
# Per-asset detail with rolling averages
# ---------------------------------------------------------------------------

st.subheader("Rolling Averages")

for asset in selected_assets:
    if asset not in funding_data:
        continue

    df = funding_data[asset]
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
    df_display = df[df["timestamp"] >= cutoff].copy()

    if df_display.empty:
        continue

    st.markdown(f"#### {asset}")

    # Compute rolling MAs on hourly data
    df_hourly = df.set_index("timestamp").resample("1h")["annualized_rate"].mean().reset_index()

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=df_hourly[df_hourly["timestamp"] >= cutoff]["timestamp"],
        y=df_hourly[df_hourly["timestamp"] >= cutoff]["annualized_rate"] * 100,
        name="Hourly Rate (ann.)",
        line=dict(color=ASSET_COLORS.get(asset, "#999"), width=1),
        opacity=0.4,
    ))

    ma_colors = ["#e15759", "#59a14f", "#4e79a7", "#f28e2b", "#b07aa1", "#76b7b2"]
    for j, period in enumerate(sorted(ma_periods)):
        ma_col = df_hourly["annualized_rate"].rolling(window=period, min_periods=1).mean()
        df_hourly[f"ma_{period}"] = ma_col

        ma_display = df_hourly[df_hourly["timestamp"] >= cutoff]
        label = f"{period}h" if period < 24 else f"{period // 24}d"

        fig_ma.add_trace(go.Scatter(
            x=ma_display["timestamp"],
            y=ma_display[f"ma_{period}"] * 100,
            name=f"{label} MA",
            line=dict(color=ma_colors[j % len(ma_colors)], width=2),
        ))

    fig_ma.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_ma.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{asset} Funding Rate with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Annualized Rate (%)",
        legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
        height=400,
    )
    st.plotly_chart(fig_ma, use_container_width=True)

    # Summary stats
    recent_rates = df_display["annualized_rate"]
    stats_cols = st.columns(5)
    with stats_cols[0]:
        st.metric("Current (ann.)", f"{recent_rates.iloc[-1] * 100:.2f}%")
    with stats_cols[1]:
        st.metric("Mean", f"{recent_rates.mean() * 100:.2f}%")
    with stats_cols[2]:
        st.metric("Median", f"{recent_rates.median() * 100:.2f}%")
    with stats_cols[3]:
        st.metric("5th Pctl", f"{recent_rates.quantile(0.05) * 100:.2f}%")
    with stats_cols[4]:
        st.metric("95th Pctl", f"{recent_rates.quantile(0.95) * 100:.2f}%")

st.divider()

# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------

st.subheader("Funding Rate Distribution")

dist_col1, dist_col2 = st.columns(2)
with dist_col1:
    n_bins = st.slider("Histogram Bins", 10, 100, 40)
with dist_col2:
    percentile_val = st.slider("Percentile Marker", 0, 100, 50, 1)
    if percentile_val == 100:
        percentile_val = 99.99

for asset in selected_assets:
    if asset not in funding_data:
        continue

    df = funding_data[asset]
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
    rates = df[df["timestamp"] >= cutoff]["annualized_rate"].dropna() * 100

    if rates.empty:
        continue

    pct_value = np.percentile(rates, percentile_val)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=rates,
        nbinsx=n_bins,
        name="Distribution",
        histnorm="probability density",
        marker_color=ASSET_COLORS.get(asset, "#999"),
        opacity=0.7,
    ))
    fig_hist.add_vline(
        x=pct_value,
        line_dash="dash",
        line_color="#2ca02c",
        annotation_text=f"{make_ordinal(percentile_val)} pctl: {pct_value:.1f}%",
        annotation_position="top",
    )
    fig_hist.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig_hist.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{asset} Annualized Funding Rate Distribution ({lookback_days}d)",
        xaxis_title="Annualized Rate (%)",
        yaxis_title="Density",
        height=350,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------------------------
# Percentile summary table
# ---------------------------------------------------------------------------

st.subheader("Percentile Summary")

pctl_rows = []
for asset in selected_assets:
    if asset not in funding_data:
        continue

    df = funding_data[asset]
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
    rates = df[df["timestamp"] >= cutoff]["annualized_rate"].dropna() * 100

    if rates.empty:
        continue

    pctl_rows.append({
        "Asset": asset,
        "5th": f"{rates.quantile(0.05):.1f}%",
        "25th": f"{rates.quantile(0.25):.1f}%",
        "Median": f"{rates.quantile(0.50):.1f}%",
        "75th": f"{rates.quantile(0.75):.1f}%",
        "95th": f"{rates.quantile(0.95):.1f}%",
        "Current": f"{rates.iloc[-1]:.1f}%",
        "Mean": f"{rates.mean():.1f}%",
    })

if pctl_rows:
    pctl_df = pd.DataFrame(pctl_rows).set_index("Asset")
    st.dataframe(pctl_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Cross-asset comparison
# ---------------------------------------------------------------------------

if len(funding_data) > 1:
    st.subheader("Cross-Asset Comparison")

    box_data = []
    for asset in selected_assets:
        if asset not in funding_data:
            continue
        df = funding_data[asset]
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
        rates = df[df["timestamp"] >= cutoff]["annualized_rate"].dropna() * 100
        if not rates.empty:
            box_data.append(go.Box(
                y=rates,
                name=asset,
                marker_color=ASSET_COLORS.get(asset, "#999"),
                boxpoints="outliers",
            ))

    if box_data:
        box_fig = go.Figure(data=box_data)
        box_fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Funding Rate Distribution ({lookback_days}d)",
            yaxis_title="Annualized Rate (%)",
            height=450,
        )
        st.plotly_chart(box_fig, use_container_width=True)
