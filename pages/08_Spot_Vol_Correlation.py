"""
MCM Analytics — Spot-Vol Correlation
DVOL vs Spot price analysis for BTC and ETH.
Uses Deribit DVOL index + index price history.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from lib.deribit import get_dvol, get_tradingview_ohlc, get_index_price
from lib.constants import ASSET_CONFIG, ASSET_COLORS, PLOTLY_LAYOUT
from lib.telegram import send_photo, is_configured
from lib import fx_style

st.set_page_config(page_title="Spot-Vol Correlation", page_icon="📈", layout="wide")
st.title("📈 Spot-Vol Correlation")
st.caption("DVOL vs Spot price • Rolling correlation • BTC & ETH only (Deribit DVOL index)")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

asset = st.sidebar.selectbox("Asset", ["BTC", "ETH"], index=0)
lookback_days = st.sidebar.selectbox("Lookback", [30, 60, 90, 180, 365], index=2)
corr_window = st.sidebar.slider("Correlation Window (days)", 7, 60, 21)

cfg = ASSET_CONFIG[asset]

if not cfg["has_dvol"]:
    st.warning(f"{asset} does not have a DVOL index on Deribit.")
    st.stop()

# ---------------------------------------------------------------------------
# Data Fetch
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_data(asset_key: str, days: int):
    """Fetch DVOL and spot price data."""
    cfg = ASSET_CONFIG[asset_key]
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000

    # DVOL daily
    dvol_df = get_dvol(cfg["deribit_ccy"], "1D", start_ms, now_ms)

    # Spot (index) daily via tradingview
    # Use the index name with the format for tradingview: btc_usd -> index
    # Deribit tradingview uses instrument names like "btc_usd" for index
    spot_df = get_tradingview_ohlc(cfg["index"], "1D", start_ms, now_ms)

    return dvol_df, spot_df


with st.spinner(f"Fetching {asset} DVOL and spot data ({lookback_days}d)..."):
    dvol_df, spot_df = fetch_data(asset, lookback_days)

if dvol_df is None or dvol_df.empty:
    st.error(f"Could not fetch DVOL data for {asset}.")
    st.stop()

if spot_df is None or spot_df.empty:
    st.error(f"Could not fetch spot price data for {asset}.")
    st.stop()

# ---------------------------------------------------------------------------
# Merge and compute
# ---------------------------------------------------------------------------

# Align on date
dvol_df = dvol_df.set_index("timestamp").sort_index()
spot_df = spot_df.set_index("timestamp").sort_index()

# Use close prices
merged = pd.DataFrame({
    "dvol": dvol_df["close"],
    "spot": spot_df["close"],
}).dropna()

if len(merged) < corr_window + 5:
    st.warning("Not enough overlapping data points for correlation analysis.")
    st.stop()

# Compute returns and correlation
merged["spot_return"] = merged["spot"].pct_change()
merged["dvol_change"] = merged["dvol"].diff()
merged["rolling_corr"] = merged["spot_return"].rolling(corr_window).corr(merged["dvol_change"])

# Current stats
current_dvol = merged["dvol"].iloc[-1]
current_spot = merged["spot"].iloc[-1]
current_corr = merged["rolling_corr"].iloc[-1]

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric(f"{asset} Spot", f"${current_spot:,.0f}")
col2.metric(f"{asset} DVOL", f"{current_dvol:.1f}%")
col3.metric(f"{corr_window}d Correlation", f"{current_corr:.3f}" if pd.notna(current_corr) else "—")
col4.metric("Avg DVOL", f"{merged['dvol'].mean():.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# Chart 1: DVOL vs Spot (dual axis)
# ---------------------------------------------------------------------------

st.subheader(f"{asset} DVOL vs Spot Price")

fig1 = make_subplots(specs=[[{"secondary_y": True}]])

fig1.add_trace(
    go.Scatter(x=merged.index, y=merged["spot"], name="Spot Price",
               line=dict(color=ASSET_COLORS[asset], width=2)),
    secondary_y=False,
)
fig1.add_trace(
    go.Scatter(x=merged.index, y=merged["dvol"], name="DVOL (%)",
               line=dict(color="#ff6b6b", width=2)),
    secondary_y=True,
)

fig1.update_layout(
    **PLOTLY_LAYOUT,
    height=400,
    yaxis_title="Spot Price (USD)",
    yaxis2_title="DVOL (%)",
    legend=dict(x=0.01, y=0.99),
)
fx_style.add_watermark(fig1)
st.plotly_chart(fx_style.apply_theme(fig1), width="stretch")

# ---------------------------------------------------------------------------
# Chart 2: Rolling Correlation
# ---------------------------------------------------------------------------

st.subheader(f"Rolling {corr_window}-Day Spot-Vol Correlation")

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(x=merged.index, y=merged["rolling_corr"], name="Correlation",
               line=dict(color="#4da6ff", width=2),
               fill="tozeroy", fillcolor="rgba(77,166,255,0.1)")
)
fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig2.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.3,
               annotation_text="Strong negative")
fig2.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.3,
               annotation_text="Strong positive")

fig2.update_layout(
    **PLOTLY_LAYOUT,
    height=300,
    yaxis_title="Correlation",
    yaxis_range=[-1, 1],
)
fx_style.add_watermark(fig2)
st.plotly_chart(fx_style.apply_theme(fig2), width="stretch")

# ---------------------------------------------------------------------------
# Chart 3: Scatter plot (spot return vs dvol change)
# ---------------------------------------------------------------------------

st.subheader("Spot Return vs DVOL Change (Scatter)")

scatter_data = merged[["spot_return", "dvol_change"]].dropna()

fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(
        x=scatter_data["spot_return"] * 100,
        y=scatter_data["dvol_change"],
        mode="markers",
        marker=dict(
            size=5,
            color=scatter_data["spot_return"],
            colorscale="RdYlGn",
            opacity=0.6,
        ),
        name="Daily observations",
    )
)

# Add regression line
if len(scatter_data) > 10:
    x = scatter_data["spot_return"].values * 100
    y = scatter_data["dvol_change"].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 10:
        coeffs = np.polyfit(x[mask], y[mask], 1)
        x_line = np.linspace(x[mask].min(), x[mask].max(), 50)
        y_line = np.polyval(coeffs, x_line)
        fig3.add_trace(
            go.Scatter(x=x_line, y=y_line, mode="lines", name=f"Regression (β={coeffs[0]:.3f})",
                       line=dict(color="yellow", dash="dash", width=2))
        )

fig3.update_layout(
    **PLOTLY_LAYOUT,
    height=400,
    xaxis_title="Spot Daily Return (%)",
    yaxis_title="DVOL Daily Change (pts)",
)
fx_style.add_watermark(fig3)
st.plotly_chart(fx_style.apply_theme(fig3), width="stretch")

# ---------------------------------------------------------------------------
# Statistics Table
# ---------------------------------------------------------------------------

st.subheader("Summary Statistics")

stats = {
    "Metric": [
        f"Mean DVOL ({lookback_days}d)",
        f"DVOL Std Dev",
        f"Current DVOL Percentile",
        f"Mean Correlation ({corr_window}d window)",
        "Current Correlation",
        "Regression β (spot ret → dvol change)",
        f"Mean Spot Return (daily)",
    ],
    "Value": [
        f"{merged['dvol'].mean():.2f}%",
        f"{merged['dvol'].std():.2f}%",
        f"{(merged['dvol'] <= current_dvol).mean() * 100:.0f}th",
        f"{merged['rolling_corr'].mean():.3f}" if pd.notna(merged['rolling_corr'].mean()) else "—",
        f"{current_corr:.3f}" if pd.notna(current_corr) else "—",
        f"{coeffs[0]:.4f}" if 'coeffs' in dir() else "—",
        f"{merged['spot_return'].mean() * 100:.3f}%",
    ],
}
st.dataframe(pd.DataFrame(stats), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Telegram Blast
# ---------------------------------------------------------------------------

if is_configured():
    st.divider()
    if st.button("📤 Send to Telegram"):
        img = fig1.to_image(format="png", width=1200, height=600)
        caption = (
            f"<b>{asset} Spot-Vol Correlation ({lookback_days}d)</b>\n"
            f"DVOL: {current_dvol:.1f}% | Spot: ${current_spot:,.0f}\n"
            f"{corr_window}d Correlation: {current_corr:.3f}"
        )
        if send_photo(img, caption):
            st.success("Sent!")
        else:
            st.error("Failed to send.")
