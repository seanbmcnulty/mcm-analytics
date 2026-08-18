"""
Realized Volatility — Multi-estimator realized vol analysis for crypto assets.

Computes annualized RV using 5 estimators (Close-to-Close, Parkinson,
Garman-Klass, Yang-Zhang, Rogers-Satchell) across multiple rolling windows.
Data sourced from Deribit TradingView OHLC (perpetual contracts).
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

from lib.deribit import get_tradingview_ohlc, clear_cache
from lib.constants import ASSET_CONFIG, ASSETS, ASSET_COLORS, PLOTLY_LAYOUT
from lib.vol_math import (
    close_to_close_vol, parkinson_vol, garman_klass_vol,
    yang_zhang_vol, rogers_satchell_vol, compute_rv_matrix, RV_ESTIMATORS,
)
from lib.telegram import send_message, send_photo, is_configured
from lib import fx_style

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Realized Volatility",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Realized Volatility")
st.caption("Multi-estimator RV from Deribit perpetual OHLC data (annualized %)")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Parameters")

    selected_assets = st.multiselect(
        "Assets",
        options=ASSETS,
        default=["BTC", "ETH"],
        help="Select one or more assets to analyze",
    )

    # Map user-facing timeframe labels to Deribit resolution values
    TIMEFRAME_MAP = {
        "1h": "60",
        "4h": "240",
        "12h": "720",
        "1D": "1D",
    }
    timeframe = st.selectbox(
        "Candle Timeframe",
        options=list(TIMEFRAME_MAP.keys()),
        index=3,  # default 1D
        help="OHLC bar size for RV calculation",
    )

    LOOKBACK_MAP = {
        "30d": 30,
        "60d": 60,
        "90d": 90,
        "180d": 180,
        "365d": 365,
    }
    lookback_label = st.selectbox(
        "Lookback Period",
        options=list(LOOKBACK_MAP.keys()),
        index=2,  # default 90d
    )
    lookback_days = LOOKBACK_MAP[lookback_label]

    estimator_name = st.selectbox(
        "Time Series Estimator",
        options=list(RV_ESTIMATORS.keys()),
        index=0,
        help="Which estimator to plot as a time series",
    )

    rv_windows = st.multiselect(
        "Matrix Windows (days)",
        options=[7, 14, 30, 60, 90],
        default=[7, 14, 30, 60, 90],
        help="Rolling window sizes for the RV matrix heatmap",
    )

    st.divider()
    if st.button("Clear Cache"):
        clear_cache()
        st.success("Cache cleared")


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _periods_per_day(resolution: str) -> float:
    """Return how many candles fit in one calendar day for annualization."""
    if resolution == "1D":
        return 1.0
    minutes = int(resolution)
    return 1440.0 / minutes


@st.cache_data(show_spinner="Fetching OHLC data...", ttl=120)
def fetch_ohlc(asset: str, resolution: str, lookback: int) -> pd.DataFrame | None:
    """Fetch OHLC candles from Deribit for the given asset's perpetual."""
    cfg = ASSET_CONFIG[asset]
    instrument = cfg["perp"]
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - lookback * 24 * 3600 * 1000

    df = get_tradingview_ohlc(instrument, resolution, start_ms, end_ms)
    if df is None or df.empty:
        return None
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def compute_time_series(df: pd.DataFrame, estimator: str, window: int,
                        resolution: str) -> pd.Series:
    """Compute a single RV estimator time series, annualized to percentage."""
    # Adjust window for intraday: if resolution is not 1D, scale window
    ppd = _periods_per_day(resolution)
    adjusted_window = max(int(window * ppd), 2)

    if estimator == "Close-to-Close":
        log_ret = np.log(df["close"] / df["close"].shift(1))
        # Annualize based on periods per year
        periods_per_year = ppd * 365
        rv = log_ret.rolling(adjusted_window).std() * np.sqrt(periods_per_year)
    elif estimator == "Parkinson":
        hl = np.log(df["high"] / df["low"]) ** 2
        factor = 1 / (4 * np.log(2))
        periods_per_year = ppd * 365
        rv = np.sqrt(hl.rolling(adjusted_window).mean() * factor * periods_per_year)
    elif estimator == "Garman-Klass":
        hl = 0.5 * np.log(df["high"] / df["low"]) ** 2
        co = -(2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2
        gk = hl + co
        periods_per_year = ppd * 365
        rv = np.sqrt(gk.rolling(adjusted_window).mean() * periods_per_year)
    elif estimator == "Yang-Zhang":
        n = adjusted_window
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        log_oc = np.log(df["open"] / df["close"].shift(1))
        log_co = np.log(df["close"] / df["open"])
        log_ho = np.log(df["high"] / df["open"])
        log_lo = np.log(df["low"] / df["open"])
        sigma_o = log_oc.rolling(n).var()
        sigma_c = log_co.rolling(n).var()
        rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(n).mean()
        sigma2 = sigma_o + k * sigma_c + (1 - k) * rs
        periods_per_year = ppd * 365
        rv = np.sqrt(np.maximum(sigma2, 0) * periods_per_year)
    elif estimator == "Rogers-Satchell":
        log_ho = np.log(df["high"] / df["open"])
        log_hc = np.log(df["high"] / df["close"])
        log_lo = np.log(df["low"] / df["open"])
        log_lc = np.log(df["low"] / df["close"])
        rs = (log_ho * log_hc + log_lo * log_lc).rolling(adjusted_window).mean()
        periods_per_year = ppd * 365
        rv = np.sqrt(np.maximum(rs, 0) * periods_per_year)
    else:
        rv = pd.Series(np.nan, index=df.index)

    return rv * 100  # Convert to percentage


def compute_matrix_for_asset(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute RV matrix (estimators x windows) using lib function.
    Returns DataFrame pivoted: rows=estimators, columns=window labels."""
    matrix_df = compute_rv_matrix(df, windows)
    if matrix_df.empty:
        return pd.DataFrame()
    # Pivot for heatmap: rows = estimator, columns = window
    pivot = matrix_df.pivot(index="estimator", columns="window", values="value")
    pivot.columns = [f"{w}d" for w in pivot.columns]
    return pivot


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

if not selected_assets:
    st.warning("Select at least one asset from the sidebar.")
    st.stop()

# Fetch data for all selected assets
ohlc_data: dict[str, pd.DataFrame] = {}
for asset in selected_assets:
    df = fetch_ohlc(asset, TIMEFRAME_MAP[timeframe], lookback_days)
    if df is not None and not df.empty:
        ohlc_data[asset] = df
    else:
        st.warning(f"No data available for {asset}")

if not ohlc_data:
    st.error("No OHLC data could be fetched. Check connectivity.")
    st.stop()

# ---------------------------------------------------------------------------
# RV Matrix heatmap (one per asset)
# ---------------------------------------------------------------------------

st.subheader("RV Matrix (Latest Values)")

matrix_cols = st.columns(len(ohlc_data))
matrices: dict[str, pd.DataFrame] = {}

for col, (asset, df) in zip(matrix_cols, ohlc_data.items()):
    with col:
        st.markdown(f"**{asset}**")
        matrix = compute_matrix_for_asset(df, sorted(rv_windows))
        if matrix.empty:
            st.info("Insufficient data for matrix")
            continue
        matrices[asset] = matrix

        # Heatmap figure
        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale="YlOrRd",
            text=np.round(matrix.values, 1),
            texttemplate="%{text}%",
            textfont={"size": 11},
            hovertemplate="Estimator: %{y}<br>Window: %{x}<br>RV: %{z:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=280,
            xaxis_title="Window",
            yaxis_title="",
            title=f"{asset} Realized Vol Matrix",
        )
        fx_style.add_watermark(fig)
        st.plotly_chart(fx_style.apply_theme(fig), width="stretch")

# ---------------------------------------------------------------------------
# Time series chart
# ---------------------------------------------------------------------------

st.subheader(f"Rolling {estimator_name} RV — {timeframe} candles")

# Let user pick which window to chart as time series
ts_window = st.select_slider(
    "Rolling window (days)", options=[7, 14, 30, 60, 90], value=30
)

fig_ts = go.Figure()
for asset, df in ohlc_data.items():
    rv_series = compute_time_series(df, estimator_name, ts_window, TIMEFRAME_MAP[timeframe])
    fig_ts.add_trace(go.Scatter(
        x=df["timestamp"],
        y=rv_series,
        name=asset,
        line=dict(color=ASSET_COLORS.get(asset, "#888"), width=2),
        hovertemplate=f"{asset}: %{{y:.1f}}%<extra></extra>",
    ))

fig_ts.update_layout(
    **PLOTLY_LAYOUT,
    height=420,
    title=f"{estimator_name} RV ({ts_window}d window, {timeframe} bars)",
    yaxis_title="Annualized RV (%)",
    xaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    hovermode="x unified",
)
fx_style.add_watermark(fig_ts)
st.plotly_chart(fx_style.apply_theme(fig_ts), width="stretch")

# ---------------------------------------------------------------------------
# Current RV summary table
# ---------------------------------------------------------------------------

st.subheader("Current RV Summary")

summary_rows = []
for asset, df in ohlc_data.items():
    for est_name in RV_ESTIMATORS:
        rv_val = compute_time_series(df, est_name, 30, TIMEFRAME_MAP[timeframe])
        latest = rv_val.iloc[-1] if len(rv_val) > 0 else np.nan
        summary_rows.append({"Asset": asset, "Estimator": est_name, "30d RV (%)": latest})

summary_df = pd.DataFrame(summary_rows)
if not summary_df.empty:
    pivot_summary = summary_df.pivot(index="Estimator", columns="Asset", values="30d RV (%)")
    st.dataframe(
        pivot_summary.style.format("{:.1f}%").background_gradient(cmap="YlOrRd", axis=None),
        width="stretch",
    )

# ---------------------------------------------------------------------------
# Telegram blast
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Telegram Report")

if not is_configured():
    st.info("Telegram not configured. Set credentials in secrets.toml or environment.")
else:
    if st.button("Send RV Matrix to Telegram", type="primary"):
        # Build text report
        lines = ["<b>Realized Volatility Matrix</b>"]
        lines.append(f"Timeframe: {timeframe} | Lookback: {lookback_label}")
        lines.append(f"Generated: {fx_style.local_now():%Y-%m-%d %H:%M} {fx_style.DISPLAY_TZ_LABEL}")
        lines.append("")

        for asset, matrix in matrices.items():
            lines.append(f"<b>{asset}</b>")
            for est_name in matrix.index:
                vals = " | ".join(f"{matrix.loc[est_name, c]:.1f}%" for c in matrix.columns)
                lines.append(f"  {est_name}: {vals}")
            lines.append("")

        msg = "\n".join(lines)
        success = send_message(msg)
        if success:
            st.success("RV matrix sent to Telegram")
        else:
            st.error("Failed to send to Telegram")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.caption(
    f"Data: Deribit TradingView OHLC (perpetual) | "
    f"Last refresh: {fx_style.local_now():%H:%M:%S} {fx_style.DISPLAY_TZ_LABEL}"
)
