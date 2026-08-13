"""
Regime Identifier — Volatility regime classification with GARCH forecasting.

Classifies the current vol regime (Low / Moderate / High) based on trailing
RV percentiles, overlays implied vol from Deribit ATM options, and forecasts
forward vol using a GARCH(1,1) model. All data from Deribit public API.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from lib.deribit import get_tradingview_ohlc, get_atm_iv_by_dte, get_index_price
from lib.constants import ASSET_CONFIG, ASSETS, ASSET_COLORS, PLOTLY_LAYOUT
from lib.vol_math import close_to_close_vol
from lib.telegram import send_message, is_configured

# Try to import arch for GARCH
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Regime Identifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Volatility Regime Identifier")
st.caption("Classify current vol regime with GARCH forecasting and IV-RV comparison")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGIME_COLORS = {
    "Low": "#4CAF50",
    "Moderate": "#FFC107",
    "High": "#F44336",
}

# Percentile thresholds for regime classification
LOW_PCTILE = 40
HIGH_PCTILE = 70

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Parameters")

    selected_asset = st.selectbox("Asset", options=ASSETS, index=0)
    cfg = ASSET_CONFIG[selected_asset]

    rv_window = st.slider(
        "RV Window (days)", min_value=7, max_value=60, value=30, step=1,
        help="Rolling window for close-to-close realized volatility",
    )

    forecast_horizon = st.slider(
        "Forecast Horizon (days)", min_value=5, max_value=60, value=30, step=5,
        help="Number of days to forecast forward (GARCH)",
    )

    st.divider()
    st.markdown("**Regime Thresholds**")
    st.markdown(
        f"- **Low:** < {LOW_PCTILE}th percentile\n"
        f"- **Moderate:** {LOW_PCTILE}th–{HIGH_PCTILE}th percentile\n"
        f"- **High:** > {HIGH_PCTILE}th percentile\n\n"
        f"_Based on trailing 365d RV distribution_"
    )


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Fetching daily OHLC...", ttl=300)
def fetch_daily_ohlc(asset: str, days: int = 400) -> pd.DataFrame | None:
    """Fetch daily OHLC from Deribit for the perpetual.
    Extra days beyond 365 to have enough history for rolling windows."""
    instrument = ASSET_CONFIG[asset]["perp"]
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000

    df = get_tradingview_ohlc(instrument, "1D", start_ms, end_ms)
    if df is None or df.empty:
        return None
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(show_spinner="Fetching ATM IV...", ttl=120)
def fetch_atm_iv(asset: str) -> pd.DataFrame | None:
    """Fetch current ATM IV term structure from Deribit options chain."""
    currency = ASSET_CONFIG[asset]["deribit_ccy"]
    spot = get_index_price(ASSET_CONFIG[asset]["index"])
    return get_atm_iv_by_dte(currency, spot)


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regime(rv_series: pd.Series) -> tuple[str, float, float]:
    """
    Classify the current vol regime based on trailing 365d percentile.
    Returns: (regime_label, current_rv, percentile_rank)
    """
    # Use last 365 data points for distribution
    lookback = rv_series.dropna().tail(365)
    if len(lookback) < 30:
        return "Unknown", np.nan, np.nan

    current_rv = lookback.iloc[-1]
    # Percentile rank of current RV within trailing distribution
    pctile = (lookback < current_rv).sum() / len(lookback) * 100

    if pctile < LOW_PCTILE:
        regime = "Low"
    elif pctile < HIGH_PCTILE:
        regime = "Moderate"
    else:
        regime = "High"

    return regime, current_rv, pctile


def compute_regime_bands(rv_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Compute rolling percentile thresholds for regime boundaries."""
    low_threshold = rv_series.rolling(365, min_periods=60).quantile(LOW_PCTILE / 100)
    high_threshold = rv_series.rolling(365, min_periods=60).quantile(HIGH_PCTILE / 100)
    return low_threshold, high_threshold


# ---------------------------------------------------------------------------
# GARCH forecasting
# ---------------------------------------------------------------------------

def run_garch_forecast(returns: pd.Series, horizon: int = 30) -> dict | None:
    """
    Fit GARCH(1,1) to returns and produce multi-step variance forecast.
    Returns dict with forecast_vol (annualized %), model params, and persistence.
    """
    if not ARCH_AVAILABLE:
        return None

    # Clean returns: drop NaN, scale to percentage for arch stability
    clean = returns.dropna() * 100  # arch works better with percent returns
    if len(clean) < 100:
        return None

    try:
        model = arch_model(clean, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
        result = model.fit(disp="off", show_warning=False)

        # Forecast variance
        forecast = result.forecast(horizon=horizon)
        # forecast.variance gives annualized-daily variance in pct^2 terms
        fcast_var = forecast.variance.iloc[-1].values  # array of length=horizon

        # Convert daily vol (pct) to annualized decimal
        fcast_daily_vol = np.sqrt(fcast_var)  # daily vol in %
        fcast_ann_vol = fcast_daily_vol * np.sqrt(365)  # annualized %

        # Model parameters
        params = result.params
        omega = params.get("omega", 0)
        alpha = params.get("alpha[1]", 0)
        beta = params.get("beta[1]", 0)
        persistence = alpha + beta

        return {
            "forecast_vol": fcast_ann_vol,
            "mean_forecast": float(np.mean(fcast_ann_vol)),
            "terminal_forecast": float(fcast_ann_vol[-1]),
            "omega": float(omega),
            "alpha": float(alpha),
            "beta": float(beta),
            "persistence": float(persistence),
            "unconditional_vol": float(np.sqrt(omega / (1 - persistence)) * np.sqrt(365))
            if persistence < 1 else np.nan,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# Fetch data
ohlc = fetch_daily_ohlc(selected_asset, days=400)

if ohlc is None or ohlc.empty:
    st.error(f"No OHLC data available for {selected_asset}.")
    st.stop()

# Compute realized volatility
rv_series = close_to_close_vol(ohlc["close"], window=rv_window) * 100  # As percentage

# Classify regime
regime, current_rv, pctile = classify_regime(rv_series)

# Compute regime bands
low_band, high_band = compute_regime_bands(rv_series)

# Log returns for GARCH
log_returns = np.log(ohlc["close"] / ohlc["close"].shift(1)).dropna()

# GARCH forecast
garch_result = run_garch_forecast(log_returns, forecast_horizon)

# Fetch IV term structure
iv_df = fetch_atm_iv(selected_asset)

# ---------------------------------------------------------------------------
# Current regime badge
# ---------------------------------------------------------------------------

st.subheader("Current Regime")

badge_cols = st.columns([1, 1, 1, 2])

with badge_cols[0]:
    color = REGIME_COLORS.get(regime, "#888")
    st.markdown(
        f'<div style="background-color:{color}; padding:1rem; border-radius:0.5rem; '
        f'text-align:center; font-weight:bold; font-size:1.5rem; color:white;">'
        f'{regime} Vol</div>',
        unsafe_allow_html=True,
    )

with badge_cols[1]:
    st.metric(
        f"Current RV ({rv_window}d)",
        f"{current_rv:.1f}%" if not np.isnan(current_rv) else "—",
    )

with badge_cols[2]:
    st.metric(
        "Percentile (365d)",
        f"{pctile:.0f}th" if not np.isnan(pctile) else "—",
    )

with badge_cols[3]:
    if garch_result:
        st.metric(
            f"GARCH Forecast ({forecast_horizon}d avg)",
            f"{garch_result['mean_forecast']:.1f}%",
        )
        st.caption(
            f"Persistence: {garch_result['persistence']:.3f} | "
            f"Unconditional: {garch_result['unconditional_vol']:.1f}%"
        )
    elif not ARCH_AVAILABLE:
        st.info("Install `arch` package for GARCH forecasting")
    else:
        st.info("Insufficient data for GARCH model")

# ---------------------------------------------------------------------------
# Price chart with regime-colored background
# ---------------------------------------------------------------------------

st.subheader("Price with Regime Overlay")

fig_price = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.6, 0.4],
    vertical_spacing=0.05,
    subplot_titles=[f"{selected_asset} Price", f"Realized Vol ({rv_window}d)"],
)

# Price line
fig_price.add_trace(
    go.Scatter(
        x=ohlc["timestamp"], y=ohlc["close"],
        name="Price", line=dict(color=ASSET_COLORS.get(selected_asset, "#fff"), width=1.5),
    ),
    row=1, col=1,
)

# Regime background shading on the RV subplot
# Color the background by regime at each point
for i in range(1, len(rv_series)):
    if pd.isna(rv_series.iloc[i]) or pd.isna(low_band.iloc[i]) or pd.isna(high_band.iloc[i]):
        continue
    val = rv_series.iloc[i]
    lo = low_band.iloc[i]
    hi = high_band.iloc[i]
    if val < lo:
        c = "rgba(76,175,80,0.08)"
    elif val > hi:
        c = "rgba(244,67,54,0.08)"
    else:
        c = "rgba(255,193,7,0.05)"

# Instead of per-bar shapes (expensive), add filled bands
fig_price.add_trace(
    go.Scatter(
        x=ohlc["timestamp"], y=low_band,
        name="Low Threshold", line=dict(color=REGIME_COLORS["Low"], dash="dot", width=1),
        showlegend=True,
    ),
    row=2, col=1,
)
fig_price.add_trace(
    go.Scatter(
        x=ohlc["timestamp"], y=high_band,
        name="High Threshold", line=dict(color=REGIME_COLORS["High"], dash="dot", width=1),
        showlegend=True,
    ),
    row=2, col=1,
)

# RV line
fig_price.add_trace(
    go.Scatter(
        x=ohlc["timestamp"], y=rv_series,
        name=f"RV ({rv_window}d)",
        line=dict(color="#fafafa", width=2),
    ),
    row=2, col=1,
)

# Fill between bands
fig_price.add_trace(
    go.Scatter(
        x=ohlc["timestamp"], y=low_band,
        fill=None, mode="lines", line=dict(width=0), showlegend=False,
    ),
    row=2, col=1,
)
fig_price.add_trace(
    go.Scatter(
        x=ohlc["timestamp"], y=high_band,
        fill="tonexty",
        fillcolor="rgba(255,193,7,0.1)",
        mode="lines", line=dict(width=0),
        name="Moderate Zone", showlegend=True,
    ),
    row=2, col=1,
)

fig_price.update_layout(
    **PLOTLY_LAYOUT,
    height=650,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)
fig_price.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig_price.update_yaxes(title_text="RV (%)", row=2, col=1)

st.plotly_chart(fig_price, use_container_width=True)

# ---------------------------------------------------------------------------
# IV vs RV spread
# ---------------------------------------------------------------------------

st.subheader("Implied vs Realized Volatility")

if iv_df is not None and not iv_df.empty:
    iv_cols = st.columns([2, 1])

    with iv_cols[0]:
        fig_ivrv = go.Figure()

        # Current RV as horizontal line
        fig_ivrv.add_hline(
            y=current_rv, line_dash="dash", line_color="#fafafa",
            annotation_text=f"RV {rv_window}d: {current_rv:.1f}%",
        )

        # IV term structure
        fig_ivrv.add_trace(go.Scatter(
            x=iv_df["dte"],
            y=iv_df["atm_iv"],
            mode="lines+markers",
            name="ATM IV",
            line=dict(color="#26a69a", width=2),
            marker=dict(size=6),
        ))

        fig_ivrv.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            title="ATM IV Term Structure vs Current RV",
            xaxis_title="Days to Expiry",
            yaxis_title="Volatility (%)",
        )
        st.plotly_chart(fig_ivrv, use_container_width=True)

    with iv_cols[1]:
        # IV-RV spread summary
        nearest_iv = iv_df.iloc[0]["atm_iv"] if len(iv_df) > 0 else np.nan
        spread = nearest_iv - current_rv if not np.isnan(nearest_iv) else np.nan

        st.metric("Nearest ATM IV", f"{nearest_iv:.1f}%" if not np.isnan(nearest_iv) else "—")
        st.metric(f"Current RV ({rv_window}d)", f"{current_rv:.1f}%")
        st.metric(
            "IV-RV Spread",
            f"{spread:+.1f}%" if not np.isnan(spread) else "—",
            delta_color="normal" if spread and spread > 0 else "inverse",
        )
        if not np.isnan(spread):
            if spread > 5:
                st.caption("IV premium — options expensive relative to realized")
            elif spread < -5:
                st.caption("RV premium — options cheap relative to realized")
            else:
                st.caption("IV and RV roughly in line")
else:
    st.info(f"No ATM IV data available for {selected_asset} (options may not be listed or no near-ATM strikes).")

# ---------------------------------------------------------------------------
# GARCH forecast detail
# ---------------------------------------------------------------------------

if garch_result:
    st.subheader("GARCH(1,1) Forecast")

    garch_cols = st.columns([2, 1])

    with garch_cols[0]:
        # Plot forecast cone
        forecast_days = np.arange(1, forecast_horizon + 1)
        fig_garch = go.Figure()
        fig_garch.add_trace(go.Scatter(
            x=forecast_days,
            y=garch_result["forecast_vol"],
            mode="lines",
            name="Forecast Vol",
            line=dict(color="#ab47bc", width=2),
        ))

        # Current RV reference
        fig_garch.add_hline(
            y=current_rv, line_dash="dash", line_color="#fafafa",
            annotation_text=f"Current RV: {current_rv:.1f}%",
        )

        # Unconditional vol reference
        if not np.isnan(garch_result["unconditional_vol"]):
            fig_garch.add_hline(
                y=garch_result["unconditional_vol"],
                line_dash="dot", line_color="#78909c",
                annotation_text=f"Unconditional: {garch_result['unconditional_vol']:.1f}%",
            )

        fig_garch.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            title=f"GARCH(1,1) Vol Forecast — {forecast_horizon}d",
            xaxis_title="Days Ahead",
            yaxis_title="Annualized Vol (%)",
        )
        st.plotly_chart(fig_garch, use_container_width=True)

    with garch_cols[1]:
        st.markdown("**Model Parameters**")
        st.markdown(f"- alpha (news): `{garch_result['alpha']:.4f}`")
        st.markdown(f"- beta (persistence): `{garch_result['beta']:.4f}`")
        st.markdown(f"- omega (constant): `{garch_result['omega']:.6f}`")
        st.markdown(f"- **Persistence (a+b):** `{garch_result['persistence']:.4f}`")
        st.divider()
        st.markdown("**Forecast Summary**")
        st.markdown(f"- Mean forecast: **{garch_result['mean_forecast']:.1f}%**")
        st.markdown(f"- Terminal ({forecast_horizon}d): **{garch_result['terminal_forecast']:.1f}%**")
        if not np.isnan(garch_result["unconditional_vol"]):
            st.markdown(f"- Unconditional: **{garch_result['unconditional_vol']:.1f}%**")

        if garch_result["persistence"] >= 0.99:
            st.warning("Very high persistence — shocks decay slowly")
        elif garch_result["persistence"] < 0.9:
            st.info("Low persistence — vol reverts quickly")

# ---------------------------------------------------------------------------
# Telegram blast
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Telegram Report")

if not is_configured():
    st.info("Telegram not configured. Set credentials in secrets.toml or environment.")
else:
    if st.button("Send Regime Report to Telegram", type="primary"):
        lines = [f"<b>{selected_asset} Volatility Regime Report</b>"]
        lines.append(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        lines.append("")
        lines.append(f"Regime: <b>{regime}</b>")
        lines.append(f"RV ({rv_window}d): {current_rv:.1f}%")
        lines.append(f"Percentile: {pctile:.0f}th (365d trailing)")
        lines.append("")

        if iv_df is not None and not iv_df.empty:
            nearest_iv_val = iv_df.iloc[0]["atm_iv"]
            spread_val = nearest_iv_val - current_rv
            lines.append(f"ATM IV (nearest): {nearest_iv_val:.1f}%")
            lines.append(f"IV-RV Spread: {spread_val:+.1f}%")
            lines.append("")

        if garch_result:
            lines.append(f"GARCH Forecast ({forecast_horizon}d):")
            lines.append(f"  Mean: {garch_result['mean_forecast']:.1f}%")
            lines.append(f"  Terminal: {garch_result['terminal_forecast']:.1f}%")
            lines.append(f"  Persistence: {garch_result['persistence']:.3f}")

        msg = "\n".join(lines)
        success = send_message(msg)
        if success:
            st.success("Regime report sent to Telegram")
        else:
            st.error("Failed to send to Telegram")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.caption(
    f"Data: Deribit 1D OHLC ({ASSET_CONFIG[selected_asset]['perp']}) | "
    f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
)
