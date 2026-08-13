"""
MCM Analytics — Vol Z-Score Analysis
How current implied volatility compares to its recent history.
Uses DVOL history for BTC/ETH and live surface data for all assets.
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
from datetime import datetime, timezone

from lib.deribit import get_dvol, get_option_chain, get_index_price
from lib.constants import ASSET_CONFIG, ASSET_COLORS, PLOTLY_LAYOUT, ASSETS
from lib.instruments import parse_instrument
from lib.vol_math import bs_delta
from lib.telegram import send_photo, is_configured

st.set_page_config(page_title="Z-Score Analysis", page_icon="📊", layout="wide")
st.title("📊 Vol Z-Score Analysis")
st.caption("How current implied volatility compares to recent history • DVOL z-scores (BTC/ETH) • Live surface percentiles (all assets)")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

asset = st.sidebar.selectbox("Asset", ASSETS, index=0)
lookback_days = st.sidebar.selectbox("DVOL Lookback (days)", [30, 60, 90, 180, 365], index=2)
z_windows = st.sidebar.multiselect(
    "Z-Score Windows",
    options=[30, 60, 90],
    default=[30, 60, 90],
)

cfg = ASSET_CONFIG[asset]


# ---------------------------------------------------------------------------
# Helper: z-score color
# ---------------------------------------------------------------------------

def zscore_color(z: float) -> str:
    """Return color for a z-score value."""
    if z <= -2:
        return "#00c853"  # deep green (very cheap)
    elif z <= -1:
        return "#69f0ae"  # light green (cheap)
    elif z >= 2:
        return "#ff1744"  # deep red (very expensive)
    elif z >= 1:
        return "#ff8a80"  # light red (expensive)
    return "#ffeb3b"  # yellow (neutral)


def zscore_label(z: float) -> str:
    """Return text label for a z-score value."""
    if z <= -2:
        return "Very Cheap"
    elif z <= -1:
        return "Cheap"
    elif z >= 2:
        return "Very Expensive"
    elif z >= 1:
        return "Expensive"
    return "Normal"


def percentile_rank(series: pd.Series, value: float) -> float:
    """Compute the percentile rank of a value within a series."""
    return (series < value).sum() / len(series) * 100


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_dvol_history(asset_key: str, days: int) -> pd.DataFrame | None:
    """Fetch DVOL daily history for an asset."""
    cfg = ASSET_CONFIG[asset_key]
    if not cfg["has_dvol"]:
        return None
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000
    return get_dvol(cfg["deribit_ccy"], "1D", start_ms, now_ms)


@st.cache_data(ttl=300)
def fetch_option_chain_data(asset_key: str) -> tuple[list[dict] | None, float | None]:
    """Fetch the full option chain and spot price for an asset."""
    cfg = ASSET_CONFIG[asset_key]
    chain = get_option_chain(cfg["deribit_ccy"], "option")
    spot = get_index_price(cfg["index"])
    return chain, spot


# ---------------------------------------------------------------------------
# DVOL Z-Score Section (BTC/ETH only)
# ---------------------------------------------------------------------------

if cfg["has_dvol"]:
    st.header(f"{asset} DVOL Z-Scores")

    with st.spinner(f"Fetching {asset} DVOL history ({lookback_days}d)..."):
        dvol_df = fetch_dvol_history(asset, lookback_days)

    if dvol_df is None or dvol_df.empty:
        st.error(f"Could not fetch DVOL data for {asset}.")
        st.stop()

    dvol_df = dvol_df.sort_values("timestamp").reset_index(drop=True)
    current_dvol = dvol_df["close"].iloc[-1]

    # Compute z-scores for each window
    zscore_results = []
    for window in z_windows:
        if len(dvol_df) < window:
            continue
        window_data = dvol_df["close"].iloc[-window:]
        mean = window_data.mean()
        std = window_data.std()
        if std > 0:
            z = (current_dvol - mean) / std
        else:
            z = 0.0
        pct = percentile_rank(window_data, current_dvol)
        zscore_results.append({
            "Window": f"{window}d",
            "Mean": mean,
            "Std Dev": std,
            "Current": current_dvol,
            "Z-Score": z,
            "Percentile": pct,
            "Signal": zscore_label(z),
        })

    # Metrics row
    if zscore_results:
        cols = st.columns(len(zscore_results) + 1)
        cols[0].metric(f"{asset} DVOL", f"{current_dvol:.1f}%")
        for i, row in enumerate(zscore_results):
            z = row["Z-Score"]
            cols[i + 1].metric(
                f"{row['Window']} Z-Score",
                f"{z:+.2f}",
                delta=f"{row['Percentile']:.0f}th pctl",
                delta_color="off",
            )

    st.divider()

    # ---------------------------------------------------------------------------
    # DVOL Historical Distribution
    # ---------------------------------------------------------------------------

    col_hist, col_ts = st.columns(2)

    with col_hist:
        st.subheader("DVOL Distribution")

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=dvol_df["close"],
                nbinsx=40,
                marker_color=ASSET_COLORS[asset],
                opacity=0.7,
                name="DVOL",
            )
        )
        # Mark current level
        fig_hist.add_vline(
            x=current_dvol,
            line_dash="dash",
            line_color="white",
            line_width=2,
            annotation_text=f"Current: {current_dvol:.1f}%",
            annotation_font_color="white",
        )
        # Mark mean
        dvol_mean = dvol_df["close"].mean()
        fig_hist.add_vline(
            x=dvol_mean,
            line_dash="dot",
            line_color="#ffeb3b",
            line_width=1,
            annotation_text=f"Mean: {dvol_mean:.1f}%",
            annotation_font_color="#ffeb3b",
            annotation_position="top left",
        )

        fig_hist.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            xaxis_title="DVOL (%)",
            yaxis_title="Frequency",
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_ts:
        st.subheader("DVOL Time Series")

        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Scatter(
                x=dvol_df["timestamp"],
                y=dvol_df["close"],
                name="DVOL",
                line=dict(color=ASSET_COLORS[asset], width=2),
            )
        )
        # Add mean and +/- 1 std bands
        dvol_std = dvol_df["close"].std()
        fig_ts.add_hline(y=dvol_mean, line_dash="dash", line_color="#ffeb3b", opacity=0.6,
                         annotation_text="Mean")
        fig_ts.add_hline(y=dvol_mean + dvol_std, line_dash="dot", line_color="#ff8a80", opacity=0.4,
                         annotation_text="+1σ")
        fig_ts.add_hline(y=dvol_mean - dvol_std, line_dash="dot", line_color="#69f0ae", opacity=0.4,
                         annotation_text="-1σ")
        fig_ts.add_hline(y=dvol_mean + 2 * dvol_std, line_dash="dot", line_color="#ff1744", opacity=0.3,
                         annotation_text="+2σ")
        fig_ts.add_hline(y=dvol_mean - 2 * dvol_std, line_dash="dot", line_color="#00c853", opacity=0.3,
                         annotation_text="-2σ")

        fig_ts.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            yaxis_title="DVOL (%)",
            showlegend=False,
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # ---------------------------------------------------------------------------
    # Z-Score Table
    # ---------------------------------------------------------------------------

    if zscore_results:
        st.subheader("Z-Score Summary Table")

        z_df = pd.DataFrame(zscore_results)
        z_df["Mean"] = z_df["Mean"].apply(lambda x: f"{x:.2f}%")
        z_df["Std Dev"] = z_df["Std Dev"].apply(lambda x: f"{x:.2f}%")
        z_df["Current"] = z_df["Current"].apply(lambda x: f"{x:.2f}%")
        z_df["Z-Score"] = z_df["Z-Score"].apply(lambda x: f"{x:+.2f}")
        z_df["Percentile"] = z_df["Percentile"].apply(lambda x: f"{x:.0f}th")

        st.dataframe(z_df, use_container_width=True, hide_index=True)

    st.divider()

else:
    st.info(
        f"**{asset}** does not have a DVOL index on Deribit. "
        "DVOL z-scores require historical volatility index data that is only available for BTC and ETH. "
        "Live surface analysis is shown below."
    )

# ---------------------------------------------------------------------------
# Live Surface Analysis (All Assets)
# ---------------------------------------------------------------------------

st.header(f"{asset} Live Term Structure Analysis")

with st.spinner(f"Fetching {asset} option chain..."):
    chain, spot = fetch_option_chain_data(asset)

if chain is None or spot is None:
    st.error(f"Could not fetch option chain data for {asset}.")
    st.stop()

# Parse the chain and compute per-expiry ATM IV
now_ms = int(time.time() * 1000)
prefix = cfg["deribit_prefix"]

expiry_data: dict[str, list[dict]] = {}

for inst in chain:
    name = inst.get("instrument_name", "")
    if not name.startswith(prefix):
        continue

    parsed = parse_instrument(name)
    if parsed is None:
        continue

    mark_iv = inst.get("mark_iv")
    if mark_iv is None or mark_iv == 0:
        continue

    # mark_iv from chain is in decimal (0.55 = 55%)
    iv_pct = mark_iv * 100

    expiry_ts = inst.get("expiration_timestamp", 0)
    dte = max((expiry_ts - now_ms) / (1000 * 86400), 0.01)
    strike = parsed.strike
    kind = parsed.kind

    # Moneyness
    moneyness = np.log(strike / spot)

    expiry_key = parsed.expiry_str
    if expiry_key not in expiry_data:
        expiry_data[expiry_key] = []

    expiry_data[expiry_key].append({
        "strike": strike,
        "kind": kind,
        "iv": iv_pct,
        "moneyness": moneyness,
        "dte": dte,
        "expiry_ts": expiry_ts,
        "delta": bs_delta(spot, strike, dte / 365, mark_iv, kind),
    })

if not expiry_data:
    st.warning("No valid option data found in the chain.")
    st.stop()

# Compute ATM IV and skew per expiry
term_structure = []
for expiry_key, options in expiry_data.items():
    if not options:
        continue
    dte = options[0]["dte"]
    expiry_ts = options[0]["expiry_ts"]

    # ATM = closest to spot (moneyness near 0)
    atm_options = [o for o in options if abs(o["moneyness"]) < 0.05]
    if not atm_options:
        # Fallback: pick the two closest to ATM
        sorted_by_m = sorted(options, key=lambda x: abs(x["moneyness"]))
        atm_options = sorted_by_m[:4]

    if not atm_options:
        continue

    atm_iv = np.mean([o["iv"] for o in atm_options])

    # 25-delta skew: put IV - call IV at ~25 delta
    calls = [o for o in options if o["kind"] == "C"]
    puts = [o for o in options if o["kind"] == "P"]

    # 25d call: delta closest to 0.25
    call_25d = None
    if calls:
        calls_by_delta = sorted(calls, key=lambda o: abs(abs(o["delta"]) - 0.25))
        if calls_by_delta and abs(abs(calls_by_delta[0]["delta"]) - 0.25) < 0.15:
            call_25d = calls_by_delta[0]["iv"]

    # 25d put: delta closest to -0.25
    put_25d = None
    if puts:
        puts_by_delta = sorted(puts, key=lambda o: abs(abs(o["delta"]) - 0.25))
        if puts_by_delta and abs(abs(puts_by_delta[0]["delta"]) - 0.25) < 0.15:
            put_25d = puts_by_delta[0]["iv"]

    skew_25d = (put_25d - call_25d) if (put_25d is not None and call_25d is not None) else None

    term_structure.append({
        "expiry": expiry_key,
        "dte": dte,
        "atm_iv": atm_iv,
        "skew_25d": skew_25d,
        "expiry_ts": expiry_ts,
    })

term_structure = sorted(term_structure, key=lambda x: x["dte"])

if not term_structure:
    st.warning("Could not compute term structure from available data.")
    st.stop()

ts_df = pd.DataFrame(term_structure)

# ---------------------------------------------------------------------------
# Term Structure Chart
# ---------------------------------------------------------------------------

st.subheader("ATM IV Term Structure")

fig_term = go.Figure()
fig_term.add_trace(
    go.Scatter(
        x=ts_df["dte"],
        y=ts_df["atm_iv"],
        mode="lines+markers",
        name="ATM IV",
        line=dict(color=ASSET_COLORS[asset], width=2),
        marker=dict(size=8),
        text=ts_df["expiry"],
        hovertemplate="%{text}<br>DTE: %{x:.0f}<br>ATM IV: %{y:.1f}%<extra></extra>",
    )
)

fig_term.update_layout(
    **PLOTLY_LAYOUT,
    height=350,
    xaxis_title="Days to Expiry",
    yaxis_title="ATM IV (%)",
)
st.plotly_chart(fig_term, use_container_width=True)

# ---------------------------------------------------------------------------
# Term Structure Z-Scores vs DVOL (BTC/ETH only)
# ---------------------------------------------------------------------------

if cfg["has_dvol"] and dvol_df is not None and not dvol_df.empty:
    st.subheader("Per-Tenor Z-Scores (vs DVOL History)")
    st.caption("How current ATM IV at each tenor compares to the DVOL distribution over the lookback window")

    tenor_z_results = []
    for _, row in ts_df.iterrows():
        atm_iv = row["atm_iv"]
        for window in z_windows:
            if len(dvol_df) < window:
                continue
            window_data = dvol_df["close"].iloc[-window:]
            mean = window_data.mean()
            std = window_data.std()
            z = (atm_iv - mean) / std if std > 0 else 0.0
            pct = percentile_rank(window_data, atm_iv)
            tenor_z_results.append({
                "Expiry": row["expiry"],
                "DTE": f"{row['dte']:.0f}d",
                "ATM IV": f"{atm_iv:.1f}%",
                f"{window}d Z": z,
                f"{window}d Pctl": pct,
            })

    if tenor_z_results:
        # Consolidate into one row per expiry with multiple windows
        consolidated = {}
        for r in tenor_z_results:
            key = r["Expiry"]
            if key not in consolidated:
                consolidated[key] = {"Expiry": r["Expiry"], "DTE": r["DTE"], "ATM IV": r["ATM IV"]}
            for k, v in r.items():
                if k not in ("Expiry", "DTE", "ATM IV"):
                    consolidated[key][k] = v

        tenor_df = pd.DataFrame(list(consolidated.values()))

        # Color-coded display using styled dataframe
        def color_zscore(val):
            """Style z-score cells."""
            if not isinstance(val, (int, float)):
                return ""
            color = zscore_color(val)
            return f"background-color: {color}; color: black; font-weight: bold"

        z_cols = [c for c in tenor_df.columns if "Z" in c]
        pctl_cols = [c for c in tenor_df.columns if "Pctl" in c]

        # Format for display
        display_df = tenor_df.copy()
        for col in pctl_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}th" if isinstance(x, (int, float)) else x)

        styled = display_df.style.applymap(color_zscore, subset=z_cols)
        for col in z_cols:
            styled = styled.format({col: "{:+.2f}"}, subset=[col])

        st.dataframe(styled, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Skew Z-Score Analysis
# ---------------------------------------------------------------------------

skew_data = ts_df[ts_df["skew_25d"].notna()].copy()

if not skew_data.empty:
    st.subheader("25-Delta Skew by Tenor")

    col_skew_chart, col_skew_table = st.columns([2, 1])

    with col_skew_chart:
        fig_skew = go.Figure()
        fig_skew.add_trace(
            go.Bar(
                x=skew_data["expiry"],
                y=skew_data["skew_25d"],
                marker_color=[
                    "#ff8a80" if s > 0 else "#69f0ae" for s in skew_data["skew_25d"]
                ],
                name="25d Skew",
                hovertemplate="%{x}<br>Skew: %{y:.1f}%<extra></extra>",
            )
        )
        fig_skew.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_skew.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            xaxis_title="Expiry",
            yaxis_title="25d Skew (Put IV - Call IV, %)",
        )
        st.plotly_chart(fig_skew, use_container_width=True)

    with col_skew_table:
        skew_display = skew_data[["expiry", "dte", "skew_25d"]].copy()
        skew_display.columns = ["Expiry", "DTE", "Skew (%)"]
        skew_display["DTE"] = skew_display["DTE"].apply(lambda x: f"{x:.0f}d")
        skew_display["Skew (%)"] = skew_display["Skew (%)"].apply(lambda x: f"{x:+.1f}%")
        st.dataframe(skew_display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Color-Coded Z-Score Heatmap (DVOL assets)
# ---------------------------------------------------------------------------

if cfg["has_dvol"] and dvol_df is not None and not dvol_df.empty:
    st.divider()
    st.subheader("Z-Score Heatmap")
    st.caption(
        "Green = cheap vol (z < -1) • Yellow = normal • Red = expensive (z > 1). "
        "Based on DVOL close prices over the lookback window."
    )

    # Build heatmap: rows = expiries, cols = windows
    heatmap_data = []
    for _, row in ts_df.iterrows():
        atm_iv = row["atm_iv"]
        row_data = {"Expiry": row["expiry"], "DTE": f"{row['dte']:.0f}d", "ATM IV (%)": f"{atm_iv:.1f}"}
        for window in z_windows:
            if len(dvol_df) < window:
                row_data[f"{window}d Z"] = np.nan
                continue
            window_data = dvol_df["close"].iloc[-window:]
            mean = window_data.mean()
            std = window_data.std()
            z = (atm_iv - mean) / std if std > 0 else 0.0
            row_data[f"{window}d Z"] = z
        heatmap_data.append(row_data)

    heatmap_df = pd.DataFrame(heatmap_data)

    # Create plotly heatmap
    z_cols = [c for c in heatmap_df.columns if c.endswith("Z")]
    if z_cols:
        z_values = heatmap_df[z_cols].values
        expiry_labels = [f"{row['Expiry']} ({row['DTE']})" for _, row in heatmap_df.iterrows()]

        fig_hm = go.Figure(data=go.Heatmap(
            z=z_values,
            x=[c.replace(" Z", "") for c in z_cols],
            y=expiry_labels,
            colorscale=[
                [0, "#00c853"],
                [0.25, "#69f0ae"],
                [0.5, "#ffeb3b"],
                [0.75, "#ff8a80"],
                [1.0, "#ff1744"],
            ],
            zmid=0,
            zmin=-3,
            zmax=3,
            text=np.round(z_values, 2),
            texttemplate="%{text:+.2f}",
            textfont=dict(size=12),
            colorbar=dict(title="Z-Score"),
            hovertemplate="Expiry: %{y}<br>Window: %{x}<br>Z-Score: %{z:+.2f}<extra></extra>",
        ))

        fig_hm.update_layout(
            **PLOTLY_LAYOUT,
            height=max(300, len(expiry_labels) * 35 + 100),
            xaxis_title="Lookback Window",
            yaxis_title="Expiry",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

# ---------------------------------------------------------------------------
# Interpretation Guide
# ---------------------------------------------------------------------------

st.divider()

with st.expander("How to interpret Z-Scores", expanded=False):
    st.markdown("""
**Z-Score** measures how many standard deviations current IV is from the mean:
- **Z <= -2** (Deep Green): Vol is very cheap relative to history — potential buy signal for options
- **Z between -1 and -2** (Light Green): Vol is below average — leaning cheap
- **Z between -1 and +1** (Yellow): Vol is in normal range
- **Z between +1 and +2** (Light Red): Vol is above average — leaning expensive
- **Z >= +2** (Deep Red): Vol is very expensive — potential sell signal for options

**Percentile Rank** shows what percentage of historical observations were below the current level.

**Caveats:**
- Z-scores assume roughly normal distribution of vol — extreme regimes may not be well-captured
- DVOL is a 30-day forward-looking index; comparing tenor-specific IVs to DVOL is approximate
- For SOL and HYPE (no DVOL), only the current snapshot of the surface is available
- Shorter lookback windows are more responsive but noisier; longer windows provide more context
""")

# ---------------------------------------------------------------------------
# Telegram Blast
# ---------------------------------------------------------------------------

if is_configured() and cfg["has_dvol"]:
    st.divider()
    if st.button("📤 Send to Telegram"):
        # Send the term structure chart
        img = fig_term.to_image(format="png", width=1200, height=500)
        z_summary = " | ".join(
            f"{r['Window']}: {r['Z-Score']:+.2f}" for r in zscore_results
        ) if zscore_results else "N/A"
        caption = (
            f"<b>{asset} Vol Z-Scores ({lookback_days}d lookback)</b>\n"
            f"DVOL: {current_dvol:.1f}% | {z_summary}\n"
            f"Signal: {zscore_results[0]['Signal'] if zscore_results else 'N/A'}"
        )
        if send_photo(img, caption):
            st.success("Sent!")
        else:
            st.error("Failed to send.")
