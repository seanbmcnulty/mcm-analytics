"""
Constant Maturity — Live constant-maturity implied volatility derived from
the Deribit option chain.

Interpolates between actual expiries to show IV at fixed tenors (7d, 14d,
30d, 60d, 90d, 180d, 365d). Includes term structure, skew analysis, and
multi-asset comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline, interp1d
import streamlit as st

from lib.deribit import get_option_chain, get_index_price
from lib.constants import ASSET_CONFIG, ASSETS, ASSET_COLORS, PLOTLY_LAYOUT
from lib.instruments import parse_instrument, parse_expiry_date
from lib.vol_math import bs_delta

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Constant Maturity",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_TENORS = [7, 14, 30, 60, 90, 180, 365]
TENOR_LABELS = ["7d", "14d", "30d", "60d", "90d", "180d", "365d"]
MIN_DTE = 1.0       # Ignore expiries with less than 1 day
MIN_OI = 10         # Minimum total OI per expiry to consider
DELTA_PUT_25 = -0.25
DELTA_CALL_25 = 0.25

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "cm_history" not in st.session_state:
    # Store historical snapshots: {asset: [{timestamp, tenors: {7: iv, 14: iv, ...}}]}
    st.session_state.cm_history = defaultdict(list)

if "cm_skew_history" not in st.session_state:
    st.session_state.cm_skew_history = defaultdict(list)

MAX_HISTORY_POINTS = 180  # ~6 hours at 2-min intervals


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120, show_spinner=False)
def fetch_chain_data(currency: str) -> list[dict] | None:
    """Fetch full option chain for a currency from Deribit."""
    cfg = ASSET_CONFIG.get(currency)
    if not cfg:
        return None
    chain = get_option_chain(cfg["deribit_ccy"], "option")
    if not chain:
        return None
    # Filter to only this asset's options
    prefix = cfg["deribit_prefix"]
    return [item for item in chain if item["instrument_name"].startswith(prefix)]


@st.cache_data(ttl=30, show_spinner=False)
def fetch_spot_price(currency: str) -> float | None:
    """Fetch current spot/index price for an asset."""
    cfg = ASSET_CONFIG.get(currency)
    if not cfg:
        return None
    return get_index_price(cfg["index"])


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def build_expiry_data(chain: list[dict], spot: float, currency: str) -> pd.DataFrame:
    """
    Process the option chain into per-expiry ATM IV and 25-delta skew.

    For each expiry:
    - Find ATM IV (strike nearest to spot, weighted by OI)
    - Find 25-delta put IV and 25-delta call IV via delta interpolation
    - Compute DTE

    Returns DataFrame with columns: expiry_str, dte, atm_iv, put25_iv, call25_iv, skew_25d
    """
    # Group options by expiry
    expiry_groups: dict[str, list[dict]] = defaultdict(list)
    for item in chain:
        parsed = parse_instrument(item["instrument_name"])
        if parsed is None:
            continue
        expiry_groups[parsed.expiry_str].append({
            **item,
            "strike": parsed.strike,
            "kind": parsed.kind,
        })

    rows = []
    now = datetime.datetime.now(datetime.timezone.utc)

    for expiry_str, options in expiry_groups.items():
        expiry_dt = parse_expiry_date(expiry_str)
        dte = (expiry_dt - now).total_seconds() / 86400.0

        if dte < MIN_DTE:
            continue

        # Check total OI for this expiry
        total_oi = sum(opt.get("open_interest", 0) or 0 for opt in options)
        if total_oi < MIN_OI:
            continue

        tte = dte / 365.0  # time to expiry in years

        # Separate calls and puts
        calls = [o for o in options if o["kind"] == "C" and o.get("mark_iv") and o["mark_iv"] > 0]
        puts = [o for o in options if o["kind"] == "P" and o.get("mark_iv") and o["mark_iv"] > 0]

        if not calls and not puts:
            continue

        # --- ATM IV ---
        # Use calls nearest to spot, weighted by OI
        atm_iv = _compute_atm_iv(calls, puts, spot)
        if atm_iv is None:
            continue

        # --- 25-delta skew ---
        put25_iv = _interpolate_delta_iv(puts, spot, tte, DELTA_PUT_25, "P")
        call25_iv = _interpolate_delta_iv(calls, spot, tte, DELTA_CALL_25, "C")

        skew_25d = None
        if put25_iv is not None and call25_iv is not None:
            skew_25d = put25_iv - call25_iv

        rows.append({
            "expiry_str": expiry_str,
            "dte": dte,
            "atm_iv": atm_iv,
            "put25_iv": put25_iv,
            "call25_iv": call25_iv,
            "skew_25d": skew_25d,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("dte").reset_index(drop=True)
    return df


def _compute_atm_iv(calls: list[dict], puts: list[dict], spot: float) -> float | None:
    """Compute ATM IV by finding strikes nearest to spot and averaging mark_iv."""
    all_options = calls + puts

    if not all_options:
        return None

    # Sort by distance from spot
    for opt in all_options:
        opt["_distance"] = abs(opt["strike"] - spot)

    sorted_opts = sorted(all_options, key=lambda x: x["_distance"])

    # Take the closest 2 strikes (one on each side if possible)
    # and OI-weight their IVs
    candidates = sorted_opts[:4]  # top 4 nearest strikes
    total_oi = sum(max(c.get("open_interest", 0) or 0, 1) for c in candidates)

    if total_oi == 0:
        # Equal-weight fallback
        return np.mean([c["mark_iv"] for c in candidates]) * 100.0

    weighted_iv = sum(
        c["mark_iv"] * max(c.get("open_interest", 0) or 0, 1) / total_oi
        for c in candidates
    )
    return weighted_iv * 100.0  # Convert to percentage


def _interpolate_delta_iv(options: list[dict], spot: float, tte: float,
                          target_delta: float, option_type: str) -> float | None:
    """
    Interpolate IV at a target delta by computing BS delta for each strike
    and finding the bracketing strikes.
    """
    if not options or tte <= 0:
        return None

    # Compute delta for each option
    delta_iv_pairs = []
    for opt in options:
        strike = opt["strike"]
        iv = opt["mark_iv"]  # Deribit mark_iv is decimal (e.g., 0.55 = 55%)
        if iv <= 0:
            continue
        d = bs_delta(spot, strike, tte, iv, option_type)
        delta_iv_pairs.append((d, iv * 100.0, strike))

    if len(delta_iv_pairs) < 2:
        return None

    # Sort by delta
    delta_iv_pairs.sort(key=lambda x: x[0])

    deltas = [p[0] for p in delta_iv_pairs]
    ivs = [p[1] for p in delta_iv_pairs]

    # Find bracketing pair
    for i in range(len(deltas) - 1):
        d_low, d_high = deltas[i], deltas[i + 1]
        if (d_low <= target_delta <= d_high) or (d_high <= target_delta <= d_low):
            # Linear interpolation
            if abs(d_high - d_low) < 1e-10:
                return ivs[i]
            frac = (target_delta - d_low) / (d_high - d_low)
            return ivs[i] + frac * (ivs[i + 1] - ivs[i])

    # If target delta is outside range, use nearest
    abs_diffs = [abs(d - target_delta) for d in deltas]
    nearest_idx = int(np.argmin(abs_diffs))
    # Only use if reasonably close (within 0.10 delta)
    if abs_diffs[nearest_idx] < 0.10:
        return ivs[nearest_idx]

    return None


def interpolate_constant_maturity(expiry_df: pd.DataFrame,
                                  tenors: list[int],
                                  method: str = "cubic") -> dict[int, float | None]:
    """
    Interpolate ATM IV at standard tenors using the term structure from actual expiries.

    Returns dict mapping tenor (days) -> interpolated IV (or None if outside range).
    """
    if expiry_df.empty or len(expiry_df) < 2:
        return {t: None for t in tenors}

    dtes = expiry_df["dte"].values
    ivs = expiry_df["atm_iv"].values

    # Remove NaN
    mask = ~np.isnan(ivs)
    dtes = dtes[mask]
    ivs = ivs[mask]

    if len(dtes) < 2:
        return {t: None for t in tenors}

    # Build interpolator
    if method == "cubic" and len(dtes) >= 4:
        interp = CubicSpline(dtes, ivs, extrapolate=False)
    else:
        interp = interp1d(dtes, ivs, kind="linear", bounds_error=False, fill_value=np.nan)

    result = {}
    for tenor in tenors:
        val = float(interp(tenor))
        result[tenor] = val if not np.isnan(val) else None

    return result


def interpolate_constant_maturity_skew(expiry_df: pd.DataFrame,
                                       tenors: list[int],
                                       method: str = "cubic") -> dict[int, float | None]:
    """Interpolate 25-delta skew at standard tenors."""
    df = expiry_df.dropna(subset=["skew_25d"])
    if df.empty or len(df) < 2:
        return {t: None for t in tenors}

    dtes = df["dte"].values
    skews = df["skew_25d"].values

    if len(dtes) < 2:
        return {t: None for t in tenors}

    if method == "cubic" and len(dtes) >= 4:
        interp = CubicSpline(dtes, skews, extrapolate=False)
    else:
        interp = interp1d(dtes, skews, kind="linear", bounds_error=False, fill_value=np.nan)

    result = {}
    for tenor in tenors:
        val = float(interp(tenor))
        result[tenor] = val if not np.isnan(val) else None

    return result


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_term_structure_chart(expiry_df: pd.DataFrame, cm_ivs: dict[int, float | None],
                               asset: str) -> go.Figure:
    """
    Build term structure chart showing actual expiry ATM IVs and interpolated
    constant-maturity line.
    """
    fig = go.Figure()
    color = ASSET_COLORS.get(asset, "#ffffff")

    # Actual expiry points
    fig.add_trace(go.Scatter(
        x=expiry_df["dte"],
        y=expiry_df["atm_iv"],
        mode="markers",
        name="Listed Expiries",
        marker=dict(size=10, color=color, symbol="circle", line=dict(width=1, color="#ffffff")),
        hovertemplate="DTE: %{x:.1f}<br>ATM IV: %{y:.1f}%<extra></extra>",
    ))

    # Interpolated constant-maturity line (smooth)
    valid_tenors = [(t, v) for t, v in cm_ivs.items() if v is not None]
    if valid_tenors:
        # Create smooth curve through all valid points
        t_vals = [p[0] for p in valid_tenors]
        iv_vals = [p[1] for p in valid_tenors]

        # Dense interpolation for smooth line
        if len(t_vals) >= 2:
            t_dense = np.linspace(min(t_vals), max(t_vals), 200)
            if len(t_vals) >= 4:
                cs = CubicSpline(t_vals, iv_vals)
                iv_dense = cs(t_dense)
            else:
                interp_fn = interp1d(t_vals, iv_vals, kind="linear")
                iv_dense = interp_fn(t_dense)

            fig.add_trace(go.Scatter(
                x=t_dense,
                y=iv_dense,
                mode="lines",
                name="CM Interpolation",
                line=dict(color=color, width=2, dash="dash"),
                hoverinfo="skip",
            ))

        # Standard tenor markers
        fig.add_trace(go.Scatter(
            x=t_vals,
            y=iv_vals,
            mode="markers+text",
            name="Standard Tenors",
            marker=dict(size=12, color=color, symbol="diamond",
                        line=dict(width=2, color="#ffffff")),
            text=[f"{t}d" for t in t_vals],
            textposition="top center",
            textfont=dict(size=10, color="#fafafa"),
            hovertemplate="Tenor: %{text}<br>CM IV: %{y:.2f}%<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{asset} ATM Term Structure",
        xaxis_title="Days to Expiry",
        yaxis_title="Implied Volatility (%)",
        height=450,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    )

    return fig


def build_skew_chart(expiry_df: pd.DataFrame, cm_skews: dict[int, float | None],
                     asset: str) -> go.Figure:
    """Build chart showing 25-delta skew at actual expiries and CM tenors."""
    fig = go.Figure()
    color = ASSET_COLORS.get(asset, "#ffffff")

    # Actual expiry skews
    df_skew = expiry_df.dropna(subset=["skew_25d"])
    if not df_skew.empty:
        fig.add_trace(go.Scatter(
            x=df_skew["dte"],
            y=df_skew["skew_25d"],
            mode="markers",
            name="Listed Expiries",
            marker=dict(size=8, color=color, symbol="circle",
                        line=dict(width=1, color="#ffffff")),
            hovertemplate="DTE: %{x:.1f}<br>Skew: %{y:.2f}%<extra></extra>",
        ))

    # CM skew markers
    valid = [(t, v) for t, v in cm_skews.items() if v is not None]
    if valid:
        t_vals = [p[0] for p in valid]
        s_vals = [p[1] for p in valid]
        fig.add_trace(go.Scatter(
            x=t_vals,
            y=s_vals,
            mode="markers+lines",
            name="CM Skew",
            marker=dict(size=10, color=color, symbol="diamond",
                        line=dict(width=2, color="#ffffff")),
            line=dict(color=color, width=1.5, dash="dot"),
            hovertemplate="Tenor: %{x}d<br>Skew: %{y:.2f}%<extra></extra>",
        ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{asset} 25-Delta Skew (Put - Call)",
        xaxis_title="Days to Expiry",
        yaxis_title="Skew (% IV)",
        height=400,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    )

    return fig


def build_multi_asset_chart(all_cm_ivs: dict[str, dict[int, float | None]]) -> go.Figure:
    """Build comparison chart of CM IVs across multiple assets."""
    fig = go.Figure()

    for asset, cm_ivs in all_cm_ivs.items():
        valid = [(t, v) for t, v in cm_ivs.items() if v is not None]
        if not valid:
            continue
        t_vals = [p[0] for p in valid]
        iv_vals = [p[1] for p in valid]
        color = ASSET_COLORS.get(asset, "#ffffff")

        fig.add_trace(go.Scatter(
            x=t_vals,
            y=iv_vals,
            mode="lines+markers",
            name=asset,
            line=dict(color=color, width=2),
            marker=dict(size=8, color=color),
            hovertemplate=f"{asset}<br>Tenor: %{{x}}d<br>IV: %{{y:.2f}}%<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Constant-Maturity IV Comparison",
        xaxis_title="Tenor (days)",
        yaxis_title="Implied Volatility (%)",
        height=450,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    )

    return fig


def build_history_chart(history: list[dict], asset: str) -> go.Figure | None:
    """Build time series of CM IVs if historical snapshots are available."""
    if not history or len(history) < 2:
        return None

    fig = go.Figure()
    color = ASSET_COLORS.get(asset, "#ffffff")

    timestamps = [h["timestamp"] for h in history]

    # Plot each tenor as a line
    tenor_colors = {
        7: "#ff6b6b", 14: "#ffa502", 30: "#2ed573",
        60: "#1e90ff", 90: "#a55eea", 180: "#ff6348", 365: "#7bed9f",
    }

    for tenor, label in zip(STANDARD_TENORS, TENOR_LABELS):
        values = [h["tenors"].get(tenor) for h in history]
        # Only plot if we have some non-None values
        if all(v is None for v in values):
            continue

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode="lines",
            name=label,
            line=dict(color=tenor_colors.get(tenor, "#ffffff"), width=1.5),
            connectgaps=True,
            hovertemplate=f"{label}<br>%{{x}}<br>IV: %{{y:.2f}}%<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{asset} Constant-Maturity IV History",
        xaxis_title="Time",
        yaxis_title="Implied Volatility (%)",
        height=400,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    )

    return fig


# ---------------------------------------------------------------------------
# Main display logic
# ---------------------------------------------------------------------------

def main():
    st.title("📏 Constant Maturity Implied Volatility")
    st.caption("Live interpolated IV at fixed tenors from the Deribit option chain")

    # --- Sidebar controls ---
    with st.sidebar:
        st.subheader("Settings")

        selected_asset = st.selectbox("Asset", ASSETS, index=0)

        interp_method = st.radio(
            "Interpolation",
            ["cubic", "linear"],
            index=0,
            horizontal=True,
            help="Cubic spline gives smoother curves; linear is more conservative",
        )

        st.divider()

        compare_assets = st.multiselect(
            "Compare assets",
            [a for a in ASSETS if a != selected_asset],
            default=[],
            help="Add more assets for side-by-side CM IV comparison",
        )

        st.divider()

        auto_refresh = st.checkbox("Auto-refresh (2 min)", value=False)
        if auto_refresh:
            try:
                from streamlit_autorefresh import st_autorefresh
                st_autorefresh(interval=120_000, limit=None, key="cm_refresh")
            except ImportError:
                st.info("Install `streamlit-autorefresh` for auto-refresh.")

    # --- Fetch data for primary asset ---
    with st.spinner(f"Fetching {selected_asset} option chain..."):
        chain = fetch_chain_data(selected_asset)
        spot = fetch_spot_price(selected_asset)

    if chain is None or spot is None:
        st.error(f"Failed to fetch data for {selected_asset}. Deribit API may be unavailable.")
        return

    cfg = ASSET_CONFIG[selected_asset]
    price_dp = cfg["price_dp"]
    spot_str = f"${spot:,.{price_dp}f}" if price_dp == 0 else f"${spot:,.{price_dp}f}"

    st.markdown(f"**{selected_asset} Spot:** {spot_str} | "
                f"**Options loaded:** {len(chain):,} | "
                f"**Updated:** {datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S UTC')}")

    # --- Build term structure from actual expiries ---
    expiry_df = build_expiry_data(chain, spot, selected_asset)

    if expiry_df.empty:
        st.warning("No valid expiries found in the option chain. "
                   "This may happen during off-hours or if OI is very low.")
        return

    # --- Interpolate constant-maturity values ---
    cm_ivs = interpolate_constant_maturity(expiry_df, STANDARD_TENORS, method=interp_method)
    cm_skews = interpolate_constant_maturity_skew(expiry_df, STANDARD_TENORS, method=interp_method)

    # --- Store historical snapshot ---
    now = datetime.datetime.now(datetime.timezone.utc)
    history = st.session_state.cm_history[selected_asset]
    if not history or (now - history[-1]["timestamp"]).total_seconds() >= 120:
        history.append({"timestamp": now, "tenors": cm_ivs.copy()})
        if len(history) > MAX_HISTORY_POINTS:
            st.session_state.cm_history[selected_asset] = history[-MAX_HISTORY_POINTS:]

    skew_history = st.session_state.cm_skew_history[selected_asset]
    if not skew_history or (now - skew_history[-1]["timestamp"]).total_seconds() >= 120:
        skew_history.append({"timestamp": now, "tenors": cm_skews.copy()})
        if len(skew_history) > MAX_HISTORY_POINTS:
            st.session_state.cm_skew_history[selected_asset] = skew_history[-MAX_HISTORY_POINTS:]

    # --- Constant Maturity Table ---
    st.subheader("Constant-Maturity IV")

    col_ivs, col_skews = st.columns(2)

    with col_ivs:
        st.markdown("**ATM Implied Volatility**")
        iv_data = {
            "Tenor": TENOR_LABELS,
            "Days": STANDARD_TENORS,
            "CM IV (%)": [f"{cm_ivs[t]:.2f}" if cm_ivs[t] is not None else "—" for t in STANDARD_TENORS],
        }
        st.dataframe(
            pd.DataFrame(iv_data),
            use_container_width=True,
            hide_index=True,
        )

    with col_skews:
        st.markdown("**25-Delta Skew (Put - Call)**")
        skew_data = {
            "Tenor": TENOR_LABELS,
            "Days": STANDARD_TENORS,
            "Skew (%)": [f"{cm_skews[t]:+.2f}" if cm_skews[t] is not None else "—" for t in STANDARD_TENORS],
        }
        st.dataframe(
            pd.DataFrame(skew_data),
            use_container_width=True,
            hide_index=True,
        )

    # --- Term Structure Chart ---
    st.subheader("Term Structure")
    ts_chart = build_term_structure_chart(expiry_df, cm_ivs, selected_asset)
    st.plotly_chart(ts_chart, use_container_width=True)

    # --- Skew Chart ---
    st.subheader("25-Delta Skew by Tenor")
    skew_chart = build_skew_chart(expiry_df, cm_skews, selected_asset)
    st.plotly_chart(skew_chart, use_container_width=True)

    # --- Historical Chart ---
    history = st.session_state.cm_history[selected_asset]
    if len(history) >= 2:
        st.subheader("CM IV History (this session)")
        hist_chart = build_history_chart(history, selected_asset)
        if hist_chart:
            st.plotly_chart(hist_chart, use_container_width=True)
    else:
        st.info("Historical CM IV chart will appear after 2+ snapshots are collected "
                "(one every 2 minutes while the page is open).")

    # --- Multi-asset comparison ---
    if compare_assets:
        st.subheader("Multi-Asset Comparison")

        all_cm_ivs = {selected_asset: cm_ivs}
        all_cm_skews = {selected_asset: cm_skews}

        for asset in compare_assets:
            asset_chain = fetch_chain_data(asset)
            asset_spot = fetch_spot_price(asset)
            if asset_chain is None or asset_spot is None:
                st.warning(f"Could not fetch data for {asset}")
                continue
            asset_expiry_df = build_expiry_data(asset_chain, asset_spot, asset)
            if asset_expiry_df.empty:
                continue
            all_cm_ivs[asset] = interpolate_constant_maturity(
                asset_expiry_df, STANDARD_TENORS, method=interp_method
            )
            all_cm_skews[asset] = interpolate_constant_maturity_skew(
                asset_expiry_df, STANDARD_TENORS, method=interp_method
            )

        # Comparison chart
        comp_chart = build_multi_asset_chart(all_cm_ivs)
        st.plotly_chart(comp_chart, use_container_width=True)

        # Comparison table
        comp_data = {"Tenor": TENOR_LABELS}
        for asset, ivs in all_cm_ivs.items():
            comp_data[f"{asset} IV (%)"] = [
                f"{ivs[t]:.2f}" if ivs[t] is not None else "—" for t in STANDARD_TENORS
            ]
        for asset, skews in all_cm_skews.items():
            comp_data[f"{asset} Skew (%)"] = [
                f"{skews[t]:+.2f}" if skews[t] is not None else "—" for t in STANDARD_TENORS
            ]

        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    # --- Raw expiry data (collapsible) ---
    with st.expander("Raw Expiry Data"):
        display_df = expiry_df.copy()
        display_df["atm_iv"] = display_df["atm_iv"].map(lambda x: f"{x:.2f}%")
        display_df["put25_iv"] = display_df["put25_iv"].map(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "—"
        )
        display_df["call25_iv"] = display_df["call25_iv"].map(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "—"
        )
        display_df["skew_25d"] = display_df["skew_25d"].map(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
        )
        display_df["dte"] = display_df["dte"].map(lambda x: f"{x:.1f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
else:
    main()
