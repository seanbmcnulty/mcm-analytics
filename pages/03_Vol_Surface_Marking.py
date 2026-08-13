"""
Vol Surface Marking — SABR/polynomial vol surface calibration on live Deribit data.

Fetches the live option chain, fits SABR (or cubic spline fallback) to each tenor's
smile, and displays an interactive 3D vol surface, per-expiry smile fits, residuals,
and calibrated parameters.
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
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import streamlit as st

from lib.deribit import get_option_chain, get_index_price
from lib.constants import ASSET_CONFIG, ASSETS, ASSET_COLORS, PLOTLY_LAYOUT
from lib.instruments import parse_instrument, dte_from_expiry

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Vol Surface Marking",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# SABR model implementation
# ---------------------------------------------------------------------------


def sabr_vol(F: float, K: float, T: float, alpha: float, beta: float,
             rho: float, nu: float) -> float:
    """
    Hagan et al. (2002) SABR implied volatility approximation.
    F: forward price, K: strike, T: time to expiry (years),
    alpha: vol of vol base, beta: CEV exponent (fixed),
    rho: correlation, nu: vol of vol.
    """
    if T <= 0:
        return alpha
    if abs(F - K) < 1e-10:
        # ATM approximation
        FK_mid = F
        term1 = alpha / (FK_mid ** (1.0 - beta))
        correction = 1.0 + (
            ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2 / (FK_mid ** (2.0 - 2.0 * beta)))
            + 0.25 * rho * beta * nu * alpha / (FK_mid ** (1.0 - beta))
            + (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0
        ) * T
        return term1 * correction

    FK = F * K
    FK_beta = FK ** ((1.0 - beta) / 2.0)
    log_FK = np.log(F / K)

    # z and x(z)
    z = (nu / alpha) * FK_beta * log_FK
    # Avoid division by zero
    if abs(z) < 1e-10:
        x_z = 1.0
    else:
        sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z ** 2) + z - rho
        if abs(sqrt_term) < 1e-10 or abs(1.0 - rho) < 1e-10:
            x_z = 1.0
        else:
            x_z = z / np.log(sqrt_term / (1.0 - rho))

    # Numerator
    denom1 = FK_beta * (
        1.0
        + ((1.0 - beta) ** 2 / 24.0) * log_FK ** 2
        + ((1.0 - beta) ** 4 / 1920.0) * log_FK ** 4
    )

    # Correction term
    correction = 1.0 + (
        ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2 / (FK ** (1.0 - beta)))
        + 0.25 * rho * beta * nu * alpha / FK_beta
        + (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0
    ) * T

    vol = (alpha / denom1) * x_z * correction
    return max(vol, 1e-6)


def sabr_vol_vec(F: float, strikes: np.ndarray, T: float, alpha: float,
                 beta: float, rho: float, nu: float) -> np.ndarray:
    """Vectorized SABR vol computation over an array of strikes."""
    return np.array([sabr_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])


def calibrate_sabr(F: float, strikes: np.ndarray, market_vols: np.ndarray,
                   T: float, beta: float = 0.5) -> dict | None:
    """
    Calibrate SABR parameters (alpha, rho, nu) for a given expiry.
    beta is fixed (commonly 0.5 for rates, 1.0 for equities — we use 0.5 for crypto).
    Returns dict with alpha, beta, rho, nu, or None if calibration fails.
    """
    if len(strikes) < 3 or T <= 0:
        return None

    # Initial guess: alpha ~ ATM vol * F^(1-beta), rho ~ 0, nu ~ 1
    atm_idx = np.argmin(np.abs(strikes - F))
    atm_vol = market_vols[atm_idx]
    alpha_init = atm_vol * (F ** (1.0 - beta))
    alpha_init = max(alpha_init, 0.01)

    x0 = np.array([alpha_init, -0.1, 0.5])

    # Bounds: alpha > 0, -0.999 < rho < 0.999, nu > 0.01
    bounds = [(0.001, 10.0), (-0.999, 0.999), (0.01, 5.0)]

    def objective(params):
        alpha, rho, nu = params
        try:
            model_vols = sabr_vol_vec(F, strikes, T, alpha, beta, rho, nu)
            residuals = model_vols - market_vols
            return np.sum(residuals ** 2)
        except (ValueError, ZeroDivisionError, OverflowError):
            return 1e10

    try:
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 500, "ftol": 1e-12})
        if result.success or result.fun < 0.01:
            alpha, rho, nu = result.x
            return {"alpha": alpha, "beta": beta, "rho": rho, "nu": nu}
    except Exception:
        pass

    # Try Nelder-Mead as fallback (no bounds, but clip)
    try:
        result = minimize(objective, x0, method="Nelder-Mead",
                          options={"maxiter": 1000, "xatol": 1e-8})
        if result.fun < 0.05:
            alpha, rho, nu = result.x
            alpha = max(alpha, 0.001)
            rho = np.clip(rho, -0.999, 0.999)
            nu = max(nu, 0.01)
            return {"alpha": alpha, "beta": beta, "rho": rho, "nu": nu}
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120, show_spinner="Fetching option chain...")
def fetch_vol_surface_data(currency: str) -> pd.DataFrame | None:
    """
    Fetch option chain from Deribit and build a DataFrame with columns:
    strike, expiry_str, expiry_date, dte, mark_iv, bid_iv, ask_iv,
    open_interest, option_type, instrument_name, underlying_price
    """
    cfg = ASSET_CONFIG.get(currency)
    if not cfg:
        return None

    chain = get_option_chain(cfg["deribit_ccy"], "option")
    if not chain:
        return None

    prefix = cfg["deribit_prefix"]
    now_ms = int(time.time() * 1000)

    rows = []
    for inst in chain:
        name = inst.get("instrument_name", "")
        if not name.startswith(prefix):
            continue

        mark_iv = inst.get("mark_iv", 0)
        open_interest = inst.get("open_interest", 0)

        # Filter: skip 0 OI or 0 mark_iv
        if not mark_iv or mark_iv == 0 or not open_interest or open_interest == 0:
            continue

        parsed = parse_instrument(name)
        if not parsed:
            continue

        expiry_ts = inst.get("expiration_timestamp", 0)
        dte = max((expiry_ts - now_ms) / (1000 * 86400), 0.001)

        # Skip expired or very short-dated (< 0.1 day ~ 2.4h)
        if dte < 0.1:
            continue

        bid_iv = inst.get("bid_iv", 0) or 0
        ask_iv = inst.get("ask_iv", 0) or 0

        rows.append({
            "instrument_name": name,
            "strike": parsed.strike,
            "expiry_str": parsed.expiry_str,
            "expiry_date": parsed.expiry_date,
            "expiry_ts": expiry_ts,
            "dte": dte,
            "tte": dte / 365.0,  # time-to-expiry in years
            "mark_iv": mark_iv / 100.0 if mark_iv > 5 else mark_iv,  # normalize
            "bid_iv": bid_iv / 100.0 if bid_iv > 5 else bid_iv,
            "ask_iv": ask_iv / 100.0 if ask_iv > 5 else ask_iv,
            "open_interest": open_interest,
            "option_type": parsed.kind,
            "underlying_price": inst.get("underlying_price", 0),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values(["dte", "strike"]).reset_index(drop=True)
    return df


@st.cache_data(ttl=30, show_spinner=False)
def fetch_spot_price(currency: str) -> float | None:
    """Fetch current spot/index price."""
    cfg = ASSET_CONFIG.get(currency)
    if not cfg:
        return None
    return get_index_price(cfg["index"])


# ---------------------------------------------------------------------------
# Surface fitting
# ---------------------------------------------------------------------------

def fit_surface(df: pd.DataFrame, spot: float, beta: float = 0.5,
                use_sabr: bool = True) -> dict:
    """
    Fit vol smiles for each tenor. Returns dict with:
    - fits: list of dicts per tenor (expiry_str, dte, params, fitted_vols, market_vols, strikes, method)
    - surface_data: DataFrame with (strike, dte, fitted_iv) for 3D plotting
    """
    results = []
    surface_rows = []

    for expiry_str, group in df.groupby("expiry_str"):
        group = group.sort_values("strike").reset_index(drop=True)
        strikes = group["strike"].values
        market_vols = group["mark_iv"].values
        dte = group["dte"].iloc[0]
        tte = group["tte"].iloc[0]

        # Use underlying_price from chain as forward proxy (close enough for crypto)
        F = group["underlying_price"].iloc[0]
        if F <= 0:
            F = spot

        fit_result = {
            "expiry_str": expiry_str,
            "dte": dte,
            "tte": tte,
            "strikes": strikes,
            "market_vols": market_vols,
            "forward": F,
            "bid_ivs": group["bid_iv"].values,
            "ask_ivs": group["ask_iv"].values,
        }

        sabr_params = None
        if use_sabr and len(strikes) >= 4:
            sabr_params = calibrate_sabr(F, strikes, market_vols, tte, beta=beta)

        if sabr_params is not None:
            fitted = sabr_vol_vec(F, strikes, tte, **sabr_params)
            fit_result["method"] = "SABR"
            fit_result["params"] = sabr_params
            fit_result["fitted_vols"] = fitted
            fit_result["residuals"] = market_vols - fitted

            # Generate smooth curve for surface
            k_min, k_max = strikes.min(), strikes.max()
            k_smooth = np.linspace(k_min, k_max, 50)
            iv_smooth = sabr_vol_vec(F, k_smooth, tte, **sabr_params)
            for k, iv in zip(k_smooth, iv_smooth):
                surface_rows.append({"strike": k, "dte": dte, "fitted_iv": iv})
        else:
            # Fallback: cubic spline interpolation
            if len(strikes) >= 4:
                try:
                    cs = CubicSpline(strikes, market_vols, bc_type="natural")
                    fitted = cs(strikes)
                    fit_result["method"] = "Cubic Spline"
                    fit_result["params"] = {"type": "cubic_spline", "knots": len(strikes)}
                    fit_result["fitted_vols"] = fitted
                    fit_result["residuals"] = market_vols - fitted

                    k_min, k_max = strikes.min(), strikes.max()
                    k_smooth = np.linspace(k_min, k_max, 50)
                    iv_smooth = cs(k_smooth)
                    for k, iv in zip(k_smooth, iv_smooth):
                        surface_rows.append({"strike": k, "dte": dte, "fitted_iv": iv})
                except Exception:
                    fit_result["method"] = "Raw"
                    fit_result["params"] = None
                    fit_result["fitted_vols"] = market_vols
                    fit_result["residuals"] = np.zeros_like(market_vols)
                    for k, iv in zip(strikes, market_vols):
                        surface_rows.append({"strike": k, "dte": dte, "fitted_iv": iv})
            else:
                # Too few points — just plot raw
                fit_result["method"] = "Raw (< 4 strikes)"
                fit_result["params"] = None
                fit_result["fitted_vols"] = market_vols
                fit_result["residuals"] = np.zeros_like(market_vols)
                for k, iv in zip(strikes, market_vols):
                    surface_rows.append({"strike": k, "dte": dte, "fitted_iv": iv})

        results.append(fit_result)

    surface_df = pd.DataFrame(surface_rows) if surface_rows else pd.DataFrame()
    return {"fits": results, "surface_data": surface_df}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def build_3d_surface(surface_df: pd.DataFrame, spot: float,
                     asset_color: str) -> go.Figure:
    """Build interactive 3D vol surface plot."""
    if surface_df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="No surface data")
        return fig

    # Pivot for surface plot
    pivot = surface_df.pivot_table(index="dte", columns="strike", values="fitted_iv")
    pivot = pivot.sort_index()

    strikes = pivot.columns.values
    dtes = pivot.index.values
    z_vals = pivot.values * 100  # Convert to percentage

    fig = go.Figure(data=[go.Surface(
        x=strikes,
        y=dtes,
        z=z_vals,
        colorscale="Viridis",
        colorbar=dict(title="IV (%)", len=0.6),
        opacity=0.9,
        hovertemplate=(
            "Strike: %{x:,.0f}<br>"
            "DTE: %{y:.1f}<br>"
            "IV: %{z:.1f}%<extra></extra>"
        ),
    )])

    # Add spot price vertical line on the surface
    fig.add_trace(go.Scatter3d(
        x=[spot] * len(dtes),
        y=dtes,
        z=[z_vals.min()] * len(dtes),
        mode="lines",
        line=dict(color="white", width=3, dash="dash"),
        name=f"Spot ({spot:,.0f})",
        showlegend=True,
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="DTE (days)",
            zaxis_title="IV (%)",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
        ),
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def build_heatmap_surface(surface_df: pd.DataFrame, spot: float) -> go.Figure:
    """Build 2D heatmap of vol surface (alternative to 3D)."""
    if surface_df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="No surface data")
        return fig

    pivot = surface_df.pivot_table(index="dte", columns="strike", values="fitted_iv")
    pivot = pivot.sort_index(ascending=False)

    z_vals = pivot.values * 100
    strikes = pivot.columns.values
    dtes = pivot.index.values

    fig = go.Figure(data=go.Heatmap(
        x=strikes,
        y=dtes,
        z=z_vals,
        colorscale="Viridis",
        colorbar=dict(title="IV (%)"),
        hovertemplate=(
            "Strike: %{x:,.0f}<br>"
            "DTE: %{y:.1f}<br>"
            "IV: %{z:.1f}%<extra></extra>"
        ),
    ))

    # Spot line
    fig.add_vline(x=spot, line_dash="dash", line_color="white",
                  annotation_text=f"Spot {spot:,.0f}")

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="IV Surface Heatmap",
        xaxis_title="Strike",
        yaxis_title="DTE (days)",
        height=500,
    )

    return fig


def build_smile_chart(fit: dict, asset_color: str) -> go.Figure:
    """Build individual smile plot for one tenor."""
    strikes = fit["strikes"]
    market_vols = fit["market_vols"] * 100
    fitted_vols = fit["fitted_vols"] * 100
    bid_ivs = fit.get("bid_ivs", np.zeros_like(strikes)) * 100
    ask_ivs = fit.get("ask_ivs", np.zeros_like(strikes)) * 100
    F = fit["forward"]

    fig = go.Figure()

    # Bid/Ask IV spread (shaded region)
    valid_ba = (bid_ivs > 0) & (ask_ivs > 0)
    if valid_ba.any():
        fig.add_trace(go.Scatter(
            x=np.concatenate([strikes[valid_ba], strikes[valid_ba][::-1]]),
            y=np.concatenate([ask_ivs[valid_ba], bid_ivs[valid_ba][::-1]]),
            fill="toself",
            fillcolor="rgba(100, 100, 100, 0.2)",
            line=dict(width=0),
            name="Bid/Ask Spread",
            hoverinfo="skip",
        ))

    # Market data points
    fig.add_trace(go.Scatter(
        x=strikes,
        y=market_vols,
        mode="markers",
        marker=dict(color=asset_color, size=8, symbol="circle"),
        name="Mark IV",
        hovertemplate="K=%{x:,.0f}, IV=%{y:.1f}%<extra></extra>",
    ))

    # Fitted curve
    if fit["method"] != "Raw" and fit["method"] != "Raw (< 4 strikes)":
        # Generate smooth fitted curve
        k_min, k_max = strikes.min(), strikes.max()
        k_smooth = np.linspace(k_min, k_max, 100)

        if fit["method"] == "SABR" and fit["params"]:
            params = fit["params"]
            iv_smooth = sabr_vol_vec(F, k_smooth, fit["tte"], **params) * 100
        else:
            try:
                cs = CubicSpline(strikes, fit["fitted_vols"])
                iv_smooth = cs(k_smooth) * 100
            except Exception:
                iv_smooth = np.interp(k_smooth, strikes, fitted_vols)

        fig.add_trace(go.Scatter(
            x=k_smooth,
            y=iv_smooth,
            mode="lines",
            line=dict(color="#ff6b6b", width=2.5),
            name=f"Fit ({fit['method']})",
        ))

    # Forward line
    fig.add_vline(x=F, line_dash="dot", line_color="rgba(255,255,255,0.5)",
                  annotation_text=f"F={F:,.0f}")

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{fit['expiry_str']} — {fit['dte']:.1f} DTE ({fit['method']})",
        xaxis_title="Strike",
        yaxis_title="IV (%)",
        height=350,
        showlegend=True,
        legend=dict(orientation="h", y=-0.2),
    )

    return fig


def build_residuals_chart(fits: list[dict]) -> go.Figure:
    """Build residuals chart across all tenors."""
    fig = go.Figure()

    colors = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab",
    ]

    for i, fit in enumerate(fits):
        if fit["residuals"] is None or len(fit["residuals"]) == 0:
            continue
        residuals_pct = fit["residuals"] * 100
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=fit["strikes"],
            y=residuals_pct,
            mode="markers+lines",
            marker=dict(size=5, color=color),
            line=dict(width=1, color=color),
            name=f"{fit['expiry_str']} ({fit['dte']:.0f}d)",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Fit Residuals (Market IV - Fitted IV)",
        xaxis_title="Strike",
        yaxis_title="Residual (% points)",
        height=400,
        legend=dict(orientation="h", y=-0.25),
    )

    return fig


def build_parameters_table(fits: list[dict]) -> pd.DataFrame:
    """Build summary table of calibrated parameters per tenor."""
    rows = []
    for fit in fits:
        row = {
            "Expiry": fit["expiry_str"],
            "DTE": f"{fit['dte']:.1f}",
            "Method": fit["method"],
            "# Strikes": len(fit["strikes"]),
            "RMSE (%)": f"{np.sqrt(np.mean(fit['residuals']**2)) * 100:.3f}"
            if fit["residuals"] is not None else "N/A",
        }
        if fit["method"] == "SABR" and fit["params"]:
            row["Alpha"] = f"{fit['params']['alpha']:.4f}"
            row["Beta"] = f"{fit['params']['beta']:.2f}"
            row["Rho"] = f"{fit['params']['rho']:.4f}"
            row["Nu (VolVol)"] = f"{fit['params']['nu']:.4f}"
        elif fit["method"] == "Cubic Spline" and fit["params"]:
            row["Alpha"] = "—"
            row["Beta"] = "—"
            row["Rho"] = "—"
            row["Nu (VolVol)"] = "—"
        else:
            row["Alpha"] = "—"
            row["Beta"] = "—"
            row["Rho"] = "—"
            row["Nu (VolVol)"] = "—"
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Term structure chart
# ---------------------------------------------------------------------------

def build_term_structure(fits: list[dict], asset_color: str) -> go.Figure:
    """ATM vol term structure from fitted smiles."""
    fig = go.Figure()

    dtes = []
    atm_vols = []

    for fit in fits:
        F = fit["forward"]
        strikes = fit["strikes"]
        market_vols = fit["market_vols"]

        # Find ATM vol (closest strike to forward)
        atm_idx = np.argmin(np.abs(strikes - F))
        atm_vols.append(market_vols[atm_idx] * 100)
        dtes.append(fit["dte"])

    if not dtes:
        fig.update_layout(**PLOTLY_LAYOUT, title="No term structure data")
        return fig

    # Sort by DTE
    order = np.argsort(dtes)
    dtes = np.array(dtes)[order]
    atm_vols = np.array(atm_vols)[order]

    fig.add_trace(go.Scatter(
        x=dtes,
        y=atm_vols,
        mode="lines+markers",
        line=dict(color=asset_color, width=2.5),
        marker=dict(size=8),
        name="ATM IV",
        hovertemplate="DTE: %{x:.1f}<br>ATM IV: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="ATM Volatility Term Structure",
        xaxis_title="DTE (days)",
        yaxis_title="ATM IV (%)",
        height=350,
    )

    return fig


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title("Vol Surface Marking")
st.caption("SABR-calibrated implied volatility surface from live Deribit option chain.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")

    currency = st.selectbox("Asset", ASSETS, index=0)
    asset_color = ASSET_COLORS.get(currency, "#4e79a7")

    st.divider()
    st.subheader("SABR Parameters")
    beta_fixed = st.slider("Beta (fixed CEV exponent)", 0.0, 1.0, 0.5, 0.1,
                           help="0 = normal, 0.5 = CIR, 1.0 = lognormal. "
                                "0.5 is standard for crypto vol.")
    use_sabr = st.checkbox("Use SABR (uncheck for spline-only)", value=True)

    st.divider()
    st.subheader("Filters")
    min_dte = st.number_input("Min DTE", min_value=0.1, max_value=365.0,
                              value=0.5, step=0.5)
    max_dte = st.number_input("Max DTE", min_value=1.0, max_value=730.0,
                              value=365.0, step=10.0)
    min_oi = st.number_input("Min Open Interest", min_value=0, max_value=10000,
                             value=1, step=10)
    min_strikes_per_expiry = st.number_input("Min Strikes per Expiry", min_value=3,
                                             max_value=50, value=5, step=1)

    st.divider()
    surface_view = st.radio("Surface View", ["3D Surface", "Heatmap"], index=0)

# Fetch data
spot = fetch_spot_price(currency)
df = fetch_vol_surface_data(currency)

if df is None or df.empty:
    st.error("No option chain data available. Check that Deribit is reachable and "
             f"{currency} options are listed.")
    st.stop()

if spot is None or spot <= 0:
    # Fallback: use underlying_price from chain
    spot = df["underlying_price"].median()

# Apply filters
df = df[(df["dte"] >= min_dte) & (df["dte"] <= max_dte)]
df = df[df["open_interest"] >= min_oi]

# Filter expiries with too few strikes
expiry_counts = df.groupby("expiry_str").size()
valid_expiries = expiry_counts[expiry_counts >= min_strikes_per_expiry].index
df = df[df["expiry_str"].isin(valid_expiries)]

if df.empty:
    st.warning("No data remaining after filters. Try relaxing the filter criteria.")
    st.stop()

# Display spot and chain info
col_info1, col_info2, col_info3, col_info4 = st.columns(4)
with col_info1:
    st.metric("Spot Price", f"${spot:,.0f}")
with col_info2:
    st.metric("Expiries", f"{df['expiry_str'].nunique()}")
with col_info3:
    st.metric("Total Strikes", f"{len(df):,}")
with col_info4:
    st.metric("DTE Range", f"{df['dte'].min():.1f} — {df['dte'].max():.0f}d")

st.divider()

# Fit the surface
with st.spinner("Calibrating vol surface..."):
    fit_results = fit_surface(df, spot, beta=beta_fixed, use_sabr=use_sabr)

fits = fit_results["fits"]
surface_df = fit_results["surface_data"]

if not fits:
    st.error("Surface calibration failed. No valid tenor fits.")
    st.stop()

# Count methods used
sabr_count = sum(1 for f in fits if f["method"] == "SABR")
spline_count = sum(1 for f in fits if f["method"] == "Cubic Spline")
raw_count = sum(1 for f in fits if "Raw" in f["method"])

method_summary = []
if sabr_count:
    method_summary.append(f"SABR: {sabr_count}")
if spline_count:
    method_summary.append(f"Spline: {spline_count}")
if raw_count:
    method_summary.append(f"Raw: {raw_count}")
st.caption(f"Calibration: {' | '.join(method_summary)} tenors fitted")

# ---------------------------------------------------------------------------
# Vol Surface (3D or Heatmap)
# ---------------------------------------------------------------------------

st.subheader("Volatility Surface")

if surface_view == "3D Surface":
    fig_surface = build_3d_surface(surface_df, spot, asset_color)
else:
    fig_surface = build_heatmap_surface(surface_df, spot)

st.plotly_chart(fig_surface, use_container_width=True)

# ---------------------------------------------------------------------------
# Term Structure
# ---------------------------------------------------------------------------

st.subheader("ATM Term Structure")
fig_term = build_term_structure(fits, asset_color)
st.plotly_chart(fig_term, use_container_width=True)

# ---------------------------------------------------------------------------
# Individual Smile Plots
# ---------------------------------------------------------------------------

st.subheader("Smile Fits by Expiry")

# Let user select which expiries to view
expiry_options = [f["expiry_str"] for f in fits]
selected_expiries = st.multiselect(
    "Select expiries to display",
    options=expiry_options,
    default=expiry_options[:6],  # Show first 6 by default
    help="Select which tenor smiles to plot individually",
)

# Display in a grid (2 columns)
selected_fits = [f for f in fits if f["expiry_str"] in selected_expiries]
if selected_fits:
    cols = st.columns(2)
    for i, fit in enumerate(selected_fits):
        with cols[i % 2]:
            fig_smile = build_smile_chart(fit, asset_color)
            st.plotly_chart(fig_smile, use_container_width=True)
else:
    st.info("Select expiries above to view individual smile fits.")

# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

st.subheader("Fit Residuals")
fig_residuals = build_residuals_chart(fits)
st.plotly_chart(fig_residuals, use_container_width=True)

# RMSE summary
total_residuals = np.concatenate([f["residuals"] for f in fits if f["residuals"] is not None])
if len(total_residuals) > 0:
    overall_rmse = np.sqrt(np.mean(total_residuals ** 2)) * 100
    max_abs_err = np.max(np.abs(total_residuals)) * 100
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.metric("Overall RMSE", f"{overall_rmse:.3f}%")
    with col_r2:
        st.metric("Max Absolute Error", f"{max_abs_err:.3f}%")

# ---------------------------------------------------------------------------
# Parameters Table
# ---------------------------------------------------------------------------

st.subheader("Calibrated Parameters")
params_df = build_parameters_table(fits)
st.dataframe(params_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    f"Data: Deribit public API | Asset: {currency} | "
    f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} | "
    f"Beta fixed at {beta_fixed:.1f}"
)
