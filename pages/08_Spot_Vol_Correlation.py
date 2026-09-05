"""
Spot Vol Correlation — how spot price relates to options volatility (CVOL),
25-delta skew, 10-delta risk reversal, and 10-delta butterfly, for BTC and
ETH.

Full port of exodus-analytics's Spot_Vol_Correlation.py (analytics_frontend/
streamlit/pages/Spot_Vol_Correlation.py, ~2840 lines) onto this app's
Deribit-only data layer. exodus fetched a 30d constant-maturity vol surface
from Amberdata; this app already has purpose-built Deribit-only history
reconstruction for exactly that (lib/history.py, lib/surface.py — see their
module docstrings), so the whole Amberdata layer is replaced with:

- **CVOL** (30d ATM IV) = ``history.iv_series_at_dte(asset, 30, "delta50")``
- **SVOL** (25Δ skew)   = 25Δ call IV − 25Δ put IV, both at 30 DTE
- **RR**   (10Δ risk reversal) = 10Δ call IV − 10Δ put IV, both at 30 DTE
- **BF**   (10Δ butterfly)     = (10Δ call + 10Δ put)/2 − ATM, all at 30 DTE
- **DVol Snapshot** panel = the real Deribit DVOL index (``lib.deribit.get_dvol``)

Scoped to **BTC and ETH only** — both have a Deribit DVOL index; SOL/HYPE do
not (``ASSET_CONFIG[...]["has_dvol"] is False``), which was exodus's own
reason for treating them differently throughout this page, so dropping them
removes an entire fallback code path (the "no DVOL, show 30d ATM instead"
branch) rather than adapting it. Matches the scope decision made with the
user for pages/07 (Regime Identifier) — see CLAUDE.md's session log.

Dropped as dead code (never called from exodus's own UI or Telegram report —
verified by reading the full source, per the page-06/pages-07 precedent):
``_chart_vol_prediction_winsorized``, ``_chart_vol_prediction_lagged_vol``,
``_chart_vol_prediction_ensemble``, ``_chart_vol_prediction_oos``. Only
``_chart_vol_prediction_vs_rr`` (linear/quadratic/cubic/exponential, 4
windows) is wired into the dashboard and the Telegram report, and it is
fully ported below.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import norm

from lib import deribit, history, surface
from lib.constants import ASSET_CONFIG, ASSET_COLORS, PLOTLY_LAYOUT
from lib import fx_style
from lib import cache as cache_lib
from lib.telegram import send_message, send_photo, is_configured

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Spot Vol Correlation", page_icon="📈", layout="wide")

SV_ASSETS = [a for a in ("BTC", "ETH") if ASSET_CONFIG.get(a, {}).get("has_dvol")]
CHART_HEIGHT = 480
VOL_PREDICTION_LOOKBACKS = (7, 14, 30, 90)  # 1 week, 2 weeks, 1 month, 3 months
MIN_POINTS_PER_DEGREE = 4
MIN_CORR_OBS = 20
MIN_ROLLING_POINTS = 20
ROLLING_WINDOW = 30
RESOLUTION_MAP = {"1 hr": "60", "4 hr": "240", "12 hr": "720", "1D": "1D"}
DATE_RANGE_PRESETS = {"Last 2 weeks": 14, "Last 1 month": 30, "Last 3 months": 90, "Last 6 months": 180, "Last 1 year": 365}


def _show(fig: go.Figure | None, key: str) -> None:
    if fig is None:
        return
    fx_style.add_watermark(fig)
    st.plotly_chart(fx_style.apply_theme(fig), width="stretch", key=key)


def _insufficient_data_fig(title: str, message: str, height: int = CHART_HEIGHT) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, align="center", font=dict(size=13))
    fig.update_layout(**PLOTLY_LAYOUT, height=height, title=title, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _safe_polyfit(x, y, degree, weights=None):
    """Weighted polyfit that returns None instead of an over-fitted or failed model
    (a degree-3 curve through 5 points is unconstrained once extrapolated)."""
    n = len(x)
    if n < degree + MIN_POINTS_PER_DEGREE or len(np.unique(x)) <= degree:
        return None
    try:
        with np.errstate(all="ignore"):
            return np.polyfit(x, y, degree, w=weights)
    except (np.linalg.LinAlgError, ValueError, TypeError):
        return None


def _prediction_windows_for_lookback(lookback_days: int) -> tuple:
    days = max(1, int(lookback_days))
    if days > VOL_PREDICTION_LOOKBACKS[-1]:
        return (7, 14, 30, days)
    return VOL_PREDICTION_LOOKBACKS


def _prediction_window_label(days: int) -> str:
    return {7: "1 week", 14: "2 weeks", 30: "1 month", 90: "3 months", 180: "6 months", 365: "1 year"}.get(days, f"{days} days")


# ============================================================================
# DATA — Deribit-only, via lib.history / lib.surface / lib.deribit
# ============================================================================

@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def _spot_series(asset: str, days: int) -> pd.Series:
    df = history.perp_ohlc(asset, days=days, resolution="60")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return _normalize_dt_index(df["close"].dropna())


@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def _cvol_series(asset: str, days: int):
    """30d ATM IV (%), (estimated, source)."""
    s, est, src = history.iv_series_at_dte(asset, 30, "delta50", days=days)
    s = _normalize_dt_index(s) if s is not None else pd.Series(dtype=float)
    return s, est, src


@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def _leg_series(asset: str, days: int, delta_key: str):
    s, est, src = history.iv_series_at_dte(asset, 30, delta_key, days=days)
    s = _normalize_dt_index(s) if s is not None else pd.Series(dtype=float)
    return s, est, src


def _svol_series(asset: str, days: int):
    """25Δ skew (%) = 25Δ call IV - 25Δ put IV, at 30 DTE."""
    call, est_c, src_c = _leg_series(asset, days, "deltaCall25")
    put, est_p, src_p = _leg_series(asset, days, "deltaPut25")
    if call.empty or put.empty:
        return pd.Series(dtype=float), True, "none"
    idx = call.index.union(put.index)
    diff = (call.reindex(idx).interpolate(limit_direction="both") - put.reindex(idx).interpolate(limit_direction="both")).dropna()
    return diff, (est_c or est_p), (src_c if est_c else src_p)


def _rr_series(asset: str, days: int):
    """10Δ risk reversal (%) = 10Δ call IV - 10Δ put IV, at 30 DTE."""
    call, est_c, src_c = _leg_series(asset, days, "deltaCall10")
    put, est_p, src_p = _leg_series(asset, days, "deltaPut10")
    if call.empty or put.empty:
        return pd.Series(dtype=float), True, "none"
    idx = call.index.union(put.index)
    diff = (call.reindex(idx).interpolate(limit_direction="both") - put.reindex(idx).interpolate(limit_direction="both")).dropna()
    return diff, (est_c or est_p), (src_c if est_c else src_p)


def _bf_series(asset: str, days: int):
    """10Δ butterfly (%) = (10Δ call + 10Δ put)/2 - ATM, at 30 DTE."""
    call, est_c, _ = _leg_series(asset, days, "deltaCall10")
    put, est_p, _ = _leg_series(asset, days, "deltaPut10")
    atm, est_a, src_a = _cvol_series(asset, days)
    if call.empty or put.empty or atm.empty:
        return pd.Series(dtype=float), True, "none"
    idx = call.index.union(put.index).union(atm.index)
    c = call.reindex(idx).interpolate(limit_direction="both")
    p = put.reindex(idx).interpolate(limit_direction="both")
    a = atm.reindex(idx).interpolate(limit_direction="both")
    bf = ((c + p) / 2 - a).dropna()
    return bf, (est_c or est_p or est_a), src_a


def _normalize_dt_index(s: pd.Series) -> pd.Series:
    """Coerce a tz-aware DatetimeIndex to a fixed (microsecond) resolution.

    pandas 2.x/3.x's ``merge_asof``/``Index.union`` require the two sides'
    datetime64 dtype to match *exactly*, not just compare equal — and this
    app's various timestamp constructions don't all land on the same
    resolution: Deribit epoch-ms timestamps (``lib/history.py:perp_ohlc``,
    ``dvol_history``) parse to ``datetime64[ms, UTC]``, while recorded
    snapshot timestamps parsed from ISO8601 strings with microsecond
    precision (``lib/history.py:_parse_snapshot_csv``) parse to
    ``datetime64[us, UTC]``. Mixing the two raises "incompatible merge keys
    ... must be the same type" — reproduced live on this page (CVOL, being
    snapshot-backed, is often `us`; spot, being Deribit-OHLC-backed, is
    always `ms`). Normalizing both sides here, right before every merge,
    fixes it regardless of which side drifts in the future."""
    if s.empty or not isinstance(s.index, pd.DatetimeIndex) or s.index.tz is None:
        return s
    s = s.copy()
    s.index = s.index.as_unit("us")
    return s


def _align_to_spot(vol_series: pd.Series, spot: pd.Series) -> pd.DataFrame:
    """Nearest-timestamp join of a (possibly sparser) vol series onto the spot
    index, tolerant of the two feeds' different native resolutions (spot is
    always hourly; vol series is hourly for the last 14 days and daily beyond
    — see lib/history.py's snapshot-thinning policy). Adds days_ago for the
    scatter colorscale."""
    if vol_series.empty or spot.empty:
        return pd.DataFrame(columns=["spot", "vol", "days_ago"])
    v = _normalize_dt_index(vol_series.sort_index())
    s = _normalize_dt_index(spot.sort_index())
    merged = pd.merge_asof(
        pd.DataFrame({"vol": v}), pd.DataFrame({"spot": s}),
        left_index=True, right_index=True, direction="nearest", tolerance=pd.Timedelta("12h"),
    ).dropna()
    if merged.empty:
        return merged
    merged["days_ago"] = (merged.index.max() - merged.index).total_seconds() / 86400.0
    return merged


@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def _current_30d_surface(asset: str):
    """Live 30d surface (ATM, 25Δ/10Δ call+put, decimal IV) from the option
    chain — replaces exodus's Amberdata constant-maturity surface fetch."""
    try:
        vols = surface.option_vols_by_dte(asset)
    except Exception:
        return None
    out = {}
    for key, bucket in (("atm", "atm"), ("call25", "call25"), ("put25", "put25"), ("call10", "call10"), ("put10", "put10")):
        by_dte = (vols or {}).get(bucket) or {}
        if not by_dte:
            continue
        try:
            iv = surface.interp_at_dte(by_dte, 30)
        except Exception:
            iv = None
        if iv is not None and np.isfinite(iv) and iv > 0:
            out[key] = float(iv)
    return out if len(out) >= 2 else None


def _strike_from_delta(spot: float, vol_decimal: float, delta: float, is_call: bool, t_years: float) -> float | None:
    """Black-Scholes strike from delta (r=0)."""
    if t_years <= 0 or vol_decimal <= 0 or spot <= 0:
        return None
    try:
        d1 = norm.ppf(max(1e-6, min(1 - 1e-6, delta))) if is_call else norm.ppf(max(1e-6, min(1 - 1e-6, 1 + delta)))
        log_k = d1 * vol_decimal * (t_years ** 0.5) - 0.5 * (vol_decimal ** 2) * t_years
        return float(spot * np.exp(-log_k))
    except (ValueError, TypeError):
        return None


def _surface_30d_to_strike_vol_points(surface_30d: dict, current_spot: float) -> list[tuple[float, float, str]]:
    """30d surface (decimal IV) -> [(strike, vol_pct, label), ...] ordered by strike, for the U-shape overlay."""
    T = 30 / 365.0
    points = []
    for key, label, delta in (("put10", "10Δ Put", -0.10), ("put25", "25Δ Put", -0.25), ("atm", "ATM", None), ("call25", "25Δ Call", 0.25), ("call10", "10Δ Call", 0.10)):
        if key not in surface_30d:
            continue
        vol_dec = surface_30d[key]
        if key == "atm":
            strike = current_spot
        else:
            strike = _strike_from_delta(current_spot, vol_dec, delta, delta > 0, T) or current_spot
        points.append((strike, vol_dec * 100.0, label))
    points.sort(key=lambda x: x[0])
    return points


# ============================================================================
# CHART BUILDERS
# ============================================================================

def _scatter_vs_spot(merged: pd.DataFrame, asset: str, title: str, y_label: str, estimated: bool, source: str) -> go.Figure:
    """Generic (spot, vol-metric) scatter: colored by days-ago recency, gray
    crosshair at the latest point, orange quadratic fit. Used for CVOL, SVOL,
    RR and BF — exodus had four near-identical copies of this
    (_scatter_cvol_spot / _scatter_svol_spot / _scatter_rr_spot /
    _scatter_bf_spot); this port keeps the one shape, parametrized."""
    if merged.empty or len(merged) < 5:
        return _insufficient_data_fig(title, f"Not enough {asset} data in this window.")
    x = merged["spot"].to_numpy(dtype=float)
    y = merged["vol"].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=7, color=merged["days_ago"], colorscale="Portland", cmin=0, cmax=max(1.0, float(merged["days_ago"].max())), colorbar=dict(title="Days Ago")),
        hovertemplate="Spot: %{x:,.0f}<br>" + y_label + ": %{y:.2f}%<extra></extra>",
        name="(spot, value)",
    ))
    coefs = _safe_polyfit(x, y, 2)
    if coefs is not None:
        x_line = np.linspace(x.min(), x.max(), 100)
        fig.add_trace(go.Scatter(x=x_line, y=np.polyval(coefs, x_line), mode="lines", line=dict(color="orange", width=2, dash="dot"), name="Quadratic fit"))
    latest = merged.iloc[merged.index.argmax()]
    fig.add_shape(type="line", x0=latest["spot"], x1=latest["spot"], y0=0, y1=1, yref="paper", line=dict(color="gray", dash="dot", width=1))
    fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=latest["vol"], y1=latest["vol"], line=dict(color="gray", dash="dot", width=1))
    note = f" — reconstructed from {source}" if estimated and source != "none" else ""
    fig.update_layout(**PLOTLY_LAYOUT, title=f"{title}{note}", xaxis_title="Spot", yaxis_title=y_label, height=CHART_HEIGHT, hovermode="closest")
    return fig


def _candlestick_dvol(asset: str, start_ms: int, end_ms: int, resolution: str, skew_series: pd.Series | None) -> go.Figure:
    df = deribit.get_dvol(ASSET_CONFIG[asset]["deribit_ccy"], resolution=resolution, start_ms=start_ms, end_ms=end_ms)
    if df is None or df.empty:
        return _insufficient_data_fig(f"DVol Snapshot — {asset}", "No DVOL candles returned for this window.")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="DVOL",
                                  increasing_line_color="#4CAF50", decreasing_line_color="#F44336"), secondary_y=False)
    if skew_series is not None and not skew_series.empty:
        skew_series = _normalize_dt_index(skew_series)
        target_index = df.index.as_unit("us") if isinstance(df.index, pd.DatetimeIndex) else df.index
        aligned = skew_series.reindex(target_index, method="nearest", tolerance=pd.Timedelta("6h")).dropna()
        if not aligned.empty:
            fig.add_trace(go.Scatter(x=aligned.index, y=aligned.values, mode="lines", name="25Δ Skew", line=dict(color="#9c27b0", width=1.5, dash="dot")), secondary_y=True)
    fig.update_yaxes(title_text="DVOL", secondary_y=False)
    fig.update_yaxes(title_text="25Δ Skew (%)", secondary_y=True, showgrid=False)
    fig.update_layout(**PLOTLY_LAYOUT, title=f"DVol Snapshot — {asset}", height=CHART_HEIGHT, xaxis_rangeslider_visible=False, hovermode="x unified")
    return fig


def _chart_rolling_correlation(spot: pd.Series, cvol: pd.Series, asset: str, window: int = ROLLING_WINDOW) -> go.Figure:
    """Rolling correlation of spot vs. CVOL (30d ATM IV) — named for the metric
    actually used (history.iv_series_at_dte), to avoid confusion with the real
    DVOL index used only in the candlestick panel (see module docstring)."""
    merged = pd.merge_asof(pd.DataFrame({"spot": _normalize_dt_index(spot.sort_index())}), pd.DataFrame({"cvol": _normalize_dt_index(cvol.sort_index())}),
                            left_index=True, right_index=True, direction="nearest", tolerance=pd.Timedelta("12h")).dropna()
    if len(merged) < window:
        return _insufficient_data_fig(f"Spot-CVOL Rolling Correlation — {asset}", f"Need a full {window}-point window; have {len(merged)}.")
    corr = merged["spot"].rolling(window).corr(merged["cvol"]).dropna()
    if len(corr) < MIN_ROLLING_POINTS:
        note = f" — thin: only {len(corr)} rolling point(s)"
    else:
        note = ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=corr.index, y=corr.values, mode="lines", name="Rolling correlation", line=dict(color="#3790C7", width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color="#9aa4b2")
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Spot-CVOL Rolling {window}d Correlation — {asset}{note}", xaxis_title="Date", yaxis_title="Correlation", height=CHART_HEIGHT, yaxis=dict(range=[-1, 1]))
    return fig


def _chart_rolling_covariance(spot: pd.Series, cvol: pd.Series, asset: str, window: int = ROLLING_WINDOW) -> go.Figure:
    merged = pd.merge_asof(pd.DataFrame({"spot": _normalize_dt_index(spot.sort_index())}), pd.DataFrame({"cvol": _normalize_dt_index(cvol.sort_index())}),
                            left_index=True, right_index=True, direction="nearest", tolerance=pd.Timedelta("12h")).dropna()
    if len(merged) < window:
        return _insufficient_data_fig(f"Spot-CVOL Rolling Covariance — {asset}", f"Need a full {window}-point window; have {len(merged)}.")
    cov = merged["spot"].rolling(window).cov(merged["cvol"]).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cov.index, y=cov.values, mode="lines", name="Rolling covariance", line=dict(color="#e65100", width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color="#9aa4b2")
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Spot-CVOL Rolling {window}d Covariance — {asset}", xaxis_title="Date", yaxis_title="Covariance", height=CHART_HEIGHT)
    return fig


def _area_ratio_spread(num: str, den: str, days: int, ratio_type: str) -> go.Figure:
    num_s, _, _ = _cvol_series(num, days)
    den_s, _, _ = _cvol_series(den, days)
    if num_s.empty or den_s.empty:
        return _insufficient_data_fig(f"{num}/{den} DVol Ratio & Spread", "Insufficient CVOL history for one or both legs.")
    idx = num_s.index.union(den_s.index)
    n = num_s.reindex(idx).interpolate(limit_direction="both")
    d = den_s.reindex(idx).interpolate(limit_direction="both")
    if ratio_type == "Ratio":
        y = (n / d).dropna()
        y_label = f"{num}/{den} Vol Ratio"
    else:
        y = (n - d).dropna()
        y_label = f"{num}-{den} Vol Spread (pts)"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y.values, mode="lines", name=y_label, line=dict(color="#3790C7", width=2), fill="tozeroy", fillcolor="rgba(55,144,199,0.12)"))
    fig.update_layout(**PLOTLY_LAYOUT, title=f"{num}/{den} DVol {ratio_type}", xaxis_title="Date", yaxis_title=y_label, height=CHART_HEIGHT, hovermode="x unified")
    return fig


def _chart_vol_prediction(asset: str, prediction_windows: tuple) -> go.Figure:
    """Vol prediction vs current pricing across four windows, four models
    (linear/quadratic/cubic/exponential, time-weighted by recency) — full
    port of exodus's _chart_vol_prediction_vs_rr, the only one of its five
    vol-prediction functions actually wired into the dashboard/report (see
    module docstring)."""
    min_points = 5
    max_days = max(prediction_windows)
    spot_full = _spot_series(asset, max_days)
    cvol_full, _, _ = _cvol_series(asset, max_days)
    if spot_full.empty or cvol_full.empty:
        return _insufficient_data_fig(f"Vol prediction vs current pricing — {asset}", "Insufficient data.", height=CHART_HEIGHT * 2)

    merged_full = _align_to_spot(cvol_full, spot_full)
    if merged_full.empty:
        return _insufficient_data_fig(f"Vol prediction vs current pricing — {asset}", "Insufficient overlapping (spot, vol) data.", height=CHART_HEIGHT * 2)

    end_t = merged_full.index.max()
    current_spot = float(merged_full["spot"].iloc[-1])
    current_vol = float(merged_full["vol"].iloc[-1])
    surface_30d = _current_30d_surface(asset)

    labels = [_prediction_window_label(int(d)) for d in prediction_windows]
    subplot_titles = list(labels)
    x_ranges: list = [None, None, None, None]
    y_ranges: list = [None, None, None, None]
    fig = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles.copy(), vertical_spacing=0.14, horizontal_spacing=0.08)

    for i, lookback_days in enumerate(prediction_windows):
        row, col = (i // 2) + 1, (i % 2) + 1
        start_t = end_t - timedelta(days=lookback_days)
        window_df = merged_full.loc[merged_full.index >= start_t]
        if len(window_df) < min_points:
            fig.add_annotation(text=f"Not enough {asset} data<br>({len(window_df)} of {min_points} points)",
                                xref=["x", "x2", "x3", "x4"][i], yref=["y", "y2", "y3", "y4"][i], x=0.5, y=0.5, showarrow=False, font=dict(size=11))
            continue

        spot = window_df["spot"].to_numpy(dtype=float)
        vol = window_df["vol"].to_numpy(dtype=float)
        end_t_window = window_df.index.max()
        days_ago = (end_t_window - window_df.index).total_seconds().to_numpy() / 86400.0
        weights = np.exp(-2.0 * days_ago / max(lookback_days, 1))

        ss_tot = np.sum((vol - np.mean(vol)) ** 2)

        def _r2(pred):
            return float(1 - np.sum((vol - pred) ** 2) / ss_tot) if pred is not None and ss_tot > 0 else np.nan

        coefs_lin = _safe_polyfit(spot, vol, 1, weights)
        coefs_quad = _safe_polyfit(spot, vol, 2, weights)
        coefs_cub = _safe_polyfit(spot, vol, 3, weights)
        vol_clip = np.maximum(vol, 0.5)
        coefs_exp = _safe_polyfit(spot, np.log(vol_clip), 1, weights)

        r2_lin = _r2(np.polyval(coefs_lin, spot) if coefs_lin is not None else None)
        r2_quad = _r2(np.polyval(coefs_quad, spot) if coefs_quad is not None else None)
        r2_cub = _r2(np.polyval(coefs_cub, spot) if coefs_cub is not None else None)
        r2_exp = _r2(np.exp(np.polyval(coefs_exp, spot)) if coefs_exp is not None else None)

        pred_lin = float(np.polyval(coefs_lin, current_spot)) if coefs_lin is not None else np.nan
        pred_quad = float(np.polyval(coefs_quad, current_spot)) if coefs_quad is not None else np.nan
        pred_cub = float(np.polyval(coefs_cub, current_spot)) if coefs_cub is not None else np.nan
        pred_exp = float(np.exp(np.polyval(coefs_exp, current_spot))) if coefs_exp is not None else np.nan

        spot_min, spot_max = float(np.min(spot)), float(np.max(spot))
        pad = (spot_max - spot_min) * 0.05 or 1
        margin = max(0.15 * current_spot, (spot_max - spot_min) * 0.4)
        x_axis_min = min(spot_min - pad, current_spot - margin)
        x_axis_max = max(spot_max + pad, current_spot + margin)
        if surface_30d:
            pts = _surface_30d_to_strike_vol_points(surface_30d, current_spot)
            if pts:
                x_axis_min = min(x_axis_min, min(p[0] for p in pts) * 0.98)
                x_axis_max = max(x_axis_max, max(p[0] for p in pts) * 1.02)
        x_ranges[i] = (x_axis_min, x_axis_max)

        r2t = lambda v: f"{v:.2f}" if np.isfinite(v) else "—"
        subplot_titles[i] = f"{labels[i]} (L/Q/C/E R²={r2t(r2_lin)}/{r2t(r2_quad)}/{r2t(r2_cub)}/{r2t(r2_exp)}, n={len(window_df)})"

        x_line = np.linspace(x_axis_min, x_axis_max, 100)
        y_lin = np.polyval(coefs_lin, x_line) if coefs_lin is not None else None
        y_quad = np.polyval(coefs_quad, x_line) if coefs_quad is not None else None
        y_cub = np.polyval(coefs_cub, x_line) if coefs_cub is not None else None
        y_exp = np.exp(np.polyval(coefs_exp, x_line)) if coefs_exp is not None else None

        show_leg = i == 0
        vol_display = np.maximum(0.0, vol)
        y_from_dots = [v for v in list(vol_display) + [current_vol, pred_lin, pred_quad, pred_cub, pred_exp] if np.isfinite(v)]
        y_from_dots = [max(0.0, v) for v in y_from_dots] or [0.0]
        if surface_30d:
            pts = _surface_30d_to_strike_vol_points(surface_30d, current_spot)
            y_from_dots.extend(max(0.0, p[1]) for p in pts if np.isfinite(p[1]))
        y_dot_min, y_dot_max = min(y_from_dots), max(y_from_dots)
        ypad = max(2.0, (y_dot_max - y_dot_min) * 0.08) if y_dot_max > y_dot_min else 2.0
        y_ranges[i] = (max(0.0, y_dot_min - ypad), y_dot_max + ypad)

        days_ago_max = max(float(days_ago.max()) if len(days_ago) else 1.0, 1.0)
        marker_kw = dict(size=8, color=days_ago, colorscale="Portland", cmin=0, cmax=days_ago_max)
        if i == 0:
            marker_kw["colorbar"] = dict(title="Days Ago")
        else:
            marker_kw["showscale"] = False
        fig.add_trace(go.Scatter(x=window_df["spot"], y=vol_display, mode="markers", marker=marker_kw,
                                  hovertemplate="Spot: %{x:,.0f}<br>CVOL: %{y:.2f}%<extra></extra>",
                                  name="(spot, vol)", showlegend=show_leg, legendgroup="scatter"), row=row, col=col)
        for y_curve, colour, dash, cname, lgroup in (
            (y_lin, "#1565c0", "solid", "Linear", "lin"), (y_quad, "#e65100", "dash", "Quadratic", "quad"),
            (y_cub, "#2e7d32", "dot", "Cubic", "cub"), (y_exp, "#7b1fa2", "dashdot", "Exponential", "exp"),
        ):
            if y_curve is None:
                continue
            fig.add_trace(go.Scatter(x=x_line, y=y_curve, mode="lines", line=dict(color=colour, width=2, dash=dash),
                                      name=cname, showlegend=show_leg, legendgroup=lgroup), row=row, col=col)
        fig.add_trace(go.Scatter(x=[current_spot], y=[max(0.0, current_vol)], mode="markers", marker=dict(size=10, color="#c62828", symbol="diamond"),
                                  name="Current vol", showlegend=show_leg, legendgroup="cur"), row=row, col=col)
        for pred, colour, symbol, pname, lgroup in (
            (pred_lin, "#1565c0", "circle", "Linear pred", "linp"), (pred_quad, "#e65100", "square", "Quadratic pred", "quadp"),
            (pred_cub, "#2e7d32", "triangle-up", "Cubic pred", "cubp"), (pred_exp, "#7b1fa2", "cross", "Exponential pred", "expp"),
        ):
            if not np.isfinite(pred):
                continue
            fig.add_trace(go.Scatter(x=[current_spot], y=[max(0.0, pred)], mode="markers", marker=dict(size=8, color=colour, symbol=symbol),
                                      name=pname, showlegend=show_leg, legendgroup=lgroup), row=row, col=col)
        if surface_30d:
            pts = _surface_30d_to_strike_vol_points(surface_30d, current_spot)
            if len(pts) >= 2:
                fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[max(0.0, p[1]) for p in pts], mode="lines+markers+text",
                                          text=[p[2] for p in pts], textposition="top center",
                                          line=dict(color="#00695c", width=2, dash="dot"),
                                          marker=dict(size=10, color="#00695c", symbol="circle-open", line=dict(width=2)),
                                          textfont=dict(size=9), name="30d surface", showlegend=show_leg, legendgroup="surface"), row=row, col=col)

    xrefs, yrefs = ["x", "x2", "x3", "x4"], ["y domain", "y2 domain", "y3 domain", "y4 domain"]
    for i in range(4):
        fig.add_shape(type="line", x0=current_spot, x1=current_spot, y0=0, y1=1, xref=xrefs[i], yref=yrefs[i], line=dict(dash="dot", color="gray", width=1))

    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=f"Vol prediction vs current pricing (linear/quadratic/cubic/exponential by window) — {asset}", font=dict(size=14)),
                       height=CHART_HEIGHT * 2, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    for i in range(4):
        r, c = (i // 2) + 1, (i % 2) + 1
        fig.update_xaxes(title_text="Spot", row=r, col=c)
        fig.update_yaxes(title_text="CVOL (%)", row=r, col=c)
        if x_ranges[i] is not None:
            fig.update_xaxes(range=x_ranges[i], row=r, col=c)
        if y_ranges[i] is not None:
            fig.update_yaxes(range=y_ranges[i], row=r, col=c)
        if i < len(fig.layout.annotations):
            fig.layout.annotations[i].text = subplot_titles[i]
    return fig


# ============================================================================
# SUMMARY STATS
# ============================================================================

@st.cache_data(ttl=300, max_entries=8, show_spinner=False)
def _latest_dvol_index(asset: str) -> float | None:
    """Latest real Deribit DVOL index value — distinct from the reconstructed
    CVOL (30d ATM IV) used everywhere else on this page; see module docstring."""
    df = deribit.get_dvol(ASSET_CONFIG[asset]["deribit_ccy"], resolution="60")
    if df is None or df.empty:
        return None
    return float(df["close"].iloc[-1])


def get_summary_stats(days: int) -> tuple[pd.DataFrame, float | None]:
    """Latest spot / CVOL / 25Δ skew / real DVOL index / rolling spot-CVOL
    correlation per asset, plus the ETH/BTC CVOL ratio — a compact,
    at-a-glance version of exodus's summary-stats block."""
    rows = []
    cvol_latest: dict[str, float] = {}
    for asset in SV_ASSETS:
        spot = _spot_series(asset, days)
        cvol, _, _ = _cvol_series(asset, days)
        svol, _, _ = _svol_series(asset, days)
        if not cvol.empty:
            cvol_latest[asset] = float(cvol.iloc[-1])
        corr = np.nan
        merged = _align_to_spot(cvol, spot)
        if len(merged) >= ROLLING_WINDOW:
            corr_series = merged["spot"].rolling(ROLLING_WINDOW).corr(merged["vol"]).dropna()
            if not corr_series.empty:
                corr = float(corr_series.iloc[-1])
        rows.append({
            "Asset": asset,
            "Spot": float(spot.iloc[-1]) if not spot.empty else np.nan,
            "CVOL 30d (%)": cvol_latest.get(asset, np.nan),
            "25Δ Skew (%)": float(svol.iloc[-1]) if not svol.empty else np.nan,
            "DVOL Index": _latest_dvol_index(asset),
            f"{ROLLING_WINDOW}d Spot-CVOL Corr": corr,
        })
    ratio = None
    if "ETH" in cvol_latest and "BTC" in cvol_latest and cvol_latest["BTC"]:
        ratio = cvol_latest["ETH"] / cvol_latest["BTC"]
    return pd.DataFrame(rows), ratio


# ============================================================================
# ASSET TAB
# ============================================================================

def render_asset_tab(asset: str, days: int, resolution: str, ratio_type: str, prediction_windows: tuple) -> None:
    spot = _spot_series(asset, days)
    if spot.empty:
        st.warning(f"No {asset} spot data available for this window.")
        return
    cvol, cvol_est, cvol_src = _cvol_series(asset, days)
    svol, svol_est, svol_src = _svol_series(asset, days)
    rr, rr_est, rr_src = _rr_series(asset, days)
    bf, bf_est, bf_src = _bf_series(asset, days)
    other = "ETH" if asset == "BTC" else "BTC"

    c1, c2 = st.columns(2)
    with c1:
        _show(_scatter_vs_spot(_align_to_spot(cvol, spot), asset, f"CVOL vs Spot — {asset}", "CVOL (%)", cvol_est, cvol_src), f"{asset}_cvol_spot")
    with c2:
        _show(_scatter_vs_spot(_align_to_spot(svol, spot), asset, f"25Δ Skew vs Spot — {asset}", "25Δ Skew (%)", svol_est, svol_src), f"{asset}_svol_spot")

    c3, c4 = st.columns(2)
    with c3:
        if other in SV_ASSETS:
            _show(_area_ratio_spread("ETH", "BTC", days, ratio_type), f"{asset}_ethbtc_ratio")
        else:
            st.info(f"Need {other} data (unavailable) to compute the ETH/BTC ratio.")
    with c4:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)
        start_ms, end_ms = int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)
        _show(_candlestick_dvol(asset, start_ms, end_ms, resolution, svol), f"{asset}_dvol_candles")

    c5, c6 = st.columns(2)
    with c5:
        _show(_scatter_vs_spot(_align_to_spot(rr, spot), asset, f"10Δ Risk Reversal vs Spot — {asset}", "10Δ RR (%)", rr_est, rr_src), f"{asset}_rr_spot")
    with c6:
        _show(_scatter_vs_spot(_align_to_spot(bf, spot), asset, f"10Δ Butterfly vs Spot — {asset}", "10Δ BF (%)", bf_est, bf_src), f"{asset}_bf_spot")

    c7, c8 = st.columns(2)
    with c7:
        _show(_chart_rolling_correlation(spot, cvol, asset), f"{asset}_roll_corr")
    with c8:
        _show(_chart_rolling_covariance(spot, cvol, asset), f"{asset}_roll_cov")

    _show(_chart_vol_prediction(asset, prediction_windows), f"{asset}_vol_prediction")


# ============================================================================
# TELEGRAM
# ============================================================================

def _send_chart(fig: go.Figure | None, caption: str) -> bool:
    """fig -> PNG (fx_style.fig_to_png / kaleido) -> telegram.send_photo.
    Mirrors the images-only pattern established in pages/01_MCM_Bot.py and
    pages/06/07 (see CLAUDE.md's 2026-08-20 session-log entries) — a render
    failure is reported, never silently swapped for a text dump."""
    if fig is None:
        return False
    img = fx_style.fig_to_png(fig)
    if img is None:
        img = fx_style.fig_to_png(fig)  # one retry — kaleido occasionally misfires cold
    if img is None:
        return False
    return send_photo(img, caption=caption[:1024])


def send_asset_report_to_telegram(asset: str, days: int, resolution: str, ratio_type: str, prediction_windows: tuple) -> tuple[int, list[str]]:
    """Every chart for one asset's tab, sent as a Telegram photo album
    preceded by a text summary. Returns (sent_count, failed_chart_names)."""
    spot = _spot_series(asset, days)
    if spot.empty:
        return 0, [f"{asset} (no spot data)"]
    cvol, cvol_est, cvol_src = _cvol_series(asset, days)
    svol, svol_est, svol_src = _svol_series(asset, days)
    rr, rr_est, rr_src = _rr_series(asset, days)
    bf, bf_est, bf_src = _bf_series(asset, days)
    other = "ETH" if asset == "BTC" else "BTC"

    latest_cvol = f"{cvol.iloc[-1]:.2f}%" if not cvol.empty else "—"
    latest_skew = f"{svol.iloc[-1]:.2f}%" if not svol.empty else "—"
    send_message(
        f"📈 <b>Spot Vol Correlation – {asset}</b>\n\n"
        f"Spot: ${spot.iloc[-1]:,.2f}\n"
        f"CVOL 30d: {latest_cvol}\n"
        f"25Δ Skew: {latest_skew}"
    )

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_ms, end_ms = int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)

    charts = [
        (_scatter_vs_spot(_align_to_spot(cvol, spot), asset, f"CVOL vs Spot — {asset}", "CVOL (%)", cvol_est, cvol_src), f"{asset} - CVOL vs Spot"),
        (_scatter_vs_spot(_align_to_spot(svol, spot), asset, f"25Δ Skew vs Spot — {asset}", "25Δ Skew (%)", svol_est, svol_src), f"{asset} - 25Δ Skew vs Spot"),
        (_area_ratio_spread("ETH", "BTC", days, ratio_type) if other in SV_ASSETS else None, "ETH/BTC Ratio"),
        (_candlestick_dvol(asset, start_ms, end_ms, resolution, svol), f"{asset} - DVol Snapshot"),
        (_scatter_vs_spot(_align_to_spot(rr, spot), asset, f"10Δ Risk Reversal vs Spot — {asset}", "10Δ RR (%)", rr_est, rr_src), f"{asset} - 10Δ RR vs Spot"),
        (_scatter_vs_spot(_align_to_spot(bf, spot), asset, f"10Δ Butterfly vs Spot — {asset}", "10Δ BF (%)", bf_est, bf_src), f"{asset} - 10Δ BF vs Spot"),
        (_chart_rolling_correlation(spot, cvol, asset), f"{asset} - Rolling Correlation"),
        (_chart_rolling_covariance(spot, cvol, asset), f"{asset} - Rolling Covariance"),
        (_chart_vol_prediction(asset, prediction_windows), f"{asset} - Vol Prediction"),
    ]
    sent, failed = 0, []
    for fig, name in charts:
        if fig is not None and _send_chart(fig, name):
            sent += 1
        elif fig is not None:
            failed.append(name)
    return sent, failed


def send_all_reports_to_telegram(days: int, resolution: str, ratio_type: str, prediction_windows: tuple) -> tuple[int, list[str]]:
    total_sent, total_failed = 0, []
    for asset in SV_ASSETS:
        sent, failed = send_asset_report_to_telegram(asset, days, resolution, ratio_type, prediction_windows)
        total_sent += sent
        total_failed.extend(failed)
    return total_sent, total_failed


# ============================================================================
# MAIN
# ============================================================================

ASSET_NAMES = {"BTC": "₿ Bitcoin (BTC)", "ETH": "⟠ Ethereum (ETH)"}


def main() -> None:
    st.title("📈 Spot Vol Correlation")

    with st.expander("📖 How to Use This Dashboard", expanded=False):
        st.markdown("""
        Explores how **spot price** relates to **options volatility** for BTC and ETH
        (both have a Deribit DVOL index — SOL/HYPE don't, so this page is scoped to
        BTC/ETH only).

        - **CVOL** — 30-day ATM implied vol, reconstructed from the live Deribit
          option chain plus recorded/re-levelled history (`lib/history.py`).
        - **25Δ Skew** — 25Δ call IV − 25Δ put IV at 30 DTE. Positive = calls bid
          over puts (bullish skew); negative = puts bid over calls.
        - **10Δ Risk Reversal / Butterfly** — the same idea further out on the
          wings (10Δ) and as a convexity (butterfly) measure.
        - **DVol Snapshot** — the real Deribit DVOL index candles, with 25Δ skew
          overlaid.
        - **Vol Prediction** — for four lookback windows, fits linear/quadratic/
          cubic/exponential (time-weighted) curves of CVOL vs. spot, then reads
          off each curve's prediction at the *current* spot — i.e. "given how
          vol has historically moved with price in this window, what would vol
          be at today's price." The dotted teal U-shape overlays the live 30d
          option-chain smile for context.

        Colors in the scatter charts = recency (yellow-to-red = most recent).
        """)

    st.markdown("---")

    with st.spinner("Loading summary stats…"):
        summary_df, eth_btc_ratio = get_summary_stats(90)
    if not summary_df.empty:
        st.dataframe(
            summary_df.style.format({
                "Spot": "${:,.2f}", "CVOL 30d (%)": "{:.2f}", "25Δ Skew (%)": "{:.2f}",
                "DVOL Index": "{:.2f}", f"{ROLLING_WINDOW}d Spot-CVOL Corr": "{:.2f}",
            }, na_rep="—"),
            hide_index=True, width="stretch",
        )
    if eth_btc_ratio is not None:
        st.caption(f"ETH/BTC CVOL ratio: **{eth_btc_ratio:.3f}**")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        range_label = st.selectbox("Date range", list(DATE_RANGE_PRESETS.keys()), index=2)
        days = DATE_RANGE_PRESETS[range_label]
    with col2:
        interval_label = st.selectbox("DVol candle interval", list(RESOLUTION_MAP.keys()), index=3)
        resolution = RESOLUTION_MAP[interval_label]
    with col3:
        ratio_type = st.radio("ETH/BTC chart", ["Ratio", "Spread"], horizontal=True)
    with col4:
        st.caption(f"📅 Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    prediction_windows = _prediction_windows_for_lookback(days)

    st.markdown("---")

    tabs = st.tabs([ASSET_NAMES.get(a, a) for a in SV_ASSETS])
    for tab, asset in zip(tabs, SV_ASSETS):
        with tab:
            render_asset_tab(asset, days, resolution, ratio_type, prediction_windows)

    with st.sidebar:
        cache_lib.render_refresh_button(help="Clear cache and refetch price/vol data from Deribit.")

        st.markdown("---")
        st.subheader("📱 Telegram Reports")
        st.caption("Send Spot Vol Correlation reports to Telegram")
        if not is_configured():
            st.warning("Telegram not configured — set bot_token and chat_id (see lib/telegram.py).")
        else:
            if st.button("📤 Send All Reports to Telegram", width="stretch", type="primary", key="sv_telegram_all"):
                with st.spinner("Generating and sending all reports to Telegram..."):
                    sent, failed = send_all_reports_to_telegram(days, resolution, ratio_type, prediction_windows)
                if failed:
                    st.warning(f"Sent {sent} chart(s). Failed: {', '.join(failed)}")
                else:
                    st.success(f"Sent {sent} chart(s) to Telegram.")
            st.caption("Or send one asset:")
            asset_cols = st.columns(len(SV_ASSETS))
            for col, asset in zip(asset_cols, SV_ASSETS):
                with col:
                    if st.button(asset, width="stretch", key=f"sv_telegram_{asset}"):
                        with st.spinner(f"Sending {asset} report..."):
                            sent, failed = send_asset_report_to_telegram(asset, days, resolution, ratio_type, prediction_windows)
                        if failed:
                            st.warning(f"Sent {sent} chart(s). Failed: {', '.join(failed)}")
                        else:
                            st.success(f"Sent {sent} chart(s) to Telegram.")


try:
    main()
except Exception as e:
    st.error(f"❌ **Error loading dashboard:** {str(e)}")
