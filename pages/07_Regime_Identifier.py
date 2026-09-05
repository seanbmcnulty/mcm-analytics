"""
Regime Identifier — Volatility regime classification, breakout-probability
model, and shock/squeeze detection for BTC and ETH perps.

Full port of exodus-analytics's Regime_Identifier.py (analytics_frontend/
streamlit/pages/Regime_Identifier.py, ~4500 lines) onto this app's
Deribit-only data layer — every chart, the calibrated breakout/return-to-low
probability model (16 factors in Low vol, 7 in Moderate/High), the Markov
P^3 forward-regime blend, GARCH(1,1) conditional vol + persistence, shock
and volume-squeeze detection, and the strategic/options-positioning playbook
are all carried over. See lib/regime.py's module docstring for exactly what
changed in the port (data source, asset scope, calibration persistence) and
CLAUDE.md's session log for the full writeup.

Scoped to **BTC and ETH only** (not SOL/HYPE) — a deliberate decision made
with the user, matching this page's DVOL-dependent sibling, Spot Vol
Correlation (pages/08). All analysis is on the perpetual, not spot — this
app has no spot leg.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from lib import regime
from lib.constants import PLOTLY_LAYOUT, ASSET_COLORS
from lib import fx_style
from lib import cache as cache_lib
from lib.telegram import send_message, send_photo, is_configured

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Regime Identifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

REGIME_ASSETS = regime.REGIME_ASSETS  # ["BTC", "ETH"]
REGIME_COLORS = {"Low": "#4CAF50", "Moderate": "#FFC107", "High": "#F44336"}
ASSET_NAMES = {"BTC": "₿ Bitcoin (BTC)", "ETH": "⟠ Ethereum (ETH)"}


def _show(fig: go.Figure | None, key: str) -> None:
    """Standard render: watermark + theme + width='stretch'. No-op on None
    (a chart builder returns None when there isn't enough data)."""
    if fig is None:
        return
    fx_style.add_watermark(fig)
    st.plotly_chart(fx_style.apply_theme(fig), width="stretch", key=key)


# ============================================================================
# CHART BUILDERS
# ============================================================================

def create_regime_gauge(current_rv: float, current_regime: str, asset: str = "BTC") -> go.Figure:
    thresholds = regime.VOL_THRESHOLDS.get(asset, {"low": regime.LOW_VOL_THRESHOLD, "high": regime.HIGH_VOL_THRESHOLD})
    low_thresh, high_thresh = thresholds["low"], thresholds["high"]
    delta_reference = (low_thresh + high_thresh) / 2
    axis_max = 150
    gauge_color = {"Low": "#4CAF50", "Moderate": "#FFC107", "High": "#F44336"}.get(current_regime, "#9aa4b2")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_rv,
        title={"text": "Current Annualized Volatility (%)", "font": {"size": 18}},
        delta={"reference": delta_reference, "position": "top"},
        gauge={
            "axis": {"range": [None, axis_max], "tickwidth": 1},
            "bar": {"color": gauge_color},
            "borderwidth": 1,
            "steps": [
                {"range": [0, low_thresh], "color": "rgba(76,175,80,0.35)"},
                {"range": [low_thresh, high_thresh], "color": "rgba(255,193,7,0.35)"},
                {"range": [high_thresh, axis_max], "color": "rgba(244,67,54,0.35)"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": current_rv},
        },
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280)
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def create_days_in_regime_gauge(days_in_regime: int, current_regime: str, breakout_prob: float) -> go.Figure:
    gauge_color = "red" if breakout_prob > 70 else ("orange" if breakout_prob > 40 else "#3790C7")
    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=days_in_regime,
        title={"text": f"Days in {current_regime} Regime", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 90]},
            "bar": {"color": gauge_color},
            "steps": [
                {"range": [0, 45], "color": "rgba(154,164,178,0.25)"},
                {"range": [45, regime.BREAKOUT_PROBABILITY_THRESHOLD], "color": "rgba(255,193,7,0.35)"},
                {"range": [regime.BREAKOUT_PROBABILITY_THRESHOLD, 90], "color": "rgba(255,152,0,0.35)"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": regime.BREAKOUT_PROBABILITY_THRESHOLD},
        },
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=250)
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def create_price_chart_with_regime(df: pd.DataFrame, asset: str = "BTC") -> go.Figure:
    fig = go.Figure()
    d = df if "regime" in df.columns else regime.add_regime_classification(df, asset=asset)
    regime_colors = {"Low": "rgba(76,175,80,0.18)", "Moderate": "rgba(255,193,7,0.18)", "High": "rgba(244,67,54,0.18)"}
    changes = d[d["regime"] != d["regime"].shift(1)].index
    prev_idx = d.index[0]
    for change_idx in changes:
        r = d.loc[prev_idx, "regime"]
        if r != "Unknown":
            fig.add_vrect(x0=prev_idx, x1=change_idx, fillcolor=regime_colors.get(r, "rgba(154,164,178,0.15)"), layer="below", line_width=0)
        prev_idx = change_idx
    final_regime = d["regime"].iloc[-1]
    if final_regime != "Unknown":
        fig.add_vrect(x0=prev_idx, x1=d.index[-1], fillcolor=regime_colors.get(final_regime, "rgba(154,164,178,0.15)"), layer="below", line_width=0)
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name=f"{asset} Price",
                              line=dict(color=ASSET_COLORS.get(asset, "#fafafa"), width=2),
                              hovertemplate="<b>%{x}</b><br>Price: $%{y:,.2f}<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, title=f"{asset} Price with Volatility Regime Overlay",
                       xaxis_title="Date", yaxis_title="Price (USD)", height=400, hovermode="x unified")
    return fig


def create_volatility_time_series(df: pd.DataFrame, asset: str = "BTC", rv_col: str = "rv_30d", rv_window_days: int = 30) -> go.Figure:
    if rv_col not in df.columns:
        rv_col, rv_window_days = "rv_30d", 30
    thresholds = regime.VOL_THRESHOLDS.get(asset, {"low": regime.LOW_VOL_THRESHOLD, "high": regime.HIGH_VOL_THRESHOLD})
    low_thresh, high_thresh = thresholds["low"], thresholds["high"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[rv_col], mode="lines", name=f"{rv_window_days}-Day RV",
                              line=dict(color="#3790C7", width=2), fill="tozeroy", fillcolor="rgba(55,144,199,0.12)",
                              hovertemplate="<b>%{x}</b><br>Volatility: %{y:.2f}%<extra></extra>"))
    fig.add_hline(y=low_thresh, line_dash="dash", line_color="#4CAF50", annotation_text=f"Low ({low_thresh:.0f}%)", annotation_position="right")
    fig.add_hline(y=high_thresh, line_dash="dash", line_color="#F44336", annotation_text=f"High ({high_thresh:.0f}%)", annotation_position="right")
    fig.update_layout(**PLOTLY_LAYOUT, title=f"{rv_window_days}-Day Rolling Annualized Realized Volatility ({asset})",
                       xaxis_title="Date", yaxis_title="Volatility (%)", height=400, hovermode="x unified")
    return fig


def create_regime_timeline(df: pd.DataFrame, asset: str = "BTC") -> go.Figure:
    d = df if "regime" in df.columns else regime.add_regime_classification(df, asset=asset)
    periods, current_regime, start_date = [], None, None
    for date, row in d.iterrows():
        r = row["regime"]
        if r != current_regime:
            if current_regime is not None and start_date is not None:
                periods.append({"regime": current_regime, "start": start_date, "end": date, "duration": int((date - start_date).total_seconds() / 86400)})
            current_regime, start_date = r, date
    if current_regime is not None and start_date is not None:
        periods.append({"regime": current_regime, "start": start_date, "end": d.index[-1], "duration": int((d.index[-1] - start_date).total_seconds() / 86400)})
    if not periods:
        fig = go.Figure()
        fig.add_annotation(text="No regime data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(**PLOTLY_LAYOUT, height=400)
        return fig
    fig = go.Figure()
    for period in periods[-10:]:
        color = REGIME_COLORS.get(period["regime"], "#9aa4b2")
        fig.add_trace(go.Bar(x=[period["duration"]], y=[f"{period['start']:%Y-%m-%d} to {period['end']:%Y-%m-%d}"],
                              orientation="h", marker_color=color, name=period["regime"],
                              text=f"{period['duration']}d ({period['regime']})", textposition="inside", showlegend=False))
    fig.update_layout(**PLOTLY_LAYOUT, title="Regime History Timeline (Last 10 Periods)",
                       xaxis_title="Duration (Days)", height=max(400, len(periods[-10:]) * 40))
    return fig


def create_volatility_distribution(df: pd.DataFrame, rv_col: str = "rv_30d", rv_window_days: int = 30) -> go.Figure:
    if rv_col not in df.columns:
        rv_col, rv_window_days = "rv_30d", 30
    vol_data = df[rv_col].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=vol_data, nbinsx=50, name="Volatility Distribution", marker_color="#3790C7", opacity=0.7))
    current_rv = df[rv_col].iloc[-1]
    fig.add_vline(x=current_rv, line_dash="dash", line_color="red", line_width=2, annotation_text=f"Current: {current_rv:.1f}%", annotation_position="top")
    fig.add_vline(x=vol_data.quantile(0.5), line_dash="dot", line_color="#9aa4b2", annotation_text="Median")
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Volatility Distribution ({rv_window_days}-Day RV)",
                       xaxis_title="Volatility (%)", yaxis_title="Frequency", height=400)
    return fig


def create_shock_analysis(df: pd.DataFrame, asset: str = "BTC") -> go.Figure:
    n_days = regime.SHOCK_VOLUME_SECTION_DAYS
    df_plot = df.iloc[-n_days:] if len(df) > n_days else df
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["rv_30d"], mode="lines", name="30-Day Volatility", line=dict(color="#3790C7", width=2)))
    shocks = regime.detect_major_shocks(df_plot, asset=asset)
    for shock_date, shock_row in shocks.iterrows():
        fig.add_shape(type="line", x0=shock_date, x1=shock_date, y0=0, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash"))
        vol_at_shock = df_plot.loc[shock_date, "rv_30d"] if shock_date in df_plot.index else df_plot["rv_30d"].max()
        fig.add_annotation(x=shock_date, y=vol_at_shock, text=f"Shock: {shock_row['shock_magnitude']:.1f}%", showarrow=True,
                            arrowhead=2, bgcolor="red", font=dict(color="white", size=10), yshift=20)
    shock_pct = regime.SHOCK_THRESHOLDS_PCT.get(asset, 4.0)
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Volatility Shocks & Persistence ({asset}) — Last 3 Months",
                       xaxis_title="Date", yaxis_title="Volatility (%)", height=400, hovermode="x unified")
    return fig, shock_pct


def create_volume_squeeze_chart(df: pd.DataFrame, asset: str = "BTC") -> go.Figure:
    d = regime.detect_squeeze_periods(df, asset=asset) if "is_squeeze" not in df.columns else df
    n_days = regime.SHOCK_VOLUME_SECTION_DAYS
    df_plot = d.iloc[-n_days:] if len(d) > n_days else d
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot["volume"], name="Daily Volume", marker_color="rgba(55,144,199,0.5)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["volume_30d_avg"], mode="lines", name="30-Day Avg Volume", line=dict(color="#3790C7", width=2)), secondary_y=False)
    squeeze_periods = df_plot[df_plot["is_squeeze"]]
    if len(squeeze_periods) > 0:
        fig.add_trace(go.Scatter(x=squeeze_periods.index, y=squeeze_periods["volume"], mode="markers", name="Squeeze Alert",
                                  marker=dict(color="red", size=9, symbol="triangle-down")), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["close"], mode="lines", name=f"{asset} Price", line=dict(color="#9aa4b2", width=1, dash="dot")), secondary_y=True)
    fig.update_yaxes(title_text="Volume", secondary_y=False)
    fig.update_yaxes(title_text="Price (USD)", secondary_y=True)
    squeeze_pct = int(regime.SQUEEZE_THRESHOLDS.get(asset, 0.5) * 100)
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Volume & Squeeze ({asset}) — Last 3 Months", height=400, hovermode="x unified")
    return fig, squeeze_pct


def create_comparative_volatility(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["rv_7d"], mode="lines", name="7-Day RV", line=dict(color="#9ecae1", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["rv_30d"], mode="lines", name="30-Day RV", line=dict(color="#3790C7", width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["rv_90d"], mode="lines", name="90-Day RV", line=dict(color="#08306b", width=1)))
    fig.update_layout(**PLOTLY_LAYOUT, title="Comparative Volatility Metrics", xaxis_title="Date", yaxis_title="Volatility (%)", height=400, hovermode="x unified")
    return fig


def create_regime_statistics(df: pd.DataFrame, asset: str = "BTC", rv_col: str = "rv_30d") -> go.Figure:
    if rv_col not in df.columns:
        rv_col = "rv_30d"
    d = df if "regime" in df.columns else regime.add_regime_classification(df, asset=asset, rv_col=rv_col)
    fig = go.Figure()
    for r in ("Low", "Moderate", "High"):
        data = d[d["regime"] == r][rv_col].dropna()
        if len(data) > 0:
            fig.add_trace(go.Box(y=data, name=r, boxmean="sd", marker_color=REGIME_COLORS.get(r, "#9aa4b2")))
    fig.update_layout(**PLOTLY_LAYOUT, title="Volatility Statistics by Regime", yaxis_title="Volatility (%)", height=400)
    return fig


def create_return_distribution_by_regime(df: pd.DataFrame, asset: str = "BTC") -> go.Figure:
    d = df if "regime" in df.columns else regime.add_regime_classification(df, asset=asset)
    fig = go.Figure()
    for r in ("Low", "Moderate", "High"):
        returns = d[d["regime"] == r]["log_returns"].dropna()
        if len(returns) > 0:
            fig.add_trace(go.Histogram(x=returns * 100, name=r, opacity=0.65, nbinsx=50, marker_color=REGIME_COLORS.get(r, "#9aa4b2")))
    fig.update_layout(**PLOTLY_LAYOUT, title="Return Distribution by Regime", xaxis_title="Daily Return (%)", yaxis_title="Frequency", height=400, barmode="overlay")
    return fig


def create_regime_transition_matrix(df: pd.DataFrame, asset: str = "BTC") -> go.Figure:
    d = df if "regime" in df.columns else regime.add_regime_classification(df, asset=asset)
    transitions = [(d["regime"].iloc[i], d["regime"].iloc[i + 1]) for i in range(len(d) - 1) if d["regime"].iloc[i] != d["regime"].iloc[i + 1]]
    regimes = ["Low", "Moderate", "High"]
    matrix = np.zeros((3, 3))
    for from_r, to_r in transitions:
        if from_r in regimes and to_r in regimes:
            matrix[regimes.index(from_r)][regimes.index(to_r)] += 1
    fig = go.Figure(data=go.Heatmap(z=matrix, x=regimes, y=regimes, colorscale="Blues", text=matrix.astype(int),
                                     texttemplate="%{text}", textfont={"size": 14}, colorbar=dict(title="Count")))
    fig.update_layout(**PLOTLY_LAYOUT, title="Regime Transition Matrix", xaxis_title="To Regime", yaxis_title="From Regime", height=400)
    return fig


def create_garch_indicators_chart(df: pd.DataFrame, rv_col: str = "rv_30d", rv_window_days: int = 30) -> go.Figure | None:
    if "garch_conditional_vol" not in df.columns:
        return None
    if rv_col not in df.columns:
        rv_col, rv_window_days = "rv_30d", 30
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df["garch_conditional_vol"], mode="lines", name="GARCH Conditional Vol", line=dict(color="#9c27b0", width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df[rv_col], mode="lines", name=f"{rv_window_days}-Day Realized Vol", line=dict(color="#3790C7", width=1, dash="dot"), opacity=0.7), secondary_y=False)
    if "garch_persistence" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["garch_persistence"], mode="lines", name="GARCH Persistence", line=dict(color="red", width=2)), secondary_y=True)
        fig.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="High Persistence (0.95)", secondary_y=True)
    fig.update_yaxes(title_text="Volatility (%)", secondary_y=False)
    fig.update_yaxes(title_text="Persistence", secondary_y=True, range=[0, 1])
    fig.update_layout(**PLOTLY_LAYOUT, title="GARCH Model Indicators", height=400, hovermode="x unified")
    return fig


def create_volatility_term_structure_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    windows = [7, 14, 30, 60, 90]
    colors = ["#9ecae1", "#4292c6", "#2171b5", "#6a51a3", "#3f007d"]
    for window, color in zip(windows, colors):
        col = f"rv_{window}d"
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=f"{window}-Day RV",
                                      line=dict(color=color, width=2 if window == 30 else 1)), secondary_y=False)
    if "term_structure_slope" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["term_structure_slope"], mode="lines", name="Term Structure Slope", line=dict(color="red", width=2)), secondary_y=True)
        fig.add_hline(y=0, line_dash="dash", line_color="#9aa4b2", secondary_y=True)
    fig.update_yaxes(title_text="Volatility (%)", secondary_y=False)
    fig.update_yaxes(title_text="Slope (7d - 90d)", secondary_y=True)
    fig.update_layout(**PLOTLY_LAYOUT, title="Volatility Term Structure Analysis", height=400, hovermode="x unified")
    return fig


def create_compression_indicators_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Compression Indicators", "Percentile Ranks"), vertical_spacing=0.15)
    if "bb_width" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_width"], mode="lines", name="Bollinger Band Width", line=dict(color="#3790C7", width=2)), row=1, col=1)
    if "atr_compression" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["atr_compression"] * 100, mode="lines", name="ATR Compression Ratio", line=dict(color="orange", width=2)), row=1, col=1)
    if "vol_percentile_rank" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["vol_percentile_rank"], mode="lines", name="Vol Percentile Rank", line=dict(color="#4CAF50", width=2), fill="tozeroy", fillcolor="rgba(76,175,80,0.12)"), row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#4CAF50", row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    if "range_compression" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["range_compression"], mode="lines", name="Price Range Compression", line=dict(color="#9c27b0", width=2)), row=2, col=1)
    fig.update_yaxes(title_text="Percentile Rank (%)", row=2, col=1, range=[0, 100])
    fig.update_layout(**PLOTLY_LAYOUT, title="Advanced Compression Indicators", height=600, hovermode="x unified")
    return fig


def create_early_warning_signals_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Volatility Acceleration & Divergence", "Regime Transition Momentum"), vertical_spacing=0.15)
    if "vol_acceleration" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["vol_acceleration"], mode="lines", name="Volatility Acceleration", line=dict(color="red", width=2), fill="tozeroy", fillcolor="rgba(244,67,54,0.12)"), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="#9aa4b2", row=1, col=1)
    if "cross_timeframe_divergence" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["cross_timeframe_divergence"], mode="lines", name="Cross-Timeframe Divergence", line=dict(color="#9c27b0", width=2)), row=1, col=1)
    if "regime_transition_momentum" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["regime_transition_momentum"], name="Regime Transitions (30d)", marker_color="orange", opacity=0.7), row=2, col=1)
    fig.update_layout(**PLOTLY_LAYOUT, title="Early Warning Signals for Regime Shifts", height=600, hovermode="x unified")
    return fig


def create_microstructure_indicators_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "volume_vol_correlation" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["volume_vol_correlation"], mode="lines", name="Volume-Vol Correlation", line=dict(color="#3790C7", width=2)), secondary_y=False)
        fig.add_hline(y=0, line_dash="dash", line_color="#9aa4b2")
    if "vol_of_vol" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["vol_of_vol"], mode="lines", name="Volatility of Volatility", line=dict(color="red", width=2)), secondary_y=True)
    if "range_compression" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["range_compression"], mode="lines", name="Range Compression %", line=dict(color="#4CAF50", width=1, dash="dot"), opacity=0.7), secondary_y=False)
    fig.update_yaxes(title_text="Correlation / Percentile", secondary_y=False, range=[-1, 1])
    fig.update_yaxes(title_text="Volatility of Volatility (%)", secondary_y=True)
    fig.update_layout(**PLOTLY_LAYOUT, title="Market Microstructure Indicators", height=400, hovermode="x unified")
    return fig


def create_cross_asset_volatility_chart(rv_frames: dict) -> go.Figure | None:
    frames = {k: v for k, v in rv_frames.items() if v is not None and not v.empty}
    if len(frames) < 2:
        return None
    common_idx = None
    for v in frames.values():
        common_idx = v.index if common_idx is None else common_idx.intersection(v.index)
    if common_idx is None or len(common_idx) < 30:
        return None
    fig = go.Figure()
    for sym, d in frames.items():
        s = d["rv_30d"].reindex(common_idx).ffill().dropna()
        if len(s) > 0:
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=f"{sym} 30d RV", line=dict(color=ASSET_COLORS.get(sym, "#9aa4b2"), width=2)))
    fig.update_layout(**PLOTLY_LAYOUT, title="Cross-Asset 30-Day Realized Volatility", xaxis_title="Date", yaxis_title="Volatility (%)", height=400, hovermode="x unified")
    return fig


def create_calibration_summary_chart(asset: str, breakout_recs, return_to_low_recs) -> go.Figure | None:
    """Reliability: when the model said 'X% chance', did it happen X% of the time?"""
    has_breakout = bool(breakout_recs) and len(breakout_recs) >= 10
    has_rtl = bool(return_to_low_recs) and len(return_to_low_recs) >= 10
    if not has_breakout and not has_rtl:
        return None
    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    subplot_titles = []
    if has_breakout:
        subplot_titles.append("When in Low vol: did price break out?")
    if has_rtl:
        subplot_titles.append("When in Moderate/High vol: did it return to Low?")
    n_cols = int(has_breakout) + int(has_rtl)
    fig = make_subplots(rows=1, cols=max(1, n_cols), subplot_titles=subplot_titles)
    col = 1
    for records in (breakout_recs, return_to_low_recs):
        if not records or len(records) < 10:
            continue
        raw = np.array([r[0] for r in records])
        y = np.array([r[1] for r in records])
        bins = [0, 20, 40, 60, 80, 100]
        actual_rates = [float(y[(raw >= bins[i]) & (raw < bins[i + 1])].mean() * 100) if ((raw >= bins[i]) & (raw < bins[i + 1])).sum() > 0 else np.nan for i in range(5)]
        fig.add_trace(go.Bar(x=bin_labels, y=actual_rates, marker_color="#3790C7"), row=1, col=col)
        fig.add_trace(go.Scatter(x=bin_labels, y=[10, 30, 50, 70, 90], mode="lines", line=dict(dash="dash", color="#9aa4b2"), name="Perfect calibration"), row=1, col=col)
        col += 1
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Is the probability reliable? ({asset})", height=350, showlegend=False)
    for c in range(1, n_cols + 1):
        fig.update_xaxes(title_text="Probability bucket (model said)", row=1, col=c)
        fig.update_yaxes(title_text="Actual % happened (3d)", row=1, col=c)
    return fig


def create_forward_outcome_chart(asset: str, breakout_recs, rtl_recs) -> go.Figure | None:
    """What actually happened after we showed each probability, next 3 days."""
    if not breakout_recs and not rtl_recs:
        return None
    bucket_labels = ["Low (0-25%)", "Medium (25-50%)", "High (50-75%)", "Very high (75-100%)"]
    rows = []
    if breakout_recs:
        rows.append(("From Low vol: % of times price broke out", breakout_recs))
    if rtl_recs:
        rows.append(("From Moderate/High vol: % of times it returned to Low", rtl_recs))
    if not rows:
        return None
    fig = make_subplots(rows=len(rows), cols=1, subplot_titles=[r[0] for r in rows], vertical_spacing=0.2)
    for row_i, (_, records) in enumerate(rows):
        raw = np.array([r[0] for r in records])
        y = np.array([r[1] for r in records])
        bins = [0, 25, 50, 75, 100]
        rates = [float(y[(raw >= bins[i]) & (raw < bins[i + 1])].mean() * 100) if ((raw >= bins[i]) & (raw < bins[i + 1])).sum() > 0 else np.nan for i in range(4)]
        fig.add_trace(go.Bar(x=bucket_labels, y=rates, marker_color="coral"), row=row_i + 1, col=1)
    fig.update_layout(**PLOTLY_LAYOUT, title=f"What actually happened after we showed each probability? ({asset}) — next 3 days",
                       height=350 if len(rows) == 1 else 450, showlegend=False)
    for r in range(1, len(rows) + 1):
        fig.update_yaxes(title_text="Actual % that made the move", row=r, col=1)
    return fig


def create_regime_duration_distribution_chart(df: pd.DataFrame, asset: str = "BTC") -> go.Figure | None:
    d = df if "regime" in df.columns else regime.add_regime_classification(df, asset=asset)
    periods, start, prev_reg = [], None, None
    for date, row in d.iterrows():
        r = row.get("regime", "Unknown")
        if r != prev_reg:
            if prev_reg is not None and start is not None:
                periods.append({"regime": prev_reg, "duration": (date - start).days})
            prev_reg, start = r, date
    if start is not None:
        periods.append({"regime": prev_reg, "duration": (d.index[-1] - start).days})
    if not periods:
        return None
    pdf = pd.DataFrame(periods)
    fig = go.Figure()
    for r in ("Low", "Moderate", "High"):
        vals = pdf[pdf["regime"] == r]["duration"]
        if len(vals) > 0:
            fig.add_trace(go.Histogram(x=vals, name=r, marker_color=REGIME_COLORS.get(r), opacity=0.7, nbinsx=min(25, max(5, len(vals)))))
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Regime Duration Distribution ({asset})", xaxis_title="Duration (days)", yaxis_title="Count", barmode="overlay", height=350)
    return fig


def create_volatility_percentile_time_series(df: pd.DataFrame, rv_col: str = "rv_30d", rv_window_days: int = 30) -> go.Figure | None:
    if rv_col not in df.columns or len(df) < 60:
        return None
    rv = df[rv_col].dropna()
    win = min(252, len(rv))
    pct = rv.rolling(window=win, min_periods=min(60, win)).apply(lambda x: (x.iloc[-1] >= x).mean() * 100 if len(x) > 0 else np.nan, raw=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pct.index, y=pct.values, mode="lines", name="RV percentile", line=dict(color="#9c27b0", width=2)))
    fig.add_hline(y=50, line_dash="dash", line_color="#9aa4b2")
    fig.update_layout(**PLOTLY_LAYOUT, title=f"{rv_window_days}-Day Volatility Percentile (rolling rank)", xaxis_title="Date", yaxis_title="Percentile (%)", height=350, yaxis=dict(range=[0, 100]))
    return fig


def create_rolling_correlation_chart(df: pd.DataFrame, window: int = 30, rv_col: str = "rv_30d", rv_window_days: int = 30) -> go.Figure | None:
    if "volume" not in df.columns or rv_col not in df.columns or len(df) < window:
        return None
    corr = df["volume"].astype(float).rolling(window=window).corr(df[rv_col])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=corr.index, y=corr.values, mode="lines", name="Volume-RV correlation", line=dict(color="teal", width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color="#9aa4b2")
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Rolling {window}-Day Correlation: Volume vs {rv_window_days}d RV", xaxis_title="Date", yaxis_title="Correlation", height=350, yaxis=dict(range=[-1, 1]))
    return fig


def create_drawdowns_by_regime_chart(df: pd.DataFrame, asset: str = "BTC") -> go.Figure | None:
    d = df if "regime" in df.columns else regime.add_regime_classification(df, asset=asset)
    cum = (1 + d["daily_return"]).cumprod()
    dd_by_regime = {"Low": [], "Moderate": [], "High": []}
    start_idx = 0
    for i in range(1, len(d)):
        if d["regime"].iloc[i] != d["regime"].iloc[i - 1]:
            run = cum.iloc[start_idx:i]
            if len(run) > 1:
                dd = (run - run.cummax()) / run.cummax() * 100
                dd_by_regime.get(d["regime"].iloc[start_idx], []).append(dd.min())
            start_idx = i
    run = cum.iloc[start_idx:]
    if len(run) > 1:
        dd = (run - run.cummax()) / run.cummax() * 100
        dd_by_regime.get(d["regime"].iloc[start_idx], []).append(dd.min())
    fig = go.Figure()
    for r in ("Low", "Moderate", "High"):
        vals = [x for x in dd_by_regime.get(r, []) if not (pd.isna(x) or np.isinf(x))]
        if vals:
            fig.add_trace(go.Box(y=vals, name=r, marker_color=REGIME_COLORS.get(r)))
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Max Drawdown During Regime Spell ({asset})", yaxis_title="Drawdown (%)", height=350)
    return fig


def create_seasonality_heatmap_chart(df: pd.DataFrame) -> go.Figure | None:
    if "rv_30d" not in df.columns or len(df) < 90:
        return None
    d = df.copy()
    d["month"], d["year"] = d.index.month, d.index.year
    this_year = d.index.year.max()
    months = list(range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure()
    past = d.loc[d["year"] < this_year]
    if not past.empty:
        by_month = past.groupby("month")["rv_30d"].mean().reindex(months)
        fig.add_trace(go.Scatter(x=months, y=by_month.values, mode="lines+markers", name="Past years avg", line=dict(color="#3790C7", width=2), marker=dict(size=8)))
    this_yr = d.loc[d["year"] == this_year]
    if not this_yr.empty:
        by_month = this_yr.groupby("month")["rv_30d"].mean().reindex(months)
        fig.add_trace(go.Scatter(x=months, y=by_month.values, mode="lines+markers", name=f"{this_year} (YTD)", line=dict(color="coral", width=2), marker=dict(size=8)))
    fig.update_layout(**PLOTLY_LAYOUT, title="30d RV by Month: This Year vs Past Years", xaxis_title="Month", yaxis_title="Avg 30d RV (%)",
                       height=350, xaxis=dict(tickvals=months, ticktext=month_names), hovermode="x unified")
    return fig


def create_expected_move_by_regime_chart(df: pd.DataFrame, asset: str = "BTC", horizon: int = 3) -> go.Figure | None:
    exp_dict = regime.get_expected_move_by_regime_dict(df, asset=asset, horizon=horizon)
    if not exp_dict:
        return None
    order = [r for r in ("Low", "Moderate", "High") if r in exp_dict]
    fig = go.Figure(go.Bar(x=order, y=[exp_dict[r] for r in order], marker_color=[REGIME_COLORS.get(r, "#9aa4b2") for r in order]))
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Avg |Return| Next {horizon}d by Regime ({asset})", xaxis_title="Regime", yaxis_title="Avg |Return| (%)", height=350)
    return fig


def create_return_skew_by_regime_chart(df: pd.DataFrame, asset: str = "BTC") -> go.Figure | None:
    d = df if "regime" in df.columns else regime.add_regime_classification(df, asset=asset)
    skews = d.groupby(d["regime"])["daily_return"].apply(lambda x: x.skew() if len(x) > 30 else np.nan).dropna()
    if skews.empty:
        return None
    fig = go.Figure(go.Bar(x=skews.index, y=skews.values, marker_color=[REGIME_COLORS.get(r, "#9aa4b2") for r in skews.index]))
    fig.add_hline(y=0, line_dash="dash", line_color="#9aa4b2")
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Return Skewness by Regime ({asset})", xaxis_title="Regime", yaxis_title="Skewness", height=350)
    return fig


def create_autocorrelation_returns_chart(df: pd.DataFrame, lag: int = 1, window: int = 60) -> go.Figure | None:
    ret = df["daily_return"].dropna()
    ac = ret.rolling(window=min(window, len(ret)), min_periods=window // 2).apply(lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan, raw=False).dropna()
    if ac.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ac.index, y=ac.values, mode="lines", name=f"Rolling {lag}-day return autocorr", line=dict(color="#08306b", width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color="#9aa4b2")
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Rolling {window}d Autocorrelation of Returns (lag={lag})", xaxis_title="Date", yaxis_title="Autocorrelation", height=350, yaxis=dict(range=[-1, 1]))
    return fig


def create_breakout_probability_time_series(df: pd.DataFrame, asset: str, learned_weights, cal: dict, rv_col: str = "rv_30d", rv_window_days: int = 30) -> go.Figure:
    """Time series of the regime-transition probability: breakout prob while
    Low, return-to-Low prob while Moderate/High. Marks every regime change."""
    lookback_days = min(730, len(df))
    breakout_probs = regime.calculate_breakout_probability_series(df, lookback_days=lookback_days, asset=asset, learned_weights=learned_weights)
    d = df if "regime" in df.columns else regime.add_regime_classification(df, asset=asset)
    calibrated = pd.Series(index=breakout_probs.index, dtype=float)
    for idx in breakout_probs.index:
        if idx in d.index:
            raw = breakout_probs.loc[idx] if pd.notna(breakout_probs.loc[idx]) else 0
            calibrated.loc[idx] = regime.calibrate_probability(raw, cal, d.loc[idx, "regime"])
        else:
            calibrated.loc[idx] = breakout_probs.loc[idx]
    probs_to_plot = calibrated.reindex(breakout_probs.index).ffill().fillna(0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=probs_to_plot.index, y=probs_to_plot.values, mode="lines", name="Transition Probability",
                              line=dict(color="#9c27b0", width=2), fill="tozeroy", fillcolor="rgba(156,39,176,0.12)"), secondary_y=False)
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="High Risk (70%)", secondary_y=False)
    fig.add_hline(y=40, line_dash="dot", line_color="orange", annotation_text="Moderate Risk (40%)", secondary_y=False)

    changes = d[d["regime"] != d["regime"].shift(1)].index
    for change_date in changes:
        prev_pos = d.index.get_loc(change_date)
        from_regime = d.loc[d.index[prev_pos - 1], "regime"] if prev_pos > 0 else "Unknown"
        to_regime = d.loc[change_date, "regime"]
        prob_at = float(probs_to_plot.loc[change_date]) if change_date in probs_to_plot.index and pd.notna(probs_to_plot.loc[change_date]) else 50
        if from_regime == "Low" and to_regime != "Low":
            fig.add_shape(type="line", x0=change_date, x1=change_date, y0=0, y1=100, line=dict(color="red", width=2, dash="dash"))
            fig.add_annotation(x=change_date, y=prob_at, text=f"Breakout<br>{from_regime}→{to_regime}", showarrow=True, arrowhead=2, bgcolor="red", font=dict(color="white", size=9), yshift=10)
        elif from_regime in ("Moderate", "High") and to_regime == "Low":
            fig.add_shape(type="line", x0=change_date, x1=change_date, y0=0, y1=100, line=dict(color="#4CAF50", width=2, dash="dot"))
            fig.add_annotation(x=change_date, y=prob_at, text="→Low", showarrow=True, arrowhead=2, bgcolor="#4CAF50", font=dict(color="white", size=9), yshift=10)

    if rv_col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[rv_col], mode="lines", name=f"{rv_window_days}-Day RV", line=dict(color="#3790C7", width=1, dash="dot"), opacity=0.5), secondary_y=True)

    fig.update_yaxes(title_text="Transition Probability (%)", secondary_y=False, range=[0, 100])
    fig.update_yaxes(title_text="Volatility (%)", secondary_y=True)
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Regime Transition Probability ({asset}) — Breakout in Low, Return-to-Low in Mod/High",
                       height=500, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


# ============================================================================
# TELEGRAM
# ============================================================================

def _send_chart(fig: go.Figure | None, caption: str) -> bool:
    """fig -> PNG (fx_style.fig_to_png / kaleido) -> telegram.send_photo.
    Mirrors the images-only pattern established in pages/01_MCM_Bot.py and
    pages/06 (see CLAUDE.md's 2026-08-20 session-log entries) — a render
    failure is reported, never silently swapped for a text dump."""
    if fig is None:
        return False
    img = fx_style.fig_to_png(fig)
    if img is None:
        img = fx_style.fig_to_png(fig)  # one retry — kaleido occasionally misfires cold
    if img is None:
        return False
    return send_photo(img, caption=caption[:1024])


def send_asset_report_to_telegram(asset: str, snap: dict, rv_window_days: int) -> tuple[int, list[str]]:
    """Every chart for one asset's dashboard, sent as a Telegram photo album
    preceded by a text summary. Returns (sent_count, failed_chart_names)."""
    df = snap["df"]
    rv_col = snap["rv_col"]
    current_regime = snap["current_regime"]
    prob_label = "Return to Low prob" if current_regime in ("Moderate", "High") else "Breakout prob"
    send_message(
        f"📊 <b>Regime Identifier – {asset}</b>\n\n"
        f"Regime: <b>{current_regime}</b> ({snap['days_in_regime']} days)\n"
        f"{rv_window_days}d RV: {snap['current_rv']:.2f}%\n"
        f"{prob_label}: {snap['breakout_prob']:.0f}%"
    )
    shock_fig, _ = create_shock_analysis(df, asset=asset)
    squeeze_fig, _ = create_volume_squeeze_chart(df, asset=asset)
    charts = [
        (create_regime_gauge(snap["current_rv"], current_regime, asset), f"{asset} - Regime Gauge"),
        (create_days_in_regime_gauge(snap["days_in_regime"], current_regime, snap["breakout_prob"]), f"{asset} - Days in Regime"),
        (create_price_chart_with_regime(df, asset), f"{asset} - Price & Regime"),
        (create_volatility_time_series(df, asset, rv_col, rv_window_days), f"{asset} - Volatility Time Series"),
        (create_regime_timeline(df, asset), f"{asset} - Regime Timeline"),
        (create_volatility_distribution(df, rv_col, rv_window_days), f"{asset} - Volatility Distribution"),
        (shock_fig, f"{asset} - Shock Analysis"),
        (squeeze_fig, f"{asset} - Volume Squeeze"),
        (create_comparative_volatility(df), f"{asset} - Comparative Volatility"),
        (create_regime_statistics(df, asset, rv_col), f"{asset} - Regime Statistics"),
        (create_return_distribution_by_regime(df, asset), f"{asset} - Return Distribution"),
        (create_regime_transition_matrix(df, asset), f"{asset} - Regime Transition Matrix"),
        (create_breakout_probability_time_series(df, asset, snap["learned_weights"], snap["calibration"], rv_col, rv_window_days), f"{asset} - Breakout Probability"),
        (create_garch_indicators_chart(df, rv_col, rv_window_days), f"{asset} - GARCH Indicators"),
        (create_volatility_term_structure_chart(df), f"{asset} - Volatility Term Structure"),
        (create_compression_indicators_chart(df), f"{asset} - Compression Indicators"),
        (create_early_warning_signals_chart(df), f"{asset} - Early Warning Signals"),
        (create_microstructure_indicators_chart(df), f"{asset} - Microstructure Indicators"),
        (create_calibration_summary_chart(asset, snap["backtest_breakout"], snap["backtest_rtl"]), f"{asset} - Calibration Summary"),
        (create_forward_outcome_chart(asset, snap["backtest_breakout"], snap["backtest_rtl"]), f"{asset} - Forward Outcome"),
        (create_regime_duration_distribution_chart(df, asset), f"{asset} - Regime Duration Distribution"),
        (create_volatility_percentile_time_series(df, rv_col, rv_window_days), f"{asset} - Volatility Percentile"),
        (create_rolling_correlation_chart(df, 30, rv_col, rv_window_days), f"{asset} - Rolling Correlation"),
        (create_drawdowns_by_regime_chart(df, asset), f"{asset} - Drawdowns by Regime"),
        (create_seasonality_heatmap_chart(df), f"{asset} - Seasonality"),
        (create_expected_move_by_regime_chart(df, asset), f"{asset} - Expected Move by Regime"),
        (create_return_skew_by_regime_chart(df, asset), f"{asset} - Return Skew by Regime"),
        (create_autocorrelation_returns_chart(df), f"{asset} - Return Autocorrelation"),
    ]
    sent, failed = 0, []
    for fig, name in charts:
        if fig is not None and _send_chart(fig, name):
            sent += 1
        elif fig is not None:
            failed.append(name)
    return sent, failed


def send_all_reports_to_telegram(days_back: int, rv_window_days: int) -> tuple[int, list[str]]:
    total_sent, total_failed = 0, []
    rv_frames = regime.get_cross_asset_rv_frames(days_back)
    cross_fig = create_cross_asset_volatility_chart(rv_frames)
    if cross_fig is not None:
        if _send_chart(cross_fig, "Cross-Asset 30d Volatility (BTC / ETH)"):
            total_sent += 1
        else:
            total_failed.append("Cross-Asset 30d Volatility")
    for asset in REGIME_ASSETS:
        df, err = regime.get_processed_df(asset, days_back)
        if err or df is None:
            total_failed.append(f"{asset} (data unavailable: {err})")
            continue
        snap = regime.compute_snapshot(asset, df, rv_window_days)
        sent, failed = send_asset_report_to_telegram(asset, snap, rv_window_days)
        total_sent += sent
        total_failed.extend(failed)
    return total_sent, total_failed


# ============================================================================
# SIDEBAR PLAYBOOK
# ============================================================================

def render_sidebar_playbook(asset: str, snap: dict) -> None:
    st.sidebar.header(f"🎯 {asset} Strategic Playbook")
    recs = regime.get_strategy_recommendation(
        snap["current_regime"], snap["current_rv"], snap["days_in_regime"], snap["breakout_prob"], snap["shock_info"], asset=asset
    )
    if recs:
        for rec in recs:
            color = {"high": "red", "medium": "orange", "low": "blue"}.get(rec["priority"], "gray")
            st.sidebar.markdown(f"**{rec['title']}**")
            st.sidebar.markdown(f"<div style='border-left: 4px solid {color}; padding-left: 10px;'>{rec['message']}</div>", unsafe_allow_html=True)
            st.sidebar.markdown("---")
    else:
        st.sidebar.info("No specific recommendations at this time.")

    st.sidebar.header(f"📈 {asset} Current Metrics")
    df = snap["df"]
    st.sidebar.metric("Current Price", f"${df['close'].iloc[-1]:,.2f}")
    st.sidebar.metric(f"{snap['rv_window_days']}-Day RV (regime)", f"{snap['current_rv']:.2f}%")
    st.sidebar.metric("7-Day RV", f"{df['rv_7d'].iloc[-1]:.2f}%")
    st.sidebar.metric("30-Day RV", f"{df['rv_30d'].iloc[-1]:.2f}%")
    st.sidebar.metric("90-Day RV", f"{df['rv_90d'].iloc[-1]:.2f}%")

    shock_info = snap["shock_info"]
    if shock_info[0]:
        st.sidebar.header("⚡ Latest Shock")
        st.sidebar.write(f"**Date:** {shock_info[0]:%Y-%m-%d}")
        st.sidebar.write(f"**Magnitude:** {shock_info[1]:.1f}%")
        st.sidebar.write(f"**Persistence:** {shock_info[2]:.0f} days remaining")


# ============================================================================
# ASSET DASHBOARD
# ============================================================================

def render_asset_dashboard(asset: str, days_back: int, rv_window_days: int, quick_view: bool) -> dict | None:
    with st.spinner(f"Loading {asset} data (fetch + indicators)..."):
        df, error_msg = regime.get_processed_df(asset, days_back)
    if error_msg or df is None or df.empty:
        st.error(f"❌ **Error loading data:** {error_msg or 'No data received'}")
        return None
    if len(df) < 2:
        st.warning("Insufficient data for analysis. Need at least 2 data points.")
        return None

    data_through = df.index[-1].strftime("%Y-%m-%d")
    st.success(f"✅ **Data through {data_through}** — {len(df)} points ({df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}) · source: **Deribit {asset}-PERPETUAL**")
    try:
        days_old = (pd.Timestamp.now().normalize() - df.index[-1].normalize()).days
        if days_old > 2:
            st.caption("⚠️ **Data may be delayed** (last close more than 2 days ago).")
    except Exception:
        pass

    snap = regime.compute_snapshot(asset, df, rv_window_days)
    df = snap["df"]
    rv_col = snap["rv_col"]
    current_rv = snap["current_rv"]
    current_regime = snap["current_regime"]
    days_in_regime = snap["days_in_regime"]
    breakout_prob = snap["breakout_prob"]
    shock_info = snap["shock_info"]
    fwd_probs = snap["fwd_probs"]
    term_structure_slope = snap["term_structure_slope"]
    backtest_breakout, backtest_rtl = snap["backtest_breakout"], snap["backtest_rtl"]

    prob_label = "Return to Low Probability" if current_regime in ("Moderate", "High") else "Breakout Probability"
    th = regime.VOL_THRESHOLDS.get(asset, {"low": regime.LOW_VOL_THRESHOLD, "high": regime.HIGH_VOL_THRESHOLD})
    low_pct, high_pct = th["low"], th["high"]

    # --- Summary ---
    st.markdown("## Summary")
    if current_regime == "Low":
        view_3d = "Likely stay Low" if breakout_prob < 50 else f"Breakout risk {breakout_prob:.0f}%"
        stance_short = "Sell vol, shorter tenor; reduce size if breakout > 70%." if breakout_prob >= 70 else "Sell vol; can extend tenor if breakout prob stays low."
    elif current_regime == "High":
        view_3d = "Elevated vol; return-to-Low possible over 3d"
        stance_short = "No new short vol; consider long vol or defined-risk."
    else:
        view_3d = f"Return-to-Low prob {breakout_prob:.0f}%"
        stance_short = "Selective premium sell; add short vol into spikes if return-to-Low prob high."
    st.info(f"**State of the book:** {asset} in **{current_regime}** ({days_in_regime}d). **3d view:** {view_3d}. **Stance:** {stance_short}")

    st.markdown("### Current Regime Status")
    st.caption(f"**Thresholds ({asset}):** Low <{low_pct:.0f}% · Moderate {low_pct:.0f}-{high_pct:.0f}% · High >{high_pct:.0f}%")

    with st.expander("ℹ️ Understanding the Metrics", expanded=False):
        st.markdown(f"""
        **Current Regime** ({asset}): based on {rv_window_days}-day realized volatility — **Low** (<{low_pct:.0f}%, compression/spring loading), **Moderate** ({low_pct:.0f}-{high_pct:.0f}%), **High** (>{high_pct:.0f}%, expansion).

        **{rv_window_days}-Day RV**: annualized realized volatility from rolling {rv_window_days}-day log returns.

        **Probability**: in **Low** vol = breakout probability (to Moderate/High); in **Moderate/High** = return-to-Low probability. Calibrated on whether the move happened **within 3 days**.

        **Days Since Last Shock**: time since last major move (>{regime.SHOCK_THRESHOLDS_PCT.get(asset, 4):.0f}% daily for {asset}). Volatility typically persists ~13 days after a shock.
        """)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Current Regime", current_regime, delta=f"{days_in_regime} days")
    with c2:
        st.metric(f"{rv_window_days}-Day RV", f"{current_rv:.2f}%", delta=f"{df[rv_col].iloc[-1] - df[rv_col].iloc[-2]:.2f}%" if len(df) > 1 else None)
    with c3:
        st.metric(f"{prob_label} (3d)", f"{breakout_prob:.0f}%", delta="High" if breakout_prob > 70 else ("Moderate" if breakout_prob > 0 else "Low"))
    with c4:
        if shock_info[0]:
            st.metric("Days Since Last Shock", f"{(df.index[-1] - shock_info[0]).days}", delta=f"{shock_info[1]:.1f}% move")
        else:
            st.metric("Days Since Last Shock", "N/A")

    rv_pct = snap["rv_percentile"]
    try:
        deribit_iv, deribit_iv_error = regime.get_deribit_atm_iv_30d(asset)
        if not np.isnan(rv_pct):
            vol_rich_cheap = "historically **cheap**" if rv_pct < 30 else ("historically **rich**" if rv_pct > 70 else "historically **neutral**")
            st.caption(f"📊 Current RV at **{rv_pct:.0f}th percentile** (last year) → vol is {vol_rich_cheap}.")
        if deribit_iv is not None:
            iv_pct = deribit_iv * 100
            st.caption(f"📈 **{rv_window_days}d RV:** {current_rv:.2f}% · **30d ATM IV (Deribit):** {iv_pct:.1f}%. {'IV rich vs RV' if iv_pct > current_rv else 'IV cheap vs RV'}.")
        else:
            st.caption(f"📈 **30d ATM IV (Deribit):** — unavailable ({deribit_iv_error or 'no quotes'}).")
    except Exception:
        st.caption(f"📈 **{rv_window_days}d RV / IV:** Unavailable (error loading data).")

    if fwd_probs:
        probs_str = " · ".join(f"P({r})={fwd_probs[r] * 100:.0f}%" for r in ("Low", "Moderate", "High"))
        most_likely = max(fwd_probs, key=fwd_probs.get)
        st.caption(f"🔮 **3d regime:** {probs_str}. Most likely: **{most_likely}**.")

    st.markdown("### Options positioning")
    for label, msg in regime.get_options_positioning(current_regime, days_in_regime, breakout_prob, shock_info, term_structure_slope, asset=asset):
        st.caption(f"**{label}:** {msg}")

    bullets = [f"**{asset}** in **{current_regime}** vol for **{days_in_regime}** days.", f"{rv_window_days}d RV: **{current_rv:.2f}%** (thresholds: <{low_pct:.0f}% / {high_pct:.0f}%+)."]
    if not np.isnan(rv_pct):
        bullets.append(f"Current RV at **{rv_pct:.0f}th percentile** of last year.")
    bullets.append(f"{prob_label}: **{breakout_prob:.0f}%**.")
    if shock_info[0]:
        bullets.append(f"Last shock **{(df.index[-1] - shock_info[0]).days}** days ago ({shock_info[1]:.1f}% move).")
    with st.expander("📌 State of the market (summary)", expanded=True):
        st.markdown("\n".join("• " + b for b in bullets))

    st.markdown("---")

    # --- Charts ---
    st.markdown("## Charts")
    st.markdown("### Key Indicators")
    st.caption("Gauge charts showing current volatility level and regime duration.")
    c1, c2 = st.columns(2)
    with c1:
        _show(create_regime_gauge(current_rv, current_regime, asset), f"{asset}_regime_gauge")
    with c2:
        _show(create_days_in_regime_gauge(days_in_regime, current_regime, breakout_prob), f"{asset}_days_in_regime_gauge")

    st.markdown("### Price & Volatility Analysis")
    st.caption("Price chart with regime overlays and volatility time series. Colored backgrounds indicate volatility regimes.")
    c1, c2 = st.columns(2)
    with c1:
        _show(create_price_chart_with_regime(df, asset), f"{asset}_price_chart")
    with c2:
        _show(create_volatility_time_series(df, asset, rv_col, rv_window_days), f"{asset}_vol_time_series")

    st.markdown("### Regime History & Patterns")
    st.caption("Historical regime durations and volatility distribution (full history).")
    c1, c2 = st.columns(2)
    with c1:
        _show(create_regime_timeline(df, asset), f"{asset}_regime_timeline")
    with c2:
        _show(create_volatility_distribution(df, rv_col, rv_window_days), f"{asset}_vol_distribution")

    st.markdown("### Shock Analysis & Volume")
    shock_fig, shock_pct = create_shock_analysis(df, asset=asset)
    squeeze_fig, squeeze_pct = create_volume_squeeze_chart(df, asset=asset)
    st.caption(f"Major price shocks (>{shock_pct:.0f}% daily) and volume squeeze alerts (<{squeeze_pct}% of 30d avg). Last 3 months.")
    c1, c2 = st.columns(2)
    with c1:
        _show(shock_fig, f"{asset}_shock_analysis")
    with c2:
        _show(squeeze_fig, f"{asset}_volume_squeeze")

    st.markdown("### Statistical Analysis")
    st.caption("Comparative volatility metrics, regime statistics, and return distributions (full history).")
    c1, c2, c3 = st.columns(3)
    with c1:
        _show(create_comparative_volatility(df), f"{asset}_comparative_vol")
    with c2:
        _show(create_regime_statistics(df, asset, rv_col), f"{asset}_regime_stats")
    with c3:
        _show(create_return_distribution_by_regime(df, asset), f"{asset}_return_dist")

    st.markdown("### Regime Transitions")
    _show(create_regime_transition_matrix(df, asset), f"{asset}_transition_matrix")

    st.markdown("### Regime Transition Probability (Reliability)")
    st.caption("When in Low = breakout probability (3d); when in Moderate/High = return-to-Low probability. Red = breakouts, green = return to Low.")
    _show(create_breakout_probability_time_series(df, asset, snap["learned_weights"], snap["calibration"], rv_col, rv_window_days), f"{asset}_breakout_ts")

    n_breakout, n_rtl = regime.get_calibration_outcome_count(backtest_breakout, backtest_rtl)
    n_total = n_breakout + n_rtl
    if n_total == 0:
        st.info("📊 **Calibration & Forward Outcomes** need more history — use 90+ days.")
    cal_fig = create_calibration_summary_chart(asset, backtest_breakout, backtest_rtl)
    fwd_fig = create_forward_outcome_chart(asset, backtest_breakout, backtest_rtl)
    if cal_fig is not None or fwd_fig is not None:
        st.markdown("### Calibration & Forward Outcomes (3-day horizon)")
        st.caption("**Left – Calibration:** when the model said 'X% chance', did it happen X% of the time within 3 days? **Right – Forward outcomes:** for each probability band shown, what % of the time did the move actually occur?")
        cal = snap["calibration"]
        if cal.get("computed_at"):
            st.caption(f"🕐 Calibration fit this session at {cal['computed_at']} UTC (in-session only — not persisted; see CLAUDE.md).")
        if n_total > 0:
            conf = "Low" if n_total < 50 else ("Medium" if n_total < 150 else "High")
            st.caption(f"📊 Based on **{n_total}** historical outcomes (breakout: {n_breakout}, return-to-Low: {n_rtl}). Confidence: **{conf}**.")
        c1, c2 = st.columns(2)
        with c1:
            _show(cal_fig, f"{asset}_calibration_summary")
        with c2:
            _show(fwd_fig, f"{asset}_forward_outcome")

    export_ok = all(c in df.columns for c in ["close", rv_col, "regime"])
    if export_ok:
        last_shock_date = shock_info[0].strftime("%Y-%m-%d") if shock_info[0] else ""
        last_shock_pct = f"{shock_info[1]:.1f}" if shock_info[0] else ""
        summary_row = (
            f"date,symbol,regime,{rv_col},probability,days_in_regime,last_shock_date,last_shock_pct\n"
            f"{data_through},{asset},{current_regime},{current_rv:.2f},{breakout_prob:.0f},{days_in_regime},{last_shock_date},{last_shock_pct}"
        )
        st.download_button("📥 Download summary (CSV)", data=summary_row, file_name=f"regime_summary_{asset}_{data_through}.csv", mime="text/csv", key=f"{asset}_export_csv")

    with st.expander("📊 Regime & Volatility Analytics (duration, percentile, correlation, drawdowns, seasonality, expected move, skew, autocorr)", expanded=True):
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            _show(create_regime_duration_distribution_chart(df, asset), f"{asset}_regime_duration")
        with r1c2:
            _show(create_volatility_percentile_time_series(df, rv_col, rv_window_days), f"{asset}_vol_percentile")
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            _show(create_rolling_correlation_chart(df, 30, rv_col, rv_window_days), f"{asset}_rolling_corr")
        with r2c2:
            _show(create_drawdowns_by_regime_chart(df, asset), f"{asset}_drawdowns")
        r3c1, r3c2 = st.columns(2)
        with r3c1:
            _show(create_seasonality_heatmap_chart(df), f"{asset}_seasonality")
        with r3c2:
            _show(create_expected_move_by_regime_chart(df, asset), f"{asset}_expected_move")
            exp_move = regime.get_expected_move_by_regime_dict(df, asset=asset, horizon=3)
            if exp_move:
                parts = [f"**{r}** ≈ {exp_move[r]:.1f}%" for r in ("Low", "Moderate", "High") if r in exp_move]
                if parts:
                    st.caption("In 3d: " + " · ".join(parts) + " (avg |return| by regime).")
        r4c1, r4c2 = st.columns(2)
        with r4c1:
            _show(create_return_skew_by_regime_chart(df, asset), f"{asset}_return_skew")
        with r4c2:
            _show(create_autocorrelation_returns_chart(df), f"{asset}_autocorr")

    if not quick_view:
        st.markdown("## Advanced")
        if regime.ARCH_AVAILABLE:
            st.markdown("### GARCH Model Indicators")
            st.caption("GARCH conditional volatility forecasts and persistence. High persistence (>0.95) → shocks persist longer, breakout risk up.")
            garch_fig = create_garch_indicators_chart(df, rv_col, rv_window_days)
            if garch_fig is not None:
                _show(garch_fig, f"{asset}_garch")
            else:
                st.info("GARCH chart not available (insufficient data or model did not converge).")
        else:
            st.markdown("### GARCH Model Indicators")
            st.info("GARCH indicators disabled. Install with: `pip install arch`")

        st.markdown("### Volatility Term Structure Analysis")
        st.caption("Volatility across 7d/14d/30d/60d/90d. Inverted term structure (negative slope) suggests compression.")
        _show(create_volatility_term_structure_chart(df), f"{asset}_term_structure")

        st.markdown("### Advanced Compression Indicators")
        st.caption("Bollinger Band width, ATR compression, volatility percentile rank, price range compression. Low values = compression.")
        _show(create_compression_indicators_chart(df), f"{asset}_compression")

        st.markdown("### Early Warning Signals")
        st.caption("Volatility acceleration, cross-timeframe divergence, regime transition momentum.")
        _show(create_early_warning_signals_chart(df), f"{asset}_early_warning")

        st.markdown("### Market Microstructure Indicators")
        st.caption("Volume-volatility correlation, volatility of volatility, price range compression.")
        _show(create_microstructure_indicators_chart(df), f"{asset}_microstructure")

    return snap


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    st.title("📊 Crypto Volatility Regime Dashboard")

    with st.expander("📖 How to Use This Dashboard", expanded=False):
        st.markdown("""
        ### Understanding Volatility Regimes

        Each tab uses **asset-specific** 30-day realized volatility thresholds:

        - **Low Volatility**: Compression phase; volatility suppressed. Like a compressed spring, these periods often precede significant price movements. Historical average duration: 45-60 days.
        - **Moderate Volatility**: Normal market conditions with balanced risk/return.
        - **High Volatility**: Expansion phase with elevated uncertainty and larger price swings.

        **Thresholds by asset:** BTC <40% / 40-60% / >60% — ETH <50% / 50-75% / >75%.

        ### Key Concepts

        **Transition Probability**: Regime-dependent (0-100%). In **Low vol**: breakout probability (to Moderate/High), using duration, compression, squeeze, GARCH, term structure, and other leading indicators. In **Moderate/High**: return-to-Low probability, using historical transition rates, days in elevated regime, proximity to low threshold, and vol decline signals. **Calibration**: fitted in-session from historical outcomes (did the transition happen within 3 days?) — see the caption under each dashboard's Calibration section for why this isn't persisted across sessions on this deployment.

        **GARCH Indicators**: conditional volatility forecasts and persistence (α+β). High persistence (>0.95) → shocks persist longer, breakout risk up.

        **Term Structure**: volatility across multiple timeframes. Inverted (negative slope) → compression, potential breakout.

        **Shocks**: major daily moves (BTC 4%, ETH 5%) that typically keep volatility elevated ~13 days (6.6-day half-life model).

        **Volume Squeeze**: volume dropping below 50% of the 30-day average, often preceding breakouts.

        ### Interpreting Charts

        Look for: compression (7d vol < 30d vol < 90d vol) → potential breakout; extended Low Vol periods (>60 days) → increasing breakout risk; volume squeezes → potential volatility expansion; recent shocks → elevated volatility persistence.
        """)

    st.markdown("---")

    if not regime.ARCH_AVAILABLE:
        st.warning("⚠️ **arch library not found. GARCH indicators will be disabled.** Install with: `pip install arch`")

    with st.spinner("Loading cross-asset 30d RV…"):
        try:
            rv_summary = regime.get_cross_asset_rv_summary(60)
        except Exception:
            rv_summary = {a: None for a in REGIME_ASSETS}
        cols = st.columns([1, 1, 2])
        with cols[0]:
            btc_rv = rv_summary.get("BTC")
            st.metric("₿ BTC 30d RV", f"{btc_rv:.1f}%" if btc_rv is not None else "—", help="Latest 30-day realized vol")
        with cols[1]:
            eth_rv = rv_summary.get("ETH")
            st.metric("⟠ ETH 30d RV", f"{eth_rv:.1f}%" if eth_rv is not None else "—", help="Latest 30-day realized vol")
        with cols[2]:
            st.caption("Select an asset below to load its full dashboard (lazy-loaded).")
        if all(v is None for v in rv_summary.values()):
            st.error("❌ **Cross-asset RV unavailable.** No Deribit perp data could be fetched for BTC or ETH.")
        stance_rows = regime.get_cross_asset_regime_stance(rv_summary)
        if stance_rows:
            st.caption("**By asset:**")
            df_stance = pd.DataFrame(
                [(r[0], r[1], f"{r[2]:.1f}" if r[2] is not None else "—", r[3]) for r in stance_rows],
                columns=["Asset", "Regime", "30d RV (%)", "Stance"],
            )
            st.dataframe(df_stance, hide_index=True, width="stretch")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
    with col1:
        rv_window_days = st.selectbox("Realized vol window", [7, 14, 21, 30], index=3, format_func=lambda x: f"{x} days",
                                       help="Lookback for daily realized vol and regime (7d = most reactive, 30d = smoothest).")
    with col2:
        days_back = st.slider("Days of History", 90, 1200, 720, help="Recommend 90+ days for calibration and forward-outcome charts.")
    with col3:
        selected_asset = st.selectbox("Select Asset", REGIME_ASSETS, format_func=lambda x: ASSET_NAMES.get(x, x), key="regime_asset_selector")
    with col4:
        quick_view = st.checkbox("Quick view", value=False, help="Hide GARCH, term structure, compression, early warning, microstructure")
    st.caption(f"📅 Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    st.markdown("---")

    snap = render_asset_dashboard(selected_asset, days_back, rv_window_days, quick_view)

    with st.expander("📊 Cross-asset 30d volatility comparison (full history)", expanded=False):
        with st.spinner("Loading BTC and ETH for comparison..."):
            rv_frames = regime.get_cross_asset_rv_frames(days_back)
        cross_fig = create_cross_asset_volatility_chart(rv_frames)
        if cross_fig is not None:
            _show(cross_fig, "cross_asset_volatility")
        else:
            st.info("Need at least 30 days of overlapping data for both assets.")

    st.markdown("---")

    with st.sidebar:
        if snap:
            render_sidebar_playbook(selected_asset, snap)
        else:
            st.info("Select an asset and wait for data to load.")

        st.markdown("---")
        cache_lib.render_refresh_button(help="Clear cache and refetch price/vol/calibration data from Deribit.")

        st.markdown("---")
        st.subheader("📱 Telegram Reports")
        st.caption("Send Regime Identifier reports to Telegram")
        if not is_configured():
            st.warning("Telegram not configured — set bot_token and chat_id (see lib/telegram.py).")
        else:
            if st.button("📤 Send All Reports to Telegram", width="stretch", type="primary", key="regime_telegram_all"):
                with st.spinner("Generating and sending all reports to Telegram..."):
                    sent, failed = send_all_reports_to_telegram(days_back, rv_window_days)
                if failed:
                    st.warning(f"Sent {sent} chart(s). Failed: {', '.join(failed)}")
                else:
                    st.success(f"Sent {sent} chart(s) to Telegram.")
            st.caption("Or send one asset:")
            asset_cols = st.columns(len(REGIME_ASSETS))
            for col, asset in zip(asset_cols, REGIME_ASSETS):
                with col:
                    if st.button(asset, width="stretch", key=f"regime_telegram_{asset}"):
                        with st.spinner(f"Sending {asset} report..."):
                            df, err = regime.get_processed_df(asset, days_back)
                            if err or df is None:
                                st.error(f"Could not load {asset} data: {err}")
                            else:
                                asset_snap = regime.compute_snapshot(asset, df, rv_window_days)
                                sent, failed = send_asset_report_to_telegram(asset, asset_snap, rv_window_days)
                                if failed:
                                    st.warning(f"Sent {sent} chart(s). Failed: {', '.join(failed)}")
                                else:
                                    st.success(f"Sent {sent} chart(s) to Telegram.")


try:
    main()
except Exception as e:
    st.error(f"❌ **Error loading dashboard:** {str(e)}")
    st.exception(e)
