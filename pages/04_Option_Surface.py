"""
Option Surface — Live option chain viewer with mark vs bid/ask skew tracking.

Shows mark price position relative to bid/ask spread, IV smile, and rolling
skew snapshots to detect market-maker positioning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.deribit import get_option_chain, get_index_price
from lib.constants import ASSET_CONFIG, ASSETS, PLOTLY_LAYOUT

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Option Surface",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "rolling_snapshots" not in st.session_state:
    st.session_state.rolling_snapshots = {}

if "last_snapshot_times" not in st.session_state:
    st.session_state.last_snapshot_times = {}

if "history_snapshots" not in st.session_state:
    st.session_state.history_snapshots = {
        "timestamp": [],
        "avg_mark_minus_mid": [],
        "avg_mark_iv": [],
    }

MAX_HISTORY_MINUTES = 6
SNAPSHOT_INTERVAL_SECONDS = 5


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=15, show_spinner=False)
def fetch_options_summary(currency: str) -> pd.DataFrame | None:
    """Fetch full option book summary from Deribit and parse into DataFrame."""
    cfg = ASSET_CONFIG.get(currency)
    if not cfg:
        return None

    chain = get_option_chain(cfg["deribit_ccy"], "option")
    if not chain:
        return None

    df = pd.DataFrame(chain)
    prefix = cfg["deribit_prefix"]
    df = df[df["instrument_name"].str.startswith(prefix)]

    required_cols = ["instrument_name", "bid_price", "ask_price", "mark_price",
                     "mark_iv", "underlying_price"]
    for col in required_cols:
        if col not in df.columns:
            return None

    df = df[required_cols].copy()
    df["expiry"] = df["instrument_name"].str.extract(r"(\d{1,2}[A-Z]{3}\d{2})")
    df["expiry_date"] = pd.to_datetime(df["expiry"], format="%d%b%y", errors="coerce")
    df["option_type"] = df["instrument_name"].str[-1]
    df["strike"] = df["instrument_name"].str.extract(r"-(\d+)-")[0].astype(float)
    df = df.sort_values("expiry_date", ascending=False)
    df = df.dropna(subset=["strike", "expiry_date", "bid_price", "ask_price",
                           "mark_price", "mark_iv"])
    return df


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def filter_strikes_combined(df: pd.DataFrame, underlying_price: float,
                            above: int = 5, below: int = 5) -> pd.DataFrame:
    """Get N strikes above (calls) and below (puts) the underlying."""
    df = df.dropna(subset=["strike"])
    df_sorted = df.sort_values("strike")
    below_puts = df_sorted[
        (df_sorted["strike"] <= underlying_price) & (df_sorted["option_type"] == "P")
    ].tail(below)
    above_calls = df_sorted[
        (df_sorted["strike"] > underlying_price) & (df_sorted["option_type"] == "C")
    ].head(above)
    combined = pd.concat([below_puts, above_calls]).sort_values("strike")
    return combined


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_skew_iv_chart(df: pd.DataFrame, history=None, window_sec: int = 180,
                        underlying_price: float = 0, strikes_each_side: int = 5) -> go.Figure:
    """Build dual-axis Plotly chart: skew (bps) on left, IV (%) on right."""
    df = filter_strikes_combined(df, underlying_price, above=strikes_each_side,
                                 below=strikes_each_side)
    df = df.dropna(subset=["bid_price", "ask_price", "mark_price", "mark_iv", "strike"])
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
    df["absolute_skew"] = df["mark_price"] - df["mid_price"]

    fig = go.Figure()

    # Live skew
    fig.add_trace(go.Scatter(
        x=df["strike"],
        y=(df["absolute_skew"] * 10000).round(2),
        mode="lines+markers",
        name="Skew Live (bps)",
        line=dict(color="#4e79a7", width=2),
        yaxis="y1",
    ))

    # Live IV
    fig.add_trace(go.Scatter(
        x=df["strike"],
        y=df["mark_iv"].round(2),
        mode="lines+markers",
        name="IV Live (%)",
        line=dict(color="#9467bd", width=2, dash="dot"),
        yaxis="y2",
    ))

    # Rolling median skew from history
    if history:
        cutoff = datetime.datetime.now() - datetime.timedelta(seconds=window_sec)
        window_data = [(ts, snap_df) for ts, snap_df in history if ts >= cutoff]

        if window_data:
            live_strikes = df["strike"].sort_values().unique()
            skew_df = pd.DataFrame(index=live_strikes)

            for _, snap_df in window_data:
                snap_series = pd.Series(
                    data=snap_df["absolute_skew"].values,
                    index=snap_df["strike"].values,
                )
                aligned = snap_series.reindex(live_strikes)
                skew_df = pd.concat([skew_df, aligned], axis=1)

            median_skew = skew_df.median(axis=1, skipna=True)

            fig.add_trace(go.Scatter(
                x=median_skew.index,
                y=(median_skew.values * 10000).round(2),
                mode="lines",
                name=f"Skew {window_sec // 60}m Median (bps)",
                line=dict(color="#f28e2b", width=2, dash="dash"),
                yaxis="y1",
            ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Skew and IV vs Strike",
        xaxis_title="Strike",
        yaxis=dict(title="Skew (bps)", side="left"),
        yaxis2=dict(title="IV (%)", overlaying="y", side="right"),
        legend=dict(x=0, y=-0.25, orientation="h"),
        height=500,
        margin=dict(t=40, b=80),
    )

    return fig


def build_consolidated_skew_table(df: pd.DataFrame) -> None:
    """Show a pivot table of mark-mid skew across expiry and strike."""
    df = df.dropna(subset=["bid_price", "ask_price", "mark_price", "strike", "expiry_date"])
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
    df["mark_minus_mid_bps"] = ((df["mark_price"] - df["mid_price"]) * 10000).round(2)
    df["expiry_date"] = df["expiry_date"].dt.date

    min_strike = int(df["strike"].min())
    max_strike = int(df["strike"].max())
    selected_range = st.slider(
        "Strike range for skew table",
        min_value=min_strike,
        max_value=max_strike,
        value=(min_strike, max_strike),
        step=100,
    )

    df = df[(df["strike"] >= selected_range[0]) & (df["strike"] <= selected_range[1])]
    pivot = df.pivot_table(index="strike", columns="expiry_date", values="mark_minus_mid_bps")
    pivot = pivot.round(2)

    st.subheader("Consolidated Skew Table (Mark - Mid in bps)")
    st.dataframe(pivot, use_container_width=True, height=600)


def build_summary_heatmap(df: pd.DataFrame) -> None:
    """Show top/bottom movers by mark-mid deviation."""
    df = df.dropna(subset=["bid_price", "ask_price", "mark_price"])
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
    df["mark_minus_mid"] = df["mark_price"] - df["mid_price"]

    if df.empty:
        st.warning("No data available for summary.")
        return

    # Filter to near-money strikes per expiry
    final_parts = []
    for _, group in df.groupby("expiry_date"):
        underlying_price = group["underlying_price"].median()
        filtered = filter_strikes_combined(group, underlying_price, above=5, below=5)
        final_parts.append(filtered)

    if not final_parts:
        st.warning("No data after filtering.")
        return

    df = pd.concat(final_parts)
    summary_df = df[["instrument_name", "mark_minus_mid"]].copy()
    summary_df = summary_df.sort_values("mark_minus_mid", ascending=False).reset_index(drop=True)

    # Top/bottom movers bar chart
    n_top = max(1, int(0.05 * len(summary_df)))
    top_movers = summary_df.head(n_top)
    bottom_movers = summary_df.tail(n_top)

    movers_df = pd.concat([
        top_movers.assign(type="Vol Bought"),
        bottom_movers.assign(type="Vol Sold"),
    ]).sort_values("mark_minus_mid", ascending=False)

    colors = movers_df["type"].map({"Vol Bought": "#2ca02c", "Vol Sold": "#d62728"})

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=movers_df["instrument_name"],
        x=movers_df["mark_minus_mid"],
        orientation="h",
        marker_color=colors.tolist(),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Top/Bottom 5% Mark-Mid Deviations",
        xaxis_title="Mark - Mid",
        yaxis_title="Instrument",
        height=max(400, n_top * 30),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Display logic
# ---------------------------------------------------------------------------

def display_options_data(df: pd.DataFrame, underlying_price: float,
                         strikes_each_side: int, expiry_key: str = "",
                         history=None, rolling_window: int = 180) -> None:
    """Render table and chart for a single expiry."""
    combined_df = filter_strikes_combined(df, underlying_price,
                                          above=strikes_each_side, below=strikes_each_side)
    combined_df["mid_price"] = (combined_df["bid_price"] + combined_df["ask_price"]) / 2
    combined_df["absolute_skew"] = combined_df["mark_price"] - combined_df["mid_price"]
    combined_df = combined_df.dropna(subset=["strike", "bid_price", "ask_price",
                                              "mark_price", "mark_iv"])

    combined_df["absolute_skew_bps"] = (combined_df["absolute_skew"] * 10000).round(2)
    display_cols = ["instrument_name", "bid_price", "ask_price", "mark_price",
                    "mark_iv", "absolute_skew_bps"]

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.dataframe(
            combined_df[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=len(combined_df) * 35 + 45,
            hide_index=True,
        )

    with col2:
        fig = build_skew_iv_chart(
            combined_df,
            history=history,
            window_sec=rolling_window,
            underlying_price=underlying_price,
            strikes_each_side=strikes_each_side,
        )
        st.plotly_chart(fig, use_container_width=True)


def display_data(df: pd.DataFrame, expiry_selection: str, strikes_each_side: int,
                 rolling_window_label: str) -> None:
    """Main display dispatcher for selected expiry."""
    rolling_window_map = {"1m": 60, "3m": 180, "5m": 300}
    rolling_window = rolling_window_map[rolling_window_label]

    rolling_snapshots = st.session_state.rolling_snapshots
    last_snapshot_times = st.session_state.last_snapshot_times

    if expiry_selection == "All":
        build_consolidated_skew_table(df)

        for expiry, group in df.groupby("expiry_date"):
            st.subheader(f"Expiry: {expiry.strftime('%Y-%m-%d')}")
            underlying_price = group["underlying_price"].median()

            group = group.dropna(subset=["bid_price", "ask_price", "mark_price",
                                          "mark_iv", "strike"])
            group["mid_price"] = (group["bid_price"] + group["ask_price"]) / 2
            group["absolute_skew"] = group["mark_price"] - group["mid_price"]

            expiry_key = expiry.strftime("%Y-%m-%d")
            now = datetime.datetime.now()

            if expiry_key not in rolling_snapshots:
                rolling_snapshots[expiry_key] = deque(maxlen=MAX_HISTORY_MINUTES * 60)
                last_snapshot_times[expiry_key] = datetime.datetime.min

            if (now - last_snapshot_times[expiry_key]).total_seconds() >= SNAPSHOT_INTERVAL_SECONDS:
                filtered_snapshot = filter_strikes_combined(
                    group, underlying_price, above=strikes_each_side, below=strikes_each_side
                )
                snapshot_df = filtered_snapshot[
                    ["strike", "mark_iv", "absolute_skew", "option_type"]
                ].copy()

                # Prune old entries
                rolling_snapshots[expiry_key] = deque([
                    snap for snap in rolling_snapshots[expiry_key]
                    if (now - snap[0]).total_seconds() <= MAX_HISTORY_MINUTES * 60
                ], maxlen=MAX_HISTORY_MINUTES * 60)

                rolling_snapshots[expiry_key].append((now, snapshot_df))
                last_snapshot_times[expiry_key] = now

            display_options_data(
                group,
                underlying_price,
                strikes_each_side,
                expiry_key=expiry_key,
                history=rolling_snapshots[expiry_key],
                rolling_window=rolling_window,
            )

    else:
        filtered_df = df[df["expiry"] == expiry_selection]
        underlying_price = filtered_df["underlying_price"].median()

        filtered_df = filtered_df.dropna(subset=["bid_price", "ask_price", "mark_price",
                                                   "mark_iv", "strike"])
        filtered_df["mid_price"] = (filtered_df["bid_price"] + filtered_df["ask_price"]) / 2
        filtered_df["absolute_skew"] = filtered_df["mark_price"] - filtered_df["mid_price"]

        expiry_key = expiry_selection
        now = datetime.datetime.now()

        if expiry_key not in rolling_snapshots:
            rolling_snapshots[expiry_key] = deque(maxlen=MAX_HISTORY_MINUTES * 60)
            last_snapshot_times[expiry_key] = datetime.datetime.min

        if (now - last_snapshot_times[expiry_key]).total_seconds() >= SNAPSHOT_INTERVAL_SECONDS:
            snapshot_df = filtered_df[
                ["strike", "mark_iv", "absolute_skew", "option_type"]
            ].copy()

            rolling_snapshots[expiry_key] = deque([
                snap for snap in rolling_snapshots[expiry_key]
                if (now - snap[0]).total_seconds() <= MAX_HISTORY_MINUTES * 60
            ], maxlen=MAX_HISTORY_MINUTES * 60)

            rolling_snapshots[expiry_key].append((now, snapshot_df))
            last_snapshot_times[expiry_key] = now

        display_options_data(
            filtered_df,
            underlying_price,
            strikes_each_side,
            expiry_key=expiry_key,
            history=rolling_snapshots[expiry_key],
            rolling_window=rolling_window,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Option Surface")
st.caption("Live option chain with mark vs bid/ask skew tracking.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    currency = st.selectbox("Currency", ASSETS, index=0)
    refresh_interval = st.selectbox(
        "Refresh Interval",
        options=["5 seconds", "15 seconds", "30 seconds", "1 minute", "5 minutes"],
        index=2,
    )
    interval_map = {
        "5 seconds": 5, "15 seconds": 15, "30 seconds": 30,
        "1 minute": 60, "5 minutes": 300,
    }
    live_data = st.checkbox("Live Data", value=False)
    strikes_each_side = st.selectbox("Strikes Each Side", [5, 10, 15, 20], index=0)

    selected_tab = st.radio("Navigation", ["Options Detail", "Summary"])
    selected_rolling_window = st.selectbox("Rolling Window", ["1m", "3m", "5m"], index=1)

# Auto-refresh
if live_data:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(
            interval=interval_map[refresh_interval] * 1000,
            key="option_surface_refresh",
        )
    except ImportError:
        st.sidebar.warning("Install streamlit-autorefresh for live updates.")

# Fetch data
df = fetch_options_summary(currency)

if df is not None and not df.empty:
    # Expiry selector
    expiries = ["All"] + sorted(df["expiry"].dropna().unique().tolist(), reverse=True)
    expiry_selection = st.sidebar.selectbox("Select Expiry", options=expiries)

    if selected_tab == "Options Detail":
        display_data(df, expiry_selection, strikes_each_side, selected_rolling_window)
    else:
        build_summary_heatmap(df)
else:
    st.warning("No option data available. Check that Deribit is reachable.")
