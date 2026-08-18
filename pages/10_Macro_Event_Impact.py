"""
Macro Event Impact — Analyze how macro events (CPI, FOMC, NFP) impact crypto
prices and volatility.

Computes surprise z-scores, fetches Deribit index price and DVOL data around
event dates, and visualizes price/vol reactions by event type.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from lib.deribit import get_tradingview_ohlc, get_dvol
from lib.constants import ASSET_CONFIG, ASSET_COLORS, PLOTLY_LAYOUT
from lib import fx_style

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Macro Event Impact",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Macro Event Impact")
st.caption(
    "Analyze how macro events (CPI, FOMC, NFP) impact crypto prices and volatility."
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CSV_PATH = Path(__file__).parent.parent / "data" / "macro_events_calendar.csv"

EVENT_TYPES = ["CPI", "FOMC", "NFP", "PPI", "PCE", "GDP", "Retail Sales", "Other"]

# Assets with index for price data
PRICE_ASSETS = {
    "BTC": ASSET_CONFIG["BTC"],
    "ETH": ASSET_CONFIG["ETH"],
}

# Window around event for price reaction analysis (hours)
REACTION_WINDOW_H = 24

SAMPLE_CSV = """date,event,actual,consensus,prior,currency
2024-01-11,CPI,3.4,3.2,3.1,USD
2024-01-31,FOMC,5.50,5.50,5.50,USD
2024-02-02,NFP,353,185,333,USD
2024-02-13,CPI,3.1,2.9,3.4,USD
2024-03-08,NFP,275,200,229,USD
2024-03-12,CPI,3.2,3.1,3.1,USD
2024-03-20,FOMC,5.50,5.50,5.50,USD
2024-04-05,NFP,303,214,270,USD
2024-04-10,CPI,3.5,3.4,3.2,USD"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_macro_calendar() -> pd.DataFrame | None:
    """Load the macro events calendar CSV. Returns None if file doesn't exist."""
    if not CSV_PATH.exists():
        return None
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=["date"])
        # Ensure required columns
        required = {"date", "event", "actual", "consensus", "prior", "currency"}
        if not required.issubset(set(df.columns)):
            st.error(
                f"CSV is missing required columns. Expected: {sorted(required)}. "
                f"Found: {sorted(df.columns)}"
            )
            return None
        df = df.sort_values("date", ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None


def compute_surprise_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute surprise z-score for each event.
    Surprise = actual - consensus.
    Z-score is standardized within each event type.
    """
    df = df.copy()
    df["actual_num"] = pd.to_numeric(df["actual"], errors="coerce")
    df["consensus_num"] = pd.to_numeric(df["consensus"], errors="coerce")
    df["prior_num"] = pd.to_numeric(df["prior"], errors="coerce")

    # Raw surprise
    df["surprise"] = df["actual_num"] - df["consensus_num"]

    # Z-score within event type
    df["surprise_zscore"] = np.nan
    for event_type in df["event"].unique():
        mask = df["event"] == event_type
        subset = df.loc[mask, "surprise"]
        std = subset.std()
        mean = subset.mean()
        if std and std > 0:
            df.loc[mask, "surprise_zscore"] = (subset - mean) / std
        elif not subset.isna().all():
            df.loc[mask, "surprise_zscore"] = 0.0

    return df


@st.cache_data(ttl=600, show_spinner=False)
def fetch_price_around_event(
    index_name: str, event_ts_ms: int, window_hours: int = 24
) -> pd.DataFrame | None:
    """
    Fetch hourly OHLC data in a window around an event timestamp.
    Returns DataFrame indexed by hours relative to event (T-24 to T+24).
    """
    start_ms = event_ts_ms - window_hours * 3600 * 1000
    end_ms = event_ts_ms + window_hours * 3600 * 1000

    # Use the perpetual for better liquidity data
    instrument = f"{index_name.split('_')[0].upper()}-PERPETUAL"
    df = get_tradingview_ohlc(instrument, "60", start_ms, end_ms)
    if df is None or df.empty:
        return None

    # Compute hours relative to event
    event_time = pd.Timestamp(event_ts_ms, unit="ms", tz="UTC")
    df["hours_from_event"] = (
        (df["timestamp"] - event_time).dt.total_seconds() / 3600
    )

    # Normalize price to % change from event time
    # Find closest candle to T=0
    closest_idx = (df["hours_from_event"].abs()).idxmin()
    base_price = df.loc[closest_idx, "close"]
    if base_price and base_price > 0:
        df["price_pct"] = ((df["close"] - base_price) / base_price) * 100
    else:
        df["price_pct"] = 0.0

    return df


@st.cache_data(ttl=600, show_spinner=False)
def fetch_dvol_around_event(
    currency: str, event_ts_ms: int, window_hours: int = 24
) -> pd.DataFrame | None:
    """
    Fetch DVOL data around an event timestamp.
    Uses hourly resolution (3600).
    """
    start_ms = event_ts_ms - window_hours * 3600 * 1000
    end_ms = event_ts_ms + window_hours * 3600 * 1000

    df = get_dvol(currency, resolution="3600", start_ms=start_ms, end_ms=end_ms)
    if df is None or df.empty:
        return None

    event_time = pd.Timestamp(event_ts_ms, unit="ms", tz="UTC")
    df["hours_from_event"] = (
        (df["timestamp"] - event_time).dt.total_seconds() / 3600
    )

    # Absolute DVOL change from T=0
    closest_idx = (df["hours_from_event"].abs()).idxmin()
    base_dvol = df.loc[closest_idx, "close"]
    if base_dvol and base_dvol > 0:
        df["dvol_change"] = df["close"] - base_dvol
    else:
        df["dvol_change"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

calendar_df = load_macro_calendar()

if calendar_df is None:
    st.warning("Macro events calendar not found.")
    st.markdown("---")
    st.subheader("Setup Instructions")
    st.markdown(
        f"""
        To use this page, create a CSV file at:

        ```
        {CSV_PATH}
        ```

        **Required columns:** `date`, `event`, `actual`, `consensus`, `prior`, `currency`

        - `date`: Event date in YYYY-MM-DD format
        - `event`: Event type (CPI, FOMC, NFP, PPI, PCE, GDP, etc.)
        - `actual`: Released value (leave blank for future events)
        - `consensus`: Market consensus/forecast
        - `prior`: Previous release value
        - `currency`: Currency (typically USD)

        **Sample CSV content:**
        """
    )
    st.code(SAMPLE_CSV, language="csv")
    st.markdown(
        """
        **Data sources:**
        - [Investing.com Economic Calendar](https://www.investing.com/economic-calendar/)
        - [ForexFactory](https://www.forexfactory.com/calendar)
        - [TradingEconomics](https://tradingeconomics.com/calendar)

        Copy historical and upcoming events into the CSV. For future events,
        leave `actual` blank — the page will still show the calendar with
        consensus estimates.
        """
    )
    st.stop()

# ---------------------------------------------------------------------------
# Compute z-scores
# ---------------------------------------------------------------------------

calendar_df = compute_surprise_zscore(calendar_df)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    # Event type filter
    available_events = sorted(calendar_df["event"].unique().tolist())
    selected_events = st.multiselect(
        "Event Types",
        options=available_events,
        default=available_events,
    )

    # Date range filter
    min_date = calendar_df["date"].min().date()
    max_date = calendar_df["date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Asset for price reaction
    reaction_asset = st.selectbox(
        "Price Reaction Asset",
        options=list(PRICE_ASSETS.keys()),
        index=0,
    )

    # Number of events to analyze (for performance)
    max_events_to_fetch = st.slider(
        "Max Events to Analyze (price data)",
        min_value=3,
        max_value=30,
        value=10,
        help="Limit API calls — each event fetches hourly candles from Deribit",
    )

# Apply filters
filtered_df = calendar_df[calendar_df["event"].isin(selected_events)].copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df["date"].dt.date >= start_date)
        & (filtered_df["date"].dt.date <= end_date)
    ]

if filtered_df.empty:
    st.info("No events match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Section 1: Event Calendar Table
# ---------------------------------------------------------------------------

st.subheader("Event Calendar")

# Format display table
display_df = filtered_df[
    ["date", "event", "actual", "consensus", "prior", "surprise", "surprise_zscore"]
].copy()
display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
display_df.columns = ["Date", "Event", "Actual", "Consensus", "Prior", "Surprise", "Z-Score"]

# Color z-scores
st.dataframe(
    display_df.style.format(
        {"Surprise": "{:.2f}", "Z-Score": "{:.2f}"},
        na_rep="—",
    ).background_gradient(
        subset=["Z-Score"],
        cmap="RdYlGn",
        vmin=-3,
        vmax=3,
    ),
    width="stretch",
    height=min(400, 35 * len(display_df) + 38),
)

st.divider()

# ---------------------------------------------------------------------------
# Section 2: Price Reaction Around Events
# ---------------------------------------------------------------------------

st.subheader(f"Price Reaction Around Events ({reaction_asset})")

# Only analyze events with actual data (past events)
past_events = filtered_df[filtered_df["actual_num"].notna()].head(max_events_to_fetch)

if past_events.empty:
    st.info(
        "No past events with actual data in the current selection. "
        "Price reaction analysis requires completed events (actual values)."
    )
else:
    cfg = PRICE_ASSETS[reaction_asset]
    index_name = cfg["index"]

    # Fetch price data around each event
    reaction_data = []
    with st.spinner(f"Fetching {reaction_asset} price data around {len(past_events)} events..."):
        for _, row in past_events.iterrows():
            event_ts_ms = int(row["date"].timestamp() * 1000)
            # Assume events happen ~8:30 AM ET (13:30 UTC) for US data
            event_ts_ms += 13 * 3600 * 1000 + 30 * 60 * 1000

            price_df = fetch_price_around_event(
                index_name, event_ts_ms, REACTION_WINDOW_H
            )
            if price_df is not None and not price_df.empty:
                price_df["event"] = row["event"]
                price_df["event_date"] = row["date"].strftime("%Y-%m-%d")
                price_df["surprise_zscore"] = row["surprise_zscore"]
                price_df["surprise"] = row["surprise"]
                reaction_data.append(price_df)

    if reaction_data:
        all_reactions = pd.concat(reaction_data, ignore_index=True)

        # --- Individual event traces ---
        st.markdown("#### Individual Event Reactions")
        fig_individual = go.Figure()

        for i, (_, row) in enumerate(past_events.iterrows()):
            event_key = f"{row['event']} {row['date'].strftime('%Y-%m-%d')}"
            event_slice = all_reactions[
                all_reactions["event_date"] == row["date"].strftime("%Y-%m-%d")
            ]
            if event_slice.empty:
                continue

            z = row["surprise_zscore"]
            if pd.notna(z) and z > 0.5:
                color = "#2ca02c"  # bullish surprise
            elif pd.notna(z) and z < -0.5:
                color = "#d62728"  # bearish surprise
            else:
                color = "#7f7f7f"  # neutral

            fig_individual.add_trace(go.Scatter(
                x=event_slice["hours_from_event"],
                y=event_slice["price_pct"],
                name=event_key,
                line=dict(width=1.5, color=color),
                opacity=0.7,
            ))

        fig_individual.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig_individual.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)
        fig_individual.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{reaction_asset} Price (% from T=0) Around Macro Events",
            xaxis_title="Hours from Event",
            yaxis_title="Price Change (%)",
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            height=500,
        )
        fx_style.add_watermark(fig_individual)
        st.plotly_chart(fx_style.apply_theme(fig_individual), width="stretch")
        st.caption(
            "Green = bullish surprise (z > 0.5), Red = bearish surprise (z < -0.5), "
            "Gray = neutral."
        )

        # --- Average reaction by event type ---
        st.markdown("#### Average Reaction by Event Type")

        # Group by event type and surprise direction
        avg_reactions = []
        for event_type in all_reactions["event"].unique():
            type_data = all_reactions[all_reactions["event"] == event_type]

            # Bullish surprises
            bullish = type_data[type_data["surprise_zscore"] > 0.5]
            if not bullish.empty:
                avg_bull = (
                    bullish.groupby(
                        bullish["hours_from_event"].round(0)
                    )["price_pct"]
                    .mean()
                    .reset_index()
                )
                avg_bull["category"] = f"{event_type} (Bullish)"
                avg_reactions.append(avg_bull)

            # Bearish surprises
            bearish = type_data[type_data["surprise_zscore"] < -0.5]
            if not bearish.empty:
                avg_bear = (
                    bearish.groupby(
                        bearish["hours_from_event"].round(0)
                    )["price_pct"]
                    .mean()
                    .reset_index()
                )
                avg_bear["category"] = f"{event_type} (Bearish)"
                avg_reactions.append(avg_bear)

            # All (for types like FOMC where direction matters less)
            avg_all = (
                type_data.groupby(
                    type_data["hours_from_event"].round(0)
                )["price_pct"]
                .mean()
                .reset_index()
            )
            avg_all["category"] = f"{event_type} (All)"
            avg_reactions.append(avg_all)

        if avg_reactions:
            avg_df = pd.concat(avg_reactions, ignore_index=True)

            fig_avg = go.Figure()
            categories = avg_df["category"].unique()
            colors = px.colors.qualitative.Set2
            for i, cat in enumerate(categories):
                cat_data = avg_df[avg_df["category"] == cat].sort_values(
                    "hours_from_event"
                )
                fig_avg.add_trace(go.Scatter(
                    x=cat_data["hours_from_event"],
                    y=cat_data["price_pct"],
                    name=cat,
                    line=dict(width=2, color=colors[i % len(colors)]),
                ))

            fig_avg.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
            fig_avg.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)
            fig_avg.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Average {reaction_asset} Reaction by Event Category",
                xaxis_title="Hours from Event",
                yaxis_title="Avg Price Change (%)",
                legend=dict(
                    orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5
                ),
                height=450,
            )
            fx_style.add_watermark(fig_avg)
            st.plotly_chart(fx_style.apply_theme(fig_avg), width="stretch")

        st.divider()

        # --- Scatter: surprise magnitude vs price move ---
        st.markdown("#### Surprise Magnitude vs. Price Move")

        # Compute T+2h and T+24h moves for each event
        scatter_data = []
        for _, row in past_events.iterrows():
            event_date_str = row["date"].strftime("%Y-%m-%d")
            event_slice = all_reactions[
                all_reactions["event_date"] == event_date_str
            ]
            if event_slice.empty or pd.isna(row["surprise_zscore"]):
                continue

            # T+2h move
            t2h = event_slice[
                (event_slice["hours_from_event"] >= 1.5)
                & (event_slice["hours_from_event"] <= 2.5)
            ]
            t2h_move = t2h["price_pct"].mean() if not t2h.empty else np.nan

            # T+24h move
            t24h = event_slice[
                (event_slice["hours_from_event"] >= 23)
                & (event_slice["hours_from_event"] <= 25)
            ]
            t24h_move = t24h["price_pct"].mean() if not t24h.empty else np.nan

            scatter_data.append({
                "event": row["event"],
                "date": event_date_str,
                "surprise_zscore": row["surprise_zscore"],
                "t2h_move": t2h_move,
                "t24h_move": t24h_move,
            })

        if scatter_data:
            scatter_df = pd.DataFrame(scatter_data)

            col_2h, col_24h = st.columns(2)

            with col_2h:
                fig_s2h = go.Figure()
                for event_type in scatter_df["event"].unique():
                    evt_data = scatter_df[scatter_df["event"] == event_type]
                    fig_s2h.add_trace(go.Scatter(
                        x=evt_data["surprise_zscore"],
                        y=evt_data["t2h_move"],
                        mode="markers+text",
                        name=event_type,
                        text=evt_data["date"],
                        textposition="top center",
                        textfont=dict(size=8),
                        marker=dict(size=10, opacity=0.7),
                    ))
                fig_s2h.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
                fig_s2h.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
                fig_s2h.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Surprise Z-Score vs. T+2h Move",
                    xaxis_title="Surprise Z-Score",
                    yaxis_title="Price Move at T+2h (%)",
                    height=400,
                )
                fx_style.add_watermark(fig_s2h)
                st.plotly_chart(fx_style.apply_theme(fig_s2h), width="stretch")

            with col_24h:
                fig_s24h = go.Figure()
                for event_type in scatter_df["event"].unique():
                    evt_data = scatter_df[scatter_df["event"] == event_type]
                    fig_s24h.add_trace(go.Scatter(
                        x=evt_data["surprise_zscore"],
                        y=evt_data["t24h_move"],
                        mode="markers+text",
                        name=event_type,
                        text=evt_data["date"],
                        textposition="top center",
                        textfont=dict(size=8),
                        marker=dict(size=10, opacity=0.7),
                    ))
                fig_s24h.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
                fig_s24h.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
                fig_s24h.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Surprise Z-Score vs. T+24h Move",
                    xaxis_title="Surprise Z-Score",
                    yaxis_title="Price Move at T+24h (%)",
                    height=400,
                )
                fx_style.add_watermark(fig_s24h)
                st.plotly_chart(fx_style.apply_theme(fig_s24h), width="stretch")

        st.divider()

    # ---------------------------------------------------------------------------
    # Section 3: DVOL Reaction Around Events
    # ---------------------------------------------------------------------------

    st.subheader("Volatility (DVOL) Reaction Around Events")

    dvol_assets = [a for a in PRICE_ASSETS if PRICE_ASSETS[a].get("has_dvol")]

    if not dvol_assets:
        st.info("No assets with DVOL data available.")
    else:
        dvol_asset = st.radio(
            "DVOL Asset",
            options=dvol_assets,
            horizontal=True,
        )
        dvol_ccy = PRICE_ASSETS[dvol_asset]["deribit_ccy"]

        dvol_reactions = []
        with st.spinner(
            f"Fetching {dvol_asset} DVOL around {len(past_events)} events..."
        ):
            for _, row in past_events.iterrows():
                event_ts_ms = int(row["date"].timestamp() * 1000)
                event_ts_ms += 13 * 3600 * 1000 + 30 * 60 * 1000

                dvol_df = fetch_dvol_around_event(
                    dvol_ccy, event_ts_ms, REACTION_WINDOW_H
                )
                if dvol_df is not None and not dvol_df.empty:
                    dvol_df["event"] = row["event"]
                    dvol_df["event_date"] = row["date"].strftime("%Y-%m-%d")
                    dvol_df["surprise_zscore"] = row["surprise_zscore"]
                    dvol_reactions.append(dvol_df)

        if dvol_reactions:
            all_dvol = pd.concat(dvol_reactions, ignore_index=True)

            fig_dvol = go.Figure()
            for _, row in past_events.iterrows():
                event_key = f"{row['event']} {row['date'].strftime('%Y-%m-%d')}"
                event_slice = all_dvol[
                    all_dvol["event_date"] == row["date"].strftime("%Y-%m-%d")
                ]
                if event_slice.empty:
                    continue

                z = row["surprise_zscore"]
                if pd.notna(z) and abs(z) > 1.0:
                    color = "#ff7f0e"  # big surprise
                else:
                    color = "#7f7f7f"

                fig_dvol.add_trace(go.Scatter(
                    x=event_slice["hours_from_event"],
                    y=event_slice["dvol_change"],
                    name=event_key,
                    line=dict(width=1.5, color=color),
                    opacity=0.7,
                ))

            fig_dvol.add_vline(
                x=0, line_dash="dash", line_color="white", opacity=0.5
            )
            fig_dvol.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)
            fig_dvol.update_layout(
                **PLOTLY_LAYOUT,
                title=f"{dvol_asset} DVOL Change (pts) Around Macro Events",
                xaxis_title="Hours from Event",
                yaxis_title="DVOL Change (pts)",
                legend=dict(
                    orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5
                ),
                height=450,
            )
            fx_style.add_watermark(fig_dvol)
            st.plotly_chart(fx_style.apply_theme(fig_dvol), width="stretch")
            st.caption(
                "Orange = large surprise (|z| > 1.0), Gray = small/no surprise."
            )

            # Average DVOL reaction
            avg_dvol = (
                all_dvol.groupby(all_dvol["hours_from_event"].round(0))["dvol_change"]
                .mean()
                .reset_index()
                .sort_values("hours_from_event")
            )
            fig_avg_dvol = go.Figure()
            fig_avg_dvol.add_trace(go.Scatter(
                x=avg_dvol["hours_from_event"],
                y=avg_dvol["dvol_change"],
                name="Avg DVOL Change",
                line=dict(width=2.5, color=ASSET_COLORS.get(dvol_asset, "#999")),
                fill="tozeroy",
                fillcolor=f"rgba(100,100,100,0.15)",
            ))
            fig_avg_dvol.add_vline(
                x=0, line_dash="dash", line_color="white", opacity=0.5
            )
            fig_avg_dvol.add_hline(
                y=0, line_dash="dot", line_color="gray", opacity=0.3
            )
            fig_avg_dvol.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Average {dvol_asset} DVOL Reaction Across All Events",
                xaxis_title="Hours from Event",
                yaxis_title="Avg DVOL Change (pts)",
                height=350,
            )
            fx_style.add_watermark(fig_avg_dvol)
            st.plotly_chart(fx_style.apply_theme(fig_avg_dvol), width="stretch")
        else:
            st.info(
                "No DVOL data available for the selected events. "
                "DVOL history may not be available for older dates."
            )

# ---------------------------------------------------------------------------
# Section 4: Summary Statistics
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Event Summary Statistics")

# Stats by event type
summary_rows = []
for event_type in filtered_df["event"].unique():
    evt_data = filtered_df[filtered_df["event"] == event_type]
    n_total = len(evt_data)
    n_with_actual = evt_data["actual_num"].notna().sum()
    avg_surprise = evt_data["surprise"].mean()
    std_surprise = evt_data["surprise"].std()
    n_bullish = (evt_data["surprise_zscore"] > 0.5).sum()
    n_bearish = (evt_data["surprise_zscore"] < -0.5).sum()

    summary_rows.append({
        "Event": event_type,
        "Total": n_total,
        "With Data": n_with_actual,
        "Avg Surprise": avg_surprise,
        "Std Surprise": std_surprise,
        "Bullish (z>0.5)": n_bullish,
        "Bearish (z<-0.5)": n_bearish,
    })

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(
        summary_df.style.format(
            {"Avg Surprise": "{:.3f}", "Std Surprise": "{:.3f}"},
            na_rep="—",
        ),
        width="stretch",
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    f"Data source: `{CSV_PATH.relative_to(CSV_PATH.parent.parent)}` | "
    f"Price data: Deribit | "
    f"Events loaded: {len(calendar_df)} | "
    f"Filtered: {len(filtered_df)}"
)
