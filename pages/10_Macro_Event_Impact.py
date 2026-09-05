"""
Macro Event Impact — how BTC/ETH price and implied vol react to scheduled
US macro releases (CPI, PCE, FOMC, NFP), and whether the reaction was
"priced in" by options ahead of time.

Full port of exodus-analytics's EMCI_vFALC.py (analytics_frontend/streamlit/
pages/EMCI_vFALC.py, 2545 lines) onto this app's Deribit-only data layer —
see lib/macro.py's module docstring for the complete list of adaptation
decisions (DVOL substituted for Amberdata's ATM IV surface, dropped
consensus-basis provenance tracking, dropped Amberdata 1W/1M term structure).
Read in full before porting, per the "verify, don't assume" precedent from
pages 06/07/08 — see CLAUDE.md.

Scoped to **BTC and ETH only** (``lib.macro.MACRO_ASSETS``) — same DVOL
dependency, same reason, as pages 07/08.

Dropped as out-of-scope chart builders (NOT verified-dead in exodus — these
ARE wired into its own UI, unlike the page-08 dead-code elimination, so this
is a deliberate scope-reduction judgment call under time pressure, flagged
to the user rather than silently made): ``_event_spider_chart`` (path
comparison overlay), ``_bloomberg_reaction_chart`` (multi-event average
response band), ``_event_timeline_candles`` (daily candlestick timeline with
event boxes + Wikipedia-sourced event context). All three need either the
Wikipedia-context enrichment exodus used (not reproduced here) or are a
"nice to have" restating information the scatter/bar/asymmetry charts below
already surface more directly. Ask if these are wanted — they can be added
without needing any new adaptation decisions, just more chart code.

Kept, and mapped onto this app's data:
- Surprise Z-Score vs Actual Move scatter, Actual vs Implied Move bar,
  asymmetry (max up/down) bars, move-size distribution histogram, Move Ratio
  over time, Decision-vs-Expectations scatter, Path-Dependency-diagnostics
  scatter — all direct ports of exodus's chart functions, unchanged math.
- exodus's "ATM 1W/1M vol change pre->post" chart is replaced by
  ``lib.macro``'s DVOL-crush-by-horizon bar chart (1h/4h/24h/48h/72h) — the
  same *idea* (how much vol was crushed after the release), simpler because
  DVOL is one index, not a 1W/1M term structure.
- The "Upcoming Event IV vs Historical IV (same days-out)" comparison is
  dropped: it needs a future event on the calendar with a live scheduled
  date, and the bundled ``data/macro_events_calendar.csv`` only has past
  dates (through 2025-08-12) — nothing to compare against right now. The
  math for it (percentile of current IV vs. history at matched days-out)
  would need a live economic calendar feed this app doesn't have; flagged
  as a possible future addition if a forward calendar is ever wired up.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import macro
from lib.constants import PLOTLY_LAYOUT
from lib import fx_style
from lib import cache as cache_lib
from lib.telegram import send_message, send_photo, is_configured

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Macro Event Impact", page_icon="📅", layout="wide")

CSV_PATH = Path(__file__).parent.parent / "data" / "macro_events_calendar.csv"
CHART_HEIGHT = 420
LOOKBACK_OPTIONS = {"Last 10 events": 10, "Last 20 events": 20, "Last 40 events": 40, "All history": 10_000}


def _show(fig: go.Figure | None, key: str) -> None:
    if fig is None:
        return
    fx_style.add_watermark(fig)
    st.plotly_chart(fx_style.apply_theme(fig), width="stretch", key=key)


def _empty_fig(title: str, message: str, height: int = CHART_HEIGHT) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, align="center", font=dict(size=13))
    fig.update_layout(**PLOTLY_LAYOUT, height=height, title=title, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ============================================================================
# CHART BUILDERS — direct ports of exodus's chart functions, ported onto
# lib.macro's column names (see build_impact_table)
# ============================================================================

def chart_scatter_z_vs_actual(df: pd.DataFrame, timeframe: str) -> go.Figure:
    valid = df.dropna(subset=["z_score", "actual_move_pct"])
    if valid.empty:
        return _empty_fig(f"Surprise Z-Score vs Actual Move ({timeframe})", "No z-scored events with a valid price reaction")
    colors = ["#2ecc71" if z > 0 else "#e74c3c" for z in valid["z_score"]]
    custom = np.column_stack([
        valid["date"].astype(str).values, valid["event"].astype(str).values,
        valid["surprise"].values, valid["expectation_bucket"].astype(str).values,
        valid["implied_move_pct"].values, valid["move_ratio"].values,
        valid["max_excursion_pct"].values, valid["first_extreme"].astype(str).values,
    ])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid["z_score"], y=valid["actual_move_pct"], mode="markers",
        marker=dict(size=10, color=colors), name="Events", customdata=custom,
        hovertemplate=("Date %{customdata[0]}<br>Event %{customdata[1]}<br>"
                       "Z-score %{x:.2f}<br>Actual move %{y:.2f}%<br>Surprise %{customdata[2]:.3f}<br>"
                       "Expectation %{customdata[3]}<br>Implied move %{customdata[4]:.2f}%<br>"
                       "Move ratio %{customdata[5]:.2f}<br>Max excursion %{customdata[6]:.2f}%<br>"
                       "First extreme %{customdata[7]}<extra></extra>"),
    ))
    x, y = valid["z_score"].to_numpy(dtype=float), valid["actual_move_pct"].to_numpy(dtype=float)
    if len(x) > 1:
        try:
            ok = np.isfinite(x) & np.isfinite(y)
            if np.sum(ok) > 1 and np.ptp(x[ok]) > 1e-10:
                coef = np.polyfit(x[ok], y[ok], 1)
                x_line = np.linspace(x.min(), x.max(), 50)
                fig.add_trace(go.Scatter(x=x_line, y=np.polyval(coef, x_line), mode="lines", name="OLS",
                                          line=dict(dash="dash", color="gray"),
                                          hovertemplate="OLS fit<br>Z %{x:.2f}<br>Pred move %{y:.2f}%<extra></extra>"))
        except (np.linalg.LinAlgError, ValueError):
            pass
    fig.update_layout(**PLOTLY_LAYOUT, height=CHART_HEIGHT, title=f"Surprise Z-Score vs Actual % Move ({timeframe})",
                       xaxis_title="Surprise Z-Score", yaxis_title="Actual % Price Change", showlegend=True)
    return fig


def chart_bar_actual_vs_implied(df: pd.DataFrame, timeframe: str) -> go.Figure:
    if df.empty:
        return _empty_fig(f"Actual vs Implied Move ({timeframe})", "No events in the selected range")
    d = df.sort_values("date")
    actual_abs = d["actual_move_pct"].abs()
    implied_abs = d["implied_move_pct"].abs()
    comparable = actual_abs.notna() & implied_abs.notna()
    outperform = comparable & (actual_abs > implied_abs)
    actual_colors = np.where(outperform, "#2ecc71", np.where(comparable, "#e74c3c", "#7f8c8d"))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["date"], y=d["implied_move_pct"], name="Implied Move %", marker_color="lightgray",
                          hovertemplate="Date %{x}<br>Implied move %{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=d["date"], y=d["actual_move_pct"], name="Actual Move %", marker_color=actual_colors,
                          hovertemplate="Date %{x}<br>Actual move %{y:.2f}%<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=CHART_HEIGHT, title=f"Actual vs Implied Move ({timeframe})",
                       barmode="group", xaxis_title="Date", yaxis_title="% Move")
    return fig


def chart_dvol_crush(df: pd.DataFrame) -> go.Figure:
    """Mean DVOL crush % at each horizon — replaces exodus's ATM 1W/1M term
    structure chart (see module docstring)."""
    cols = {"1h": "dvol_crush_1h", "4h": "dvol_crush_4h", "24h": "dvol_crush_24h",
            "48h": "dvol_crush_48h", "72h": "dvol_crush_72h"}
    means, counts = [], []
    for c in cols.values():
        v = df[c].dropna() if c in df.columns else pd.Series(dtype=float)
        means.append(float(v.mean()) if len(v) else np.nan)
        counts.append(int(len(v)))
    if all(pd.isna(m) for m in means):
        return _empty_fig("Mean DVOL Crush % by Horizon (vs T−1h)", "No DVOL data for these events (SOL/HYPE have no DVOL index)")
    fig = go.Figure(go.Bar(x=list(cols.keys()), y=means, marker_color="coral", customdata=np.column_stack([counts]),
                            hovertemplate="Horizon %{x}<br>Mean DVOL crush %{y:.2f}%<br>Events %{customdata[0]}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(**PLOTLY_LAYOUT, height=CHART_HEIGHT, title="Mean DVOL Crush % by Horizon (vs T−1h baseline)",
                       xaxis_title="Hours after release", yaxis_title="DVOL Crush % (positive = vol fell)")
    return fig


def chart_asymmetry(df: pd.DataFrame) -> go.Figure:
    valid = df.dropna(subset=["max_up_pct", "max_down_pct"])
    if valid.empty:
        return _empty_fig("Asymmetry: Max Up % vs Max Down %", "No path data available")
    sample = valid.tail(30)
    labels = [f"{d} {str(e)[:12]}" for d, e in zip(sample["date"], sample["event"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=sample["max_up_pct"], name="Max Up %", marker_color="#2ecc71",
                          hovertemplate="%{x}<br>Max up %{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=-sample["max_down_pct"], name="Max Down %", marker_color="#e74c3c",
                          hovertemplate="%{x}<br>Max down %{customdata:.2f}%<extra></extra>", customdata=sample["max_down_pct"]))
    fig.update_layout(**PLOTLY_LAYOUT, height=CHART_HEIGHT, title="Asymmetry: Max Up % vs Max Down % (last 30 events)",
                       barmode="group", xaxis_tickangle=-45)
    return fig


def chart_distribution(df: pd.DataFrame, col: str = "actual_move_pct") -> go.Figure:
    v = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
    if v.empty:
        return _empty_fig(f"Distribution of {col.replace('_', ' ').title()}", "No data available")
    fig = go.Figure(go.Histogram(x=v, nbinsx=25, marker_color="steelblue",
                                  hovertemplate="Bin center %{x:.2f}%<br>Count %{y}<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=CHART_HEIGHT, title=f"Distribution of {col.replace('_', ' ').title()}",
                       xaxis_title="%", yaxis_title="Count")
    return fig


def chart_move_ratio_over_time(df: pd.DataFrame) -> go.Figure:
    valid = df.dropna(subset=["date", "move_ratio"]).sort_values("date")
    if valid.empty:
        return _empty_fig("Move Ratio (Actual/Implied) over Time", "No move-ratio data available")
    fig = go.Figure(go.Scatter(x=valid["date"], y=valid["move_ratio"], mode="markers+lines", name="Move Ratio",
                                text=valid["event"], hovertemplate="Date %{x}<br>Event %{text}<br>Move ratio %{y:.2f}<extra></extra>"))
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    fig.update_layout(**PLOTLY_LAYOUT, height=CHART_HEIGHT, title="Move Ratio (Actual/Implied) over Time",
                       xaxis_title="Date", yaxis_title="Move Ratio")
    return fig


def chart_decision_vs_expectations(df: pd.DataFrame, timeframe: str) -> go.Figure:
    valid = df.dropna(subset=["surprise", "actual_move_pct"])
    if valid.empty:
        return _empty_fig(f"Decision vs Expectations ({timeframe})", "No actual/consensus data available")
    color_map = {"Aligned": "#2ecc71", "Diverged": "#e74c3c", "N/A": "#95a5a6"}
    colors = [color_map.get(v, "#95a5a6") for v in valid["response_alignment"]]
    fig = go.Figure(go.Scatter(
        x=valid["surprise"], y=valid["actual_move_pct"], mode="markers+text", text=valid["date"],
        textposition="top center", marker=dict(size=9, color=colors),
        customdata=np.column_stack([valid["event"].astype(str).values, valid["response_alignment"].astype(str).values]),
        hovertemplate="Date %{text}<br>Event %{customdata[0]}<br>Surprise %{x:.3f}<br>Actual %{y:.2f}%<br>Response %{customdata[1]}<extra></extra>",
        name="Events",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(**PLOTLY_LAYOUT, height=CHART_HEIGHT, title=f"Decision vs Expectations ({timeframe} response)",
                       xaxis_title="Surprise (Actual − Consensus)", yaxis_title="Actual Move (%)")
    return fig


def chart_path_dependency(df: pd.DataFrame, timeframe: str) -> go.Figure:
    valid = df.dropna(subset=["actual_move_pct", "max_excursion_pct"])
    if valid.empty:
        return _empty_fig(f"Path Dependency: Excursion vs Close Move ({timeframe})", "No path diagnostics available")
    color_map = {"Up first": "#2ecc71", "Down first": "#e74c3c", "Same bar": "#f1c40f"}
    colors = [color_map.get(v, "#95a5a6") for v in valid["first_extreme"]]
    fig = go.Figure(go.Scatter(
        x=valid["actual_move_pct"], y=valid["max_excursion_pct"], mode="markers", marker=dict(size=10, color=colors),
        text=valid["date"], customdata=valid["first_extreme"].astype(str).values,
        hovertemplate="Date %{text}<br>Close move %{x:.2f}%<br>Max excursion %{y:.2f}%<br>First extreme %{customdata}<extra></extra>",
        name="Events",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(**PLOTLY_LAYOUT, height=CHART_HEIGHT, title=f"Path Dependency: Excursion vs Close Move ({timeframe})",
                       xaxis_title="Close Move (%)", yaxis_title="Max Excursion (%)")
    return fig


# ============================================================================
# DATA ORCHESTRATION
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def _load_calendar() -> pd.DataFrame | None:
    return macro.load_macro_calendar(CSV_PATH)


def _selected_events(cal: pd.DataFrame, event_types: list[str], n_events: int) -> pd.DataFrame:
    now_utc = pd.Timestamp.now(tz=timezone.utc)
    past = cal[(cal["event"].isin(event_types)) & (cal["release_time_utc"] <= now_utc)]
    return past.sort_values("date", ascending=False).head(n_events).sort_values("date")


@st.cache_data(ttl=300, show_spinner=False)
def build_impact_table_cached(asset: str, event_types: list[str], n_events: int, timeframe: str,
                               window_minutes: int, window_before_h: int, window_after_h: int) -> pd.DataFrame:
    cal = _load_calendar()
    if cal is None or cal.empty:
        return pd.DataFrame()
    scored = macro.compute_surprise_zscores(cal)
    events = _selected_events(scored, event_types, n_events)
    if events.empty:
        return pd.DataFrame()
    return macro.build_impact_table(events, asset, timeframe, window_minutes, window_before_h, window_after_h)


# ============================================================================
# RENDER
# ============================================================================

def render_asset_tab(asset: str, event_types: list[str], n_events: int, timeframe: str) -> pd.DataFrame:
    window_minutes = macro.TIMEFRAME_MINUTES[timeframe]
    window_before_h, window_after_h = 4, max(24, window_minutes // 60 + 4)

    with st.spinner(f"Fetching {asset} event reactions from Deribit…"):
        impact = build_impact_table_cached(asset, event_types, n_events, timeframe, window_minutes, window_before_h, window_after_h)

    if impact.empty:
        st.info("No matching events with fetchable Deribit history for this asset/selection.")
        return impact

    if not impact["actual"].notna().any() or not impact["consensus"].notna().any():
        st.info("Actual/consensus fields are largely empty for these dates, so expectation-alignment "
                "charts will be sparse — the bundled calendar's own data, not a fetch problem.")

    kpis = macro.summary_kpis(impact)
    st.subheader("At-a-glance summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events compared", f"{kpis.get('events', 0)}")
    c2.metric("Avg |Actual Move|", f"{kpis.get('avg_abs_actual', np.nan):.2f}%")
    c3.metric("Avg Implied Move", f"{kpis.get('avg_implied', np.nan):.2f}%")
    c4.metric("Move Ratio > 1", f"{kpis.get('priced_outside_pct', np.nan):.1f}%")
    c5, c6, c7 = st.columns(3)
    c5.metric("Expectation alignment", f"{kpis.get('aligned_pct', np.nan):.1f}%")
    c6.metric("Up-first paths", f"{kpis.get('up_first_pct', np.nan):.1f}%")
    c7.metric("Typical first touch", kpis.get("first_touch", "—"))

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Z-Score vs Actual Move: right/up = positive surprise with positive response. "
                   "Points far from the OLS line are outlier reactions.")
        _show(chart_scatter_z_vs_actual(impact, timeframe), key=f"{asset}_scatter_z")
    with col2:
        st.caption("Actual vs Implied Move: green bars beat the implied move (from DVOL); "
                   "red bars fell short of it.")
        _show(chart_bar_actual_vs_implied(impact, timeframe), key=f"{asset}_bar_ai")

    st.subheader("DVOL Crush")
    st.caption("Mean % change in Deribit's DVOL index at each horizon after release, vs. a T−1h baseline. "
               "Positive = vol crush (typical post-event pattern); negative = vol expansion.")
    _show(chart_dvol_crush(impact), key=f"{asset}_dvol_crush")

    with st.expander("Asymmetry & distributions", expanded=False):
        st.markdown(
            "**How to read this section** — Asymmetry bars compare upside vs downside excursion per "
            "event. Distribution shows typical move size and tail risk. Move ratio over time (>1 means "
            "realized exceeded implied) shows whether the market has been persistently under/over-pricing "
            "these releases."
        )
        a1, a2 = st.columns(2)
        with a1:
            _show(chart_asymmetry(impact), key=f"{asset}_asym")
        with a2:
            _show(chart_distribution(impact), key=f"{asset}_dist")
        _show(chart_move_ratio_over_time(impact), key=f"{asset}_moveratio")

    with st.expander("Decision vs expectations & path diagnostics", expanded=True):
        st.markdown(
            "**How to read this section** — Decision vs expectations compares the macro surprise "
            "(Actual − Consensus) to the realized move; same-sign quadrants are an *aligned* reaction, "
            "opposite-sign quadrants are *diverged* (positioning/liquidity dominated). Path diagnostics "
            "compares the endpoint move to the max excursion — high excursion with a small close move is a "
            "noisy/mean-reverting path."
        )
        d1, d2 = st.columns(2)
        with d1:
            _show(chart_decision_vs_expectations(impact, timeframe), key=f"{asset}_decision")
        with d2:
            _show(chart_path_dependency(impact, timeframe), key=f"{asset}_pathdiag")

        exp_summary = macro.expectations_summary_table(impact)
        if not exp_summary.empty:
            st.markdown("**Expectations response summary**")
            st.dataframe(exp_summary.round(3), hide_index=True, width="stretch")
        else:
            st.info("No actual/consensus values available for this selection yet.")

    st.subheader("Impact table")
    st.caption("`move_ratio > 1`: realized move exceeded the DVOL-implied move. `response_alignment`: "
               "whether the move direction matched the surprise direction. `first_extreme`/`time_to_high/"
               "low_min`: compact path-dependency diagnostics.")
    display_cols = ["date", "event", "actual", "consensus", "surprise", "expectation_bucket", "response_alignment",
                     "z_score", "dvol_before", "implied_move_pct", "actual_move_pct", "move_ratio",
                     "dvol_crush_1h", "dvol_crush_4h", "dvol_crush_24h", "dvol_crush_48h", "dvol_crush_72h",
                     "max_up_pct", "max_down_pct", "range_pct", "pre_event_drift_pct", "first_extreme",
                     "time_to_high_min", "time_to_low_min", "realized_vol_window_pct"]
    available = [c for c in display_cols if c in impact.columns]
    table_df = impact[available].copy()
    numeric_cols = table_df.select_dtypes(include="number").columns
    table_df[numeric_cols] = table_df[numeric_cols].round(3)

    def _highlight_mr(val):
        if pd.isna(val):
            return ""
        return "background-color: rgba(231, 76, 60, 0.25)" if val > 1 else ""

    if "move_ratio" in table_df.columns:
        st.dataframe(table_df.style.map(_highlight_mr, subset=["move_ratio"]), hide_index=True, width="stretch", height=420)
    else:
        st.dataframe(table_df, hide_index=True, width="stretch", height=420)

    return impact


def _send_chart(fig: go.Figure | None, caption: str) -> bool:
    """Same images-only pattern established in pages/01/06/07/08 — a render
    failure is reported, never silently swapped for a text dump."""
    if fig is None:
        return False
    img = fx_style.fig_to_png(fig)
    if img is None:
        img = fx_style.fig_to_png(fig)  # one retry — kaleido occasionally misfires cold
    if img is None:
        return False
    return send_photo(img, caption=caption[:1024])


def send_asset_report_to_telegram(asset: str, event_types: list[str], n_events: int, timeframe: str) -> tuple[int, list[str]]:
    window_minutes = macro.TIMEFRAME_MINUTES[timeframe]
    window_before_h, window_after_h = 4, max(24, window_minutes // 60 + 4)
    impact = build_impact_table_cached(asset, event_types, n_events, timeframe, window_minutes, window_before_h, window_after_h)
    if impact.empty:
        return 0, [f"{asset} (no data)"]

    kpis = macro.summary_kpis(impact)
    send_message(
        f"📅 <b>Macro Event Impact – {asset}</b>\n\n"
        f"Events compared: {kpis.get('events', 0)}\n"
        f"Avg |Actual Move| ({timeframe}): {kpis.get('avg_abs_actual', float('nan')):.2f}%\n"
        f"Avg Implied Move: {kpis.get('avg_implied', float('nan')):.2f}%\n"
        f"Expectation alignment: {kpis.get('aligned_pct', float('nan')):.1f}%"
    )
    charts = [
        (chart_scatter_z_vs_actual(impact, timeframe), f"{asset} - Z-Score vs Actual Move"),
        (chart_bar_actual_vs_implied(impact, timeframe), f"{asset} - Actual vs Implied Move"),
        (chart_dvol_crush(impact), f"{asset} - DVOL Crush"),
        (chart_asymmetry(impact), f"{asset} - Asymmetry"),
        (chart_distribution(impact), f"{asset} - Distribution"),
        (chart_move_ratio_over_time(impact), f"{asset} - Move Ratio Over Time"),
        (chart_decision_vs_expectations(impact, timeframe), f"{asset} - Decision vs Expectations"),
        (chart_path_dependency(impact, timeframe), f"{asset} - Path Diagnostics"),
    ]
    sent, failed = 0, []
    for fig, name in charts:
        if _send_chart(fig, name):
            sent += 1
        else:
            failed.append(name)
    return sent, failed


def send_all_reports_to_telegram(event_types: list[str], n_events: int, timeframe: str) -> tuple[int, list[str]]:
    total_sent, total_failed = 0, []
    for asset in macro.MACRO_ASSETS:
        sent, failed = send_asset_report_to_telegram(asset, event_types, n_events, timeframe)
        total_sent += sent
        total_failed.extend(failed)
    return total_sent, total_failed


# ============================================================================
# MAIN
# ============================================================================

ASSET_NAMES = {"BTC": "₿ Bitcoin (BTC)", "ETH": "⟠ Ethereum (ETH)"}


def main() -> None:
    st.title("📅 Macro Event Impact")

    with st.expander("📖 How to Use This Dashboard", expanded=False):
        st.markdown("""
        Explores how **BTC/ETH price and implied vol** react to scheduled US macro releases
        (CPI, PCE, FOMC, NFP — from the bundled `data/macro_events_calendar.csv`), and
        whether the reaction was already "priced in" ahead of time. Scoped to **BTC/ETH
        only** (both have a Deribit DVOL index — SOL/HYPE don't).

        - **Surprise Z-score** — `(actual − consensus) / historical std of that event
          type's surprises` (expanding-window, so early history isn't scored against a
          full-sample std it couldn't have known at the time).
        - **Implied move** — DVOL (annualized) scaled down to the selected reaction
          window: `daily_vol = DVOL/√365`, `implied_move = daily_vol·√(window/1440)·100`.
        - **Move ratio** — actual move ÷ implied move. >1 means the release moved price
          further than options had priced in.
        - **DVOL Crush** — % change in the DVOL index at 1h/4h/24h/48h/72h after release
          vs. a T−1h baseline — the vol-crush that typically follows a scheduled event.
        - **Path diagnostics** — which extreme (high/low) came first and how long each
          took, plus the larger of the two excursions vs. the close-to-close move.

        Release times are recovered via a fixed per-event-type ET lookup (DST-aware,
        `lib/macro.py:RELEASE_TIME_ET`) since the bundled calendar has no `time` column —
        FOMC decisions are pinned to 14:00 ET, everything else to the standard 8:30am ET
        slot most US macro prints share.
        """)

    st.markdown("---")

    cal = _load_calendar()
    if cal is None or cal.empty:
        st.error(f"❌ Could not load the macro events calendar from `{CSV_PATH.name}`.")
        return

    event_types_all = sorted(cal["event"].unique().tolist())

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        event_types = st.multiselect("Event types", event_types_all, default=event_types_all)
    with col2:
        lookback_label = st.selectbox("History depth", list(LOOKBACK_OPTIONS.keys()), index=1)
        n_events = LOOKBACK_OPTIONS[lookback_label]
    with col3:
        timeframe = st.selectbox("Reaction timeframe", macro.TIMEFRAME_OPTIONS, index=macro.TIMEFRAME_OPTIONS.index("1h"))

    if not event_types:
        st.warning("Select at least one event type.")
        return

    st.caption(f"📅 Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    st.markdown("---")

    tabs = st.tabs([ASSET_NAMES.get(a, a) for a in macro.MACRO_ASSETS])
    for tab, asset in zip(tabs, macro.MACRO_ASSETS):
        with tab:
            render_asset_tab(asset, event_types, n_events, timeframe)

    now_utc = pd.Timestamp.now(tz=timezone.utc)
    future = cal[(cal["event"].isin(event_types)) & (cal["release_time_utc"] > now_utc)]
    if not future.empty:
        st.markdown("---")
        st.subheader("Upcoming events")
        st.dataframe(future[["date", "event", "release_time_utc"]].head(20), hide_index=True, width="stretch")

    with st.sidebar:
        cache_lib.render_refresh_button(help="Clear cache and refetch calendar/price/DVOL data from Deribit.")

        st.markdown("---")
        st.subheader("📱 Telegram Reports")
        st.caption("Send Macro Event Impact reports to Telegram")
        if not is_configured():
            st.warning("Telegram not configured — set bot_token and chat_id (see lib/telegram.py).")
        else:
            if st.button("📤 Send All Reports to Telegram", width="stretch", type="primary", key="macro_telegram_all"):
                with st.spinner("Generating and sending all reports to Telegram..."):
                    sent, failed = send_all_reports_to_telegram(event_types, n_events, timeframe)
                if failed:
                    st.warning(f"Sent {sent} chart(s). Failed: {', '.join(failed)}")
                else:
                    st.success(f"Sent {sent} chart(s) to Telegram.")


main()
