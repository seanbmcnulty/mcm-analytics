"""
MCM Analytics — Home Page
Crypto derivatives analytics powered by Deribit public API data.
"""

import sys
from pathlib import Path

# Ensure lib/ is importable
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import time
from datetime import datetime, timezone

from lib.deribit import get_index_price, get_option_chain, get_ticker
from lib.constants import ASSET_CONFIG, ASSETS, ASSET_COLORS
from lib.telegram import is_configured
from lib import cache as cache_lib
from lib import commands as cmdreg

st.set_page_config(
    page_title="MCM Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 MCM Analytics")
st.caption("Crypto derivatives analytics • Deribit public API • No authentication required")

# ---------------------------------------------------------------------------
# Market Overview
# ---------------------------------------------------------------------------

st.subheader("Market Overview")

cols = st.columns(len(ASSETS))
for i, asset in enumerate(ASSETS):
    cfg = ASSET_CONFIG[asset]
    with cols[i]:
        price = get_index_price(cfg["index"])
        if price:
            # Get perp data for funding + OI
            ticker = get_ticker(cfg["perp"])
            funding = ticker.get("current_funding", 0) if ticker else 0
            funding_pct = funding * 100 if funding else 0

            st.metric(
                label=f"{asset}",
                value=f"${price:,.{cfg['price_dp']}f}",
                delta=f"Funding: {funding_pct:+.4f}%/8h" if ticker else None,
            )
        else:
            st.metric(label=asset, value="—")

# ---------------------------------------------------------------------------
# Options Activity Summary
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Options Activity")

option_cols = st.columns(4)
headers = ["Currency", "Total OI (contracts)", "24h Volume", "Active Instruments"]
for col, h in zip(option_cols, headers):
    col.markdown(f"**{h}**")

for asset in ASSETS:
    cfg = ASSET_CONFIG[asset]
    chain = get_option_chain(cfg["deribit_ccy"], "option")
    if chain:
        # Filter to this asset's prefix
        prefix = cfg["deribit_prefix"]
        asset_chain = [c for c in chain if c["instrument_name"].startswith(prefix)]

        total_oi = sum(c.get("open_interest", 0) for c in asset_chain)
        total_vol = sum(c.get("volume", 0) for c in asset_chain)
        n_instruments = len(asset_chain)

        row_cols = st.columns(4)
        row_cols[0].markdown(f"**{asset}**")
        row_cols[1].markdown(f"{total_oi:,.0f}")
        row_cols[2].markdown(f"{total_vol:,.0f}")
        row_cols[3].markdown(f"{n_instruments}")

# ---------------------------------------------------------------------------
# System Status
# ---------------------------------------------------------------------------

st.divider()
st.subheader("System Status")

status_cols = st.columns(3)
with status_cols[0]:
    st.markdown("**Telegram**")
    if is_configured():
        st.success("✅ Configured")
    else:
        st.warning("⚠️ Not configured — see secrets.toml.example")

with status_cols[1]:
    st.markdown("**Deribit API**")
    # Quick connectivity check
    test = get_index_price("btc_usd")
    if test:
        st.success("✅ Connected")
    else:
        st.error("❌ Unreachable")

with status_cols[2]:
    st.markdown("**Assets**")
    st.info(f"📈 {', '.join(ASSETS)} (Deribit public data)")

# ---------------------------------------------------------------------------
# Quick Actions
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Quick Actions")

st.session_state.setdefault("auto_pipeline", None)
st.session_state.setdefault("auto_pipeline_started_at", None)

if cache_lib.expire_stale_auto_pipeline():
    st.caption("Auto pipeline timed out after 15 minutes and was cleared — "
               "you can start it again.")

_tg_ready = is_configured()
_pipeline_running = st.session_state.get("auto_pipeline") is not None
if st.button(
    "🔄📤 Refresh BTC/ETH & send to Telegram (MCM Bot → Block Trades → Time Based RV → Spot Vol Correlation)",
    width="stretch",
    type="primary",
    disabled=not _tg_ready or _pipeline_running,
    help="Clears cached data and re-runs every MCM Bot command for BTC and "
         "ETH, sends those reports to Telegram, then does the same for the "
         "Block Trades — Deribit page's BTC and ETH charts, then the Time "
         "Based Realized Vol page's BTC and ETH reports, then Spot Vol "
         "Correlation BTC/ETH Telegram reports as the final step. Takes "
         "several minutes — you'll land on each page as its step runs.",
):
    st.session_state["auto_pipeline"] = "mcm_bot"
    st.session_state["auto_pipeline_started_at"] = datetime.now(timezone.utc)
    st.switch_page("pages/01_MCM_Bot.py")

if not _tg_ready:
    st.caption("Configure Telegram (see secrets.toml.example) to enable this.")
elif _pipeline_running:
    st.caption("Auto pipeline running — Home button disabled until it finishes "
               "or times out (15 min).")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Pages")

page_info = [
    ("01 MCM Bot", f"Full markets bot: {len(cmdreg.COMMAND_NAMES)} commands — vol/skew term structure, forward vols, carry, basis, flow, RV"),
    ("02 Block Trades", "Deribit block trade analysis with Greeks and Telegram reporting"),
    ("06 Time Based Realized Vol", "RV across hedging frequencies + lookbacks (BTC/ETH/SOL/HYPE perps), 7 estimators, decision matrix"),
    ("07 Regime Identifier", "Vol regime classification (GARCH + implied vol)"),
    ("08 Spot-Vol Correlation", "DVOL vs spot analysis (BTC/ETH; SOL/HYPE on RV-scaled vol)"),
    ("10 Macro Event Impact", "CPI/FOMC/NFP surprise z-scores + price reactions"),
    ("11 Fear & Greed Signal", "Contrarian delta-lean backtest vs alternative.me Fear & Greed Index"),
]

grid_cols = st.columns(3)
for i, (name, desc) in enumerate(page_info):
    with grid_cols[i % 3]:
        st.markdown(f"**{name}**")
        st.caption(desc)


# ---------------------------------------------------------------------------
# Cross-exchange option block flow (Deribit / Derive / Paradex)
# ---------------------------------------------------------------------------
# Placed last so everything above renders before the (slower) three-venue fetch.

st.divider()
st.subheader("Option Block Flow — Deribit vs Derive vs Paradex")

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go

from lib import flow_summary
from lib.constants import TTL_MEDIUM


@st.cache_data(ttl=TTL_MEDIUM, show_spinner=False)
def _flow_payloads(start_ms: int):
    return flow_summary.collect(start_ms)


_win = st.radio("Window", ["Last 12 Hours", "Last 24 Hours"], index=1, horizontal=True,
                key="home_flow_window")
_hours = 12 if _win == "Last 12 Hours" else 24
_start_ms = int((datetime.now(timezone.utc) - timedelta(hours=_hours))
                .replace(second=0, microsecond=0).timestamp() * 1000)

with st.spinner("Loading block flow from Deribit, Derive and Paradex..."):
    _by_venue, _by_asset, _notes = flow_summary.summarize(_flow_payloads(_start_ms))

if _by_venue.empty:
    st.info("No block flow found in this window.")
else:
    _money = lambda x: f"${x:,.0f}" if pd.notna(x) else "—"  # noqa: E731
    _pct = lambda x: f"{x:.0%}" if pd.notna(x) else "—"  # noqa: E731
    _fmt = {"Gross premium $": _money, "Net delta $": _money, "Net vega $": _money,
            "Call %": _pct, "Taker-buy %": _pct, "Greeks cover": _pct, "Largest $": _money,
            "Blocks": "{:,}", "Underlyings": "{:,}"}

    st.dataframe(_by_venue.style.format({k: v for k, v in _fmt.items() if k in _by_venue.columns}),
                 width="stretch", hide_index=True)

    _c1, _c2 = st.columns(2)
    for _col, _title, _col_name, _a, _b in (
            (_c1, "Calls vs Puts (% of premium)", "Call %", "Calls", "Puts"),
            (_c2, "Taker buys vs sells (% of premium)", "Taker-buy %", "Buys", "Sells")):
        _fig = go.Figure()
        _fig.add_bar(y=_by_venue["Exchange"], x=_by_venue[_col_name], orientation="h", name=_a,
                     marker_color="#2E8B57", text=_by_venue[_col_name].map(_pct), textposition="inside")
        _fig.add_bar(y=_by_venue["Exchange"], x=1 - _by_venue[_col_name], orientation="h", name=_b,
                     marker_color="#C0392B", text=(1 - _by_venue[_col_name]).map(_pct), textposition="inside")
        _fig.update_layout(barmode="stack", title=_title, height=240, margin=dict(l=10, r=10, t=40, b=10),
                           xaxis=dict(tickformat=".0%", range=[0, 1]), yaxis=dict(autorange="reversed"),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))
        _col.plotly_chart(_fig, width="stretch", key=f"home_flow_{_col_name}")

    with st.expander("By underlying"):
        st.dataframe(_by_asset.style.format({k: v for k, v in _fmt.items() if k in _by_asset.columns}),
                     width="stretch", hide_index=True)

    for _v, _n in _notes.items():
        st.warning(f"{_v}: {_n}")
    st.caption(
        "Block definitions differ by venue: Deribit = trades at/above the per-asset minimum size used on the "
        "Block Trades — Deribit page (its public feed has no block flag); Derive = RFQ fills (quote_id/rfq_id "
        "present); Paradex = trade_type BLOCK_TRADE. Premium is USD (coin-margined Deribit BTC/ETH converted at "
        "each trade's index). Call %, Taker-buy % are premium-weighted; Net delta/vega are dollar Greeks at "
        "execution, signed from the taker's side (buy +, sell -), using each trade's own IV. "
        "Greeks cover = share of premium with an index price to compute them."
    )
