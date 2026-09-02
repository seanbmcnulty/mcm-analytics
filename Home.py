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
from lib.deribit import get_index_price, get_option_chain, get_ticker
from lib.constants import ASSET_CONFIG, ASSETS, ASSET_COLORS
from lib.telegram import is_configured

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

_tg_ready = is_configured()
_pipeline_running = st.session_state.get("auto_pipeline") is not None
if st.button(
    "🔄📤 Refresh BTC/ETH & send to Telegram (MCM Bot, then Block Trades)",
    width="stretch",
    type="primary",
    disabled=not _tg_ready or _pipeline_running,
    help="Clears cached data and re-runs every MCM Bot command for BTC and "
         "ETH, sends those reports to Telegram, then does the same for the "
         "Block Trades — Deribit page's BTC and ETH charts. Takes a few "
         "minutes — you'll land on each page as its step runs.",
):
    st.session_state["auto_pipeline"] = "mcm_bot"
    st.switch_page("pages/01_MCM_Bot.py")

if not _tg_ready:
    st.caption("Configure Telegram (see secrets.toml.example) to enable this.")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Pages")

page_info = [
    ("01 MCM Bot", "Full markets bot: 21 commands — vol/skew term structure, forward vols, carry, basis, flow, RV"),
    ("02 Block Trades", "Deribit block trade analysis with Greeks and Telegram reporting"),
    ("06 Time Based Realized Vol", "RV across hedging frequencies + lookbacks (BTC/ETH perps), 7 estimators, decision matrix"),
    ("07 Regime Identifier", "Vol regime classification (GARCH + implied vol)"),
    ("08 Spot-Vol Correlation", "DVOL vs spot analysis (BTC/ETH)"),
    ("10 Macro Event Impact", "CPI/FOMC/NFP surprise z-scores + price reactions"),
    ("11 Fear & Greed Signal", "Contrarian delta-lean backtest vs alternative.me Fear & Greed Index"),
]

grid_cols = st.columns(3)
for i, (name, desc) in enumerate(page_info):
    with grid_cols[i % 3]:
        st.markdown(f"**{name}**")
        st.caption(desc)
