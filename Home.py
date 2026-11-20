"""
MCM Analytics — Home Page
Crypto derivatives analytics powered by Deribit public API data.
"""

import sys
from pathlib import Path

# Ensure lib/ is importable
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from datetime import datetime, timezone

from lib.deribit import get_index_price, get_option_chain, get_ticker
from lib.constants import ASSET_CONFIG, ASSETS
from lib.telegram import is_configured
from lib import cache as cache_lib
from lib import commands as cmdreg
from lib import blast, freshness

freshness.reload_if_changed(blast)

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
_pipeline_running = blast.running()

# --- Report selector -------------------------------------------------------
# The canonical selection lives in non-widget keys (``blast_sel_items`` /
# ``blast_sel_assets``) because Streamlit drops a widget's state when you
# leave the page; the checkboxes are re-seeded from it on every run. A fresh
# session starts from the saved default (💾 below) or, if none, from exactly
# what this button sent before the selector existed.
_SEL, _SEL_A = "blast_sel_items", "blast_sel_assets"
if _SEL not in st.session_state:
    _saved = blast.load_saved()
    st.session_state[_SEL] = set(_saved[0]) if _saved else blast.default_items()
    st.session_state[_SEL_A] = list(_saved[1]) if _saved else list(blast.DEFAULT_ASSETS)


def _cb_key(item_key: str) -> str:
    return f"blast_cb::{item_key}"


def _asset_key(asset: str) -> str:
    return f"blast_asset::{asset}"


def _apply(items=None, assets=None) -> None:
    """Set the selection (and the widgets showing it) from a button callback."""
    if items is not None:
        st.session_state[_SEL] = set(items)
        for it in blast.ITEMS:
            st.session_state[_cb_key(it.key)] = it.key in st.session_state[_SEL]
    if assets is not None:
        st.session_state[_SEL_A] = list(assets)
        for a in blast.ALL_ASSETS:
            st.session_state[_asset_key(a)] = a in assets


def _set_group(keys, on: bool) -> None:
    cur = set(st.session_state[_SEL])
    _apply(cur | set(keys) if on else cur - set(keys))


def _save_default() -> None:
    ok = blast.save(st.session_state[_SEL], st.session_state[_SEL_A])
    st.session_state["blast_saved_msg"] = ("💾 Saved — new sessions (including the scheduled runs) start "
                                           "from this selection." if ok else "Couldn't save the selection.")


def _reset_default() -> None:
    blast.forget_saved()
    _apply(blast.default_items(), blast.DEFAULT_ASSETS)
    st.session_state["blast_saved_msg"] = "↩️ Back to the standard pipeline selection (saved default removed)."


for _it in blast.ITEMS:
    st.session_state.setdefault(_cb_key(_it.key), _it.key in st.session_state[_SEL])
for _a in blast.ALL_ASSETS:
    st.session_state.setdefault(_asset_key(_a), _a in st.session_state[_SEL_A])

st.markdown("**Reports to send** — tick what the button below sends. Pages run in app order "
            "(MCM Bot → Block Trades → Time Based RV → Regime → Spot-Vol → Macro → Fear & Greed → "
            "IBIT → Ecoinometrics); pages with nothing ticked are skipped.")
_ac = st.columns([1, 1, 1, 1, 4])
for _col, _a in zip(_ac, blast.ALL_ASSETS):
    _col.checkbox(_a, key=_asset_key(_a), disabled=_pipeline_running)
_ac[4].caption("Assets (pages that don't support an asset skip it; Regime and Macro are BTC/ETH only; "
               "IBIT and Ecoinometrics aren't asset-specific).")

_bc = st.columns(4)
_bc[0].button("☑️ Select all", key="blast_all", width="stretch", disabled=_pipeline_running,
              on_click=_apply, kwargs={"items": [i.key for i in blast.ITEMS]})
_bc[1].button("⬜ Clear all", key="blast_none", width="stretch", disabled=_pipeline_running,
              on_click=_apply, kwargs={"items": []})
_bc[2].button("↩️ Standard pipeline", key="blast_reset", width="stretch", disabled=_pipeline_running,
              on_click=_reset_default,
              help="MCM Bot (all commands), Block Trades — Deribit, Time Based RV and Spot-Vol "
                   "Correlation for BTC/ETH: what this button always sent. Also removes a saved default.")
_bc[3].button("💾 Save as default", key="blast_save", width="stretch", disabled=_pipeline_running,
              on_click=_save_default,
              help="New sessions (e.g. the 7 AM / 7 PM scheduled click) start from this selection. "
                   "Kept until the app restarts or redeploys, then it's back to the standard pipeline.")
if st.session_state.get("blast_saved_msg"):
    st.caption(st.session_state.pop("blast_saved_msg"))

for _gid, _gtitle, _subs in blast.GROUPS:
    _gkeys = [it.key for _, its in _subs for it in its]
    with st.expander(_gtitle):
        _gc = st.columns([1, 1, 4])
        _gc[0].button("Select all", key=f"blast_gall_{_gid}", width="stretch", disabled=_pipeline_running,
                      on_click=_set_group, args=(_gkeys, True))
        _gc[1].button("Clear", key=f"blast_gnone_{_gid}", width="stretch", disabled=_pipeline_running,
                      on_click=_set_group, args=(_gkeys, False))
        for _stitle, _its in _subs:
            _sc = st.columns([5, 1, 1])
            _sc[0].markdown(f"*{_stitle}*")
            _skeys = [it.key for it in _its]
            _sc[1].button("All", key=f"blast_sall_{_gid}_{_stitle}", width="stretch",
                          disabled=_pipeline_running, on_click=_set_group, args=(_skeys, True))
            _sc[2].button("None", key=f"blast_snone_{_gid}_{_stitle}", width="stretch",
                          disabled=_pipeline_running, on_click=_set_group, args=(_skeys, False))
            _cols = st.columns(3)
            for _i, _it in enumerate(_its):
                _cols[_i % 3].checkbox(_it.label, key=_cb_key(_it.key), disabled=_pipeline_running)

# Widgets -> canonical selection.
st.session_state[_SEL] = {it.key for it in blast.ITEMS if st.session_state.get(_cb_key(it.key))}
st.session_state[_SEL_A] = [a for a in blast.ALL_ASSETS if st.session_state.get(_asset_key(a))]
_items, _assets = st.session_state[_SEL], st.session_state[_SEL_A]
_steps, _skipped = blast.build_plan(_items, _assets)

_counts = []
for _gid, _gtitle, _subs in blast.GROUPS:
    _gk = [it.key for _, its in _subs for it in its]
    _n = sum(k in _items for k in _gk)
    if _n:
        _counts.append(f"{_gtitle.split(' ', 1)[1]} {_n}/{len(_gk)}")
st.caption("Selected: " + (" · ".join(_counts) if _counts else "nothing")
           + f" — assets: {', '.join(_assets) or 'none'} — {len(_steps)} page step(s)."
           + (f" Skipped: {'; '.join(_skipped)}." if _skipped else ""))

if st.button(
    blast.BUTTON_LABEL,
    key="blast_go",
    width="stretch",
    type="primary",
    disabled=not _tg_ready or _pipeline_running or not _steps,
    help="Clears cached data on each page it visits, re-runs the selected reports and sends them "
         "to Telegram one page at a time, in app order (by default: every MCM Bot command, the "
         "Block Trades — Deribit charts, the Time Based Realized Vol report, then Spot Vol "
         "Correlation, for BTC and ETH). Takes several minutes — you'll land on each page as its "
         "step runs, and the last page shows a summary of what was sent and what failed.",
):
    _first = blast.start(_items, _assets)
    if _first:
        st.switch_page(_first)

if not _tg_ready:
    st.caption("Configure Telegram (see secrets.toml.example) to enable this.")
elif _pipeline_running:
    st.caption("Auto pipeline running — Home button disabled until it finishes "
               "or a step times out (15 min).")
elif not _steps:
    st.caption("Nothing to send — tick at least one report and an asset it supports "
               "(Block Trades needs a venue and at least one chart).")

if not _pipeline_running and st.session_state.get(blast.SUMMARY):
    blast.render_summary(st.session_state[blast.SUMMARY], heading="Last chart blast")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Pages")

page_info = [
    ("01 MCM Bot", f"Full markets bot: {len(cmdreg.COMMAND_NAMES)} commands — vol/skew term structure, forward vols, carry, basis, flow, RV"),
    ("02 Block Trades — Binance", "Binance Options blockTrades tape (BTC/ETH/SOL/BNB + others), USDT-settled"),
    ("02 Block Trades — Bullish", "Bullish BTC option tape (no block flag); every print at/above a minimum size"),
    ("02 Block Trades — Bybit", "Bybit isBlockTrade option prints; public feed only reaches back the latest 1,000 trades per coin"),
    ("02 Block Trades — Deribit", "Deribit block trade analysis with Greeks and Telegram reporting"),
    ("02 Block Trades — Derive", "Derive RFQ block flow (BTC/ETH/SOL/HYPE)"),
    ("02 Block Trades — OKX", "OKX RFQ block tape (BTC/ETH/SOL/XAU), coin-margined prices converted to USD"),
    ("02 Block Trades — Paradex", "Paradex BLOCK_TRADE tape (BTC/ETH/SOL/HYPE)"),
    ("06 Time Based Realized Vol", "RV across hedging frequencies + lookbacks (BTC/ETH/SOL/HYPE perps), 7 estimators, decision matrix"),
    ("07 Regime Identifier", "Vol regime classification (GARCH + implied vol)"),
    ("08 Spot-Vol Correlation", "DVOL vs spot analysis (BTC/ETH; SOL/HYPE on RV-scaled vol)"),
    ("10 Macro Event Impact", "CPI/FOMC/NFP surprise z-scores + price reactions"),
    ("11 Fear & Greed Signal", "Contrarian delta-lean backtest vs alternative.me Fear & Greed Index"),
    ("12 IBIT", "IBIT (iShares Bitcoin ETF) options vs Deribit BTC: vol surface, skew, OI, gamma, implied distribution (Cboe delayed chain)"),
]

grid_cols = st.columns(3)
for i, (name, desc) in enumerate(page_info):
    with grid_cols[i % 3]:
        st.markdown(f"**{name}**")
        st.caption(desc)


# ---------------------------------------------------------------------------
# Cross-exchange option block flow (Deribit / Derive / Paradex / OKX / Bybit / Bullish / Binance)
# ---------------------------------------------------------------------------
# Placed last so everything above renders before the (slower) seven-venue fetch.

st.divider()
st.subheader("Option Block Flow — Deribit vs Derive vs Paradex vs OKX vs Bybit vs Bullish vs Binance")

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

with st.spinner("Loading block flow from Deribit, Derive, Paradex, OKX, Bybit, Bullish and Binance..."):
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
        "present); Paradex = trade_type BLOCK_TRADE; OKX = RFQ block tape (rfq/public-trades); Bybit = "
        "isBlockTrade prints (its feed only reaches back the latest 1,000 trades per coin); Bullish = no block flag, "
        "so every print at/above its 0.0005 BTC minimum order; Binance = blockTrades tape (side assumed to be the taker's). "
        "Venue minimum sizes are each exchange's minimum order. Premium is USD (coin-margined Deribit and OKX BTC/ETH converted at "
        "each trade's index). Call %, Taker-buy % are premium-weighted; Net delta/vega are dollar Greeks at "
        "execution, signed from the taker's side (buy +, sell -), using each trade's own IV. "
        "Greeks cover = share of premium with an index price to compute them."
    )
