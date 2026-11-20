"""Home-page "chart blast": pick reports on Home, then send them to Telegram
page by page, in the app's page order.

Pieces:

* A registry (``GROUPS`` / ``ITEMS`` / ``PAGES``) of every report or chart a
  page can send to Telegram, grouped the way the Home selector shows them.
* A plan: the ordered steps (one per page, or one per asset / view on pages
  that show one at a time) for a selection. ``st.session_state["auto_pipeline"]``
  holds the current step id (``"mcm"``, ``"tbrv:BTC"``, ``"ibit:3"`` ...) and
  ``"auto_pipeline_started_at"`` when that step began; ``lib.cache.
  expire_stale_auto_pipeline`` clears a step that has been running for over
  15 minutes. The started-at time is reset at every hand-off, so the timeout
  is per step (a long selection can't time out half way through) while a page
  that hangs still clears.
* Page hooks: ``active_step(page)`` on entry, ``finish(step, sent, failed)``
  when that page has sent its share. ``finish`` moves to the next step
  (``st.switch_page`` / ``st.rerun``) or, after the last one, clears the flag
  and shows the summary banner.

Default selection == what the Home button sent before the selector existed:
every MCM Bot command, the Block Trades — Deribit charts, the Time Based RV
report and the Spot Vol Correlation charts, for BTC and ETH.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

FLAG = "auto_pipeline"
STARTED = "auto_pipeline_started_at"
PLAN = "blast_plan"
SUMMARY = "blast_summary"

ALL_ASSETS = ("BTC", "ETH", "SOL", "HYPE")
DEFAULT_ASSETS = ("BTC", "ETH")

BUTTON_LABEL = "🔄📤 Refresh BTC/ETH & send to Telegram (selected reports, in page order)"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Page:
    id: str
    title: str
    path: str
    assets: tuple | None          # None = not asset-specific
    per_asset: bool = False       # one pass per asset (page shows one asset at a time)
    views: bool = False           # one pass per view (IBIT: only the selected view renders)


@dataclass(frozen=True)
class Item:
    key: str                      # unique id, "<page>.<short>"
    label: str
    page: str                     # page id ("bt" = chart type for every block-trade venue)
    default: bool = False
    sub: str | None = None        # IBIT view index

    @property
    def short(self) -> str:
        return self.key.rsplit(".", 1)[1]


BT_VENUES = (  # (venue key, title, page file) — Deribit first, then the sidebar order
    ("deribit", "Deribit", "pages/02_Block_Trades_-_Deribit.py"),
    ("binance", "Binance", "pages/02_Block_Trades_-_Binance.py"),
    ("bullish", "Bullish", "pages/02_Block_Trades_-_Bullish.py"),
    ("bybit", "Bybit", "pages/02_Block_Trades_-_Bybit.py"),
    ("derive", "Derive", "pages/02_Block_Trades_-_Derive.py"),
    ("okx", "OKX", "pages/02_Block_Trades_-_OKX.py"),
    ("paradex", "Paradex", "pages/02_Block_Trades_-_Paradex.py"),
)

IBIT_VIEWS = ("Deribit vs IBIT", "IBIT surface", "IBIT positioning", "Gamma & max pain",
              "Activity", "Large tickets")

_PAGE_LIST = [
    Page("mcm", "MCM Bot", "pages/01_MCM_Bot.py", ALL_ASSETS),
    *[Page(f"bt_{k}", f"Block Trades — {t}", p, ALL_ASSETS) for k, t, p in BT_VENUES],
    Page("tbrv", "Time Based Realized Vol", "pages/06_Time_Based_Realized_Vol.py", ALL_ASSETS, per_asset=True),
    Page("regime", "Regime Identifier", "pages/07_Regime_Identifier.py", ("BTC", "ETH")),
    Page("spot_vol", "Spot Vol Correlation", "pages/08_Spot_Vol_Correlation.py", ALL_ASSETS),
    Page("macro", "Macro Event Impact", "pages/10_Macro_Event_Impact.py", ("BTC", "ETH")),
    Page("fng", "Fear & Greed Signal", "pages/11_Fear_Greed_Signal.py", ALL_ASSETS, per_asset=True),
    Page("ibit", "IBIT", "pages/12_IBIT.py", None, views=True),
    Page("eco", "Ecoinometrics", "pages/13_Ecoinometrics_Charts.py", None),
]
PAGES = {p.id: p for p in _PAGE_LIST}
PAGE_ORDER = [p.id for p in _PAGE_LIST]


def _items(page: str, default: bool, pairs, sub: str | None = None, prefix: str | None = None):
    pre = prefix or page
    return [Item(f"{pre}.{s}", label, page, default, sub) for s, label in pairs]


# (group id, title, [(sub-group title, [Item, ...]), ...])
GROUPS = [
    ("mcm", "🤖 MCM Bot", [
        ("Vol surface, smile & skew", _items("mcm", True, [
            ("vol_run", "Vol run (table)"),
            ("vol_surfaces", "Vol surface by expiry (vol run charts)"),
            ("vol_term_structure", "Vol term structure"),
            ("skew_term_structure", "Skew term structure (25Δ)"),
            ("skew_term_structure_10d", "Skew term structure (10Δ)"),
            ("butterfly_term_structure", "Butterfly term structure"),
            ("vol_smile", "Vol smile"),
            ("atm_iv_box_plot", "ATM IV box plot"),
        ])),
        ("Forward vol & carry", _items("mcm", True, [
            ("forward_vols", "Forward vols"),
            ("forward_vol_steepness", "Forward vol steepness (ATM)"),
            ("forward_vol_steepness_25d_call", "Forward vol steepness (25Δ call)"),
            ("forward_vol_steepness_25d_put", "Forward vol steepness (25Δ put)"),
            ("forward_vol_steepness_multidelta", "Forward vol steepness (multi-delta)"),
            ("forward_vol_matrix", "Forward vol matrix"),
        ])),
        ("Intraday & history", _items("mcm", True, [
            ("intraday_vol", "Intraday vol"),
            ("intraday_skew", "Intraday skew"),
            ("vol_time_series", "Vol time series"),
            ("skew_time_series", "Skew time series"),
        ])),
        ("Basis, funding & realized vol", _items("mcm", True, [
            ("basis_run", "Basis run (table)"),
            ("intraday_basis", "Intraday basis"),
            ("plot_funding_rates", "Funding rates"),
            ("rv_plot", "Realized vol (RV plot)"),
        ])),
        ("Flows & other", _items("mcm", True, [
            ("block_trades_summary", "Block trades summary (table)"),
            ("moonphase", "Moonphase"),
        ])),
    ]),
    ("bt", "📊 Block Trades", [
        ("Venues", [Item(f"bt.venue.{k}", t, f"bt_{k}", k == "deribit") for k, t, _ in BT_VENUES]),
        ("Charts (sent for each selected venue)", _items("bt", True, [
            ("scatter", "Trades vs spot & DVOL"),
            ("strike_expiry", "Strike vs expiry bubbles"),
            ("net_heatmap", "Net flow heatmap"),
            ("gross_heatmap", "Gross volume heatmap"),
            ("cumulative", "Cumulative flow"),
            ("iv_surface", "IV by tenor"),
            ("term_flow", "Term-structure flow"),
            ("put_call", "Put/call ratio"),
            ("delta_term", "Delta term structure"),
            ("vega_expiry", "Net vega by expiry"),
            ("vega_strike", "Net vega by strike"),
            ("vega_expiry_w", "30d-weighted vega by expiry"),
            ("vega_strike_w", "30d-weighted vega by strike"),
            ("aggression", "Cumulative aggression (not Binance/Bullish/Paradex)"),
        ], prefix="bt.chart")),
    ]),
    ("tbrv", "⏱️ Time Based Realized Vol", [
        ("Overview (all lookbacks)", _items("tbrv", True, [
            ("overview_text", "Overview message"),
            ("cross_heatmap", "Cross-lookback heatmap"),
            ("matrix3d", "3D decision matrix"),
            ("recommendations", "Recommendations message"),
        ])),
        ("Detailed lookback (7d)", _items("tbrv", True, [
            ("detail_text", "Lookback summary message"),
            ("rolling", "Rolling comparison"),
            ("ranking", "Interval ranking"),
            ("distribution", "Rolling distribution"),
            ("metric_heatmap", "Metric heatmap"),
            ("forecast", "Forecast"),
            ("completeness", "Completeness"),
            ("estimator_heatmap", "Estimator heatmap"),
            ("estimator_bars", "Estimator grouped bars"),
        ])),
    ]),
    ("regime", "🧭 Regime Identifier (BTC/ETH)", [
        ("Current regime", _items("regime", False, [
            ("summary", "Summary message"),
            ("cross_asset", "Cross-asset 30d vol (BTC vs ETH)"),
            ("gauge", "Regime gauge"),
            ("days_in_regime", "Days in regime"),
            ("price_regime", "Price & regime"),
            ("vol_ts", "Volatility time series"),
            ("timeline", "Regime timeline"),
            ("breakout_prob", "Breakout probability"),
            ("duration_dist", "Regime duration distribution"),
        ])),
        ("Distributions & statistics", _items("regime", False, [
            ("vol_dist", "Volatility distribution"),
            ("comparative_vol", "Comparative volatility"),
            ("statistics", "Regime statistics"),
            ("return_dist", "Return distribution by regime"),
            ("transition", "Regime transition matrix"),
            ("vol_percentile", "Volatility percentile"),
            ("drawdowns", "Drawdowns by regime"),
            ("expected_move", "Expected move by regime"),
            ("return_skew", "Return skew by regime"),
            ("seasonality", "Seasonality"),
        ])),
        ("Indicators & backtest", _items("regime", False, [
            ("shock", "Shock analysis"),
            ("squeeze", "Volume squeeze"),
            ("garch", "GARCH indicators"),
            ("vol_term", "Volatility term structure"),
            ("compression", "Compression indicators"),
            ("early_warning", "Early warning signals"),
            ("microstructure", "Microstructure indicators"),
            ("rolling_corr", "Rolling correlation"),
            ("autocorr", "Return autocorrelation"),
            ("calibration", "Calibration summary"),
            ("forward_outcome", "Forward outcome"),
        ])),
    ]),
    ("spot_vol", "📈 Spot-Vol Correlation", [
        ("Per asset", _items("spot_vol", True, [
            ("summary", "Summary message"),
            ("cvol_scatter", "CVOL vs spot"),
            ("skew_scatter", "25Δ skew vs spot"),
            ("rr_scatter", "10Δ risk reversal vs spot"),
            ("bf_scatter", "10Δ butterfly vs spot"),
            ("dvol_snapshot", "DVol snapshot (BTC/ETH)"),
            ("rolling_corr", "Rolling correlation"),
            ("rolling_cov", "Rolling covariance"),
            ("rv_iv_basis", "RV-IV basis (BTC/ETH)"),
            ("vol_prediction", "Vol prediction"),
        ])),
        ("ETH–BTC pair (sent once, needs BTC or ETH)", _items("spot_vol", True, [
            ("dvol_spread", "ETH/BTC DVOL spread"),
            ("rv_iv_spread", "ETH-BTC RV-IV basis spread"),
            ("dvol_spread_z", "DVOL spread z-score"),
            ("dvol_beta", "ETH DVOL beta to BTC"),
            ("skew_spread", "ETH-BTC 25Δ skew spread"),
            ("skew_spread_z", "Skew spread z-score"),
        ])),
    ]),
    ("macro", "📅 Macro Event Impact (BTC/ETH)", [
        ("Charts (last 5 events, 1h reaction)", _items("macro", False, [
            ("summary", "Summary message"),
            ("timeline", "Price timeline"),
            ("bloomberg", "Bloomberg-style reaction"),
            ("spider", "Path comparison"),
            ("z_scatter", "Z-score vs actual move"),
            ("actual_implied", "Actual vs implied move"),
            ("dvol_crush", "DVOL crush"),
            ("asymmetry", "Asymmetry"),
            ("distribution", "Distribution"),
            ("move_ratio", "Move ratio over time"),
            ("decision", "Decision vs expectations"),
            ("path_diag", "Path diagnostics"),
        ])),
    ]),
    ("fng", "😨 Fear & Greed Signal", [
        ("Per asset", _items("fng", False, [("summary", "Signal summary message")])),
    ]),
    ("ibit", "🪙 IBIT", [
        ("Deribit vs IBIT", _items("ibit", False, [
            ("smile", "Smile at a constant tenor"),
            ("ts_skew", "Term structure & skew"),
            ("ts_deribit", "Term structure on Deribit expiries"),
            ("surfaces", "Vol surfaces side by side"),
            ("oi_compare", "Open interest comparison"),
            ("density", "Market-implied distribution"),
            ("size", "Size & activity"),
        ], sub="0")),
        ("IBIT surface", _items("ibit", False, [
            ("surface3d", "IBIT implied-vol surface"),
            ("smiles", "Smiles by expiry"),
            ("term", "Term structure, skew & quote quality"),
        ], sub="1")),
        ("IBIT positioning", _items("ibit", False, [
            ("oi_strike", "Open interest by strike / expiry"),
            ("oi_cum", "Cumulative OI distribution"),
        ], sub="2")),
        ("Gamma & max pain", _items("ibit", False, [
            ("gex", "Dealer gamma (GEX) & vega"),
            ("max_pain", "Max pain"),
        ], sub="3")),
        ("Activity", _items("ibit", False, [
            ("most_active", "Most active contracts"),
            ("vol_strike", "Volume by strike"),
        ], sub="4")),
        ("Large tickets", _items("ibit", False, [
            ("oi_change", "Open-interest change"),
            ("unusual", "Unusual activity"),
            ("lean", "Buyer / seller lean"),
            ("flow", "Flow summary"),
        ], sub="5")),
    ]),
    ("eco", "📰 Ecoinometrics", [
        ("Flows & BTC", _items("eco", False, [
            ("etf_streaks", "ETF flow streaks"),
            ("etf_regime", "ETF flow regime"),
            ("etf_30d", "30-day ETF flow"),
            ("btc_200dma", "BTC vs 200DMA (with Nasdaq panel)"),
            ("ratio_200dma", "BTC / 200DMA ratio"),
            ("drawdowns", "Drawdowns"),
            ("hacks", "Exchange hacks"),
            ("strategy", "Strategy funding"),
        ])),
        ("Macro", _items("eco", False, [
            ("fed_tone", "Fed tone"),
            ("nasdaq_corr", "BTC–Nasdaq correlation"),
            ("nasdaq_down", "Nasdaq down days"),
            ("inflation", "Inflation & yields"),
        ])),
    ]),
]

ITEMS: list[Item] = [it for _, _, subs in GROUPS for _, its in subs for it in its]
ITEM_BY_KEY = {it.key: it for it in ITEMS}

# IBIT: the page's section titles (``sec(...)``) that belong to each item.
IBIT_SECTIONS = {
    "smile": ("Smile at a constant tenor",),
    "ts_skew": ("Term structure & skew",),
    "ts_deribit": ("Term structure on Deribit expiries",),
    "surfaces": ("Vol surfaces side by side",),
    "oi_compare": ("Open interest comparison",),
    "density": ("Market-implied distribution",),
    "size": ("Size & activity",),
    "surface3d": ("IBIT implied-vol surface",),
    "smiles": ("Smiles by expiry",),
    "term": ("Term structure",),
    "oi_strike": ("Open interest by strike",),
    "oi_cum": ("Cumulative OI distribution",),
    "gex": ("Dealer gamma exposure (GEX) by strike",),
    "max_pain": ("Max pain",),
    "most_active": ("Most active contracts",),
    "vol_strike": ("Volume by strike",),
    "oi_change": ("Open-interest change",),
    "unusual": ("Unusual activity",),
    "lean": ("Buyer / seller lean (last print)",),
    "flow": ("Flow summary",),
}
IBIT_ITEM_BY_SECTION = {sec: k for k, secs in IBIT_SECTIONS.items() for sec in secs}


def default_items() -> set[str]:
    return {it.key for it in ITEMS if it.default}


def page_title(page_id: str) -> str:
    return PAGES[page_id].title


def items_for_page(items, page_id: str) -> set[str]:
    """Short keys selected for one page. Block-trade venues get the selected
    chart types when the venue itself is ticked."""
    items = set(items)
    if page_id.startswith("bt_"):
        if f"bt.venue.{page_id[3:]}" not in items:
            return set()
        return {ITEM_BY_KEY[k].short for k in items if k in ITEM_BY_KEY and ITEM_BY_KEY[k].page == "bt"}
    return {ITEM_BY_KEY[k].short for k in items if k in ITEM_BY_KEY and ITEM_BY_KEY[k].page == page_id}


def page_assets(page_id: str, assets) -> list[str]:
    meta = PAGES[page_id]
    if meta.assets is None:
        return []
    return [a for a in assets if a in meta.assets]


def build_plan(items, assets) -> tuple[list[str], list[str]]:
    """(steps, skipped notes) for a selection, in page order."""
    steps: list[str] = []
    skipped: list[str] = []
    items = set(items)
    assets = [a for a in ALL_ASSETS if a in set(assets)]
    for pid in PAGE_ORDER:
        sel = items_for_page(items, pid)
        if not sel:
            continue
        meta = PAGES[pid]
        if meta.views:
            subs = sorted({ITEM_BY_KEY[k].sub for k in items
                           if k in ITEM_BY_KEY and ITEM_BY_KEY[k].page == pid}, key=int)
            steps += [f"{pid}:{s}" for s in subs]
        elif meta.assets is None:
            steps.append(pid)
        else:
            eff = page_assets(pid, assets)
            if not eff:
                skipped.append(f"{meta.title}: only supports {'/'.join(meta.assets)}")
                continue
            steps += [f"{pid}:{a}" for a in eff] if meta.per_asset else [pid]
    return steps, skipped


# ---------------------------------------------------------------------------
# Running a plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    id: str
    page: str
    sub: str | None
    index: int
    total: int

    @property
    def asset(self) -> str | None:
        return self.sub if self.sub in ALL_ASSETS else None

    @property
    def label(self) -> str:
        return f"step {self.index + 1}/{self.total}"


# Flags the pages used before the selector (a session mid-run across a deploy).
_LEGACY = {"mcm_bot": "mcm", "block_trades": "bt_deribit", "tbrv_btc": "tbrv:BTC",
           "tbrv_eth": "tbrv:ETH", "spot_vol": "spot_vol"}


def _now():
    return datetime.now(timezone.utc)


def start(items, assets) -> str | None:
    """Store the plan and point the flag at its first step. Returns the first
    page's path (caller switches to it), or None when nothing would run."""
    steps, skipped = build_plan(items, assets)
    st.session_state.pop(SUMMARY, None)
    if not steps:
        return None
    st.session_state[PLAN] = {
        "steps": steps,
        "items": sorted(set(items)),
        "assets": [a for a in ALL_ASSETS if a in set(assets)],
        "skipped": skipped,
        "log": [],
        "started": _now().isoformat(),
    }
    st.session_state[FLAG] = steps[0]
    st.session_state[STARTED] = _now()
    return PAGES[steps[0].split(":")[0]].path


def _plan() -> dict | None:
    flag = st.session_state.get(FLAG)
    if flag is None:
        return None
    plan = st.session_state.get(PLAN)
    if plan and flag in plan.get("steps", []):
        return plan
    legacy = _LEGACY.get(flag)
    if legacy:  # rebuild the old fixed pipeline around the legacy flag
        steps, skipped = build_plan(default_items(), DEFAULT_ASSETS)
        plan = {"steps": steps, "items": sorted(default_items()), "assets": list(DEFAULT_ASSETS),
                "skipped": skipped, "log": [], "started": _now().isoformat()}
        st.session_state[PLAN] = plan
        st.session_state[FLAG] = legacy
        return plan
    # Unknown flag (or the plan was lost): clear rather than get stuck.
    clear()
    return None


def current_step() -> Step | None:
    plan = _plan()
    if plan is None:
        return None
    flag = st.session_state.get(FLAG)
    i = plan["steps"].index(flag)
    page, _, sub = flag.partition(":")
    return Step(flag, page, sub or None, i, len(plan["steps"]))


def running() -> bool:
    return st.session_state.get(FLAG) is not None


def clear() -> None:
    st.session_state[FLAG] = None
    st.session_state[STARTED] = None
    st.session_state.pop(PLAN, None)


def active_step(page_id: str) -> Step | None:
    """The current step when it belongs to ``page_id`` (else None). Clears the
    run when Telegram isn't configured (the Home button is disabled then, but
    never leave the flag stuck)."""
    step = current_step()
    if step is None or step.page != page_id:
        return None
    from lib import telegram
    if not telegram.is_configured():
        clear()
        return None
    return step


def selected(step: Step) -> set[str]:
    plan = st.session_state.get(PLAN) or {}
    return items_for_page(plan.get("items", []), step.page)


def assets_for(step: Step) -> list[str]:
    """Assets this step sends: its own asset on per-asset pages, else the
    selected assets the page supports."""
    if step.asset:
        return [step.asset]
    plan = st.session_state.get(PLAN) or {}
    return page_assets(step.page, plan.get("assets", DEFAULT_ASSETS))


def banner(step: Step, what: str = "") -> None:
    """The blue "Auto pipeline — step i/n" note at the top of a page."""
    title = page_title(step.page)
    extra = f" ({step.asset})" if step.asset else ""
    if step.page == "ibit" and step.sub is not None:
        extra = f" ({IBIT_VIEWS[int(step.sub)]})"
    st.info(f"🔄📤 Auto pipeline — {step.label}: {title}{extra}"
            + (f" — {what}" if what else " — refreshing and sending to Telegram…"))


def finish(step: Step, sent: int, failed=None, note: str | None = None) -> None:
    """Record this step's result and hand off to the next step. After the last
    step: clear the flag and show the summary banner (returns normally)."""
    plan = st.session_state.get(PLAN)
    if plan is None or st.session_state.get(FLAG) != step.id:
        return
    title = page_title(step.page)
    if step.asset:
        title += f" ({step.asset})"
    elif step.page == "ibit" and step.sub is not None:
        title += f" ({IBIT_VIEWS[int(step.sub)]})"
    else:
        a = assets_for(step)
        if a:
            title += f" ({', '.join(a)})"
    plan["log"].append({"step": step.id, "title": title, "sent": int(sent or 0),
                        "failed": [str(f) for f in (failed or [])], "note": note})
    st.session_state[PLAN] = plan
    nxt = step.index + 1
    if nxt < len(plan["steps"]):
        nxt_id = plan["steps"][nxt]
        st.session_state[FLAG] = nxt_id
        st.session_state[STARTED] = _now()   # per-step timeout
        nxt_page = nxt_id.split(":")[0]
        if nxt_page == step.page:
            st.rerun()
        st.switch_page(PAGES[nxt_page].path)
    summary = {"log": plan["log"], "skipped": plan.get("skipped", []),
               "finished": _now().isoformat()}
    clear()
    st.session_state[SUMMARY] = summary
    render_summary(summary)


def abort(page_id: str, reason: str) -> None:
    """A page can't build its reports (fetch failed etc.): record the failure
    and move on. No-op when no step for ``page_id`` is running."""
    step = active_step(page_id)
    if step is not None:
        finish(step, 0, [reason])


def stop(page_id: str, reason: str) -> None:
    """Drop-in for ``st.stop()`` on a page's early-exit paths: a running blast
    moves on to the next page instead of sitting on the flag until timeout."""
    abort(page_id, reason)
    st.stop()


def summary_lines(summary: dict) -> tuple[int, int, list[str]]:
    sent = sum(e["sent"] for e in summary.get("log", []))
    failed = sum(len(e["failed"]) for e in summary.get("log", []))
    lines = []
    for e in summary.get("log", []):
        icon = "⚠️" if e["failed"] else ("✅" if e["sent"] else "➖")
        line = f"{icon} **{e['title']}** — {e['sent']} sent"
        if e["failed"]:
            shown = ", ".join(e["failed"][:8]) + (" …" if len(e["failed"]) > 8 else "")
            line += f", {len(e['failed'])} failed: {shown}"
        if e.get("note"):
            line += f" ({e['note']})"
        lines.append(line)
    for s in summary.get("skipped", []):
        lines.append(f"➖ Skipped {s}")
    return sent, failed, lines


def render_summary(summary: dict | None, heading: str = "Chart blast finished") -> None:
    if not summary:
        return
    sent, failed, lines = summary_lines(summary)
    when = ""
    try:
        from lib import fx_style
        ts = datetime.fromisoformat(summary["finished"]).astimezone(fx_style.local_now().tzinfo)
        when = f" at {ts:%H:%M} {fx_style.DISPLAY_TZ_LABEL}"
    except Exception:
        pass
    head = f"{'⚠️' if failed else '✅'} {heading}{when}: {sent} sent to Telegram" + (
        f", {failed} failed." if failed else ".")
    body = head + "\n\n" + "\n".join(f"- {ln}" for ln in lines)
    (st.warning if failed else st.success)(body)


# ---------------------------------------------------------------------------
# Saved default selection
# ---------------------------------------------------------------------------

def saved_path() -> Path:
    env = os.environ.get("MCM_BLAST_SELECTION_PATH")
    return Path(env) if env else Path(tempfile.gettempdir()) / "mcm_blast_selection.json"


def load_saved() -> tuple[set[str], list[str]] | None:
    try:
        raw = json.loads(saved_path().read_text())
        items = {k for k in raw.get("items", []) if k in ITEM_BY_KEY}
        assets = [a for a in ALL_ASSETS if a in set(raw.get("assets", []))]
        return items, assets or list(DEFAULT_ASSETS)
    except Exception:
        return None


def save(items, assets) -> bool:
    try:
        p = saved_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"items": sorted(set(items) & set(ITEM_BY_KEY)),
                                 "assets": [a for a in ALL_ASSETS if a in set(assets)],
                                 "saved": _now().isoformat()}))
        return True
    except Exception:
        return False


def forget_saved() -> None:
    try:
        saved_path().unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Sending helpers shared by pages without their own send path (IBIT, Eco)
# ---------------------------------------------------------------------------

def send_figs(pairs, light_theme: bool = True) -> tuple[int, list[str]]:
    """Send (fig, caption) pairs as photos. ``light_theme`` exports through
    fx_style.fig_to_png (dark text on white); False renders the figure as
    styled (Ecoinometrics charts carry their own look)."""
    from lib import fx_style, telegram
    sent, failed = 0, []
    for fig, caption in pairs:
        if fig is None:
            continue
        png = None
        try:
            png = fx_style.fig_to_png(fig) if light_theme else fig.to_image(format="png", width=1200, height=800)
        except Exception:
            png = None
        if png is None:
            failed.append(f"{caption} (render)")
        elif telegram.send_photo(png, caption[:1024]):
            sent += 1
        else:
            failed.append(f"{caption} (send)")
    return sent, failed
