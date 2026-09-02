"""
MCM Bot — the full markets bot, Deribit public data only.

Layout: a compact toolbar (expiry + load/refresh/send), then one tab per
asset for the report grid plus a final "Run single command" tab — so
switching between assets (or to the ad-hoc single-command tool) is a tab
click instead of a long scroll. "What each report shows" lives in the
sidebar as reference material.
"""

import sys
import html
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from lib import cache as cache_lib
from lib import commands as cmdreg
from lib import fx_style, history, surface, telegram
from lib.constants import ASSETS

st.set_page_config(page_title="MCM Bot", page_icon="🤖", layout="wide",
                   initial_sidebar_state="expanded")

DASHBOARD_CACHE_TTL = 300        # seconds before results are called stale
DASHBOARD_CACHE_MAX_AGE = 1800   # seconds before results are dropped entirely

_TODAY = datetime.now(timezone.utc).date()

# Fixed display order: every asset, each command in COMMANDS order.
MCM_ORDER = [(a, cn) for a in ASSETS for cn in cmdreg.COMMAND_NAMES]

# These render wide tables (not charts) that get cut off when squeezed into a
# 1/3-width grid column, so each gets a dedicated full-width row instead of
# sharing a row of 3 with neighboring commands — same treatment vol_run's
# table already gets (see the dashboard grid loop below).
FULL_ROW_COMMANDS = {"basis_run", "block_trades_summary"}

for _k, _v in (("mcm_all_results", None), ("mcm_all_results_ts", None),
               ("mcm_fig", None), ("mcm_df", None), ("mcm_text", None)):
    st.session_state.setdefault(_k, _v)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stretch(**kw):
    """use_container_width was deprecated in favour of width='stretch'."""
    return dict(width="stretch", **kw)


def _finite(v) -> bool:
    return surface.finite(v)


def _expiry_options() -> tuple[list[int], bool]:
    try:
        dtes = surface.listed_expiries(ASSETS[0])
    except Exception:
        dtes = []
    if not dtes:
        return [7, 30, 90], True
    return dtes, False


def _expiry_label(dte: int) -> str:
    try:
        return f"{dte} DTE → {(_TODAY + timedelta(days=int(dte))):%d-%b-%Y}"
    except Exception:
        return f"{dte} days"


def _table_height(n_rows: int) -> int:
    return 40 + 35 * max(0, n_rows)


def _style_table(df: pd.DataFrame):
    """Number formatting and colouring, matching the FalconX tables."""
    disp = df.copy()
    cols = set(disp.columns)

    basis_col = ("Basis % (1Y APR)" if "Basis % (1Y APR)" in cols
                 else ("Basis %" if "Basis %" in cols else None))
    if basis_col and "Expiry" in cols:
        fmt = {basis_col: "{:+.3f}%", "Basis Low %": "{:+.3f}%",
               "Basis High %": "{:+.3f}%",
               "P": lambda x: f"P{int(x)}" if _finite(x) else "-",
               "Basis $": lambda x: f"${x:,.0f}" if pd.notna(x) else "",
               "Mid Price": lambda x: f"${float(x):,.2f}" if _finite(x) else "",
               "DTE": "{:.0f}"}
        style = disp.style.format({k: v for k, v in fmt.items() if k in cols},
                                  na_rep="-")
        style = style.apply(
            lambda s: s.apply(
                lambda v: "background-color: rgba(200,230,200,0.5)"
                if pd.notna(v) and v >= 0 else
                ("color: #c00;" if pd.notna(v) and v < 0 else "")),
            subset=[basis_col])
        return style

    if "ATM σ%" in cols and "RV%" in cols:
        fmt = {"DTE": "{:.0f}", "ATM σ%": "{:.1f}", "Daily Move %": "{:.2f}%",
               "BE Move %": "{:.2f}%", "ATM 3h": "{:+.1f}", "ATM Open": "{:+.1f}",
               "RV%": "{:.1f}", "IV−RV": "{:+.1f}", "25d σ Call": "{:.1f}",
               "25d σ Put": "{:.1f}", "Fwd IV": "{:.1f}"}
        style = disp.style.format({k: v for k, v in fmt.items() if k in cols},
                                  na_rep="-")
        style = style.apply(
            lambda row: ["background-color: rgba(0,0,0,0.04);"
                         if disp.index.get_loc(row.name) % 2 else ""] * len(row),
            axis=1)
        for c in ("ATM 3h", "ATM Open", "IV−RV"):
            if c in cols:
                style = style.apply(
                    lambda s: s.apply(
                        lambda v: "color: #c00;" if pd.notna(v) and _finite(v) and v < 0
                        else ("color: #0a0;" if pd.notna(v) and _finite(v) and v > 0
                              else "")), subset=[c])
        return style

    for c in ("Delta", "Vega", "Gamma (1%)"):
        if c in cols and pd.api.types.is_numeric_dtype(disp[c]):
            disp[c] = disp[c].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "-")
    for c in ("Net Puts", "Net Calls", "Gross Notional"):
        if c in cols and pd.api.types.is_numeric_dtype(disp[c]):
            disp[c] = disp[c].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "-")
    return disp


def render_output(fig, df, text, key_prefix: str = "out",
                  full_width_charts: bool = False, skip_figures: bool = False):
    """Render one command result: header text, table, then chart(s)."""
    figs = [] if skip_figures else (
        fig if isinstance(fig, list) else ([fig] if fig is not None else []))
    has_table = df is not None and not getattr(df, "empty", True)

    if text and has_table:
        st.markdown(text)

    if has_table:
        try:
            st.dataframe(_style_table(df), hide_index=True,
                         height=_table_height(len(df)), **_stretch())
        except Exception:
            st.dataframe(df, hide_index=True, **_stretch())

    valid = [(i, f) for i, f in enumerate(figs) if f is not None]
    if full_width_charts and valid:
        for start in range(0, len(valid), 3):
            chunk = valid[start:start + 3]
            cols = st.columns(3)
            for j, (i, f) in enumerate(chunk):
                with cols[j]:
                    st.plotly_chart(fx_style.apply_theme(f),
                                    key=f"plotly_{key_prefix}_{i}", **_stretch())
    else:
        for i, f in valid:
            st.plotly_chart(fx_style.apply_theme(f),
                            key=f"plotly_{key_prefix}_{i}", **_stretch())

    if text and not figs and not has_table:
        st.markdown(text)


_SPAN_TAG_RE = re.compile(r"</?span[^>]*>", re.IGNORECASE)
_CAPTION_SAFE_LEN = 700  # generous for a chart title; keeps well under
                          # Telegram's 1024-char caption cap even after
                          # HTML-escaping expands some characters


def _caption_from_title(title_text: str | None, fallback: str) -> str:
    """Turn a Plotly figure's title.text into a safe Telegram HTML-mode
    caption.

    Plotly titles carry Plotly's own light markup (<br> for a line
    break, <span style=...> for the smaller reconstruction-note line
    that fx_style.finalize() adds) — but note text itself can contain
    literal characters that aren't markup at all, e.g.
    forward_vol_steepness's note reads "excludes <=3DTE". Sent verbatim
    under Telegram's parse_mode=HTML, that "<=" combined with a real
    closing </span> tag later in the string reads as one huge malformed
    tag to Telegram's parser, which rejects the whole message with a 400
    ("can't parse entities") — every single time, deterministically, for
    any chart whose note/title contains that pattern. That's why
    forward_vol_steepness and its 25d_call/25d_put variants failed to
    send 100% of the time while spacing/rate-limit fixes did nothing for
    them: it was never a rate limit.

    Fix: convert <br> to a newline, strip Plotly's <span> tags (Telegram
    doesn't support arbitrary style attributes on <span> anyway), then
    html.escape() whatever's left so any remaining literal '<'/'&'/'>'
    renders as itself instead of being parsed as markup. This does mean
    a title's own bold/italic styling (none currently used in title text)
    would render as escaped text rather than formatting — an acceptable
    trade for "always delivers."
    """
    if not title_text:
        return fallback
    text = (title_text.replace("<br>", "\n")
                       .replace("<br/>", "\n")
                       .replace("<br />", "\n"))
    text = _SPAN_TAG_RE.sub("", text).strip()
    if len(text) > _CAPTION_SAFE_LEN:
        text = text[:_CAPTION_SAFE_LEN].rstrip() + "…"
    text = html.escape(text)
    return text or fallback


def _send_photo(png: bytes | None, caption: str) -> bool:
    """Send one PNG to Telegram. telegram.send_photo already retries a
    429 (rate limit) with Telegram's own retry_after, and separately
    downgrades to a plain-text caption if HTML parsing is ever still
    rejected — this wrapper just guards against a missing image."""
    if not png:
        return False
    return telegram.send_photo(png, caption)


def send_result_to_telegram(asset: str, cmd_name: str, fig, df, text) -> tuple[bool, list[str]]:
    """Send one command's result to Telegram as image(s) only: a table
    image for the dataframe (if any), one photo per chart (if any).

    This never dumps a table or chart as raw text — if image rendering
    fails, that piece just isn't sent. The only text message ever sent is
    the command's own status text, and only when there's no table or
    chart to render in the first place (e.g. "No basis data available.").

    Returns (ok, reasons): ok is True if at least one piece went out;
    reasons lists exactly which piece(s) failed and at which stage
    (render vs. send) — the render/send distinction matters because
    they point at different root causes (kaleido vs. Telegram itself),
    and a flat "failed" count doesn't tell you which one to chase.
    """
    ok = False
    reasons: list[str] = []
    has_table = df is not None and not getattr(df, "empty", True)
    figs = fig if isinstance(fig, list) else ([fig] if fig is not None else [])
    figs = [f for f in figs if f is not None]
    caption_base = f"{asset} /{cmd_name}"

    if has_table:
        png = fx_style.dataframe_to_table_image(df, header_text=text or caption_base)
        if png is None:
            png = fx_style.dataframe_to_table_image(df, header_text=text or caption_base)
        if png is None:
            reasons.append(f"{caption_base}: table image failed to **render**")
        elif _send_photo(png, f"<b>{caption_base}</b>"):
            ok = True
        else:
            reasons.append(f"{caption_base}: table image failed to **send**")

    for idx, f in enumerate(figs, start=1):
        title_obj = getattr(getattr(f, "layout", None), "title", None)
        title_text = getattr(title_obj, "text", None) if title_obj else None
        caption = _caption_from_title(title_text, caption_base)
        label = f"{caption_base} chart {idx}/{len(figs)}"

        png = fx_style.fig_to_png(fx_style.apply_theme(f, "light"), width=1200, height=800)
        if png is None:
            png = fx_style.fig_to_png(fx_style.apply_theme(f, "light"), width=1200, height=800)
        if png is None:
            reasons.append(f"{label}: failed to **render**")
        elif _send_photo(png, caption):
            ok = True
        else:
            reasons.append(f"{label}: failed to **send**")

    if not has_table and not figs and text:
        # text here is a command's own status/error string (can carry an
        # exception message verbatim, e.g. cmd_block_trades_summary's
        # "Failed to fetch trades: {exc}") — escape it too, same reason
        # as chart captions above.
        if telegram.send_message(f"<b>{caption_base}</b>\n\n{html.escape(str(text))}"):
            ok = True
        else:
            reasons.append(f"{caption_base}: status text failed to **send**")

    return ok, reasons


def _telegram_queue(assets: list[str], results: dict) -> list[tuple]:
    """Ordered (asset, cmd_name, fig, df, text) tuples to send for `assets`,
    in dashboard order. vol_run's table is queued under its own command
    name; its per-expiry vol-surface charts are queued right after
    moonphase, matching where they render in the dashboard."""
    order = [(a, cn) for a in assets for cn in cmdreg.COMMAND_NAMES]
    to_send = []
    for (a, cn) in order:
        if (a, cn) not in results:
            continue
        fig, df, text = results[(a, cn)]
        to_send.append((a, cn, None, df, text) if cn == "vol_run"
                       else (a, cn, fig, df, text))
        if cn == "moonphase" and (a, "vol_run") in results:
            f_list, _, _ = results[(a, "vol_run")]
            if isinstance(f_list, list):
                for i, f in enumerate(f_list):
                    if f is not None:
                        to_send.append((a, f"vol_surface_{i}", f, None, None))
    return to_send


def _send_to_telegram(assets: list[str], label: str) -> None:
    """Ensure every command is loaded for `assets` (running only what's
    missing — a single-asset button doesn't force-load the other 3), then
    send them all to Telegram as images. Shared by the per-asset buttons
    and the "All" button."""
    results = st.session_state.mcm_all_results or {}
    missing = [a for a in assets
              if any((a, cn) not in results for cn in cmdreg.COMMAND_NAMES)]
    if missing:
        with st.spinner(f"Running commands for {', '.join(missing)}…"):
            fresh = run_reports(missing, list(cmdreg.COMMAND_NAMES),
                                st.session_state.get("mcm_dte_days", 30))
            results = {**results, **fresh}
            st.session_state.mcm_all_results = results
            st.session_state.mcm_all_results_ts = datetime.now(timezone.utc)

    to_send = _telegram_queue(assets, results)
    sent = failed = 0
    all_reasons: list[str] = []
    with st.spinner(f"Sending {label} to Telegram…"):
        for (a, cn, fig, df, text) in to_send:
            ok, reasons = send_result_to_telegram(a, cn, fig, df, text)
            if ok:
                sent += 1
            else:
                failed += 1
            all_reasons.extend(reasons)
    if sent:
        st.success(f"Sent {sent} report(s) to Telegram."
                   + (f" ({failed} failed to render/send — see detail below.)"
                      if failed else ""))
    else:
        st.error("Failed to send reports to Telegram.")

    if all_reasons:
        n_render = sum(1 for r in all_reasons if "render" in r)
        n_send = sum(1 for r in all_reasons if "send" in r)
        with st.expander(f"⚠️ {len(all_reasons)} image(s)/message(s) had "
                         f"trouble — {n_render} failed to render, {n_send} "
                         "failed to send. Click for detail."):
            st.caption(
                "**Render** failures happen before Telegram is ever "
                "contacted (the chart/table image itself couldn't be "
                "built — usually kaleido). **Send** failures mean the "
                "image rendered fine but Telegram rejected or didn't "
                "receive it (rate limiting, network, timeout) even after "
                "the built-in retries.")
            for r in all_reasons:
                st.markdown(f"- {r}")


def run_reports(assets: list[str], cmd_names: list[str], target_days: int) -> dict:
    results = {}
    total = max(1, len(assets) * len(cmd_names))
    bar = st.progress(0.0)
    done = 0
    for a in assets:
        for cn in cmd_names:
            results[(a, cn)] = cmdreg.run_command(
                a, "/" + cn,
                expiry_target_days=target_days if cn in cmdreg.EXPIRY_COMMANDS else None)
            done += 1
            bar.progress(done / total)
    bar.empty()
    return results


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🤖 MCM Bot")
    st.caption("Deribit public API · no authentication required")
    st.radio("Chart theme", ["Auto", "Light", "Dark"], index=0,
             key="mcm_chart_theme", horizontal=True,
             help="Auto follows the app theme (☰ → Settings → Appearance). "
                  "Charts sent to Telegram are always exported light.")
    st.divider()
    if telegram.is_configured():
        st.success("Telegram ready")
    else:
        st.warning("Telegram not configured")
        st.caption(telegram.config_status())
    st.divider()
    if st.button("Clear data cache", **_stretch()):
        cache_lib.clear_all_caches()
        st.rerun()
    st.divider()
    with st.expander("What each report shows"):
        st.markdown("""
| Command | What it shows |
|--------|----------------|
| **Vol run** | Implied volatility by expiry (ATM σ, 3h/open change, RV, IV−RV, 25Δ wings, forward IV). Green/red = up/down. |
| **Vol term structure** | ATM vol across expiries; current vs 24h / 1w / 1m ago. |
| **Skew term structure** | 25Δ call minus put skew by expiry (call–put vol spread). |
| **Butterfly term structure** | 10Δ butterfly by expiry: average of 10Δ call/put vol minus ATM vol. |
| **Basis run** | Perp and dated futures basis vs index. Basis % (1Y APR) = annualized; green/red = contango/backwardation. |
| **Forward vols** | Combined ATM, 25d call, 25d put vols and their forward vols by expiry date. |
| **Forward vol steepness** | ATM-only $100k vega daily carry waterfall by adjacent tenor pair (excludes <=3DTE; weighted to 30D vega). |
| **Forward vol steepness 25d call / put** | Same waterfall for the 25d call or 25d put wing. |
| **Forward vol steepness multidelta** | Grouped bar chart of carry (pts/30d) for ATM vs 25d call vs 25d put by tenor pair. |
| **ATM IV box plot** | 90-day ATM IV distribution per tenor with today's curve overlaid; P-number = current percentile. |
| **Forward vol matrix** | Forward vol between every pair of expiries, as a heatmap. |
| **Intraday basis / vol / skew** | Last 24 hours for the selected expiry; perp price on the secondary axis. |
| **Vol time series** | Last month of ATM and 25Δ/10Δ vol for the selected expiry (Singapore time). |
| **Skew time series** | 25Δ and 10Δ skew (call − put) over the last month. |
| **Vol smile** | Implied vol across deltas for current, 1d ago, 1w ago, 1m ago. |
| **Funding rates** | Annualized Deribit perpetual funding APR (last 30 days). |
| **RV plot** | Realized volatility (Parkinson) over 3d, 7d, 30d, 90d. |
| **Block trades summary** | Last 24h large option trades by expiry: **Delta, Vega, Gamma** = net $ exposure (green = net long, red = net short). Net Puts/Calls = contract flow (+ buy, − sell). |
| **Moonphase** | Perp price with lunar phases; bands show full/new moon windows. |
""")

for _asset in ASSETS:
    try:
        history.record_snapshot(_asset)
    except Exception:
        pass

st.title("🤖 MCM Bot")
st.caption("Crypto derivatives analytics · Deribit public API · "
           f"{fx_style.local_now():%d-%b-%Y %H:%M} "
           f"{fx_style.DISPLAY_TZ_LABEL} time")

# ---------------------------------------------------------------------------
# Auto pipeline (triggered from the Home page's "Refresh BTC/ETH & send to
# Telegram" button): refresh BTC+ETH here and send to Telegram, then hand
# off to the Block Trades page for its own BTC+ETH step. Single-shot: the
# flag is advanced to "block_trades" (or cleared) before this block can
# ever run twice.
# ---------------------------------------------------------------------------
if st.session_state.get("auto_pipeline") == "mcm_bot":
    if telegram.is_configured():
        _auto_assets = [a for a in ASSETS if a in ("BTC", "ETH")]
        st.info(f"🔄📤 Auto pipeline — step 1/2: refreshing MCM Bot data for "
                f"{', '.join(_auto_assets)} and sending to Telegram…")
        with st.spinner(f"Refreshing MCM Bot commands for {', '.join(_auto_assets)}…"):
            cache_lib.clear_all_caches()
            _fresh = run_reports(_auto_assets, list(cmdreg.COMMAND_NAMES),
                                 st.session_state.get("mcm_dte_days", 30))
            st.session_state.mcm_all_results = {
                **(st.session_state.mcm_all_results or {}), **_fresh}
            st.session_state.mcm_all_results_ts = datetime.now(timezone.utc)
        _send_to_telegram(_auto_assets, "BTC+ETH")
        st.session_state["auto_pipeline"] = "block_trades"
        st.switch_page("pages/02_Block_Trades_-_Deribit.py")
    else:
        # Shouldn't happen — the Home page button is disabled when Telegram
        # isn't configured — but clear the flag rather than getting stuck.
        st.session_state["auto_pipeline"] = None

# ---------------------------------------------------------------------------
# Toolbar — expiry + load/refresh/send, one compact row instead of three
# stacked sections. Everything here used to take ~40 lines of subheaders and
# always-visible captions before a single chart appeared; the explanatory
# text now lives in tooltips (hover ⓘ). The old "Load selected…" popover
# (pick assets + commands from two multiselects, then click through) was
# replaced by a "Run <asset>" button inside each asset's own tab — one click
# for whichever asset you're already looking at, instead of a menu.
# ---------------------------------------------------------------------------

_expiries, _using_fallback = _expiry_options()
_default_idx = _expiries.index(min(_expiries, key=lambda d: abs(d - 30)))

_tb1, _tb2, _tb3 = st.columns([2.6, 1.2, 1.2])
with _tb1:
    st.selectbox(
        "Default expiry", options=_expiries, index=_default_idx,
        format_func=_expiry_label, key="mcm_dte_days",
        help="DTE = days to expiry. Used for: Vol Time Series, Skew Time Series, "
             "Intraday Basis/Vol/Skew, and Vol Smile. Only Deribit-listed option "
             "expiries are shown." + (
                 " Deribit's expiry list is unavailable right now, so 7/30/90d "
                 "are shown instead." if _using_fallback else ""))
with _tb2:
    st.write("")
    load_all_btn = st.button("Load all", key="mcm_load_all", **_stretch(),
                             help=f"Run all {len(cmdreg.COMMAND_NAMES)} commands for "
                                  f"{', '.join(ASSETS)}. May take several minutes. "
                                  "To load just one asset, use that asset's own "
                                  "tab below instead.")
with _tb3:
    st.write("")
    refresh_all_btn = st.button("Refresh all", key="mcm_refresh_all", **_stretch(),
                                help="Re-run all commands for every asset and "
                                     "replace the dashboard with the latest data.")

# Send to Telegram — one button per asset, a BTC+ETH combo, plus "All",
# instead of a single all-or-nothing send. A per-asset (or combo) button
# only loads whatever's missing rather than forcing a full 4-asset run.
_sb_label, *_sb_asset_cols, _sb_pair, _sb_all = st.columns(
    [1.5] + [0.8] * len(ASSETS) + [1.0, 0.9])
with _sb_label:
    st.write("")
    st.caption("Send to Telegram (as images):")
send_asset_clicked = {}
for _a, _col in zip(ASSETS, _sb_asset_cols):
    with _col:
        send_asset_clicked[_a] = st.button(
            _a, key=f"mcm_send_{_a}", **_stretch(),
            disabled=not telegram.is_configured(),
            help=f"Send {_a}'s reports to Telegram as images. Loads {_a}'s "
                 "commands first if they aren't cached yet.")
_BTC_ETH = [a for a in ASSETS if a in ("BTC", "ETH")]
with _sb_pair:
    send_btc_eth_btn = st.button(
        "BTC+ETH", key="mcm_send_btc_eth", **_stretch(),
        disabled=not telegram.is_configured() or not _BTC_ETH,
        help="Send BTC and ETH's reports to Telegram as images. Loads "
             "whichever of the two aren't cached yet first.")
with _sb_all:
    send_all_btn = st.button(
        "All", key="mcm_send_all", **_stretch(),
        disabled=not telegram.is_configured(),
        help="Send every asset's reports to Telegram as images, in "
             "dashboard order. Loads anything not already cached first.")

if load_all_btn or refresh_all_btn:
    with st.spinner(f"Running all commands for {', '.join(ASSETS)}…"):
        st.session_state.mcm_all_results = run_reports(
            list(ASSETS), list(cmdreg.COMMAND_NAMES),
            st.session_state.get("mcm_dte_days", 30))
        st.session_state.mcm_all_results_ts = datetime.now(timezone.utc)
    st.success("All reports loaded." if load_all_btn else "All reports refreshed.")
    st.rerun()

for _a in ASSETS:
    if send_asset_clicked.get(_a):
        _send_to_telegram([_a], _a)

if send_btc_eth_btn and _BTC_ETH:
    _send_to_telegram(_BTC_ETH, "BTC+ETH")

if send_all_btn:
    _send_to_telegram(list(ASSETS), "all assets")

# Drop very old results rather than showing stale data indefinitely.
_ts = st.session_state.get("mcm_all_results_ts")
if _ts is not None and st.session_state.mcm_all_results is not None:
    if (datetime.now(timezone.utc) - _ts).total_seconds() > DASHBOARD_CACHE_MAX_AGE:
        st.session_state.mcm_all_results = None
        st.session_state.mcm_all_results_ts = None

if st.session_state.mcm_all_results is None:
    st.session_state.mcm_all_results = {}

results = st.session_state.mcm_all_results or {}
_ts = st.session_state.get("mcm_all_results_ts")

# ---------------------------------------------------------------------------
# Dashboard — one tab per asset, plus a tab for the single-command tool.
# Replaces the old layout of "every asset's full grid, stacked vertically,
# then the single-command panel below all of them" — which meant scrolling
# past BTC's ~21 charts to see ETH's, then SOL's, then HYPE's, then the
# single-command tool at the very bottom. Now switching between them is a
# tab click, not a scroll. ("What each report shows" moved to the sidebar.)
# ---------------------------------------------------------------------------

st.divider()

if _ts is not None and results:
    age_sec = (datetime.now(timezone.utc) - _ts).total_seconds()
    age_min = int(age_sec // 60)
    if age_sec < DASHBOARD_CACHE_TTL:
        st.caption(f"Last updated {age_min} min ago. Data is cached for "
                   f"{DASHBOARD_CACHE_TTL // 60} min — use **Refresh all** for "
                   "latest. Cache persists across page navigations.")
    else:
        st.info(f"Data is {age_min} min old. Click **Refresh all** above for "
                "the latest.")
elif not results:
    st.info("No reports loaded yet. Use **Load all** above, or open an "
            "asset's tab below and click its **Run <asset>** button to "
            "load just that one.")

_tab_labels = [f"📊 {a}" for a in ASSETS] + ["🔎 Run single command"]
asset_tabs = st.tabs(_tab_labels)

for ia, a in enumerate(ASSETS):
    with asset_tabs[ia]:
        _run_a_col, _ = st.columns([1, 4])
        with _run_a_col:
            run_asset_btn = st.button(
                f"Run {a}", key=f"mcm_run_{a}", **_stretch(),
                help=f"Run all {len(cmdreg.COMMAND_NAMES)} commands for {a} "
                     "only, and replace just this tab's data with the "
                     "latest — quicker than Load all/Refresh all when you "
                     "only need this asset.")
        if run_asset_btn:
            with st.spinner(f"Running all commands for {a}…"):
                if st.session_state.mcm_all_results is None:
                    st.session_state.mcm_all_results = {}
                st.session_state.mcm_all_results.update(
                    run_reports([a], list(cmdreg.COMMAND_NAMES),
                                st.session_state.get("mcm_dte_days", 30)))
                st.session_state.mcm_all_results_ts = datetime.now(timezone.utc)
            st.rerun()

        order_a = [(x, cn) for x, cn in MCM_ORDER if x == a]
        n_a = len(order_a)
        i = 0
        while i < n_a:
            cn0 = order_a[i][1]
            if cn0 in FULL_ROW_COMMANDS:
                st.caption(f"**{a}** /{cn0}")
                if (a, cn0) in results:
                    f2, d2, t2 = results[(a, cn0)]
                    render_output(f2, d2, t2, key_prefix=f"{a}_{cn0}")
                else:
                    st.caption(f"Not loaded. Use **Load all** above, or "
                               f"**Run {a}** at the top of this tab.")
                i += 1
                if i < n_a:
                    st.write("")
                continue

            # Next row: up to 3 items, stopping before a full-row command so
            # that command starts its own row instead of being pulled in.
            row_items = []
            j = i
            while j < n_a and len(row_items) < 3 and order_a[j][1] not in FULL_ROW_COMMANDS:
                row_items.append(order_a[j])
                j += 1
            i = j

            multi_idx = None
            for k, (_, cn) in enumerate(row_items):
                entry = results.get((a, cn))
                if entry and isinstance(entry[0], list) and entry[0]:
                    multi_idx = k
                    break

            if multi_idx is not None:
                _, cn = row_items[multi_idx]
                fig, df, text = results[(a, cn)]
                st.caption(f"**{a}** /{cn}")
                render_output(fig, df, text, key_prefix=f"{a}_{cn}", skip_figures=True)
                others = [row_items[k] for k in range(len(row_items)) if k != multi_idx]
                if others:
                    cols = st.columns(len(others))
                    for k, (_, cn2) in enumerate(others):
                        with cols[k]:
                            st.caption(f"**{a}** /{cn2}")
                            if (a, cn2) in results:
                                f2, d2, t2 = results[(a, cn2)]
                                render_output(f2, d2, t2, key_prefix=f"{a}_{cn2}")
                            else:
                                st.caption(
                                    f"Not loaded. Use **Load all** above, or "
                                    f"**Run {a}** at the top of this tab.")
            else:
                cols = st.columns(len(row_items))
                for k, (_, cn) in enumerate(row_items):
                    with cols[k]:
                        st.caption(f"**{a}** /{cn}")
                        if (a, cn) in results:
                            f2, d2, t2 = results[(a, cn)]
                            render_output(f2, d2, t2, key_prefix=f"{a}_{cn}",
                                          skip_figures=(cn == "vol_run"
                                                        and isinstance(f2, list)))
                        else:
                            st.caption(
                                f"Not loaded. Use **Load all** above, or "
                                f"**Run {a}** at the top of this tab.")
            if i < n_a:
                st.write("")

        if (a, "vol_run") in results:
            f_list, _, _ = results[(a, "vol_run")]
            if isinstance(f_list, list) and f_list:
                st.caption(f"**{a}** Vol surface by expiry")
                render_output(f_list, None, None,
                              key_prefix=f"{a}_vol_surface", full_width_charts=True)

# ---------------------------------------------------------------------------
# Run single command
# ---------------------------------------------------------------------------

with asset_tabs[-1]:
    st.caption("Run one command for one asset and view or send the result. Useful "
               "for quick checks or sending a single chart/table to Telegram.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        asset = st.selectbox("Asset", ASSETS, key="mcm_asset",
                             help=", ".join(ASSETS) + ".")
    with col2:
        command_input = st.selectbox("Command", cmdreg.COMMANDS, key="mcm_cmd",
                                     help="Choose which report to run.")
    with col3:
        show_help = st.checkbox("Show help", key="mcm_help",
                                help="Show a short description of the selected command.")

    cmd_name = command_input.strip().lstrip("/").split()[0] if command_input else ""
    if cmd_name in cmdreg.EXPIRY_COMMANDS:
        st.caption("This command uses the **default expiry** selected in the "
                   "toolbar above.")
    if show_help:
        st.info(f"**/{cmd_name}** — "
                f"{cmdreg.COMMAND_HELP.get(cmd_name, 'No help for this command.')}")

    run_btn = st.button("Run", type="primary", key="mcm_run")

    if run_btn and command_input:
        with st.spinner("Running..."):
            target = (st.session_state.get("mcm_dte_days", 30)
                      if cmd_name in cmdreg.EXPIRY_COMMANDS else None)
            fig, df, text = cmdreg.run_command(asset, command_input,
                                               expiry_target_days=target)
        st.session_state.mcm_fig = fig
        st.session_state.mcm_df = df
        st.session_state.mcm_text = text
        if text and fig is None and (df is None or getattr(df, "empty", True)):
            st.warning(text)

    _fig = st.session_state.mcm_fig
    _df = st.session_state.mcm_df
    _text = st.session_state.mcm_text
    _has_output = _fig is not None or (_df is not None and not getattr(_df, "empty", True))

    if _has_output or _text:
        render_output(_fig, _df, _text, key_prefix="single_cmd",
                      full_width_charts=isinstance(_fig, list) and bool(_fig))

    send_tg_btn = st.button("Send to Telegram", key="mcm_send_tg",
                            disabled=not (_has_output and telegram.is_configured()))
    if send_tg_btn and _has_output:
        _ok, _reasons = send_result_to_telegram(asset, cmd_name, _fig, _df, _text)
        if _ok:
            st.success("Sent to Telegram."
                       + (f" ({len(_reasons)} piece(s) had trouble — see below.)"
                          if _reasons else ""))
        else:
            st.error("Failed to send to Telegram.")
        for _r in _reasons:
            st.caption(_r)

st.caption("Data provided by the Deribit public API.")