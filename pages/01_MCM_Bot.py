"""
MCM Bot — the full markets bot, Deribit public data only.

Command surface mirrors the FalconX bot: pick an asset and a command, or sweep
a set of commands into a dashboard grid, and blast any output to Telegram.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from lib import commands as cmdreg
from lib import fx_style, history, surface, telegram
from lib.constants import ASSETS

st.set_page_config(page_title="MCM Bot", page_icon="🤖", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🤖 MCM Bot")
    asset = st.selectbox("Asset", ASSETS, key="mcm_asset")
    st.divider()
    st.caption("Deribit public API · no authentication")
    if telegram.is_configured():
        st.success("Telegram ready")
    else:
        st.warning("Telegram not configured")
    st.divider()
    if st.button("Clear data cache", use_container_width=True):
        from lib import deribit
        deribit.clear_cache()
        st.rerun()

# Record a surface snapshot so historical charts improve the longer this runs.
try:
    history.record_snapshot(asset)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _expiry_options(asset_key: str) -> list[int]:
    try:
        dtes = surface.listed_expiries(asset_key)
    except Exception:
        dtes = []
    return dtes or [7, 30, 90]


def _telegram_controls(key: str, fig=None, df=None, text: str = "",
                       caption: str = ""):
    """Send-to-Telegram button for one command output."""
    if not telegram.is_configured():
        return
    if st.button("📤 Send to Telegram", key=f"tg_{key}"):
        ok = False
        figs = fig if isinstance(fig, list) else ([fig] if fig is not None else [])
        for i, f in enumerate(figs):
            png = fx_style.fig_to_png(f, width=1200, height=800)
            if png:
                ok = telegram.send_photo(png, caption if i == 0 else "") or ok
        if df is not None and not df.empty:
            png = fx_style.dataframe_to_table_image(df, header_text=text or caption)
            if png:
                ok = telegram.send_photo(png, caption) or ok
            else:
                ok = telegram.send_message(
                    f"<b>{caption}</b>\n<pre>{df.to_string(index=False)}</pre>") or ok
        if not figs and (df is None or df.empty) and text:
            ok = telegram.send_message(f"<b>{caption}</b>\n{text}") or ok
        st.success("Sent to Telegram") if ok else st.error("Telegram send failed")


def _render_output(fig, df, text, key: str, caption: str = "",
                   show_telegram: bool = True):
    """Render one command result: banner, figures, table, message."""
    is_error = (fig is None and (df is None or getattr(df, "empty", True))
                and bool(text))
    if is_error:
        st.info(text)
        return

    if text and not is_error:
        st.markdown(text.replace("**", "**"))

    figs = fig if isinstance(fig, list) else ([fig] if fig is not None else [])
    for i, f in enumerate(figs):
        st.plotly_chart(f, use_container_width=True, key=f"{key}_fig{i}")

    if df is not None and not getattr(df, "empty", True):
        try:
            st.dataframe(fx_style.style_dataframe(df), use_container_width=True,
                         hide_index=True)
        except Exception:
            st.dataframe(df, use_container_width=True, hide_index=True)

    if show_telegram:
        _telegram_controls(key, fig=fig, df=df, text=text, caption=caption)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title(f"🤖 MCM Bot — {asset}")
st.caption("Crypto derivatives analytics · Deribit public API · "
           f"{datetime.now(timezone.utc):%d-%b-%Y %H:%M} UTC")

tab_single, tab_dashboard = st.tabs(["Single command", "Dashboard"])

# ---------------------------------------------------------------------------
# Single command
# ---------------------------------------------------------------------------

with tab_single:
    col_cmd, col_dte = st.columns([3, 2])
    with col_cmd:
        command = st.selectbox("Command", cmdreg.COMMANDS, key="mcm_cmd",
                               help="Choose which report to run.")
    cmd_name = command.lstrip("/")

    target_days = None
    with col_dte:
        if cmd_name in cmdreg.EXPIRY_COMMANDS:
            opts = _expiry_options(asset)
            default = min(opts, key=lambda d: abs(d - 30))
            target_days = st.selectbox(
                "Expiry (DTE)", opts, index=opts.index(default),
                format_func=lambda d: f"{d}d", key="mcm_dte",
                help="Listed Deribit expiries only.")
        else:
            st.write("")

    st.caption(cmdreg.COMMAND_HELP.get(cmd_name, ""))

    if st.button("Run", type="primary", key="mcm_run"):
        st.session_state["mcm_last"] = (asset, command, target_days)

    last = st.session_state.get("mcm_last")
    if last:
        a, c, d = last
        with st.spinner(f"Running {c} for {a}…"):
            fig, df, text = cmdreg.run_command(a, c, expiry_target_days=d)
        st.subheader(f"{a} {c}")
        _render_output(fig, df, text, key=f"single_{a}_{c}",
                       caption=f"{a} {c}")

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

with tab_dashboard:
    st.markdown("Run several commands at once. Results appear in order.")
    dash_assets = st.multiselect("Assets", ASSETS, default=[asset],
                                 key="mcm_dash_assets")
    dash_cmds = st.multiselect("Commands", list(cmdreg.COMMAND_NAMES),
                               default=["vol_run", "vol_term_structure",
                                        "skew_term_structure", "basis_run"],
                               key="mcm_dash_cmds")
    dash_opts = _expiry_options(asset)
    dash_default = min(dash_opts, key=lambda d: abs(d - 30))
    dash_dte = st.selectbox("Expiry (DTE) for expiry-based commands", dash_opts,
                            index=dash_opts.index(dash_default),
                            format_func=lambda d: f"{d}d", key="mcm_dash_dte")

    run_dash = st.button("Run dashboard", type="primary", key="mcm_dash_run")
    if run_dash:
        st.session_state["mcm_dash"] = (tuple(dash_assets), tuple(dash_cmds),
                                        dash_dte)

    dash_state = st.session_state.get("mcm_dash")
    if dash_state:
        d_assets, d_cmds, d_dte = dash_state
        if not d_assets or not d_cmds:
            st.info("Pick at least one asset and one command.")
        else:
            total = len(d_assets) * len(d_cmds)
            progress = st.progress(0.0)
            done = 0
            results = []
            for a in d_assets:
                for cn in d_cmds:
                    fig, df, text = cmdreg.run_command(
                        a, "/" + cn,
                        expiry_target_days=d_dte if cn in cmdreg.EXPIRY_COMMANDS else None)
                    results.append((a, cn, fig, df, text))
                    done += 1
                    progress.progress(done / total)
            progress.empty()

            for a, cn, fig, df, text in results:
                with st.expander(f"{a} · /{cn}", expanded=True):
                    _render_output(fig, df, text, key=f"dash_{a}_{cn}",
                                   caption=f"{a} /{cn}")

            if telegram.is_configured():
                st.divider()
                if st.button("📤 Blast entire dashboard to Telegram",
                             key="mcm_dash_blast"):
                    sent = 0
                    for a, cn, fig, df, text in results:
                        caption = f"{a} /{cn}"
                        figs = fig if isinstance(fig, list) else (
                            [fig] if fig is not None else [])
                        for i, f in enumerate(figs):
                            png = fx_style.fig_to_png(f, width=1200, height=800)
                            if png and telegram.send_photo(png, caption if i == 0 else ""):
                                sent += 1
                        if df is not None and not df.empty:
                            png = fx_style.dataframe_to_table_image(
                                df, header_text=text or caption)
                            if png and telegram.send_photo(png, caption):
                                sent += 1
                    st.success(f"Sent {sent} images to Telegram")
