"""
MCM Analytics — shared cache-invalidation controls.

Every page reads through some combination of:

  * this app's own ``st.cache_data``-wrapped fetch functions (page-local
    TTLs — see ``lib.constants``' ``TTL_*`` tiers),
  * ``lib.deribit``'s process-wide request cache (raw Deribit HTTP
    responses, already shared across every page and user, since Streamlit
    Community Cloud runs one process for the whole multi-page app),
  * ``lib.surface``'s computed-vol-surface cache, and
  * ``lib.history``'s historical-IV reconstruction caches.

``clear_all_caches()`` is the one place that knows about all four, so
"refresh" means the same thing wherever a page offers it. Before this,
``pages/01_MCM_Bot.py`` hand-rolled the full clear (correctly), while
``pages/06_Time_Based_Realized_Vol.py`` only cleared ``st.cache_data`` plus
``lib.deribit``'s cache — incomplete by luck rather than by design, since
that page happens not to read through ``lib.surface``/``lib.history``. The
other five analytics pages had no refresh control at all and simply waited
out whatever TTL their own fetchers used (up to an hour, for the daily
external Fear & Greed feed).

``render_refresh_button()`` gives every page the same control without each
one hand-rolling the wiring.
"""

from __future__ import annotations

import streamlit as st

from lib import deribit, history, surface


def clear_all_caches() -> None:
    """Clear every cache layer in the app. Safe to call from any page,
    regardless of which of the four layers that page actually reads."""
    try:
        st.cache_data.clear()
    except Exception:
        pass
    deribit.clear_cache()
    surface.clear_cache()
    history.clear_cache()


def render_refresh_button(
    label: str = "🔄 Refresh data (clear cache)",
    help: str | None = (
        "Clears every cached fetch/computation across the app and reloads "
        "this page with fresh data from Deribit."
    ),
    **button_kwargs,
) -> None:
    """Render a standard refresh button.

    On click, clears every cache layer (see ``clear_all_caches``) and
    reruns the page. Callers just place this once — typically in the
    sidebar, alongside a page's other controls — and don't need an
    ``if clicked:`` branch of their own.
    """
    button_kwargs.setdefault("width", "stretch")
    if st.button(label, help=help, **button_kwargs):
        clear_all_caches()
        st.rerun()
