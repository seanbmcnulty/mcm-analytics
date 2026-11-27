"""
10x Research — our own versions of the charts in 10x Research's emails
(reports@update.10xresearch.com, Gmail label "Markets").

Started 2026-11-27 at Sean's request: every time a 10x email comes in, the
reproducible charts are rebuilt here from free public data, newest first.
The registry, data and maths are in lib/tenx.py; figures in lib/tenx_charts.py.
10x's images and text are licensed to the recipient, so none of them is
stored here: each tab cites the post (title, date, link) and our data source.

Data: Yahoo Finance daily closes (BTC-USD, GC=F, COIN), live, with the
committed data/tenx_prices.csv (real Yahoo closes) for anything live misses.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from lib.constants import TTL_DAILY_EXTERNAL
from lib import cache as cache_lib
from lib import fx_style
from lib import tenx
from lib import tenx_charts

st.set_page_config(page_title="10x Research", page_icon="🔟", layout="wide",
                   initial_sidebar_state="expanded")
st.title("🔟 10x Research charts")
st.caption("Our own rebuilds of the charts in 10x Research's emails, newest first, drawn from public "
           "data (Yahoo Finance daily closes). Signals and channel lines are MCM's reading of each chart, "
           "not 10x's model. Each tab links the original post.")

with st.sidebar:
    st.header("10x Research")
    cache_lib.render_refresh_button(help="Clears cached fetches across the app and reloads this page.")


@st.cache_data(ttl=TTL_DAILY_EXTERNAL, show_spinner="Fetching Yahoo daily closes…")
def _prices():
    return tenx.load_prices()


prices = _prices()
_now = fx_style.local_now()
st.sidebar.caption(f"Loaded {_now:%d %b %Y %H:%M} SGT. Charts: {len(tenx.REGISTRY)}.")
if prices.errors:
    st.warning("Some live Yahoo fetches failed, so those series use the committed CSV of earlier Yahoo "
               "closes: " + "; ".join(prices.errors))

tabs = st.tabs([c.tab for c in tenx.REGISTRY])
for tab, chart in zip(tabs, tenx.REGISTRY):
    with tab:
        missing = [c for c in chart.series if prices.source.get(c) == "none"]
        if missing:
            st.error(f"No data for {', '.join(missing)}; chart skipped.")
            continue
        try:
            fig, _stats = tenx_charts.BUILDERS[chart.key](prices)
        except Exception as ex:   # one bad chart must not take the page down
            st.error(f"{chart.tab}: could not draw the chart ({type(ex).__name__}: {ex}).")
            continue
        st.plotly_chart(fig, theme=None, width="stretch", config={"displaylogo": False},
                        key=f"tenx_{chart.key}")
        post_day = pd.Timestamp(chart.post_date)
        st.caption(f"Rebuilt from 10x Research {chart.exhibit} in [{chart.post_title}]({chart.post_url}), "
                   f"{post_day:%d %b %Y}. {chart.summary}")
