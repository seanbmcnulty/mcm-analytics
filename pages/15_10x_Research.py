"""
10x Research — our own versions of the charts in 10x Research's emails
(reports@update.10xresearch.com, Gmail label "Markets").

Started 2026-11-27 at Sean's request: every time a 10x email comes in, the
reproducible charts are rebuilt here from free public data, newest first.
The registry, data and maths are in lib/tenx.py; figures in lib/tenx_charts.py.
10x's images and text are licensed to the recipient, so none of them is
stored here: each chart cites the post (title, date, link) and our data source.

One scrolling page (28 Nov 2026), like Daily Shot and Ecoinometrics: charts
grouped by post, a contents list with anchors, a collapsed Refresh data
control, white chart cards.

Data: Yahoo Finance daily closes (BTC-USD, GC=F, COIN), live, with the
committed data/tenx_prices.csv (real Yahoo closes) for anything live misses.
"""
import importlib
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

# Streamlit Cloud pulls a new commit into the running process and keeps lib
# modules in sys.modules, and lib.freshness's mtime check did not fire here on
# 27 Nov 2026 (the RSI tick fix in #29 stayed invisible). Both modules are small
# and new charts land in them every time a 10x email comes in, so reload them on
# every run. tenx first: tenx_charts imports it.
importlib.reload(tenx)
importlib.reload(tenx_charts)

st.set_page_config(page_title="10x Research", page_icon="🔟", layout="wide",
                   initial_sidebar_state="expanded")
st.title("🔟 10x Research charts")
st.caption("Our own rebuilds of the charts in 10x Research's emails, on one scrolling page, "
           "newest post first, drawn from public data (Yahoo Finance daily closes). "
           "Signals and channel lines are MCM's reading of each chart, not 10x's model. "
           "Each section links the original post.")

with st.sidebar:
    st.header("10x Research")


@st.cache_data(ttl=TTL_DAILY_EXTERNAL, show_spinner="Fetching Yahoo daily closes…")
def _prices():
    # Plain tuple, not tenx.Prices: the module is reloaded every run, and pickling an
    # instance of a class that was just replaced would fail.
    pr = tenx.load_prices()
    return pr.df, dict(pr.source), list(pr.errors)


with st.expander("Refresh data", expanded=False):
    st.caption("Clears cached Yahoo fetches across the app and redraws every chart on this page.")
    cache_lib.render_refresh_button(help="Clears cached fetches across the app and reloads this page.")

prices = tenx.Prices(*_prices())
_now = fx_style.local_now()
groups = tenx.charts_by_post()
st.sidebar.caption(f"Loaded {_now:%d %b %Y %H:%M} SGT. "
                   f"{len(tenx.REGISTRY)} charts in {len(groups)} posts.")
st.caption(f"Data last updated **{_now:%d %b %Y %H:%M} SGT**.")
if prices.errors:
    st.info("Some live Yahoo fetches failed, so those series use the committed CSV of earlier Yahoo "
            "closes: " + "; ".join(prices.errors))

_contents = []
for pid, charts in groups.items():
    day = pd.Timestamp(charts[0].post_date)
    sec = f"{day:%d %b %Y} — {charts[0].post_title}"
    bits = " · ".join(f"[{c.tab}](#{tenx.anchor(pid, c.tab)})" for c in charts)
    _contents.append(f"**[{sec}](#{tenx.anchor(pid)})** — {bits}")
st.markdown("#### Contents\n" + "\n\n".join(_contents))

for pid, charts in groups.items():
    st.divider()
    day = pd.Timestamp(charts[0].post_date)
    st.header(f"{day:%d %b %Y} — {charts[0].post_title}", anchor=tenx.anchor(pid))
    st.caption(f"Original post: [{charts[0].post_title}]({charts[0].post_url})")
    for chart in charts:
        st.subheader(chart.tab, anchor=tenx.anchor(pid, chart.tab))
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
