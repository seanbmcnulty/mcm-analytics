"""
Daily Shot — a standing, Daily Shot-style chart book.

Sean (27 Nov 2026): one long page laid out like The Daily Shot newsletter,
with its sections in order and a bunch of charts under each, refreshed on
demand, not tied to any issue. Every chart is our own, from free public data
on a rolling window ending at the latest data point (Yahoo Finance, FRED, US
Treasury, Deribit), with committed CSVs of real values as fallbacks.
Chart list, loaders and maths: lib/dailyshot.py; figures: lib/dailyshot_charts.py.
"""
import concurrent.futures as cf
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from lib import fx_style
from lib import dailyshot
from lib import dailyshot_charts

# Streamlit Cloud keeps lib modules in sys.modules across commits; reload so
# chart-list changes in lib/ show up without a reboot (dailyshot first).
importlib.reload(dailyshot)
importlib.reload(dailyshot_charts)

st.set_page_config(page_title="Daily Shot", page_icon="📈", layout="wide", initial_sidebar_state="expanded")


# Plain tuples in the cache, not dailyshot.Dataset: the module is reloaded every run
# and pickling an instance of a just-replaced class fails.
@st.cache_data(ttl=3600, show_spinner="Fetching market and macro data…")
def _load_all():
    t0 = time.time()
    loaders = {"prices": dailyshot.load_prices, "fred": dailyshot.load_fred,
               "yields": dailyshot.load_yields, "dvol": dailyshot.load_dvol}
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {k: ex.submit(fn) for k, fn in loaders.items()}
        out = {k: (f.result().df, dict(f.result().source), list(f.result().errors)) for k, f in futs.items()}
    return out, fx_style.local_now(), time.time() - t0


st.title("📈 Daily Shot charts")
st.caption("A standing chart book in the style of The Daily Shot, section by section, from public data "
           "(Yahoo Finance, FRED, US Treasury, Deribit). Each chart runs to the latest data point; "
           "press Refresh to pull fresh data.")

with st.sidebar:
    st.header("Daily Shot")
    if st.button("🔄 Refresh data", width="stretch", help="Re-fetch every source for this page and redraw."):
        _load_all.clear()
        st.rerun()

data, loaded_at, secs = _load_all()
ctx = dailyshot_charts.Ctx(*(dailyshot.Dataset(*data[k]) for k in ("prices", "fred", "yields", "dvol")))
groups = dailyshot.sections()
st.sidebar.caption(f"Data last updated {loaded_at:%d %b %Y %H:%M} SGT (fetch {secs:.1f}s). "
                   f"{len(dailyshot.CHARTS)} charts in {len(groups)} sections.")
st.caption(f"Data last updated **{loaded_at:%d %b %Y %H:%M} SGT**.")
notes = [e for k in ("prices", "fred", "yields", "dvol") for e in data[k][2]]
if notes:
    st.info("Some live sources did not answer, so those charts use the committed CSV of earlier real values "
            "(see each chart's source line): " + "; ".join(notes[:6]) + (" …" if len(notes) > 6 else ""))

st.markdown("#### Contents\n" + " · ".join(f"[{s}](#{dailyshot.anchor(s)})" for s, _ in groups))

for section, charts in groups:
    st.divider()
    st.header(section, anchor=dailyshot.anchor(section))
    for chart in charts:
        try:
            fig, info = dailyshot_charts.build(chart, ctx)
        except Exception as ex:   # one bad chart must not take the page down
            st.caption(f"**{chart.title}**: not drawn ({type(ex).__name__}: {ex}).")
            continue
        st.plotly_chart(fig, theme=None, width="stretch", config={"displaylogo": False}, key=f"ds_{chart.key}")
        st.caption(f"{chart.caption} Source: {info['source']}. Data to {info['as_of']:%d %b %Y}.")
