"""
Daily Shot — our own versions of charts from The Daily Shot
(https://thedailyshot.com/, Lev Borodovsky's global macro newsletter).

Started 2026-11-27 at Sean's request. Issue bodies sit behind the subscriber
login, so each tab rebuilds the issue's public headline story (or a section
item named in the public excerpt) from free public data, newest first.
Registry, data and maths: lib/dailyshot.py; figures: lib/dailyshot_charts.py.
Nothing of The Daily Shot's is stored here: each tab cites the issue
(headline, date, link) and our data source.

Data: Yahoo Finance daily closes (live, else data/dailyshot_prices.csv), FRED
(live, else data/dailyshot_fred.csv) and Treasury par yields (lib.eco_more).
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
from lib import eco_more
from lib import dailyshot
from lib import dailyshot_charts

# Same as the 10x page: Streamlit Cloud keeps lib modules in sys.modules across
# commits, and new charts land in these two modules with every issue, so reload
# them on every run. dailyshot first: dailyshot_charts imports it.
importlib.reload(dailyshot)
importlib.reload(dailyshot_charts)

st.set_page_config(page_title="Daily Shot", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")
st.title("📈 The Daily Shot charts")
st.caption("Our own rebuilds of the lead stories in The Daily Shot, newest first, drawn from public data "
           "(Yahoo Finance, FRED, US Treasury). Issue bodies are for subscribers, so each chart follows the "
           "issue's public headline and is MCM's reading of it, not The Daily Shot's chart. Each issue "
           "header links the original.")

with st.sidebar:
    st.header("Daily Shot")
    cache_lib.render_refresh_button(help="Clears cached fetches across the app and reloads this page.")


# Plain tuples, not dailyshot.Prices: the module is reloaded every run, and pickling an
# instance of a class that was just replaced would fail.
@st.cache_data(ttl=TTL_DAILY_EXTERNAL, show_spinner="Fetching Yahoo daily closes…")
def _prices():
    pr = dailyshot.load_prices()
    return pr.df, dict(pr.source), list(pr.errors)


@st.cache_data(ttl=TTL_DAILY_EXTERNAL, show_spinner="Fetching FRED series…")
def _fred():
    fr = dailyshot.load_fred()
    return fr.df, dict(fr.source), list(fr.errors)


@st.cache_data(ttl=TTL_DAILY_EXTERNAL, show_spinner="Fetching Treasury yields…")
def _yields():
    r = eco_more.fetch_yield_curve_result(timeout=10)
    label = {"live": "US Treasury par yields (live)", "csv": "US Treasury par yields (committed CSV)"}
    return r.df, label.get(r.source, "Treasury yields unavailable")


prices = dailyshot.Prices(*_prices())
fred = dailyshot.Fred(*_fred())
ydf, ysrc = _yields()
ctx = dailyshot_charts.Ctx(prices, fred, ydf, ysrc)

_now = fx_style.local_now()
st.sidebar.caption(f"Loaded {_now:%d %b %Y %H:%M} SGT. Charts: {len(dailyshot.REGISTRY)} from "
                   f"{len(dailyshot.charts_by_issue())} issues.")
_errs = prices.errors + fred.errors
if _errs:
    st.info("Some live fetches failed, so those series use the committed CSV of earlier real values: "
            + "; ".join(_errs))



def _render(chart) -> None:
    missing = [c for c in chart.series if prices.source.get(c) == "none"]
    missing += [c for c in chart.fred if fred.source.get(c) == "none"]
    if missing:
        st.error(f"No data for {', '.join(missing)}; chart skipped.")
        return
    try:
        fig, _stats = dailyshot_charts.BUILDERS[chart.key](ctx)
    except Exception as ex:   # one bad chart must not take the page down
        st.error(f"{chart.tab}: could not draw the chart ({type(ex).__name__}: {ex}).")
        return
    st.plotly_chart(fig, theme=None, width="stretch", config={"displaylogo": False}, key=f"ds_{chart.key}")
    st.caption(chart.summary)


# One long page laid out like the newsletter: issues newest first, each issue's
# sections in its own order, charts stacked under each section. Contents first.
issues = dailyshot.layout()
toc = []
for iss in issues:
    day = pd.Timestamp(iss["date"])
    secs = " · ".join(f"[{s}](#{dailyshot.anchor(iss['date'], s)})" for s in iss["sections"])
    toc.append(f"- **[{day:%a %d %b %Y}](#{dailyshot.anchor(iss['date'])})**: {iss['title']} ({secs})")
st.markdown("#### Contents\n" + "\n".join(toc))

for iss in issues:
    day = pd.Timestamp(iss["date"])
    st.divider()
    st.header(f"The Daily Shot: {day:%d %b %Y}", anchor=dailyshot.anchor(iss["date"]))
    st.markdown(f"**[{iss['title']}]({iss['url']})**")
    if iss["all_sections"]:
        st.caption("Sections in this issue: " + " • ".join(iss["all_sections"]))
    for section, charts in iss["sections"].items():
        st.subheader(section, anchor=dailyshot.anchor(iss["date"], section))
        for chart in charts:
            _render(chart)

st.divider()
with st.expander(f"Recent issues not rebuilt ({len(dailyshot.NOT_REBUILT)})"):
    st.markdown("\n".join(f"- {pd.Timestamp(d):%d %b %Y}: [{t}]({u}). {why}"
                           for d, t, u, why in dailyshot.NOT_REBUILT))
