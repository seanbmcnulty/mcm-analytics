"""
Ecoinometrics-style charts — our own versions (and extensions) of charts from the
Ecoinometrics newsletter, restyled to match their look (lib/eco_style.py).

Exceptions to "Deribit-only" (user-approved 2026-11-01): Farside ETF flows, Bitstamp daily
closes (hack events before Aug 2018 only), FRED (fed funds, Nasdaq, CPI, 10Y), and
federalreserve.gov transcripts. Fed tone is a lexicon proxy, NOT their Fed Communication Index.

The "BTC / 200DMA ratio" and "ETF flow regime" tabs (2026-11-09) are extra views
inspired by the 6 Nov 2026 Ecoinometrics issue. The ratio is the Deribit daily
close over a 200-calendar-day average. The flow colours are MCM's own regime,
not the unpublished Ecoinometrics model. Both are drawn with fx_style.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.deribit import get_tradingview_ohlc
from lib.constants import TTL_SLOW, TTL_DAILY_EXTERNAL
from lib import cache as cache_lib
from lib import ecoinometrics as eco
from lib import eco_charts
from lib import eco_style as es
from lib import fx_style

st.set_page_config(page_title="Ecoinometrics Charts", page_icon="📰",
                   layout="wide", initial_sidebar_state="expanded")
st.title("📰 Ecoinometrics-style charts")
st.caption("Our own versions of recurring Ecoinometrics charts, plus extras. Sources: Farside, "
           "Deribit BTC-PERP, Bitstamp (pre-2018), FRED, federalreserve.gov. "
           "The flow-regime colours are an MCM approximation, not the Ecoinometrics model.")

with st.sidebar:
    st.header("Parameters")
    lookback = st.selectbox("ETF / 200DMA / correlation window", ["6 months", "1 year", "2 years", "All"], index=1)
    min_streak = st.number_input("Min streak length (days)", 1, 10, 1)
    horizon = st.selectbox("Hack study horizon (days)", [1, 3, 5, 7, 14, 30], index=2)
    corr_win = st.number_input("Correlation window (trading days)", 20, 250, 90)
    st.divider()
    cache_lib.render_refresh_button()

LB_DAYS = {"6 months": 183, "1 year": 365, "2 years": 730, "All": 10_000}[lookback]
# Display window is anchored on the SGT calendar date. The ratio and the
# flow regime are still computed on the full history, then sliced — see
# eco.display_from.
_now_sgt = fx_style.local_now()
as_of_date = pd.Timestamp(_now_sgt.year, _now_sgt.month, _now_sgt.day)
cutoff = as_of_date - pd.Timedelta(days=LB_DAYS)
st.sidebar.caption(
    f"Display window from {cutoff:%d %b %Y}. "
    f"As of {_now_sgt:%d %b %Y %H:%M} SGT. "
    "The 200-day average and the flow-regime scale use the full history, then this window is applied."
)


def show(fig):
    st.plotly_chart(fig, theme=None, width="stretch", config={"displaylogo": False})


@st.cache_data(ttl=TTL_DAILY_EXTERNAL, show_spinner="Fetching ETF flows…")
def _etf():
    return eco.fetch_etf_flows()


@st.cache_data(ttl=TTL_SLOW, show_spinner="Fetching BTC-PERP daily closes…")
def _btc_daily():
    start = int(pd.Timestamp("2018-08-01").timestamp() * 1000)
    out, cur, end = [], start, int(time.time() * 1000)
    while cur < end:
        nxt = min(cur + 1000 * 86_400_000, end)
        df = get_tradingview_ohlc("BTC-PERPETUAL", "1D", cur, nxt)
        if df is not None and len(df):
            out.append(df)
        cur = nxt
    return pd.concat(out).drop_duplicates("timestamp")[["timestamp", "close"]] if out else None


@st.cache_data(ttl=TTL_DAILY_EXTERNAL, show_spinner="Fetching pre-2018 BTC closes…")
def _btc_early():
    return eco.fetch_bitstamp_daily("2014-12-01", "2018-08-02")


@st.cache_data(ttl=TTL_DAILY_EXTERNAL, show_spinner="Fetching FRED…")
def _fred(sid):
    return eco.fred_series(sid)


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as ex:
        st.error(f"Data unavailable: {ex}")
        return None


tabs = st.tabs(["ETF flow streaks", "Exchange hacks", "Fed tone", "BTC vs 200DMA",
                "BTC–Nasdaq correlation", "Inflation & yields",
                "BTC / 200DMA ratio", "ETF flow regime"])
px = _btc_daily()

# ------------------------------------------------------------------ ETF flows
with tabs[0]:
    flows = _safe(_etf)
    if flows is not None and px is not None:
        fb = eco.flows_to_btc(flows, px)
        fb = fb[fb["date"] >= cutoff]
        sp = eco.streak_paths(fb)
        sp = sp[sp.groupby("streak")["date"].transform("size") >= min_streak]
        r30 = eco.rolling_flow(fb, 30)
        cur = eco.flow_streaks(fb).iloc[-1]
        c = st.columns(3)
        c[0].metric("30d net (BTC)", f"{r30.iloc[-1]:,.0f}")
        c[1].metric("Current streak", f"{int(cur['days'])}d {'in' if cur['total'] > 0 else 'out'}", f"{cur['total']:,.0f} BTC")
        c[2].metric("Last day (BTC)", f"{fb.iloc[-1]['flow']:,.0f}")
        fig = go.Figure()
        for sign, col in ((1, es.RED), (-1, es.TEAL)):
            xs, ys = [], []
            for _, g in sp[sp["sign"] == sign].groupby("streak"):
                xs += [g["date"].iloc[0], *g["date"], None]
                ys += [0, *g["cum"], None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=2.5),
                                     hovertemplate="%{x|%d %b %Y}: %{y:,.0f} BTC<extra></extra>"))
        tot = sp.groupby("streak").agg(end=("date", "last"), cum=("cum", "last"))
        if len(tot):
            hi, lo = tot["cum"].idxmax(), tot["cum"].idxmin()
            es.note(fig, tot.loc[hi, "end"], tot.loc[hi, "cum"], "Strongest inflow streak<br>in window", color=es.RED, ay=-40, ax=-60)
            es.note(fig, tot.loc[lo, "end"], tot.loc[lo, "cum"], "Worst outflow streak<br>in window", color=es.TEAL, ay=40, ax=-60)
        es.style(fig, "Bitcoin ETF Flows by Streak", "Bitcoin",
                 f"Cumulative inflows/outflows over consecutive trading-day streaks. 30-day net flow: {r30.iloc[-1]:,.0f} BTC.",
                 "Notes: daily Farside US$ totals converted to BTC at that day's close. Source: Farside, Deribit. Not Ecoinometrics' data.",
                 yfmt=",.0f")
        fig.update_yaxes(title_text="Cumulative inflows/outflows by streak (BTC)", title_font=dict(size=14))
        show(fig)

# ----------------------------------------------------------------- Hacks study
with tabs[1]:
    hacks = eco.load_hacks()
    early = _safe(_btc_early) if (hacks["date"] < "2018-08-02").any() else None
    parts = [d for d in (early, px) if d is not None and len(d)]
    if not parts:
        st.error("No BTC price history available.")
    else:
        allpx = pd.concat(parts).assign(d=lambda x: pd.to_datetime(x["timestamp"]).dt.normalize()) \
                  .drop_duplicates("d", keep="last").drop(columns="d")
        res = eco.hack_event_returns(allpx, hacks, int(horizon))
        if res.empty:
            st.warning("No events overlap the price history.")
        else:
            m = res["ret"].mean()
            fig = go.Figure(go.Scatter(
                x=res["loss_usd_m"], y=res["ret"] * 100, mode="markers+text",
                text=res["exchange"] + " (" + res["date"].dt.year.astype(str) + ")",
                textposition=["top left" if v > 400 else ("bottom right" if r < 0 else "top right") for v, r in zip(res["loss_usd_m"], res["ret"])], textfont=dict(family="Arial", size=12, color=es.INK),
                marker=dict(size=13, color=[es.RED if v >= 0 else es.TEAL for v in res["ret"]]),
                hovertemplate="%{text}<br>%{y:.1f}% | $%{x:,.0f}M<extra></extra>"))
            es.style(fig, "Bitcoin Returns After Major Crypto Hacks", "Bitcoin",
                     f"Across {len(res)} exchange hacks the average {horizon}-day Bitcoin return was {m:.1%}. "
                     "Size of the hack tells us little about what happens next.",
                     f"Notes: {len(res)} selected events; close-to-close return from event date to date + {horizon} days. "
                     "Event list hand-curated in data/exchange_hacks.csv (verify). Prices: Deribit, Bitstamp pre-2018.")
            fig.update_xaxes(type="log", title_text="Reported hack size (USD millions, log scale)",
                             tickvals=[5, 10, 30, 100, 300, 1000], ticktext=["$5M", "$10M", "$30M", "$100M", "$300M", "$1,000M"])
            fig.update_yaxes(ticksuffix="%", title_text=f"Bitcoin return after {horizon} days")
            es.note(fig, res["loss_usd_m"].max(), m * 100, f"On average, returns<br>are {m:.1%}", ax=-20, ay=-50)
            show(fig)

# --------------------------------------------------------------------- Fed tone
with tabs[2]:
    ft = eco.load_fed_tone()
    if ft.empty:
        st.info("No data/fed_tone.csv yet. Run `python tools/build_fed_tone.py 2019` locally "
                "(needs `pip install pypdf requests`), then push.")
    else:
        dff = _safe(_fred, "DFF")
        if dff is not None:
            ft = eco.add_fed_funds(ft, dff)
        fig = go.Figure()
        have = "ffr" in ft
        # colour driver: fed funds when FRED is available, else the tone level itself
        drv = ft["ffr"] if have else ft["avg6"]
        lo, hi = (0.0, max(5.5, float(drv.max()))) if have else (-50.0, 100.0)
        ramp = [[0, "#1B4F72"], [0.2, es.TEAL], [0.4, "#8CC7A1"], [0.55, es.YELLOW],
                [0.78, "#F08A0A"], [1, "#B3123F"]]
        def _col(v):
            t = min(1.0, max(0.0, (v - lo) / (hi - lo)))
            for (t0, c0), (t1, c1) in zip(ramp, ramp[1:]):
                if t <= t1:
                    f = (t - t0) / (t1 - t0) if t1 > t0 else 0
                    a_, b_ = [int(c0[i:i + 2], 16) for i in (1, 3, 5)], [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
                    return "#%02X%02X%02X" % tuple(round(x + (y - x) * f) for x, y in zip(a_, b_))
            return ramp[-1][1]
        ymax = float(max(100, ft["idx"].max() + 5)); ymin = float(min(-50, ft["idx"].min() - 5))
        fig.add_hrect(y0=0, y1=ymax, fillcolor="#F51E0A", opacity=0.06, line_width=0,
                      annotation_text="HAWKISH", annotation_position="top right",
                      annotation_font=dict(color="#B3123F", size=11))
        fig.add_hrect(y0=ymin, y1=0, fillcolor="#4A9AB5", opacity=0.12, line_width=0,
                      annotation_text="DOVISH", annotation_position="bottom right",
                      annotation_font=dict(color="#1B4F72", size=11))
        # lollipop stems from zero to each meeting's score
        for d, v, c in zip(ft["date"], ft["idx"], drv):
            fig.add_shape(type="line", x0=d, x1=d, y0=0, y1=v, line=dict(color=_col(c), width=1.5), opacity=0.45)
        # smoothed line: one glowing segment per meeting pair
        for i in range(1, len(ft)):
            seg = dict(x=ft["date"].iloc[i - 1:i + 1], y=ft["avg6"].iloc[i - 1:i + 1], mode="lines",
                       hoverinfo="skip", showlegend=False)
            fig.add_trace(go.Scatter(**seg, line=dict(color=_col(drv.iloc[i]), width=11), opacity=0.18))
            fig.add_trace(go.Scatter(**seg, line=dict(color=_col(drv.iloc[i]), width=4)))
        fig.add_trace(go.Scatter(
            x=ft["date"], y=ft["idx"], mode="markers", showlegend=False,
            marker=dict(size=11, symbol=["triangle-up" if v > 0 else "triangle-down" for v in ft["idx"]],
                        color=drv, colorscale=ramp, cmin=lo, cmax=hi,
                        line=dict(color="white", width=1.2),
                        showscale=True, colorbar=dict(title="Fed funds %" if have else "Tone", orientation="h",
                                                      y=1.02, len=0.4, thickness=10, x=0.35)),
            hovertemplate="%{x|%d %b %Y}: %{y:.1f}<extra></extra>"))
        es.style(fig, "Fed Press-Conference Tone", "Tone",
                 "Hawkish minus dovish language in the Chair's FOMC press-conference remarks; line is the six-meeting average.",
                 "Notes: lexicon proxy (phrase lists in lib/ecoinometrics.py), not the Ecoinometrics Fed Communication Index. "
                 "Source: federalreserve.gov, FRED.")
        fig.update_yaxes(title_text="Tone index (hawkish +, dovish −)")
        show(fig)

# ------------------------------------------------------------------ 200DMA
with tabs[3]:
    if px is not None:
        g = eco.sma_gap(px, 200)
        gw = g[g["date"] >= cutoff]
        last = g.dropna().iloc[-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gw["date"], y=gw["sma"], line=dict(color=es.ORANGE, width=3), name="200DMA"))
        fig.add_trace(go.Scatter(x=gw["date"], y=gw["close"], line=dict(color=es.INK, width=2), name="BTC"))
        es.style(fig, "Bitcoin vs Its 200-Day Average", "Bitcoin",
                 f"Bitcoin trades {last['gap']:+.1%} versus its 200-day moving average ({last['close']:,.0f} vs {last['sma']:,.0f}).",
                 "Notes: Deribit BTC-PERPETUAL daily closes. Source: Deribit.", yfmt=",.0f")
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.04, x=0.6))
        show(fig)

# ------------------------------------------------------------ Nasdaq correlation
with tabs[4]:
    nq = _safe(_fred, "NASDAQCOM")
    if nq is not None and px is not None:
        c = eco.rolling_corr(px, nq, int(corr_win))
        cw = c[c["date"] >= cutoff]
        avg = c["corr"].mean()
        fig = go.Figure(go.Scatter(x=cw["date"], y=cw["corr"], line=dict(color=es.ORANGE, width=3)))
        fig.add_hline(y=avg, line_dash="dash", line_color=es.TEAL)
        es.note(fig, cw["date"].iloc[len(cw) // 8], avg, f"Long-run average {avg:.2f}", color=es.TEAL, ay=-35)
        es.style(fig, "Bitcoin and the Nasdaq", "Bitcoin",
                 f"{corr_win}-day correlation of daily returns is {c['corr'].iloc[-1]:.2f} versus an average of {avg:.2f} since 2018.",
                 "Notes: rolling correlation of daily log returns on Nasdaq trading days. Source: Deribit, FRED (NASDAQCOM).",
                 yfmt=".1f")
        show(fig)

# ------------------------------------------------------------- Inflation / yields
with tabs[5]:
    cpi, y10 = _safe(_fred, "CPIAUCSL"), _safe(_fred, "DGS10")
    if cpi is not None:
        cpi = cpi.assign(yoy=cpi["value"].pct_change(12) * 100).dropna()
        cw = cpi[cpi["date"] >= cutoff - pd.Timedelta(days=365)]
        fig = go.Figure(go.Bar(x=cw["date"], y=cw["yoy"], marker_color=[es.RED if v > 2 else es.TEAL for v in cw["yoy"]]))
        fig.add_hline(y=2, line_dash="dash", line_color=es.MUTED)
        es.style(fig, "US Inflation", "Inflation",
                 f"Headline CPI is running at {cw['yoy'].iloc[-1]:.1f}% year on year versus the 2% target.",
                 "Notes: CPIAUCSL, 12-month change. Source: FRED.", height=520)
        fig.update_yaxes(ticksuffix="%")
        show(fig)
    if y10 is not None:
        yw = y10[y10["date"] >= cutoff]
        fig = go.Figure(go.Scatter(x=yw["date"], y=yw["value"], line=dict(color=es.RED, width=3)))
        es.style(fig, "The 10-Year Treasury Yield", "10-Year",
                 f"The 10-year yield is {yw['value'].iloc[-1]:.2f}%, a key competitor for capital.",
                 "Notes: DGS10. Source: FRED.", height=520)
        fig.update_yaxes(ticksuffix="%")
        show(fig)

# ------------------------------------------------------- close / 200-day ratio
with tabs[6]:
    st.caption(
        "Bitcoin close divided by its 200-calendar-day average, coloured by the close. "
        "Inspired by the 6 Nov 2026 Ecoinometrics chart "
        "\"Bitcoin On Its Way To Retest The Long Term Trend\". "
        f"As of {_now_sgt:%d %b %Y %H:%M} SGT."
    )
    if px is None:
        st.error("No Deribit BTC-PERP daily history, so the ratio cannot be computed.")
    else:
        ratio_full = eco.price_sma_ratio(px, eco.SMA_CALENDAR_DAYS)
        ratio_view = eco.display_from(ratio_full, cutoff)
        valid = ratio_view.dropna(subset=["ratio"])
        if valid.empty:
            st.info(
                f"No complete {eco.SMA_CALENDAR_DAYS}-calendar-day window inside this display range. "
                "The average is left blank until every day in it has a real close."
            )
        else:
            last = valid.iloc[-1]
            c = st.columns(3)
            c[0].metric("Close / 200-day average", f"{last['ratio']:.3f}")
            c[1].metric("BTC close", f"{last['close']:,.0f} USD")
            c[2].metric("200-day average", f"{last['sma']:,.0f} USD")
            st.caption(
                f"Latest valid session {pd.Timestamp(last['date']):%d %b %Y} "
                f"(UTC date of the daily bar; Deribit stamps it at 08:00 UTC, "
                f"16:00 SGT the same calendar date). "
                f"The close is {last['ratio'] - 1:+.1%} versus the average."
            )
        show(eco_charts.figure_price_sma_ratio(ratio_view))
    with st.expander("How the 200-day ratio is calculated"):
        st.markdown(eco.sma_ratio_methodology())

# ------------------------------------------------------- ETF flow regime
with tabs[7]:
    st.caption(
        "Daily US spot BTC ETF net flow in BTC, coloured by MCM's trailing-flow regime. "
        "Inspired by the 6 Nov 2026 Ecoinometrics chart "
        "\"Bitcoin ETFs Regime Has Switched To Strong Outflows\". "
        "The regime is our approximation, not their unpublished model. "
        f"As of {_now_sgt:%d %b %Y %H:%M} SGT."
    )
    flows = _safe(_etf)
    if flows is None or px is None:
        st.error("The flow regime needs the Farside total and Deribit BTC-PERP daily closes.")
    else:
        btc_flow = eco.flows_to_btc(flows, px)
        if btc_flow.empty:
            st.warning(
                "No Farside session shared a UTC date with a positive Deribit close, "
                "so no bar was converted. Missing prices were not filled with zero or with an earlier close."
            )
        else:
            regime_full = eco.etf_flow_regime(btc_flow)
            regime_view = eco.display_from(regime_full, cutoff)
            if regime_view.empty:
                st.info("No aligned sessions inside this display window.")
            else:
                last = regime_view.iloc[-1]
                label = eco_charts.REGIME_LABELS.get(str(last["regime"]), str(last["regime"]))
                c = st.columns(3)
                c[0].metric("Latest daily flow", f"{last['flow']:,.0f} BTC")
                c[1].metric("MCM regime", label)
                sum_txt = "warmup" if pd.isna(last["sum_window"]) else f"{last['sum_window']:,.0f} BTC"
                c[2].metric(f"{eco.FLOW_REGIME_WINDOW}-session sum", sum_txt)
                if pd.notna(last["scale_median"]):
                    st.caption(
                        f"Session {pd.Timestamp(last['date']):%d %b %Y}. "
                        f"Scale median {last['scale_median']:,.0f} BTC. "
                        f"Mild cutoff ±{last['mild_cutoff']:,.0f} BTC. "
                        f"Strong cutoff ±{last['strong_cutoff']:,.0f} BTC. "
                        f"{len(regime_full)} aligned sessions in the full tape; "
                        f"the first {eco.FLOW_REGIME_SCALE - 1} stay unclassified."
                    )
                else:
                    st.caption(
                        f"Session {pd.Timestamp(last['date']):%d %b %Y} is unclassified. "
                        f"The scale median needs {eco.FLOW_REGIME_SCALE} aligned sessions; "
                        f"the tape has {len(regime_full)}."
                    )
                audit = regime_view.tail(12).copy()
                audit["date"] = pd.to_datetime(audit["date"]).dt.strftime("%d %b %Y")
                audit["regime"] = audit["regime"].map(lambda r: eco_charts.REGIME_LABELS.get(str(r), str(r)))
                show_cols = ["date", "flow", "sum_window", "scale_median", "mild_cutoff", "strong_cutoff", "regime"]
                st.dataframe(
                    audit[show_cols].rename(columns={
                        "date": "Session",
                        "flow": "Daily flow (BTC)",
                        "sum_window": f"{eco.FLOW_REGIME_WINDOW}-session sum (BTC)",
                        "scale_median": f"{eco.FLOW_REGIME_SCALE}-session median |flow| (BTC)",
                        "mild_cutoff": "Mild cutoff (BTC)",
                        "strong_cutoff": "Strong cutoff (BTC)",
                        "regime": "Regime",
                    }),
                    width="stretch", hide_index=True,
                )
            show(eco_charts.figure_etf_flow_regime(regime_view))
    with st.expander("How the flow regime is calculated"):
        st.markdown(eco.flow_regime_methodology())

st.divider()
st.caption(
    "Our own charts inspired by the Ecoinometrics newsletter, built from public data. "
    "The ETF regime colours are an MCM approximation, not their model or their data. No trading advice."
)
