"""
TradFi vol vs crypto vol: does the VIX (and MOVE, VVIX, the VIX curve, DXY,
HY credit spreads, Nasdaq realized vol) lead or predict BTC / ETH DVOL? And is
crypto vol rich or cheap vs TradFi vol right now?

Data (free, public, no tokens): Deribit DVOL (hourly, snapped to the 16:00
New York close), Cboe VIX/VVIX/VIX9D/VIX3M, Yahoo ^MOVE / DX-Y.NYB / ^NDX,
FRED BAMLH0A0HYM2. Committed CSVs data/vl_*.csv, refreshed by the buttons
below (this machine only) and weekly by .github/workflows/update_vol_lead_data.yml.
Analysis lives in lib/vol_lead.py (tests/test_vol_lead.py).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from lib import freshness, fx_style, blast
from lib import vol_lead_data as vld
from lib import vol_lead as vl
from lib import vol_lead_charts as vlc
from lib import vol_lead_intraday as vi
from lib import vol_lead_explain as vx

# Streamlit Cloud keeps old lib modules in memory after a deploy.
_RELOADED = freshness.reload_if_changed(fx_style, vld, vl, vi, vx, vlc) + freshness.reload_if_changed(blast)

st.set_page_config(page_title="TradFi Vol vs DVOL", page_icon="📉", layout="wide",
                   initial_sidebar_state="expanded")
st.title("📉 TradFi vol vs crypto vol (VIX → DVOL)")
st.caption("Two questions: does traditional-market vol (VIX, MOVE and friends) lead BTC / ETH DVOL, and is crypto "
           "vol rich or cheap vs TradFi vol right now? Read top to bottom: verdict → rich/cheap → intraday → daily → "
           "deeper stats. Sources: Deribit DVOL, Cboe (VIX, VVIX, VIX9D, VIX3M), Yahoo (VIX intraday, MOVE, DXY, "
           "Nasdaq-100), FRED (HY OAS). Daily rows are US trading days with DVOL at the 16:00 New York close.")

# ------------------------------------------------------------------ blast step
_blast_step = blast.active_step("vixdvol")
_SHOWN: list = []
_cur = None
if _blast_step is not None:
    blast.banner(_blast_step, "drawing the charts and sending the selected ones to Telegram…")

WINDOWS = {"3 months": 63, "6 months": 126, "1 year": 252, "2 years": 504}
FV_WINDOWS = {"3 months": 63, "6 months": 126}

with st.sidebar:
    st.header("Parameters")
    _assets = list(vl.ASSETS)
    _default_asset = _blast_step.asset if _blast_step is not None and _blast_step.asset in _assets else "BTC"
    asset = st.radio("Crypto vol index", _assets, index=_assets.index(_default_asset), horizontal=True)
    lookback = st.selectbox("Analysis window", list(WINDOWS), index=2,
                            help="Sample for every statistic, daily and intraday (2 years at most). "
                                 "5-minute data only goes back to late Aug 2026 (Yahoo keeps ~60 days; "
                                 "the stored file grows from there).")
    fv_label = st.selectbox("Fair-value / z-score window", list(FV_WINDOWS), index=1,
                            help="Rolling window for the TradFi fair-value regression and the ratio z-scores.")
    pred_key = st.selectbox("Predictor for the daily lead-lag / rolling views", ["VIX", "MOVE", "VVIX"], index=0)
    st.caption(f"As of {fx_style.local_now():%d %b %Y %H:%M} SGT.")

N_WIN = WINDOWS[lookback]
FV_WIN = FV_WINDOWS[fv_label]
WIN_DAYS = int(N_WIN * 365 / 252)


def show(fig, key: str, takeaway: str | None = None):
    if _blast_step is not None:
        _SHOWN.append((key, fig, takeaway))
    st.plotly_chart(fig, theme=None, width="stretch", config={"displaylogo": False})


def _md(t: str) -> str:
    # Streamlit markdown: a single "~" can strike text through, "$" starts LaTeX.
    return t.replace("~", "≈").replace("$", "\\$")


def chart(fig, key: str, takeaway: str | None = None, how: str | None = None):
    """Takeaway line above, chart, collapsed 'How to read this' below."""
    if takeaway:
        st.markdown(f"**→** {_md(takeaway)}")
    show(fig, key, takeaway)
    if how:
        with st.expander("How to read this"):
            st.markdown(_md(how))


def table(df_: pd.DataFrame, fmt: dict | None = None, verdict_cols=None, mask=None, how: str | None = None):
    sty = df_.style
    if fmt:
        sty = sty.format({k: v for k, v in fmt.items() if k in df_.columns}, na_rep="—")
    sty = vx.style_verdicts(sty, [c_ for c_ in (verdict_cols or []) if c_ in df_.columns])
    if mask is not None:
        css = pd.DataFrame(np.where(mask.values.astype(bool), "background-color: rgba(14,122,56,0.14); font-weight: 600", ""),
                           index=mask.index, columns=mask.columns)
        sty = sty.apply(lambda _d: css, axis=None)
    st.dataframe(sty, hide_index=True, width="stretch")
    if how:
        with st.expander("How to read this"):
            st.markdown(_md(how))


# ------------------------------------------------------------------ data
def _mtime(key):
    p = vld.csv_path(key)
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _load(key: str, mtime: float):
    return vld.load_csv(key)


def _get(key):
    good = st.session_state.get("vl_good", {}).get(key)
    if good is not None:
        return good
    return _load(key, _mtime(key))


def _refresh(keys):
    msgs = []
    with st.spinner("Refreshing " + ", ".join(vld.DATASETS[k][1] for k in keys) + "…"):
        for k in keys:
            res = vld.refresh(k)
            msgs.append((res.ok, res.message))
            if res.ok:
                st.session_state.setdefault("vl_good", {})[k] = res
                _load.clear()
    for ok, m in msgs:
        (st.success if ok else st.error)(m)


with st.expander("🔄 Data sources and refresh", expanded=False):
    st.caption("Each button fetches live from the public source and rewrites the CSV on this machine "
               "(a redeploy goes back to the committed file; the weekly GitHub Action commits fresh ones). "
               "No tokens are used.")
    cols = st.columns(len(vld.DATASETS) + 1)
    for i, (k, (_f, label)) in enumerate(vld.DATASETS.items()):
        if cols[i].button(label.split(" (")[0].replace("Cboe VIX / VVIX / VIX9D / VIX3M", "Cboe VIX family")
                          .replace("Yahoo MOVE / DXY / NDX", "Yahoo MOVE/DXY/NDX"),
                          key=f"vl_refresh_{k}", width="stretch", help=f"Refresh {label}"):
            _refresh([k])
    if cols[-1].button("Refresh all", key="vl_refresh_all", width="stretch", type="primary"):
        _refresh(list(vld.DATASETS))
    for k in vld.DATASETS:
        r = _get(k)
        tag = "✅" if r.ok else "⚠️"
        asof = f", latest {r.as_of:%d %b %Y}" if r.as_of is not None else ""
        st.markdown(f"- {tag} {r.message}{asof} ({r.source})")

@st.cache_data(ttl=900, show_spinner="Topping up intraday VIX / DVOL…")
def _intraday(kind: str, mtime: float, bump: int):
    """Committed CSV + live top-up (Yahoo / Deribit). Falls back to the file."""
    return vld.refresh_intraday(kind, write=True)


def _imtime(kind):
    p = vld.intraday_path(kind)
    return p.stat().st_mtime if p.exists() else 0.0


with st.expander("⏱️ Intraday data (5-min and hourly)", expanded=False):
    st.caption("Topped up live every 15 minutes from Yahoo ^VIX and Deribit DVOL; the stored files are the fallback.")
    ic = st.columns(3)
    if ic[0].button("Top up 5-min now", key="vl_refresh_5m", width="stretch"):
        st.session_state["vl_bump"] = st.session_state.get("vl_bump", 0) + 1
    if ic[1].button("Top up hourly now", key="vl_refresh_1h", width="stretch"):
        st.session_state["vl_bump"] = st.session_state.get("vl_bump", 0) + 1
    _bump = st.session_state.get("vl_bump", 0)
    I5 = _intraday("5m", 0.0, _bump)
    I1 = _intraday("1h", 0.0, _bump)
    for r_ in (I5, I1):
        st.markdown(f"- {'✅' if r_.ok else '⚠️'} {r_.message} ({r_.source})")

frames = {k: (_get(k).df if _get(k).ok else pd.DataFrame()) for k in vld.DATASETS}
if frames["dvol"].empty or frames["cboe"].empty:
    st.error("DVOL or VIX data is missing. Use Refresh all above.")
    blast.stop("vixdvol", "DVOL/VIX data missing")

raw = vld.merge_panel(frames, vld.ndx_fallback())
panel = vl.align_panel(raw)
missing = [k for k in vl.PREDICTORS if k not in panel or panel[k].notna().sum() < 100]
if missing:
    st.info("Not available right now (left out of the ranking): "
            + ", ".join(vl.PREDICTORS[k] for k in missing)
            + (". FRED often times out from Streamlit Cloud; the weekly GitHub Action fills data/vl_fred.csv."
               if "HY_OAS" in missing else ". Try that source's refresh button."))

full = vl.asset_frame(panel, asset)
df = full.iloc[-N_WIN:]
min_train = int(max(40, len(df) // 2))
avail = [k for k in vl.PREDICTORS if k not in missing]


@st.cache_data(show_spinner="Running the lead-lag and forecast tests…")
def _analysis(asset: str, n_win: int, fv_win: int, pred_key: str, _df: pd.DataFrame, _full: pd.DataFrame,
              stamp: tuple, avail: tuple, min_train: int):
    x = vl.change(_df, pred_key)
    y = _df["DVOL"].diff()
    out = {
        "cc": vl.cross_corr(x, y, 10),
        "cc_vix": vl.cross_corr(vl.change(_df, "VIX"), y, 10),
        "gr_fwd": vl.granger(y, vl.change(_df, "VIX"), 5),
        "gr_rev": vl.granger(vl.change(_df, "VIX"), y, 5),
        "gr_pred_fwd": vl.granger(y, x, 5),
        "gr_pred_rev": vl.granger(x, y, 5),
        "rank": vl.ranking_table(_df, list(avail), vl.HORIZONS, min_train, spec="change"),
        "rank_full": vl.ranking_table(_df, list(avail), vl.HORIZONS, min_train, spec="full"),
        "vix_reg": [vl.predictive_regression(_df, "VIX", h, min_train) for h in vl.HORIZONS],
        "rc": vl.rich_cheap(_full, window=fv_win, long_window=max(n_win, fv_win)),
        "roll": vl.rolling_beta_corr(y, x, 21 if n_win <= 126 else 63),
        "roll_w": 21 if n_win <= 126 else 63,
    }
    rc = out["rc"].reindex(_df.index)
    out["rc_q"] = vl.score_quintiles(rc["score"], _df["DVOL"], (5, 10, 20), _df["SPREAD_VIX"])
    out["rc_sig"] = vl.score_signal(rc["score"], _df["DVOL"], (5, 10, 20), _df["SPREAD_VIX"], min_train)
    ev = {}
    start = _df.index.min()
    for k in ("VIX", "MOVE"):
        if k in _full and _full[k].notna().sum() > 200:
            lvl = _full[k].ffill()
            def _w(e):
                return e[e["date"] >= start] if not e.empty else e
            e_spike = _w(vl.find_events(lvl, 2.0, None))
            e_cross = _w(vl.find_events(lvl, 99.0, 25.0 if k == "VIX" else 130.0))
            ev[k] = {"spike": (e_spike, vl.event_path(_full["DVOL"], e_spike)),
                     "cross": (e_cross, vl.event_path(_full["DVOL"], e_cross))}
    out["events"] = ev
    out["uncond"] = vl.unconditional_path(_df["DVOL"])
    return out


@st.cache_data(show_spinner="Running the intraday tests…")
def _intraday_analysis(asset: str, win_days: int, stamp5, stamp1, _m5: pd.DataFrame, _h1: pd.DataFrame):
    col = f"DVOL_{asset}"
    out = {}
    b5, bh = pd.Timedelta(minutes=5), pd.Timedelta(hours=1)
    m5 = vi.window(_m5, win_days)
    h1 = vi.window(_h1, win_days)
    if m5 is not None and not m5.empty:
        dx, dy = vi.bar_change(m5["VIX"], b5), vi.bar_change(m5[col], b5)
        out.update(cc5=vi.cross_corr(dx, dy, b5, 12), gr5=vi.granger(dy, dx, b5, 6), gr5_rev=vi.granger(dx, dy, b5, 6),
                   fit5=vi.ols_beta(dx, dy), dx5=dx, dy5=dy)
        up, dn = vi.find_moves(dx, 3.0, sign=1), vi.find_moves(dx, 3.0, sign=-1)
        out["ev5"] = {"VIX jumps up": vi.event_path(m5[col], up, b5), "VIX drops": vi.event_path(m5[col], dn, b5)}
        out["n_sessions5"] = vi.sessions(m5.index).nunique()
    if h1 is not None and not h1.empty:
        dx, dy = vi.bar_change(h1["VIX"], bh), vi.bar_change(h1[col], bh)
        out.update(cch=vi.cross_corr(dx, dy, bh, 5), grh=vi.granger(dy, dx, bh, 4), grh_rev=vi.granger(dx, dy, bh, 4),
                   fith=vi.ols_beta(dx, dy), dxh=dx, dyh=dy,
                   rollh={"1 week (~30 hourly changes)": vi.rolling_beta_corr(dx, dy, 30),
                          "1 month (~126)": vi.rolling_beta_corr(dx, dy, 126)},
                   sp=vi.session_predict(h1, asset))
        out["n_sessions1"] = vi.sessions(h1.index).nunique()
    return out


stamp = tuple(round(_mtime(k)) for k in vld.DATASETS) + tuple(
    sorted(st.session_state.get("vl_good", {}).keys()))
A = _analysis(asset, N_WIN, FV_WIN, pred_key, df, full, stamp, tuple(avail), min_train)
rc_full = A["rc"]
rc = rc_full.reindex(df.index)
M5 = I5.df if I5.ok else pd.DataFrame()
H1 = I1.df if I1.ok else pd.DataFrame()
_s5 = (len(M5), str(M5.index.max()) if len(M5) else "")
_s1 = (len(H1), str(H1.index.max()) if len(H1) else "")
B = _intraday_analysis(asset, WIN_DAYS, _s5, _s1, M5, H1)

st.caption(f"Daily sample: {df.index.min():%d %b %Y} → {df.index.max():%d %b %Y}, {len(df)} trading days "
           f"(out-of-sample tests train on at least {min_train}). Intraday: "
           f"{B.get('n_sessions5', 0)} sessions of 5-min bars, {B.get('n_sessions1', 0)} sessions of hourly bars.")
if len(df) - min_train < 40:
    st.warning("This window leaves fewer than 40 out-of-sample days, so the out-of-sample numbers are very noisy. "
               "Use 1 year or 2 years for the forecast tests.")

sig_rc = A["rc_sig"]
rcw = rc_full.reindex(df.index)
pct_score = vl.window_pct(rcw["score"]) if "score" in rcw else np.nan

# ------------------------------------------------------------------ verdict
st.header(f"🧭 Verdict: does TradFi vol predict {asset} DVOL?")
bullets = vx.verdict_bullets(asset, lookback, cc5=B.get("cc5"), cch=B.get("cch"), cc_daily=A["cc_vix"],
                             gr_daily=A["gr_fwd"], sp=B.get("sp"), rank=A["rank"], rc=rc_full, pct=pct_score, sig=sig_rc)
st.info("\n".join(f"- {_md(b)}" for b in bullets))
with st.expander("All verdict details (every test, with the numbers)"):
    intra_lines = vi.verdict_lines(asset, B.get("cc5"), B.get("gr5"), B.get("gr5_rev"), B.get("cch"), B.get("sp"))
    lines = vl.verdict(asset, A["cc_vix"], A["gr_fwd"], A["gr_rev"], A["rank"])
    st.markdown(f"**Intraday (last {lookback} where data allows)**\n" + "\n".join(f"- {_md(ln)}" for ln in intra_lines)
                + f"\n\n**Daily (last {lookback})**\n" + "\n".join(f"- {_md(ln)}" for ln in lines))

# ------------------------------------------------------------------ rich / cheap
st.header(f"💸 Is {asset} vol rich or cheap vs TradFi vol?")
last = rc_full.dropna(subset=["score"])
if last.empty:
    st.warning("Not enough history for the rich/cheap model.")
else:
    r = last.iloc[-1]
    pct_rv = vl.window_pct(rcw["ratio_vix"])
    pct_rm = vl.window_pct(rcw["ratio_move"]) if "ratio_move" in rcw else np.nan
    st.markdown(f"**{_md(vl.rich_cheap_headline(asset, rc_full, pct_score, 'the last ' + lookback))}**  \n"
                f"<span style='color:#888'>Reading for the {last.index[-1]:%a %d %b %Y} New York close.</span>",
                unsafe_allow_html=True)
    st.plotly_chart(vlc.rich_cheap_gauge(pct_score, r["score"], r["resid"], asset, "the last " + lookback),
                    theme=None, width="stretch", config={"displaylogo": False, "displayModeBar": False})
    with st.expander("How to read this"):
        st.markdown(_md(vx.HOW["gauge"]))
    c = st.columns(5)
    c[0].metric(f"{asset} DVOL", f"{full['DVOL'].iloc[-1]:.1f}")
    c[1].metric("TradFi fair value", f"{r['fair']:.1f}", f"{r['resid']:+.1f} vols ({'rich' if r['resid'] > 0 else 'cheap'})",
                delta_color="off", help=f"Rolling {fv_label} regression of DVOL on VIX and MOVE (fit R² {r['fv_r2']:.2f}).")
    c[2].metric("Composite score", f"{r['score']:+.2f} sd", help="Average of the three component z-scores below; "
                "below 0 = cheap, above 0 = rich.")
    c[3].metric(f"Percentile ({lookback})", vl.ordinal(pct_score) if np.isfinite(pct_score) else "—",
                vx.zone(pct_score), delta_color="off")
    c[4].metric("DVOL / VIX", f"{r['ratio_vix']:.2f}x",
                f"{vl.ordinal(pct_rv)} pct ({lookback})" if np.isfinite(pct_rv) else None, delta_color="off")

    def _f(v, fmt):
        return fmt.format(v) if v is not None and np.isfinite(v) else "—"

    _rows = [("Gap to fair value (vols)", r["resid"], r["resid_z"], vl.window_pct(rcw["resid"]), "{:+.2f}"),
             ("DVOL / VIX", r["ratio_vix"], r["ratio_vix_z"], pct_rv, "{:.2f}x")]
    if "ratio_move" in r:
        _rows.append(("DVOL / MOVE", r["ratio_move"], r["ratio_move_z"], pct_rm, "{:.3f}x"))
    comp = pd.DataFrame([{"Component": n, "Now": _f(a, fm), f"z-score ({fv_label})": _f(b, "{:+.2f}"),
                          f"Percentile ({lookback})": _f(c1, "{:.0f}"), "Reads as": vx.component_signal(b)}
                         for n, a, b, c1, fm in _rows])
    st.dataframe(comp.style.map(lambda v: f"color: {vlc.RICH_C}; font-weight: 600" if isinstance(v, str) and "rich" in v.lower()
                                else f"color: {vlc.CHEAP_C}; font-weight: 600" if isinstance(v, str) and "cheap" in v.lower()
                                else "", subset=["Reads as"]), hide_index=True, width="stretch")
    chart(vlc.rich_cheap(df, rcw, asset, fv_label), "rich_cheap", vx.rich_cheap_takeaway(rcw, asset), vx.HOW["rich_cheap"])
    cq1, cq2 = st.columns([3, 2])
    with cq1:
        if not A["rc_q"].empty:
            chart(vlc.quintile_bars(A["rc_q"], asset), "rich_cheap_q", vx.quintile_takeaway(A["rc_q"], asset),
                  vx.HOW["quintiles"])
    with cq2:
        st.markdown("**Does the gauge forecast DVOL?**")
        if not sig_rc.empty:
            st.markdown(f"**→** {_md(vx.signal_takeaway(sig_rc, asset))}")
            sp_ = vx.signal_plain(sig_rc)
            table(sp_, {"Forecast gain % (out-of-sample)": "{:+.1f}"},
                  ["Rich → falls? (vols per +1 sd)", "Beats naive forecast?"], how=vx.HOW["signal"])
    # ---- today: intraday DVOL vs fair value (5-min)
    coef = rc_full.attrs.get("coef", {})
    if not M5.empty and coef:
        day = vi.last_sessions(M5, 1)
        move_last = float(full["MOVE"].dropna().iloc[-1]) if "MOVE" in full and full["MOVE"].notna().any() else None
        fvi = vi.intraday_fair(day, asset, coef, move_last).dropna()
        if not fvi.empty:
            st.subheader("Today, bar by bar")
            dlab = f"{fvi.index[-1].tz_convert('America/New_York'):%a %d %b} NY session, last bar " \
                   f"{fvi.index[-1].tz_convert('Asia/Singapore'):%H:%M} SGT"
            tcols = st.columns(3)
            tcols[0].metric(f"{asset} DVOL (latest 5-min)", f"{fvi['DVOL'].iloc[-1]:.1f}")
            tcols[1].metric("Fair value now", f"{fvi['fair'].iloc[-1]:.1f}",
                            f"{fvi['resid'].iloc[-1]:+.1f} vols ({'rich' if fvi['resid'].iloc[-1] > 0 else 'cheap'})",
                            delta_color="off")
            tcols[2].metric("Session range of the gap", f"{fvi['resid'].min():+.1f} → {fvi['resid'].max():+.1f}")
            chart(vlc.today_fair(fvi, asset, dlab), "today", vx.today_takeaway(fvi, asset), vx.HOW["today"])

# ------------------------------------------------------------------ short-term (intraday)
st.header("⚡ Short-term: VIX vs DVOL intraday")
b5, bh = pd.Timedelta(minutes=5), pd.Timedelta(hours=1)
if M5.empty and H1.empty:
    st.warning("No intraday data available right now.")
else:
    st.subheader(f"Does VIX lead {asset} DVOL within the day?")
    lc1, lc2 = st.columns(2)
    with lc1:
        if "cc5" in B:
            chart(vlc.intraday_cross_corr(B["cc5"], asset, "min"), "intraday_leadlag",
                  vx.leadlag_takeaway(B["cc5"], "min", asset), vx.HOW["leadlag5"])
    with lc2:
        if "cch" in B:
            chart(vlc.intraday_cross_corr(B["cch"], asset, "h"), "intraday_leadlag_h",
                  vx.leadlag_takeaway(B["cch"], "h", asset), vx.HOW["leadlagh"])
    gcols = st.columns(2)
    if "gr5" in B:
        with gcols[0]:
            st.markdown("**Granger test, 5-min bars**")
            st.markdown(f"**→** {_md(vx.granger_takeaway(B['gr5'], B['gr5_rev'], 'min', asset))}")
            g5 = vx.granger_table(B["gr5"], B["gr5_rev"], "min")
            table(g5, None, [c_ for c_ in g5.columns if "?" in c_], how=vx.HOW["granger"])
    if "grh" in B:
        with gcols[1]:
            st.markdown("**Granger test, hourly bars**")
            st.markdown(f"**→** {_md(vx.granger_takeaway(B['grh'], B['grh_rev'], 'h', asset))}")
            gh = vx.granger_table(B["grh"], B["grh_rev"], "h")
            table(gh, None, [c_ for c_ in gh.columns if "?" in c_], how=vx.HOW["granger"])

    t1, t2, t3, t4, t5 = st.tabs(["📈 Intraday overlay", "📊 Normalised overlay", "🔵 Scatter of moves",
                                  "🔁 Rolling corr & beta", "⚡ After big VIX moves"])
    with t1:
        if not M5.empty:
            days = sorted(set(vi.sessions(M5.index)), reverse=True)
            opts = ["Last 5 sessions", "Latest session"] + [f"{d:%a %d %b %Y}" for d in days[1:30]]
            pick = st.selectbox("Sessions", opts, index=0, key="vl_overlay_pick")
            if pick == "Last 5 sessions":
                view = vi.last_sessions(M5, 5)
            elif pick == "Latest session":
                view = vi.last_sessions(M5, 1)
            else:
                d0 = days[opts.index(pick) - 1]
                view = M5[vi.sessions(M5.index) == d0]
            chart(vlc.intraday_overlay(view, asset, b5, f"{asset} DVOL vs VIX through the US session ({pick.lower()}, 5-min)"),
                  "intraday_overlay", vx.overlay_takeaway(view, asset, b5), vx.HOW["overlay"])
    with t2:
        span = st.radio("Window", ["1 month", "3 months"], horizontal=True, key="vl_norm_span")
        src = H1 if not H1.empty else M5
        zv = vi.window(src, 31 if span == "1 month" else 92)
        zcols = [c_ for c_ in ("VIX", "DVOL_BTC", "DVOL_ETH") if c_ in zv]
        zf = vi.zscore_frame(zv, zcols)
        chart(vlc.norm_overlay(zf, f"Do VIX and DVOL trend together? (hourly, scaled over the last {span})", bh),
              "norm_overlay", vx.norm_takeaway(zf, asset, span), vx.HOW["norm"])
    with t3:
        freq = st.radio("Bar", ["5-min", "Hourly"], horizontal=True, key="vl_scatter_bar")
        if freq == "5-min" and "fit5" in B:
            chart(vlc.change_scatter(B["dx5"], B["dy5"], B["fit5"], asset, "5-min"), "scatter",
                  vx.scatter_takeaway(B["fit5"], asset, "5-min"), vx.HOW["scatter"])
        elif "fith" in B:
            chart(vlc.change_scatter(B["dxh"], B["dyh"], B["fith"], asset, "Hourly"), "scatter",
                  vx.scatter_takeaway(B["fith"], asset, "Hourly"), vx.HOW["scatter"])
    with t4:
        if "rollh" in B:
            chart(vlc.rolling_intraday(B["rollh"], asset), "intraday_rolling", vx.rolling_takeaway(B["rollh"], asset),
                  vx.HOW["rolling_i"])
    with t5:
        if "ev5" in B:
            up = B["ev5"]["VIX jumps up"]
            dn = B["ev5"]["VIX drops"]
            tk = vx.event_takeaway(up, "big 5-min VIX jumps", asset, (0, 30, 120), "min") + " " + \
                vx.event_takeaway(dn, "big 5-min VIX drops", asset, (0, 30, 120), "min")
            chart(vlc.intraday_events(B["ev5"], asset), "intraday_events", tk, vx.HOW["events_i"])

    if "sp" in B and not B["sp"].empty:
        st.subheader(f"Does the morning VIX move (10:00 → 12:00 New York) predict {asset} DVOL later?")
        st.markdown(f"**→** {_md(vx.session_takeaway(B['sp'], asset))}")
        spt = vx.session_plain(B["sp"])
        table(spt, {"DVOL move per 1-pt morning VIX move": "{:+.2f}", "Forecast gain % vs DVOL-only": "{:+.1f}",
                    "Direction right %": "{:.0f}"}, ["Significant in-sample?", "Beats DVOL-only out-of-sample?"],
              how=vx.HOW["session"])
    st.caption("Intraday caveat: DVOL is an index of Deribit option prices. A short VIX lead can partly reflect "
               "Deribit quotes catching up to US moves, so it is only tradeable if the options themselves lag.")

# ------------------------------------------------------------------ daily
st.header("📅 Daily")
st.subheader("Levels and the crypto vol premium")
chart(vlc.levels(df, asset), "levels", vx.levels_takeaway(df, asset), vx.HOW["levels"])
z = {k: vl.rolling_z(full[k], FV_WIN).iloc[-1] for k in ("SPREAD_VIX", "RATIO_VIX", "RATIO_MOVE") if k in df}
p = {k: vl.window_pct(df[k]) for k in z}
c = st.columns(3)
for i, (k, lab, fmt) in enumerate((("SPREAD_VIX", "DVOL − VIX", "{:+.1f} pts"), ("RATIO_VIX", "DVOL / VIX", "{:.2f}x"),
                                   ("RATIO_MOVE", "DVOL / MOVE", "{:.3f}x"))):
    if k in df:
        c[i].metric(lab, fmt.format(df[k].iloc[-1]), f"z {z[k]:+.2f}, {vl.ordinal(p[k])} pct ({lookback})"
                    if np.isfinite(z[k]) else None, delta_color="off")
chart(vlc.spread_ratio(df, asset), "spreads", vx.spread_takeaway(df, asset), vx.HOW["spreads"])

st.subheader(f"Does {pred_key} lead {asset} DVOL day to day?")
chart(vlc.cross_corr(A["cc"], asset, pred_key), "leadlag", vx.leadlag_takeaway(A["cc"], "d", asset, pred_key),
      vx.HOW["leadlagd"])

st.subheader("What did DVOL do after TradFi vol spikes?")
uncond = A["uncond"]
for k, evs in A["events"].items():
    paths = {f"{k} 1-day jump > 2 sd": evs["spike"][1],
             f"{k} crosses above {25 if k == 'VIX' else 130}": evs["cross"][1]}
    tk = vx.event_takeaway(evs["spike"][1], f"{k} jump days", asset, (0, 5, 20), "d")
    chart(vlc.event_paths(paths, f"What does {asset} DVOL do after a {k} spike?", uncond, asset), f"events_{k.lower()}",
          tk, vx.HOW["events_d"])

st.subheader("Is the link stable over time?")
chart(vlc.rolling(A["roll"], asset, pred_key, A.get("roll_w", 63)), "rolling",
      vx.rolling_daily_takeaway(A["roll"], asset, pred_key), vx.HOW["rolling_d"])

# ------------------------------------------------------------------ deeper stats
st.header("🔬 Deeper stats")
st.subheader(f"Granger test, daily changes ({pred_key})")
st.markdown(f"**→** {_md(vx.granger_takeaway(A['gr_pred_fwd'], A['gr_pred_rev'], 'd', asset, pred_key))}")
gd = vx.granger_table(A["gr_pred_fwd"], A["gr_pred_rev"], "d", pred_key)
if pred_key != "VIX":
    gv = vx.granger_table(A["gr_fwd"], A["gr_rev"], "d", "VIX")
    gd = pd.concat([gd.drop(columns=["Observations"]), gv.drop(columns=["Past moves included"])], axis=1)
table(gd, None, [c_ for c_ in gd.columns if "?" in c_], how=vx.HOW["granger"])

st.subheader(f"Does today's VIX move forecast {asset} DVOL over the next days?")
st.markdown(f"**→** {_md(vx.regression_takeaway(A['vix_reg'], asset))}")
regp = vx.regression_plain(A["vix_reg"])
if not regp.empty:
    table(regp, {"DVOL move per 1-pt VIX move today": "{:+.3f}", "Forecast gain vs DVOL-only %": "{:+.1f}"},
          ["Significant in-sample?", "Beats DVOL-only out-of-sample?"], how=vx.HOW["regression"])
    with st.expander("Full regression output"):
        rows = []
        for res in A["vix_reg"]:
            if not res.get("ok"):
                continue
            rows.append({"h (days)": res["h"], "β ΔVIX": res["coef"]["dVIX"], "t ΔVIX": res["t"]["dVIX"],
                         "β VIX level": res["coef"]["VIX_lvl"], "t VIX level": res["t"]["VIX_lvl"],
                         "β DVOL−VIX": res["coef"]["SPREAD_VIX"], "t DVOL−VIX": res["t"]["SPREAD_VIX"],
                         "β ΔDVOL (own)": res["coef"]["dDVOL"], "R²": res["r2"], "R² DVOL-only": res["r2_base"],
                         "OOS R² vs DVOL-only %": 100 * res["oos_r2_vs_base"],
                         "OOS R² vs no-change %": 100 * res["oos_r2_vs_rw"], "CW p": res["cw_p"], "n": res["n"]})
        reg = pd.DataFrame(rows)
        st.dataframe(reg.style.format({c_: "{:+.3f}" for c_ in reg.columns if c_.startswith("β")}
                                      | {c_: "{:+.2f}" for c_ in reg.columns if c_.startswith("t ")}
                                      | {"R²": "{:.3f}", "R² DVOL-only": "{:.3f}", "OOS R² vs DVOL-only %": "{:+.1f}",
                                         "OOS R² vs no-change %": "{:+.1f}", "CW p": "{:.3f}"}),
                     hide_index=True, width="stretch")
        st.caption("Controls: today's DVOL change and level. t-stats are Newey-West HAC (overlapping horizons). "
                   "Out-of-sample R² uses expanding-window forecasts made only with data known at the time.")

st.subheader("Which TradFi gauge adds the most forecasting power?")
spec_choice = st.radio("Information added", ["Daily change only", "Change + level"], horizontal=True,
                       key="vl_rank_spec", help="Change only = the day's news (1 extra parameter). Change + level "
                       "also adds where the gauge sits (2 parameters); more parameters cost more estimation noise.")
rank = A["rank"] if spec_choice == "Daily change only" else A["rank_full"]
if not rank.empty:
    chart(vlc.ranking_heatmap(rank, asset, "daily change" if spec_choice == "Daily change only" else "change and level"),
          "ranking", vx.ranking_takeaway(rank, asset), vx.HOW["ranking"])
    rp = vx.ranking_plain(rank)
    gain_cols = [c_ for c_ in rp.columns if c_.endswith("d gain %")] + ["Best forecast gain %"]
    rp = rp[["Predictor", "Improves forecast?", "Best forecast gain %", "Best at", "Significance (best)"]
            + [c_ for c_ in rp.columns if c_.endswith("d gain %")]]
    table(rp, {c_: "{:+.2f}" for c_ in gain_cols}, ["Improves forecast?", "Significance (best)"],
          mask=vx.ranking_sig_mask(rank, rp))
    st.caption("Sorted by usefulness. Gains are out-of-sample R² improvements (%) vs a DVOL-only model; green cells "
               "are significant (Clark-West p < 0.05). The verdict uses the change-only version.")

st.divider()
st.caption("Research view, not trading advice. MOVE is in bp of Treasury yield vol, so DVOL/MOVE is a relative "
           "gauge, not a vol spread. FRED only serves the last ≈3 years of the ICE BofA HY spread.")

# ------------------------------------------------------------------ blast send
if _blast_step is not None:
    import html as _html
    _want = {k.rsplit(".", 1)[-1] for k in blast.selected(_blast_step)}
    _titles = {"rich_cheap": "Rich/cheap vs TradFi", "rich_cheap_q": "Rich/cheap quintiles",
               "levels": "DVOL vs VIX & MOVE", "spreads": "Spreads & ratios", "leadlag": "Lead-lag",
               "events_vix": "After VIX spikes", "events_move": "After MOVE spikes", "rolling": "Rolling beta",
               "ranking": "Predictor ranking", "today": "Today vs fair value (5-min)",
               "intraday_overlay": "Intraday overlay (5-min)", "norm_overlay": "Normalised overlay (hourly)",
               "scatter": "Scatter of changes", "intraday_rolling": "Rolling 1w/1m corr & beta",
               "intraday_events": "After big 5-min VIX moves", "intraday_leadlag": "Intraday lead-lag (5-min)",
               "intraday_leadlag_h": "Intraday lead-lag (hourly)"}

    def _cap(k, tk):
        head = f"TradFi vol vs {asset} DVOL — {_titles.get(k, k)}"
        if k == "rich_cheap" and not last.empty:
            tk = vl.rich_cheap_headline(asset, rc_full, pct_score, "the last " + lookback)
        return head + (f"\n{_html.escape(tk)}" if tk else "")

    _pairs = [(f, _cap(k, tk)) for k, f, tk in _SHOWN
              if k in _want or (k == "rich_cheap_q" and "rich_cheap" in _want)
              or (k.startswith("events_") and "events" in _want)
              or (k == "intraday_leadlag_h" and "intraday_leadlag" in _want)]
    with st.spinner("Sending to Telegram…"):
        _sent, _failed = blast.send_figs(_pairs, light_theme=True)
    blast.finish(_blast_step, _sent, _failed)
