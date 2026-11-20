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

# Streamlit Cloud keeps old lib modules in memory after a deploy.
_RELOADED = freshness.reload_if_changed(fx_style, vld, vl, vi, vlc) + freshness.reload_if_changed(blast)

st.set_page_config(page_title="TradFi Vol vs DVOL", page_icon="📉", layout="wide",
                   initial_sidebar_state="expanded")
st.title("📉 TradFi vol vs crypto vol (VIX → DVOL)")
st.caption("Does traditional-market vol lead BTC / ETH DVOL, and is crypto vol rich or cheap vs TradFi vol? "
           "Sources: Deribit DVOL, Cboe (VIX, VVIX, VIX9D, VIX3M), Yahoo (MOVE, DXY, Nasdaq-100), "
           "FRED (HY OAS). One row per US trading day; DVOL taken at the 16:00 New York close, so a Monday "
           "change spans Friday → Monday and includes the weekend.")

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


def show(fig, key: str):
    if _blast_step is not None:
        _SHOWN.append((key, fig))
    st.plotly_chart(fig, theme=None, width="stretch", config={"displaylogo": False})


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

# ------------------------------------------------------------------ verdict
st.subheader(f"Verdict: does TradFi vol predict {asset} DVOL?")
intra_lines = vi.verdict_lines(asset, B.get("cc5"), B.get("gr5"), B.get("gr5_rev"), B.get("cch"), B.get("sp"))
lines = vl.verdict(asset, A["cc_vix"], A["gr_fwd"], A["gr_rev"], A["rank"])
sig_rc = A["rc_sig"]
if not sig_rc.empty:
    d = sig_rc[sig_rc["target"] == "DVOL"]
    if not d.empty:
        neg = d[(d["slope"] < 0) & (d["t"].abs() > 1.96)]
        best = d.sort_values("oos_r2_pct", ascending=False).iloc[0]
        if not neg.empty:
            lines.append(
                f"Rich/cheap: a rich score is followed by falling {asset} DVOL in-sample "
                f"(HAC t {', '.join(f'{t:.1f} at {h}d' for t, h in zip(neg['t'], neg['h']))}); real-time "
                f"out-of-sample R² is {best['oos_r2_pct']:+.1f}% at {int(best['h'])}d "
                f"(CW p={best['cw_p']:.2f}).")
        else:
            lines.append(f"Rich/cheap: the score does not significantly predict {asset} DVOL changes "
                         f"(best out-of-sample R² {best['oos_r2_pct']:+.1f}% at {int(best['h'])}d).")
st.info(f"**Intraday (real time, last {lookback} where data allows)**\n"
        + "\n".join(f"- {ln}" for ln in intra_lines)
        + f"\n\n**Daily (last {lookback})**\n" + "\n".join(f"- {ln}" for ln in lines))

# ------------------------------------------------------------------ rich / cheap
st.subheader(f"💸 Is {asset} vol rich or cheap vs TradFi vol?")
last = rc_full.dropna(subset=["score"])
if last.empty:
    st.warning("Not enough history for the rich/cheap model.")
else:
    r = last.iloc[-1]
    rcw = rc_full.reindex(df.index)
    pct_score = vl.window_pct(rcw["score"])
    pct_rv = vl.window_pct(rcw["ratio_vix"])
    pct_rm = vl.window_pct(rcw["ratio_move"]) if "ratio_move" in rcw else np.nan
    st.markdown(f"**{vl.rich_cheap_headline(asset, rc_full, pct_score, 'the last ' + lookback)}**  \n"
                f"<span style='color:#888'>Reading for {last.index[-1]:%a %d %b %Y} New York close.</span>",
                unsafe_allow_html=True)
    c = st.columns(5)
    c[0].metric(f"{asset} DVOL", f"{full['DVOL'].iloc[-1]:.1f}")
    c[1].metric("TradFi fair value", f"{r['fair']:.1f}", f"{r['resid']:+.1f} vols", delta_color="off",
                help=f"Rolling {fv_label} regression of DVOL on VIX and MOVE (fit R² {r['fv_r2']:.2f}).")
    c[2].metric("Composite score", f"{r['score']:+.2f} sd", help="Mean of the three component z-scores.")
    c[3].metric(f"Score percentile ({lookback})", vl.ordinal(pct_score) if np.isfinite(pct_score) else "—")
    c[4].metric("DVOL / VIX", f"{r['ratio_vix']:.2f}x",
                f"{vl.ordinal(pct_rv)} pct ({lookback})" if np.isfinite(pct_rv) else None, delta_color="off")
    def _f(v, fmt):
        return fmt.format(v) if v is not None and np.isfinite(v) else "—"

    _rows = [("Fair-value residual (vols)", r["resid"], r["resid_z"], vl.window_pct(rcw["resid"])),
             ("DVOL / VIX", r["ratio_vix"], r["ratio_vix_z"], pct_rv)]
    if "ratio_move" in r:
        _rows.append(("DVOL / MOVE", r["ratio_move"], r["ratio_move_z"], pct_rm))
    comp = pd.DataFrame([{"Component": n, "Reading": _f(a, "{:+.2f}" if i == 0 else "{:.3f}"),
                          f"z ({fv_label})": _f(b, "{:+.2f}"), f"Percentile ({lookback})": _f(c1, "{:.0f}")}
                         for i, (n, a, b, c1) in enumerate(_rows)])
    st.dataframe(comp, hide_index=True, width="stretch")
    show(vlc.rich_cheap(df, rc, asset), "rich_cheap")
    cq1, cq2 = st.columns([3, 2])
    with cq1:
        if not A["rc_q"].empty:
            show(vlc.quintile_bars(A["rc_q"], asset), "rich_cheap_q")
    with cq2:
        st.markdown("**Does the score predict DVOL?**")
        if not sig_rc.empty:
            st.dataframe(sig_rc[["target", "h", "slope", "t", "r2", "oos_r2_pct", "cw_p"]].rename(columns={
                "h": "Days", "slope": "Slope (pts per sd)", "t": "HAC t", "r2": "R²",
                "oos_r2_pct": "OOS R² %", "cw_p": "CW p"}).style.format(
                {"Slope (pts per sd)": "{:+.2f}", "HAC t": "{:+.2f}", "R²": "{:.3f}", "OOS R² %": "{:+.1f}",
                 "CW p": "{:.3f}"}), hide_index=True, width="stretch")
            st.caption("Negative slope = rich crypto vol tends to cheapen. OOS R² vs a historical-mean "
                       "forecast, expanding window; CW = Clark-West one-sided p.")
    # ---- today: intraday DVOL vs fair value (5-min)
    coef = rc_full.attrs.get("coef", {})
    if not M5.empty and coef:
        day = vi.last_sessions(M5, 1)
        move_last = float(full["MOVE"].dropna().iloc[-1]) if "MOVE" in full and full["MOVE"].notna().any() else None
        fvi = vi.intraday_fair(day, asset, coef, move_last).dropna()
        if not fvi.empty:
            dlab = f"{fvi.index[-1].tz_convert('America/New_York'):%a %d %b} NY session, last bar " \
                   f"{fvi.index[-1].tz_convert('Asia/Singapore'):%H:%M} SGT"
            tcols = st.columns(3)
            tcols[0].metric(f"{asset} DVOL (latest 5-min)", f"{fvi['DVOL'].iloc[-1]:.1f}")
            tcols[1].metric("Fair value now", f"{fvi['fair'].iloc[-1]:.1f}", f"{fvi['resid'].iloc[-1]:+.1f} vols",
                            delta_color="off")
            tcols[2].metric("Intraday range of rich/cheap", f"{fvi['resid'].min():+.1f} → {fvi['resid'].max():+.1f}")
            show(vlc.today_fair(fvi, asset, dlab), "today")

# ------------------------------------------------------------------ short horizon
st.subheader("⚡ Short-horizon: VIX vs DVOL intraday")
if M5.empty and H1.empty:
    st.warning("No intraday data available right now.")
else:
    t1, t2, t3, t4, t5 = st.tabs(["Intraday overlay", "Normalised overlay", "Scatter of changes",
                                  "Rolling corr & beta", "After big VIX moves"])
    b5, bh = pd.Timedelta(minutes=5), pd.Timedelta(hours=1)
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
            show(vlc.intraday_overlay(view, asset, b5, f"{asset} DVOL vs VIX, 5-min ({pick.lower()})"),
                 "intraday_overlay")
    with t2:
        span = st.radio("Window", ["1 month", "3 months"], horizontal=True, key="vl_norm_span")
        src = H1 if not H1.empty else M5
        zv = vi.window(src, 31 if span == "1 month" else 92)
        zcols = [c_ for c_ in ("VIX", "DVOL_BTC", "DVOL_ETH") if c_ in zv]
        show(vlc.norm_overlay(vi.zscore_frame(zv, zcols), f"VIX vs DVOL, hourly, z-scored over {span}", bh),
             "norm_overlay")
    with t3:
        freq = st.radio("Bar", ["5-min", "Hourly"], horizontal=True, key="vl_scatter_bar")
        if freq == "5-min" and "fit5" in B:
            show(vlc.change_scatter(B["dx5"], B["dy5"], B["fit5"], asset, "5-min"), "scatter")
        elif "fith" in B:
            show(vlc.change_scatter(B["dxh"], B["dyh"], B["fith"], asset, "Hourly"), "scatter")
    with t4:
        if "rollh" in B:
            show(vlc.rolling_intraday(B["rollh"], asset), "intraday_rolling")
    with t5:
        if "ev5" in B:
            show(vlc.intraday_events(B["ev5"], asset), "intraday_events")
            pu = B["ev5"]["VIX jumps up"].set_index("minutes")
            if pu["n"].max() > 0:
                st.caption(f"{int(pu['n'].max())} VIX jumps. Average {asset} DVOL move vs the bar before: "
                           f"{pu.loc[0, 'mean']:+.2f} in the jump bar, {pu.loc[30, 'mean']:+.2f} after 30 min, "
                           f"{pu.loc[120, 'mean']:+.2f} after 2 hours.")

st.subheader("⏱️ Intraday lead-lag, Granger and session forecasts")
lc1, lc2 = st.columns(2)
with lc1:
    if "cc5" in B:
        show(vlc.intraday_cross_corr(B["cc5"], asset, "min"), "intraday_leadlag")
with lc2:
    if "cch" in B:
        show(vlc.intraday_cross_corr(B["cch"], asset, "h"), "intraday_leadlag_h")
gcols = st.columns(2)
if "gr5" in B:
    g5 = pd.DataFrame({"Lags (min)": B["gr5"]["minutes"], "VIX → DVOL p": B["gr5"]["p_value"],
                       "DVOL → VIX p": B["gr5_rev"]["p_value"], "n": B["gr5"]["n"]})
    gcols[0].markdown("**5-min Granger**")
    gcols[0].dataframe(g5.style.format({"VIX → DVOL p": "{:.4f}", "DVOL → VIX p": "{:.4f}"}), hide_index=True,
                       width="stretch")
if "grh" in B:
    gh = pd.DataFrame({"Lags (h)": B["grh"]["minutes"] // 60, "VIX → DVOL p": B["grh"]["p_value"],
                       "DVOL → VIX p": B["grh_rev"]["p_value"], "n": B["grh"]["n"]})
    gcols[1].markdown("**Hourly Granger**")
    gcols[1].dataframe(gh.style.format({"VIX → DVOL p": "{:.4f}", "DVOL → VIX p": "{:.4f}"}), hide_index=True,
                       width="stretch")
if "sp" in B and not B["sp"].empty:
    st.markdown(f"**Does the morning VIX move (10:00 → 12:00 New York) predict {asset} DVOL later?**")
    sp = B["sp"].rename(columns={"target": "DVOL change", "beta_vix": "β ΔVIX", "t_vix": "HAC t ΔVIX",
                                 "beta_own": "β own ΔDVOL", "t_own": "HAC t own", "r2": "R²",
                                 "oos_r2_pct": "OOS R² % (vs mean)", "oos_r2_vs_own_pct": "OOS R² % (vs DVOL-only)",
                                 "cw_p": "CW p", "same_sign_pct": "Same sign %", "n": "Sessions"})
    st.dataframe(sp.drop(columns=["n_oos"]).style.format(
        {"β ΔVIX": "{:+.3f}", "HAC t ΔVIX": "{:+.2f}", "β own ΔDVOL": "{:+.3f}", "HAC t own": "{:+.2f}", "R²": "{:.3f}",
         "OOS R² % (vs mean)": "{:+.1f}", "OOS R² % (vs DVOL-only)": "{:+.1f}", "CW p": "{:.3f}",
         "Same sign %": "{:.0f}"}), hide_index=True, width="stretch")
    st.caption("One row per session, hourly bars. Controls for DVOL's own morning move. Out-of-sample uses "
               "an expanding window over sessions.")
st.caption("Intraday caveat: DVOL is an index of Deribit option prices. A short VIX lead can partly reflect "
           "Deribit quotes catching up to US moves, so it is only tradeable if the options themselves lag.")

# ------------------------------------------------------------------ 1. levels
st.subheader("📅 Daily analysis")
st.subheader("1 · Levels, spreads and ratios")
show(vlc.levels(df, asset), "levels")
z = {k: vl.rolling_z(full[k], FV_WIN).iloc[-1] for k in ("SPREAD_VIX", "RATIO_VIX", "RATIO_MOVE") if k in df}
p = {k: vl.window_pct(df[k]) for k in z}
c = st.columns(3)
for i, (k, lab, fmt) in enumerate((("SPREAD_VIX", "DVOL − VIX", "{:+.1f} pts"), ("RATIO_VIX", "DVOL / VIX", "{:.2f}x"),
                                   ("RATIO_MOVE", "DVOL / MOVE", "{:.3f}x"))):
    if k in df:
        c[i].metric(lab, fmt.format(df[k].iloc[-1]), f"z {z[k]:+.2f}, {vl.ordinal(p[k])} pct ({lookback})"
                    if np.isfinite(z[k]) else None, delta_color="off")
show(vlc.spread_ratio(df, asset), "spreads")

# ------------------------------------------------------------------ 2. lead-lag
st.subheader(f"2 · Lead-lag: does Δ{pred_key} lead Δ{asset} DVOL?")
show(vlc.cross_corr(A["cc"], asset, pred_key), "leadlag")

# ------------------------------------------------------------------ 3. granger
st.subheader("3 · Granger causality (daily changes, lags 1–5)")
g = pd.DataFrame({"Lags": A["gr_pred_fwd"]["lag"],
                  f"{pred_key} → DVOL p": A["gr_pred_fwd"]["p_value"],
                  f"DVOL → {pred_key} p": A["gr_pred_rev"]["p_value"]})
if pred_key != "VIX":
    g["VIX → DVOL p"] = A["gr_fwd"]["p_value"]
    g["DVOL → VIX p"] = A["gr_rev"]["p_value"]
st.dataframe(g.style.format({c_: "{:.4f}" for c_ in g.columns if c_ != "Lags"})
             .map(lambda v: "background-color: rgba(14,122,56,0.15)" if isinstance(v, float) and v < 0.05 else "",
                  subset=[c_ for c_ in g.columns if c_ != "Lags"]), hide_index=True, width="stretch")
st.caption("p < 0.05 (green): past changes of the first series help explain today's change of the second, "
           "beyond its own past. Both directions are shown; feedback both ways is common.")

# ------------------------------------------------------------------ 4. predictive regression
st.subheader(f"4 · Predictive regressions: {asset} DVOL change over the next h days on today's VIX")
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
if rows:
    reg = pd.DataFrame(rows)
    st.dataframe(reg.style.format({c_: "{:+.3f}" for c_ in reg.columns if c_.startswith("β")}
                                  | {c_: "{:+.2f}" for c_ in reg.columns if c_.startswith("t ")}
                                  | {"R²": "{:.3f}", "R² DVOL-only": "{:.3f}", "OOS R² vs DVOL-only %": "{:+.1f}",
                                     "OOS R² vs no-change %": "{:+.1f}", "CW p": "{:.3f}"}),
                 hide_index=True, width="stretch")
    st.caption("Controls: today's DVOL change and level. t-stats are Newey-West HAC (overlapping horizons). "
               "Out-of-sample R² uses expanding-window forecasts made only with data known at the time.")

# ------------------------------------------------------------------ 5. events
st.subheader("5 · Event study: what DVOL did after TradFi vol spikes")
uncond = A["uncond"]
for k, evs in A["events"].items():
    paths = {f"{k} 1-day jump > 2 sd": evs["spike"][1],
             f"{k} crosses above {25 if k == 'VIX' else 130}": evs["cross"][1]}
    show(vlc.event_paths(paths, f"{asset} DVOL around {k} spikes", uncond), f"events_{k.lower()}")
    e = evs["spike"][0]
    pth = evs["spike"][1].set_index("offset")
    if not e.empty:
        def _at(o):
            return pth.loc[o, "mean"] if o in pth.index else np.nan
        st.caption(f"{len(e)} {k} jump days. Average {asset} DVOL move: day 0 {_at(0):+.1f}, "
                   f"day 5 {_at(5):+.1f}, day 20 {_at(20):+.1f} vol pts (vs day −1). After day 0 the extra "
                   f"move is {_at(20) - _at(0):+.1f} pts by day 20.")

# ------------------------------------------------------------------ 6. rolling
st.subheader("6 · Regime dependence")
show(vlc.rolling(A["roll"], asset, pred_key), "rolling")

# ------------------------------------------------------------------ 7. ranking
st.subheader("7 · Ranking: which TradFi gauge adds the most out-of-sample power?")
spec_choice = st.radio("Predictor information added", ["Daily change only", "Change + level"], horizontal=True,
                       key="vl_rank_spec", help="Change only = the day's news (1 extra parameter). Change + level "
                       "also adds where the gauge sits (2 parameters); more parameters cost more estimation noise.")
rank = A["rank"] if spec_choice == "Daily change only" else A["rank_full"]
if not rank.empty:
    show(vlc.ranking_heatmap(rank, asset, "daily change" if spec_choice == "Daily change only" else "change and level"),
         "ranking")
    tbl = rank.pivot_table(index="label", columns="h", values="oos_r2_gain_pct")
    tbl.columns = [f"{h}d OOS gain %" for h in tbl.columns]
    pv = rank.pivot_table(index="label", columns="h", values="cw_p")
    pv.columns = [f"{h}d CW p" for h in pv.columns]
    out = pd.concat([tbl, pv], axis=1)
    out = out[[c_ for h in vl.HORIZONS for c_ in (f"{h}d OOS gain %", f"{h}d CW p") if c_ in out]]
    out = out.sort_values(out.columns[0], ascending=False)
    st.dataframe(out.style.format("{:+.2f}"), width="stretch")
    st.caption("Baseline: DVOL's own daily change and level. Positive = better real-time forecasts than DVOL alone. "
               "CW p < 0.05 = the predictor's forecasts beat the baseline after adjusting for estimation noise "
               "(Clark-West), even when the raw OOS R² gain is tiny. The verdict uses the change-only ranking.")

st.divider()
st.caption("Research view, not trading advice. MOVE is in bp of Treasury yield vol, so DVOL/MOVE is a relative "
           "gauge, not a vol spread. FRED only serves the last ~3 years of the ICE BofA HY spread.")

# ------------------------------------------------------------------ blast send
if _blast_step is not None:
    _want = {k.rsplit(".", 1)[-1] for k in blast.selected(_blast_step)}
    _titles = {"rich_cheap": "Rich/cheap vs TradFi", "rich_cheap_q": "Rich/cheap quintiles",
               "levels": "DVOL vs VIX & MOVE", "spreads": "Spreads & ratios", "leadlag": "Lead-lag",
               "events_vix": "After VIX spikes", "events_move": "After MOVE spikes", "rolling": "Rolling beta",
               "ranking": "Predictor ranking", "today": "Today vs fair value (5-min)",
               "intraday_overlay": "Intraday overlay (5-min)", "norm_overlay": "Normalised overlay (hourly)",
               "scatter": "Scatter of changes", "intraday_rolling": "Rolling 1w/1m corr & beta",
               "intraday_events": "After big 5-min VIX moves", "intraday_leadlag": "Intraday lead-lag (5-min)",
               "intraday_leadlag_h": "Intraday lead-lag (hourly)"}
    _pairs = [(f, f"TradFi vol vs {asset} DVOL — {_titles.get(k, k)}") for k, f in _SHOWN
              if k in _want or (k == "rich_cheap_q" and "rich_cheap" in _want)
              or (k.startswith("events_") and "events" in _want)
              or (k == "intraday_leadlag_h" and "intraday_leadlag" in _want)]
    with st.spinner("Sending to Telegram…"):
        _sent, _failed = blast.send_figs(_pairs, light_theme=True)
    blast.finish(_blast_step, _sent, _failed)
