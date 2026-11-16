"""
IBIT — iShares Bitcoin Trust options vs Deribit BTC options.

Chain analytics from Cboe's free ~15-min-delayed option-chain snapshot
(see lib/ibit.py for scope, the "Deribit-only" exception and the maths).
The headline view is a like-for-like Deribit-vs-IBIT comparison of the vol
surface, skew, term structure, open interest and market-implied distribution;
the remaining tabs go deep on IBIT alone (surface, positioning, gamma, max
pain, activity, large tickets) and expose the raw data.

The large-tickets view is still not a trade tape.  Cboe does not publish
prints.  Open-interest change uses only daily closes this app has recorded;
unusual volume, last-print side and the Deribit comparison are estimates
built from the snapshot.

IBIT only trades in US hours, so outside them the snapshot is the last close
while Deribit is live -- the banner at the top says how old the data is.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import html

from lib import cache as cache_lib
from lib import fx_style
from lib import ibit as L
from lib.constants import TTL_DAILY_EXTERNAL, TTL_LONG, TTL_MEDIUM, TTL_SLOW

st.set_page_config(page_title="IBIT", page_icon="🪙", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Look & feel — same chrome as the MCM Bot page (CSS, tile badges, pills,
# sidebar chart-theme selector, one-view-at-a-time radio).
# ---------------------------------------------------------------------------

IB, DER = "#3790C7", "#f7931a"           # IBIT blue (house blue), BTC orange
CALL, PUT, VIO, MUTED = "#22a35a", "#e5574a", "#8e6fd8", "#8B95A5"

st.markdown("""
<style>
.block-container{padding-top:2.2rem;max-width:1500px}
h1{letter-spacing:-0.02em;font-weight:800}
div[data-testid="stButton"]>button{border-radius:10px;font-weight:600;transition:all .15s ease}
div[data-testid="stButton"]>button:hover{transform:translateY(-1px);
  box-shadow:0 3px 10px rgba(55,144,199,.28);border-color:#3790C7}
div[data-testid="stDataFrame"]{border:1px solid rgba(128,128,128,.25);border-radius:12px;overflow:hidden}
div[data-testid="stPlotlyChart"]{border:1px solid rgba(128,128,128,.2);border-radius:12px}
div[data-testid="stExpander"]{border-radius:12px}
.mcm-tile{display:flex;align-items:center;gap:.55rem;margin:1.1rem 0 .4rem}
.mcm-tile .a{font-size:.68rem;font-weight:800;letter-spacing:.09em;padding:.14rem .5rem;
  border-radius:6px;color:#fff}
.mcm-tile .c{font-size:1rem;font-weight:700;letter-spacing:-0.01em}
.mcm-pills{display:flex;flex-wrap:wrap;gap:.45rem;margin:.15rem 0 .75rem}
.mcm-pill{border:1px solid rgba(128,128,128,.3);border-radius:999px;padding:.24rem .75rem;
  font-size:.88rem;background:rgba(128,128,128,.08);white-space:nowrap}
.mcm-pill b{opacity:.6;font-weight:700;margin-right:.4rem;font-size:.72rem;letter-spacing:.06em}
.mcm-pill.lead{font-weight:700}
.mcm-pill .pos{color:#22a35a;font-weight:700}
.mcm-pill .neg{color:#e5574a;font-weight:700}
</style>
""", unsafe_allow_html=True)

_cur_badge = "IBIT×BTC"


def sec(title: str, sub: str = "", color: str = IB, badge: str | None = None) -> None:
    """Coloured badge + report name (MCM Bot tile label), optional caption."""
    b = badge or _cur_badge
    st.markdown(f'<div class="mcm-tile"><span class="a" style="background:{color}">'
                f'{html.escape(b)}</span><span class="c">{html.escape(title)}</span></div>',
                unsafe_allow_html=True)
    if sub:
        st.caption(sub)


def kpis(items: list[tuple]) -> None:
    """MCM Bot header pills. items: (label, value, sub, color[, direction])."""
    ps = []
    for it in items:
        lab, val, sub = it[0], str(it[1]), it[2]
        d = it[4] if len(it) > 4 else ""
        v = html.escape(val)
        if d:
            v = f'<span class="{"pos" if d == "up" else "neg"}">{v}</span>'
        ps.append(f'<span class="mcm-pill" title="{html.escape(sub)}"><b>{html.escape(lab.upper())}</b>{v}</span>')
    st.markdown(f'<div class="mcm-pills">{"".join(ps)}</div>', unsafe_allow_html=True)


def _chart_theme() -> str:
    t = str(st.session_state.get("ibit_chart_theme") or "Auto").lower()
    return t if t in fx_style.CHART_THEMES else "auto"


def sty(fig: go.Figure, h: int = 420, title: str | None = None,
        xt: str | None = None, yt: str | None = None) -> go.Figure:
    fig.update_layout(
        height=h, margin=dict(l=50, r=20, t=50 if title else 30, b=45),
        title=dict(text=title, x=0.01, font=dict(size=15)) if title else None,
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        hovermode="x unified")
    fig.update_xaxes(zeroline=False, title=xt)
    fig.update_yaxes(zeroline=False, title=yt)
    return fig


def pct(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def usd(x):
    if x is None or not np.isfinite(x):
        return "–"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"${x / div:.2f}{suf}"
    return f"${x:,.0f}"


def susd(x):
    """``usd`` with the sign in front of the dollar (``-$23.61M``)."""
    if x is None or not np.isfinite(x):
        return "–"
    return ("-" if x < 0 else "") + usd(abs(float(x)))


def show(fig, key):
    th = _chart_theme()
    fx_style.apply_chart_theme(fig, th)
    fx_style.add_watermark(fig)
    st.plotly_chart(fig, theme=fx_style.streamlit_chart_theme(th), width="stretch", key=key,
                    config={"displaylogo": False})


def _share_phrase(info: dict) -> str:
    """One line naming where the BTC-per-share ratio came from."""
    if info.get("fallback"):
        return (f"iShares holdings fallback as of {info['asof']} "
                "(live file unavailable)")
    return f"iShares holdings as of {info['asof']}"


def _ticket_name(row) -> str:
    exp = pd.to_datetime(row["expiry"], utc=True, errors="coerce")
    when = exp.strftime("%d %b %y") if pd.notna(exp) else "?"
    strike = row["strike"]
    k = f"{float(strike):g}" if pd.notna(strike) and np.isfinite(strike) else "?"
    return f"{when} {k}{row['type']}"


def _oi_bars(frame, title, color):
    if frame is None or len(frame) == 0:
        return sty(go.Figure(), 380, title)
    names = frame.apply(_ticket_name, axis=1)
    f = go.Figure(go.Bar(
        y=names[::-1], x=frame["oi_change"][::-1], orientation="h", marker_color=color,
        hovertemplate="%{y}<br>%{x:+,.0f}<extra></extra>"))
    return sty(f, 420, title).update_layout(hovermode="closest")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_ibit():
    return L.parse_cboe(L.fetch_raw("IBIT"), datetime.now(timezone.utc))


@st.cache_data(ttl=TTL_LONG, show_spinner=False)
def load_deribit():
    return L.deribit_chain(datetime.now(timezone.utc))


@st.cache_data(ttl=TTL_DAILY_EXTERNAL, show_spinner=False)
def load_share_ratio():
    # iShares publishes the holdings file once a day.  The fallback inside
    # lib.ibit is used when that fetch fails, and is labelled with its date.
    return L.btc_per_share_ratio()


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_ibit_history():
    # Daily closes.  Same 5-minute remote TTL as the vol snapshots so a close
    # recorded this evening shows up without waiting out an hour-long cache.
    return L.load_chain_history()


@st.cache_data(ttl=TTL_MEDIUM, show_spinner=False)
def load_btc_blocks(start_ms: int, end_ms: int):
    from lib.flow_summary import DERIBIT_MIN_SIZES, deribit_blocks
    return deribit_blocks(int(start_ms), int(end_ms),
                          min_sizes={"BTC": DERIBIT_MIN_SIZES["BTC"]})


@st.cache_data(ttl=TTL_MEDIUM, show_spinner=False)
def load_btc_comparison(session_start_ms: int, session_end_ms: int, now_ms: int):
    """Session blocks when the public tape still has them, else the last 24h.

    Returns ``(payload, used_fallback)``.  A session with trades but no
    size-threshold blocks is not a fallback — the tape covered the window.
    """
    from lib import deribit
    from lib.flow_summary import DERIBIT_MIN_SIZES, deribit_blocks
    mins = {"BTC": DERIBIT_MIN_SIZES["BTC"]}
    primary = deribit_blocks(int(session_start_ms), int(session_end_ms), min_sizes=mins)
    frame = (primary.get("frames") or {}).get("BTC")
    covered = frame is not None and len(frame) > 0
    if not covered:
        probe = deribit.get_last_trades_by_currency_and_time(
            "BTC", kind="option", start_ms=int(session_start_ms), end_ms=int(session_end_ms),
            count=1, sorting="desc", ttl=TTL_MEDIUM)
        covered = bool(probe and probe.get("trades"))
    if covered:
        return primary, False
    fallback = deribit_blocks(int(now_ms) - 24 * 3600 * 1000, int(now_ms), min_sizes=mins)
    return fallback, True


with st.sidebar:
    st.title("🪙 IBIT")
    st.caption("Cboe delayed option chain (free, ~15 min delayed, snapshot) vs Deribit public API")
    st.radio("Chart theme", ["Auto", "Light", "Dark"], index=0, key="ibit_chart_theme", horizontal=True,
             help="Auto follows the app theme (☰ → Settings → Appearance).")
    st.divider()
    cache_lib.render_refresh_button()

st.title("🪙 IBIT")
st.caption("IBIT (iShares Bitcoin Trust) options vs Deribit BTC options · Cboe delayed chain · "
           f"{fx_style.local_now():%d-%b-%Y %H:%M} {fx_style.DISPLAY_TZ_LABEL} time")

with st.expander("What each view shows"):
    st.markdown("""
| View | What it shows |
|------|----------------|
| **Deribit vs IBIT** | Smile at a fixed tenor, ATM / 25Δ risk-reversal / 25Δ butterfly / skew-% term structures, tenor table, side-by-side vol surfaces + difference heatmap, OI by moneyness & tenor, put/call, market-implied distribution. |
| **IBIT surface** | 3D vol surface, smiles by expiry, ATM/25Δ term structure, skew & wings, quote quality. |
| **IBIT positioning** | OI by strike / expiry, volume by expiry, put/call by expiry, strike × expiry OI heatmap, cumulative OI. |
| **Gamma & max pain** | Dealer-gamma proxy by strike and expiry, gamma flip, vega by expiry, max pain per expiry. |
| **Activity** | Most active contracts, vol/OI turnover, volume bubbles and volume by strike. |
| **Large tickets** | Recorded day-over-day OI change, unusual volume, last-print buyer/seller lean, and a snapshot flow summary next to Deribit BTC blocks. Not a trade tape. |
| **Data & notes** | Normalised chains (CSV download) and methodology. |

IBIT trades only in US hours, so outside them the snapshot is the last close while Deribit is live — the age pill shows how stale it is.
""")

_tb1, _tb2, _tb3 = st.columns([1.4, 1.4, 1.2])
with _tb1:
    tenor = st.selectbox("Comparison tenor", options=L.TENORS, index=L.TENORS.index(30),
                         format_func=lambda d: f"{d} DTE", key="ibit_tenor",
                         help="Tenor used for the smile, KPI pills and implied distribution.")
with _tb2:
    kmax = st.selectbox("Strike window", options=[10, 15, 20, 25, 30, 40], index=3,
                        format_func=lambda p: f"±{p}% of forward", key="ibit_kmax",
                        help="Strike range shown on smile / surface charts.") / 100
with _tb3:
    st.write("")
    if st.button("Reload chains", type="primary", width="stretch", key="ibit_reload",
                 help="Re-fetch Cboe and Deribit option chains (clears cached data)."):
        load_ibit.clear()
        load_deribit.clear()
        load_ibit_history.clear()
        load_share_ratio.clear()
        load_btc_blocks.clear()
        load_btc_comparison.clear()
        st.rerun()

try:
    with st.spinner("Fetching IBIT chain from Cboe…"):
        ib, ibm = load_ibit()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load IBIT options from Cboe: {e}")
    st.stop()

der, derm, der_err = None, None, None
try:
    der, derm = load_deribit()
except Exception as e:  # noqa: BLE001
    der_err = str(e)
HAS_D = der is not None and len(der) > 0

now = datetime.now(timezone.utc)
snap = ibm.get("snapshot_utc")
age_h = (pd.Timestamp(now) - snap).total_seconds() / 3600 if snap is not None else np.nan
stale = (not np.isfinite(age_h)) or age_h > 0.6
snap_txt = "n/a"
if snap is not None:
    snap_txt = (f"{snap.tz_convert('America/New_York'):%a %d %b %H:%M} ET · "
                f"{snap.tz_convert('Asia/Singapore'):%H:%M} SGT")
age_txt = "" if not np.isfinite(age_h) else (
    f"{age_h * 60:.0f} min old" if age_h < 2 else f"{age_h:.1f} h old")

kpis([("IBIT", f"${ibm['spot']:.2f}", "IBIT last", IB),
      ("BTC", f"${(derm or {}).get('spot', float('nan')):,.0f}", "Deribit index", DER),
      ("Cboe snapshot", f"{snap_txt} · {age_txt}" + (" · stale" if stale else ""), "delayed ~15 min", IB,
       "dn" if stale else "up"),
      ("IBIT chain", f"{ibm['n_contracts']:,} contracts · {ibm['n_expiries']} expiries", "", IB),
      ("Deribit chain", f"{(derm or {}).get('n_contracts', 0):,} contracts", "", DER)])
if der_err:
    st.warning(f"Deribit chain unavailable ({der_err}) — comparison tabs show IBIT only.")

# ---------------------------------------------------------------------------
# Derived frames
# ---------------------------------------------------------------------------

KG = np.round(np.linspace(-0.40, 0.40, 81), 4)
mI = L.expiry_metrics(ib)
zI = L.surface_grid(ib, KG)
if HAS_D:
    mD = L.expiry_metrics(der)
    zD = L.surface_grid(der, KG)
share_ratio = load_share_ratio()
btc_per_share = float(share_ratio["btc_per_share"])

# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------

def tv(m, col, t):
    return L.at_tenors(m, [col], [t])[col].iloc[0]


atmI, rrI, flyI = tv(mI, "atm", tenor), tv(mI, "rr25", tenor), tv(mI, "fly25", tenor)
k = [("IBIT ATM IV", pct(atmI), f"{tenor}d interpolated · Cboe iv30 {ibm['iv30']:.1f}%", IB),
     ("IBIT 25Δ risk reversal", f"{rrI * 100:+.1f} pts" if np.isfinite(rrI) else "–",
      "calls − puts · negative = put skew", IB, "dn" if rrI < 0 else "up"),
     ("IBIT P/C open interest", f"{ib[ib.type == 'P'].oi.sum() / ib[ib.type == 'C'].oi.sum():.2f}",
      f"{usd(ib.oi_usd.sum())} notional OI", IB)]
if HAS_D:
    atmD, rrD, flyD = tv(mD, "atm", tenor), tv(mD, "rr25", tenor), tv(mD, "fly25", tenor)
    k += [("Deribit ATM IV", pct(atmD), f"{tenor}d interpolated", DER),
          ("Deribit 25Δ risk reversal", f"{rrD * 100:+.1f} pts" if np.isfinite(rrD) else "–",
           "calls − puts", DER, "dn" if rrD < 0 else "up"),
          ("IBIT − Deribit ATM", f"{(atmI - atmD) * 100:+.1f} pts" if np.isfinite(atmI - atmD) else "–",
           "positive = IBIT vol richer", VIO, "up" if atmI > atmD else "dn"),
          ("Deribit P/C open interest",
           f"{der[der.type == 'P'].oi.sum() / der[der.type == 'C'].oi.sum():.2f}",
           f"{usd(der.oi_usd.sum())} notional OI", DER)]
kpis(k)

_VIEWS = ["⚔️ Deribit vs IBIT", "🌋 IBIT surface", "📊 IBIT positioning",
          "🧲 Gamma & max pain", "🔥 Activity", "🧱 Large tickets", "🗂 Data & notes"]
# Only the selected view renders (like MCM Bot).
_view = st.radio("View", _VIEWS, key="ibit_view", horizontal=True, label_visibility="collapsed")
_cur_badge = "IBIT×BTC" if _view == _VIEWS[0] else "IBIT"

# ===========================================================================
# TAB 1 — head to head
# ===========================================================================
if _view == _VIEWS[0]:
    if not HAS_D:
        st.info("Deribit data is required for this tab.")
    else:
        # -- smile -----------------------------------------------------------
        sec("Smile at a constant tenor",
            f"{tenor}-day implied vol by strike (K/F − 1), interpolated between listed expiries")
        sI = L.on_tenor_grid(zI, [tenor]).iloc[0]
        sD = L.on_tenor_grid(zD, [tenor]).iloc[0]
        x = KG * 100
        msk = np.abs(KG) <= kmax
        c1, c2 = st.columns([3, 2])
        with c1:
            f = go.Figure()
            f.add_scatter(x=x[msk], y=sD[msk] * 100, name="Deribit BTC", mode="lines",
                          line=dict(color=DER, width=3), connectgaps=False)
            f.add_scatter(x=x[msk], y=sI[msk] * 100, name="IBIT", mode="lines",
                          line=dict(color=IB, width=3), connectgaps=False,
                          fill="tonexty", fillcolor="rgba(120,140,170,.12)")
            f.add_vline(x=0, line=dict(color=MUTED, dash="dot"))
            show(sty(f, 420, xt="Strike vs forward (%)", yt="Implied vol (%)"), "smile")
        with c2:
            d = (sI - sD) * 100
            f = go.Figure(go.Bar(x=x[msk], y=d[msk], marker_color=np.where(d[msk] >= 0, IB, DER),
                                 hovertemplate="%{x:.0f}%: %{y:+.1f} pts<extra></extra>"))
            f.add_hline(y=0, line=dict(color=MUTED))
            show(sty(f, 420, "IBIT − Deribit (vol pts)", "Strike vs forward (%)", "pts"), "smile_d")

        # -- term structure + skew grid ---------------------------------------
        sec("Term structure & skew", "Per-expiry ATM vol, 25Δ risk reversal, 25Δ butterfly and skew as % of ATM",
            VIO)
        defs = [("atm", "ATM implied vol (%)", 100, True), ("rr25", "25Δ risk reversal (vol pts)", 100, False),
                ("fly25", "25Δ butterfly (vol pts)", 100, False), ("rr25_n", "Skew: RR25 / ATM (%)", 100, False)]
        r1 = st.columns(2)
        r2 = st.columns(2)
        for (col, ttl, mult, logx), cell in zip(defs, [r1[0], r1[1], r2[0], r2[1]]):
            with cell:
                f = go.Figure()
                for nm, m, colr in (("Deribit BTC", mD, DER), ("IBIT", mI, IB)):
                    s = m[m.dte >= 2].dropna(subset=[col])
                    f.add_scatter(x=s.dte, y=s[col] * mult, name=nm, mode="lines+markers",
                                  line=dict(color=colr, width=2.5), marker=dict(size=6))
                if col == "atm":
                    f.add_scatter(x=[30], y=[ibm["iv30"]], name="Cboe iv30", mode="markers",
                                  marker=dict(symbol="diamond", size=12, color=IB,
                                              line=dict(color="#13314F", width=1)))
                    try:
                        from lib import surface
                        dv = surface.dvol_now("BTC")
                        if dv:
                            f.add_scatter(x=[30], y=[dv * (100 if dv < 5 else 1)], name="DVOL",
                                          mode="markers", marker=dict(symbol="diamond", size=12,
                                                                      color=DER, line=dict(color="#13314F", width=1)))
                    except Exception:  # noqa: BLE001
                        pass
                if col != "atm":
                    f.add_hline(y=0, line=dict(color=MUTED, dash="dot"))
                show(sty(f, 340, ttl, "Days to expiry", None).update_xaxes(type="log"), f"ts_{col}")

        # -- term structure on Deribit's fixed expiries ---------------------------
        sec("Term structure on Deribit expiries",
            "Deribit's listed expiry dates on the x-axis; IBIT interpolated (in DTE) to each one. "
            "Blank = Deribit expiry outside IBIT's listed range", VIO)
        _TS_MET = {"atm": ("ATM implied vol", "%"), "rr25": ("25Δ risk reversal", "vol pts"),
                   "fly25": ("25Δ butterfly", "vol pts"), "rr10": ("10Δ risk reversal", "vol pts"),
                   "fly10": ("10Δ butterfly", "vol pts"), "rr25_n": ("Skew: RR25 / ATM", "%")}
        e1, e2, e3 = st.columns([1.3, 1.1, 1.3])
        with e1:
            ts_kind = st.selectbox("Deribit expiries", L.EXPIRY_FILTERS, index=0, key="ibit_ts_kind",
                                   help="All = dailies + weeklies + monthlies + quarterlies. "
                                        "Monthlies/quarterlies = last Friday of the month.")
        with e2:
            ts_max = st.selectbox("Max DTE", [30, 60, 90, 180, 365, 730], index=4, key="ibit_ts_max",
                                  format_func=lambda d: f"{d} DTE")
        with e3:
            ts_met = st.selectbox("Chart metric", list(_TS_MET), index=0, key="ibit_ts_met",
                                  format_func=lambda c: _TS_MET[c][0])
        eg = L.deribit_expiry_grid(mD, mI, kind=ts_kind, max_dte=ts_max)
        if eg.empty:
            st.info("No Deribit expiries match this filter.")
        else:
            lab = [f"{e:%d %b %y}" for e in eg["expiry"]]
            cd = np.column_stack([eg["dte"].round(1), eg["type"]])
            f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                              vertical_spacing=0.06)
            for nm, pre, colr in (("Deribit BTC", "D_", DER), ("IBIT (interp.)", "I_", IB)):
                f.add_scatter(x=lab, y=eg[f"{pre}{ts_met}"] * 100, name=nm, mode="lines+markers",
                              line=dict(color=colr, width=2.5), marker=dict(size=7), customdata=cd,
                              connectgaps=False,
                              hovertemplate="%{x} · %{customdata[0]}d · %{customdata[1]}<br>"
                                            "%{y:.2f}<extra>" + nm + "</extra>", row=1, col=1)
            dx = eg[f"X_{ts_met}"] * 100
            f.add_bar(x=lab, y=dx, name="IBIT − Deribit", marker_color=np.where(dx >= 0, IB, DER),
                      showlegend=False, customdata=cd,
                      hovertemplate="%{x} · %{customdata[0]}d<br>%{y:+.2f} pts<extra>IBIT − Deribit</extra>",
                      row=2, col=1)
            f.add_hline(y=0, line=dict(color=MUTED, dash="dot"), row=2, col=1)
            f.update_xaxes(type="category", tickangle=-45)
            f.update_yaxes(title_text=_TS_MET[ts_met][1], row=1, col=1)
            f.update_yaxes(title_text="IBIT − Deribit", row=2, col=1)
            show(sty(f, 520, f"{_TS_MET[ts_met][0]} by Deribit expiry"), "ts_dexp")

            tt = pd.DataFrame({"Expiry": lab, "Type": eg["type"], "DTE": eg["dte"].round(1)})
            for c, nm in (("atm", "ATM"), ("rr25", "RR25"), ("fly25", "Fly25")):
                tt[f"Deribit {nm}"] = eg[f"D_{c}"] * 100
                tt[f"IBIT {nm}"] = eg[f"I_{c}"] * 100
                tt[f"Δ {nm}"] = eg[f"X_{c}"] * 100
            st.dataframe(
                tt.style.format({c: "{:+.1f}" if c.startswith("Δ") else "{:.1f}" for c in tt.columns[3:]},
                                na_rep="–")
                .format({"DTE": "{:.1f}"})
                .background_gradient(subset=[c for c in tt.columns if c.startswith("Δ")],
                                     cmap="RdBu", vmin=-8, vmax=8),
                hide_index=True, width="stretch")
            st.caption("Vol points; Δ = IBIT − Deribit. IBIT is linearly interpolated in DTE between its listed "
                       "expiries (no extrapolation), so very short Deribit dailies and the far wings of the curve "
                       "can be blank. Deribit expires 08:00 UTC, IBIT at the 16:00 ET close — matched on time to expiry.")

        # -- tenor table ---------------------------------------------------------
        cols = ["atm", "rr25", "fly25", "rr10", "fly10"]
        tI, tD = L.at_tenors(mI, cols), L.at_tenors(mD, cols)
        t = pd.DataFrame({"Tenor": [f"{x}d" for x in L.TENORS]})
        for c, lab in zip(cols, ["ATM", "RR25", "Fly25", "RR10", "Fly10"]):
            t[f"IBIT {lab}"] = tI[c].values * 100
            t[f"Deribit {lab}"] = tD[c].values * 100
            t[f"Δ {lab}"] = (tI[c].values - tD[c].values) * 100
        st.dataframe(
            t.style.format({c: "{:+.1f}" if c.startswith("Δ") else "{:.1f}" for c in t.columns[1:]}, na_rep="–")
            .background_gradient(subset=[c for c in t.columns if c.startswith("Δ")], cmap="RdBu", vmin=-8, vmax=8),
            hide_index=True, width="stretch")
        st.caption("Vol points; Δ = IBIT − Deribit. RR = call IV − put IV; Fly = (call IV + put IV)/2 − ATM IV. "
                   "Blank = tenor outside the listed range or too few quotes.")

        # -- surfaces ---------------------------------------------------------------
        sec("Vol surfaces side by side", "Same colour scale, strike (K/F − 1) × days to expiry", DER)
        kw = KG[np.abs(KG) <= kmax]
        grid_t = [3, 7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365]
        gI = L.on_tenor_grid(zI[kw], grid_t) * 100
        gD = L.on_tenor_grid(zD[kw], grid_t) * 100
        both = np.concatenate([gI.values.ravel(), gD.values.ravel()])
        zmin, zmax = np.nanpercentile(both, 2), np.nanpercentile(both, 98)
        hc = st.columns(2)
        for cell, g, nm in ((hc[0], gD, "Deribit BTC"), (hc[1], gI, "IBIT")):
            with cell:
                f = go.Figure(go.Heatmap(z=g.values, x=kw * 100, y=[f"{t}d" for t in grid_t],
                                         colorscale="Turbo", zmin=zmin, zmax=zmax,
                                         colorbar=dict(title="IV %", len=0.9),
                                         hovertemplate="K/F %{x:.0f}% · %{y}<br>IV %{z:.1f}%<extra></extra>"))
                show(sty(f, 380, nm, "Strike vs forward (%)", None).update_layout(hovermode="closest"),
                     f"hm_{nm}")
        diff = gI - gD
        lim = float(np.nanpercentile(np.abs(diff.values), 95) or 5)
        f = go.Figure(go.Heatmap(z=diff.values, x=kw * 100, y=[f"{t}d" for t in grid_t],
                                 colorscale="RdBu", zmid=0, zmin=-lim, zmax=lim,
                                 colorbar=dict(title="pts"),
                                 hovertemplate="K/F %{x:.0f}% · %{y}<br>IBIT−Deribit %{z:+.1f} pts<extra></extra>"))
        show(sty(f, 380, "IBIT − Deribit implied vol (pts) — blue = IBIT richer, red = Deribit richer",
                 "Strike vs forward (%)").update_layout(hovermode="closest"), "hm_diff")

        # -- open interest -----------------------------------------------------------
        sec("Open interest comparison",
            "USD notional (OI × multiplier × spot), shown as % of each venue's total so size differences wash out", CALL)
        oI, oD = L.oi_by_moneyness(ib), L.oi_by_moneyness(der)
        f = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03,
                          subplot_titles=("Deribit BTC", "IBIT"))
        for j, o in enumerate((oD, oI), start=1):
            f.add_bar(x=o["mid"] * 100, y=o["C_pct"], name="Calls", marker_color=CALL,
                      showlegend=(j == 1), row=1, col=j)
            f.add_bar(x=o["mid"] * 100, y=-o["P_pct"], name="Puts", marker_color=PUT,
                      showlegend=(j == 1), row=1, col=j)
        f.update_layout(barmode="relative")
        f.update_xaxes(title_text="Strike vs spot (%)")
        show(sty(f, 400, yt="% of venue OI"), "oi_m")
        st.caption("Outer buckets are open-ended (≤ −30% / ≥ +30%). Calls up, puts down.")

        c1, c2 = st.columns(2)
        tI_, tD_ = L.oi_by_tenor(ib), L.oi_by_tenor(der)
        with c1:
            f = go.Figure()
            for nm, o, cc in (("Deribit BTC", tD_, DER), ("IBIT", tI_, IB)):
                f.add_bar(x=o.bucket, y=o.call_pct + o.put_pct, name=nm, marker_color=cc)
            f.update_layout(barmode="group")
            show(sty(f, 360, "Where the OI sits by tenor (% of venue OI)", None, "%"), "oi_t")
        with c2:
            f = go.Figure()
            for nm, o, cc in (("Deribit BTC", tD_, DER), ("IBIT", tI_, IB)):
                f.add_scatter(x=o.bucket, y=o.pc_oi, name=f"{nm} OI", mode="lines+markers",
                              line=dict(color=cc, width=3))
                f.add_scatter(x=o.bucket, y=o.pc_vol, name=f"{nm} volume", mode="lines+markers",
                              line=dict(color=cc, width=2, dash="dot"))
            f.add_hline(y=1, line=dict(color=MUTED, dash="dash"))
            show(sty(f, 360, "Put/call ratio by tenor (solid = OI, dotted = volume)", None, "P/C"), "pc_t")

        # -- implied distribution ------------------------------------------------------
        sec("Market-implied distribution", f"Breeden-Litzenberger on each venue's smoothed smile, nearest expiry to {tenor}d",
            VIO)

        def nearest(df_, tgt):
            e = df_.groupby("expiry")["dte"].first()
            e = e[e >= 3]
            return e.index[(e - tgt).abs().argmin()]

        dens = {}
        for nm, df_ in (("Deribit BTC", der), ("IBIT", ib)):
            e = nearest(df_, tenor)
            g_ = df_[df_.expiry == e]
            r = L.implied_density(g_)
            if r is not None:
                dens[nm] = (r, float(g_.dte.iloc[0]), L.tail_probs(r[0], r[1]))
        if dens:
            c1, c2 = st.columns([3, 2])
            with c1:
                f = go.Figure()
                for nm, cc in (("Deribit BTC", DER), ("IBIT", IB)):
                    if nm in dens:
                        (xx, ff, _), dte_, _ = dens[nm]
                        f.add_scatter(x=(np.exp(xx) - 1) * 100, y=ff, name=f"{nm} ({dte_:.0f}d)",
                                      mode="lines", line=dict(color=cc, width=3),
                                      fill="tozeroy", fillcolor="rgba(120,140,170,.10)")
                f.add_vline(x=0, line=dict(color=MUTED, dash="dot"))
                show(sty(f, 400, xt="Terminal return vs forward (%)", yt="Probability density"), "dens")
            with c2:
                rows = []
                for nm, (rr_, dte_, tp) in dens.items():
                    rows.append({"Venue": nm, "Expiry (d)": round(dte_), "P(≤ −10%)": tp["down"] * 100,
                                 "P(≥ +10%)": tp["up"] * 100,
                                 "Down/Up": tp["down"] / tp["up"] if tp["up"] else np.nan})
                st.dataframe(pd.DataFrame(rows).style.format(
                    {"P(≤ −10%)": "{:.1f}%", "P(≥ +10%)": "{:.1f}%", "Down/Up": "{:.2f}x"}, na_rep="–"),
                    hide_index=True, width="stretch")
                st.caption("Risk-neutral (not real-world) probabilities, normalised inside the quoted strike range — "
                           "indicative only. Different expiry lengths will move the numbers.")

        sec("Size & activity", "Notional open interest and volume in USD", MUTED)
        f = make_subplots(rows=1, cols=2, subplot_titles=("Open interest (USD)", "Volume today (USD)"))
        for nm, df_, cc in (("Deribit BTC", der, DER), ("IBIT", ib, IB)):
            f.add_bar(x=[nm], y=[df_.oi_usd.sum()], marker_color=cc, showlegend=False, row=1, col=1)
            f.add_bar(x=[nm], y=[df_.vol_usd.sum()], marker_color=cc, showlegend=False, row=1, col=2)
        show(sty(f, 320).update_layout(hovermode="closest"), "size")
        st.caption(f"1 IBIT share = {btc_per_share:.6f} BTC ({_share_phrase(share_ratio)}). "
                   "1 IBIT option = 100 shares, so that many BTC is one Deribit contract. "
                   "Deribit volume is rolling 24h; IBIT volume is the latest US session.")

# ===========================================================================
# TAB 2 — IBIT surface
# ===========================================================================
if _view == _VIEWS[1]:
    sec("IBIT implied-vol surface", "Interpolated OTM mid-quote IV from Cboe; blank = no live two-sided quotes")
    zw = zI[[c for c in KG if abs(c) <= kmax]]
    zw = zw[zw.index <= 400]
    f = go.Figure(go.Surface(x=zw.columns * 100, y=zw.index, z=zw.values * 100, colorscale="Turbo",
                             connectgaps=False,
                             colorbar=dict(title="IV %", len=0.7),
                             hovertemplate="K/F %{x:.0f}%<br>%{y:.0f}d<br>IV %{z:.1f}%<extra></extra>"))
    f.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0),
                    scene=dict(xaxis_title="Strike vs fwd (%)", yaxis_title="DTE", zaxis_title="IV %",
                               camera=dict(eye=dict(x=1.7, y=-1.6, z=0.9))))
    show(f, "surf3d")

    c1, c2 = st.columns(2)
    with c1:
        sec("Smiles by expiry", "pick expiries", IB)
        exps = sorted(ib.expiry.unique())
        usable = [e for e in exps if (ib[ib.expiry == e].dte.iloc[0] >= 2) and ib[(ib.expiry == e) & ib.ok].shape[0] >= 8]
        default = [usable[i] for i in np.linspace(0, len(usable) - 1, min(6, len(usable))).astype(int)]
        pick = st.multiselect("Expiries", usable, default=default,
                              format_func=lambda e: f"{e:%d %b %y} ({ib[ib.expiry == e].dte.iloc[0]:.0f}d)")
        f = go.Figure()
        pal = ["#4DA6FF", "#2ECC71", "#F7C948", "#FF8A5C", "#FF5C8A", "#A78BFA", "#5CE1E6", "#C0C7D4"]
        for i, e in enumerate(pick):
            s = L.otm_smile(ib[ib.expiry == e])
            s = s[np.abs(s.k) <= kmax]
            f.add_scatter(x=s.k * 100, y=s.iv * 100, mode="lines+markers", name=f"{e:%d %b %y}",
                          line=dict(color=pal[i % len(pal)], width=2), marker=dict(size=4))
        f.add_vline(x=0, line=dict(color=MUTED, dash="dot"))
        show(sty(f, 420, xt="Strike vs forward (%)", yt="IV (%)"), "smiles")
    with c2:
        sec("Term structure", "ATM, 25Δ call and 25Δ put IV by expiry", VIO)
        s = mI[mI.dte >= 2]
        f = go.Figure()
        f.add_scatter(x=s.dte, y=s.p25 * 100, name="25Δ put", line=dict(color=PUT, width=2))
        f.add_scatter(x=s.dte, y=s.atm * 100, name="ATM", line=dict(color="#7f8fa6", width=3))
        f.add_scatter(x=s.dte, y=s.c25 * 100, name="25Δ call", line=dict(color=CALL, width=2))
        f.add_scatter(x=[30], y=[ibm["iv30"]], name="Cboe iv30", mode="markers",
                      marker=dict(symbol="diamond", size=12, color=IB, line=dict(color="#13314F", width=1)))
        show(sty(f, 420, xt="Days to expiry", yt="IV (%)").update_xaxes(type="log"), "ib_ts")

    c1, c2 = st.columns(2)
    with c1:
        s = mI[(mI.dte >= 2)]
        f = make_subplots(specs=[[{"secondary_y": True}]])
        f.add_bar(x=s.dte, y=s.rr25 * 100, name="RR25 (pts)", marker_color=np.where(s.rr25 >= 0, CALL, PUT))
        f.add_scatter(x=s.dte, y=s.fly25 * 100, name="Fly25 (pts)", mode="lines+markers",
                      line=dict(color=VIO, width=3), secondary_y=True)
        show(sty(f, 380, "Skew (bars) and wing premium (line) by expiry", "Days to expiry", None)
             .update_xaxes(type="log"), "ib_skew")
    with c2:
        g = ib[ib.ok & (ib.dte >= 2) & (np.abs(ib.k) <= kmax)].copy()
        g["sp"] = (g.ask - g.bid) / ((g.ask + g.bid) / 2) * 100
        g = g[g.sp.between(0, 60)]
        f = go.Figure(go.Scattergl(x=g.k * 100, y=g.sp, mode="markers",
                                   marker=dict(size=5, color=np.log10(g.dte.clip(lower=1)), colorscale="Viridis",
                                               opacity=.6, colorbar=dict(title="log10 DTE"))))
        show(sty(f, 380, "Quote quality: bid-ask spread (% of mid) by strike", "Strike vs forward (%)", "%")
             .update_layout(hovermode="closest"), "ib_q")

# ===========================================================================
# TAB 3 — IBIT positioning
# ===========================================================================
if _view == _VIEWS[2]:
    sec("Open interest by strike", "Calls up, puts down — contracts (×100 shares)", CALL)
    opts = ["All expiries"] + [f"{e:%d %b %y} ({ib[ib.expiry == e].dte.iloc[0]:.0f}d)" for e in sorted(ib.expiry.unique())]
    pick = st.selectbox("Expiry", opts, index=0)
    sub = ib if pick == "All expiries" else ib[ib.expiry == sorted(ib.expiry.unique())[opts.index(pick) - 1]]
    sub = sub[np.abs(sub.m - 1) <= max(kmax, 0.3)]
    byk = sub.pivot_table(index="strike", columns="type", values="oi", aggfunc="sum").fillna(0)
    byv = sub.pivot_table(index="strike", columns="type", values="volume", aggfunc="sum").fillna(0)
    for c in "CP":
        for t_ in (byk, byv):
            if c not in t_:
                t_[c] = 0
    f = go.Figure()
    f.add_bar(x=byk.index, y=byk.C, name="Call OI", marker_color=CALL)
    f.add_bar(x=byk.index, y=-byk.P, name="Put OI", marker_color=PUT)
    f.add_vline(x=ibm["spot"], line=dict(color=MUTED, dash="dash"),
                annotation_text=f"spot {ibm['spot']:.2f}")
    mp = L.max_pain(sub if pick != "All expiries" else ib[ib.dte > 0])
    if pick != "All expiries" and len(mp):
        f.add_vline(x=float(mp.max_pain.iloc[0]), line=dict(color=VIO, dash="dot"),
                    annotation_text=f"max pain {mp.max_pain.iloc[0]:.1f}", annotation_position="bottom")
    f.update_layout(barmode="relative")
    show(sty(f, 440, xt="Strike", yt="Open interest (contracts)"), "oi_k")

    c1, c2 = st.columns(2)
    with c1:
        e = ib.groupby(["expiry", "type"]).oi_usd.sum().unstack().fillna(0)
        e = e[e.index.map(lambda x: (x - pd.Timestamp(now)).days) <= 450]
        f = go.Figure()
        f.add_bar(x=e.index, y=e.get("C", 0), name="Calls", marker_color=CALL)
        f.add_bar(x=e.index, y=e.get("P", 0), name="Puts", marker_color=PUT)
        f.update_layout(barmode="stack")
        show(sty(f, 380, "Notional OI by expiry (USD)", None, "USD"), "oi_e")
    with c2:
        e = ib.groupby(["expiry", "type"]).vol_usd.sum().unstack().fillna(0)
        e = e[e.index.map(lambda x: (x - pd.Timestamp(now)).days) <= 120]
        f = go.Figure()
        f.add_bar(x=e.index, y=e.get("C", 0), name="Calls", marker_color=CALL)
        f.add_bar(x=e.index, y=e.get("P", 0), name="Puts", marker_color=PUT)
        f.update_layout(barmode="stack")
        show(sty(f, 380, "Notional volume by expiry (USD, last session)", None, "USD"), "vol_e")

    c1, c2 = st.columns(2)
    with c1:
        mm = mI[mI.dte >= 1].copy()
        mm = mm[mm.dte <= 450]
        f = go.Figure()
        f.add_scatter(x=mm.expiry, y=mm.pc_oi, name="P/C OI", mode="lines+markers", line=dict(color=IB, width=3))
        f.add_scatter(x=mm.expiry, y=mm.pc_vol, name="P/C volume", mode="lines+markers",
                      line=dict(color=DER, width=2, dash="dot"))
        f.add_hline(y=1, line=dict(color=MUTED, dash="dash"))
        show(sty(f, 360, "Put/call ratios by expiry", None, "P/C"), "ib_pc")
    with c2:
        hh = ib[(ib.dte >= 1) & (ib.dte <= 120) & (np.abs(ib.m - 1) <= 0.3)]
        hp = hh.pivot_table(index="strike", columns=hh.expiry.dt.strftime("%d %b"), values="oi",
                            aggfunc="sum").fillna(0)
        order = sorted(hh.expiry.unique())
        hp = hp[[pd.Timestamp(x).strftime("%d %b") for x in order if pd.Timestamp(x).strftime("%d %b") in hp.columns]]
        f = go.Figure(go.Heatmap(z=np.log10(hp.values + 1), x=hp.columns, y=hp.index, colorscale="Magma",
                                 colorbar=dict(title="log10 OI"),
                                 hovertext=hp.values, hovertemplate="%{x} · K=%{y}<br>OI %{hovertext:,.0f}<extra></extra>"))
        f.add_hline(y=ibm["spot"], line=dict(color="white", dash="dash"))
        show(sty(f, 360, "OI heatmap: strike × expiry (≤120d)", None, "Strike").update_layout(hovermode="closest"),
             "oi_hm")

    sec("Cumulative OI distribution", "Where the open interest sits relative to spot", VIO)
    cum = ib.groupby(["strike", "type"]).oi.sum().unstack().fillna(0)
    cum = cum[np.abs(cum.index / ibm["spot"] - 1) <= 0.5]
    f = go.Figure()
    for c, nm, cc in (("C", "Calls", CALL), ("P", "Puts", PUT)):
        if c in cum:
            f.add_scatter(x=cum.index, y=cum[c].cumsum() / cum[c].sum() * 100, name=nm,
                          line=dict(color=cc, width=3, shape="hv"))
    f.add_vline(x=ibm["spot"], line=dict(color=MUTED, dash="dash"))
    show(sty(f, 340, xt="Strike", yt="Cumulative % of OI"), "cum")

# ===========================================================================
# TAB 4 — Gamma & max pain
# ===========================================================================
if _view == _VIEWS[3]:
    sec("Dealer gamma exposure (GEX) by strike",
        "$ gamma per 1% spot move. Assumes customers are long calls & puts → dealers short: calls +, puts −. "
        "A rough positioning proxy, not observed dealer inventory.", VIO)
    horizon = st.select_slider("Expiries included (days)", options=[7, 30, 90, 180, 400, 900], value=90)
    g = L.gex_by_strike(ib, horizon)
    gz = g[np.abs(g.strike / ibm["spot"] - 1) <= max(kmax, 0.3)]
    flip = L.gamma_flip(g)
    kpis([("Net GEX", usd(g.gex.sum()), "per 1% move, within horizon", VIO, "up" if g.gex.sum() >= 0 else "dn"),
          ("Gamma flip strike", f"{flip:.1f}" if np.isfinite(flip) else "–",
           "cumulative GEX crosses zero", IB),
          ("Largest +GEX strike", f"{gz.loc[gz.gex.idxmax(), 'strike']:.1f}" if len(gz) else "–", "call wall", CALL),
          ("Largest −GEX strike", f"{gz.loc[gz.gex.idxmin(), 'strike']:.1f}" if len(gz) else "–", "put wall", PUT)])
    f = go.Figure(go.Bar(x=gz.strike, y=gz.gex, marker_color=np.where(gz.gex >= 0, CALL, PUT)))
    f.add_vline(x=ibm["spot"], line=dict(color=MUTED, dash="dash"), annotation_text="spot")
    if np.isfinite(flip):
        f.add_vline(x=flip, line=dict(color=VIO, dash="dot"), annotation_text="flip", annotation_position="bottom")
    show(sty(f, 420, xt="Strike", yt="GEX ($ per 1% move)"), "gex")

    c1, c2 = st.columns(2)
    with c1:
        x = ib[ib.gamma.notna() & (ib.dte <= 400)].copy()
        sgn = np.where(x.type == "C", 1, -1)
        x["gex"] = x.gamma * x.oi * x.mult * x.spot ** 2 * 0.01 * sgn
        x["vega"] = x.gamma * x.fwd ** 2 * x.iv * x["T"] * x.oi * x.mult / 100
        ge = x.groupby("expiry")[["gex", "vega"]].sum()
        f = go.Figure(go.Bar(x=ge.index, y=ge.gex, marker_color=np.where(ge.gex >= 0, CALL, PUT)))
        show(sty(f, 360, "Net GEX by expiry", None, "$ per 1%"), "gex_e")
    with c2:
        f = go.Figure(go.Bar(x=ge.index, y=ge.vega, marker_color=IB))
        show(sty(f, 360, "Vega notional by expiry ($ per 1 vol pt, gross OI)", None, "$"), "vega_e")

    sec("Max pain", "Strike at which option holders collectively lose the most (OI-based)", DER)
    mp = L.max_pain(ib)
    mp = mp[(mp.dte >= 1) & (mp.dte <= 120) & (mp.oi_usd > 5e6)]
    f = go.Figure()
    f.add_scatter(x=mp.expiry, y=mp.max_pain, name="Max pain", mode="lines+markers", line=dict(color=DER, width=3))
    f.add_hline(y=ibm["spot"], line=dict(color=MUTED, dash="dash"), annotation_text="spot")
    show(sty(f, 360, xt=None, yt="Strike"), "pain")
    mp2 = mp.assign(expiry=mp.expiry.dt.strftime("%d %b %y"), dte=mp.dte.round(0))
    st.dataframe(mp2[["expiry", "dte", "max_pain", "pain_vs_spot", "oi_usd"]].rename(columns={
        "expiry": "Expiry", "dte": "DTE", "max_pain": "Max pain", "pain_vs_spot": "vs spot", "oi_usd": "OI (USD)"})
        .style.format({"Max pain": "{:.1f}", "vs spot": "{:+.1%}", "OI (USD)": "${:,.0f}"}),
        hide_index=True, width="stretch")

# ===========================================================================
# TAB 5 — Activity
# ===========================================================================
if _view == _VIEWS[4]:
    sec("Most active contracts", "Ranked by notional volume in the latest session; vol/OI > 1 = new positioning", "#FF8A5C")
    mv = st.slider("Minimum contracts traded", 50, 5000, 300, step=50)
    act = L.activity(ib, mv)
    if act.empty:
        st.info("No contracts above that volume.")
    else:
        top = act.head(40).copy()
        top["Contract"] = top.expiry.dt.strftime("%d %b %y") + " " + top.strike.map("{:g}".format) + top.type
        f = go.Figure(go.Bar(y=top.Contract.head(15)[::-1], x=top.vol_usd.head(15)[::-1], orientation="h",
                             marker_color=np.where(top.type.head(15)[::-1] == "C", CALL, PUT)))
        c1, c2 = st.columns(2)
        with c1:
            show(sty(f, 440, "Top 15 by notional volume (USD)").update_layout(hovermode="closest"), "act_bar")
        with c2:
            size_by = st.radio(
                "Bubble size", ["IBIT contracts", "BTC equivalent"], horizontal=True,
                key="ibit_bub_size",
                help="BTC equivalent = contracts × 100 × iShares BTC per share. "
                     "That is the Deribit size (1 Deribit BTC option = 1 BTC). "
                     "The two choices draw the same marks; the key changes units.")
            b = act[act.dte <= 200].copy()
            if b.empty:
                st.info("No contracts inside 200 days at this volume floor.")
                b = None
            if b is not None:
                unit = "btc" if size_by.startswith("BTC") else "contracts"
                b["btc_eq"] = L.contracts_to_btc(b["volume"], btc_per_share)
                b["k_btc"] = L.strike_btc_equivalent(b["strike"], btc_per_share)
                divisor = L.bubble_size_divisor(unit, btc_per_share)
                qty = b["btc_eq"] if unit == "btc" else b["volume"]
                b["mark"] = L.bubble_marker_size(qty, divisor)
                # Bottom row is the size key. Plotly draws every legend marker at one
                # size, so the key has to be real marks. xaxis2 belongs to that row;
                # the BTC-equivalent strike scale is xaxis3 over the bubbles.
                f = make_subplots(rows=2, cols=1, row_heights=[0.70, 0.30], vertical_spacing=0.12)
                for typ, name, color in (("C", "Calls", CALL), ("P", "Puts", PUT)):
                    s = b[b["type"] == typ]
                    if s.empty:
                        continue
                    custom = np.column_stack([s["k_btc"], s["volume"], s["btc_eq"], s["vol_usd"]])
                    f.add_scatter(
                        x=s["strike"], y=s["dte"], mode="markers", name=name, row=1, col=1,
                        marker=dict(size=s["mark"], color=color, opacity=0.65, line=dict(width=0)),
                        customdata=custom,
                        hovertemplate=("IBIT strike %{x:.2f} · BTC-eq strike %{customdata[0]:,.0f}<br>"
                                       "%{y:.0f} days<br>"
                                       "%{customdata[1]:,.0f} IBIT contracts<br>"
                                       "%{customdata[2]:.2f} BTC equiv (Deribit contracts)<br>"
                                       "USD notional %{customdata[3]:,.0f}<extra>" + name + "</extra>"))
                f.add_scatter(x=[None, None], y=[None, None], mode="lines", name="IBIT spot",
                              line=dict(color=MUTED, dash="dash", width=2), hoverinfo="skip",
                              row=1, col=1)
                keys = L.bubble_size_key(btc_per_share, unit)
                f.add_scatter(
                    x=list(range(1, len(keys) + 1)), y=[0] * len(keys),
                    mode="markers+text", showlegend=False, hoverinfo="skip", row=2, col=1,
                    text=[f"{k['contracts']:,.0f} contracts<br>{k['btc']:.0f} BTC" for k in keys],
                    textposition="top center",
                    marker=dict(size=[k["size"] for k in keys], color=MUTED, opacity=0.8,
                                line=dict(width=0)))
                f.add_vline(x=float(ibm["spot"]), line=dict(color=MUTED, dash="dash"),
                            annotation_text="spot", annotation_position="top", row=1, col=1)
                xmin, xmax = float(b["strike"].min()), float(b["strike"].max())
                pad = max((xmax - xmin) * 0.04, 0.5)
                x0, x1 = xmin - pad, xmax + pad
                f.add_scatter(x=[x0 / btc_per_share, x1 / btc_per_share],
                              y=[float(b["dte"].min())] * 2,
                              xaxis="x3", yaxis="y", mode="lines", line=dict(width=0),
                              opacity=0, showlegend=False, hoverinfo="skip")
                fig = sty(f, 640, None, "IBIT strike (USD)", "Days to expiry")
                fig.update_xaxes(range=[x0, x1], row=1, col=1)
                fig.update_xaxes(visible=False, range=[0.35, len(keys) + 0.65], row=2, col=1)
                fig.update_yaxes(visible=False, range=[-0.6, 2.2], row=2, col=1)
                fig.update_layout(
                    hovermode="closest",
                    xaxis3=dict(overlaying="x", side="top", anchor="y", showgrid=False,
                                tickformat=",.0f", range=[x0 / btc_per_share, x1 / btc_per_share],
                                title="BTC-equivalent strike (USD per BTC)"),
                    legend=dict(orientation="h", y=1.08, x=0, xanchor="left"),
                    margin=dict(l=50, r=20, t=56, b=24))
                show(fig, "act_bub")
                bits = L.bubble_size_key(btc_per_share, "contracts")
                key_txt = " · ".join(
                    f"{k['contracts']:,.0f} contracts = {k['btc']:.1f} BTC" for k in bits)
                st.caption(
                    "Green = calls, red = puts. The dashed line is IBIT spot. "
                    "Bubble area follows the square root of size and is capped, so a large print is not "
                    f"to scale with a small one. Size key: {key_txt}. "
                    f"1 contract = 100 shares × {btc_per_share:.6f} BTC ({_share_phrase(share_ratio)}), "
                    "which is the Deribit-contract equivalent. "
                    "The top axis is the IBIT strike divided by that BTC-per-share. "
                    "USD notional is contracts × 100 × the IBIT share price.")
        tbl = top.assign(IV=top.iv * 100)[["Contract", "dte", "volume", "oi", "vol_oi", "IV", "vol_usd",
                                           "spread_pct"]].rename(columns={
            "dte": "DTE", "volume": "Volume", "oi": "OI", "vol_oi": "Vol/OI",
            "vol_usd": "Notional vol", "spread_pct": "Spread %"})
        st.dataframe(
            tbl, hide_index=True, width="stretch",
            column_config={"DTE": st.column_config.NumberColumn(format="%.0f"),
                           "Volume": st.column_config.NumberColumn(format="%d"),
                           "OI": st.column_config.NumberColumn(format="%d"),
                           "Vol/OI": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=5),
                           "IV": st.column_config.NumberColumn(format="%.1f"),
                           "Notional vol": st.column_config.NumberColumn(format="$%d"),
                           "Spread %": st.column_config.NumberColumn(format="%.1f")})

    sec("Volume by strike", "Calls up, puts down (contracts, all expiries)", "#FF8A5C")
    vv = ib[np.abs(ib.m - 1) <= 0.35].pivot_table(index="strike", columns="type", values="volume",
                                                  aggfunc="sum").fillna(0)
    f = go.Figure()
    if "C" in vv:
        f.add_bar(x=vv.index, y=vv.C, name="Calls", marker_color=CALL)
    if "P" in vv:
        f.add_bar(x=vv.index, y=-vv.P, name="Puts", marker_color=PUT)
    f.add_vline(x=ibm["spot"], line=dict(color=MUTED, dash="dash"))
    f.update_layout(barmode="relative")
    show(sty(f, 380, xt="Strike", yt="Volume (contracts)"), "vol_k")

# ===========================================================================
# TAB 6 — Large tickets (snapshot, not a tape)
# ===========================================================================
if _view == _VIEWS[5]:
    sec("Large tickets and estimated flow",
        "Cboe's free feed is one delayed chain snapshot per contract: bid, ask, last, "
        "volume and open interest. It is not a trade tape, so these are not exchange "
        "block prints (that tape is OPRA, which is paid). Open-interest changes use "
        "only closes this app has recorded — nothing is backfilled.",
        "#FF8A5C")

    snap_ts = ibm.get("snapshot_ny") or now
    session = L.session_date_of(snap_ts)
    session_s = session.isoformat()
    try:
        store = load_ibit_history()
    except Exception as e:  # noqa: BLE001
        store = None
        st.caption(f"Recorded OI history could not be read ({e}). Volume screens still run.")
    notice = L.history_notice(store, session_s)
    if notice["ready"]:
        st.caption(notice["text"])
    else:
        st.info(notice["text"])

    _fc1, _fc2, _fc3 = st.columns(3)
    with _fc1:
        min_premium = st.selectbox(
            "Min premium", [10_000, 25_000, 50_000, 100_000, 250_000], index=1,
            format_func=lambda v: usd(v), key="ibit_flow_premium",
            help="Premium = contracts × price × 100. Price is the last print, or the mid when there is no last.")
    with _fc2:
        min_contracts = st.selectbox(
            "Min contracts", [10, 25, 50, 100, 250, 500], index=2, key="ibit_flow_contracts",
            help="Day volume on the contract. The flow summary below still uses the whole chain.")
    with _fc3:
        rank_by = st.selectbox("Rank unusual by", ["Premium", "|Delta notional|"], index=0,
                               key="ibit_flow_rank")
    opening_only = st.checkbox("Only likely-opening contracts", value=False, key="ibit_flow_opening")

    prior_day, prior_rows = L.prior_chain(store, session_s)
    screen = L.unusual_activity(ib, prior_rows, min_premium=min_premium, min_contracts=min_contracts)
    if opening_only and len(screen):
        screen = screen[screen["likely_opening"]].reset_index(drop=True)
    if len(screen) and rank_by.startswith("|"):
        screen = screen.reindex(screen["delta_usd"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    split = L.premium_split(ib)
    lean = L.lean_table(ib).set_index("side")
    n_open = int(screen["likely_opening"].sum()) if len(screen) else 0
    call_share = split["call_premium"] / split["premium"] if split["premium"] else np.nan

    kpis([
        ("Session premium", usd(split["premium"]), "volume × price × 100, whole chain", IB),
        ("Call share", pct(call_share) if np.isfinite(call_share) else "–", "of session premium", CALL,
         "up" if np.isfinite(call_share) and call_share >= 0.5 else "dn"),
        ("Est. net premium", susd(lean.loc["Total", "net_premium"]),
         "last print's side × the whole session's volume", VIO,
         "up" if lean.loc["Total", "net_premium"] >= 0 else "dn"),
        ("Est. net delta", susd(lean.loc["Total", "net_delta_usd"]),
         "same last-print estimate, share delta × spot", DER,
         "up" if lean.loc["Total", "net_delta_usd"] >= 0 else "dn"),
        ("Unusual", f"{len(screen):,}  ·  {n_open:,} opening",
         f"≥ {usd(min_premium)} and ≥ {min_contracts:g} contracts", "#FF8A5C"),
        ("OI history", f"{notice['n_sessions']} close{'s' if notice['n_sessions'] != 1 else ''}",
         f"since {notice['since']}" if notice["since"] else "none stored yet", MUTED),
    ])

    # -- open-interest change ------------------------------------------------
    sec("Open-interest change",
        "Where published open interest increased or decreased versus the previous recorded close. "
        "Cboe updates OI overnight, so a change is the prior session's net opening and closing, "
        "not today's prints. Contracts that expired and left the chain are excluded.",
        CALL)
    if not notice["ready"]:
        st.caption("No prior close to diff against yet — this block fills in once the next session is stored.")
    else:
        changes = L.oi_change(ib, prior_rows, asof=snap_ts if ibm.get("snapshot_ny") is not None else now)
        up, down = L.largest_oi_moves(changes, n=12)
        n_expired = int((changes["status"] == "expired").sum())
        net_oi = float(changes.loc[changes["status"] != "expired", "oi_change"].sum())
        kpis([
            ("Versus", prior_day, f"history since {notice['since']}", IB),
            ("Net OI change", f"{net_oi:+,.0f} contracts", "excludes expirations", VIO,
             "up" if net_oi >= 0 else "dn"),
            ("Largest increase", f"{up['oi_change'].iloc[0]:+,.0f}" if len(up) else "–",
             _ticket_name(up.iloc[0]) if len(up) else "", CALL),
            ("Largest decrease", f"{down['oi_change'].iloc[0]:+,.0f}" if len(down) else "–",
             _ticket_name(down.iloc[0]) if len(down) else "", PUT),
        ])
        c1, c2 = st.columns(2)
        with c1:
            show(_oi_bars(up, "Biggest OI increases (contracts)", CALL), "oi_up")
        with c2:
            show(_oi_bars(down, "Biggest OI decreases (contracts)", PUT), "oi_dn")
        movers = pd.concat([up, down], ignore_index=True)
        if len(movers):
            show_m = pd.DataFrame({
                "Contract": movers.apply(_ticket_name, axis=1),
                "Status": movers["status"],
                "OI": movers["oi"],
                "Prior OI": movers["oi_prev"],
                "Change": movers["oi_change"],
                "Change BTC": movers["oi_change"] * L.IBIT_MULT * btc_per_share,
                "Change USD": movers["oi_change_usd"],
                "Volume": movers["volume"],
            })
            st.dataframe(show_m, hide_index=True, width="stretch",
                         column_config={
                             "OI": st.column_config.NumberColumn(format="%.0f"),
                             "Prior OI": st.column_config.NumberColumn(format="%.0f"),
                             "Change": st.column_config.NumberColumn(format="%+d"),
                             "Change BTC": st.column_config.NumberColumn(format="%+.2f"),
                             "Change USD": st.column_config.NumberColumn(format="$%d"),
                             "Volume": st.column_config.NumberColumn(format="%.0f"),
                         })
        if n_expired:
            st.caption(f"{n_expired:,} contract{'s' if n_expired != 1 else ''} expired and left the chain. "
                       "Those drops are not counted as closing trades.")

    # -- unusual activity ----------------------------------------------------
    sec("Unusual activity",
        "High volume relative to open interest, among contracts above the premium and size floors. "
        "Likely opening: today's volume exceeds prior OI (a new listing counts as zero) and/or "
        "published OI rose. With no prior close, the flag is volume greater than the OI on the chain. "
        "Delta notional = volume × delta × 100 shares × IBIT spot.",
        "#FF8A5C")
    if screen.empty:
        st.info("No contracts above those floors.")
    else:
        screen = screen.copy()
        screen["Contract"] = screen.apply(_ticket_name, axis=1)
        top = screen.head(15)
        f = go.Figure(go.Bar(
            y=top["Contract"][::-1],
            x=(top["delta_usd"].abs() if rank_by.startswith("|") else top["premium"])[::-1],
            orientation="h",
            marker_color=np.where(top["type"][::-1] == "C", CALL, PUT),
            hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
        show(sty(f, 440, f"Top 15 by {'|delta notional|' if rank_by.startswith('|') else 'premium'} (USD)")
             .update_layout(hovermode="closest"), "unusual_bar")
        tbl = pd.DataFrame({
            "Contract": screen["Contract"],
            "DTE": screen["dte"],
            "Volume": screen["volume"],
            "OI": screen["oi"],
            "Prior OI": screen["oi_prev"],
            "OI change": screen["oi_change"],
            "Vol/OI": screen["vol_oi"],
            "Premium": screen["premium"],
            "BTC equiv": L.contracts_to_btc(screen["volume"], btc_per_share),
            "Delta shares": screen["delta_shares"],
            "Delta BTC": L.ibit_btc_equivalent(screen["delta_shares"], btc_per_share),
            "Delta USD": screen["delta_usd"],
            "Last": screen["last"] if "last" in screen.columns else np.nan,
            "Bid": screen["bid"],
            "Ask": screen["ask"],
            "Side": screen["side"],
            "Flag": np.where(screen["likely_opening"], "likely opening", ""),
        })
        st.dataframe(
            tbl, hide_index=True, width="stretch",
            column_config={
                "DTE": st.column_config.NumberColumn(format="%.0f"),
                "Volume": st.column_config.NumberColumn(format="%.0f"),
                "OI": st.column_config.NumberColumn(format="%.0f"),
                "Prior OI": st.column_config.NumberColumn(format="%.0f"),
                "OI change": st.column_config.NumberColumn(format="%+.0f"),
                "Vol/OI": st.column_config.NumberColumn(format="%.2f"),
                "Premium": st.column_config.NumberColumn(format="$%.0f"),
                "BTC equiv": st.column_config.NumberColumn(format="%.2f"),
                "Delta shares": st.column_config.NumberColumn(format="%+.0f"),
                "Delta BTC": st.column_config.NumberColumn(format="%+.2f"),
                "Delta USD": st.column_config.NumberColumn(format="$%+.0f"),
                "Last": st.column_config.NumberColumn(format="%.2f"),
                "Bid": st.column_config.NumberColumn(format="%.2f"),
                "Ask": st.column_config.NumberColumn(format="%.2f"),
            })
        st.caption("Side is where the last print sat versus the current bid/ask "
                   "(at/above ask = bought, at/below bid = sold, inside = ambiguous). "
                   "It is the last print only — the premium column is the whole session's volume.")

    # -- buyer / seller lean -------------------------------------------------
    sec("Buyer / seller lean (last print)",
        "Estimate only. Each contract's entire session volume is signed by where its last "
        "trade printed versus the current bid and ask: at or above the ask counts as bought, "
        "at or below the bid as sold, inside the spread or a locked market as ambiguous and "
        "left out of the nets. This is not the aggressor on each fill.",
        VIO)
    total_vol = float(lean.loc["Total", "volume"])
    mix = []
    for lab, col in (("bought", "volume_bought"), ("sold", "volume_sold"),
                     ("ambiguous", "volume_ambiguous"), ("unknown", "volume_unknown")):
        v = float(lean.loc["Total", col])
        mix.append(f"{lab} {v / total_vol * 100:.0f}%" if total_vol > 0 else lab)
    st.caption("Session volume by last-print bucket: " + " · ".join(mix) + ".")
    c1, c2 = st.columns(2)
    with c1:
        f = go.Figure()
        f.add_bar(x=["Calls", "Puts"], y=[lean.loc["Calls", "bought_premium"], lean.loc["Puts", "bought_premium"]],
                  name="Bought (paid)", marker_color=CALL)
        f.add_bar(x=["Calls", "Puts"], y=[-lean.loc["Calls", "sold_premium"], -lean.loc["Puts", "sold_premium"]],
                  name="Sold (received)", marker_color=PUT)
        f.add_hline(y=0, line=dict(color=MUTED, dash="dot"))
        show(sty(f, 380, "Premium by last-print side", None, "USD"), "lean_prem")
    with c2:
        f = go.Figure(go.Bar(
            x=["Calls", "Puts", "Total"],
            y=[lean.loc[s, "net_delta_usd"] for s in ("Calls", "Puts", "Total")],
            marker_color=[CALL, PUT, VIO],
            hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>"))
        f.add_hline(y=0, line=dict(color=MUTED, dash="dot"))
        show(sty(f, 380, "Estimated net delta (USD)", None, "USD")
             .update_layout(hovermode="closest"), "lean_delta")
    st.caption("Bought premium is what the last-print rule calls customer buying; sold premium is "
               "shown negative. Net delta uses the same sign × Black-Scholes delta × 100 × spot. "
               "Ambiguous and unknown prints are omitted from both charts.")

    # -- flow summary + Deribit blocks --------------------------------------
    sec("Flow summary",
        "Put/call split of the session snapshot, a strike × expiry cluster of that premium, "
        "and the same clock window on Deribit BTC blocks (a block is a trade at or above "
        "the Deribit minimum, 12.5 BTC). "
        f"IBIT sizes use {btc_per_share:.6f} BTC per share ({_share_phrase(share_ratio)}): "
        "1 contract = 100 × that ratio, in Deribit BTC.",
        DER)
    c1, c2 = st.columns(2)
    with c1:
        f = go.Figure()
        f.add_bar(x=["Calls", "Puts"], y=[split["call_premium"], split["put_premium"]],
                  name="Premium", marker_color=[CALL, PUT])
        show(sty(f, 360, "Premium by put/call", None, "USD"), "flow_prem")
    with c2:
        f = go.Figure()
        f.add_bar(x=["Calls", "Puts"],
                  y=[split["call_abs_delta_usd"], split["put_abs_delta_usd"]],
                  name="|Delta| notional", marker_color=[CALL, PUT])
        show(sty(f, 360, "Absolute delta notional by put/call", None, "USD"), "flow_dd")
    st.caption(f"Signed delta notional (puts carry negative delta): calls {susd(split['call_delta_usd'])}, "
               f"puts {susd(split['put_delta_usd'])}, net {susd(split['delta_usd'])}. "
               f"Share delta {split['delta_shares']:+,.0f}.")

    heat = L.flow_heatmap_frame(ib)
    if heat.empty:
        st.info("No session volume inside 120 days and ±30% of spot to cluster.")
    else:
        exp_order = sorted(heat["expiry"].unique())
        strikes = sorted(heat["strike"].unique())
        pivot = heat.pivot(index="strike", columns="expiry", values="net_premium").reindex(
            index=strikes, columns=exp_order)
        call_p = heat.pivot(index="strike", columns="expiry", values="call_premium").reindex(
            index=strikes, columns=exp_order)
        put_p = heat.pivot(index="strike", columns="expiry", values="put_premium").reindex(
            index=strikes, columns=exp_order)
        labels = [f"{pd.Timestamp(e):%d %b}" for e in exp_order]
        finite = np.abs(pivot.to_numpy())
        finite = finite[np.isfinite(finite)]
        lim = float(np.percentile(finite, 95)) if len(finite) else 1.0
        if not np.isfinite(lim) or lim <= 0:
            lim = 1.0
        f = go.Figure(go.Heatmap(
            z=pivot.to_numpy(), x=labels, y=strikes, colorscale="RdBu", zmid=0,
            zmin=-lim, zmax=lim, colorbar=dict(title="USD"),
            customdata=np.dstack([call_p.to_numpy(), put_p.to_numpy()]),
            hovertemplate="%{x} · K=%{y}<br>call premium %{customdata[0]:,.0f}"
                          "<br>put premium %{customdata[1]:,.0f}<br>net %{z:,.0f}<extra></extra>"))
        f.add_hline(y=ibm["spot"], line=dict(color="white", dash="dash"))
        show(sty(f, 460, "Session premium by strike × expiry (≤120d, ±30% of spot) — blue calls, red puts",
                 None, "Strike").update_layout(hovermode="closest"), "flow_heat")

    start, end = L.cash_session_bounds(session, now)
    now_ms = int(pd.Timestamp(now).timestamp() * 1000)
    now_ms -= now_ms % 60_000
    der_blocks = None
    blocks_err = None
    used_fallback = False
    try:
        with st.spinner("Loading Deribit BTC blocks…"):
            der_blocks, used_fallback = load_btc_comparison(
                int(start.timestamp() * 1000), int(end.timestamp() * 1000), now_ms)
    except Exception as e:  # noqa: BLE001
        blocks_err = str(e)
    asof = datetime.fromtimestamp(now_ms / 1000, timezone.utc)
    cmp_start, cmp_end, fallback_note = L.deribit_compare_window(
        start, end, asof, session_has_trades=not used_fallback)
    cst, cen = cmp_start.astimezone(L.NY), cmp_end.astimezone(L.NY)
    sst, sen = fx_style.to_local_ts(cmp_start), fx_style.to_local_ts(cmp_end)
    if fallback_note:
        st.caption(fallback_note + f" Window: {cst:%d %b %H:%M}–{cen:%d %b %H:%M} ET "
                   f"({sst:%d %b %H:%M}–{sen:%d %b %H:%M} SGT).")
    else:
        st.caption(f"Deribit window matches the IBIT session the chain describes: "
                   f"{cst:%d %b %H:%M}–{cen:%d %b %H:%M} ET "
                   f"({sst:%d %b %H:%M}–{sen:%d %b %H:%M} SGT).")
    if blocks_err:
        st.warning(f"Deribit block tape unavailable ({blocks_err}). IBIT figures above still stand.")
    else:
        frame = (der_blocks or {}).get("frames", {}).get("BTC")
        der_label = "Deribit BTC blocks (last 24h)" if used_fallback else "Deribit BTC blocks"
        der_sum = L.summarize_deribit_blocks(frame)
        ibit_btc = L.ibit_btc_equivalent(float(lean.loc["Total", "net_delta_shares"]), btc_per_share)
        ibit_col, der_col = "IBIT session (snapshot)", der_label
        cmp = pd.DataFrame([
            {"Metric": "Gross premium", ibit_col: usd(split["premium"]), der_col: usd(der_sum["premium"])},
            {"Metric": "Call premium", ibit_col: usd(split["call_premium"]), der_col: usd(der_sum["call_premium"])},
            {"Metric": "Put premium", ibit_col: usd(split["put_premium"]), der_col: usd(der_sum["put_premium"])},
            {"Metric": "Net premium", ibit_col: susd(lean.loc["Total", "net_premium"]),
             der_col: susd(der_sum["net_premium"])},
            {"Metric": "Net delta USD", ibit_col: susd(lean.loc["Total", "net_delta_usd"]),
             der_col: susd(der_sum["net_delta_usd"])},
            {"Metric": "Net delta BTC",
             ibit_col: f"{ibit_btc:+,.1f} BTC" if np.isfinite(ibit_btc) else "–",
             der_col: (f"{der_sum['net_delta_btc']:+,.1f} BTC"
                       if np.isfinite(der_sum["net_delta_btc"]) else "–")},
            {"Metric": "Prints", ibit_col: "not on the free feed",
             der_col: f"{der_sum['blocks']:,} blocks"},
        ])
        st.dataframe(cmp, hide_index=True, width="stretch")
        f = go.Figure()
        f.add_bar(name="IBIT session", x=["Call premium", "Put premium"],
                  y=[split["call_premium"], split["put_premium"]], marker_color=IB)
        f.add_bar(name="Deribit blocks", x=["Call premium", "Put premium"],
                  y=[der_sum["call_premium"], der_sum["put_premium"]], marker_color=DER)
        f.update_layout(barmode="group")
        cmp_title = ("Call vs put premium — IBIT session and Deribit last 24h"
                     if used_fallback else
                     "Call vs put premium — IBIT session and Deribit blocks")
        show(sty(f, 360, cmp_title, None, "USD"), "flow_cmp")
        der_when = "over the last 24 hours" if used_fallback else "in that US session"
        st.caption(
            "IBIT net premium and net delta attribute every contract's session volume to the side of its "
            "last print. Deribit nets are the actual taker side on trades at or above the block minimum, "
            f"{der_when}. The BTC-equivalent column is share delta × {btc_per_share:.6f} "
            f"({_share_phrase(share_ratio)}); 1 IBIT contract = 100 shares = "
            f"{100 * btc_per_share:.4f} BTC, and 1 Deribit BTC option = 1 BTC. "
            "The two venues settle differently (IBIT American, share-delivered, 16:00 ET; Deribit European, "
            "cash-settled, 08:00 UTC), so the comparison is positioning context, not a matched tape.")
        if (der_blocks or {}).get("meta", {}).get("truncated"):
            st.caption("Deribit hit the page-fetch cap for this window — those block totals are a lower bound.")

# ===========================================================================
# TAB 7 — Data & notes
# ===========================================================================
if _view == _VIEWS[6]:
    sec("Chains", "Normalised schema, identical for both venues", MUTED)
    which = st.radio("Dataset", ["IBIT", "Deribit BTC"] if HAS_D else ["IBIT"], horizontal=True)
    d = ib if which == "IBIT" else der
    show_d = d.copy()
    show_d["expiry"] = show_d.expiry.dt.strftime("%Y-%m-%d %H:%M UTC")
    st.dataframe(show_d, hide_index=True, width="stretch", height=380)
    st.download_button("⬇️ Download CSV", show_d.to_csv(index=False).encode(),
                       file_name=f"{which.split()[0].lower()}_options_chain.csv", mime="text/csv")
    sec("Methodology & caveats", "", MUTED)
    st.markdown(f"""
- **Source:** Cboe free delayed option-chain JSON — a snapshot (≈15 min delayed during the US session, last close otherwise).
  It has **no per-trade tape**; OPRA trade data is licensed. Large-ticket figures are snapshot estimates, not block prints.
  Day-over-day open interest comes only from closes recorded after the US session (`IBIT_chain.parquet` on the `snapshots` branch). Nothing is backfilled.
  This is a deliberate exception to the app's Deribit-only rule (like the Fear & Greed page).
- **Last-print side:** at or above the ask = bought, at or below the bid = sold, inside the spread = ambiguous.
  The whole session's volume is then attributed to that one print. It is an estimate.
- **BTC-equivalent:** iShares bitcoin quantity ÷ shares outstanding ({_share_phrase(share_ratio)}, {btc_per_share:.6f} BTC/share). 1 IBIT option = 100 shares, so contracts × 100 × that ratio is the Deribit-contract equivalent (1 Deribit BTC option = 1 BTC). A BTC-equivalent strike is the IBIT strike divided by the same ratio. The dated fallback is the 13 Nov 2026 holdings file and is used only when the live CSV cannot be fetched.
- **Greeks / IV:** IV is each venue's own (Cboe mid-quote IV; Deribit mark IV). Delta & gamma are **recomputed** with Black-Scholes
  (r = 0) from that IV on both sides so they are strictly comparable. IBIT forward = spot; Deribit forward = each expiry's underlying price.
- **IBIT IV is only used where there is a live two-sided quote.** Deep ITM / illiquid strikes show IV = 0 at Cboe and are excluded.
- **Surface / smile** use OTM options only (puts below the forward, calls above) and linear interpolation; blanks mean no coverage.
- **Skew:** RR25 = IV(25Δ call) − IV(25Δ put); Fly25 = ½(IV25c + IV25p) − ATM. Compared at fixed tenors by interpolating between listed expiries.
- **Settlement differs:** IBIT options are American-style, physically settled in shares, expire 16:00 ET; Deribit options are European,
  cash-settled in BTC at 08:00 UTC. Early-exercise value and ETF premium/discount are ignored.
- **Notional:** IBIT OI × 100 × share price; Deribit OI (BTC) × index. 1 share = {btc_per_share:.6f} BTC ({_share_phrase(share_ratio)}).
- **GEX** assumes customers are net long options — an assumption, not observed dealer inventory.
- **Implied distribution** is risk-neutral and approximated by a quartic fit of the smile; treat tail probabilities as indicative.
- IBIT trades only in US hours; Deribit is 24/7, so comparisons outside those hours mix a stale IBIT snapshot with live Deribit data.
""")
