"""
IBIT — iShares Bitcoin Trust options vs Deribit BTC options.

Chain analytics from Cboe's free ~15-min-delayed option-chain snapshot
(see lib/ibit.py for scope, the "Deribit-only" exception and the maths).
The headline view is a like-for-like Deribit-vs-IBIT comparison of the vol
surface, skew, term structure, open interest and market-implied distribution;
the remaining tabs go deep on IBIT alone (surface, positioning, gamma, max
pain, activity) and expose the raw data.

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
from lib.constants import TTL_LONG, TTL_SLOW

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


def show(fig, key):
    th = _chart_theme()
    fx_style.apply_chart_theme(fig, th)
    fx_style.add_watermark(fig)
    st.plotly_chart(fig, theme=fx_style.streamlit_chart_theme(th), width="stretch", key=key,
                    config={"displaylogo": False})


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def load_ibit():
    return L.parse_cboe(L.fetch_raw("IBIT"), datetime.now(timezone.utc))


@st.cache_data(ttl=TTL_LONG, show_spinner=False)
def load_deribit():
    return L.deribit_chain(datetime.now(timezone.utc))


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
btc_per_share = ibm["spot"] / derm["spot"] if HAS_D else np.nan

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
          "🧲 Gamma & max pain", "🔥 Activity", "🗂 Data & notes"]
# Only the selected view renders (like MCM Bot): ~30 charts across six views.
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
        st.caption(f"1 IBIT share ≈ {btc_per_share:.6f} BTC (ratio of spot prices). Deribit volume is rolling 24h; "
                   "IBIT volume is the latest US session.")

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
            b = act[act.dte <= 200]
            f = go.Figure(go.Scattergl(
                x=b.strike, y=b.dte, mode="markers", text=b.volume,
                marker=dict(size=np.clip(np.sqrt(b.volume) / 4, 5, 40), color=np.where(b.type == "C", CALL, PUT),
                            opacity=.65, line=dict(width=0)),
                hovertemplate="K %{x} · %{y:.0f}d<br>%{text:,.0f} contracts<extra></extra>"))
            f.add_vline(x=ibm["spot"], line=dict(color=MUTED, dash="dash"))
            show(sty(f, 440, "Volume bubbles: strike × DTE", "Strike", "Days to expiry")
                 .update_layout(hovermode="closest"), "act_bub")
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
# TAB 6 — Data & notes
# ===========================================================================
if _view == _VIEWS[5]:
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
  It has **no per-trade tape**; OPRA trade data is licensed. This page is therefore *chain* analytics, not block-flow analytics.
  This is a deliberate exception to the app's Deribit-only rule (like the Fear & Greed page).
- **Greeks / IV:** IV is each venue's own (Cboe mid-quote IV; Deribit mark IV). Delta & gamma are **recomputed** with Black-Scholes
  (r = 0) from that IV on both sides so they are strictly comparable. IBIT forward = spot; Deribit forward = each expiry's underlying price.
- **IBIT IV is only used where there is a live two-sided quote.** Deep ITM / illiquid strikes show IV = 0 at Cboe and are excluded.
- **Surface / smile** use OTM options only (puts below the forward, calls above) and linear interpolation; blanks mean no coverage.
- **Skew:** RR25 = IV(25Δ call) − IV(25Δ put); Fly25 = ½(IV25c + IV25p) − ATM. Compared at fixed tenors by interpolating between listed expiries.
- **Settlement differs:** IBIT options are American-style, physically settled in shares, expire 16:00 ET; Deribit options are European,
  cash-settled in BTC at 08:00 UTC. Early-exercise value and ETF premium/discount are ignored.
- **Notional:** IBIT OI × 100 × share price; Deribit OI (BTC) × index. 1 share ≈ {btc_per_share:.6f} BTC.
- **GEX** assumes customers are net long options — an assumption, not observed dealer inventory.
- **Implied distribution** is risk-neutral and approximated by a quartic fit of the smile; treat tail probabilities as indicative.
- IBIT trades only in US hours; Deribit is 24/7, so comparisons outside those hours mix a stale IBIT snapshot with live Deribit data.
""")
