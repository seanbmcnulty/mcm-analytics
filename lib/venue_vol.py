"""
Deribit vs OKX implied-vol comparison (BTC, ETH): ATM term structure, forward vol,
25-delta risk reversal and 10-delta butterfly term structures.

Both venues are normalised into the ``lib.ibit`` chain schema so ``ibit.expiry_metrics`` runs
identically on each:

* Deribit - ``get_book_summary_by_currency`` ``mark_iv`` (inverse BTC / ETH options).
* OKX     - public ``/api/v5/public/opt-summary`` ``markVol`` + per-instrument ``fwdPx``, INVERSE family
            only (``BTC-USD-...``, same coin-margined structure as Deribit).  The USD-settled ``_UM`` family is
            deliberately excluded: it carries its own forward and a different (much flatter) smile model, and
            mixing it in corrupts the skew (checked 2026-10-30: BTC 270326 RR25 +1.9 merged vs -2.0 inverse-only).

Deltas are recomputed from each venue's own mark IV with Black-Scholes (r = 0) on the venue's own forward,
so 25d / 10d are defined identically on both sides.  Definitions (vol points):

    ATM   = OTM-smile IV interpolated at K = F
    RR25  = IV(25d call) - IV(25d put)
    Fly10 = (IV(10d call) + IV(10d put)) / 2 - ATM
    Fwd   = sqrt((w2 - w1) / (T2 - T1)),  w = ATM^2 * T   (forward ATM vol between two tenors)

Term structures are put on a common tenor grid (ATM by linear interpolation in total variance, RR / fly
linear in DTE; no extrapolation) so the venues are compared at identical maturities.

Pure functions up top (offline-testable); Streamlit / plotly only inside ``render``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from lib import ibit, okx

UTC = timezone.utc
ASSETS = ["BTC", "ETH"]
TENORS = [7, 14, 21, 30, 45, 60, 90, 120, 180, 270]
YEAR = 365.25
DERIBIT_C, OKX_C = "#3790C7", "#E8710A"


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------
def okx_chain(asset: str, now: datetime | None = None) -> tuple[pd.DataFrame, dict]:
    """OKX option summary -> ibit schema."""
    now = now or datetime.now(UTC)
    data = okx._get("/public/opt-summary", {"uly": f"{asset}-USD"}).get("data", [])
    rows = []
    for x in data:
        base, exp, strike, cp = okx.parse_option(x.get("instId"))
        if "_UM" in (x.get("instId") or ""):
            continue
        if base != asset or cp is None or pd.isna(exp):
            continue
        iv, fwd = okx._f(x.get("markVol")), okx._f(x.get("fwdPx"))
        if not (np.isfinite(iv) and iv > 0 and np.isfinite(fwd) and fwd > 0):
            continue
        rows.append(dict(venue="OKX", expiry=pd.Timestamp(exp).tz_localize(UTC) + pd.Timedelta(hours=8),
                         type=cp, strike=strike, fwd=fwd, spot=np.nan, iv=iv,
                         bid=okx._f(x.get("bidVol")), ask=okx._f(x.get("askVol")),
                         volume=0.0, oi=0.0, mult=1))
    if not rows:
        raise RuntimeError("OKX option summary unavailable")
    df = pd.DataFrame(rows).drop_duplicates(["expiry", "strike", "type"])
    spot = okx.fetch_spot(asset) or float(df["fwd"].median())
    df["spot"] = spot
    df = ibit._finish(df, now)
    df["ok"] = df["iv"].between(0.05, 5.0)
    return df[ibit.COLS], dict(spot=float(spot), n_contracts=len(df), n_expiries=int(df["expiry"].nunique()))


def deribit_chain(asset: str, now: datetime | None = None) -> tuple[pd.DataFrame, dict]:
    return ibit.deribit_chain(now, asset)


# ---------------------------------------------------------------------------
# Term structures
# ---------------------------------------------------------------------------
def _interp(x: np.ndarray, y: np.ndarray, t: float) -> float:
    return ibit._at(x, y, t)


def tenor_grid(metrics: pd.DataFrame, tenors: list[float] = TENORS) -> pd.DataFrame:
    """Per-expiry metrics -> fixed tenors: columns tenor, atm, rr25, fly10, w (total variance)."""
    out = {"tenor": list(tenors), "atm": [], "rr25": [], "fly10": [], "w": []}
    a = metrics[["dte", "atm"]].dropna()
    a = a[a["atm"] > 0]
    w_x = a["dte"].values.astype(float)
    w_y = (a["atm"].values ** 2) * (w_x / YEAR)
    r = metrics[["dte", "rr25"]].dropna()
    f = metrics[["dte", "fly10"]].dropna()
    for t in tenors:
        w = _interp(w_x, w_y, t)
        out["w"].append(w)
        out["atm"].append(np.sqrt(w / (t / YEAR)) if np.isfinite(w) and w > 0 else np.nan)
        out["rr25"].append(_interp(r["dte"].values.astype(float), r["rr25"].values, t))
        out["fly10"].append(_interp(f["dte"].values.astype(float), f["fly10"].values, t))
    return pd.DataFrame(out)


def forward_vols(grid: pd.DataFrame) -> pd.DataFrame:
    """Forward ATM vol between consecutive grid tenors from total variance (NaN if variance is not increasing)."""
    rows = []
    g = grid.dropna(subset=["w"]).reset_index(drop=True)
    for i in range(1, len(g)):
        t1, t2 = float(g.loc[i - 1, "tenor"]), float(g.loc[i, "tenor"])
        dw = g.loc[i, "w"] - g.loc[i - 1, "w"]
        fv = float(np.sqrt(dw / ((t2 - t1) / YEAR))) if dw > 0 else np.nan
        rows.append(dict(pair=f"{t1:g}d→{t2:g}d", t1=t1, t2=t2, fwd=fv))
    return pd.DataFrame(rows, columns=["pair", "t1", "t2", "fwd"])


def compare(d_chain: pd.DataFrame, o_chain: pd.DataFrame, tenors: list[float] = TENORS) -> dict:
    """-> {'grid': merged tenor table (Deribit/OKX/diff), 'fwd': merged forward table, 'dm','om': per-expiry metrics}."""
    dm, om = ibit.expiry_metrics(d_chain), ibit.expiry_metrics(o_chain)
    dg, og = tenor_grid(dm, tenors), tenor_grid(om, tenors)
    g = dg[["tenor"]].copy()
    for c in ("atm", "rr25", "fly10"):
        g[f"{c}_d"], g[f"{c}_o"] = dg[c].values, og[c].values
        g[f"{c}_diff"] = g[f"{c}_o"] - g[f"{c}_d"]
    fd, fo = forward_vols(dg), forward_vols(og)
    f = fd.merge(fo, on=["pair", "t1", "t2"], how="outer", suffixes=("_d", "_o")).sort_values(["t1", "t2"])
    f["diff"] = f["fwd_o"] - f["fwd_d"]
    return dict(grid=g, fwd=f.reset_index(drop=True), dm=dm, om=om)


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------
def _line(x, yd, yo, title, ytitle, xtitle="Tenor (days)", scale=100.0, bars=False):
    import plotly.graph_objects as go
    fig = go.Figure()
    for name, y, col in (("Deribit", yd, DERIBIT_C), ("OKX", yo, OKX_C)):
        y = np.asarray(y, dtype=float) * scale
        if bars:
            fig.add_bar(x=x, y=y, name=name, marker_color=col)
        else:
            fig.add_scatter(x=x, y=y, name=name, mode="lines+markers", line=dict(color=col, width=2.5),
                            marker=dict(size=7), connectgaps=True)
    fig.update_layout(title=title, height=380, margin=dict(l=40, r=20, t=60, b=40), hovermode="x unified",
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="#000000"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                      barmode="group")
    fig.update_xaxes(title=xtitle, gridcolor="#E5E5E5", type="category" if bars else "linear")
    fig.update_yaxes(title=ytitle, gridcolor="#E5E5E5", zeroline=True, zerolinecolor="#999999")
    return fig


def figures(res: dict, asset: str) -> list:
    from lib import fx_style
    g, f = res["grid"], res["fwd"]
    figs = [
        _line(g["tenor"], g["atm_d"], g["atm_o"], f"{asset} ATM IV term structure", "IV (vol %)"),
        _line(f["pair"], f["fwd_d"], f["fwd_o"], f"{asset} forward ATM vol (between tenors)", "Fwd vol (%)",
              xtitle="Forward window", bars=True),
        _line(g["tenor"], g["rr25_d"], g["rr25_o"], f"{asset} 25Δ risk reversal term structure (call IV − put IV)",
              "RR25 (vol pts)"),
        _line(g["tenor"], g["fly10_d"], g["fly10_o"], f"{asset} 10Δ butterfly term structure ((C+P)/2 − ATM)",
              "Fly10 (vol pts)"),
    ]
    for fig in figs:
        fx_style.add_watermark(fig)
    return figs


def _fmt_table(res: dict):
    g, f = res["grid"].copy(), res["fwd"].copy()
    t = pd.DataFrame({"Tenor": g["tenor"].map(lambda x: f"{x:g}d")})
    for c, lab in (("atm", "ATM"), ("rr25", "RR25"), ("fly10", "Fly10")):
        t[f"{lab} Deribit"], t[f"{lab} OKX"], t[f"{lab} Δ (OKX−D)"] = (g[f"{c}_d"] * 100, g[f"{c}_o"] * 100,
                                                                      g[f"{c}_diff"] * 100)
    ft = pd.DataFrame({"Window": f["pair"], "Fwd Deribit": f["fwd_d"] * 100, "Fwd OKX": f["fwd_o"] * 100,
                       "Fwd Δ (OKX−D)": f["diff"] * 100})
    return t, ft


def render() -> None:
    import streamlit as st
    from lib.constants import TTL_LONG

    @st.cache_data(ttl=TTL_LONG, show_spinner=False)
    def _load(asset: str):
        now = datetime.now(UTC)
        d, dmeta = deribit_chain(asset, now)
        o, ometa = okx_chain(asset, now)
        return compare(d, o), dmeta, ometa, now

    st.markdown('<div class="asset-header">⚖️ Deribit vs OKX — implied vol term structure, forward vol, '
                '25Δ risk reversal, 10Δ butterfly</div>', unsafe_allow_html=True)
    st.caption("Live mark IVs (Deribit `mark_iv`, OKX `opt-summary` `markVol`); deltas recomputed per venue with "
               "Black-Scholes (r=0) on each venue's own forward. Compared at identical tenors: ATM interpolated in "
               "total variance, RR / fly linear in DTE, no extrapolation. Forward vol = √((w₂−w₁)/(T₂−T₁)), "
               "w = ATM²·T. OKX = inverse (coin-margined) family, like Deribit.")
    tabs = st.tabs(ASSETS)
    for tab, asset in zip(tabs, ASSETS):
        with tab:
            try:
                res, dmeta, ometa, ts = _load(asset)
            except Exception as e:  # noqa: BLE001
                st.error(f"{asset}: could not load both chains — {e}")
                continue
            g = res["grid"]
            i30 = g.index[g["tenor"] == 30]
            if len(i30):
                r = g.loc[i30[0]]
                c = st.columns(4)
                c[0].metric("30d ATM  Deribit / OKX", f"{r['atm_d']*100:.1f} / {r['atm_o']*100:.1f}",
                            f"{r['atm_diff']*100:+.2f} vol", delta_color="off")
                c[1].metric("30d RR25  Deribit / OKX", f"{r['rr25_d']*100:+.1f} / {r['rr25_o']*100:+.1f}",
                            f"{r['rr25_diff']*100:+.2f} vol", delta_color="off")
                c[2].metric("30d Fly10  Deribit / OKX", f"{r['fly10_d']*100:.1f} / {r['fly10_o']*100:.1f}",
                            f"{r['fly10_diff']*100:+.2f} vol", delta_color="off")
                c[3].metric("Spot  Deribit / OKX", f"{dmeta['spot']:,.0f} / {ometa['spot']:,.0f}")
            figs = figures(res, asset)
            c1, c2 = st.columns(2)
            c1.plotly_chart(figs[0], width="stretch", key=f"vv_{asset}_atm")
            c2.plotly_chart(figs[1], width="stretch", key=f"vv_{asset}_fwd")
            c3, c4 = st.columns(2)
            c3.plotly_chart(figs[2], width="stretch", key=f"vv_{asset}_rr")
            c4.plotly_chart(figs[3], width="stretch", key=f"vv_{asset}_fly")
            t, ft = _fmt_table(res)
            fmt = {c: "{:+.2f}" if "Δ" in c or c.startswith("RR") else "{:.2f}" for c in t.columns if c != "Tenor"}
            st.dataframe(t.style.format(fmt, na_rep="–"), width="stretch", hide_index=True)
            st.dataframe(ft.style.format({c: ("{:+.2f}" if "Δ" in c else "{:.2f}") for c in ft.columns if c != "Window"},
                                         na_rep="–"), width="stretch", hide_index=True)
            st.caption(f"All values in vol points. Snapshot {ts:%Y-%m-%d %H:%M} UTC · Deribit {dmeta['n_contracts']} "
                       f"contracts / {dmeta['n_expiries']} expiries · OKX {ometa['n_contracts']} / {ometa['n_expiries']}. "
                       "Blank = tenor outside a venue's listed expiries or wing quotes missing.")
