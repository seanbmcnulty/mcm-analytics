"""Plain-English takeaways, readable tables and the headline verdict for
pages/14_VIX_vs_DVOL.py. Every sentence is computed from the stats passed in;
only the "How to read this" notes (HOW) are fixed text.

Pure functions (pandas / numpy only), tested in tests/test_vol_lead.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA = 0.05
YES, NO = "✅", "—"


# ------------------------------------------------------------------ formatting
def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "p n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"


def sig_label(p: float, alpha: float = ALPHA, yes: str = "significant", no: str = "no") -> str:
    """'✅ significant (p=0.012)' / '— no (p=0.35)'."""
    if p is None or not np.isfinite(p):
        return f"{NO} n/a"
    return f"{YES} {yes} ({fmt_p(p)})" if p < alpha else f"{NO} {no} ({fmt_p(p)})"


def lag_txt(v: float, unit: str) -> str:
    v = float(v)
    if unit == "min":
        return f"{int(round(v))} min"
    if unit == "h":
        return f"{v:g} h"
    return f"{int(round(v))} day" + ("" if abs(v) == 1 else "s")


def _f(v, fmt: str, na: str = "n/a") -> str:
    try:
        return fmt.format(v) if v is not None and np.isfinite(v) else na
    except (TypeError, ValueError):
        return na


def style_verdicts(styler, cols=None):
    """Green background for ✅ cells, grey text for '—' cells."""
    def _c(v):
        if isinstance(v, str) and v.startswith(YES):
            return "background-color: rgba(14,122,56,0.14); color: #0B5A2A; font-weight: 600"
        if isinstance(v, str) and v.startswith(NO):
            return "color: #8A8A8A"
        return ""
    return styler.map(_c, subset=cols) if cols is not None else styler.map(_c)


# ------------------------------------------------------------------ lead-lag
def _lag_x(cc: pd.DataFrame, unit: str) -> pd.Series:
    if unit == "min":
        return cc["minutes"]
    if unit == "h":
        return cc["minutes"] / 60
    return cc["lag"]


def leadlag_stats(cc: pd.DataFrame, unit: str) -> dict:
    """Same-time corr, the strongest positive VIX lead, which lags clear the band."""
    if cc is None or cc.empty or cc["n"].max() <= 0:
        return {}
    cc = cc.assign(x=_lag_x(cc, unit).values)
    c0 = cc.loc[cc["lag"] == 0].iloc[0]
    lead = cc[cc["lag"] > 0]
    rev = cc[cc["lag"] < 0]
    sig_lead = lead[lead["corr"] > lead["band"]]
    sig_neg = lead[lead["corr"] < -lead["band"]]
    sig_rev = []
    for _, row in rev.sort_values("lag", ascending=False).iterrows():  # -1, -2, ... unbroken
        if abs(row["corr"]) > row["band"]:
            sig_rev.append(row)
        else:
            break
    sig_rev = pd.DataFrame(sig_rev, columns=cc.columns)
    best = lead.loc[lead["corr"].abs().idxmax()]
    # A lead only counts if it starts at the first lag and runs unbroken; isolated bars further out are
    # what 1 in 20 noise bars look like.
    run = []
    for _, row in lead.sort_values("lag").iterrows():
        if row["corr"] > row["band"]:
            run.append(row)
        else:
            break
    run = pd.DataFrame(run)
    peak = run.loc[run["corr"].idxmax()] if not run.empty else None
    run_x = [float(v) for v in run["x"]] if not run.empty else []
    isolated = [float(v) for v in sig_lead["x"] if float(v) not in run_x]
    l1 = lead.loc[lead["lag"] == 1].iloc[0] if (lead["lag"] == 1).any() else None
    return {"same": float(c0["corr"]), "same_sig": bool(abs(c0["corr"]) > c0["band"]),
            "band": float(cc["band"].median()), "best_x": float(best["x"]), "best_corr": float(best["corr"]),
            "peak_x": float(peak["x"]) if peak is not None else None,
            "peak_corr": float(peak["corr"]) if peak is not None else None,
            "sig_x": run_x, "isolated_x": isolated, "neg_x": [float(v) for v in sig_neg["x"]],
            "rev_x": [float(v) for v in sig_rev["x"]],
            "lag1_corr": float(l1["corr"]) if l1 is not None else None,
            "lag1_sig": bool(abs(l1["corr"]) > l1["band"]) if l1 is not None else False, "unit": unit}


def leadlag_takeaway(cc: pd.DataFrame, unit: str, asset: str, pred: str = "VIX") -> str:
    s = leadlag_stats(cc, unit)
    if not s:
        return "Not enough data."
    same = f"same-{'bar' if unit == 'min' else 'hour' if unit == 'h' else 'day'} corr {s['same']:+.2f}"
    iso = (f" Isolated significant bar{'s' if len(s['isolated_x']) > 1 else ''} at "
           f"{', '.join(lag_txt(v, unit) for v in s['isolated_x'][:4])} (likely noise)." if s["isolated_x"] else "")
    if s["peak_x"] is None:
        first = lag_txt(_lag_x(cc, unit)[cc["lag"] == 1].iloc[0], unit) if (cc["lag"] == 1).any() else "the first lag"
        return (f"No meaningful {pred} lead: the {first} lead is {s['lag1_corr']:+.2f}, inside the noise band "
                f"(±{s['band']:.2f}). The {same} is co-movement, not prediction.{iso}")
    txt = (f"{pred} leads {asset} DVOL by about {lag_txt(s['peak_x'], unit)} (corr {s['peak_corr']:+.2f} vs noise "
           f"±{s['band']:.2f}); ")
    last = max(s["sig_x"])
    txt += (f"the lead stays significant out to {lag_txt(last, unit)}, then fades. " if last > s["peak_x"]
            else "no meaningful lead beyond that. ")
    txt += same[0].upper() + same[1:] + "." + iso
    if s["rev_x"]:
        txt += f" DVOL also leads {pred} at {', '.join(lag_txt(-v, unit) for v in s['rev_x'][:4])}."
    return txt


# ------------------------------------------------------------------ Granger
def granger_table(gr: pd.DataFrame, gr_rev: pd.DataFrame, unit: str, pred: str = "VIX") -> pd.DataFrame:
    x = _lag_x(gr, unit)
    return pd.DataFrame({
        "Past moves included": [f"up to {lag_txt(v, unit)}" for v in x],
        f"Past {pred} helps predict DVOL?": [sig_label(p, yes="yes") for p in gr["p_value"]],
        f"Past DVOL helps predict {pred}?": [sig_label(p, yes="yes") for p in gr_rev["p_value"]],
        "Observations": gr["n"].astype(int).values})


def granger_takeaway(gr: pd.DataFrame, gr_rev: pd.DataFrame, unit: str, asset: str, pred: str = "VIX") -> str:
    if gr is None or gr.empty or not gr["p_value"].notna().any():
        return "Not enough data."
    pf, pr = gr["p_value"].min(), gr_rev["p_value"].min() if gr_rev is not None and not gr_rev.empty else np.nan
    xf = _lag_x(gr, unit)[gr["p_value"].idxmin()]
    fwd = (f"Past {pred} moves help predict {asset} DVOL ({fmt_p(pf)}, best with up to {lag_txt(xf, unit)} of history)"
           if pf < ALPHA else f"Past {pred} moves do not help predict {asset} DVOL (best {fmt_p(pf)})")
    if np.isfinite(pr) and pr < ALPHA:
        rev = f"; DVOL also helps predict {pred} ({fmt_p(pr)}), so the information flows both ways."
    else:
        rev = f"; the reverse is not significant ({fmt_p(pr)})."
    return fwd + rev


# ------------------------------------------------------------------ forecast ranking
def ranking_plain(rank: pd.DataFrame) -> pd.DataFrame:
    """One row per predictor, sorted by its best forecast gain; verdict + gain per horizon."""
    if rank is None or rank.empty:
        return pd.DataFrame()
    gain = rank.pivot_table(index="label", columns="h", values="oos_r2_gain_pct")
    p = rank.pivot_table(index="label", columns="h", values="cw_p")
    rows = []
    for lab in gain.index:
        g, pp = gain.loc[lab], p.loc[lab]
        good = [h for h in gain.columns if g[h] > 0 and pp[h] < ALPHA]
        hb = max(good, key=lambda h: g[h]) if good else g.idxmax()
        row = {"Predictor": lab,
               "Improves forecast?": (f"{YES} yes, {', '.join(f'{h}d' for h in good)}" if good else f"{NO} no"),
               "Best forecast gain %": float(g[hb]), "Best at": f"{hb}d", "_p": float(pp[hb])}
        for h in gain.columns:
            row[f"{h}d gain %"] = float(g[h])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_good"] = out["Improves forecast?"].str.startswith(YES)
    out = out.sort_values(["_good", "Best forecast gain %"], ascending=[False, False]).drop(columns=["_good"])
    out["Significance (best)"] = [sig_label(v) for v in out.pop("_p")]
    return out.reset_index(drop=True)


def ranking_sig_mask(rank: pd.DataFrame, plain: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame aligned to ``plain``: True where that horizon's gain is positive and CW p < 0.05."""
    p = rank.pivot_table(index="label", columns="h", values="cw_p")
    g = rank.pivot_table(index="label", columns="h", values="oos_r2_gain_pct")
    m = pd.DataFrame(False, index=plain.index, columns=plain.columns)
    for i, lab in plain["Predictor"].items():
        for h in p.columns:
            c = f"{h}d gain %"
            if c in m:
                m.loc[i, c] = bool(p.loc[lab, h] < ALPHA and g.loc[lab, h] > 0)
    return m


def ranking_takeaway(rank: pd.DataFrame, asset: str) -> str:
    t = ranking_plain(rank)
    if t.empty:
        return "Not enough data."
    good = t[t["Improves forecast?"].str.startswith(YES)]
    vix = t[t["Predictor"] == "VIX"]
    vtxt = ""
    if not vix.empty:
        v = vix.iloc[0]
        vtxt = (f" VIX itself helps ({v['Best forecast gain %']:+.1f}% at {v['Best at']})." if v["Improves forecast?"].startswith(YES)
                else f" VIX itself never helps (best {v['Best forecast gain %']:+.1f}%).")
    if good.empty:
        return (f"No TradFi gauge significantly improves {asset} DVOL forecasts over DVOL's own history "
                f"(best: {t.iloc[0]['Predictor']} {t.iloc[0]['Best forecast gain %']:+.1f}% at {t.iloc[0]['Best at']})." + vtxt)
    b = good.iloc[0]
    size = "small" if b["Best forecast gain %"] < 2 else "meaningful"
    return (f"Best add-on: {b['Predictor']} at {b['Best at']}, {b['Best forecast gain %']:+.1f}% forecast gain "
            f"({b['Significance (best)'].split('(')[-1].rstrip(')')}), which is {size}; "
            f"{len(good)} of {len(t)} gauges help at some horizon." + vtxt)


# ------------------------------------------------------------------ predictive regressions
def regression_plain(regs: list, pred: str = "VIX") -> pd.DataFrame:
    rows = []
    for r in regs:
        if not r.get("ok"):
            continue
        t = r["t"].get(f"d{pred}", np.nan)
        rows.append({"Horizon": f"{r['h']} day" + ("" if r["h"] == 1 else "s"),
                     f"DVOL move per 1-pt {pred} move today": r["coef"].get(f"d{pred}", np.nan),
                     "Significant in-sample?": (f"{YES} yes (t {t:+.1f})" if abs(t) > 1.96 else f"{NO} no (t {t:+.1f})"),
                     "Forecast gain vs DVOL-only %": 100 * r["oos_r2_vs_base"],
                     "Beats DVOL-only out-of-sample?": sig_label(r["cw_p"], yes="yes") if r["oos_r2_vs_base"] > 0
                     else f"{NO} no ({fmt_p(r['cw_p'])})",
                     "Days": int(r["n"])})
    return pd.DataFrame(rows)


def regression_takeaway(regs: list, asset: str, pred: str = "VIX") -> str:
    t = regression_plain(regs, pred)
    if t.empty:
        return "Not enough data."
    good = t[t["Beats DVOL-only out-of-sample?"].str.startswith(YES)]
    best = t.loc[t["Forecast gain vs DVOL-only %"].idxmax()]
    if good.empty:
        return (f"Today's {pred} move does not improve {asset} DVOL forecasts at any horizon "
                f"(best {best['Forecast gain vs DVOL-only %']:+.1f}% at {best['Horizon']}).")
    g = good.loc[good["Forecast gain vs DVOL-only %"].idxmax()]
    return f"Today's {pred} move improves {asset} DVOL forecasts at {g['Horizon']} ({g['Forecast gain vs DVOL-only %']:+.1f}%)."


# ------------------------------------------------------------------ rich / cheap
ZONES = ((0, 20, "Cheap"), (20, 40, "Slightly cheap"), (40, 60, "Fair"), (60, 80, "Slightly rich"), (80, 100, "Rich"))


def zone(pct: float) -> str:
    if pct is None or not np.isfinite(pct):
        return "n/a"
    for lo, hi, name in ZONES:
        if pct <= hi:
            return name
    return "Rich"


def component_signal(z: float) -> str:
    if z is None or not np.isfinite(z):
        return "n/a"
    return "Rich" if z > 1 else "Cheap" if z < -1 else "Slightly rich" if z > 0.5 else "Slightly cheap" if z < -0.5 else "Fair"


def rich_cheap_takeaway(rc: pd.DataFrame, asset: str) -> str:
    s = rc["score"].dropna()
    if s.empty:
        return "Not enough history."
    rich, cheap = 100 * (s > 1).mean(), 100 * (s < -1).mean()
    last = rc.dropna(subset=["score"]).iloc[-1]
    word = "above" if last["resid"] > 0 else "below"
    return (f"{asset} DVOL is {abs(last['resid']):.1f} vols {word} its TradFi fair value ({last['fair']:.1f}); score "
            f"{last['score']:+.2f} sd. Over the window it was rich (> +1 sd) {rich:.0f}% of days and cheap (< −1 sd) "
            f"{cheap:.0f}%.")


def quintile_takeaway(q: pd.DataFrame, asset: str) -> str:
    if q is None or q.empty:
        return "Not enough data."
    h = int(q["h"].max())
    d = q[q["h"] == h].set_index("quintile")["fwd_dvol_mean"]
    if "Q1" not in d or "Q5" not in d:
        return "Not enough data."
    spread = d["Q1"] - d["Q5"]
    shape = ("cheap readings were followed by higher DVOL than rich ones, as mean reversion predicts"
             if spread > 0 else "rich readings were NOT followed by lower DVOL than cheap ones")
    return (f"Over the next {h} days {asset} DVOL moved {d['Q1']:+.1f} vols after the cheapest fifth of readings and "
            f"{d['Q5']:+.1f} after the richest: {shape}.")


def signal_plain(sig: pd.DataFrame) -> pd.DataFrame:
    if sig is None or sig.empty:
        return pd.DataFrame()
    rows = []
    for _, r in sig.iterrows():
        down = r["slope"] < 0 and abs(r["t"]) > 1.96
        rows.append({"What is forecast": "DVOL" if r["target"] == "DVOL" else "DVOL − VIX spread",
                     "Horizon": f"{int(r['h'])} days",
                     "Rich → falls? (vols per +1 sd)": (f"{YES} yes, {r['slope']:+.1f} (t {r['t']:+.1f})" if down
                                                        else f"{NO} no, {r['slope']:+.1f} (t {r['t']:+.1f})"),
                     "Forecast gain % (out-of-sample)": r["oos_r2_pct"],
                     "Beats naive forecast?": sig_label(r["cw_p"], yes="yes") if r["oos_r2_pct"] > 0
                     else f"{NO} no ({fmt_p(r['cw_p'])})"})
    return pd.DataFrame(rows)


def signal_takeaway(sig: pd.DataFrame, asset: str) -> str:
    if sig is None or sig.empty:
        return "Not enough data."
    d = sig[sig["target"] == "DVOL"]
    if d.empty:
        return "Not enough data."
    b = d.sort_values("oos_r2_pct", ascending=False).iloc[0]
    if b["oos_r2_pct"] > 0 and b["cw_p"] < ALPHA and b["slope"] < 0:
        return (f"Yes: a rich score has been followed by falling {asset} DVOL; forecast gain {b['oos_r2_pct']:+.1f}% "
                f"at {int(b['h'])} days ({fmt_p(b['cw_p'])}).")
    return (f"No reliable signal: best out-of-sample gain {b['oos_r2_pct']:+.1f}% at {int(b['h'])} days "
            f"({fmt_p(b['cw_p'])}).")


def today_takeaway(fv: pd.DataFrame, asset: str) -> str:
    if fv is None or fv.empty:
        return "No intraday data."
    r = fv["resid"]
    now = float(r.iloc[-1])
    word = "rich" if now > 0 else "cheap"

    def _w(v):
        return f"{abs(v):.1f} {'rich' if v > 0 else 'cheap'}"
    return (f"Latest 5-min bar: {asset} DVOL {fv['DVOL'].iloc[-1]:.1f} vs fair {fv['fair'].iloc[-1]:.1f}, "
            f"{abs(now):.1f} vols {word}. During the session it ranged from {_w(r.min())} to {_w(r.max())}.")


# ------------------------------------------------------------------ levels / short horizon
def levels_takeaway(df: pd.DataFrame, asset: str) -> str:
    last = df.dropna(subset=["DVOL", "VIX"]).iloc[-1]
    sp = df["DVOL"] - df["VIX"]
    corr = df["DVOL"].corr(df["VIX"])
    mv = f", MOVE {last['MOVE']:.0f}" if "MOVE" in df and np.isfinite(last.get("MOVE", np.nan)) else ""
    return (f"Now: {asset} DVOL {last['DVOL']:.1f}, VIX {last['VIX']:.1f}{mv}. DVOL sits {sp.iloc[-1]:+.1f} pts over VIX "
            f"(window average {sp.mean():+.1f}); level correlation over the window {corr:+.2f}.")


def spread_takeaway(df: pd.DataFrame, asset: str) -> str:
    r = df["RATIO_VIX"].dropna()
    if r.empty:
        return "Not enough data."
    pct = 100 * (r <= r.iloc[-1]).mean()
    return (f"{asset} DVOL is {r.iloc[-1]:.2f}x VIX, vs a window average of {r.mean():.2f}x "
            f"(range {r.min():.2f}x to {r.max():.2f}x; now at the {int(round(pct))}th percentile).")


def overlay_takeaway(view: pd.DataFrame, asset: str, bar: pd.Timedelta) -> str:
    col = f"DVOL_{asset}"
    v = view[["VIX", col]].dropna()
    if len(v) < 3:
        return "Not enough data."
    dx = v["VIX"].diff()
    dy = v[col].diff()
    ok = (v.index.to_series().diff() == bar)
    c = dx[ok].corr(dy[ok])
    return (f"Over this view VIX went {v['VIX'].iloc[0]:.1f} → {v['VIX'].iloc[-1]:.1f} "
            f"({v['VIX'].iloc[-1] - v['VIX'].iloc[0]:+.1f}) and {asset} DVOL {v[col].iloc[0]:.1f} → {v[col].iloc[-1]:.1f} "
            f"({v[col].iloc[-1] - v[col].iloc[0]:+.1f}); correlation of their {int(bar / pd.Timedelta(minutes=1))}-min "
            f"moves {c:+.2f}.")


def norm_takeaway(z: pd.DataFrame, asset: str, span: str) -> str:
    col = f"DVOL_{asset}"
    if col not in z or "VIX" not in z or z[[col, "VIX"]].dropna().empty:
        return "Not enough data."
    zz = z[[col, "VIX"]].dropna()
    c = zz[col].corr(zz["VIX"])
    gap = zz[col].iloc[-1] - zz["VIX"].iloc[-1]
    rel = ("in line with" if abs(gap) < 0.5 else "high relative to" if gap > 0 else "low relative to")
    return (f"Over the last {span} the levels correlate {c:+.2f}. Now {asset} DVOL is {zz[col].iloc[-1]:+.1f} sd and VIX "
            f"{zz['VIX'].iloc[-1]:+.1f} sd from their averages, so DVOL is {rel} VIX.")


def scatter_takeaway(fit: dict, asset: str, label: str) -> str:
    b, c, t = fit.get("beta", np.nan), fit.get("corr", np.nan), fit.get("t", np.nan)
    if not np.isfinite(b):
        return "Not enough data."
    sig = "significant" if abs(t) > 1.96 else "not significant"
    return (f"A 1-pt VIX move comes with a {b:+.2f} vol move in {asset} DVOL in the same {label.lower()} bar "
            f"(corr {c:+.2f}, t {t:.1f}, {sig}; {fit.get('n', 0):,} bars). Low corr = most DVOL moves are "
            f"not explained by VIX.")


def rolling_takeaway(rbs: dict, asset: str) -> str:
    parts = []
    for name, rb in rbs.items():
        c = rb["corr"].dropna()
        if c.empty:
            continue
        parts.append(f"{name.split(' (')[0]} corr {c.iloc[-1]:+.2f} now (window range {c.min():+.2f} to {c.max():+.2f})")
    if not parts:
        return "Not enough data."
    return "; ".join(parts) + ". A rising line means DVOL is tracking VIX more closely."


def rolling_daily_takeaway(rb: pd.DataFrame, asset: str, pred: str) -> str:
    c, b = rb["corr"].dropna(), rb["beta"].dropna()
    if c.empty:
        return "Not enough data."
    return (f"Same-day correlation of daily {pred} and {asset} DVOL moves is {c.iloc[-1]:+.2f} now "
            f"(window range {c.min():+.2f} to {c.max():+.2f}); beta {b.iloc[-1]:+.2f} DVOL pts per {pred} pt.")


def event_takeaway(path: pd.DataFrame, name: str, asset: str, at: tuple, unit: str) -> str:
    """Average DVOL change at the listed offsets (days or minutes) after the event."""
    if path is None or path.empty or path["n"].max() == 0:
        return f"No {name} events in this window."
    key = "minutes" if unit == "min" else "offset"
    p = path.set_index(key)
    bits = []
    for o in at:
        if o in p.index and np.isfinite(p.loc[o, "mean"]):
            sig = p.loc[o, "lo"] > 0 or p.loc[o, "hi"] < 0
            when = ("on the day" if o == 0 else f"after {o} days") if unit == "d" else \
                   ("in the jump bar" if o == 0 else f"after {o // 60}h" if o >= 60 and o % 60 == 0 else f"after {o} min")
            bits.append(f"{p.loc[o, 'mean']:+.1f} vols {when}" + ("" if sig else " (not significant)"))
    return f"After {int(path['n'].max())} {name}, {asset} DVOL averaged " + ", ".join(bits) + "."


def session_plain(sp: pd.DataFrame) -> pd.DataFrame:
    if sp is None or sp.empty:
        return pd.DataFrame()
    rows = []
    for _, r in sp.iterrows():
        rows.append({"DVOL change over": r["target"].split(" (")[0],
                     "DVOL move per 1-pt morning VIX move": r["beta_vix"],
                     "Significant in-sample?": (f"{YES} yes (t {r['t_vix']:+.1f})" if abs(r["t_vix"]) > 1.96
                                                else f"{NO} no (t {r['t_vix']:+.1f})"),
                     "Forecast gain % vs DVOL-only": r["oos_r2_vs_own_pct"],
                     "Beats DVOL-only out-of-sample?": sig_label(r["cw_p"], yes="yes") if r["oos_r2_vs_own_pct"] > 0
                     else f"{NO} no ({fmt_p(r['cw_p'])})",
                     "Direction right %": r["same_sign_pct"], "Sessions": int(r["n"])})
    return pd.DataFrame(rows)


def session_takeaway(sp: pd.DataFrame, asset: str) -> str:
    t = session_plain(sp)
    if t.empty:
        return "Not enough data."
    good = t[t["Beats DVOL-only out-of-sample?"].str.startswith(YES)]
    if good.empty:
        return (f"No: the 10:00→12:00 New York VIX move does not forecast {asset} DVOL for the rest of the session, "
                f"overnight or to the next close (forecast gains {', '.join(f'{v:+.1f}%' for v in t['Forecast gain % vs DVOL-only'])}).")
    g = good.iloc[0]
    return f"Yes for {g['DVOL change over']}: forecast gain {g['Forecast gain % vs DVOL-only']:+.1f}%."


# ------------------------------------------------------------------ headline verdict
def verdict_bullets(asset: str, lookback: str, cc5=None, cch=None, cc_daily=None, gr_daily=None, sp=None,
                    rank=None, rc=None, pct=None, sig=None) -> list[str]:
    """3-4 plain bullets for the top of the page."""
    out = []
    s5 = leadlag_stats(cc5, "min") if cc5 is not None else {}
    sh = leadlag_stats(cch, "h") if cch is not None else {}
    if s5:
        if s5["peak_x"] is not None:
            txt = (f"**Minutes, not hours.** VIX leads {asset} DVOL by about {lag_txt(s5['peak_x'], 'min')} on 5-min bars "
                   f"(corr {s5['peak_corr']:+.2f} vs noise ±{s5['band']:.2f}), fading out by "
                   f"{lag_txt(max(s5['sig_x']), 'min')}.")
        else:
            txt = f"**No intraday lead.** 5-min VIX moves do not lead {asset} DVOL beyond noise."
        if sh and sh.get("lag1_corr") is not None:
            txt += (f" On hourly bars the 1-hour lead is {sh['lag1_corr']:+.2f} "
                    f"({'significant' if sh['lag1_sig'] else 'not significant'}).")
        out.append(txt)
    sd = leadlag_stats(cc_daily, "d") if cc_daily is not None else {}
    bits = []
    if sd:
        bits.append(f"daily VIX moves move with DVOL on the same day (corr {sd['same']:+.2f})" +
                    (f" but do not lead it (largest lagged corr {sd['best_corr']:+.2f}, noise ±{sd['band']:.2f})"
                     if sd["peak_x"] is None else f" and lead it by {lag_txt(sd['peak_x'], 'd')} (corr {sd['peak_corr']:+.2f})"))
    if sp is not None and not sp.empty:
        r = sp.iloc[0]
        bits.append(f"the morning VIX move {'does' if (r['oos_r2_vs_own_pct'] > 0 and r['cw_p'] < ALPHA) else 'does not'} "
                    f"forecast DVOL for the rest of the session (forecast gain {r['oos_r2_vs_own_pct']:+.1f}%)")
    if bits:
        head = "**No day-ahead edge.** " if sd and sd["peak_x"] is None else "**Daily.** "
        out.append(head + "; ".join(bits)[0].upper() + "; ".join(bits)[1:] + ".")
    if rank is not None and not rank.empty:
        out.append("**Best TradFi predictor.** " + ranking_takeaway(rank, asset))
    if rc is not None and not rc.dropna(subset=["score"]).empty:
        last = rc.dropna(subset=["score"]).iloc[-1]
        word = "rich" if last["resid"] > 0 else "cheap"
        ptxt = f", {_ordinal(pct)} percentile of the last {lookback}" if pct is not None and np.isfinite(pct) else ""
        txt = (f"**Right now {asset} vol is {abs(last['resid']):.1f} vols {word}** vs TradFi fair value "
               f"(score {last['score']:+.2f} sd{ptxt}).")
        if sig is not None and not sig.empty:
            d = sig[sig["target"] == "DVOL"]
            if not d.empty:
                b = d.sort_values("oos_r2_pct", ascending=False).iloc[0]
                if b["oos_r2_pct"] > 0 and b["cw_p"] < ALPHA and b["slope"] < 0:
                    txt += (f" The gauge has mean-reverted: forecast gain {b['oos_r2_pct']:+.1f}% at {int(b['h'])} days "
                            f"({fmt_p(b['cw_p'])}).")
                else:
                    txt += f" The gauge has not reliably forecast DVOL in this window (best gain {b['oos_r2_pct']:+.1f}%)."
        out.append(txt)
    return out[:4]


def _ordinal(x: float) -> str:
    n = min(max(int(round(float(x))), 0), 100)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


# ------------------------------------------------------------------ "How to read this"
HOW = {
    "gauge": "The marker shows where today's composite score sits among all days in the analysis window "
             "(0 = cheapest day, 100 = richest). Green zones = crypto vol cheap vs what VIX and MOVE imply, red = rich.",
    "rich_cheap": "Top: the orange line is DVOL, the dashed navy line is the fair value implied by a rolling regression "
                  "on VIX and MOVE; red shading = DVOL above fair (rich), green = below (cheap). Bottom: the composite "
                  "score in standard deviations; beyond ±1 sd counts as clearly rich or cheap.",
    "quintiles": "Days are sorted by the rich/cheap score into five equal buckets. Each bar is the average DVOL change "
                 "over the following 5, 10 or 20 days. If the gauge mean-reverts, bars fall from left (cheap) to right (rich).",
    "signal": "Regression of the future DVOL change on today's score. A negative slope means rich vol tends to "
              "cheapen. 'Beats naive forecast?' is the real-time test: forecasts built only from data known at the "
              "time, compared with a historical-mean forecast (Clark-West test).",
    "today": "Same fair-value model as above, applied bar by bar to live VIX (MOVE held at its last close). "
             "Shows whether crypto vol cheapened or richened vs TradFi during the current US session.",
    "leadlag5": "Each bar is the correlation between VIX's 5-min move and DVOL's move N minutes later. Bars right of 0 "
                "= VIX moved first; left of 0 = DVOL moved first. Bars inside the grey band are noise; coloured bars "
                "are significant at 95%.",
    "leadlagh": "Same as the 5-min chart but with hourly moves. If the right-hand bars are grey, the VIX lead is "
                "shorter than an hour.",
    "leadlagd": "Correlation between today's change in the predictor and the DVOL change N days later. Right of 0 = the "
                "predictor moved first. Bars inside the grey band are noise.",
    "granger": "Granger test: do past moves of one series improve a forecast of the other beyond its own past? "
               "✅ = yes at 95% confidence. It tests timing, not cause.",
    "session": "For each session, regress DVOL's change after 12:00 New York on VIX's move from 10:00 to 12:00 "
               "(controlling for DVOL's own morning move). The out-of-sample columns test whether that would have "
               "helped in real time.",
    "overlay": "DVOL (left axis) and VIX (right axis) during US cash hours, overnight gaps removed. Look for whether "
               "the VIX line turns first.",
    "norm": "Both series rescaled to standard deviations from their own average over the window, so the shapes "
            "can be compared directly. A gap between the lines = one is high or low relative to the other.",
    "scatter": "Each dot is one bar: VIX's move (x) against DVOL's move (y) over the same bar. The slope is the beta; "
               "a tight cloud means a strong link, a round cloud a weak one.",
    "rolling_i": "Correlation and beta of hourly VIX vs DVOL moves over a moving 1-week and 1-month window. Shows "
                 "whether the link is getting stronger or weaker.",
    "events_i": "Average DVOL path around every 5-min VIX move bigger than 3 trailing sd, measured from the bar "
                "before the move. Shaded = 95% confidence band; if the band includes 0 the move is not significant.",
    "levels": "Levels on the 16:00 New York close. MOVE (Treasury vol, bp) is on the right axis because its units differ.",
    "spreads": "DVOL minus VIX is the vol-point premium crypto pays over equities; the ratios scale-adjust it. "
               "The dashed line is the window average.",
    "events_d": "Average DVOL path around each TradFi vol spike day (day 0), measured from the day before. Shaded = "
                "95% band; the dotted line is DVOL's average drift on all days for comparison.",
    "rolling_d": "Rolling same-day correlation and beta of daily moves. Shows when DVOL has been more or less tied to TradFi vol.",
    "regression": "Regression of the DVOL change over the next h days on today's VIX move, VIX level and DVOL−VIX spread, "
                  "with DVOL's own move as a control. 'Forecast gain' compares real-time forecasts with a DVOL-only model.",
    "ranking": "Each cell is how much adding that gauge improves real-time DVOL forecasts versus a DVOL-only model "
               "(out-of-sample R² gain). Green = helps, red = hurts; ✅ = significant (Clark-West p < 0.05).",
}
