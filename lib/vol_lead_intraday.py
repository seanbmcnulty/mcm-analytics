"""Intraday (5-minute / hourly) VIX vs DVOL analysis for pages/14_VIX_vs_DVOL.py.

Pure functions, no network. Frames are indexed by observation time (UTC,
tz-aware) and hold only US cash-session observations (lib/vol_lead_data).

Overnight handling: a change is X(t) - X(t - bar) and both points must be in
the frame, so the first bar of a session (whose previous point is the prior
day's close) has no change. Lags use the same time arithmetic, so a lag
never reaches across the overnight gap. Overnight moves are studied
separately in :func:`session_predict`.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from lib import vol_lead as vl

NY = "America/New_York"


def bar_change(s: pd.Series, bar: pd.Timedelta) -> pd.Series:
    """X(t) - X(t-bar) when both exist (same session by construction)."""
    prev = s.reindex(s.index - bar)
    prev.index = s.index
    return s - prev


def lag_series(s: pd.Series, k: int, bar: pd.Timedelta, index: pd.DatetimeIndex | None = None) -> pd.Series:
    """Value of ``s`` k bars earlier (k<0: later), aligned to ``index``; NaN across gaps."""
    idx = s.index if index is None else index
    out = s.reindex(idx - k * bar)
    out.index = idx
    return out


def cross_corr(dx: pd.Series, dy: pd.Series, bar: pd.Timedelta, max_lag: int = 12) -> pd.DataFrame:
    """corr(dx[t - k bars], dy[t]); k>0 = x leads. Only within-session pairs."""
    rows = []
    for k in range(-max_lag, max_lag + 1):
        pair = pd.concat([lag_series(dx, k, bar, dy.index), dy], axis=1).dropna()
        n = len(pair)
        c = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if n > 10 else np.nan
        rows.append({"lag": k, "minutes": int(k * bar / pd.Timedelta(minutes=1)), "corr": c, "n": n,
                     "band": 1.96 / math.sqrt(n) if n else np.nan})
    return pd.DataFrame(rows)


def granger(dy: pd.Series, dx: pd.Series, bar: pd.Timedelta, max_lag: int = 6) -> pd.DataFrame:
    """Within-session Granger F test: do x's last p bar-changes help explain y's?"""
    rows = []
    for p in range(1, max_lag + 1):
        cols = {"y": dy}
        for l in range(1, p + 1):
            cols[f"y{l}"] = lag_series(dy, l, bar)
            cols[f"x{l}"] = lag_series(dx, l, bar, dy.index)
        d = pd.DataFrame(cols).dropna()
        n = len(d)
        if n < 5 * p + 20:
            rows.append({"lag": p, "minutes": int(p * bar / pd.Timedelta(minutes=1)), "F": np.nan,
                         "p_value": np.nan, "n": n})
            continue
        Y = d["y"].values
        Xr = np.column_stack([np.ones(n)] + [d[f"y{l}"].values for l in range(1, p + 1)])
        Xu = np.column_stack([Xr] + [d[f"x{l}"].values for l in range(1, p + 1)])
        rss_r = np.sum((Y - Xr @ np.linalg.lstsq(Xr, Y, rcond=None)[0]) ** 2)
        rss_u = np.sum((Y - Xu @ np.linalg.lstsq(Xu, Y, rcond=None)[0]) ** 2)
        df2 = n - Xu.shape[1]
        F = ((rss_r - rss_u) / p) / (rss_u / df2)
        rows.append({"lag": p, "minutes": int(p * bar / pd.Timedelta(minutes=1)), "F": F,
                     "p_value": float(stats.f.sf(F, p, df2)), "n": n})
    return pd.DataFrame(rows)


def ols_beta(dx: pd.Series, dy: pd.Series) -> dict:
    d = pd.concat([dx.rename("x"), dy.rename("y")], axis=1).dropna()
    if len(d) < 10:
        return {"beta": np.nan, "alpha": np.nan, "corr": np.nan, "t": np.nan, "n": len(d)}
    r = vl.ols_hac(d["y"].values, np.column_stack([np.ones(len(d)), d["x"].values]))
    return {"beta": r.coef[1], "alpha": r.coef[0], "corr": d["x"].corr(d["y"]), "t": r.t[1], "n": len(d)}


def rolling_beta_corr(dx: pd.Series, dy: pd.Series, n_obs: int) -> pd.DataFrame:
    """Rolling corr/beta over the last ``n_obs`` within-session change pairs."""
    d = pd.concat([dx.rename("x"), dy.rename("y")], axis=1).dropna()
    mp = max(10, int(n_obs * 0.7))
    cov = d["y"].rolling(n_obs, min_periods=mp).cov(d["x"])
    var = d["x"].rolling(n_obs, min_periods=mp).var()
    return pd.DataFrame({"corr": d["y"].rolling(n_obs, min_periods=mp).corr(d["x"]), "beta": cov / var})


def sessions(index: pd.DatetimeIndex) -> pd.Index:
    return index.tz_convert(NY).normalize().tz_localize(None)


def last_sessions(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    s = sessions(df.index)
    keep = sorted(set(s))[-n:]
    return df[s.isin(keep)]


def window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df[df.index >= df.index.max() - pd.Timedelta(days=days)]


def zscore_frame(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c]
        out[c] = (s - s.mean()) / s.std()
    return out


# ------------------------------------------------------------------ events
def find_moves(dx: pd.Series, z: float = 3.0, lookback: int = 780, min_gap: pd.Timedelta = pd.Timedelta(minutes=30),
               sign: int = 1) -> pd.DataFrame:
    """Large bar moves: dx / trailing sd (sd from earlier bars only) beyond ``z``
    in direction ``sign``. De-clustered by ``min_gap``."""
    s = dx.dropna()
    sd = s.shift(1).rolling(lookback, min_periods=max(60, lookback // 4)).std()
    zz = s / sd
    hit = (zz * sign) > z
    rows, last = [], None
    for t in zz.index[hit.fillna(False).values]:
        if last is not None and t - last < min_gap:
            continue
        last = t
        rows.append({"obs": t, "move": s.loc[t], "z": zz.loc[t]})
    return pd.DataFrame(rows)


def event_path(level: pd.Series, events: pd.DataFrame, bar: pd.Timedelta, pre: int = 6, post: int = 24) -> pd.DataFrame:
    """Mean change of ``level`` vs one bar before each event, within the same
    session only (points across the overnight gap are skipped)."""
    rows = []
    ev = list(events["obs"]) if events is not None and not events.empty else []
    sess = {t: s for t, s in zip(level.index, sessions(level.index))}
    for k in range(-pre, post + 1):
        xs = []
        for t in ev:
            a, b = t - bar, t + k * bar
            if a in sess and b in sess and sess[a] == sess[t] == sess[b]:
                va, vb = level.get(a), level.get(b)
                if np.isfinite(va) and np.isfinite(vb):
                    xs.append(vb - va)
        n = len(xs)
        m = float(np.mean(xs)) if n else np.nan
        half = stats.t.ppf(0.975, n - 1) * float(np.std(xs, ddof=1)) / math.sqrt(n) if n > 1 else np.nan
        rows.append({"offset": k, "minutes": int(k * bar / pd.Timedelta(minutes=1)), "mean": m,
                     "lo": m - half, "hi": m + half, "n": n})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ session test
def session_table(h1: pd.DataFrame, asset: str, cut_hour: int = 12, open_hour: int = 10,
                  close_hour: int = 16) -> pd.DataFrame:
    """One row per session with VIX/DVOL at the open (10:00 NY), cut and close."""
    col = f"DVOL_{asset}"
    ny = h1.index.tz_convert(NY)
    d = h1.assign(_s=sessions(h1.index), _h=ny.hour + ny.minute / 60.0)
    piv = {}
    for name, hr in (("open", open_hour), ("cut", cut_hour), ("close", close_hour)):
        x = d[d["_h"] == hr].set_index("_s")
        piv[f"VIX_{name}"] = x["VIX"]
        piv[f"DVOL_{name}"] = x[col]
    t = pd.DataFrame(piv).sort_index()
    t = t.dropna(subset=["VIX_open", "VIX_cut", "DVOL_open", "DVOL_cut"])
    return t


def session_predict(h1: pd.DataFrame, asset: str, cut_hour: int = 12, min_train: int | None = None) -> pd.DataFrame:
    """Does the VIX move from 10:00 to ``cut_hour`` NY predict DVOL later?

    Targets (all strictly after the cut): DVOL cut->16:00 (rest of session),
    16:00->next session 10:00 (overnight), cut->next 16:00. Regressors:
    VIX's 10:00->cut change and DVOL's own 10:00->cut change. HAC t-stats;
    OOS R^2 vs the historical mean (expanding, by session). Values are
    clipped at 5 robust sd (centre/scale from the training half only).
    """
    t = session_table(h1, asset, cut_hour)
    if t.empty:
        return pd.DataFrame()
    t["dVIX"] = t["VIX_cut"] - t["VIX_open"]
    t["dDVOL"] = t["DVOL_cut"] - t["DVOL_open"]
    nxt = t.shift(-1)
    targets = {
        "rest of session (cut → 16:00)": t["DVOL_close"] - t["DVOL_cut"],
        "overnight (16:00 → next 10:00)": nxt["DVOL_open"] - t["DVOL_close"],
        "to next close (cut → next 16:00)": nxt["DVOL_close"] - t["DVOL_cut"],
    }
    rows = []
    for name, y in targets.items():
        d = pd.concat([y.rename("y"), t["dVIX"], t["dDVOL"]], axis=1).dropna()
        n = len(d)
        if n < 30:
            continue
        # One crypto-only blow-up (e.g. 5 Feb 2026, DVOL +22 in an afternoon) can swamp a
        # few hundred sessions. Clip each column at 5 robust sd, with the centre and
        # scale taken from the first-half (training) sessions only, so no look-ahead.
        mt0 = min_train or max(20, n // 2)
        head = d.iloc[:mt0]
        med = head.median()
        mad = (head - med).abs().median() * 1.4826
        d = d.clip(lower=med - 5 * mad, upper=med + 5 * mad, axis=1)
        X = np.column_stack([np.ones(n), d["dVIX"].values, d["dDVOL"].values])
        r = vl.ols_hac(d["y"].values, X, lags=vl.nw_lags(n))
        mt = min_train or max(20, n // 2)
        f_m = vl.expanding_forecasts(d["y"], d[["dVIX", "dDVOL"]], 1, mt, refit_every=1)
        f_0 = vl.expanding_forecasts(d["y"], pd.DataFrame(index=d.index), 1, mt, refit_every=1)
        f_b = vl.expanding_forecasts(d["y"], d[["dDVOL"]], 1, mt, refit_every=1)
        r2o, n_oos = vl.oos_r2(d["y"], f_m, f_0)
        r2b, _ = vl.oos_r2(d["y"], f_m, f_b)
        hit = float(((np.sign(d["dVIX"]) == np.sign(d["y"])) & (d["dVIX"] != 0)).mean() * 100)
        rows.append({"target": name, "beta_vix": r.coef[1], "t_vix": r.t[1], "beta_own": r.coef[2], "t_own": r.t[2],
                     "r2": r.r2, "oos_r2_pct": 100 * r2o, "oos_r2_vs_own_pct": 100 * r2b,
                     "cw_p": vl.clark_west(d["y"], f_b, f_m, 1), "same_sign_pct": hit, "n": n, "n_oos": n_oos})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ intraday fair value
def intraday_fair(day: pd.DataFrame, asset: str, coef: dict, move_level: float | None) -> pd.DataFrame:
    """Apply the latest daily fair-value fit (DVOL ~ a + b*VIX [+ c*MOVE]) to
    intraday VIX (MOVE held at its latest daily close)."""
    col = f"DVOL_{asset}"
    out = pd.DataFrame(index=day.index)
    out["DVOL"] = day[col]
    fair = coef.get("const", np.nan) + coef.get("VIX", 0.0) * day["VIX"]
    if "MOVE" in coef:
        fair = fair + coef["MOVE"] * (move_level if move_level is not None else np.nan)
    out["fair"] = fair
    out["resid"] = out["DVOL"] - out["fair"]
    return out


def verdict_lines(asset: str, cc5: pd.DataFrame | None, gr5: pd.DataFrame | None, gr5_rev: pd.DataFrame | None,
                  cch: pd.DataFrame | None, sp: pd.DataFrame | None, alpha: float = 0.05) -> list[str]:
    lines = []
    if cc5 is not None and not cc5.empty and cc5["n"].max() > 0:
        c0 = cc5.loc[cc5["lag"] == 0].iloc[0]
        lead = cc5[cc5["lag"] > 0]
        best = lead.loc[lead["corr"].abs().idxmax()]
        sig = lead[lead["corr"].abs() > lead["band"]]
        txt = (f"5-min bars: same-bar correlation {c0['corr']:+.2f}; best VIX lead {best['corr']:+.2f} at "
               f"{int(best['minutes'])} min (band ±{best['band']:.2f}). ")
        if sig.empty:
            txt += "No 5-min lead beyond noise."
        else:
            txt += "Significant VIX leads at " + ", ".join(f"{int(m)}m" for m in sig["minutes"]) + "."
        lines.append(txt)
    if gr5 is not None and not gr5.empty and gr5["p_value"].notna().any():
        pf = gr5["p_value"].min()
        pr = gr5_rev["p_value"].min() if gr5_rev is not None and not gr5_rev.empty else np.nan
        lines.append(f"5-min Granger (lags 5-30 min): VIX → {asset} DVOL best p = {pf:.3f}"
                     f" ({'significant' if pf < alpha else 'not significant'}); DVOL → VIX best p = {pr:.3f}.")
    if cch is not None and not cch.empty and cch["n"].max() > 0:
        c0 = cch.loc[cch["lag"] == 0].iloc[0]
        c1 = cch.loc[cch["lag"] == 1].iloc[0]
        sig = "significant" if abs(c1["corr"]) > c1["band"] else "not significant"
        lines.append(f"Hourly bars: same-hour correlation {c0['corr']:+.2f}; last hour's VIX move vs this hour's "
                     f"DVOL move {c1['corr']:+.2f} ({sig}, band ±{c1['band']:.2f}).")
    if sp is not None and not sp.empty:
        r = sp.iloc[0]
        lines.append(f"Morning VIX move (10:00→12:00 NY) vs {asset} DVOL for the rest of the session: beta "
                     f"{r['beta_vix']:+.2f} (HAC t {r['t_vix']:+.1f}), out-of-sample R² {r['oos_r2_pct']:+.1f}%.")
    return lines
