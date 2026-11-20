"""Does TradFi vol lead crypto vol? Pure analysis functions for
pages/14_VIX_vs_DVOL.py (no Streamlit, no network; tested in
tests/test_vol_lead.py).

Calendar: every row is a TradFi trading day (a Cboe VIX close). DVOL is the
value at the 16:00 New York close (lib/vol_lead_data.snap_to_ny_close), so a
Monday change is Friday 16:00 -> Monday 16:00 and includes the weekend's
crypto move. Changes are in vol points unless noted.

Conventions: in :func:`cross_corr` a positive lag ``k`` means the predictor
leads: corr(x[t-k], y[t]). Regressions use Newey-West (Bartlett) HAC errors.
Out-of-sample R^2 is Campbell-Thompson: 1 - SSE(model) / SSE(baseline), both
from expanding-window forecasts that only use data known at the forecast date.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

ASSETS = ("BTC", "ETH")
HORIZONS = (1, 5, 10, 20)
TRADING_YEAR = 252

# predictor key -> (label, how to build its daily change and level columns)
PREDICTORS = {
    "VIX": "VIX",
    "MOVE": "MOVE (Treasury vol)",
    "VVIX": "VVIX (vol of VIX)",
    "TS_SLOPE": "VIX term slope (VIX3M - VIX)",
    "VIX9D_GAP": "VIX9D - VIX (short-end stress)",
    "DXY": "Dollar index (DXY)",
    "HY_OAS": "HY credit spread (OAS)",
    "NDX_RV": "Nasdaq-100 21d realized vol",
}


# ------------------------------------------------------------------ panel
def align_panel(raw: pd.DataFrame, ffill_limit: int = 3) -> pd.DataFrame:
    """Raw outer-joined panel -> one row per VIX trading day.

    Series that trade on a different holiday calendar (MOVE, DXY, HY OAS,
    NDX) are forward-filled for at most ``ffill_limit`` rows. DVOL is never
    filled: a missing DVOL row stays missing.
    """
    if raw is None or raw.empty or "VIX" not in raw:
        return pd.DataFrame()
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    out = pd.DataFrame(index=raw.index)
    for col in raw.columns:
        s = pd.to_numeric(raw[col], errors="coerce")
        out[col] = s if col.startswith("DVOL") or col == "VIX" else s.ffill(limit=ffill_limit)
    if "NDX" in out:
        # realized vol uses NDX's own trading days, then aligns
        ndx = pd.to_numeric(raw["NDX"], errors="coerce").dropna()
        rv = np.log(ndx).diff().rolling(21, min_periods=18).std() * math.sqrt(TRADING_YEAR) * 100
        out["NDX_RV"] = rv.reindex(out.index).ffill(limit=ffill_limit)
    out = out[out["VIX"].notna()]
    if "VIX3M" in out:
        out["TS_SLOPE"] = out["VIX3M"] - out["VIX"]
    if "VIX9D" in out:
        out["VIX9D_GAP"] = out["VIX9D"] - out["VIX"]
    return out


def asset_frame(panel: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Panel rows where this asset's DVOL exists, with spreads and ratios."""
    col = f"DVOL_{asset}"
    if panel is None or panel.empty or col not in panel:
        return pd.DataFrame()
    df = panel[panel[col].notna()].copy()
    df["DVOL"] = df[col]
    df["SPREAD_VIX"] = df["DVOL"] - df["VIX"]
    df["RATIO_VIX"] = df["DVOL"] / df["VIX"]
    if "MOVE" in df:
        # MOVE is in bp of Treasury yield vol; the ratio is relative, not a vol spread
        df["RATIO_MOVE"] = df["DVOL"] / df["MOVE"]
    return df


def change(df: pd.DataFrame, col: str) -> pd.Series:
    """Daily change on the TradFi calendar. DXY is a % log return."""
    s = df[col]
    if col == "DXY":
        return np.log(s).diff() * 100
    return s.diff()


def rolling_z(s: pd.Series, window: int = TRADING_YEAR, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or max(20, window // 2)
    m = s.rolling(window, min_periods=mp).mean()
    sd = s.rolling(window, min_periods=mp).std()
    return (s - m) / sd.replace(0, np.nan)


def rolling_pct(s: pd.Series, window: int = TRADING_YEAR, min_periods: int | None = None) -> pd.Series:
    """Percentile (0-100) of today's value within the trailing window (incl. today)."""
    mp = min_periods or max(20, window // 2)

    def _p(a):
        last = a[-1]
        return 100.0 * (np.sum(a <= last) - 0.5) / len(a)

    return s.rolling(window, min_periods=mp).apply(_p, raw=True)


# ------------------------------------------------------------------ OLS / HAC
@dataclass
class OLS:
    coef: np.ndarray
    se: np.ndarray
    t: np.ndarray
    r2: float
    n: int
    names: list


def nw_lags(n: int, h: int = 1) -> int:
    return int(max(h, math.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


def ols_hac(y, X, lags: int | None = None, names=None) -> OLS:
    """OLS with Newey-West (Bartlett kernel) standard errors."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    L = nw_lags(n) if lags is None else int(lags)
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, min(L, n - 1) + 1):
        w = 1.0 - l / (L + 1.0)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - float(e @ e) / tss if tss > 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    return OLS(beta, se, t, r2, n, list(names) if names is not None else [f"x{i}" for i in range(k)])


# ------------------------------------------------------------------ lead-lag
def cross_corr(x: pd.Series, y: pd.Series, max_lag: int = 10) -> pd.DataFrame:
    """corr(x[t-k], y[t]) for k in -max_lag..max_lag; k>0 = x leads y.

    ``band`` is the +/-1.96/sqrt(n) white-noise band.
    """
    rows = []
    for k in range(-max_lag, max_lag + 1):
        pair = pd.concat([x.shift(k), y], axis=1).dropna()
        n = len(pair)
        c = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if n > 3 else np.nan
        rows.append({"lag": k, "corr": c, "n": n, "band": 1.96 / math.sqrt(n) if n else np.nan})
    return pd.DataFrame(rows)


def granger(y: pd.Series, x: pd.Series, max_lag: int = 5) -> pd.DataFrame:
    """Does x Granger-cause y? F test of x's lags 1..p in y on own lags + x lags."""
    rows = []
    base = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    for p in range(1, max_lag + 1):
        d = base.copy()
        for l in range(1, p + 1):
            d[f"y{l}"] = d["y"].shift(l)
            d[f"x{l}"] = d["x"].shift(l)
        d = d.dropna()
        n = len(d)
        if n < 3 * p + 10:
            rows.append({"lag": p, "F": np.nan, "p_value": np.nan, "n": n})
            continue
        Y = d["y"].values
        Xr = np.column_stack([np.ones(n)] + [d[f"y{l}"].values for l in range(1, p + 1)])
        Xu = np.column_stack([Xr] + [d[f"x{l}"].values for l in range(1, p + 1)])
        rss_r = np.sum((Y - Xr @ np.linalg.lstsq(Xr, Y, rcond=None)[0]) ** 2)
        rss_u = np.sum((Y - Xu @ np.linalg.lstsq(Xu, Y, rcond=None)[0]) ** 2)
        df2 = n - Xu.shape[1]
        F = ((rss_r - rss_u) / p) / (rss_u / df2)
        rows.append({"lag": p, "F": F, "p_value": float(stats.f.sf(F, p, df2)), "n": n})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ prediction
def forward_change(s: pd.Series, h: int) -> pd.Series:
    return s.shift(-h) - s


def baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """DVOL's own information: today's change and level (mean reversion)."""
    return pd.DataFrame({"dDVOL": df["DVOL"].diff(), "DVOL": df["DVOL"]}, index=df.index)


SPECS = ("change", "full")


def predictor_features(df: pd.DataFrame, key: str, spec: str = "full") -> pd.DataFrame:
    """One predictor's information at t.

    ``change``: today's change only (the "news"). ``full``: change and level
    (plus the DVOL-VIX spread for the VIX).
    """
    if key not in df or df[key].notna().sum() < 50:
        return pd.DataFrame(index=df.index)
    d = change(df, key)
    if spec == "change":
        return pd.DataFrame({f"d{key}": d}, index=df.index)
    lvl = df[key] if key != "DXY" else rolling_z(np.log(df[key]), 63)
    out = pd.DataFrame({f"d{key}": d, f"{key}_lvl": lvl}, index=df.index)
    if key == "VIX":
        out["SPREAD_VIX"] = df["DVOL"] - df["VIX"]
    return out


def expanding_forecasts(y: pd.Series, X: pd.DataFrame, h: int, min_train: int = TRADING_YEAR,
                        refit_every: int = 5) -> pd.Series:
    """Out-of-sample forecasts of y[t] (= forward h-day change at t).

    At forecast date t the model is fit only on rows s with s + h <= t
    (targets already observed). Refit every ``refit_every`` rows.
    """
    d = pd.concat([y.rename("_y"), X], axis=1).dropna()
    idx = d.index
    Y = d["_y"].values
    Z = np.column_stack([np.ones(len(d)), d.drop(columns="_y").values])
    pos = {t: i for i, t in enumerate(idx)}
    # positions in the full calendar so "s + h <= t" counts trading rows
    full_pos = {t: i for i, t in enumerate(y.index)}
    fp = np.array([full_pos[t] for t in idx])
    pred = np.full(len(d), np.nan)
    beta = None
    last_fit = -10 ** 9
    for i in range(len(d)):
        train = fp <= fp[i] - h
        if train.sum() < min_train:
            continue
        if beta is None or i - last_fit >= refit_every:
            beta = np.linalg.lstsq(Z[train], Y[train], rcond=None)[0]
            last_fit = i
        pred[i] = Z[i] @ beta
    del pos
    return pd.Series(pred, index=idx)


def clark_west(y: pd.Series, f_base: pd.Series, f_model: pd.Series, h: int) -> float:
    """One-sided Clark-West p-value that the bigger (nested) model forecasts better."""
    d = pd.concat([y, f_base, f_model], axis=1).dropna()
    if len(d) < 30:
        return float("nan")
    yy, fb, fm = d.iloc[:, 0].values, d.iloc[:, 1].values, d.iloc[:, 2].values
    adj = (yy - fb) ** 2 - ((yy - fm) ** 2 - (fb - fm) ** 2)
    r = ols_hac(adj, np.ones((len(adj), 1)), lags=nw_lags(len(adj), h))
    return float(stats.norm.sf(r.t[0]))


def oos_r2(y: pd.Series, f_model: pd.Series, f_base: pd.Series) -> tuple[float, int]:
    d = pd.concat([y, f_model, f_base], axis=1).dropna()
    if len(d) < 30:
        return float("nan"), len(d)
    e_m = d.iloc[:, 0] - d.iloc[:, 1]
    e_b = d.iloc[:, 0] - d.iloc[:, 2]
    return 1.0 - float((e_m ** 2).sum() / (e_b ** 2).sum()), len(d)


def predictive_regression(df: pd.DataFrame, key: str, h: int, min_train: int = TRADING_YEAR,
                          spec: str = "full") -> dict:
    """DVOL(t+h) - DVOL(t) on DVOL's own change/level plus one predictor."""
    y = forward_change(df["DVOL"], h)
    base = baseline_features(df)
    feats = predictor_features(df, key, spec)
    if feats.empty:
        return {"predictor": key, "h": h, "ok": False}
    full = pd.concat([y.rename("y"), base, feats], axis=1).dropna()
    if len(full) < min_train + 60:
        return {"predictor": key, "h": h, "ok": False, "n": len(full)}
    names = ["const"] + list(base.columns) + list(feats.columns)
    Xf = np.column_stack([np.ones(len(full)), full[names[1:]].values])
    fit = ols_hac(full["y"].values, Xf, lags=nw_lags(len(full), h), names=names)
    Xb = np.column_stack([np.ones(len(full)), full[list(base.columns)].values])
    r2_base = ols_hac(full["y"].values, Xb, lags=1).r2
    yy = full["y"]
    f_base = expanding_forecasts(yy, full[list(base.columns)], h, min_train)
    f_mod = expanding_forecasts(yy, full[list(base.columns) + list(feats.columns)], h, min_train)
    r2oos, n_oos = oos_r2(yy, f_mod, f_base)
    # vs a zero-change (random-walk) forecast as well
    zero = pd.Series(0.0, index=yy.index)
    r2oos_rw, _ = oos_r2(yy, f_mod, zero)
    return {
        "predictor": key, "h": h, "ok": True, "n": fit.n, "n_oos": n_oos,
        "r2": fit.r2, "r2_base": r2_base, "r2_gain": fit.r2 - r2_base,
        "oos_r2_vs_base": r2oos, "oos_r2_vs_rw": r2oos_rw,
        "cw_p": clark_west(yy, f_base, f_mod, h),
        "coef": dict(zip(names, fit.coef)), "t": dict(zip(names, fit.t)),
    }


def ranking_table(df: pd.DataFrame, predictors=None, horizons=HORIZONS,
                  min_train: int = TRADING_YEAR, spec: str = "change") -> pd.DataFrame:
    """OOS R^2 gain of each predictor over the DVOL-only model, per horizon.

    Default ``spec="change"`` adds only the predictor's daily change (fewest
    parameters, so estimation noise does not swamp a small signal).
    """
    rows = []
    for key in (predictors or PREDICTORS):
        for h in horizons:
            r = predictive_regression(df, key, h, min_train, spec)
            if not r.get("ok"):
                continue
            # the t-stat of the predictor's own change (its "news")
            rows.append({
                "predictor": key, "label": PREDICTORS.get(key, key), "h": h, "spec": spec,
                "oos_r2_gain_pct": 100 * r["oos_r2_vs_base"],
                "cw_p": r["cw_p"], "in_sample_r2_gain_pct": 100 * r["r2_gain"],
                "t_change": r["t"].get(f"d{key}", np.nan), "coef_change": r["coef"].get(f"d{key}", np.nan),
                "t_level": r["t"].get(f"{key}_lvl", np.nan), "n": r["n"], "n_oos": r["n_oos"],
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ events
def find_events(level: pd.Series, z_thresh: float = 2.0, cross_level: float | None = 25.0,
                window: int = TRADING_YEAR, min_gap: int = 10) -> pd.DataFrame:
    """Spike days: 1-day change > z_thresh trailing sd (sd from t-1 back), or an
    upward cross of ``cross_level``. De-clustered: later events within
    ``min_gap`` rows of the previous one are dropped."""
    d = level.diff()
    sd = d.shift(1).rolling(window, min_periods=60).std()
    z = d / sd
    is_spike = z > z_thresh
    is_cross = pd.Series(False, index=level.index)
    if cross_level is not None:
        is_cross = (level > cross_level) & (level.shift(1) <= cross_level)
    flag = (is_spike | is_cross).fillna(False)
    rows, last = [], -10 ** 9
    pos = np.arange(len(level))
    for i, t in zip(pos[flag.values], level.index[flag.values]):
        if i - last < min_gap:
            continue
        last = i
        rows.append({"date": t, "pos": int(i), "level": level.iloc[i], "change": d.iloc[i],
                     "z": z.iloc[i], "kind": "spike" if bool(is_spike.iloc[i]) else "cross"})
    return pd.DataFrame(rows)


def event_path(series: pd.Series, events: pd.DataFrame, pre: int = 5, post: int = 20,
               base_offset: int = -1) -> pd.DataFrame:
    """Average path of ``series`` around events, relative to its value at
    ``base_offset`` (default the day before the event, so day 0 includes the
    same-day reaction). Returns offset, mean, lo, hi (95% t CI), n."""
    vals = series.values
    rows = []
    for k in range(-pre, post + 1):
        xs = []
        for p in (events["pos"] if not events.empty else []):
            a, b = p + base_offset, p + k
            if 0 <= a < len(vals) and 0 <= b < len(vals) and np.isfinite(vals[a]) and np.isfinite(vals[b]):
                xs.append(vals[b] - vals[a])
        n = len(xs)
        m = float(np.mean(xs)) if n else np.nan
        if n > 1:
            half = stats.t.ppf(0.975, n - 1) * float(np.std(xs, ddof=1)) / math.sqrt(n)
        else:
            half = np.nan
        rows.append({"offset": k, "mean": m, "lo": m - half, "hi": m + half, "n": n})
    return pd.DataFrame(rows)


def unconditional_path(series: pd.Series, pre: int = 5, post: int = 20, base_offset: int = -1) -> pd.DataFrame:
    """Same as event_path with every day as an event (the drift to beat)."""
    ev = pd.DataFrame({"pos": np.arange(len(series))})
    p = event_path(series, ev, pre, post, base_offset)
    return p[["offset", "mean"]]


# ------------------------------------------------------------------ rolling
def rolling_beta_corr(y: pd.Series, x: pd.Series, window: int = 63) -> pd.DataFrame:
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    cov = d["y"].rolling(window).cov(d["x"])
    var = d["x"].rolling(window).var()
    return pd.DataFrame({"corr": d["y"].rolling(window).corr(d["x"]), "beta": cov / var})


# ------------------------------------------------------------------ rich / cheap
FAIR_VALUE_INPUTS = ("VIX", "MOVE")


def fair_value(df: pd.DataFrame, inputs=FAIR_VALUE_INPUTS, window: int = TRADING_YEAR,
               min_periods: int | None = None) -> pd.DataFrame:
    """Rolling-window regression of DVOL level on TradFi vol levels.

    Each date's fit uses only the trailing ``window`` rows up to and
    including that date (all observed at t), so the residual is a real-time
    reading. Returns fair, resid (vol points), resid_z (resid / window
    residual sd) and the fit's R^2.
    """
    cols = [c for c in inputs if c in df and df[c].notna().sum() > 50]
    mp = min_periods or max(60, window // 2)
    d = df[["DVOL"] + cols].dropna()
    out = pd.DataFrame(index=d.index, columns=["fair", "resid", "resid_z", "r2"], dtype=float)
    if len(d) < mp or not cols:
        return out.reindex(df.index)
    Y = d["DVOL"].values
    X = np.column_stack([np.ones(len(d)), d[cols].values])
    for i in range(mp - 1, len(d)):
        lo = max(0, i - window + 1)
        Xi, Yi = X[lo:i + 1], Y[lo:i + 1]
        b = np.linalg.lstsq(Xi, Yi, rcond=None)[0]
        e = Yi - Xi @ b
        sd = float(np.std(e, ddof=X.shape[1]))
        fair = float(X[i] @ b)
        tss = float(np.sum((Yi - Yi.mean()) ** 2))
        out.iloc[i] = [fair, Y[i] - fair, (Y[i] - fair) / sd if sd > 0 else np.nan,
                       1 - float(e @ e) / tss if tss > 0 else np.nan]
    out.attrs["inputs"] = cols
    return out.reindex(df.index)


def rich_cheap(df: pd.DataFrame, window: int = TRADING_YEAR, long_window: int = 3 * TRADING_YEAR,
               inputs=FAIR_VALUE_INPUTS) -> pd.DataFrame:
    """Composite rich/cheap score for crypto vol vs TradFi vol.

    Components (each a z-score, + = crypto vol rich):
      * fair-value residual z (DVOL vs rolling regression on VIX, MOVE)
      * DVOL/VIX ratio z vs trailing ``window``
      * DVOL/MOVE ratio z vs trailing ``window`` (if MOVE exists)
    ``score`` is their mean; ``score_pct`` its percentile vs its own trailing
    ``long_window`` history. Ratio percentiles vs 1y and 3y are included.
    """
    fv = fair_value(df, inputs, window)
    out = pd.DataFrame(index=df.index)
    out["fair"] = fv["fair"]
    out["resid"] = fv["resid"]
    out["resid_z"] = fv["resid_z"]
    out["fv_r2"] = fv["r2"]
    out["ratio_vix"] = df["RATIO_VIX"]
    out["ratio_vix_z"] = rolling_z(df["RATIO_VIX"], window)
    out["ratio_vix_pct_1y"] = rolling_pct(df["RATIO_VIX"], window)
    out["ratio_vix_pct_3y"] = rolling_pct(df["RATIO_VIX"], long_window, min_periods=window)
    comps = ["resid_z", "ratio_vix_z"]
    if "RATIO_MOVE" in df and df["RATIO_MOVE"].notna().sum() > 50:
        out["ratio_move"] = df["RATIO_MOVE"]
        out["ratio_move_z"] = rolling_z(df["RATIO_MOVE"], window)
        out["ratio_move_pct_1y"] = rolling_pct(df["RATIO_MOVE"], window)
        out["ratio_move_pct_3y"] = rolling_pct(df["RATIO_MOVE"], long_window, min_periods=window)
        comps.append("ratio_move_z")
    out["score"] = out[comps].mean(axis=1, skipna=False)
    out["score_pct"] = rolling_pct(out["score"], long_window, min_periods=window // 2)
    out.attrs["components"] = comps
    return out


def score_quintiles(score: pd.Series, dvol: pd.Series, horizons=(5, 10, 20), spread: pd.Series | None = None,
                    n_bins: int = 5) -> pd.DataFrame:
    """Average forward DVOL change (and spread change) by score quintile.

    Quintile edges are full-sample (descriptive); see :func:`score_signal` for
    the honest real-time test.
    """
    s = score.dropna()
    if len(s) < n_bins * 10:
        return pd.DataFrame()
    q = pd.qcut(s, n_bins, labels=[f"Q{i}" for i in range(1, n_bins + 1)])
    rows = []
    for h in horizons:
        fwd = forward_change(dvol, h).reindex(s.index)
        fsp = forward_change(spread, h).reindex(s.index) if spread is not None else None
        for lab in q.cat.categories:
            m = q == lab
            f = fwd[m].dropna()
            row = {"h": h, "quintile": lab, "score_mean": float(s[m].mean()), "n": int(len(f)),
                   "fwd_dvol_mean": float(f.mean()) if len(f) else np.nan,
                   "hit_down_pct": 100.0 * float((f < 0).mean()) if len(f) else np.nan}
            if fsp is not None:
                row["fwd_spread_mean"] = float(fsp[m].dropna().mean())
            rows.append(row)
    return pd.DataFrame(rows)


def score_signal(score: pd.Series, dvol: pd.Series, horizons=(5, 10, 20), spread: pd.Series | None = None,
                 min_train: int = TRADING_YEAR) -> pd.DataFrame:
    """Does the rich/cheap score predict DVOL (or the DVOL-VIX spread)?

    In-sample: slope of the forward h-day change on the score, HAC t.
    Real time: out-of-sample R^2 of an expanding regression on the score vs
    the historical-mean forecast, and Clark-West p. Negative slope = rich vol
    tends to cheapen (mean reversion).
    """
    rows = []
    targets = {"DVOL": dvol}
    if spread is not None:
        targets["DVOL-VIX spread"] = spread
    for name, ser in targets.items():
        for h in horizons:
            y = forward_change(ser, h)
            d = pd.concat([y.rename("y"), score.rename("s")], axis=1).dropna()
            if len(d) < min_train + 60:
                continue
            fit = ols_hac(d["y"].values, np.column_stack([np.ones(len(d)), d["s"].values]),
                          lags=nw_lags(len(d), h))
            f_mean = expanding_forecasts(d["y"], pd.DataFrame(index=d.index), h, min_train)
            f_mod = expanding_forecasts(d["y"], d[["s"]], h, min_train)
            r2o, n_oos = oos_r2(d["y"], f_mod, f_mean)
            rows.append({"target": name, "h": h, "slope": fit.coef[1], "t": fit.t[1], "r2": fit.r2,
                         "oos_r2_pct": 100 * r2o, "cw_p": clark_west(d["y"], f_mean, f_mod, h),
                         "n": fit.n, "n_oos": n_oos})
    return pd.DataFrame(rows)


def rich_cheap_headline(asset: str, rc: pd.DataFrame) -> str:
    last = rc.dropna(subset=["score"]).tail(1)
    if last.empty:
        return f"{asset}: not enough history for a rich/cheap reading."
    r = last.iloc[0]
    word = "rich" if r["resid"] > 0 else "cheap"
    pct = r.get("score_pct", np.nan)
    pct_txt = f", {ordinal(pct)} percentile of the last 3y" if np.isfinite(pct) else ""
    return (f"{asset} vol is {abs(r['resid']):.1f} vols {word} vs TradFi fair value "
            f"({r['fair']:.1f}); composite score {r['score']:+.2f} sd{pct_txt}.")


def ordinal(x: float) -> str:
    n = int(round(float(x)))
    n = min(max(n, 0), 100)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


# ------------------------------------------------------------------ verdict
def verdict(asset: str, cc: pd.DataFrame, gr_fwd: pd.DataFrame, gr_rev: pd.DataFrame,
            rank: pd.DataFrame, alpha: float = 0.05) -> list[str]:
    """Plain-English lines computed from the stats (nothing hardcoded)."""
    lines = []
    if cc is not None and not cc.empty:
        c0 = cc.loc[cc["lag"] == 0].iloc[0]
        lead = cc[cc["lag"] > 0]
        sig = lead[lead["corr"].abs() > lead["band"]]
        best = lead.loc[lead["corr"].abs().idxmax()]
        lines.append(
            f"Same-day link: a VIX move and the {asset} DVOL move on the same day correlate "
            f"{c0['corr']:+.2f}. That is co-movement, not prediction.")
        if sig.empty:
            lines.append(
                f"Lead: no day-ahead VIX change (lags 1-10) correlates with {asset} DVOL changes "
                f"beyond the noise band (largest {best['corr']:+.2f} at {int(best['lag'])}d, band ±{best['band']:.2f}).")
        else:
            lines.append(
                f"Lead: VIX changes lead {asset} DVOL at {', '.join(f'{int(l)}d' for l in sig['lag'])} "
                f"(peak {best['corr']:+.2f} at {int(best['lag'])}d, band ±{best['band']:.2f}).")
    if gr_fwd is not None and not gr_fwd.empty:
        pf = gr_fwd["p_value"].min()
        pr = gr_rev["p_value"].min() if gr_rev is not None and not gr_rev.empty else np.nan
        fwd_txt = "significant" if pf < alpha else "not significant"
        lines.append(
            f"Granger: VIX → {asset} DVOL is {fwd_txt} (best p = {pf:.3f} over lags 1-5); "
            f"DVOL → VIX best p = {pr:.3f}.")
    if rank is not None and not rank.empty:
        good = rank[(rank["oos_r2_gain_pct"] > 0) & (rank["cw_p"] < alpha)]
        for h in sorted(rank["h"].unique()):
            rh = rank[rank["h"] == h].sort_values("oos_r2_gain_pct", ascending=False)
            top = rh.iloc[0]
            vix = rh[rh["predictor"] == "VIX"]
            vtxt = (f"VIX {vix.iloc[0]['oos_r2_gain_pct']:+.1f}%" if not vix.empty else "VIX n/a")
            lines.append(
                f"{h}d ahead: best add-on is {top['label']} ({top['oos_r2_gain_pct']:+.1f}% out-of-sample R², "
                f"CW p={top['cw_p']:.2f}); {vtxt}.")
        if good.empty:
            lines.append(
                f"Bottom line: no TradFi gauge adds significant out-of-sample forecasting power for "
                f"{asset} DVOL beyond DVOL's own history at 1-20 days.")
        else:
            g = good.sort_values("oos_r2_gain_pct", ascending=False).iloc[0]
            lines.append(
                f"Bottom line: {g['label']} adds the most real-time forecasting power for {asset} DVOL "
                f"({g['oos_r2_gain_pct']:+.1f}% out-of-sample R² at {int(g['h'])}d, CW p={g['cw_p']:.3f}); "
                f"{len(good)} predictor/horizon pairs are significant at {int(alpha*100)}%.")
    return lines
