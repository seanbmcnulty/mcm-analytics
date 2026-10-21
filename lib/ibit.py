"""
IBIT (iShares Bitcoin Trust ETF) options vs Deribit BTC options.

DELIBERATE EXCEPTION to "Deribit-only", like lib/fng.py: IBIT options trade on
US exchanges, so the only free source is Cboe's public *delayed* option-chain
JSON (https://cdn.cboe.com/api/global/delayed_quotes/options/IBIT.json).  It is
a ~15-minute-delayed SNAPSHOT of every listed contract (bid/ask/last, IV,
Greeks, volume, open interest) -- there is no per-trade tape (OPRA is paid), so
this page is chain analytics, not flow-tape analytics.

Both venues are normalised into one schema (``COLS``) so every analytic below
runs identically on IBIT and on Deribit BTC options:

    venue, expiry (UTC Timestamp), dte, T, type C/P, strike, fwd, spot, iv
    (decimal), bid, ask, mid, volume, oi, mult, oi_usd, vol_usd, delta, gamma,
    k = ln(K/F), m = K/spot, ok (IV usable)

Delta / gamma are recomputed from each venue's own IV with Black-Scholes at
r = 0 so the two sides are strictly comparable (Cboe's own Greeks are ignored).
IBIT uses spot as the forward (carry is small vs the smile); Deribit uses each
expiry's own underlying/forward price.

All functions are pure (no Streamlit) so they can be unit-tested offline.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

CBOE_URLS = (
    "https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json",
    "https://cdn-api.cboe.com/api/global/delayed_quotes/options/{sym}.json",
)
NY = ZoneInfo("America/New_York")
UTC = timezone.utc
IBIT_MULT = 100                 # shares per option contract
_OCC_RE = re.compile(r"^(?P<root>[A-Z]+\d?)(?P<ymd>\d{6})(?P<cp>[CP])(?P<k>\d{8})$")

COLS = ["venue", "expiry", "dte", "T", "type", "strike", "fwd", "spot", "iv",
        "bid", "ask", "mid", "volume", "oi", "mult", "oi_usd", "vol_usd",
        "delta", "gamma", "k", "m", "ok"]

TENORS = [7, 14, 30, 60, 90, 180]


# --------------------------------------------------------------------------
# Fetch + parse
# --------------------------------------------------------------------------

def fetch_raw(symbol: str = "IBIT", timeout: float = 25.0) -> dict:
    """Download the Cboe delayed chain (follows the cdn -> cdn-api redirect)."""
    last: Exception | None = None
    for url in CBOE_URLS:
        try:
            r = requests.get(url.format(sym=symbol), timeout=timeout,
                             headers={"User-Agent": "Mozilla/5.0 (MCM Analytics)"})
            r.raise_for_status()
            payload = r.json()
            if payload.get("data", {}).get("options"):
                return payload
        except Exception as e:  # noqa: BLE001 - try the next host
            last = e
    raise RuntimeError(f"Cboe delayed chain unavailable for {symbol}: {last}")


def _bs_greeks(F, K, T, iv, is_call):
    """Vectorised BS delta/gamma (r = 0).  NaN where inputs are unusable."""
    F, K, T, iv = (np.asarray(x, dtype=float) for x in (F, K, T, iv))
    with np.errstate(all="ignore"):
        sd = iv * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd
        delta = np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)
        gamma = norm.pdf(d1) / (F * sd)
    bad = ~np.isfinite(sd) | (sd <= 0)
    delta[bad] = np.nan
    gamma[bad] = np.nan
    return delta, gamma


def _finish(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    df = df.copy()
    df["dte"] = (df["expiry"] - pd.Timestamp(now)).dt.total_seconds() / 86400.0
    df["T"] = df["dte"].clip(lower=1e-6) / 365.25
    df["mid"] = np.where((df["bid"] > 0) & (df["ask"] > 0),
                         (df["bid"] + df["ask"]) / 2.0, np.nan)
    df["oi_usd"] = df["oi"] * df["mult"] * df["spot"]
    df["vol_usd"] = df["volume"] * df["mult"] * df["spot"]
    df["k"] = np.log(df["strike"] / df["fwd"])
    df["m"] = df["strike"] / df["spot"]
    is_call = (df["type"] == "C").values
    df["delta"], df["gamma"] = _bs_greeks(df["fwd"], df["strike"], df["T"],
                                           df["iv"], is_call)
    return df[df["dte"] > 0].reset_index(drop=True)


def parse_cboe(payload: dict, now: datetime | None = None
               ) -> tuple[pd.DataFrame, dict]:
    """Cboe payload -> (normalised chain, meta)."""
    now = now or datetime.now(UTC)
    data = payload["data"]
    spot = float(data.get("current_price") or data.get("close") or np.nan)
    rows = []
    for o in data["options"]:
        m = _OCC_RE.match(o.get("option", ""))
        if not m:
            continue
        y, mo, d = int(m["ymd"][:2]) + 2000, int(m["ymd"][2:4]), int(m["ymd"][4:])
        exp = datetime(y, mo, d, 16, 0, tzinfo=NY).astimezone(UTC)
        iv = float(o.get("iv") or 0.0)
        bid, ask = float(o.get("bid") or 0), float(o.get("ask") or 0)
        rows.append(dict(
            venue="IBIT", expiry=pd.Timestamp(exp), type=m["cp"],
            strike=int(m["k"]) / 1000.0, fwd=spot, spot=spot,
            iv=iv if iv > 0 else np.nan, bid=bid, ask=ask,
            volume=float(o.get("volume") or 0), oi=float(o.get("open_interest") or 0),
            mult=IBIT_MULT))
    df = pd.DataFrame(rows)
    df = _finish(df, now)
    # IV is only trusted when there is a live two-sided quote.
    df["ok"] = df["iv"].between(0.02, 5.0) & (df["bid"] > 0) & (df["ask"] > 0)
    df = df[COLS]
    ts = None
    try:
        ts = pd.Timestamp(payload["timestamp"]).tz_localize(NY)
    except Exception:  # noqa: BLE001
        pass
    meta = dict(
        spot=spot, iv30=float(data.get("iv30") or np.nan),
        prev_close=float(data.get("prev_day_close") or np.nan),
        change_pct=float(data.get("price_change_percent") or np.nan),
        stock_volume=float(data.get("volume") or 0),
        snapshot_ny=ts, snapshot_utc=ts.tz_convert(UTC) if ts is not None else None,
        n_contracts=len(df), n_expiries=int(df["expiry"].nunique()))
    return df, meta


def deribit_chain(now: datetime | None = None) -> tuple[pd.DataFrame, dict]:
    """Deribit BTC option book summary -> same schema as parse_cboe."""
    from lib import deribit, surface
    from lib.constants import ASSET_CONFIG

    now = now or datetime.now(UTC)
    spot = deribit.get_index_price(ASSET_CONFIG["BTC"]["index"])
    chain = deribit.get_option_chain("BTC", "option") or []
    rows = []
    for r in chain:
        parts = (r.get("instrument_name") or "").split("-")
        if len(parts) != 4 or parts[0] != "BTC":
            continue
        exp_d = surface.parse_expiry_date(parts[1])
        strike = surface.parse_strike(parts[2])
        if exp_d is None or strike is None:
            continue
        exp = pd.Timestamp(datetime(exp_d.year, exp_d.month, exp_d.day, 8, tzinfo=UTC))
        miv = r.get("mark_iv")
        fwd = r.get("underlying_price") or spot
        rows.append(dict(
            venue="Deribit", expiry=exp, type="C" if parts[3].upper().startswith("C") else "P",
            strike=float(strike), fwd=float(fwd), spot=float(spot or fwd),
            iv=(float(miv) / 100.0) if miv else np.nan,
            bid=float(r.get("bid_price") or 0), ask=float(r.get("ask_price") or 0),
            volume=float(r.get("volume") or 0), oi=float(r.get("open_interest") or 0),
            mult=1))
    if not rows:
        raise RuntimeError("Deribit option chain unavailable")
    df = _finish(pd.DataFrame(rows), now)
    # Deribit marks always exist; drop absurd wings only.
    df["ok"] = df["iv"].between(0.05, 5.0)
    df = df[COLS]
    return df, dict(spot=float(spot), n_contracts=len(df),
                    n_expiries=int(df["expiry"].nunique()))


# --------------------------------------------------------------------------
# Smile helpers
# --------------------------------------------------------------------------

def otm_smile(g: pd.DataFrame) -> pd.DataFrame:
    """One expiry -> OTM-only (puts left of forward, calls right), sorted by k."""
    s = g[g["ok"]]
    s = s[((s["type"] == "P") & (s["k"] < 0)) | ((s["type"] == "C") & (s["k"] >= 0))]
    return s.sort_values("k")[["k", "strike", "iv", "type", "delta", "volume", "oi"]]


def _at(x: np.ndarray, y: np.ndarray, t: float) -> float:
    if len(x) < 2 or t < x.min() or t > x.max():
        return np.nan
    return float(np.interp(t, x, y))


def _delta_iv(g: pd.DataFrame, typ: str, d: float) -> float:
    s = g[(g["type"] == typ) & g["ok"]].copy()
    s["ad"] = s["delta"].abs()
    s = s[s["ad"].between(0.02, 0.65)].sort_values("ad")
    return _at(s["ad"].values, s["iv"].values, d)


def expiry_metrics(df: pd.DataFrame, min_dte: float = 1.0) -> pd.DataFrame:
    """Per-expiry ATM / 25d & 10d wings / risk reversals / butterflies."""
    out = []
    for exp, g in df.groupby("expiry"):
        dte = float(g["dte"].iloc[0])
        if dte < min_dte:
            continue
        sm = otm_smile(g)
        atm = _at(sm["k"].values, sm["iv"].values, 0.0) if len(sm) >= 3 else np.nan
        c25, p25 = _delta_iv(g, "C", 0.25), _delta_iv(g, "P", 0.25)
        c10, p10 = _delta_iv(g, "C", 0.10), _delta_iv(g, "P", 0.10)
        calls, puts = g[g["type"] == "C"], g[g["type"] == "P"]
        out.append(dict(
            expiry=exp, dte=dte, atm=atm, c25=c25, p25=p25, c10=c10, p10=p10,
            rr25=c25 - p25, rr10=c10 - p10,
            fly25=(c25 + p25) / 2 - atm, fly10=(c10 + p10) / 2 - atm,
            oi_usd=g["oi_usd"].sum(), vol_usd=g["vol_usd"].sum(),
            pc_oi=puts["oi"].sum() / calls["oi"].sum() if calls["oi"].sum() else np.nan,
            pc_vol=puts["volume"].sum() / calls["volume"].sum()
            if calls["volume"].sum() else np.nan,
            n=len(sm)))
    m = pd.DataFrame(out)
    if len(m):
        m["rr25_n"] = m["rr25"] / m["atm"]       # skew as a fraction of ATM vol
    return m.sort_values("dte").reset_index(drop=True)


def at_tenors(metrics: pd.DataFrame, cols: list[str],
              tenors: list[float] = TENORS) -> pd.DataFrame:
    """Linear interpolation of per-expiry metrics onto fixed tenors (no extrapolation)."""
    res = {"tenor": tenors}
    for c in cols:
        s = metrics[["dte", c]].dropna()
        res[c] = [_at(s["dte"].values, s[c].values, t) for t in tenors]
    return pd.DataFrame(res)


def surface_grid(df: pd.DataFrame, kgrid: np.ndarray, min_dte: float = 2.0
                 ) -> pd.DataFrame:
    """Rows = expiry DTE, cols = log-moneyness; NaN outside the quoted range."""
    rows, idx = [], []
    for exp, g in df.groupby("expiry"):
        dte = float(g["dte"].iloc[0])
        sm = otm_smile(g)
        if dte < min_dte or len(sm) < 5:
            continue
        x, y = sm["k"].values, sm["iv"].values
        rows.append([_at(x, y, k) for k in kgrid])
        idx.append(dte)
    z = pd.DataFrame(rows, index=idx, columns=kgrid).sort_index()
    return z


def on_tenor_grid(z: pd.DataFrame, tenors: list[float]) -> pd.DataFrame:
    """Interpolate a surface_grid frame onto fixed tenors (column-wise)."""
    out = {}
    for c in z.columns:
        s = z[c].dropna()
        out[c] = [_at(s.index.values.astype(float), s.values, t) for t in tenors]
    return pd.DataFrame(out, index=tenors)


# --------------------------------------------------------------------------
# Positioning
# --------------------------------------------------------------------------

M_BINS = np.arange(-0.30, 0.3001, 0.05)       # K/spot - 1 buckets
TENOR_BUCKETS = [(0, 7, "<1w"), (7, 30, "1w-1m"), (30, 90, "1-3m"),
                 (90, 180, "3-6m"), (180, 1e9, ">6m")]


def oi_by_moneyness(df: pd.DataFrame) -> pd.DataFrame:
    """USD OI per moneyness bucket (K/spot-1), split by call/put, as % of venue total."""
    x = df.copy()
    x["mb"] = pd.cut(x["m"] - 1.0, bins=np.r_[-9, M_BINS, 9], right=False)
    t = x.pivot_table(index="mb", columns="type", values="oi_usd", aggfunc="sum",
                      observed=False).fillna(0.0)
    for c in ("C", "P"):
        if c not in t:
            t[c] = 0.0
    tot = float(t.values.sum()) or np.nan
    t["C_pct"], t["P_pct"] = t["C"] / tot * 100, t["P"] / tot * 100
    t["mid"] = [(iv.left + iv.right) / 2 if iv.left > -9 and iv.right < 9
                else (iv.right - 0.025 if iv.left <= -9 else iv.left + 0.025)
                for iv in t.index]
    return t.reset_index(drop=True)


def oi_by_tenor(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tot = float(df["oi_usd"].sum()) or np.nan
    for lo, hi, lab in TENOR_BUCKETS:
        g = df[(df["dte"] >= lo) & (df["dte"] < hi)]
        c, p = g[g["type"] == "C"], g[g["type"] == "P"]
        rows.append(dict(bucket=lab, call_usd=c["oi_usd"].sum(), put_usd=p["oi_usd"].sum(),
                         call_pct=c["oi_usd"].sum() / tot * 100,
                         put_pct=p["oi_usd"].sum() / tot * 100,
                         pc_oi=p["oi"].sum() / c["oi"].sum() if c["oi"].sum() else np.nan,
                         pc_vol=p["volume"].sum() / c["volume"].sum()
                         if c["volume"].sum() else np.nan,
                         vol_usd=g["vol_usd"].sum()))
    return pd.DataFrame(rows)


def max_pain(df: pd.DataFrame) -> pd.DataFrame:
    """Strike minimising total option-holder payout, per expiry (OI-based)."""
    out = []
    for exp, g in df.groupby("expiry"):
        c = g[g["type"] == "C"].groupby("strike")["oi"].sum()
        p = g[g["type"] == "P"].groupby("strike")["oi"].sum()
        ks = np.array(sorted(set(c.index) | set(p.index)))
        if len(ks) < 3 or (c.sum() + p.sum()) <= 0:
            continue
        cc, pp = c.reindex(ks).fillna(0).values, p.reindex(ks).fillna(0).values
        pay = np.array([(cc * np.maximum(s - ks, 0)).sum() + (pp * np.maximum(ks - s, 0)).sum()
                        for s in ks])
        i = int(np.argmin(pay))
        out.append(dict(expiry=exp, dte=float(g["dte"].iloc[0]), max_pain=float(ks[i]),
                        spot=float(g["spot"].iloc[0]),
                        oi_usd=float(g["oi_usd"].sum()),
                        pain_vs_spot=float(ks[i] / g["spot"].iloc[0] - 1)))
    return pd.DataFrame(out).sort_values("dte").reset_index(drop=True)


def gex_by_strike(df: pd.DataFrame, max_dte: float | None = None) -> pd.DataFrame:
    """
    Dealer-gamma proxy: $ gamma per 1% spot move = gamma * OI * mult * spot^2 * 1%.
    Convention (assumption, flagged on the page): customers are long calls and
    long puts, dealers short them -> call GEX positive, put GEX negative.
    """
    x = df[df["gamma"].notna()].copy()
    if max_dte is not None:
        x = x[x["dte"] <= max_dte]
    x["gex"] = x["gamma"] * x["oi"] * x["mult"] * x["spot"] ** 2 * 0.01
    x.loc[x["type"] == "P", "gex"] *= -1
    g = x.groupby("strike")["gex"].sum().reset_index()
    return g


def gamma_flip(g: pd.DataFrame) -> float:
    """Strike where cumulative net GEX crosses zero (nearest crossing), else NaN."""
    if len(g) < 3:
        return np.nan
    g = g.sort_values("strike")
    cum = g["gex"].cumsum().values
    s = np.sign(cum)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    return float(g["strike"].values[idx[0] + 1]) if len(idx) else np.nan


def activity(df: pd.DataFrame, min_vol: float = 200) -> pd.DataFrame:
    """Most active contracts with vol/OI turnover (unusual-activity screen)."""
    x = df[df["volume"] >= min_vol].copy()
    x["vol_oi"] = x["volume"] / x["oi"].replace(0, np.nan)
    x["spread_pct"] = (x["ask"] - x["bid"]) / x["mid"] * 100
    return x.sort_values("vol_usd", ascending=False)


# --------------------------------------------------------------------------
# Market-implied distribution (Breeden-Litzenberger on the fitted smile)
# --------------------------------------------------------------------------

def implied_density(g: pd.DataFrame, n: int = 400):
    """Return (x = ln(K/F), density, covered_range) or None if the smile is too thin."""
    sm = otm_smile(g)
    if len(sm) < 8:
        return None
    k, iv = sm["k"].values, sm["iv"].values
    T, F = float(g["T"].iloc[0]), float(g["fwd"].iloc[0])
    coef = np.polyfit(k, iv, 4)
    kk = np.linspace(k.min(), k.max(), n)
    sig = np.clip(np.polyval(coef, kk), 0.03, 4.0)
    K = F * np.exp(kk)
    sd = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd
    C = F * norm.cdf(d1) - K * norm.cdf(d1 - sd)
    d2C = np.gradient(np.gradient(C, K), K)
    f = np.clip(d2C, 0, None) * K                   # density in x = ln(K/F)
    area = np.trapezoid(f, kk) if hasattr(np, "trapezoid") else np.trapz(f, kk)
    if not np.isfinite(area) or area <= 0:
        return None
    return kk, f / area, (float(k.min()), float(k.max()))


def tail_probs(x: np.ndarray, f: np.ndarray, thresh: float = 0.10) -> dict:
    """P(return < -thresh) and P(return > +thresh) inside the covered range."""
    r = np.exp(x) - 1.0
    cdf = np.cumsum(f) * (x[1] - x[0])
    cdf = cdf / cdf[-1]
    lo = float(np.interp(-thresh, r, cdf)) if r.min() < -thresh else np.nan
    hi = 1.0 - float(np.interp(thresh, r, cdf)) if r.max() > thresh else np.nan
    return dict(down=lo, up=hi)
