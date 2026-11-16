"""
IBIT (iShares Bitcoin Trust ETF) options vs Deribit BTC options.

DELIBERATE EXCEPTION to "Deribit-only", like lib/fng.py: IBIT options trade on
US exchanges, so the only free source is Cboe's public *delayed* option-chain
JSON (https://cdn.cboe.com/api/global/delayed_quotes/options/IBIT.json).  It is
a ~15-minute-delayed SNAPSHOT of every listed contract (bid/ask/last, IV,
Greeks, volume, open interest) -- there is no per-trade tape (OPRA is paid), so
individual block prints are not available.  Day-over-day open-interest change
uses only closes this app has recorded itself (``IBIT_chain.parquet`` on the
``snapshots`` branch).  Nothing here is backfilled.

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

import csv
import io
import re
import time as _time
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
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
# Present on the Cboe chain only.  Deribit rows do not have them.
FLOW_EXTRA = ["option", "last", "bid_size", "ask_size"]

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
            mult=IBIT_MULT,
            option=m.group(0),
            last=float(o.get("last_trade_price") or 0),
            bid_size=float(o.get("bid_size") or 0),
            ask_size=float(o.get("ask_size") or 0)))
    df = pd.DataFrame(rows)
    df = _finish(df, now)
    # IV is only trusted when there is a live two-sided quote.
    df["ok"] = df["iv"].between(0.02, 5.0) & (df["bid"] > 0) & (df["ask"] > 0)
    df = df[COLS + [c for c in FLOW_EXTRA if c in df.columns]]
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


def deribit_chain(now: datetime | None = None, asset: str = "BTC") -> tuple[pd.DataFrame, dict]:
    """Deribit (inverse BTC / ETH) option book summary -> same schema as parse_cboe."""
    from lib import deribit, surface
    from lib.constants import ASSET_CONFIG

    now = now or datetime.now(UTC)
    spot = deribit.get_index_price(ASSET_CONFIG[asset]["index"])
    chain = deribit.get_option_chain(asset, "option") or []
    rows = []
    for r in chain:
        parts = (r.get("instrument_name") or "").split("-")
        if len(parts) != 4 or parts[0] != asset:
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


EXPIRY_FILTERS = ("All expiries", "Fridays (weekly+)", "Monthlies & quarterlies")
TS_COLS = ["atm", "rr25", "fly25", "rr10", "fly10", "rr25_n"]


def _last_friday(ts: pd.Timestamp) -> bool:
    return ts.weekday() == 4 and (ts + pd.Timedelta(days=7)).month != ts.month


def deribit_expiry_grid(mD: pd.DataFrame, mI: pd.DataFrame, cols: list[str] = TS_COLS,
                        kind: str = "All expiries", min_dte: float = 2.0,
                        max_dte: float = 1e9) -> pd.DataFrame:
    """Term structure on Deribit's *listed* expiry dates.

    One row per Deribit expiry; ``D_<c>`` is Deribit's own value, ``I_<c>`` is IBIT
    linearly interpolated (in DTE, no extrapolation) at that expiry's DTE, and
    ``X_<c>`` = IBIT - Deribit.  ``kind`` filters the Deribit expiries (Fridays =
    weeklies and longer; monthlies & quarterlies = last Friday of the month).
    """
    d = mD[(mD["dte"] >= min_dte) & (mD["dte"] <= max_dte)].copy()
    if kind != EXPIRY_FILTERS[0]:
        ex = pd.to_datetime(d["expiry"])
        d = d[ex.map(lambda t: t.weekday() == 4)] if kind == EXPIRY_FILTERS[1] \
            else d[ex.map(_last_friday)]
    if d.empty:
        return pd.DataFrame()
    out = d[["expiry", "dte"]].reset_index(drop=True)
    out["type"] = ["Quarterly" if (_last_friday(pd.Timestamp(e)) and pd.Timestamp(e).month % 3 == 0)
                   else "Monthly" if _last_friday(pd.Timestamp(e))
                   else "Weekly" if pd.Timestamp(e).weekday() == 4 else "Daily"
                   for e in out["expiry"]]
    ti = at_tenors(mI, cols, out["dte"].tolist()) if len(mI) else None
    for c in cols:
        out[f"D_{c}"] = d[c].values
        out[f"I_{c}"] = ti[c].values if ti is not None else np.nan
        out[f"X_{c}"] = out[f"I_{c}"] - out[f"D_{c}"]
    return out


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


# --------------------------------------------------------------------------
# Large tickets — snapshot OI, unusual volume, last-print lean, flow
# --------------------------------------------------------------------------
# Cboe still has no prints.  Everything below is either the live snapshot or a
# diff against a close this process recorded.  There is no synthetic history.

# Equity options close 16:15 ET; the free feed is ~15 min delayed.  16:45 ET is
# the first time a snapshot can see the close in both EST and EDT.
OPTIONS_CLOSE_GATE = time(16, 45)
CASH_OPEN = time(9, 30)
OPTIONS_CLOSE = time(16, 15)
# Float slack so 1.50 vs 1.5000001 stays "at the ask" without pulling a
# one-cent-inside print out of the ambiguous bucket.
_AGG_EPS = 1e-4

# Full-day NYSE/Nasdaq closures, from the NYSE holiday calendar published for
# 2026–2028 (early closes are trading days and are not listed).  1 Jan 2028
# falls on Saturday and is not observed.  Years outside this set are not
# guessed: weekends are still skipped, holidays are not.
NYSE_HOLIDAYS = frozenset({
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
    date(2028, 1, 17), date(2028, 2, 21), date(2028, 4, 14), date(2028, 5, 29),
    date(2028, 6, 19), date(2028, 7, 4), date(2028, 9, 4), date(2028, 11, 23),
    date(2028, 12, 25),
})
HOLIDAY_YEARS = range(2026, 2029)

CHAIN_FILENAME = "IBIT_chain.parquet"
CHAIN_COLUMNS = ["session_date", "recorded_at", "option", "expiry", "type",
                 "strike", "spot", "bid", "ask", "last", "iv", "volume", "oi",
                 "delta"]
_CHAIN_REMOTE_TTL = 300.0   # same cadence as the vol-snapshot remote cache
_CHAIN_REMOTE_CACHE: dict[str, tuple[float, "pd.DataFrame | None"]] = {}


def is_us_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def holiday_list_covers(d: date) -> bool:
    return d.year in HOLIDAY_YEARS


def _as_ny(ts) -> datetime:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(NY)
    else:
        t = t.tz_convert(NY)
    return t.to_pydatetime()


def _previous_trading_day(d: date) -> date:
    d = d - timedelta(days=1)
    while not is_us_trading_day(d):
        d = d - timedelta(days=1)
    return d


def session_date_of(ts) -> date:
    """US session the chain describes.

    Before 09:30 ET the snapshot is still the previous session (volume has
    not reset).  Weekends and full holidays roll back to the last session.
    """
    ny = _as_ny(ts)
    d = ny.date()
    if (ny.hour, ny.minute, ny.second) < (CASH_OPEN.hour, CASH_OPEN.minute, CASH_OPEN.second):
        d = d - timedelta(days=1)
    while not is_us_trading_day(d):
        d = d - timedelta(days=1)
    return d


def recording_decision(now: datetime) -> tuple[str, date | None]:
    """``(status, session)`` for a recorder run at ``now``.

    ``ok`` after 16:45 America/New_York on a trading day — that wall time is
    21:45 UTC in winter and 20:45 UTC in summer, so a single 21:45 UTC cron
    (and a later retry) is past the gate under both daylight-saving offsets.
    ``closed`` on weekends and NYSE holidays.  ``too_early`` before the gate.
    """
    ny = _as_ny(now)
    if not is_us_trading_day(ny.date()):
        return "closed", None
    if (ny.hour, ny.minute, ny.second) < (OPTIONS_CLOSE_GATE.hour,
                                          OPTIONS_CLOSE_GATE.minute,
                                          OPTIONS_CLOSE_GATE.second):
        return "too_early", None
    return "ok", ny.date()


def deribit_compare_window(session_start: datetime, session_end: datetime, now: datetime,
                          session_has_trades: bool) -> tuple[datetime, datetime, str | None]:
    """Window for the Deribit block column.

    The public tape only keeps about a day.  When it still contains the US
    session the IBIT chain describes, that session is the comparison.  When it
    does not, the column is the last 24 hours and the note says so — an empty
    column would look like Deribit printed nothing.
    """
    if session_has_trades:
        return session_start, session_end, None
    end = _as_ny(now).astimezone(UTC)
    start = end - timedelta(hours=24)
    note = ("Deribit's public option tape only keeps about the last day, so the "
            "US session on this IBIT snapshot is no longer available. "
            "The Deribit column is BTC blocks from the last 24 hours, not that session.")
    return start, end, note


def cash_session_bounds(session: date, now: datetime) -> tuple[datetime, datetime]:
    """UTC start/end of the US options session used for the Deribit comparison.

    The end is the 16:15 ET close, or ``now`` when that session is still open.
    """
    start = datetime.combine(session, CASH_OPEN, tzinfo=NY)
    close = datetime.combine(session, OPTIONS_CLOSE, tzinfo=NY)
    ny = _as_ny(now)
    end = ny if start <= ny < close else close
    return start.astimezone(UTC), end.astimezone(UTC)


def trade_price(df: pd.DataFrame) -> pd.Series:
    """Last print when there is one, otherwise the mid."""
    last = (pd.to_numeric(df["last"], errors="coerce") if "last" in df.columns
            else pd.Series(np.nan, index=df.index))
    mid = (pd.to_numeric(df["mid"], errors="coerce") if "mid" in df.columns
           else pd.Series(np.nan, index=df.index))
    return last.where(last > 0, mid)


def classify_aggressor(last, bid, ask) -> np.ndarray:
    """Last-print side versus the current quote.

    At or above the ask → ``bought``.  At or below the bid → ``sold``.
    Strictly inside, or a locked market (bid == ask), → ``ambiguous``.
    Missing/crossed quotes → ``unknown``.  This is the last print only.
    """
    last = np.asarray(last, dtype=float)
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    out = np.full(last.shape, "unknown", dtype=object)
    tradable = (np.isfinite(last) & np.isfinite(bid) & np.isfinite(ask)
                & (last > 0) & (bid > 0) & (ask > 0) & (ask + _AGG_EPS >= bid))
    locked = tradable & (np.abs(ask - bid) <= _AGG_EPS)
    bought = tradable & ~locked & (last + _AGG_EPS >= ask)
    sold = tradable & ~locked & ~bought & (last - _AGG_EPS <= bid)
    ambiguous = tradable & ~bought & ~sold
    out[bought] = "bought"
    out[sold] = "sold"
    out[ambiguous] = "ambiguous"
    return out


def _prior_oi_series(prior_oi) -> pd.Series | None:
    if prior_oi is None:
        return None
    if isinstance(prior_oi, pd.DataFrame):
        if prior_oi.empty or "option" not in prior_oi.columns:
            return pd.Series(dtype=float)
        return (prior_oi.drop_duplicates("option")
                .set_index("option")["oi"].astype(float))
    return pd.Series(prior_oi, dtype=float)


def unusual_activity(df: pd.DataFrame, prior_oi=None,
                     min_premium: float = 25_000.0,
                     min_contracts: float = 50.0) -> pd.DataFrame:
    """Contracts at or above the premium and size floors, largest premium first.

    ``likely_opening`` is true when today's volume exceeds prior open interest
    (a new listing counts as prior OI of 0) and/or published OI rose.  With no
    prior close, the flag falls back to volume greater than the OI on the chain.
    """
    x = df.copy()
    mult = x["mult"] if "mult" in x.columns else IBIT_MULT
    x["price"] = trade_price(x)
    x["premium"] = x["volume"] * x["price"] * mult
    x["delta_shares"] = x["volume"] * x["delta"] * mult
    x["delta_usd"] = x["delta_shares"] * x["spot"]
    prev = _prior_oi_series(prior_oi)
    if prev is None:
        x["oi_prev"] = np.nan
        x["oi_change"] = np.nan
        x["likely_opening"] = x["volume"] > x["oi"]
    else:
        x["oi_prev"] = x["option"].map(prev) if "option" in x.columns else np.nan
        x["oi_change"] = x["oi"] - x["oi_prev"]
        x["likely_opening"] = ((x["volume"] > x["oi_prev"].fillna(0.0))
                               | (x["oi_change"] > 0).fillna(False))
    if {"last", "bid", "ask"} <= set(x.columns):
        x["side"] = classify_aggressor(x["last"], x["bid"], x["ask"])
    else:
        x["side"] = "unknown"
    x["vol_oi"] = x["volume"] / x["oi"].replace(0, np.nan)
    x = x[(x["premium"] >= min_premium) & (x["volume"] >= min_contracts)]
    return x.sort_values(["premium", "option"], ascending=[False, True]).reset_index(drop=True)


def lean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Net premium and net delta by call/put, signed from the last print.

    The session's whole volume is attributed to that one print's side.  Mid
    and unknown prints contribute to gross premium and volume, not to the nets.
    """
    x = df.copy()
    if x.empty:
        x = pd.DataFrame(columns=list(df.columns) + ["side", "price", "signed_premium"])
    mult = x["mult"] if "mult" in x.columns else IBIT_MULT
    if {"last", "bid", "ask"} <= set(x.columns) and len(x):
        x["side"] = classify_aggressor(x["last"], x["bid"], x["ask"])
    else:
        x["side"] = "unknown" if len(x) else pd.Series(dtype=object)
    x["price"] = trade_price(x) if len(x) else pd.Series(dtype=float)
    sign = np.where(x["side"] == "bought", 1.0, np.where(x["side"] == "sold", -1.0, 0.0))
    spot = x["spot"] if "spot" in x.columns else np.nan
    delta = x["delta"] if "delta" in x.columns else np.nan
    volume = x["volume"] if "volume" in x.columns else 0.0
    x["gross_premium"] = volume * x["price"] * mult
    x["signed_premium"] = sign * x["gross_premium"]
    x["signed_delta_shares"] = sign * volume * delta * mult
    x["signed_delta_usd"] = x["signed_delta_shares"] * spot
    rows = []
    for typ, name in (("C", "Calls"), ("P", "Puts"), (None, "Total")):
        g = x if typ is None else x[x["type"] == typ] if "type" in x.columns else x.iloc[0:0]
        def _sum(col, side=None, frame=g):
            if not len(frame) or col not in frame.columns:
                return 0.0
            s = frame if side is None else frame[frame["side"] == side]
            v = pd.to_numeric(s[col], errors="coerce").sum()
            return float(v) if np.isfinite(v) else 0.0
        vol = _sum("volume") if "volume" in g.columns else 0.0
        rows.append(dict(
            side=name,
            net_premium=_sum("signed_premium"),
            gross_premium=_sum("gross_premium"),
            bought_premium=_sum("gross_premium", "bought"),
            sold_premium=_sum("gross_premium", "sold"),
            net_delta_shares=_sum("signed_delta_shares"),
            net_delta_usd=_sum("signed_delta_usd"),
            volume=vol,
            volume_bought=_sum("volume", "bought") if "volume" in g.columns else 0.0,
            volume_sold=_sum("volume", "sold") if "volume" in g.columns else 0.0,
            volume_ambiguous=_sum("volume", "ambiguous") if "volume" in g.columns else 0.0,
            volume_unknown=_sum("volume", "unknown") if "volume" in g.columns else 0.0,
        ))
    return pd.DataFrame(rows)


def premium_split(df: pd.DataFrame) -> dict:
    """Put/call premium and delta notional from session volume (not aggressor-signed)."""
    x = df.copy()
    mult = x["mult"] if "mult" in x.columns else IBIT_MULT
    price = trade_price(x) if len(x) else pd.Series(dtype=float)
    premium = (x["volume"] * price * mult) if len(x) else pd.Series(dtype=float)
    delta_shares = (x["volume"] * x["delta"] * mult) if len(x) else pd.Series(dtype=float)
    delta_usd = delta_shares * x["spot"] if len(x) else pd.Series(dtype=float)
    abs_delta_usd = delta_shares.abs() * x["spot"] if len(x) else pd.Series(dtype=float)
    out = {}
    for typ, name in (("C", "call"), ("P", "put")):
        m = (x["type"] == typ).to_numpy() if len(x) else np.array([], dtype=bool)
        out[f"{name}_premium"] = float(premium[m].sum()) if m.any() else 0.0
        out[f"{name}_delta_usd"] = float(delta_usd[m].sum()) if m.any() else 0.0
        out[f"{name}_delta_shares"] = float(delta_shares[m].sum()) if m.any() else 0.0
        out[f"{name}_abs_delta_usd"] = float(abs_delta_usd[m].sum()) if m.any() else 0.0
    out["premium"] = out["call_premium"] + out["put_premium"]
    out["delta_usd"] = out["call_delta_usd"] + out["put_delta_usd"]
    out["delta_shares"] = out["call_delta_shares"] + out["put_delta_shares"]
    out["abs_delta_usd"] = out["call_abs_delta_usd"] + out["put_abs_delta_usd"]
    return out


def flow_heatmap_frame(df: pd.DataFrame, max_dte: float = 120.0,
                       moneyness: float = 0.30) -> pd.DataFrame:
    """Premium by strike and expiry.  ``net_premium`` is calls minus puts."""
    if df.empty:
        return pd.DataFrame(columns=["expiry", "strike", "call_premium", "put_premium",
                                     "net_premium", "premium"])
    x = df[(df["dte"] <= max_dte) & ((df["m"] - 1).abs() <= moneyness) & (df["volume"] > 0)].copy()
    if x.empty:
        return pd.DataFrame(columns=["expiry", "strike", "call_premium", "put_premium",
                                     "net_premium", "premium"])
    mult = x["mult"] if "mult" in x.columns else IBIT_MULT
    x["premium"] = x["volume"] * trade_price(x) * mult
    x["expiry"] = pd.to_datetime(x["expiry"], utc=True)
    rows = []
    for (exp, strike), g in x.groupby(["expiry", "strike"], sort=True):
        call = float(g.loc[g["type"] == "C", "premium"].sum())
        put = float(g.loc[g["type"] == "P", "premium"].sum())
        rows.append(dict(expiry=exp, strike=float(strike), call_premium=call, put_premium=put,
                         net_premium=call - put, premium=call + put))
    return pd.DataFrame(rows)


# iShares publishes bitcoin quantity and shares outstanding on the free holdings
# CSV.  One option contract is 100 shares, and one Deribit BTC option is 1 BTC,
# so contracts × 100 × (bitcoin / shares) is the Deribit-contract equivalent.
# The fallback is the 13 Nov 2026 file (783,448.30480 BTC / 1,387,960,000 shares)
# and is used only when that CSV cannot be fetched.  It is not a price ratio.
ISHARES_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/333011/x/latest-holdings.csv"
)
_FALLBACK_BTC = 783_448.30480
_FALLBACK_SHARES = 1_387_960_000.0
IBIT_BTC_PER_SHARE_FALLBACK = _FALLBACK_BTC / _FALLBACK_SHARES
IBIT_BTC_PER_SHARE_ASOF = "2026-11-13"
_HOLDINGS_TTL = 3600.0
_HOLDINGS_CACHE: dict[str, tuple[float, dict]] = {}

# Bubble marks on the activity chart: pixel size = clip(sqrt(qty) / divisor, 5, 40).
# The divisor is 4 for IBIT contracts.  For BTC-equivalent size it is scaled by
# sqrt(100 × BTC-per-share) so a contract and its BTC equivalent draw the same
# mark — the conversion is linear, so the picture matches and only the key changes.
BUBBLE_DIVISOR = 4.0
BUBBLE_SIZE_MIN = 5.0
BUBBLE_SIZE_MAX = 40.0
BUBBLE_KEY_CONTRACTS = (1_000, 5_000, 20_000)


def parse_ishares_holdings(text: str) -> dict:
    """Bitcoin quantity and shares outstanding from the iShares holdings CSV."""
    asof_m = re.search(r'Fund Holdings as of,"([^"]+)"', text)
    sh_m = re.search(r'Shares Outstanding,"([^"]+)"', text)
    if not asof_m or not sh_m:
        raise ValueError("iShares holdings header missing")
    shares = float(sh_m.group(1).replace(",", ""))
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header_i = next((i for i, ln in enumerate(lines) if ln.startswith("Ticker,")), None)
    if header_i is None:
        raise ValueError("iShares holdings table missing")
    header = next(csv.reader([lines[header_i]]))
    qty_i = header.index("Quantity")
    btc = None
    for ln in lines[header_i + 1:]:
        if not (ln.startswith('"BTC"') or ln.startswith("BTC,")):
            continue
        row = next(csv.reader([ln]))
        btc = float(row[qty_i].replace(",", ""))
        break
    if btc is None or btc <= 0 or shares <= 0:
        raise ValueError("iShares holdings had no bitcoin quantity")
    asof = datetime.strptime(asof_m.group(1), "%b %d, %Y").date().isoformat()
    return dict(btc=btc, shares=shares, asof=asof, btc_per_share=btc / shares,
                source="iShares", fallback=False)


def _holdings_fallback() -> dict:
    return dict(btc=_FALLBACK_BTC, shares=_FALLBACK_SHARES,
                asof=IBIT_BTC_PER_SHARE_ASOF,
                btc_per_share=IBIT_BTC_PER_SHARE_FALLBACK,
                source="fallback", fallback=True)


def btc_per_share_ratio() -> dict:
    """Live iShares BTC-per-share, or the dated fallback.  Never raises."""
    now = _time.time()
    hit = _HOLDINGS_CACHE.get("ibit")
    if hit is not None and now - hit[0] < _HOLDINGS_TTL:
        return hit[1]
    info = _holdings_fallback()
    try:
        resp = requests.get(ISHARES_HOLDINGS_URL, timeout=8,
                            headers={"User-Agent": "Mozilla/5.0 (MCM Analytics)"})
        if resp.status_code == 200 and "Shares Outstanding" in resp.text:
            info = parse_ishares_holdings(resp.text)
    except Exception:
        info = _holdings_fallback()
    _HOLDINGS_CACHE["ibit"] = (now, info)
    return info


def ibit_btc_equivalent(shares, btc_per_share: float):
    """Shares (or share-delta) → BTC using the fund's BTC per share.

    One IBIT option is 100 shares, so contract volume × 100 × this ratio is
    the Deribit equivalent (one Deribit BTC option is 1 BTC).  A strike in
    dollars per share divided by the same ratio is the dollars-per-BTC strike.
    """
    bps = float(btc_per_share)
    out = np.asarray(shares, dtype=float) * bps
    if not np.isfinite(bps) or bps <= 0:
        out = np.full(np.shape(shares), np.nan, dtype=float)
    if np.shape(out) == ():
        return float(out)
    return out


def contracts_to_btc(contracts, btc_per_share: float):
    """IBIT contracts → BTC, i.e. Deribit-contract equivalent."""
    return ibit_btc_equivalent(np.asarray(contracts, dtype=float) * IBIT_MULT, btc_per_share)


def strike_btc_equivalent(strike, btc_per_share: float):
    """IBIT strike (USD per share) → BTC-equivalent strike (USD per BTC)."""
    bps = float(btc_per_share)
    values = np.asarray(strike, dtype=float)
    if not np.isfinite(bps) or bps <= 0:
        out = np.full(values.shape, np.nan, dtype=float)
    else:
        out = values / bps
    if out.shape == ():
        return float(out)
    return out


def bubble_size_divisor(unit: str, btc_per_share: float) -> float:
    """Pixel divisor.  ``btc`` is scaled so a contract and its BTC size match."""
    if unit == "btc":
        return BUBBLE_DIVISOR * float(np.sqrt(IBIT_MULT * float(btc_per_share)))
    return BUBBLE_DIVISOR


def bubble_marker_size(quantity, divisor: float = BUBBLE_DIVISOR) -> np.ndarray:
    """Sqrt size, floored and capped.  Same function the chart and its key use."""
    q = np.asarray(quantity, dtype=float)
    return np.clip(np.sqrt(np.maximum(q, 0.0)) / float(divisor),
                   BUBBLE_SIZE_MIN, BUBBLE_SIZE_MAX)


def bubble_size_key(btc_per_share: float, unit: str = "contracts") -> list[dict]:
    """Three reference marks: 1k / 5k / 20k IBIT contracts and their BTC size."""
    divisor = bubble_size_divisor("btc" if unit == "btc" else "contracts", btc_per_share)
    out = []
    for n in BUBBLE_KEY_CONTRACTS:
        btc = float(contracts_to_btc(n, btc_per_share))
        qty = btc if unit == "btc" else float(n)
        out.append(dict(contracts=int(n), btc=btc,
                        size=float(bubble_marker_size(qty, divisor))))
    return out


def summarize_deribit_blocks(df: pd.DataFrame | None) -> dict:
    """Gross premium and taker-signed delta for a Deribit block frame.

    A block is a trade at or above the venue minimum (see ``lib.flow_summary``).
    Premium is already in USD (coin price × index).  Empty input → zeros.
    """
    empty = dict(blocks=0, premium=0.0, call_premium=0.0, put_premium=0.0,
                 net_premium=0.0, net_delta_usd=np.nan, net_delta_btc=np.nan)
    if df is None or df.empty:
        return empty
    gross = pd.to_numeric(df["abs_amount"], errors="coerce") * pd.to_numeric(df["price"], errors="coerce")
    signed = pd.to_numeric(df["amount"], errors="coerce") * pd.to_numeric(df["price"], errors="coerce")
    call = df["option_type"] == "C"
    out = dict(
        blocks=int(len(df)),
        premium=float(gross.sum()),
        call_premium=float(gross[call].sum()),
        put_premium=float(gross[~call].sum()),
        net_premium=float(signed.sum()),
        net_delta_usd=np.nan,
        net_delta_btc=np.nan,
    )
    try:
        from lib import block_flow
        idx = pd.to_numeric(df["index_price"], errors="coerce").dropna()
        spot = float(idx.iloc[-1]) if len(idx) else float("nan")
        enriched = block_flow.enrich(df.copy(), spot, "BTC", True)
        out["net_delta_usd"] = float(enriched["dollar_delta"].sum())
        out["net_delta_btc"] = float((enriched["delta"] * enriched["amount"]).sum())
    except Exception:
        pass
    return out


def oi_change(current: pd.DataFrame, prior: pd.DataFrame,
              asof: datetime | None = None) -> pd.DataFrame:
    """Contract-level OI change.  Contracts that left because they expired are
    ``expired`` (not a close).  A name only on the prior close is ``closed``;
    only on the current chain is ``new``.
    """
    cur = current.drop_duplicates("option").set_index("option")
    pri = prior.drop_duplicates("option").set_index("option")
    keys = cur.index.union(pri.index)

    def _col(frame, name):
        if name not in frame.columns:
            return pd.Series(np.nan, index=keys)
        return frame[name].reindex(keys)

    oi_now = pd.to_numeric(_col(cur, "oi"), errors="coerce").fillna(0.0)
    oi_prev = pd.to_numeric(_col(pri, "oi"), errors="coerce").fillna(0.0)
    expiry = _col(cur, "expiry").fillna(_col(pri, "expiry"))
    spot = pd.to_numeric(_col(cur, "spot"), errors="coerce").fillna(
        pd.to_numeric(_col(pri, "spot"), errors="coerce"))
    out = pd.DataFrame({
        "option": keys.astype(str),
        "expiry": expiry.values,
        "type": _col(cur, "type").fillna(_col(pri, "type")).values,
        "strike": pd.to_numeric(_col(cur, "strike"), errors="coerce").fillna(
            pd.to_numeric(_col(pri, "strike"), errors="coerce")).values,
        "spot": spot.values,
        "oi": oi_now.values,
        "oi_prev": oi_prev.values,
        "oi_change": (oi_now - oi_prev).values,
        "volume": pd.to_numeric(_col(cur, "volume"), errors="coerce").fillna(0.0).values,
    })
    in_cur = out["option"].isin(set(cur.index.astype(str)))
    in_pri = out["option"].isin(set(pri.index.astype(str)))
    exp_ts = pd.to_datetime(out["expiry"], utc=True, errors="coerce")
    asof_ts = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now(tz=UTC)
    if asof_ts.tzinfo is None:
        asof_ts = asof_ts.tz_localize(UTC)
    expired = ~in_cur & exp_ts.notna() & (exp_ts <= asof_ts)
    status = np.full(len(out), "flat", dtype=object)
    status = np.where(~in_pri, "new", status)
    status = np.where(~in_cur & ~expired, "closed", status)
    status = np.where(expired, "expired", status)
    both = in_cur & in_pri
    status = np.where(both & (out["oi_change"] > 0), "increase", status)
    status = np.where(both & (out["oi_change"] < 0), "decrease", status)
    out["status"] = status
    out["oi_change_usd"] = out["oi_change"] * IBIT_MULT * out["spot"]
    return out.sort_values(["oi_change", "option"], ascending=[False, True]).reset_index(drop=True)


def largest_oi_moves(changes: pd.DataFrame, n: int = 15
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Biggest increases and decreases, excluding expiries that simply dropped off."""
    if changes is None or changes.empty:
        empty = changes.iloc[0:0] if changes is not None else pd.DataFrame()
        return empty, empty
    live = changes[changes["status"] != "expired"]
    up = live[live["oi_change"] > 0].head(n)
    down = live[live["oi_change"] < 0].sort_values("oi_change", ascending=True).head(n)
    return up.reset_index(drop=True), down.reset_index(drop=True)


def history_notice(store: pd.DataFrame | None, current_session: str) -> dict:
    """Copy for the page, and whether a real prior close exists to diff against."""
    dates: list[str] = []
    if store is not None and len(store):
        dates = sorted({str(x)[:10] for x in store["session_date"]})
    current_session = str(current_session)[:10]
    prior = [d for d in dates if d < current_session]
    since = dates[0] if dates else None
    ready = bool(prior)
    if not dates:
        text = ("Collecting history — no daily close has been stored yet. "
                "Open-interest change appears once a prior close is on record. "
                "The screens below use the latest chain's volume.")
    elif not ready:
        text = (f"Collecting history since {since}. "
                "Open-interest change needs a prior recorded close; "
                "volume-based screens below use the latest chain.")
    else:
        noun = "session" if len(dates) == 1 else "sessions"
        text = (f"Open-interest change versus the {prior[-1]} close "
                f"(history since {since}, {len(dates)} recorded {noun}). "
                "Cboe publishes open interest overnight, so this is net opening and closing "
                "between recorded closes, not a trade-by-trade tape.")
    return dict(ready=ready, since=since, prior=prior[-1] if prior else None,
                n_sessions=len(dates), latest=dates[-1] if dates else None, text=text)


def prior_chain(store: pd.DataFrame | None, current_session: str
                ) -> tuple[str | None, pd.DataFrame | None]:
    """The newest stored session strictly before ``current_session``."""
    notice = history_notice(store, current_session)
    if not notice["ready"] or store is None:
        return None, None
    day = notice["prior"]
    rows = store[store["session_date"].astype(str).str.slice(0, 10) == day].copy()
    return day, rows


# --------------------------------------------------------------------------
# Daily chain store (parquet on the snapshots branch)
# --------------------------------------------------------------------------

def chain_store_path(base: Path | None = None) -> Path:
    if base is None:
        from lib.history import SNAPSHOT_DIR
        base = SNAPSHOT_DIR
    return Path(base) / CHAIN_FILENAME


def _normalize_store(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in CHAIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"IBIT chain store missing {missing}")
    out = df[CHAIN_COLUMNS].copy()
    for c in ("session_date", "recorded_at", "option", "expiry", "type"):
        out[c] = out[c].astype(str)
    out["session_date"] = out["session_date"].str.slice(0, 10)
    for c in ("strike", "spot", "bid", "ask", "last", "iv", "volume", "oi", "delta"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.drop_duplicates(["session_date", "option"], keep="last")
    return out.sort_values(["session_date", "option"]).reset_index(drop=True)


def read_chain_store(base: Path | None = None) -> pd.DataFrame | None:
    path = chain_store_path(base)
    if not path.exists():
        return None
    return _normalize_store(pd.read_parquet(path))


def merge_chain_frames(base: pd.DataFrame | None, incoming: pd.DataFrame | None,
                       incoming_wins: bool = False) -> pd.DataFrame:
    """Union on ``(session_date, option)``.  Existing rows win unless ``incoming_wins``."""
    if base is None or (isinstance(base, pd.DataFrame) and base.empty):
        if incoming is None or incoming.empty:
            return pd.DataFrame(columns=CHAIN_COLUMNS)
        return _normalize_store(incoming)
    if incoming is None or incoming.empty:
        return _normalize_store(base)
    left, right = (_normalize_store(incoming), _normalize_store(base)) if not incoming_wins \
        else (_normalize_store(base), _normalize_store(incoming))
    # ``right`` is concatenated last so drop_duplicates(keep="last") prefers it.
    both = pd.concat([left, right], ignore_index=True)
    return _normalize_store(both)


def append_chain_snapshot(frame: pd.DataFrame, base: Path | None = None) -> Path:
    """Append rows.  Refuses an empty frame and never drops an existing key.

    Overlapping ``(session_date, option)`` keys keep the values already stored.
    """
    if frame is None or len(frame) == 0:
        raise ValueError("refusing to write an empty IBIT chain snapshot")
    path = chain_store_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_chain_store(base) if path.exists() else None
    merged = merge_chain_frames(existing, frame, incoming_wins=False)
    if existing is not None:
        old_keys = set(zip(existing["session_date"], existing["option"]))
        new_keys = set(zip(merged["session_date"], merged["option"]))
        if not old_keys <= new_keys:
            raise RuntimeError("IBIT chain append would drop published rows; not writing")
    tmp = path.with_suffix(".parquet.tmp")
    try:
        merged.to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def validate_chain_history(old: pd.DataFrame | None, new: pd.DataFrame | None,
                           max_new_sessions: int | None = 1) -> list[str]:
    """Problems that must block a publish.  Empty means the candidate is safe.

    Published ``(session_date, option)`` keys and their OI/volume/price fields
    have to survive unchanged.  After the first file exists, at most
    ``max_new_sessions`` new dates may appear, and none of them may be older
    than the newest published date (that would be a backfill).
    """
    if old is None or (isinstance(old, pd.DataFrame) and old.empty):
        if new is None or (isinstance(new, pd.DataFrame) and new.empty):
            return []
        return _sanity_new_rows(_normalize_store(new), [])
    if new is None or (isinstance(new, pd.DataFrame) and new.empty):
        return ["candidate file missing or empty; published history would be lost"]
    try:
        old_n, new_n = _normalize_store(old), _normalize_store(new)
    except ValueError as exc:
        return [str(exc)]
    errs: list[str] = []
    if old_n.duplicated(["session_date", "option"]).any():
        errs.append("published file has duplicate keys")
    if new_n.duplicated(["session_date", "option"]).any():
        errs.append("duplicate session/contract keys")
    old_keys = set(zip(old_n["session_date"], old_n["option"]))
    new_keys = set(zip(new_n["session_date"], new_n["option"]))
    lost = old_keys - new_keys
    if lost:
        sample = next(iter(sorted(lost)))
        errs.append(f"{len(lost)} published rows would be lost (e.g. {sample[0]} {sample[1]})")
    merged = old_n.merge(new_n, on=["session_date", "option"], suffixes=("_o", "_n"), how="left")
    for col in ("oi", "volume", "strike", "last", "delta", "spot"):
        a = pd.to_numeric(merged[f"{col}_o"], errors="coerce").to_numpy()
        b = pd.to_numeric(merged[f"{col}_n"], errors="coerce").to_numpy()
        if not np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-6):
            errs.append(f"published {col} values would change")
    if not (merged["recorded_at_o"].astype(str) == merged["recorded_at_n"].astype(str)).all():
        errs.append("published recorded_at values would change")
    new_sessions = sorted(set(new_n["session_date"]) - set(old_n["session_date"]))
    if max_new_sessions is not None and len(new_sessions) > max_new_sessions:
        errs.append(f"{len(new_sessions)} new sessions (max {max_new_sessions})")
    if new_sessions and min(new_sessions) < old_n["session_date"].max():
        errs.append("new session is older than the newest published session")
    added = new_n[new_n["session_date"].isin(new_sessions)] if new_sessions else new_n.iloc[0:0]
    return _sanity_new_rows(added, errs)


def _sanity_new_rows(df: pd.DataFrame, errs: list[str]) -> list[str]:
    if df is None or df.empty:
        return errs
    oi = pd.to_numeric(df["oi"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce")
    if oi.isna().any() or (oi < 0).any():
        errs.append("oi missing or negative")
    if vol.isna().any() or (vol < 0).any():
        errs.append("volume missing or negative")
    if df["option"].eq("").any() or df["option"].eq("nan").any() or df["session_date"].eq("").any():
        errs.append("blank option or session_date")
    return errs


def snapshot_rows_from_chain(df: pd.DataFrame, session: date, recorded_at: datetime) -> pd.DataFrame:
    """One stored row per contract.  Real chain only — an empty chain stays empty."""
    if df is None or df.empty or "option" not in df.columns:
        return pd.DataFrame(columns=CHAIN_COLUMNS)
    exp = pd.to_datetime(df["expiry"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    recorded = pd.Timestamp(recorded_at)
    if recorded.tzinfo is None:
        recorded = recorded.tz_localize(UTC)
    else:
        recorded = recorded.tz_convert(UTC)
    out = pd.DataFrame({
        "session_date": session.isoformat(),
        "recorded_at": recorded.isoformat(),
        "option": df["option"].astype(str),
        "expiry": exp.fillna(""),
        "type": df["type"].astype(str),
        "strike": pd.to_numeric(df["strike"], errors="coerce"),
        "spot": pd.to_numeric(df["spot"], errors="coerce"),
        "bid": pd.to_numeric(df["bid"], errors="coerce"),
        "ask": pd.to_numeric(df["ask"], errors="coerce"),
        "last": pd.to_numeric(df["last"], errors="coerce") if "last" in df.columns else 0.0,
        "iv": pd.to_numeric(df["iv"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce").fillna(0.0),
        "oi": pd.to_numeric(df["oi"], errors="coerce").fillna(0.0),
        "delta": pd.to_numeric(df["delta"], errors="coerce"),
    })
    out = out[out["option"].str.len() > 0]
    return _normalize_store(out)


def record_chain_snapshot(now: datetime, fetch, base: Path | None = None) -> str:
    """Append today's close, or say why it didn't.

    ``recorded`` / ``skipped`` (already stored) / ``closed`` (weekend or
    holiday) / ``too_early`` / ``stale`` (the chain is not this session) /
    ``no_data`` / ``error``.  A failure leaves the file untouched.  ``fetch``
    returns ``(chain dataframe, meta)`` and is not called on a skip, a closure,
    or before the gate.
    """
    try:
        decision, session = recording_decision(now)
        if decision != "ok":
            return decision
        existing = None
        path = chain_store_path(base)
        if path.exists():
            existing = read_chain_store(base)
            have = set(existing["session_date"]) if existing is not None else set()
            if session.isoformat() in have:
                return "skipped"
        chain, meta = fetch()
        snap = (meta or {}).get("snapshot_ny")
        if snap is not None and session_date_of(snap) != session:
            return "stale"
        rows = snapshot_rows_from_chain(chain, session, now)
        if rows.empty:
            return "no_data"
        append_chain_snapshot(rows, base)
        return "recorded"
    except Exception:
        return "error"


def clear_cache() -> None:
    """Drop the in-process copies of the remote chain file and the iShares ratio."""
    _CHAIN_REMOTE_CACHE.clear()
    _HOLDINGS_CACHE.clear()


def _fetch_remote_chain() -> pd.DataFrame | None:
    now = _time.time()
    hit = _CHAIN_REMOTE_CACHE.get("ibit")
    if hit is not None and now - hit[0] < _CHAIN_REMOTE_TTL:
        return hit[1]
    df = None
    try:
        from lib.history import _REMOTE_SNAPSHOT_BASE
        url = f"{_REMOTE_SNAPSHOT_BASE}/{CHAIN_FILENAME}"
        resp = requests.get(url, timeout=8,
                            headers={"User-Agent": "Mozilla/5.0 (MCM Analytics)"})
        if resp.status_code == 200 and resp.content:
            df = _normalize_store(pd.read_parquet(io.BytesIO(resp.content)))
    except Exception:
        df = None
    _CHAIN_REMOTE_CACHE["ibit"] = (now, df)
    return df


def load_chain_history(base: Path | None = None) -> pd.DataFrame | None:
    """Recorded closes: local file overlaid on the snapshots branch.

    Local wins where a key overlaps (it is at least as fresh).  A missing
    file or a failed fetch returns whatever the other side has, or None.
    Never raises — the page falls back to volume screens.
    """
    local = remote = None
    try:
        local = read_chain_store(base)
    except Exception:
        local = None
    try:
        remote = _fetch_remote_chain()
    except Exception:
        remote = None
    if local is None:
        return remote
    if remote is None:
        return local
    return merge_chain_frames(remote, local, incoming_wins=True)
