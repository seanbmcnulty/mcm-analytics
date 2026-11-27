"""10x Research chart collection: data, maths and the chart registry. No Streamlit here.

Sean asked (27 Nov 2026) for our own copy of the charts in the 10x Research
emails (reports@update.10xresearch.com, Gmail label "Markets"). Each chart is
rebuilt from free public data. The 10x images and text are licensed to the
recipient only, so nothing of theirs is stored in this (public) repo: each
registry entry carries just the post title, date and link, and our own
one-line description.

Adding a chart from a new email
-------------------------------
1. Write the maths here (pure functions on pandas objects) and a figure
   builder in ``lib.tenx_charts``.
2. Add any new Yahoo symbol to ``YAHOO_SYMBOLS`` and run
   ``python3 -c "from lib import tenx; tenx.write_prices_csv()"`` so the
   committed fallback CSV carries it.
3. Put a ``TenXChart`` at the TOP of ``REGISTRY`` (newest first) and map its
   ``key`` to the builder in ``lib.tenx_charts.BUILDERS``.

The registry is keyed by post: ``post_id`` is "YYYY-MM-DD | post title" (date
the email arrived, SGT). Before adding a chart, check ``has_post(date, title)``
so a re-run of the daily 10x check never adds the same post twice;
``charts_by_post()`` groups the registry per post, newest first.

Data
----
Daily closes come from Yahoo Finance's chart API (live, the same endpoint
``lib.vol_lead_data`` uses). ``data/tenx_prices.csv`` is a committed copy of
those same real Yahoo closes. The page uses live data where it gets it and
the CSV for anything live did not return. Nothing is filled or invented: a
missing day stays missing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_CSV = "tenx_prices.csv"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}
PRICES_START = "2023-01-01"
# column name -> Yahoo symbol
YAHOO_SYMBOLS = {"BTC": "BTC-USD", "GOLD": "GC=F", "COIN": "COIN"}
SYMBOL_LABELS = {"BTC": "Bitcoin (Yahoo BTC-USD, daily close UTC)",
                 "GOLD": "Gold front-month future (Yahoo GC=F, COMEX settle)",
                 "COIN": "Coinbase (Yahoo COIN, Nasdaq close)"}


# ------------------------------------------------------------------ registry
@dataclass(frozen=True)
class TenXChart:
    key: str              # stable id, maps to a builder in lib.tenx_charts
    tab: str              # short tab label
    exhibit: str          # 10x's exhibit name
    post_title: str
    post_date: str        # email received date, SGT, YYYY-MM-DD
    post_url: str
    summary: str          # our own one-line description of the chart
    series: tuple = field(default_factory=tuple)  # columns of the prices frame it needs

    @property
    def post_id(self) -> str:
        return post_id(self.post_date, self.post_title)


REGISTRY: tuple[TenXChart, ...] = (
    TenXChart(
        key="btc_rsi",
        tab="BTC daily RSI",
        exhibit="Exhibit 2: Bitcoin daily RSI",
        post_title="Bitcoin and Gold: The Thanksgiving Charts",
        post_date="2026-11-27",
        post_url="https://update.10xresearch.com/p/bitcoin-and-gold-the-thanksgiving-charts",
        summary=("Bitcoin price against its 14-day RSI, with 30 / 70 bands. 10x flagged RSI near 26 as "
                 "a buy reversal signal, like the Feb and Jun 2026 lows."),
        series=("BTC",),
    ),
    TenXChart(
        key="gold_channel",
        tab="Gold channel",
        exhibit="Exhibit 30: Gold (GC)",
        post_title="Bitcoin and Gold: The Thanksgiving Charts",
        post_date="2026-11-27",
        post_url="https://update.10xresearch.com/p/bitcoin-and-gold-the-thanksgiving-charts",
        summary=("Gold futures with 7-day and 30-day moving averages, the bearish channel from the "
                 "2026 peak, and gold's 1-year return relative to Bitcoin. 10x: gold has reached "
                 "the channel's lower boundary."),
        series=("GOLD", "BTC"),
    ),
    TenXChart(
        key="btc_coin",
        tab="BTC vs Coinbase",
        exhibit="Exhibit 8: Bitcoin vs. Coinbase",
        post_title="One Liners: Crypto Market Drivers — 6 Charts That Matter Now",
        post_date="2026-11-27",
        post_url="https://update.10xresearch.com/p/one-liners-crypto-market-drivers-6-charts-that-matter-now-d444",
        summary=("Bitcoin (left axis) against Coinbase stock (right axis) since Dec 2023. COIN has "
                 "fallen further than BTC in the latest leg down."),
        series=("BTC", "COIN"),
    ),
)


def post_id(post_date: str, post_title: str) -> str:
    """Registry key for one 10x post: "YYYY-MM-DD | title" (whitespace-normalised)."""
    return f"{pd.Timestamp(post_date):%Y-%m-%d} | {' '.join(str(post_title).split())}"


def registry_keys() -> list[str]:
    return [c.key for c in REGISTRY]


def charts_by_post() -> dict[str, list[TenXChart]]:
    """post_id -> its charts, posts newest first (registry order kept within a post)."""
    out: dict[str, list[TenXChart]] = {}
    for c in sorted(REGISTRY, key=lambda c: c.post_date, reverse=True):
        out.setdefault(c.post_id, []).append(c)
    return out


def has_post(post_date: str, post_title: str) -> bool:
    return post_id(post_date, post_title) in charts_by_post()


# ---------------------------------------------------------------------- data
@dataclass
class Prices:
    df: pd.DataFrame                 # index date (naive), one column per YAHOO_SYMBOLS key
    source: dict                     # column -> "live" | "csv" | "live+csv" | "none"
    errors: list                     # human-readable fetch errors


def parse_yahoo_chart(payload: dict, name: str) -> pd.Series:
    """Yahoo v8 chart JSON -> Series of daily closes on the exchange's calendar date."""
    res = (payload.get("chart") or {}).get("result") or []
    if not res:
        raise ValueError(f"Yahoo {name}: empty result")
    res = res[0]
    ts = res.get("timestamp") or []
    closes = ((res.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    if len(ts) != len(closes):
        raise ValueError(f"Yahoo {name}: {len(ts)} timestamps vs {len(closes)} closes")
    tz = (res.get("meta") or {}).get("exchangeTimezoneName") or "America/New_York"
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    s = pd.Series(pd.to_numeric(pd.Series(closes, dtype="float64"), errors="coerce").values,
                  index=idx, name=name)
    s = s[np.isfinite(s.to_numpy()) & (s.to_numpy() > 0)]
    s = s[~s.index.duplicated(keep="last")]
    s.index.name = "date"
    return s.sort_index()


def fetch_yahoo(sym: str, name: str, start: str = PRICES_START, get=None, timeout: int = 20) -> pd.Series:
    get = get or requests.get
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(time.time()) + 86400
    r = get(YAHOO_URL.format(sym=sym), params={"period1": p1, "period2": p2, "interval": "1d"},
            headers=UA, timeout=timeout)
    r.raise_for_status()
    return parse_yahoo_chart(r.json(), name)


def load_prices_csv(data_dir: Path | None = None) -> pd.DataFrame:
    p = (data_dir or DATA_DIR) / PRICES_CSV
    if not p.exists():
        return pd.DataFrame(columns=list(YAHOO_SYMBOLS)).rename_axis("date")
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def merge_live_csv(live: dict, csv: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Live closes win on a shared date; the CSV supplies dates live did not return."""
    cols, source = {}, {}
    for name in YAHOO_SYMBOLS:
        lv = live.get(name)
        cv = csv[name].dropna() if name in csv.columns else pd.Series(dtype=float)
        if lv is not None and len(lv):
            extra = cv[~cv.index.isin(lv.index)]
            s = pd.concat([lv, extra]).sort_index()
            source[name] = "live+csv" if len(extra) else "live"
        elif len(cv):
            s, source[name] = cv, "csv"
        else:
            s, source[name] = pd.Series(dtype=float), "none"
        cols[name] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df, source


def load_prices(get=None, data_dir: Path | None = None, live: bool = True) -> Prices:
    live_cols, errors = {}, []
    if live:
        for name, sym in YAHOO_SYMBOLS.items():
            try:
                live_cols[name] = fetch_yahoo(sym, name, get=get)
            except Exception as ex:
                errors.append(f"{sym}: {ex}")
    df, source = merge_live_csv(live_cols, load_prices_csv(data_dir))
    return Prices(df, source, errors)


def write_prices_csv(data_dir: Path | None = None, get=None) -> Path:
    """Refresh the committed fallback from live Yahoo (real closes only)."""
    pr = load_prices(get=get, data_dir=data_dir)
    if pr.errors:
        raise RuntimeError("; ".join(pr.errors))
    p = (data_dir or DATA_DIR) / PRICES_CSV
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    df = pr.df[(pr.df.index >= pd.Timestamp(PRICES_START)) & (pr.df.index < today)]  # drop today's partial bar
    df.to_csv(p, date_format="%Y-%m-%d", float_format="%.4f")
    return p


# --------------------------------------------------------------------- maths
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI: the first average gain / loss is the simple mean of the first ``n``
    changes, then avg = (prev * (n - 1) + change) / n. The first ``n`` bars are NaN."""
    c = pd.Series(close, dtype="float64").dropna()
    out = pd.Series(np.nan, index=c.index, name="rsi")
    if len(c) <= n:
        return out
    d = np.diff(c.to_numpy())
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()
    vals = np.full(len(c), np.nan)

    def _r(au, ad):
        if ad == 0:
            return 100.0 if au > 0 else 50.0
        return 100 - 100 / (1 + au / ad)

    vals[n] = _r(au, ad)
    for i in range(n, len(d)):
        au = (au * (n - 1) + up[i]) / n
        ad = (ad * (n - 1) + dn[i]) / n
        vals[i + 1] = _r(au, ad)
    out[:] = vals
    return out


def rsi_view(r: float, diff: float, lo: float = 30, hi: float = 70) -> str:
    """Label in the spirit of 10x's 'View' cell: oversold and turning up = bullish."""
    if not np.isfinite(r):
        return "n/a"
    if r <= lo:
        return "bullish" if diff > 0 else "oversold"
    if r >= hi:
        return "bearish" if diff < 0 else "overbought"
    return "neutral"


def rsi_frame(close: pd.Series, n: int = 14, ma: int = 5, lo: float = 30, hi: float = 70,
              start=None, gap_days: int = 20) -> tuple[pd.DataFrame, dict]:
    """Price, RSI, RSI moving average and buy-reversal markers.

    A buy reversal is the first day the RSI, while at or below ``lo``, crosses
    up through its own ``ma``-day average. Signals within ``gap_days`` of the
    previous one are dropped so one washout gives one arrow.
    """
    c = pd.Series(close, dtype="float64").dropna().sort_index()
    df = pd.DataFrame({"close": c, "rsi": rsi(c, n)})
    df["ma"] = df["rsi"].rolling(ma, min_periods=ma).mean()
    cross = (df["rsi"] > df["ma"]) & (df["rsi"].shift(1) <= df["ma"].shift(1))
    raw = cross & (df["rsi"] <= lo)
    sig, last = [], None
    for d in df.index[raw.fillna(False).to_numpy()]:
        if last is None or (d - last).days > gap_days:
            sig.append(d)
            last = d
    df["signal"] = df.index.isin(sig)
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    df = df.dropna(subset=["rsi"])
    if df.empty:
        return df, {}
    last_row = df.iloc[-1]
    diff = float(last_row["rsi"] - last_row["ma"]) if np.isfinite(last_row["ma"]) else float("nan")
    stats = {"date": df.index[-1], "level": float(last_row["close"]), "rsi": float(last_row["rsi"]),
             "ma": float(last_row["ma"]), "diff": diff, "view": rsi_view(float(last_row["rsi"]), diff, lo, hi),
             "n": n, "ma_n": ma, "lo": lo, "hi": hi, "signals": list(df.index[df["signal"]])}
    return df, stats


def relative_return(a: pd.Series, b: pd.Series, days: int = 365) -> pd.Series:
    """a's trailing ``days``-calendar-day return minus b's, in percent, on a's dates.

    Each leg uses the last close on or before (date - days); a date whose
    look-back has no close yet is NaN, not filled.
    """
    a = pd.Series(a, dtype="float64").dropna().sort_index()
    b = pd.Series(b, dtype="float64").dropna().sort_index()

    def trailing(s: pd.Series, on: pd.DatetimeIndex) -> pd.Series:
        now = s.reindex(s.index.union(on)).ffill().reindex(on)  # close on/before each date (same-day for a)
        past_idx = on - pd.Timedelta(days=days)
        past = s.reindex(s.index.union(past_idx)).ffill().reindex(past_idx)
        past.index = on
        past[past_idx < s.index[0]] = np.nan
        return now / past - 1

    if a.empty or b.empty:
        return pd.Series(dtype=float)
    on = a.index
    out = (trailing(a, on) - trailing(b, on)) * 100
    return out.rename("relative")


@dataclass
class Channel:
    start: pd.Timestamp
    end: pd.Timestamp
    slope_per_day: float
    mid0: float          # midline value at ``start``
    upper_off: float
    lower_off: float

    def at(self, d, which: str = "mid") -> float:
        t = (pd.Timestamp(d) - self.start).days
        base = self.mid0 + self.slope_per_day * t
        return base + {"mid": 0.0, "upper": self.upper_off, "lower": self.lower_off}[which]

    def position(self, d, price: float) -> float:
        """0 = on the lower line, 1 = on the upper line."""
        lo, hi = self.at(d, "lower"), self.at(d, "upper")
        return (price - lo) / (hi - lo) if hi != lo else float("nan")


def fit_channel(close: pd.Series, start=None, lookback_days: int = 400) -> Channel | None:
    """Least-squares trend channel from ``start`` (default: the highest close in the last
    ``lookback_days``) to the latest close, with parallel lines through the largest
    residual above and below the trend."""
    c = pd.Series(close, dtype="float64").dropna().sort_index()
    if c.empty:
        return None
    if start is None:
        recent = c[c.index >= c.index[-1] - pd.Timedelta(days=lookback_days)]
        start = recent.idxmax()
    seg = c[c.index >= pd.Timestamp(start)]
    if len(seg) < 20:
        return None
    t = np.asarray((seg.index - seg.index[0]).days, dtype=float)
    slope, icpt = np.polyfit(t, seg.to_numpy(), 1)
    resid = seg.to_numpy() - (icpt + slope * t)
    return Channel(seg.index[0], seg.index[-1], float(slope), float(icpt),
                   float(resid.max()), float(resid.min()))


def gold_frame(gold: pd.Series, btc: pd.Series, start=None, rel_days: int = 365,
               ma_fast: int = 7, ma_slow: int = 30) -> tuple[pd.DataFrame, dict]:
    g = pd.Series(gold, dtype="float64").dropna().sort_index()
    df = pd.DataFrame({"gold": g})
    df["ma_fast"] = g.rolling(ma_fast, min_periods=ma_fast).mean()
    df["ma_slow"] = g.rolling(ma_slow, min_periods=ma_slow).mean()
    df["relative"] = relative_return(g, btc, rel_days)
    ch = fit_channel(g)
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if df.empty:
        return df, {}
    last = df.iloc[-1]
    stats = {"date": df.index[-1], "gold": float(last["gold"]), "ma_fast": float(last["ma_fast"]),
             "ma_slow": float(last["ma_slow"]), "relative": float(last["relative"]),
             "channel": ch, "ma_fast_n": ma_fast, "ma_slow_n": ma_slow, "rel_days": rel_days}
    if ch is not None:
        stats["chan_lower"] = ch.at(df.index[-1], "lower")
        stats["chan_upper"] = ch.at(df.index[-1], "upper")
        stats["chan_pos"] = ch.position(df.index[-1], float(last["gold"]))
    return df, stats


def pair_frame(btc: pd.Series, other: pd.Series, start=None, corr_days: int = 90,
               chg_days: int = 90) -> tuple[pd.DataFrame, dict]:
    """BTC and another asset on the other asset's trading dates (BTC trades every day;
    an equity does not), with their trailing return correlation and % changes."""
    o = pd.Series(other, dtype="float64").dropna().sort_index()
    b = pd.Series(btc, dtype="float64").dropna().sort_index()
    df = pd.DataFrame({"other": o, "btc": b.reindex(o.index)}).dropna()
    full = df.copy()
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if df.empty:
        return df, {}
    rets = np.log(full).diff().dropna()
    rets = rets[rets.index > rets.index[-1] - pd.Timedelta(days=corr_days)]
    end = df.index[-1]
    ref = full[full.index <= end - pd.Timedelta(days=chg_days)]
    stats = {"date": end, "btc": float(df["btc"].iloc[-1]), "other": float(df["other"].iloc[-1]),
             "corr": float(rets["btc"].corr(rets["other"])) if len(rets) > 5 else float("nan"),
             "corr_days": corr_days, "chg_days": chg_days}
    if len(ref):
        stats["btc_chg"] = float(df["btc"].iloc[-1] / ref["btc"].iloc[-1] - 1) * 100
        stats["other_chg"] = float(df["other"].iloc[-1] / ref["other"].iloc[-1] - 1) * 100
    return df, stats
