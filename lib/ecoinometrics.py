"""
Ecoinometrics-style charts — pure data/analytics helpers (no Streamlit).

DELIBERATE EXCEPTIONS to "Deribit-only" (requested by the user 2026-11-01):
  * US spot BTC ETF daily flows  -> Farside (scraped HTML) or data/etf_flows.csv
  * Fed press-conference tone    -> data/fed_tone.csv built by tools/build_fed_tone.py
    from federalreserve.gov transcripts (dictionary score, NOT Ecoinometrics' index)
BTC prices for the hack study come from Deribit (lib/deribit.py) in the page.

Charts replicated (Ecoinometrics, 2026-10-30 issue):
  1. ETF flow streaks: cumulative flow over consecutive same-sign trading days,
     plus rolling 30-trading-day net flow.
  2. BTC N-day return after major exchange hacks (data/exchange_hacks.csv).
  3. Fed hawk/dove tone per FOMC press conference + 6-meeting average.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FARSIDE_URLS = ("https://farside.co.uk/btc/", "https://farside.co.uk/bitcoin-etf-flow-all-data/")
UA = {"User-Agent": "Mozilla/5.0 (compatible; mcm-analytics)"}

# ---------------------------------------------------------------- ETF flows
def _num(x) -> float:
    s = str(x).strip().replace(",", "")
    if s in ("", "-", "nan", "NaN"):
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -v if neg else v


def parse_farside_html(html: str) -> pd.DataFrame:
    """Farside BTC table -> DataFrame[date, flow] (US$ m, total of all ETFs)."""
    for t in pd.read_html(io.StringIO(html)):
        cols = [str(c).strip().lower() for c in t.columns.get_level_values(-1)]
        t.columns = cols
        if "date" in cols and "total" in cols:
            d = pd.to_datetime(t["date"], format="%d %b %Y", errors="coerce")
            out = pd.DataFrame({"date": d, "flow": t["total"].map(_num)})
            return out.dropna().sort_values("date").reset_index(drop=True)
    raise ValueError("no Farside flow table found")


def fetch_etf_flows(timeout: int = 20) -> pd.DataFrame:
    """Try Farside; fall back to data/etf_flows.csv (date,flow in US$ m)."""
    err = None
    for url in FARSIDE_URLS:
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return parse_farside_html(r.text)
        except Exception as e:  # blocked / format change
            err = e
    p = DATA_DIR / "etf_flows.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"])
        return df[["date", "flow"]].sort_values("date").reset_index(drop=True)
    raise RuntimeError(f"Farside unavailable ({err}) and no data/etf_flows.csv")


def flow_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive same-sign trading days. Zero-flow days are dropped."""
    d = df[df["flow"] != 0].reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(columns=["start", "end", "days", "total"])
    sign = np.sign(d["flow"])
    grp = (sign != sign.shift()).cumsum()
    g = d.groupby(grp)
    return pd.DataFrame({
        "start": g["date"].first(), "end": g["date"].last(),
        "days": g["flow"].size(), "total": g["flow"].sum(),
    }).reset_index(drop=True)


def rolling_flow(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """Rolling net flow over `window` trading days."""
    return df.set_index("date")["flow"].rolling(window).sum()


# ----------------------------------------------------------- external helpers
def fred_series(series_id: str, timeout: int = 15, start: str = "2013-01-01", retries: int = 2,
                live_only: bool = False) -> pd.DataFrame:
    """FRED public CSV (no key). Returns DataFrame[date, value].
    Falls back to data/fred_<ID>.csv (kept fresh by tools/build_fred_cache.py + GH Action)
    because Streamlit Cloud frequently times out reaching FRED."""
    err = None
    for _ in range(retries):
        try:
            r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                             params={"id": series_id, "cosd": start}, headers=UA, timeout=timeout)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            err = e
    else:
        p = DATA_DIR / f"fred_{series_id}.csv"
        if live_only or not p.exists():
            raise err
        df = pd.read_csv(p, parse_dates=["date"])
        return df.dropna().reset_index(drop=True)
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def fetch_bitstamp_daily(start: str, end: str, timeout: int = 20) -> pd.DataFrame:
    """Daily BTC/USD closes from Bitstamp (history to 2011) -> DataFrame[timestamp, close].
    Used only for hack events before Deribit's perp history (Aug 2018)."""
    out, cur, end_ts = [], int(pd.Timestamp(start, tz="UTC").timestamp()), int(pd.Timestamp(end, tz="UTC").timestamp())
    while cur < end_ts:
        r = requests.get("https://www.bitstamp.net/api/v2/ohlc/btcusd/", headers=UA, timeout=timeout,
                         params={"step": 86400, "limit": 1000, "start": cur})
        r.raise_for_status()
        rows = r.json()["data"]["ohlc"]
        if not rows:
            break
        out += rows
        nxt = int(rows[-1]["timestamp"]) + 86400
        if nxt <= cur:
            break
        cur = nxt
    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df["close"] = df["close"].astype(float)
    return df[["timestamp", "close"]].drop_duplicates("timestamp")


def flows_to_btc(flows: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """US$ m flows -> BTC using that day's close (px: DataFrame[timestamp, close])."""
    p = px.assign(date=pd.to_datetime(px["timestamp"]).dt.normalize()).drop_duplicates("date").set_index("date")["close"]
    d = flows.copy()
    d["close"] = d["date"].dt.normalize().map(p)
    d = d.dropna(subset=["close"])
    d["flow"] = d["flow"] * 1e6 / d["close"]
    return d[["date", "flow"]].reset_index(drop=True)


def streak_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day cumulative flow within each streak: DataFrame[streak, date, cum, sign]."""
    d = df[df["flow"] != 0].reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(columns=["streak", "date", "cum", "sign"])
    sign = np.sign(d["flow"])
    d["streak"] = (sign != sign.shift()).cumsum()
    d["cum"] = d.groupby("streak")["flow"].cumsum()
    d["sign"] = sign
    return d[["streak", "date", "cum", "sign"]]


def sma_gap(px: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    d = px.assign(date=pd.to_datetime(px["timestamp"]).dt.normalize()).set_index("date")[["close"]]
    d["sma"] = d["close"].rolling(window).mean()
    d["gap"] = d["close"] / d["sma"] - 1
    return d.reset_index()


def rolling_corr(btc: pd.DataFrame, other: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """Rolling correlation of daily log returns, on `other`'s trading days. other: [date, value]."""
    b = btc.assign(date=pd.to_datetime(btc["timestamp"]).dt.normalize()).set_index("date")["close"]
    o = other.set_index("date")["value"]
    j = pd.concat([b.rename("btc"), o.rename("oth")], axis=1, join="inner").dropna()
    r = np.log(j).diff().dropna()
    c = r["btc"].rolling(window).corr(r["oth"]).dropna()
    return c.rename("corr").reset_index()

# ------------------------------------------------------------- Exchange hacks
def load_hacks() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "exchange_hacks.csv", parse_dates=["date"])


def hack_event_returns(prices: pd.DataFrame, hacks: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Return from close of hack day to close `horizon` days later.
    prices: DataFrame[timestamp, close] daily. Events outside price history dropped."""
    px = prices.assign(d=pd.to_datetime(prices["timestamp"]).dt.normalize()).set_index("d")["close"]
    rows = []
    for _, h in hacks.iterrows():
        d0, d1 = h["date"].normalize(), h["date"].normalize() + pd.Timedelta(days=horizon)
        if d0 in px.index and d1 in px.index:
            rows.append({**h.to_dict(), "ret": px[d1] / px[d0] - 1})
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ Fed tone
HAWK = [r"inflation", r"restrictive", r"tighten", r"\bhike", r"raise (the )?(rate|policy)",
        r"upside risk", r"elevated", r"persistent", r"overheat", r"strong(er)? (growth|labor|demand)",
        r"further (firming|increase)", r"above (our )?(2 percent|target)"]
DOVE = [r"\bcut", r"easing", r"accommodat", r"downside risk", r"weak(er|ening)? (labor|growth|demand)",
        r"softening", r"slowing", r"lower (the )?(rate|policy)", r"disinflation", r"unemployment (has )?(risen|rising)",
        r"restrain(t|ing) (growth|activity)"]
_H = [re.compile(p, re.I) for p in HAWK]
_D = [re.compile(p, re.I) for p in DOVE]


def chair_text(full: str) -> str:
    """Keep only the Chair's speaking turns (heuristic on 'CHAIR NAME.' labels)."""
    parts = re.split(r"\n\s*(?=[A-Z][A-Za-z.' -]{2,40}\.\s)", full)
    keep = [p for p in parts if re.match(r"\s*(CHAIR|Chair)\b", p)]
    return "\n".join(keep) if keep else full


def tone_score(text: str) -> dict:
    h = sum(len(p.findall(text)) for p in _H)
    d = sum(len(p.findall(text)) for p in _D)
    n = len(text.split())
    return {"hawk": h, "dove": d, "words": n,
            "score": (h - d) / (h + d) if h + d else 0.0,
            "net_per_1k": 1000 * (h - d) / n if n else 0.0}


def load_fed_tone() -> pd.DataFrame:
    p = DATA_DIR / "fed_tone.csv"
    if not p.exists():
        return pd.DataFrame(columns=["date", "score", "idx", "avg6"])
    df = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
    df["idx"] = df["score"] * 100
    df["avg6"] = df["idx"].rolling(6, min_periods=1).mean()
    return df.reset_index(drop=True)


def add_fed_funds(ft: pd.DataFrame, dff: pd.DataFrame) -> pd.DataFrame:
    """Attach the effective fed funds rate (FRED DFF) on/before each meeting date."""
    m = pd.merge_asof(ft.sort_values("date"), dff.rename(columns={"value": "ffr"}).sort_values("date"), on="date")
    return m
