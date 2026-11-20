"""Data for the "TradFi vol vs crypto vol" page (pages/14_VIX_vs_DVOL.py).

Every series is free and public, and none needs a token:

* BTC / ETH DVOL: Deribit ``public/get_volatility_index_data`` hourly
  candles, snapped to the New York cash close (16:00 America/New_York, the
  hour the VIX settles), so the crypto and TradFi numbers on a row were
  observed at the same moment. The 00:00 UTC daily close is kept too.
* VIX, VVIX, VIX9D, VIX3M: Cboe daily-price CSVs (cdn.cboe.com). Cboe's
  Cloudflare front wants a cboe.com Referer, like lib/ibit.fetch_raw.
* MOVE, DXY (ICE dollar index) and the Nasdaq-100 index: Yahoo Finance chart
  API (^MOVE, DX-Y.NYB, ^NDX). The NDX falls back to data/fred_NASDAQ100.csv.
* US high-yield OAS: FRED BAMLH0A0HYM2 (public CSV, no key). FRED only
  serves the last ~3 years of ICE BofA series.

Each source is a committed CSV in data/ (``vl_*.csv``), rebuilt by the page's
refresh buttons (written on that machine only, like the Ecoinometrics page)
and by ``tools/refresh_vol_lead_data.py`` / the weekly GitHub Action, which
commits them. Nothing here is synthetic: a source that fails stays missing.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "Mozilla/5.0 (compatible; mcm-analytics)"}
CBOE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/csv,*/*;q=0.8",
    "Referer": "https://www.cboe.com/",
    "Origin": "https://www.cboe.com",
}
DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

DVOL_START = "2021-03-24"          # first DVOL print
START = "2021-01-01"
CBOE_SERIES = ("VIX", "VVIX", "VIX9D", "VIX3M")
YAHOO_SERIES = {"MOVE": "^MOVE", "DXY": "DX-Y.NYB", "NDX": "^NDX"}
FRED_SERIES = {"HY_OAS": "BAMLH0A0HYM2"}

# dataset key -> (csv file, label)
DATASETS = {
    "dvol": ("vl_dvol.csv", "Deribit DVOL (BTC, ETH)"),
    "cboe": ("vl_cboe.csv", "Cboe VIX / VVIX / VIX9D / VIX3M"),
    "yahoo": ("vl_yahoo.csv", "Yahoo MOVE / DXY / NDX"),
    "fred": ("vl_fred.csv", "FRED HY OAS"),
}


@dataclass
class Result:
    df: pd.DataFrame
    ok: bool
    source: str           # "live" | "csv" | "none"
    message: str
    as_of: pd.Timestamp | None = None


def _as_of(df: pd.DataFrame) -> pd.Timestamp | None:
    if df is None or df.empty or "date" not in df:
        return None
    return pd.to_datetime(df["date"]).max()


# ---------------------------------------------------------------- Deribit
def snap_to_ny_close(hourly: pd.DataFrame, hour: int = 16) -> pd.Series:
    """Hourly DVOL (``ts`` UTC ms, ``close``) -> one value per NY calendar day.

    Takes the last close at or before ``hour``:00 New York time, so the
    value lines up with the 16:00 ET cash close (DST handled by the tz
    conversion). Days without a print before the cut are dropped.
    """
    if hourly is None or hourly.empty:
        return pd.Series(dtype=float)
    ts = pd.to_datetime(hourly["ts"], unit="ms", utc=True)
    # Candle at ts covers [ts, ts+1h): its close is observed at ts+1h.
    obs = (ts + pd.Timedelta(hours=1)).dt.tz_convert("America/New_York")
    frame = pd.DataFrame({"obs": obs, "close": hourly["close"].astype(float).values})
    frame["day"] = frame["obs"].dt.tz_localize(None).dt.normalize()
    frame["hr"] = frame["obs"].dt.hour + frame["obs"].dt.minute / 60.0
    # Only days the feed has reached the cut (an unfinished "today" is not a close).
    reached = frame.groupby("day")["hr"].max() >= hour - 1
    frame = frame[(frame["hr"] <= hour) & frame["day"].map(reached).fillna(False).astype(bool)]
    out = frame.sort_values("obs").groupby("day")["close"].last()
    out.index.name = "date"
    return out


def fetch_dvol_hourly(currency: str, start: str = DVOL_START, get=None,
                      max_pages: int = 120, pause: float = 0.12) -> pd.DataFrame:
    get = get or requests.get
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(time.time() * 1000)
    rows: list = []
    for _ in range(max_pages):
        r = get(DERIBIT_URL, params={
            "currency": currency, "resolution": "3600",
            "start_timestamp": start_ms, "end_timestamp": end_ms}, timeout=20)
        r.raise_for_status()
        res = r.json().get("result") or {}
        data = res.get("data") or []
        if not data:
            break
        rows.extend(data)
        cont = res.get("continuation")
        if not cont or cont <= start_ms or cont >= end_ms:
            break
        end_ms = int(cont)
        time.sleep(pause)
    if not rows:
        raise RuntimeError(f"Deribit returned no {currency} DVOL rows")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def fetch_dvol_daily(currency: str, start: str = DVOL_START, get=None) -> pd.Series:
    """Deribit 1D candles; close = 00:00 UTC of the next day. Index = UTC date."""
    get = get or requests.get
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(time.time() * 1000)
    rows: list = []
    for _ in range(10):
        r = get(DERIBIT_URL, params={"currency": currency, "resolution": "1D",
                                     "start_timestamp": start_ms, "end_timestamp": end_ms}, timeout=20)
        r.raise_for_status()
        res = r.json().get("result") or {}
        rows.extend(res.get("data") or [])
        cont = res.get("continuation")
        if not cont or cont <= start_ms:
            break
        end_ms = int(cont)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).drop_duplicates("ts")
    s = pd.Series(df["close"].astype(float).values,
                  index=pd.to_datetime(df["ts"], unit="ms").dt.normalize(), name=currency).sort_index()
    s.index.name = "date"
    return s


def build_dvol_frame(get=None) -> pd.DataFrame:
    cols = {}
    for cur in ("BTC", "ETH"):
        cols[f"DVOL_{cur}"] = snap_to_ny_close(fetch_dvol_hourly(cur, get=get))
        cols[f"DVOL_{cur}_UTC"] = fetch_dvol_daily(cur, get=get)
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df.reset_index()


# ---------------------------------------------------------------- Cboe
def parse_cboe_csv(text: str, sym: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]
    if "DATE" not in df.columns:
        raise ValueError(f"Cboe {sym}: no DATE column")
    col = "CLOSE" if "CLOSE" in df.columns else (sym.upper() if sym.upper() in df.columns else df.columns[-1])
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").values,
                  index=pd.to_datetime(df["DATE"], format="%m/%d/%Y", errors="coerce"), name=sym)
    s = s[s.index.notna()].dropna()
    s = s[s > 0]
    s.index.name = "date"
    return s.sort_index()


def build_cboe_frame(get=None, start: str = START) -> pd.DataFrame:
    get = get or requests.get
    cols, errors = {}, []
    for sym in CBOE_SERIES:
        try:
            r = get(CBOE_URL.format(sym=sym), headers=CBOE_HEADERS, timeout=25)
            r.raise_for_status()
            cols[sym] = parse_cboe_csv(r.text, sym)
        except Exception as ex:   # keep the others
            errors.append(f"{sym}: {ex}")
    if not cols:
        raise RuntimeError("Cboe unavailable (" + "; ".join(errors) + ")")
    df = pd.DataFrame(cols).sort_index()
    df = df[df.index >= pd.Timestamp(start)]
    df.index.name = "date"
    out = df.reset_index()
    out.attrs["errors"] = errors
    return out


# ---------------------------------------------------------------- Yahoo
def parse_yahoo_chart(payload: dict, name: str) -> pd.Series:
    res = (payload.get("chart") or {}).get("result") or []
    if not res:
        raise ValueError(f"Yahoo {name}: empty result")
    res = res[0]
    ts = res.get("timestamp") or []
    closes = ((res.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    tz = (res.get("meta") or {}).get("exchangeTimezoneName") or "America/New_York"
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    s = pd.Series(pd.to_numeric(pd.Series(closes), errors="coerce").values, index=idx, name=name)
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")]
    s.index.name = "date"
    return s.sort_index()


def build_yahoo_frame(get=None, start: str = START) -> pd.DataFrame:
    get = get or requests.get
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(time.time()) + 86400
    cols, errors = {}, []
    for name, sym in YAHOO_SERIES.items():
        try:
            r = get(YAHOO_URL.format(sym=sym), params={"period1": p1, "period2": p2, "interval": "1d"},
                    headers={"User-Agent": CBOE_HEADERS["User-Agent"]}, timeout=25)
            r.raise_for_status()
            cols[name] = parse_yahoo_chart(r.json(), name)
        except Exception as ex:
            errors.append(f"{name}: {ex}")
    if not cols:
        raise RuntimeError("Yahoo unavailable (" + "; ".join(errors) + ")")
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    out = df.reset_index()
    out.attrs["errors"] = errors
    return out


# ---------------------------------------------------------------- FRED
def build_fred_frame(get=None, start: str = START) -> pd.DataFrame:
    get = get or requests.get
    cols, errors = {}, []
    for name, sid in FRED_SERIES.items():
        try:
            r = get(FRED_URL, params={"id": sid, "cosd": start}, headers=UA, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["date", "value"]
            s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                          index=pd.to_datetime(df["date"]), name=name).dropna()
            s.index.name = "date"
            cols[name] = s
        except Exception as ex:
            errors.append(f"{name}: {ex}")
    if not cols:
        raise RuntimeError("FRED unavailable (" + "; ".join(errors) + ")")
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    out = df.reset_index()
    out.attrs["errors"] = errors
    return out


BUILDERS = {"dvol": build_dvol_frame, "cboe": build_cboe_frame,
            "yahoo": build_yahoo_frame, "fred": build_fred_frame}


# ---------------------------------------------------------------- disk
def csv_path(key: str, data_dir: Path | None = None) -> Path:
    return (data_dir or DATA_DIR) / DATASETS[key][0]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.4f")


def load_csv(key: str, data_dir: Path | None = None) -> Result:
    path = csv_path(key, data_dir)
    label = DATASETS[key][1]
    if not path.exists():
        return Result(pd.DataFrame(), False, "none", f"{label}: no {path.name} yet; press Refresh.")
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception as ex:
        return Result(pd.DataFrame(), False, "none", f"{label}: could not read {path.name} ({ex}).")
    return Result(df, not df.empty, "csv", f"{label}: {path.name}, {len(df)} rows", _as_of(df))


def refresh(key: str, *, write: bool = True, data_dir: Path | None = None, get=None) -> Result:
    """Live rebuild of one dataset. A failure leaves the existing CSV alone."""
    label = DATASETS[key][1]
    try:
        df = BUILDERS[key](get=get)
    except Exception as ex:
        return Result(pd.DataFrame(), False, "none", f"{label}: refresh failed ({ex}).")
    errs = df.attrs.get("errors") or []
    msg = f"{label}: live, {len(df)} rows to {_as_of(df):%d %b %Y}"
    if errs:
        msg += " (partial: " + "; ".join(errs) + ")"
    if write:
        try:
            write_csv(df, csv_path(key, data_dir))
            msg += f"; wrote {csv_path(key, data_dir).name} on this machine"
        except Exception as ex:
            msg += f"; in-session only ({ex})"
    return Result(df, True, "live", msg, _as_of(df))


def ndx_fallback(data_dir: Path | None = None) -> pd.Series:
    path = (data_dir or DATA_DIR) / "fred_NASDAQ100.csv"
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values, index=df["date"], name="NDX")
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float, name="NDX")


def merge_panel(frames: dict[str, pd.DataFrame], ndx_extra: pd.Series | None = None) -> pd.DataFrame:
    """Outer-join the dataset frames on date (one column per series)."""
    parts = []
    for df in frames.values():
        if df is None or df.empty:
            continue
        parts.append(df.assign(date=pd.to_datetime(df["date"])).set_index("date"))
    panel = pd.concat(parts, axis=1).sort_index() if parts else pd.DataFrame()
    panel = panel.loc[:, ~panel.columns.duplicated()]
    if ndx_extra is not None and not ndx_extra.empty:
        if "NDX" in panel:
            panel["NDX"] = panel["NDX"].combine_first(ndx_extra)
        else:
            panel = panel.join(ndx_extra.rename("NDX"), how="outer")
    return panel
