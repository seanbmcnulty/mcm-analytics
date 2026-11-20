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
    s = s[s.index < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()]   # today's candle is unfinished
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


# ================================================================ intraday
# Two small committed stores, topped up live by the page (TTL cache):
#   vl_intraday_5m.csv  5-minute bars, US cash session (obs 09:35-16:00 NY).
#                       VIX from Yahoo ^VIX 5m (Yahoo keeps ~60 days), DVOL
#                       from Deribit 1-minute candles (kept ~3-4 months).
#                       The file keeps growing past Yahoo's window (capped 2y).
#   vl_intraday_1h.csv  hourly, obs 10:00-16:00 NY (on the hour; Yahoo ^VIX
#                       1h bars and Deribit hourly DVOL both start on the hour).
# ``obs`` is the bar END time in UTC (a value observed at that instant), so
# a 5m bar stamped 10:00 by Yahoo is the 10:05 observation.
INTRADAY = {
    "5m": {"file": "vl_intraday_5m.csv", "label": "Intraday 5-min (Yahoo ^VIX + Deribit 1-min DVOL)",
           "freq": pd.Timedelta(minutes=5), "first": (9, 35), "last": (16, 0)},
    "1h": {"file": "vl_intraday_1h.csv", "label": "Intraday hourly (Yahoo ^VIX + Deribit hourly DVOL)",
           "freq": pd.Timedelta(hours=1), "first": (10, 0), "last": (16, 0)},
}
INTRADAY_COLS = ["VIX", "DVOL_BTC", "DVOL_ETH"]
INTRADAY_MAX_DAYS = 730


def cash_session_mask(obs_utc: pd.DatetimeIndex, first=(9, 35), last=(16, 0)) -> np.ndarray:
    """True for observations inside the US cash session (NY wall clock), Mon-Fri."""
    ny = obs_utc.tz_convert("America/New_York")
    mins = ny.hour * 60 + ny.minute
    return ((mins >= first[0] * 60 + first[1]) & (mins <= last[0] * 60 + last[1])
            & (ny.dayofweek < 5) & (ny.second == 0))


def parse_yahoo_intraday(payload: dict, interval: pd.Timedelta, now=None) -> pd.Series:
    """Yahoo chart JSON -> close at bar END (UTC index). Unfinished bars dropped."""
    res = ((payload.get("chart") or {}).get("result") or [None])[0]
    if not res:
        raise ValueError("Yahoo intraday: empty result")
    ts = res.get("timestamp") or []
    closes = ((res.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    start = pd.to_datetime(ts, unit="s", utc=True)
    obs = start + interval
    s = pd.Series(pd.to_numeric(pd.Series(closes), errors="coerce").values, index=obs).dropna()
    s = s[(s.index.second == 0)]
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    s = s[s.index <= now]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.index.name = "obs"
    return s


def fetch_yahoo_intraday(sym: str, interval: str, rng: str, get=None, timeout: int = 12) -> pd.Series:
    get = get or requests.get
    r = get(YAHOO_URL.format(sym=sym), params={"range": rng, "interval": interval},
            headers={"User-Agent": CBOE_HEADERS["User-Agent"]}, timeout=timeout)
    r.raise_for_status()
    iv = pd.Timedelta(minutes=5) if interval == "5m" else pd.Timedelta(hours=1)
    return parse_yahoo_intraday(r.json(), iv)


def _deribit_candles(currency: str, resolution: str, start_ms: int, end_ms: int, get=None,
                     max_pages: int = 40, timeout: int = 12) -> pd.DataFrame:
    get = get or requests.get
    rows: list = []
    for _ in range(max_pages):
        r = get(DERIBIT_URL, params={"currency": currency, "resolution": resolution,
                                     "start_timestamp": start_ms, "end_timestamp": end_ms}, timeout=timeout)
        r.raise_for_status()
        res = r.json().get("result") or {}
        data = res.get("data") or []
        rows.extend(data)
        cont = res.get("continuation")
        if not data or not cont or cont <= start_ms or cont >= end_ms:
            break
        end_ms = int(cont)
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).drop_duplicates("ts")


def dvol_at_obs(candles: pd.DataFrame, bar: pd.Timedelta, obs: pd.DatetimeIndex) -> pd.Series:
    """DVOL observed exactly at each ``obs`` = close of the candle ENDING there.

    Candle ``ts`` covers [ts, ts+bar); its close is the value at ts+bar.
    No interpolation and no later value: a missing candle stays missing.
    """
    if candles is None or candles.empty:
        return pd.Series(np.nan, index=obs)
    end = pd.to_datetime(candles["ts"], unit="ms", utc=True) + bar
    s = pd.Series(candles["close"].astype(float).values, index=end)
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(obs)


def _sessions_ny(obs: pd.DatetimeIndex) -> pd.Index:
    return obs.tz_convert("America/New_York").normalize().tz_localize(None)


def build_intraday(kind: str, existing: pd.DataFrame | None = None, get=None, now=None,
                   max_days: int = INTRADAY_MAX_DAYS, pause: float = 0.05) -> pd.DataFrame:
    """Fetch the intraday store ``kind`` ("5m" or "1h"), merging into ``existing``.

    Only sessions not already complete in ``existing`` are fetched from
    Deribit (one 1-minute window per session for 5m). The latest stored
    session is always re-fetched so an unfinished day fills in.
    """
    spec = INTRADAY[kind]
    freq = spec["freq"]
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    old = existing.copy() if existing is not None and not existing.empty else pd.DataFrame(columns=INTRADAY_COLS)
    if not old.empty:
        old.index = pd.DatetimeIndex(old.index).tz_convert("UTC") if pd.DatetimeIndex(old.index).tz else \
            pd.DatetimeIndex(old.index).tz_localize("UTC")
    if kind == "5m":
        rng = "60d" if old.empty or (now - old.index.max()) > pd.Timedelta(days=4) else "5d"
        vix = fetch_yahoo_intraday("^VIX", "5m", rng, get)
    else:
        rng = "730d" if old.empty or (now - old.index.max()) > pd.Timedelta(days=25) else "1mo"
        vix = fetch_yahoo_intraday("^VIX", "1h", rng, get)
    vix = vix[cash_session_mask(vix.index, spec["first"], spec["last"])]
    if vix.empty:
        raise RuntimeError(f"Yahoo ^VIX {kind}: no cash-session bars")
    sessions = _sessions_ny(vix.index)
    latest_old = _sessions_ny(old.index).max() if not old.empty else None
    need = sorted({d for d in sessions if latest_old is None or d >= latest_old})
    new = pd.DataFrame({"VIX": vix})
    new = new[_sessions_ny(new.index).isin(need)]
    for cur in ("BTC", "ETH"):
        vals = []
        if kind == "5m":
            for d in need:
                o = new.index[_sessions_ny(new.index) == d]
                if len(o) == 0:
                    continue
                s_ms = int((o.min() - pd.Timedelta(minutes=2)).timestamp() * 1000)
                e_ms = int(o.max().timestamp() * 1000)
                c = _deribit_candles(cur, "60", s_ms, e_ms, get)
                vals.append(dvol_at_obs(c, pd.Timedelta(minutes=1), o))
                time.sleep(pause)
        else:
            if len(new):
                s_ms = int((new.index.min() - pd.Timedelta(hours=2)).timestamp() * 1000)
                e_ms = int(new.index.max().timestamp() * 1000)
                c = _deribit_candles(cur, "3600", s_ms, e_ms, get)
                vals.append(dvol_at_obs(c, pd.Timedelta(hours=1), new.index))
        new[f"DVOL_{cur}"] = pd.concat(vals).reindex(new.index) if vals else np.nan
    keep_old = old[~_sessions_ny(old.index).isin(need)] if not old.empty else old
    out = pd.concat([keep_old, new]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out[out.index >= now - pd.Timedelta(days=max_days)]
    out.index.name = "obs"
    return out[INTRADAY_COLS].astype(float)


def intraday_path(kind: str, data_dir: Path | None = None) -> Path:
    return (data_dir or DATA_DIR) / INTRADAY[kind]["file"]


def load_intraday(kind: str, data_dir: Path | None = None) -> pd.DataFrame:
    p = intraday_path(kind, data_dir)
    if not p.exists():
        return pd.DataFrame(columns=INTRADAY_COLS)
    df = pd.read_csv(p)
    idx = pd.to_datetime(df["obs"], utc=True)
    out = df.drop(columns="obs").set_index(idx).astype(float)
    out.index.name = "obs"
    return out


def write_intraday(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    out.index.name = "obs"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, float_format="%.3f")


def refresh_intraday(kind: str, *, write: bool = True, data_dir: Path | None = None, get=None,
                     now=None) -> Result:
    label = INTRADAY[kind]["label"]
    old = load_intraday(kind, data_dir)
    try:
        df = build_intraday(kind, old, get=get, now=now)
    except Exception as ex:
        msg = f"{label}: live top-up failed ({ex}); using the stored file ({len(old)} rows)."
        return Result(old, not old.empty, "csv" if not old.empty else "none", msg,
                      old.index.max() if not old.empty else None)
    msg = (f"{label}: live, {len(df)} rows, {_sessions_ny(df.index).nunique()} sessions to "
           f"{df.index.max().tz_convert('Asia/Singapore'):%d %b %H:%M} SGT")
    if write:
        try:
            write_intraday(df, intraday_path(kind, data_dir))
        except Exception as ex:
            msg += f"; in-session only ({ex})"
    return Result(df, True, "live", msg, df.index.max())
