"""
Bybit public REST client — option block trades (V5 API).

No auth needed.  Used by ``pages/05_Block_Trades_-_Bybit.py`` via
``lib.block_flow.render_page(bybit.VENUE)``.

Public endpoints used (all GET ``https://api.bybit.com/v5/...``, no auth):
  market/recent-trade?category=option&baseCoin=X&limit=1000
        option executions with an ``isBlockTrade`` flag, plus ``mP`` (mark price), ``iP`` (index price),
        ``mIv`` (mark IV) and ``iv`` (trade IV), all decimals.  ``side`` is the taker side.
  market/tickers?category=linear&symbol=XUSDT           live index price (``indexPrice``)
  market/index-price-kline?category=linear&symbol=XUSDT index OHLC for the chart overlay

Checked and NOT used: ``/v5/rfq/public-trades`` (Bybit's RFQ block feed) answers "apiKey is missing" —
it is not public.

Limitations — read these before trusting a quiet page:
* ``recent-trade`` has NO time or cursor parameter: it only ever returns the latest 1,000 option trades
  per coin.  That is ~35 min of BTC and ~1 h of ETH but days of the thinner coins (checked 2026-10-21), so
  a block older than that horizon is simply not obtainable.  To widen the window, every block seen is
  kept in a process-wide store (``_STORE``) and merged on each fetch, so the page's history grows for as
  long as the Streamlit server process stays up (it is lost on redeploy / sleep).
* Bybit geo-blocks some regions at its CloudFront edge (US IPs get 403 "configured to block access from your
  country").  Streamlit Community Cloud runs in the US, so the deployed app may be blocked even though the
  page works locally.  Set ``BYBIT_API_BASE`` (env var, or ``[bybit] api_base`` in secrets) to a reachable
  proxy / mirror host to override.
* Linear (USDT-settled) options: ``price`` is USDT per 1 underlying and ``size`` is in underlying units, so
  premium = price * size, treated as USD.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import threading
import time
from datetime import timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

DEFAULT_API_BASE = "https://api.bybit.com"
SGT = timezone(timedelta(hours=8))

ASSETS = ["BTC", "ETH", "SOL"]
# Default minimum size = Bybit's minimum order quantity per coin (instruments-info lotSizeFilter.minOrderQty,
# checked 2026-10-21); blocks are already flagged by the exchange, so this shows every one.
DEFAULT_MIN_SIZES: Dict[str, float] = {"BTC": 0.01, "ETH": 0.1, "SOL": 1.0, "XRP": 10.0, "DOGE": 100.0,
                                       "MNT": 10.0, "HYPE": 1.0}
# Bybit has no endpoint listing option underlyings; probe these (those without options return nothing).
CANDIDATE_BASES = ["BTC", "ETH", "SOL", "XRP", "DOGE", "MNT", "HYPE"]
_KEEP_MS = 14 * 86400_000          # how long the in-memory block store keeps rows

_session = requests.Session()
_lock = threading.Lock()
_STORE: Dict[str, Dict[str, Dict]] = {}     # base -> execId -> raw block row
_COVERAGE: Dict[str, Dict] = {}             # base -> {"oldest_ms", "n"} of the latest recent-trade fetch


def _api_base() -> str:
    base = os.environ.get("BYBIT_API_BASE", "")
    if not base:
        try:
            import streamlit as st
            base = st.secrets.get("bybit", {}).get("api_base", "")
        except Exception:
            base = ""
    return (base or DEFAULT_API_BASE).rstrip("/")


def _get(path: str, params: Dict, timeout: int = 30, retries: int = 3) -> Dict:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = _session.get(f"{_api_base()}{path}", params=params, timeout=timeout)
            if r.status_code == 403 or "block access from your country" in r.text[:300]:
                raise PermissionError(
                    "Bybit blocks requests from this server's region (HTTP 403 geo-block). Set the BYBIT_API_BASE "
                    "env var or [bybit] api_base in secrets to a reachable proxy host.")
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            j = r.json()
            if j.get("retCode", 0) != 0:
                raise RuntimeError(f"{j.get('retCode')}: {j.get('retMsg')}")
            return j.get("result", {})
        except (PermissionError, RuntimeError):
            raise
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"Bybit {path} failed: {last}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_OPT_RE = re.compile(r"^(?P<base>[A-Z0-9]+)-(?P<exp>\d{1,2}[A-Z]{3}\d{2})-(?P<strike>[0-9.]+)-(?P<cp>[CP])-(?P<q>[A-Z]+)$")


def parse_option(symbol: str):
    """'BTC-22OCT26-82500-C-USDT' -> (base, expiry Timestamp, strike, 'C'/'P')."""
    m = _OPT_RE.match(symbol or "")
    if not m:
        return None, pd.NaT, np.nan, None
    return (m["base"], pd.to_datetime(m["exp"], format="%d%b%y", errors="coerce"),
            float(m["strike"]), m["cp"])


def _f(x) -> float:
    try:
        return float(x) if x not in (None, "") else np.nan
    except (TypeError, ValueError):
        return np.nan


def _normalize(rows: List[Dict], asset: str) -> pd.DataFrame:
    recs = []
    for r in rows:
        base, exp, strike, cp = parse_option(r.get("symbol"))
        if base != asset or cp is None or pd.isna(exp):
            continue
        iv = _f(r.get("iv"))
        recs.append((r["execId"], int(r["time"]), r["symbol"], strike, exp, cp,
                     str(r.get("side", "")).lower(), _f(r.get("size")), _f(r.get("price")),
                     _f(r.get("mP")), _f(r.get("iP")), iv * 100.0 if iv > 0 else np.nan))
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs, columns=["trade_id", "ts_ms", "instrument_name", "strike", "expiry", "option_type",
                                     "direction", "abs_amount", "price", "mark_price", "index_price", "iv"])
    df = df[df["direction"].isin(["buy", "sell"])].sort_values("ts_ms").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(SGT)
    df["amount"] = np.where(df["direction"] == "buy", df["abs_amount"], -df["abs_amount"])
    df["is_block"] = True
    # Multi-leg blocks print in the same millisecond; group those legs into one package.
    df["block_id"] = df["ts_ms"].astype(str)
    df["wallet"] = None
    return df.drop(columns="ts_ms")


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------
def _recent_trades(base: str, limit: int = 1000) -> List[Dict]:
    return _get("/v5/market/recent-trade",
                {"category": "option", "baseCoin": base, "limit": limit}).get("list", [])


def _store_blocks(base: str, rows: List[Dict], now_ms: int) -> None:
    with _lock:
        d = _STORE.setdefault(base, {})
        for r in rows:
            if r.get("isBlockTrade"):
                d[r["execId"]] = r
        cut = now_ms - _KEEP_MS
        for k in [k for k, v in d.items() if int(v["time"]) < cut]:
            del d[k]
        if rows:
            _COVERAGE[base] = {"oldest_ms": min(int(r["time"]) for r in rows), "n": len(rows)}


def listed_bases() -> List[str]:
    """Underlyings (from ``CANDIDATE_BASES``) that currently return option trades."""
    def probe(b):
        try:
            return b if _recent_trades(b, 1) else None
        except Exception:
            return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(CANDIDATE_BASES)) as ex:
        return sorted(b for b in ex.map(probe, CANDIDATE_BASES) if b)


def fetch_all(start_ms: int, end_ms: int, assets: List[str]) -> Dict:
    """-> {'frames': {base: df (block rows only)}, 'meta': {...}}.  Raises if every underlying fails
    (e.g. geo-block) so the page shows the reason instead of an empty dashboard."""
    listed = sorted(set(listed_bases()) | set(assets))
    now_ms = int(time.time() * 1000)
    errors: Dict[str, str] = {}

    def one(base):
        try:
            _store_blocks(base, _recent_trades(base), now_ms)
        except Exception as e:  # noqa: BLE001
            errors[base] = str(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(listed), 8)) as ex:
        list(ex.map(one, listed))
    if len(errors) == len(listed):
        raise RuntimeError(next(iter(errors.values())))

    frames: Dict[str, pd.DataFrame] = {}
    for a in listed:
        with _lock:
            rows = [r for r in _STORE.get(a, {}).values() if start_ms <= int(r["time"]) <= end_ms]
        frames[a] = _normalize(rows, a)
    cov = {a: dict(_COVERAGE[a]) for a in listed if a in _COVERAGE}
    return {"frames": frames,
            "meta": {"listed": listed, "coverage": cov, "start_ms": start_ms, "end_ms": end_ms,
                     "errors": errors, "core": list(assets)}}


def fetch_spot(asset: str) -> float:
    try:
        lst = _get("/v5/market/tickers", {"category": "linear", "symbol": f"{asset}USDT"}).get("list", [])
        return float(lst[0]["indexPrice"]) if lst else 0.0
    except Exception:
        return 0.0


def fetch_hist_spot(asset: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    span_h = (end_ms - start_ms) / 3.6e6
    interval = 5 if span_h <= 48 else 30       # minutes; 1000 bars max per call
    try:
        lst = _get("/v5/market/index-price-kline",
                   {"category": "linear", "symbol": f"{asset}USDT", "interval": interval,
                    "start": int(start_ms), "end": int(end_ms), "limit": 1000}).get("list", [])
    except Exception:
        return pd.DataFrame()
    if not lst:
        return pd.DataFrame()
    d = pd.DataFrame([r[:5] for r in lst], columns=["ts_ms", "open", "high", "low", "close"])
    d["ts_ms"] = d["ts_ms"].astype("int64")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.drop_duplicates("ts_ms").sort_values("ts_ms")
    ts = pd.to_datetime(d["ts_ms"], unit="ms", utc=True).dt.tz_convert(SGT)
    return pd.DataFrame({"timestamp": ts.values, "close": d["close"].values}).assign(
        timestamp=lambda x: pd.to_datetime(x["timestamp"], utc=True).dt.tz_convert(SGT)).reset_index(drop=True)


def feed_status(meta: Dict, now_ms: Optional[int] = None) -> Optional[str]:
    """Explain the 1,000-trade horizon whenever it does not reach back to the window start."""
    start_ms, end_ms = meta.get("start_ms"), meta.get("end_ms")
    cov = meta.get("coverage", {})
    if not start_ms or not cov:
        return None
    short = {a: c for a, c in cov.items() if c["oldest_ms"] > start_ms}
    if not short:
        return None
    want_h = (end_ms - start_ms) / 3.6e6
    txt = ", ".join(f"{a} {(end_ms - c['oldest_ms']) / 3.6e6:,.1f}h" for a, c in
                    sorted(short.items(), key=lambda kv: (kv[0] not in meta.get("core", []), kv[0])))
    return (f"Bybit's public feed only returns the latest 1,000 option trades per coin (no time parameter), so for a "
            f"{want_h:,.0f}h window the real lookback is shorter: {txt}. Older blocks can't be fetched; blocks seen "
            f"while this app has been running are kept in memory, so history builds up over time.")


class _Venue:
    key = "bybit"
    title = "Bybit"
    icon = "🟠"
    assets = ASSETS
    default_min_sizes = DEFAULT_MIN_SIZES
    has_mark = True
    has_iv = True
    perp_label = "Index"
    definition = ("**Block definition:** option executions with `isBlockTrade=true` in "
                  "`GET /v5/market/recent-trade?category=option` (USDT-settled linear options). `side` is the "
                  "taker side; multi-leg blocks printing in the same millisecond are grouped into one package. Default "
                  "minimum size = Bybit's minimum order quantity per coin. "
                  "Mark price, index and trade IV come straight from the feed. DVOL overlay is Deribit's.")
    listed = staticmethod(listed_bases)
    fetch_all = staticmethod(fetch_all)
    fetch_spot = staticmethod(fetch_spot)
    fetch_hist_spot = staticmethod(fetch_hist_spot)
    feed_status = staticmethod(feed_status)


VENUE = _Venue()
