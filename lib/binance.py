"""
Binance Options (eapi) public REST client - block-trade tape.

No auth needed.  Used by ``pages/02_Block_Trades_-_Binance.py`` via
``lib.block_flow.render_page(binance.VENUE)``.

Public endpoints used (no auth):
  GET https://eapi.binance.com/eapi/v1/blockTrades?limit=500
        the exchange-wide block-trade tape (``symbol`` is optional; omitting it returns every market).
        Newest-first, max 500 rows, NO time / cursor parameters.  Fields: id, tradeId, symbol, price, qty,
        quoteQty, side (-1 / 1), time (ms).  500 rows reached back ~53 days on 2026-10-21, so a window up to
        a week is covered in full; ``meta['truncated']`` flags the case where it is not.
  GET https://eapi.binance.com/eapi/v1/exchangeInfo   listed underlyings (``optionContracts``) + lot sizes
  GET https://eapi.binance.com/eapi/v1/index?underlying=BTCUSDT   live index price
  GET https://fapi.binance.com/fapi/v1/indexPriceKlines?pair=BTCUSDT&interval=5m   index OHLC for the chart
        overlay and the index at trade time (futures host; weight 1)

Units: USDT-settled linear European options, ``price`` is USDT per 1 underlying and ``qty`` is in underlying
units (min lot 0.01 for every underlying), so premium = price * qty.

Limitations / assumptions:
* ``side`` is undocumented for this endpoint.  It is treated as the TAKER side (-1 sell / 1 buy), matching the
  documented meaning on Binance's regular ``/eapi/v1/trades`` ("completed trade direction"), so a "buy" is the
  taker lifting an offer.  Verify against Binance if the sign of net delta/vega matters to you.
* The tape has no mark, IV or index, and Binance's ``/mark`` is live only, so the mark-based Edge /
  Aggression views are not available and IV is backed out of price (Black-Scholes, r=0) with the index at the trade.
* Multi-leg blocks print in the same millisecond; those legs are grouped into one package.
* Binance blocks restricted regions (US IPs get HTTP 451 / "restricted location"), which may include Streamlit
  Community Cloud.  Override the hosts with ``BINANCE_EAPI_BASE`` / ``BINANCE_FAPI_BASE`` (env vars or
  ``[binance] eapi_base`` / ``fapi_base`` in secrets).
"""

from __future__ import annotations

import os
import re
import time
from datetime import timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

DEFAULT_EAPI = "https://eapi.binance.com/eapi/v1"
DEFAULT_FAPI = "https://fapi.binance.com/fapi/v1"
SGT = timezone(timedelta(hours=8))

ASSETS = ["BTC", "ETH", "SOL", "BNB"]
# Default minimum size = Binance's minimum lot (LOT_SIZE.minQty = 0.01 on every option symbol, 2026-10-21):
# the tape is already blocks only, so this shows every one.
DEFAULT_MIN_SIZES: Dict[str, float] = {b: 0.01 for b in
                                       ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "XAU", "XAG", "CL", "BZ")}

_session = requests.Session()


def _base(kind: str) -> str:
    env, sec, default = (("BINANCE_EAPI_BASE", "eapi_base", DEFAULT_EAPI) if kind == "eapi"
                         else ("BINANCE_FAPI_BASE", "fapi_base", DEFAULT_FAPI))
    base = os.environ.get(env, "")
    if not base:
        try:
            import streamlit as st
            base = st.secrets.get("binance", {}).get(sec, "")
        except Exception:
            base = ""
    return (base or default).rstrip("/")


def _get(kind: str, path: str, params: Dict, timeout: int = 30, retries: int = 3):
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = _session.get(f"{_base(kind)}{path}", params=params, timeout=timeout)
            if r.status_code in (403, 451) or "restricted location" in r.text[:300]:
                raise PermissionError(
                    "Binance blocks requests from this server's region (restricted location). Set BINANCE_EAPI_BASE / "
                    "BINANCE_FAPI_BASE (env vars or [binance] eapi_base / fapi_base in secrets) to a reachable proxy host.")
            if r.status_code in (418, 429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            return r.json()
        except (PermissionError, RuntimeError):
            raise
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"Binance {path} failed: {last}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_OPT_RE = re.compile(r"^(?P<base>[A-Z0-9]+)-(?P<exp>\d{6})-(?P<strike>[0-9.]+)-(?P<cp>[CP])$")


def parse_option(symbol: str):
    """'BTC-261225-85000-C' -> (base, expiry Timestamp, strike, 'C'/'P')."""
    m = _OPT_RE.match(symbol or "")
    if not m:
        return None, pd.NaT, np.nan, None
    return (m["base"], pd.to_datetime(m["exp"], format="%y%m%d", errors="coerce"),
            float(m["strike"]), m["cp"])


def _normalize(rows: List[Dict], asset: str, candles: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for r in rows:
        base, exp, strike, cp = parse_option(r.get("symbol"))
        if base != asset or cp is None or pd.isna(exp) or r.get("side") not in (1, -1):
            continue
        recs.append((str(r["id"]), int(r["time"]), r["symbol"], strike, exp, cp,
                     "buy" if r["side"] == 1 else "sell", abs(float(r["qty"])), float(r["price"])))
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs, columns=["trade_id", "ts_ms", "instrument_name", "strike", "expiry", "option_type",
                                     "direction", "abs_amount", "price"])
    df = df.drop_duplicates("trade_id").sort_values("ts_ms").reset_index(drop=True)
    if not candles.empty:
        k = candles.rename(columns={"ts_ms": "k_ms", "open": "index_price"})[["k_ms", "index_price"]]
        df = pd.merge_asof(df, k, left_on="ts_ms", right_on="k_ms", direction="backward",
                           tolerance=6 * 3_600_000).drop(columns="k_ms")
    else:
        df["index_price"] = np.nan
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(SGT)
    df["amount"] = np.where(df["direction"] == "buy", df["abs_amount"], -df["abs_amount"])
    df["mark_price"] = np.nan      # not published historically
    df["iv"] = np.nan              # backed out by lib.block_flow
    df["is_block"] = True
    df["block_id"] = df["ts_ms"].astype(str)   # legs of one block print in the same millisecond
    df["wallet"] = None
    return df.drop(columns="ts_ms")


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------
def fetch_block_tape(start_ms: int, end_ms: int) -> tuple:
    """Latest 500 block prints filtered to the window -> (rows, truncated)."""
    rows = _get("eapi", "/blockTrades", {"limit": 500})
    if not isinstance(rows, list):
        raise RuntimeError(f"Binance blockTrades: unexpected response {str(rows)[:150]}")
    truncated = len(rows) >= 500 and min(int(r["time"]) for r in rows) > start_ms
    return [r for r in rows if start_ms <= int(r["time"]) <= end_ms], truncated


def fetch_index_klines(asset: str, start_ms: int, end_ms: int, interval: str = "5m") -> pd.DataFrame:
    """Index OHLC (``{asset}USDT``) -> DataFrame[ts_ms (bar open), open, close], oldest first."""
    try:
        data = _get("fapi", "/indexPriceKlines", {"pair": f"{asset}USDT", "interval": interval,
                                                 "startTime": int(start_ms), "endTime": int(end_ms), "limit": 1500})
    except PermissionError:
        raise
    except Exception:
        return pd.DataFrame(columns=["ts_ms", "open", "close"])
    if not isinstance(data, list) or not data:
        return pd.DataFrame(columns=["ts_ms", "open", "close"])
    d = pd.DataFrame([r[:5] for r in data], columns=["ts_ms", "open", "high", "low", "close"])
    d["ts_ms"] = d["ts_ms"].astype("int64")
    for c in ("open", "close"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[["ts_ms", "open", "close"]].drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True)


def _interval(start_ms: int, end_ms: int) -> str:
    return "5m" if (end_ms - start_ms) <= 48 * 3.6e6 else "1h"      # <= 1,500 bars either way up to ~60d


def listed_bases() -> List[str]:
    """Option underlyings (exchangeInfo.optionContracts baseAsset)."""
    try:
        info = _get("eapi", "/exchangeInfo", {}, timeout=60)
        return sorted({c["baseAsset"] for c in info.get("optionContracts", [])})
    except Exception:
        return []


def fetch_all(start_ms: int, end_ms: int, assets: List[str]) -> Dict:
    """-> {'frames': {base: df}, 'meta': {...}}.  Raises when Binance is unreachable (e.g. region block)
    so the page shows the reason rather than an empty dashboard."""
    rows, truncated = fetch_block_tape(start_ms, end_ms)
    present = sorted({parse_option(r["symbol"])[0] for r in rows} - {None})
    listed = sorted(set(listed_bases()) | set(assets) | set(present))
    frames: Dict[str, pd.DataFrame] = {a: pd.DataFrame() for a in listed}
    k_start = start_ms - 3_600_000
    for a in present:
        try:
            k = fetch_index_klines(a, k_start, end_ms, _interval(start_ms, end_ms))
        except PermissionError:
            k = pd.DataFrame(columns=["ts_ms", "open", "close"])
        frames[a] = _normalize(rows, a, k)
    return {"frames": frames,
            "meta": {"listed": listed, "tape_rows": len(rows), "truncated": truncated,
                     "other_bases": sorted(b for b in present if b not in assets)}}


def fetch_spot(asset: str) -> float:
    try:
        return float(_get("eapi", "/index", {"underlying": f"{asset}USDT"}).get("indexPrice") or 0.0)
    except Exception:
        return 0.0


def fetch_hist_spot(asset: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    try:
        k = fetch_index_klines(asset, start_ms, end_ms, _interval(start_ms, end_ms))
    except PermissionError:
        return pd.DataFrame()
    if k.empty:
        return pd.DataFrame()
    ts = pd.to_datetime(k["ts_ms"], unit="ms", utc=True).dt.tz_convert(SGT)
    return pd.DataFrame({"timestamp": ts, "close": k["close"].values})


def feed_status(meta: Dict, now_ms: Optional[int] = None) -> Optional[str]:
    if meta.get("tape_rows", 0) == 0:
        return "Binance returned no block trades in this window."
    txt = f"Window contains {meta['tape_rows']:,} block prints (Binance serves only its latest 500 block trades)."
    if meta.get("truncated"):
        txt += " The window reaches back further than those 500 prints, so the oldest blocks are missing."
    return txt


class _Venue:
    key = "binance"
    title = "Binance"
    icon = "🟡"
    assets = ASSETS
    default_min_sizes = DEFAULT_MIN_SIZES
    has_mark = False
    has_iv = False
    perp_label = "Index"
    definition = ("**Block definition:** every print on Binance Options' public block-trade tape "
                  "(`GET /eapi/v1/blockTrades`, the latest 500 across all markets). USDT-settled linear options; "
                  "`side` is undocumented and treated as the taker side; legs printing in the same millisecond are "
                  "grouped into one package. Default minimum size = Binance's minimum lot (0.01). The tape has no "
                  "mark, IV or index: IV is backed out of price (Black-Scholes, r=0) using the Binance index price "
                  "at the trade, so the mark-based Edge and Aggression views are not available. DVOL overlay is "
                  "Deribit's.")
    listed = staticmethod(listed_bases)
    fetch_all = staticmethod(fetch_all)
    fetch_spot = staticmethod(fetch_spot)
    fetch_hist_spot = staticmethod(fetch_hist_spot)
    feed_status = staticmethod(feed_status)


VENUE = _Venue()
