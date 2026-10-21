"""
Bullish (bullish.com) public REST client — option trade tape (BTC options).

No auth needed.  Used by ``pages/02_Block_Trades_-_Bullish.py`` via
``lib.block_flow.render_page(bullish.VENUE)``.

Public endpoints used (all GET ``https://api.exchange.bullish.com/trading-api/v1/...``, no auth):
  history/option-trades           exchange-wide option trades, newest-first, 5/25/50/100 per page, filtered
                                  with ``createdAtTimestamp[gte|lte]`` (epoch ms).  The server rejects any
                                  range wider than 3 days, so longer windows are fetched in <=2.9-day chunks,
                                  and each chunk is paged by moving ``[lte]`` back to the oldest row seen.
                                  Fields: symbol, tradeId, price, quantity, side, isTaker, createdAtTimestamp.
  markets                         option market list (used to discover the listed underlyings)
  markets/{symbol}/candle         spot candles, used as the index (``BTCUSDC``; ``createdAtDatetime[gte|lte]``
                                  ISO filters, 100 bars/page, bar timestamp = bar open)
  index-prices/{asset}            live index price

Block definition
----------------
Bullish publishes NO block / RFQ / OTC flag on its public option tape (``otcBaseVolume`` exists on tickers
but is 0 on every option market checked, and the OTC trade endpoints are private).  So — exactly like
``pages/02_Block_Trades_-_Deribit.py`` — a "block" is a trade at or above a per-underlying minimum size
(``DEFAULT_MIN_SIZES``, editable in the page sidebar) - by default the exchange minimum, so every print.
``fetch_all`` returns every option trade; the page applies the size filter.  Checked 2026-10-21 over 7 days: 1,179 trades, median 0.35 BTC, 84 trades >= 2 BTC (~12/day), 21 >= 5 BTC.

Notes
-----
* Options are USDC-linear: ``price`` is USDC per 1 BTC and ``quantity`` is in BTC, so premium = price * qty.
* ``side`` is the taker side (every row has ``isTaker=true``).
* The tape carries no mark, index or IV.  Index comes from BTCUSDC candles (bar open at/before the trade);
  IV is backed out of price by lib.block_flow (Black-Scholes, r=0), so the mark-based Edge / Aggression
  views are not available.
* Older history pages can take 10-30 s per request, so fetches use a long timeout and run chunks in parallel.
"""

from __future__ import annotations

import concurrent.futures
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

API_BASE = "https://api.exchange.bullish.com/trading-api/v1"
SGT = timezone(timedelta(hours=8))

ASSETS = ["BTC"]                      # Bullish lists options on BTC only (checked 2026-10-21)
# Default minimum size = Bullish's minimum order (minQuantityLimit 0.0005 BTC), i.e. show every print.  Raise it
# in the sidebar to isolate bigger trades (checked 2026-10-21: median print 0.35 BTC; >= 2 BTC is ~12/day).
DEFAULT_MIN_SIZES: Dict[str, float] = {"BTC": 0.0005}

_CHUNK_MS = int(2.9 * 86400_000)      # server limit is 3 days per request
_session = requests.Session()


def _get(path: str, params: Dict, timeout: int = 90, retries: int = 3):
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = _session.get(f"{API_BASE}{path}", params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            return r.json()
        except RuntimeError as e:
            last = e
            break
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Bullish {path} failed: {last}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_OPT_RE = re.compile(r"^(?P<base>[A-Z0-9]+)-(?P<q>[A-Z]+)-(?P<exp>\d{8})-(?P<strike>[0-9.]+)-(?P<cp>[CP])$")


def parse_option(symbol: str):
    """'BTC-USDC-20261021-76000-C' -> (base, expiry Timestamp, strike, 'C'/'P')."""
    m = _OPT_RE.match(symbol or "")
    if not m:
        return None, pd.NaT, np.nan, None
    return (m["base"], pd.to_datetime(m["exp"], format="%Y%m%d", errors="coerce"),
            float(m["strike"]), m["cp"])


def _normalize(rows: List[Dict], asset: str, candles: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for r in rows:
        base, exp, strike, cp = parse_option(r.get("symbol"))
        if base != asset or cp is None or pd.isna(exp) or r.get("isTaker") is False:
            continue
        recs.append((r["tradeId"], int(r["createdAtTimestamp"]), r["symbol"], strike, exp, cp,
                     str(r["side"]).lower(), float(r["quantity"]), float(r["price"])))
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
    df["mark_price"] = np.nan      # not published
    df["iv"] = np.nan              # backed out by lib.block_flow
    df["is_block"] = df["abs_amount"] >= DEFAULT_MIN_SIZES.get(asset, 0.0)
    df["block_id"] = None
    df["wallet"] = None
    return df.drop(columns="ts_ms")


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------
def _fetch_chunk(lo: int, hi: int, max_pages: int) -> Tuple[List[Dict], bool]:
    rows: List[Dict] = []
    seen = set()
    cur = hi
    for _ in range(max_pages):
        page = _get("/history/option-trades", {"_pageSize": 100, "createdAtTimestamp[gte]": str(lo),
                                               "createdAtTimestamp[lte]": str(cur)})
        new = [x for x in page if x["tradeId"] not in seen]
        seen.update(x["tradeId"] for x in new)
        rows.extend(new)
        if len(page) < 100 or not new:
            return rows, False
        cur = int(page[-1]["createdAtTimestamp"])
    return rows, True


def fetch_option_trades(start_ms: int, end_ms: int, max_pages: int = 40) -> Tuple[List[Dict], bool]:
    """All option trades in [start_ms, end_ms] -> (rows, truncated)."""
    chunks, hi = [], int(end_ms)
    while hi > start_ms:
        lo = max(int(start_ms), hi - _CHUNK_MS)
        chunks.append((lo, hi))
        hi = lo - 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chunks), 3)) as ex:
        res = list(ex.map(lambda c: _fetch_chunk(c[0], c[1], max_pages), chunks))
    rows: Dict[str, Dict] = {}
    for r, _ in res:
        for x in r:
            rows[x["tradeId"]] = x
    return list(rows.values()), any(t for _, t in res)


def fetch_candles(asset: str, start_ms: int, end_ms: int, bucket: str = "5m") -> pd.DataFrame:
    """Spot candles (``{asset}USDC``) -> DataFrame[ts_ms (bar open), open, close], oldest first."""
    iso = lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")  # noqa: E731
    rows: List[Dict] = []
    hi = int(end_ms)
    for _ in range(40):
        try:
            page = _get(f"/markets/{asset}USDC/candle",
                        {"timeBucket": bucket, "_pageSize": 100,
                         "createdAtDatetime[gte]": iso(start_ms), "createdAtDatetime[lte]": iso(hi)})
        except RuntimeError:
            break
        if not isinstance(page, list) or not page:
            break
        rows.extend(page)
        oldest = int(page[-1]["createdAtTimestamp"])
        if len(page) < 100 or oldest <= start_ms or oldest >= hi:
            break
        hi = oldest - 1000
    if not rows:
        return pd.DataFrame(columns=["ts_ms", "open", "close"])
    d = pd.DataFrame(rows)
    d["ts_ms"] = pd.to_numeric(d["createdAtTimestamp"]).astype("int64")
    for c in ("open", "close"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return (d[["ts_ms", "open", "close"]].drop_duplicates("ts_ms")
            .sort_values("ts_ms").reset_index(drop=True))


def _bucket(start_ms: int, end_ms: int) -> str:
    return "5m" if (end_ms - start_ms) <= 48 * 3.6e6 else "30m"


def listed_bases() -> List[str]:
    """Underlyings with option markets (``markets`` where marketType == OPTION)."""
    try:
        m = _get("/markets", {}, timeout=60)
        return sorted({x["underlyingBaseSymbol"] for x in m
                       if x.get("marketType") == "OPTION" and x.get("underlyingBaseSymbol")})
    except Exception:
        return []


def fetch_all(start_ms: int, end_ms: int, assets: List[str]) -> Dict:
    """-> {'frames': {base: df}, 'meta': {...}}.  Frames hold EVERY option trade (``is_block`` marks those at/above
    the default min size); the page / home summary apply their own size filter."""
    rows, truncated = fetch_option_trades(start_ms, end_ms)
    present = sorted({parse_option(r["symbol"])[0] for r in rows} - {None})
    listed = sorted(set(listed_bases()) | set(assets) | set(present))
    k_start = start_ms - 3_600_000
    frames: Dict[str, pd.DataFrame] = {a: pd.DataFrame() for a in listed}
    for a in present:
        frames[a] = _normalize(rows, a, fetch_candles(a, k_start, end_ms, _bucket(start_ms, end_ms)))
    return {"frames": frames,
            "meta": {"listed": listed, "tape_rows": len(rows), "truncated": truncated,
                     "other_bases": sorted(b for b in present if b not in assets)}}


def fetch_spot(asset: str) -> float:
    try:
        return float(_get(f"/index-prices/{asset}", {}, timeout=30).get("price") or 0.0)
    except Exception:
        return 0.0


def fetch_hist_spot(asset: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    k = fetch_candles(asset, start_ms, end_ms, _bucket(start_ms, end_ms))
    if k.empty:
        return pd.DataFrame()
    ts = pd.to_datetime(k["ts_ms"], unit="ms", utc=True).dt.tz_convert(SGT)
    return pd.DataFrame({"timestamp": ts, "close": k["close"].values})


def feed_status(meta: Dict, now_ms: Optional[int] = None) -> Optional[str]:
    if meta.get("tape_rows", 0) == 0:
        return "Bullish returned no option trades in this window."
    txt = f"Window contains {meta['tape_rows']:,} option trades; the page shows those at/above the minimum size set in the sidebar."
    if meta.get("truncated"):
        txt += " The window exceeded the page-fetch cap, so the oldest trades are missing."
    return txt


class _Venue:
    key = "bullish"
    title = "Bullish"
    icon = "🐂"
    assets = ASSETS
    default_min_sizes = DEFAULT_MIN_SIZES
    has_mark = False
    has_iv = False
    perp_label = "Spot"
    definition = ("**Block definition:** Bullish has no block/RFQ flag on its public option tape "
                  "(`GET /v1/history/option-trades`), so every print at or above the minimum size in "
                  "the sidebar is shown (default 0.0005 BTC = Bullish's minimum order; raise it to isolate blocks - the "
                  "median Bullish print is ~0.35 BTC and >= 2 BTC is ~12 a day). `side` is the taker side. "
                  "USDC-linear BTC options. The tape has no mark, IV or index: IV is backed out of price "
                  "(Black-Scholes, r=0) using BTCUSDC spot at the trade, so the mark-based Edge and Aggression "
                  "views are not available. DVOL overlay is Deribit's.")
    listed = staticmethod(listed_bases)
    fetch_all = staticmethod(fetch_all)
    fetch_spot = staticmethod(fetch_spot)
    fetch_hist_spot = staticmethod(fetch_hist_spot)
    feed_status = staticmethod(feed_status)


VENUE = _Venue()
