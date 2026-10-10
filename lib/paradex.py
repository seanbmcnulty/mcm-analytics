"""
Paradex public REST client — block trade tape (options).

No auth needed.  ``GET https://api.prod.paradex.trade/v1/trades`` with
``trade_type=BLOCK_TRADE`` returns the exchange-wide block tape with ``market``
and ``base_asset`` both omitted — a single cursor-paged feed covering every
market, so there is no per-market looping.  Used by
``pages/04_Block_Trades_-_Paradex.py`` via ``lib.block_flow.render_page``.

Per the Paradex docs:
* each leg of a block package is its own print; legs share one ``block_id``
* ``side`` is the taker side
* perp delta hedges ride along on the exchange-wide block feed; they are split
  out here (option legs go to the charts, perp legs are only counted)

Limitations vs. Deribit/Derive:
* the tape has no mark price, IV, wallet or index price — only id, market, side,
  size, price, created_at, block_id.  IV is backed out of price (Black-Scholes,
  r=0) using the Paradex perp kline close at the trade time, and the mark-based
  charts (edge / aggression) are not available.
* klines are capped at a limited number of bars per call, so they are fetched in
  chunks.
"""

from __future__ import annotations

import concurrent.futures
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

API_BASE = "https://api.prod.paradex.trade/v1"
SGT = timezone(timedelta(hours=8))

ASSETS = ["BTC", "ETH", "SOL", "HYPE"]
DEFAULT_MIN_SIZES = {"BTC": 0.01, "ETH": 0.1, "SOL": 1.0, "HYPE": 10.0}

_session = requests.Session()
_KLINE_CHUNK_BARS = 500


def _get(path: str, params: Dict, timeout: int = 30, retries: int = 3) -> Dict:
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
        except requests.RequestException as e:
            last = e
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"Paradex {path} failed: {last}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
# Parsed from the symbol, not /v1/markets: that list only holds *live* markets, and
# ~1/3 of a week's block prints are on options that have since expired.  (Checked
# 2026-10-10: /v1/markets has only OPTION/PERP/SPOT kinds, no PERP_OPTION or
# RFQ_ONLY markets, and every non-regex tape market is a perp hedge.)  Crypto
# options expire 08:00 UTC, which lib.block_flow assumes.
_OPT_RE = re.compile(r"^(?P<base>[A-Z0-9]+)-USD-(?P<exp>\d{1,2}[A-Z]{3}\d{2})-(?P<strike>[0-9_.]+)-(?P<cp>[CP])$")


def parse_option(market: str):
    m = _OPT_RE.match(market or "")
    if not m:
        return None, pd.NaT, np.nan, None
    return (m["base"], pd.to_datetime(m["exp"], format="%d%b%y", errors="coerce"),
            float(m["strike"].replace("_", ".")), m["cp"])


def fetch_block_tape(start_ms: int, end_ms: int, max_pages: int = 60) -> List[Dict]:
    """Raw exchange-wide block prints in [start_ms, end_ms], cursor-paged."""
    rows: List[Dict] = []
    params: Dict = {"trade_type": "BLOCK_TRADE", "page_size": 1000,
                    "start_at": int(start_ms), "end_at": int(end_ms)}
    for _ in range(max_pages):
        j = _get("/trades", params)
        rows.extend(j.get("results", []))
        cur = j.get("next")
        if not cur:
            break
        params = {"cursor": cur}  # the scope of the request rides inside the cursor
    seen, out = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def fetch_klines(base: str, start_ms: int, end_ms: int, resolution: int = 1) -> pd.DataFrame:
    """Perp klines -> DataFrame[ts_ms, open, close]; chunked to respect the bar cap."""
    step = resolution * 60_000 * _KLINE_CHUNK_BARS
    rows: List[List[float]] = []
    t = int(start_ms)
    while t < end_ms:
        e = min(t + step, int(end_ms))
        try:
            j = _get("/markets/klines", {"symbol": f"{base}-USD-PERP", "resolution": resolution,
                                         "start_at": t, "end_at": e})
            rows.extend(j.get("results", []))
        except RuntimeError:
            pass
        t = e
    if not rows:
        return pd.DataFrame(columns=["ts_ms", "open", "close"])
    d = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    return (d[["ts_ms", "open", "close"]].drop_duplicates("ts_ms")
            .sort_values("ts_ms").reset_index(drop=True))


def _normalize(rows: List[Dict], asset: str, klines: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for r in rows:
        base, exp, strike, cp = parse_option(r["market"])
        if base != asset or cp is None or pd.isna(exp):
            continue
        recs.append((r["id"], r["created_at"], r["market"], strike, exp, cp,
                     str(r["side"]).lower(), float(r["size"]), float(r["price"]),
                     r.get("block_id")))
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs, columns=["trade_id", "ts_ms", "instrument_name", "strike", "expiry",
                                     "option_type", "direction", "abs_amount", "price", "block_id"])
    df = df.sort_values("ts_ms").reset_index(drop=True)
    if not klines.empty:
        k = klines.rename(columns={"ts_ms": "k_ms", "open": "index_price"})[["k_ms", "index_price"]]
        df = pd.merge_asof(df, k, left_on="ts_ms", right_on="k_ms", direction="backward",
                           tolerance=6 * 3_600_000).drop(columns="k_ms")
    else:
        df["index_price"] = np.nan
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(SGT)
    df["amount"] = np.where(df["direction"] == "buy", df["abs_amount"], -df["abs_amount"])
    df["mark_price"] = np.nan  # not published on the public tape
    df["iv"] = np.nan          # backed out by lib.block_flow
    df["is_block"] = True
    df["wallet"] = None
    # Perp delta hedge legs that ride in the same package (shared block_id):
    # signed perp size (taker side, buy +) in underlying units, per package.
    hedge: Dict[str, float] = {}
    for r in rows:
        if r["market"] == f"{asset}-USD-PERP" and r.get("block_id"):
            sz = float(r["size"]) * (1 if str(r["side"]).upper() == "BUY" else -1)
            hedge[r["block_id"]] = hedge.get(r["block_id"], 0.0) + sz
    df["hedge_size"] = df["block_id"].map(hedge).fillna(0.0)
    return df.drop(columns="ts_ms")


# ---------------------------------------------------------------------------
# Public fetchers
# ---------------------------------------------------------------------------
def listed_bases() -> List[str]:
    """Underlyings with options on /v1/markets (live markets only)."""
    try:
        m = _get("/markets", {}).get("results", [])
        return sorted({x["base_currency"] for x in m if x.get("asset_kind") in ("OPTION", "PERP_OPTION")})
    except Exception:
        return []


def fetch_all(start_ms: int, end_ms: int, assets: List[str]) -> Dict:
    """One tape fetch + one kline fetch per underlying on the tape
    -> {'frames': {base: df}, 'meta': {...}}.  Every listed underlying that printed a
    block is normalised (``meta['listed']`` = all listed bases); only ``assets`` get
    their own tab, the rest feed the page headline / home-page summary."""
    tape = fetch_block_tape(start_ms, end_ms)
    n_opt = n_perp = 0
    present_set = set()
    for r in tape:
        base, _, _, cp = parse_option(r["market"])
        if cp:
            n_opt += 1
            present_set.add(base)
        else:
            n_perp += 1
    k_start = start_ms - 6 * 3_600_000   # klines only exist for minutes with trades
    res = 1 if (end_ms - start_ms) <= 48 * 3.6e6 else 5
    listed = sorted(set(listed_bases()) | set(assets) | present_set)
    frames: Dict[str, pd.DataFrame] = {a: pd.DataFrame() for a in listed}
    todo = [a for a in listed if a in present_set]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max(len(todo), 1), 8)) as ex:
        kl = {a: ex.submit(fetch_klines, a, k_start, end_ms, res) for a in todo}
        for a in todo:
            try:
                k = kl[a].result()
            except Exception:
                k = pd.DataFrame(columns=["ts_ms", "open", "close"])
            frames[a] = _normalize(tape, a, k)
    return {"frames": frames,
            "meta": {"tape_rows": len(tape), "option_legs": n_opt, "non_option_legs": n_perp,
                     "listed": listed,
                     "other_bases": sorted(b for b in present_set if b and b not in assets)}}


def fetch_spot(asset: str) -> float:
    """Live perp index from /markets/summary (klines are sparse on quiet perps)."""
    try:
        j = _get("/markets/summary", {"market": f"{asset}-USD-PERP"})
        row = (j.get("results") or [{}])[0]
        return float(row.get("underlying_price") or row.get("mark_price") or 0.0)
    except Exception:
        return 0.0


def fetch_hist_spot(asset: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    k = fetch_klines(asset, start_ms, end_ms, 5)
    if k.empty:
        return pd.DataFrame()
    ts = pd.to_datetime(k["ts_ms"], unit="ms", utc=True).dt.tz_convert(SGT)
    return pd.DataFrame({"timestamp": ts, "close": k["close"].values})


def feed_status(meta: Dict, now_ms: Optional[int] = None) -> Optional[str]:
    if meta.get("tape_rows", 0) == 0:
        return "Paradex returned no block prints in this window."
    parts = [f"{meta['option_legs']} option legs on the block tape"]
    if meta.get("non_option_legs"):
        parts.append(f"{meta['non_option_legs']} perp hedge legs (not charted)")
    if meta.get("other_bases"):
        parts.append("other underlyings (headline only, no tab): " + ", ".join(meta["other_bases"]))
    return "Window contains " + "; ".join(parts) + "."


class _Venue:
    key = "paradex"
    title = "Paradex"
    icon = "🔷"
    assets = ASSETS
    default_min_sizes = DEFAULT_MIN_SIZES
    has_mark = False
    has_iv = False
    perp_label = "Perp"
    definition = ("**Block definition:** prints with `trade_type=BLOCK_TRADE` on the public trade tape "
                  "(exchange-wide, `GET /v1/trades`). `side` is the taker side; legs of one package share "
                  "a `block_id`. The tape has no mark or IV, so IV is backed out of price (Black-Scholes, "
                  "r=0) using the Paradex perp price at the trade minute, and the mark-based Edge and "
                  "Aggression views are not available. DVOL overlay is Deribit's.")
    fetch_all = staticmethod(fetch_all)
    fetch_spot = staticmethod(fetch_spot)
    fetch_hist_spot = staticmethod(fetch_hist_spot)
    feed_status = staticmethod(feed_status)


VENUE = _Venue()
