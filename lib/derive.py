"""
Derive (derive.xyz, formerly Lyra) public REST client — block/RFQ option trades.

No auth needed: every call here is a ``POST https://api.lyra.finance/public/<method>``
with a JSON body.  Used by ``pages/03_Block_Trades_-_Derive.py`` via
``lib.block_flow.render_page(derive.VENUE)``.

Block definition
----------------
Derive has no separate "block trade" product; the closest equivalent is an
RFQ fill — a trade negotiated off-book via a quote.  ``public/get_trade_history``
tags those with a non-null ``quote_id`` / ``rfq_id``, so a trade is treated as a
block when either is present (same rule as the exodus-analytics Derive page).
Multi-leg RFQs share one ``rfq_id``, which is used to group legs into packages.

Notes
-----
* Every match is published twice (one maker row, one taker row, same
  ``trade_id``).  We keep the taker row, so ``direction`` is the taker's side —
  the same convention as ``lib.deribit`` trades.
* Options are USDC-quoted/settled: ``trade_price`` is USD per 1 underlying and
  ``trade_amount`` is in underlying units, so premium = price * amount.
* ``get_trade_history`` returns newest-first and caps ``page_size`` at 1000;
  the response carries ``pagination.num_pages`` so remaining pages can be
  fetched concurrently.
* The public trade-history feed has no IV field.  ``lib.block_flow`` backs IV
  out of the trade price (Black-Scholes, r=0) using the per-trade index price.
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

API_BASE = "https://api.lyra.finance"
SGT = timezone(timedelta(hours=8))

ASSETS = ["BTC", "ETH", "SOL", "HYPE"]
# Minimum trade size (underlying units) below which a block is hidden.
# Deliberately low: Derive RFQ fills are far smaller than Deribit blocks.
DEFAULT_MIN_SIZES = {"BTC": 0.1, "ETH": 1.0, "SOL": 10.0, "HYPE": 100.0}

_session = requests.Session()


def _post(method: str, params: Dict, timeout: int = 30, retries: int = 3) -> Dict:
    url = f"{API_BASE}/public/{method}"
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = _session.post(url, json=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            return data.get("result", {})
        except (requests.RequestException, RuntimeError) as e:
            last_err = e
            if isinstance(e, RuntimeError):
                break  # API-level error: retrying won't help
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"Derive {method} failed: {last_err}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_INSTR_RE = re.compile(r"^(?P<base>[A-Z0-9]+)-(?P<exp>\d{8})-(?P<strike>[0-9.]+)-(?P<cp>[CP])$")


def parse_option(name: str):
    """'BTC-20261010-86000-C' -> (base, expiry Timestamp, strike, 'C'/'P')."""
    m = _INSTR_RE.match(name or "")
    if not m:
        return None, pd.NaT, np.nan, None
    return (m["base"], pd.to_datetime(m["exp"], format="%Y%m%d"),
            float(m["strike"]), m["cp"])


def _normalize(rows: List[Dict], asset: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ("trade_amount", "trade_price", "mark_price", "index_price"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    # One row per match: prefer the taker row.
    if "liquidity_role" in df.columns:
        df["_is_taker"] = df["liquidity_role"].astype(str).str.lower().eq("taker")
        df = (df.sort_values("_is_taker", ascending=False)
                .drop_duplicates("trade_id", keep="first"))
    df = df[df["instrument_name"].str.startswith(f"{asset}-")].copy()
    if df.empty:
        return pd.DataFrame()

    parsed = df["instrument_name"].map(parse_option)
    df["expiry"] = parsed.map(lambda x: x[1])
    df["strike"] = parsed.map(lambda x: x[2])
    df["option_type"] = parsed.map(lambda x: x[3])
    df = df.dropna(subset=["expiry", "strike", "option_type"])

    # Some low-priced assets (HYPE) occasionally list strikes in scaled units.
    # Only touch strikes that are implausible versus the contemporaneous index.
    bad = (df["index_price"] > 0) & (
        (df["strike"] / df["index_price"] > 8) | (df["strike"] / df["index_price"] < 0.1))
    if bad.any():
        for scale in (10.0, 100.0):
            cand = df.loc[bad, "strike"] / scale
            ratio = cand / df.loc[bad, "index_price"]
            ok = (ratio >= 0.1) & (ratio <= 8)
            idx = ok[ok].index
            df.loc[idx, "strike"] = cand.loc[idx]
            bad.loc[idx] = False

    quote = df["quote_id"] if "quote_id" in df.columns else pd.Series(None, index=df.index)
    rfq = df["rfq_id"] if "rfq_id" in df.columns else pd.Series(None, index=df.index)
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(SGT)

    out = pd.DataFrame({
        "trade_id": df["trade_id"].values,
        "timestamp": ts.values,
        "instrument_name": df["instrument_name"].values,
        "strike": df["strike"].values,
        "expiry": df["expiry"].values,
        "option_type": df["option_type"].values,
        "direction": df["direction"].astype(str).str.lower().values,
        "abs_amount": df["trade_amount"].abs().values,
        "price": df["trade_price"].values,
        "mark_price": df["mark_price"].values,
        "index_price": df["index_price"].values,
        "iv": np.nan,
        "is_block": (quote.notna() | rfq.notna()).values,
        "block_id": rfq.where(rfq.notna(), quote).values,
        "wallet": df["wallet"].values if "wallet" in df.columns else None,
    })
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(SGT)
    out["amount"] = np.where(out["direction"] == "buy", out["abs_amount"], -out["abs_amount"])
    return out.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public fetchers
# ---------------------------------------------------------------------------
def fetch_option_trades(currency: str, start_ms: int, end_ms: int,
                        max_pages: int = 30, workers: int = 4) -> List[Dict]:
    """All settled option trade rows (both maker+taker) for one currency."""
    def page(n: int) -> Dict:
        return _post("get_trade_history", {
            "currency": currency, "instrument_type": "option",
            "from_timestamp": int(start_ms), "to_timestamp": int(end_ms),
            "tx_status": "settled", "page": n, "page_size": 1000})

    first = page(1)
    rows = list(first.get("trades", []))
    num_pages = min(int(first.get("pagination", {}).get("num_pages", 1) or 1), max_pages)
    if num_pages > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(page, range(2, num_pages + 1)):
                rows.extend(res.get("trades", []))
    return rows


def fetch_all(start_ms: int, end_ms: int, assets: List[str]) -> Dict:
    """-> {'frames': {asset: DataFrame (block rows only)}, 'meta': {...}}.

    Frames hold block/RFQ trades only (the page is a *block* page).  ``meta``
    carries the count of all option trades seen and the latest option trade
    timestamp per asset, so the page can say when the feed has gone quiet.
    """
    frames: Dict[str, pd.DataFrame] = {}
    meta: Dict = {"all_option_rows": {}, "latest_trade_ms": {}}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(assets)) as ex:
        futs = {a: ex.submit(fetch_option_trades, a, start_ms, end_ms) for a in assets}
        latest = {a: ex.submit(latest_option_trade_ms, a) for a in assets}
        for a in assets:
            rows = futs[a].result()
            df = _normalize(rows, a)
            meta["all_option_rows"][a] = len(df)
            frames[a] = df[df["is_block"]].reset_index(drop=True) if not df.empty else df
            try:
                meta["latest_trade_ms"][a] = latest[a].result()
            except Exception:
                meta["latest_trade_ms"][a] = None
    return {"frames": frames, "meta": meta}


def latest_option_trade_ms(currency: str) -> Optional[int]:
    """Timestamp (ms) of the most recent settled option trade, any time."""
    res = _post("get_trade_history", {"currency": currency, "instrument_type": "option",
                                      "tx_status": "settled", "page": 1, "page_size": 1})
    trades = res.get("trades", [])
    return int(trades[0]["timestamp"]) if trades else None


def fetch_spot(asset: str) -> float:
    """Latest index price from the spot feed.

    Deliberately NOT ``get_ticker().index_price``: when the venue is quiet that
    field can lag by days (observed 2026-10-10: ticker index was ~3 days old and
    ~1-3% off, while the spot feed matched Deribit/Paradex)."""
    now_s = int(time.time())
    try:
        res = _post("get_spot_feed_history", {"currency": asset, "period": 300,
                                              "start_timestamp": now_s - 3600, "end_timestamp": now_s})
        feed = res.get("spot_feed_history", [])
        if feed:
            return float(feed[-1]["price"])
    except Exception:
        pass
    try:
        return float(_post("get_ticker", {"instrument_name": f"{asset}-PERP"}).get("index_price") or 0.0)
    except Exception:
        return 0.0


def fetch_hist_spot(asset: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Index price history (5-min buckets) -> DataFrame[timestamp (SGT), close]."""
    try:
        res = _post("get_spot_feed_history", {
            "currency": asset, "period": 300,
            "start_timestamp": int(start_ms // 1000), "end_timestamp": int(end_ms // 1000)})
    except Exception:
        return pd.DataFrame()
    feed = res.get("spot_feed_history", [])
    if not feed:
        return pd.DataFrame()
    d = pd.DataFrame(feed)
    ts = pd.to_datetime(pd.to_numeric(d["timestamp"]), unit="s", utc=True).dt.tz_convert(SGT)
    return pd.DataFrame({"timestamp": ts, "close": pd.to_numeric(d["price"], errors="coerce")}
                        ).dropna().sort_values("timestamp").reset_index(drop=True)


def feed_status(meta: Dict, now_ms: Optional[int] = None) -> Optional[str]:
    """A human note when Derive's trade feed looks stale; None when healthy."""
    now_ms = now_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    latest = [v for v in meta.get("latest_trade_ms", {}).values() if v]
    if not latest:
        return None
    age_h = (now_ms - max(latest)) / 3.6e6
    if age_h < 6:
        return None
    return (f"Derive's public trade-history feed last printed an option trade "
            f"{age_h:,.0f}h ago, so windows shorter than that will be empty. "
            f"The API itself is responding (live index/ticker data is fine).")


class _Venue:
    key = "derive"
    title = "Derive"
    icon = "🟣"
    assets = ASSETS
    default_min_sizes = DEFAULT_MIN_SIZES
    has_mark = True
    has_iv = False  # backed out of price by lib.block_flow
    perp_label = "Index"
    definition = ("**Block definition:** a trade is a block when it was filled via RFQ "
                  "(`quote_id` / `rfq_id` present in `public/get_trade_history`). "
                  "Taker rows only; direction = taker side. IV is backed out of trade price "
                  "(Black-Scholes, r=0, per-trade index price). DVOL overlay is Deribit's.")
    fetch_all = staticmethod(fetch_all)
    fetch_spot = staticmethod(fetch_spot)
    fetch_hist_spot = staticmethod(fetch_hist_spot)
    feed_status = staticmethod(feed_status)


VENUE = _Venue()
