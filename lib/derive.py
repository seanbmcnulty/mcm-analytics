"""
Derive (derive.xyz, formerly Lyra) public REST client — block/RFQ option trades.

No auth needed: every call here is a ``POST https://api.derive.xyz/v3/public/<method>``
with a JSON body (the v3 API).  NOTE: the legacy v2 host ``api.lyra.finance`` is still
up but its trade history is frozen/stale (checked 2026-10-10: ~80h behind), so it must
not be used.  Used by ``pages/03_Block_Trades_-_Derive.py`` via
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
* v3 renamed ``tx_status`` -> ``batch_status`` (``Batching`` ... ``Settled``, plus
  ``*Error`` variants).  Unset returns every state, so fresh not-yet-settled trades
  show up; ``*Error`` rows are dropped here.  The window is capped at 30 days.
* v3 has no ``get_spot_feed_history``; index history comes from
  ``public/get_index_chart_data`` (OHLC candles, UTC seconds) and live spot from
  ``public/get_ticker`` (``I`` = index price).

Public endpoints used (all POST ``/v3/public/<method>``, no auth):
  get_trade_history      option trades incl. quote_id / rfq_id  -> the block tape
  get_ticker             live index price (short key ``I``)
  get_index_chart_data   index OHLC candles for the overlay
  get_live_incidents     unresolved exchange incidents (shown as a page notice)

Checked and deliberately NOT used: every RFQ/quote method (send_rfq, poll_rfqs,
get_rfqs, get_quotes, ...) is ``private/*`` — Derive publishes no public RFQ or
block-trade feed, so the ``quote_id`` / ``rfq_id`` tag on public trades is the
only public block signal.  ``public/get_tickers`` would give live mark/IV per
option (``option_pricing.i``) but only for *current* marks, not at trade time.
Websocket ``trades.option.{currency}`` carries the same data as get_trade_history.
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

API_BASE = "https://api.derive.xyz/v3"
SGT = timezone(timedelta(hours=8))

ASSETS = ["BTC", "ETH", "SOL", "HYPE"]
# Minimum trade size (underlying units) below which a block is hidden.
# Deliberately low: Derive RFQ fills are far smaller than Deribit blocks.
DEFAULT_MIN_SIZES: Dict[str, float] = {}   # 0 for every underlying: show all blocks (flows are small)

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
_INSTR_RE = re.compile(r"^(?P<base>[A-Z0-9]+)-(?P<exp>\d{8})-(?P<strike>[0-9_.]+)-(?P<cp>[CP])$")


def parse_option(name: str):
    """'BTC-20261010-86000-C' -> (base, expiry Timestamp, strike, 'C'/'P')."""
    m = _INSTR_RE.match(name or "")
    if not m:
        return None, pd.NaT, np.nan, None
    return (m["base"], pd.to_datetime(m["exp"], format="%Y%m%d"),
            float(m["strike"].replace("_", ".")), m["cp"])  # sub-$1 strikes: "0_14" -> 0.14


def _normalize(rows: List[Dict], asset: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "batch_status" in df.columns:
        df = df[~df["batch_status"].astype(str).str.endswith("Error")]
        if df.empty:
            return pd.DataFrame()
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
            "page": n, "page_size": 1000})

    first = page(1)
    rows = list(first.get("trades", []))
    num_pages = min(int(first.get("pagination", {}).get("num_pages", 1) or 1), max_pages)
    if num_pages > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(page, range(2, num_pages + 1)):
                rows.extend(res.get("trades", []))
    return rows


def listed_bases() -> List[str]:
    """Underlyings with live options (public/get_all_live_instruments)."""
    try:
        res = _post("get_all_live_instruments", {"instrument_type": "option"})
        names = res if isinstance(res, list) else res.get("instruments", [])
        return sorted({n.split("-")[0] for n in names if "-" in n})
    except Exception:
        return []


def fetch_all(start_ms: int, end_ms: int, assets: List[str]) -> Dict:
    """-> {'frames': {asset: DataFrame (block rows only)}, 'meta': {...}}.

    Fetches every underlying that has live options (``meta['listed']``), not just
    the charted ``assets``, so the page headline can cover them all.  Frames hold
    block/RFQ trades only (the page is a *block* page).  ``meta`` also carries the
    count of all option trades seen and the latest option trade timestamp per
    charted asset, so the page can say when the feed has gone quiet.
    """
    listed = sorted(set(listed_bases()) | set(assets))
    frames: Dict[str, pd.DataFrame] = {}
    meta: Dict = {"all_option_rows": {}, "latest_trade_ms": {}, "listed": listed}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(listed), 8)) as ex:
        futs = {a: ex.submit(fetch_option_trades, a, start_ms, end_ms) for a in listed}
        latest = {a: ex.submit(latest_option_trade_ms, a) for a in assets}
        for a in listed:
            try:
                df = _normalize(futs[a].result(), a)
            except Exception:
                df = pd.DataFrame()
            meta["all_option_rows"][a] = len(df)
            frames[a] = df[df["is_block"]].reset_index(drop=True) if not df.empty else df
        for a in assets:
            try:
                meta["latest_trade_ms"][a] = latest[a].result()
            except Exception:
                meta["latest_trade_ms"][a] = None
    meta["incidents"] = live_incidents()
    return {"frames": frames, "meta": meta}


def latest_option_trade_ms(currency: str) -> Optional[int]:
    """Timestamp (ms) of the most recent settled option trade, any time."""
    res = _post("get_trade_history", {"currency": currency, "instrument_type": "option",
                                      "page": 1, "page_size": 1})
    trades = res.get("trades", [])
    return int(trades[0]["timestamp"]) if trades else None


def fetch_spot(asset: str) -> float:
    """Live index price: ``get_ticker`` on the perp, field ``I`` (v3 slim ticker)."""
    try:
        res = _post("get_ticker", {"instrument_name": f"{asset}-PERP"})
        return float(res.get("I") or res.get("index_price") or 0.0)
    except Exception:
        return 0.0


def fetch_hist_spot(asset: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Index price history (5-min close) via public/get_index_chart_data,
    fetched in 3-day chunks (the endpoint clamps the bucket count)."""
    step = 3 * 86400
    t, end = int(start_ms // 1000), int(end_ms // 1000)
    rows: List[Dict] = []
    while t < end:
        e = min(t + step, end)
        try:
            res = _post("get_index_chart_data", {"currency": asset, "period": 300,
                                                 "start_timestamp": t, "end_timestamp": e})
            rows.extend(res if isinstance(res, list) else res.get("candles", []))
        except Exception:
            pass
        t = e
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows).drop_duplicates("timestamp_bucket").sort_values("timestamp_bucket")
    ts = pd.to_datetime(pd.to_numeric(d["timestamp_bucket"]), unit="s", utc=True).dt.tz_convert(SGT)
    return pd.DataFrame({"timestamp": ts.values, "close": pd.to_numeric(d["close_price"], errors="coerce").values}
                        ).dropna().reset_index(drop=True).assign(
        timestamp=lambda x: pd.to_datetime(x["timestamp"], utc=True).dt.tz_convert(SGT))


def live_incidents() -> List[Dict]:
    """Unresolved exchange incidents (public/get_live_incidents); [] on failure."""
    try:
        return list(_post("get_live_incidents", {}).get("incidents", []))
    except Exception:
        return []


def feed_status(meta: Dict, now_ms: Optional[int] = None) -> Optional[str]:
    """A human note when Derive's trade feed looks stale; None when healthy."""
    now_ms = now_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    inc = meta.get("incidents") or []
    inc_txt = ("Live Derive incident(s): " + "; ".join(
        f"[{i.get('severity', '?')}] {i.get('label', '')}: {i.get('message', '')}" for i in inc) + " ") if inc else ""
    latest = [v for v in meta.get("latest_trade_ms", {}).values() if v]
    if not latest:
        return inc_txt.strip() or None
    age_h = (now_ms - max(latest)) / 3.6e6
    if age_h < 6:
        return inc_txt.strip() or None
    return inc_txt + (f"Derive's public trade-history feed last printed an option trade "
            f"{age_h:,.0f}h ago, so windows shorter than that will be empty.")


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
                  "(Black-Scholes, r=0, per-trade index price). Index line: public/get_index_chart_data. "
                  "DVOL overlay is Deribit's.")
    listed = staticmethod(listed_bases)
    fetch_all = staticmethod(fetch_all)
    fetch_spot = staticmethod(fetch_spot)
    fetch_hist_spot = staticmethod(fetch_hist_spot)
    feed_status = staticmethod(feed_status)


VENUE = _Venue()
