"""
Deribit Public API Client — shared across all MCM Analytics pages.

All endpoints are public (no authentication required). Provides:
- In-memory LRU cache with configurable TTL
- Automatic retry with exponential backoff
- Rate limiting (10 req/s default)
- Common fetcher functions for all pages
"""

import time
import threading
from typing import Any

import requests
import pandas as pd

from lib.constants import (
    TTL_FAST, TTL_QUICK, TTL_MEDIUM, TTL_LONG, TTL_SLOW,
)

BASE_URL = "https://www.deribit.com/api/v2/public"

# Rate limiting
_rate_lock = threading.Lock()
_last_request_time = 0.0
_MIN_INTERVAL = 0.1  # 10 req/s

# Simple TTL cache
_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()

# Single-flight: when several worker threads miss the cache on the same key at
# once (the MCM Bot page runs commands 4-wide), only the first one hits the
# network; the rest wait for its result instead of firing duplicate requests.
_inflight: dict[str, threading.Event] = {}
_inflight_lock = threading.Lock()

# One keep-alive HTTPS session per thread. A bare requests.get() opens a new
# TCP + TLS connection (and rebuilds the SSL context) on every call, which
# measured ~340 ms/call to Deribit vs ~175 ms on a reused connection.
_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        _tls.session = s
    return s

DEFAULT_TIMEOUT = 15  # seconds
# Backoff between attempts (3 attempts total -> waits 2s, then 5s; a third
# entry is used if retries is raised above 3).
RETRY_BACKOFF_S = (2.0, 5.0, 15.0)


def _rate_limit():
    """Enforce minimum interval between requests."""
    global _last_request_time
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_request_time = time.time()


def _cache_get(key: str, ttl: float) -> Any | None:
    """Return cached value if still fresh, else None."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry[0]) < ttl:
            return entry[1]
    return None


def _cache_set(key: str, value: Any):
    """Store value in cache with current timestamp."""
    with _cache_lock:
        _cache[key] = (time.time(), value)
        # Prune if too large (simple LRU-ish: drop oldest 25%)
        if len(_cache) > 500:
            sorted_keys = sorted(_cache, key=lambda k: _cache[k][0])
            for k in sorted_keys[:125]:
                del _cache[k]


# Time-window endpoints whose callers pass ``end_timestamp=now`` (ms). With the
# raw millisecond value in the cache key, no two calls ever shared a cache
# entry (basis_run re-fetched the same perp candles once per future). Widen the
# window to whole ``ttl`` buckets instead -- start floored, end ceiled -- so
# every call within one bucket gets one key and one fetch, and no candle that
# the caller asked for is ever dropped. Not applied to paginated endpoints
# (e.g. get_last_trades_by_currency_and_time), whose start is a cursor.
_WINDOW_ENDPOINTS = {
    "get_tradingview_chart_data",
    "get_volatility_index_data",
    "get_funding_rate_history",
}
_MIN_BUCKET_MS = 60_000


def _bucket_window(endpoint: str, params: dict | None, ttl: float) -> dict | None:
    if (not params or endpoint not in _WINDOW_ENDPOINTS
            or "end_timestamp" not in params or "start_timestamp" not in params):
        return params
    try:
        b = max(int(float(ttl) * 1000), _MIN_BUCKET_MS)
        start, end = int(params["start_timestamp"]), int(params["end_timestamp"])
    except (TypeError, ValueError):
        return params
    out = dict(params)
    out["start_timestamp"] = (start // b) * b
    out["end_timestamp"] = -(-end // b) * b
    return out


def _request(endpoint: str, params: dict | None = None, ttl: float = 60.0,
             retries: int = 3) -> dict | None:
    """
    Make a GET request to Deribit public API with caching and retry.
    Returns the JSON response dict, or None on failure.
    """
    url = f"{BASE_URL}/{endpoint}"
    params = _bucket_window(endpoint, params, ttl)
    cache_key = f"{endpoint}|{sorted(params.items()) if params else ''}"

    # Check cache
    cached = _cache_get(cache_key, ttl)
    if cached is not None:
        return cached

    with _inflight_lock:
        event = _inflight.get(cache_key)
        leader = event is None
        if leader:
            event = threading.Event()
            _inflight[cache_key] = event
    if not leader:
        # Another thread is already fetching this exact request.
        event.wait(timeout=DEFAULT_TIMEOUT * retries + sum(RETRY_BACKOFF_S))
        cached = _cache_get(cache_key, ttl)
        if cached is not None:
            return cached
        # Leader failed or returned an uncacheable body: fetch ourselves.
        return _fetch(url, params, cache_key, retries)
    try:
        return _fetch(url, params, cache_key, retries)
    finally:
        with _inflight_lock:
            _inflight.pop(cache_key, None)
        event.set()


def _fetch(url: str, params: dict | None, cache_key: str, retries: int):
    """Network part of ``_request``: rate limit, GET with retry, cache."""
    _rate_limit()

    for attempt in range(retries):
        retryable = False
        try:
            resp = _session().get(url, params=params, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                if "result" in data:
                    _cache_set(cache_key, data["result"])
                    return data["result"]
                return data
            # 429 (rate limit) and 5xx (Deribit hiccup) are transient; other
            # 4xx are our fault and won't get better on retry.
            retryable = resp.status_code == 429 or resp.status_code >= 500
        except (requests.RequestException, ValueError):
            retryable = True
        if not retryable or attempt >= retries - 1:
            break
        time.sleep(RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)])
    return None


# ---------------------------------------------------------------------------
# Public API fetchers
# ---------------------------------------------------------------------------

def get_index_price(index_name: str) -> float | None:
    """Get current index price. index_name: btc_usd, eth_usd, sol_usdc, hype_usdc"""
    result = _request("get_index_price", {"index_name": index_name}, ttl=TTL_FAST)
    if result and "index_price" in result:
        return result["index_price"]
    return None


def get_option_chain(currency: str, kind: str = "option") -> list[dict] | None:
    """
    Get full book summary for a currency.
    currency: BTC, ETH, USDC (for SOL/HYPE linear options)
    kind: option, future
    Returns list of instrument summaries.
    """
    result = _request("get_book_summary_by_currency",
                      {"currency": currency, "kind": kind}, ttl=TTL_LONG)
    return result if isinstance(result, list) else None


def get_instruments(currency: str, kind: str = "option",
                    expired: bool = False) -> list[dict] | None:
    """Get list of active instruments."""
    result = _request("get_instruments",
                      {"currency": currency, "kind": kind, "expired": str(expired).lower()},
                      ttl=TTL_SLOW)
    return result if isinstance(result, list) else None


def get_tradingview_ohlc(instrument_name: str, resolution: str | int,
                         start_ms: int, end_ms: int) -> pd.DataFrame | None:
    """
    Get OHLC candles from Deribit's tradingview endpoint.
    resolution: 1, 3, 5, 10, 15, 30, 60, 120, 180, 360, 720, 1D
    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    """
    result = _request("get_tradingview_chart_data", {
        "instrument_name": instrument_name,
        "resolution": str(resolution),
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
    }, ttl=TTL_MEDIUM)

    if not result or "ticks" not in result:
        return None

    df = pd.DataFrame({
        "timestamp": result["ticks"],
        "open": result["open"],
        "high": result["high"],
        "low": result["low"],
        "close": result["close"],
        "volume": result.get("volume", [0] * len(result["ticks"])),
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def get_dvol(currency: str, resolution: str | int = "1D",
             start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame | None:
    """
    Get DVOL (Deribit Volatility Index) history. BTC and ETH only.
    resolution: 1, 60, 3600, 43200, 1D
    Returns DataFrame with columns: timestamp, open, high, low, close
    """
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    if start_ms is None:
        start_ms = end_ms - 90 * 24 * 3600 * 1000  # 90 days default

    result = _request("get_volatility_index_data", {
        "currency": currency,
        "resolution": str(resolution),
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
    }, ttl=TTL_SLOW)

    if not result or "data" not in result:
        return None

    rows = result["data"]
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def get_last_trades_by_currency_and_time(
    currency: str,
    kind: str = "option",
    start_ms: int | None = None,
    end_ms: int | None = None,
    count: int = 1000,
    sorting: str = "asc",
    ttl: float = TTL_MEDIUM,
) -> dict | None:
    """One page of ``get_last_trades_by_currency_and_time``.

    Returns Deribit's result dict (``trades``, ``has_more``, …) so callers
    that need pagination — notably ``pages/02_Block_Trades_-_Deribit.py`` —
    can advance ``start_timestamp`` without bypassing this module's shared
    rate budget and TTL cache. Prefer this over raw ``requests.get``.
    """
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    if start_ms is None:
        start_ms = end_ms - 24 * 3600 * 1000

    result = _request("get_last_trades_by_currency_and_time", {
        "currency": currency,
        "kind": kind,
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "count": count,
        "sorting": sorting,
    }, ttl=ttl)
    return result if isinstance(result, dict) else None


_FUNDING_HISTORY_MAX_SPAN_MS = 30 * 24 * 3600 * 1000  # Deribit caps this endpoint at ~31d/request


def get_funding_history(instrument_name: str,
                        start_ms: int | None = None,
                        end_ms: int | None = None) -> pd.DataFrame | None:
    """
    Get funding rate history for a perpetual instrument.
    Returns DataFrame with columns: timestamp, interest_1h, index_price

    Deribit's get_funding_rate_history endpoint silently caps each request
    to ~31 days of history regardless of the requested start_timestamp, so
    a wide range is paginated here into <=30-day chunks and concatenated
    (each chunk still goes through the shared TTL cache/rate limiter).
    """
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    if start_ms is None:
        start_ms = end_ms - 30 * 24 * 3600 * 1000  # 30 days default

    chunks = []
    chunk_end = end_ms
    while chunk_end > start_ms:
        chunk_start = max(start_ms, chunk_end - _FUNDING_HISTORY_MAX_SPAN_MS)
        result = _request("get_funding_rate_history", {
            "instrument_name": instrument_name,
            "start_timestamp": chunk_start,
            "end_timestamp": chunk_end,
        }, ttl=TTL_SLOW)
        if result and isinstance(result, list):
            chunks.append(pd.DataFrame(result))
        chunk_end = chunk_start

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    if "timestamp" in df.columns:
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.reset_index(drop=True)


def get_ticker(instrument_name: str) -> dict | None:
    """Get ticker data for an instrument (mark price, funding, OI, etc.).

    The public endpoint is ``public/ticker`` -- there is no ``get_ticker``
    (Deribit answers HTTP 400 "Method not found"), which silently made every
    caller see None: no PERPETUAL row in /basis_run, no PERP pill in the
    /vol_run header, no funding delta on the Home page.
    """
    return _request("ticker", {"instrument_name": instrument_name}, ttl=TTL_QUICK)


def get_expirations(currency: str, kind: str = "option") -> list[int] | None:
    """Get list of available expiration timestamps for a currency."""
    # Deribit doesn't have a direct expirations endpoint for all currencies;
    # derive from instruments list.
    instruments = get_instruments(currency, kind)
    if not instruments:
        return None
    expirations = sorted(set(i["expiration_timestamp"] for i in instruments))
    return expirations


# ---------------------------------------------------------------------------
# Higher-level helpers
# ---------------------------------------------------------------------------

_cache_gen = 0


def cache_generation() -> int:
    """Bumped by ``clear_cache`` so derived memos elsewhere can tell that a
    "Clear data cache" / refresh happened and drop their own copies."""
    return _cache_gen


def clear_cache():
    """Clear the entire request cache."""
    global _cache_gen
    with _cache_lock:
        _cache.clear()
        _cache_gen += 1
