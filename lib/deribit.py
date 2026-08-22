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
from functools import lru_cache
from typing import Any

import requests
import pandas as pd
import numpy as np

from lib.constants import (
    TTL_FAST, TTL_QUICK, TTL_SHORT, TTL_MEDIUM, TTL_LONG, TTL_SLOW,
)

BASE_URL = "https://www.deribit.com/api/v2/public"

# Rate limiting
_rate_lock = threading.Lock()
_last_request_time = 0.0
_MIN_INTERVAL = 0.1  # 10 req/s

# Simple TTL cache
_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()

DEFAULT_TIMEOUT = 15  # seconds


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


def _request(endpoint: str, params: dict | None = None, ttl: float = 60.0,
             retries: int = 3) -> dict | None:
    """
    Make a GET request to Deribit public API with caching and retry.
    Returns the JSON response dict, or None on failure.
    """
    url = f"{BASE_URL}/{endpoint}"
    cache_key = f"{endpoint}|{sorted(params.items()) if params else ''}"

    # Check cache
    cached = _cache_get(cache_key, ttl)
    if cached is not None:
        return cached

    _rate_limit()

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                if "result" in data:
                    _cache_set(cache_key, data["result"])
                    return data["result"]
                return data
            elif resp.status_code == 429:
                # Rate limited — back off
                time.sleep(2 ** attempt)
            else:
                # Non-retryable HTTP error
                return None
        except (requests.RequestException, ValueError):
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            continue
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


def get_trades(currency: str, kind: str = "option",
               start_ms: int | None = None, end_ms: int | None = None,
               count: int = 1000) -> list[dict] | None:
    """
    Get recent trades for a currency/kind.
    Returns list of trade dicts with: instrument_name, price, amount, direction, timestamp, etc.
    """
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    if start_ms is None:
        start_ms = end_ms - 24 * 3600 * 1000  # 24h default

    result = _request("get_last_trades_by_currency_and_time", {
        "currency": currency,
        "kind": kind,
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "count": count,
        "sorting": "desc",
    }, ttl=TTL_MEDIUM)

    if result and "trades" in result:
        return result["trades"]
    return result if isinstance(result, list) else None


def get_funding_history(instrument_name: str,
                        start_ms: int | None = None,
                        end_ms: int | None = None) -> pd.DataFrame | None:
    """
    Get funding rate history for a perpetual instrument.
    Returns DataFrame with columns: timestamp, interest_1h, index_price
    """
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    if start_ms is None:
        start_ms = end_ms - 30 * 24 * 3600 * 1000  # 30 days default

    result = _request("get_funding_rate_history", {
        "instrument_name": instrument_name,
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
    }, ttl=TTL_SLOW)

    if not result or not isinstance(result, list):
        return None

    df = pd.DataFrame(result)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def get_ticker(instrument_name: str) -> dict | None:
    """Get ticker data for an instrument (mark price, funding, OI, etc.)."""
    return _request("get_ticker", {"instrument_name": instrument_name}, ttl=TTL_QUICK)


def get_expirations(currency: str, kind: str = "option") -> list[int] | None:
    """Get list of available expiration timestamps for a currency."""
    # Deribit doesn't have a direct expirations endpoint for all currencies;
    # derive from instruments list.
    instruments = get_instruments(currency, kind)
    if not instruments:
        return None
    expirations = sorted(set(i["expiration_timestamp"] for i in instruments))
    return expirations


def get_book_summary_by_instrument(instrument_name: str) -> dict | None:
    """Get book summary for a single instrument."""
    result = _request("get_book_summary_by_instrument",
                      {"instrument_name": instrument_name}, ttl=TTL_SHORT)
    if isinstance(result, list) and len(result) > 0:
        return result[0]
    return result if isinstance(result, dict) else None


# ---------------------------------------------------------------------------
# Higher-level helpers
# ---------------------------------------------------------------------------

def get_atm_iv_by_dte(currency: str, spot: float | None = None) -> pd.DataFrame | None:
    """
    Compute ATM implied volatility for each listed expiry from the option chain.
    Returns DataFrame with columns: dte, atm_iv, expiry_ts
    """
    chain = get_option_chain(currency, "option")
    if not chain:
        return None

    if spot is None:
        index_name = {"BTC": "btc_usd", "ETH": "eth_usd", "USDC": "sol_usdc"}.get(currency)
        if index_name:
            spot = get_index_price(index_name)
    if not spot:
        return None

    now_ms = int(time.time() * 1000)
    rows = []
    for inst in chain:
        if inst.get("mark_iv") is None or inst["mark_iv"] == 0:
            continue
        name = inst["instrument_name"]
        # Parse strike from instrument name: BTC-27JUN26-120000-C
        parts = name.split("-")
        if len(parts) < 4:
            continue
        try:
            strike = float(parts[2].replace("d", "."))
        except ValueError:
            continue

        expiry_ts = inst.get("expiration_timestamp", 0)
        dte = max((expiry_ts - now_ms) / (1000 * 86400), 0.01)

        # ATM = strike closest to spot
        moneyness = abs(np.log(strike / spot))
        if moneyness < 0.05:  # within 5% of spot = near-ATM
            rows.append({
                "dte": dte,
                "atm_iv": inst["mark_iv"],
                "expiry_ts": expiry_ts,
                "instrument": name,
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    # Average multiple near-ATM options per expiry
    df = df.groupby("expiry_ts").agg({"dte": "first", "atm_iv": "mean"}).reset_index()
    df = df.sort_values("dte").reset_index(drop=True)
    return df


def clear_cache():
    """Clear the entire request cache."""
    with _cache_lock:
        _cache.clear()
