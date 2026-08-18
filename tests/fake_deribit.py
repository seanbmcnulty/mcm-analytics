"""Synthetic Deribit feed so the command layer can be exercised offline."""
from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

SPOT = {"btc_usd": 64000.0, "eth_usd": 1900.0, "sol_usdc": 75.0, "hype_usdc": 59.0}
PERP = {"BTC-PERPETUAL": 64080.0, "ETH-PERPETUAL": 1903.0,
        "SOL_USDC-PERPETUAL": 75.1, "HYPE_USDC-PERPETUAL": 59.2}
_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
DTES = [1, 2, 8, 15, 29, 57, 92, 183, 365]
_PREFIXES = {"BTC": ["BTC"], "ETH": ["ETH"],
             "USDC": ["SOL_USDC", "HYPE_USDC"]}


def _tok(d):
    dt = datetime.now(timezone.utc).date() + timedelta(days=d)
    return f"{dt.day}{_MONTHS[dt.month - 1]}{dt.year % 100:02d}"


def _iv(dte, moneyness):
    """Smile: term structure in sqrt(t) plus a put-skewed quadratic wing."""
    base = 0.45 + 0.10 * math.sqrt(max(dte, 1) / 365.0)
    return max(0.05, base + 0.55 * moneyness ** 2 - 0.18 * moneyness)


def get_index_price(index_name):
    return SPOT.get(index_name)


def get_ticker(instrument_name):
    return {"mark_price": PERP.get(instrument_name, 100.0),
            "current_funding": 0.0001, "open_interest": 1000}


_SPOT_FOR_PREFIX = {"BTC": "btc_usd", "ETH": "eth_usd",
                    "SOL_USDC": "sol_usdc", "HYPE_USDC": "hype_usdc"}


def get_option_chain(currency, kind="option"):
    rows = []
    for prefix in _PREFIXES.get(currency, ["BTC"]):
        spot = SPOT[_SPOT_FOR_PREFIX[prefix]]
        if kind == "future":
            for d in DTES[2:]:
                rows.append({"instrument_name": f"{prefix}-{_tok(d)}",
                             "mark_price": spot * (1 + 0.00025 * d)})
            rows.append({"instrument_name": f"{prefix}-PERPETUAL",
                         "mark_price": spot * 1.0008})
            continue
        for d in DTES:
            for mult in (0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4):
                strike = round(spot * mult, 2)
                m = math.log(strike / spot)
                for cp in ("C", "P"):
                    rows.append({
                        "instrument_name": f"{prefix}-{_tok(d)}-{strike:g}-{cp}",
                        "mark_iv": _iv(d, m) * 100.0,
                        "open_interest": 100, "volume": 10,
                        "mark_price": 0.02, "underlying_price": spot,
                    })
    return rows


def get_instruments(currency, kind="option", expired=False):
    now = datetime.now(timezone.utc)
    out = []
    for prefix in _PREFIXES.get(currency, ["BTC"]):
        for d in DTES:
            out.append({
                "instrument_name": f"{prefix}-{_tok(d)}-1000-C",
                "expiration_timestamp": int((now + timedelta(days=d)).timestamp() * 1000)})
    return out


def get_expirations(currency, kind="option"):
    now = datetime.now(timezone.utc)
    return sorted(int((now + timedelta(days=d)).timestamp() * 1000) for d in DTES)


def get_tradingview_ohlc(instrument_name, resolution, start_ms, end_ms):
    step_ms = {"1D": 86400000, "60": 3600000, "15": 900000}.get(str(resolution), 86400000)
    n = max(3, min(2000, int((end_ms - start_ms) / step_ms)))
    ts = [start_ms + i * step_ms for i in range(n)]
    base = PERP.get(instrument_name)
    if base is None:
        base = SPOT["btc_usd"] * 1.01 if instrument_name.startswith("BTC") else 1900.0
    rng = np.random.default_rng(abs(hash(instrument_name)) % (2 ** 31))
    walk = base * np.cumprod(1 + rng.normal(0, 0.012, n))
    high = walk * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = walk * (1 - np.abs(rng.normal(0, 0.008, n)))
    return pd.DataFrame({"timestamp": pd.to_datetime(ts, unit="ms"),
                         "open": walk, "high": high, "low": low,
                         "close": walk, "volume": np.full(n, 100.0)})


def get_dvol(currency, resolution="1D", start_ms=None, end_ms=None):
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    if start_ms is None:
        start_ms = end_ms - 90 * 86400000
    step = {"1D": 86400000, "60": 3600000}.get(str(resolution), 3600000)
    n = max(5, min(3000, int((end_ms - start_ms) / step)))
    ts = [start_ms + i * step for i in range(n)]
    rng = np.random.default_rng(7)
    vals = 50 + np.cumsum(rng.normal(0, 0.25, n))
    vals = np.clip(vals, 20, 110)
    return pd.DataFrame({"timestamp": pd.to_datetime(ts, unit="ms"),
                         "open": vals, "high": vals + 1, "low": vals - 1,
                         "close": vals})


def get_funding_history(instrument_name, start_ms=None, end_ms=None):
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    if start_ms is None:
        start_ms = end_ms - 7 * 86400000
    n = max(3, int((end_ms - start_ms) / 3600000))
    ts = [start_ms + i * 3600000 for i in range(n)]
    rng = np.random.default_rng(3)
    return pd.DataFrame({"timestamp": pd.to_datetime(ts, unit="ms"),
                         "interest_8h": rng.normal(0.0001, 0.00005, n),
                         "index_price": np.full(n, 64000.0)})


def _request(endpoint, params=None, ttl=60.0, retries=3):
    if endpoint == "get_last_trades_by_currency_and_time":
        now_ms = int(time.time() * 1000)
        rng = np.random.default_rng(11)
        prefix = _PREFIXES.get((params or {}).get("currency", "BTC"), ["BTC"])[0]
        trades = []
        for i in range(60):
            d = DTES[i % len(DTES)]
            trades.append({
                "instrument_name": f"{prefix}-{_tok(d)}-{64000 + 1000 * (i % 5)}-"
                                   f"{'C' if i % 2 else 'P'}",
                "amount": float(rng.choice([15, 25, 50, 130, 400, 6000, 9000])),
                "direction": "buy" if i % 3 else "sell",
                "mark_iv": 55.0, "index_price": 64000.0,
                "timestamp": now_ms - i * 60000,
            })
        return {"trades": trades, "has_more": False}
    return None


def clear_cache():
    pass
