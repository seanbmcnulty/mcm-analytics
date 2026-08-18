"""
MCM Analytics — Deribit live volatility surface.

Replaces the Amberdata "delta surface" feed used by the FalconX bot with an
equivalent built purely from Deribit's public option book summary:

  * ``mark_iv`` per listed instrument  -> smile by Black-Scholes delta
  * nearest-strike-to-index            -> ATM
  * BS delta search                    -> 10 / 25 / 35 delta calls and puts

Everything is returned keyed by integer DTE (days to expiry), IV in DECIMAL
(0.55 == 55%).  Deribit reports ``mark_iv`` in percent, so it is normalised
once here and the rest of the codebase can stop guessing at units.
"""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from lib import deribit
from lib.constants import ASSET_CONFIG

# Delta buckets we extract from the live chain.  Names mirror the Amberdata
# keys the FalconX page used so the chart code reads the same.
DELTA_KEYS = [
    "deltaCall10", "deltaCall25", "deltaCall35",
    "delta50",
    "deltaPut35", "deltaPut25", "deltaPut10",
]

# Position of each delta bucket on the smile x-axis ("Call Delta").
DELTA_X_MAP = {
    "deltaCall10": 10, "deltaCall25": 25, "deltaCall35": 35,
    "delta50": 50,
    "deltaPut35": 65, "deltaPut25": 75, "deltaPut10": 90,
}

# Target BS delta for each bucket (puts are negative).
_DELTA_TARGET = {
    "deltaCall10": 0.10, "deltaCall25": 0.25, "deltaCall35": 0.35,
    "deltaPut35": -0.35, "deltaPut25": -0.25, "deltaPut10": -0.10,
}

_EXPIRY_RE = re.compile(r"^(\d{1,2})([A-Z]{3})(\d{2})$")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}

# option_vols_by_dte() re-parses the whole chain and re-runs the BS delta
# search for every bucket/DTE. Every one of atm_iv_by_dte, iv_by_dte_for_delta
# and current_smile funnels through it, and a single "Load all" pass calls
# those a dozen-plus times per asset in the space of a couple of seconds —
# same underlying chain, same answer, recomputed from scratch each time.
# Cache it for as long as the least-fresh input (get_index_price, ttl=10s)
# is considered current, so a render pass shares one computation without
# masking a genuinely new chain on the next user-triggered refresh.
_VOLS_CACHE: dict[str, tuple[float, dict]] = {}
_VOLS_TTL = 10.0


def acfg(asset: str) -> dict:
    return ASSET_CONFIG.get(asset, ASSET_CONFIG["BTC"])


def finite(v: Any) -> bool:
    """True when v converts to a finite float."""
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def parse_strike(tok: Any) -> float | None:
    """Deribit USDC-linear names use 'd' as the decimal point ('6d4' -> 6.4)."""
    try:
        return float(str(tok).replace("d", "."))
    except (TypeError, ValueError):
        return None


def parse_expiry_date(token: str):
    """'26SEP25' -> date(2025, 9, 26).  None when unparseable."""
    m = _EXPIRY_RE.match(str(token).upper())
    if not m:
        return None
    day, mon, yy = m.groups()
    month = _MONTHS.get(mon)
    if not month:
        return None
    try:
        return datetime(2000 + int(yy), month, int(day), tzinfo=timezone.utc).date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Black-Scholes (r = 0, matching the FalconX bot)
# ---------------------------------------------------------------------------

def bs_delta(spot: float, strike: float, t_years: float, sigma: float,
             is_call: bool) -> float:
    """BS delta with r = 0.  Call ~ +0.25, put ~ -0.25."""
    try:
        if t_years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
            return 0.0
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * t_years) / (
            sigma * math.sqrt(t_years))
        return float(norm.cdf(d1)) if is_call else float(norm.cdf(d1) - 1.0)
    except (ValueError, ZeroDivisionError):
        return 0.0


def bs_price(spot: float, strike: float, t_years: float, sigma: float,
             is_call: bool) -> float:
    """BS price with r = 0, in the same units as spot."""
    if t_years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return max(0.0, (spot - strike) if is_call else (strike - spot))
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if is_call:
        return float(spot * norm.cdf(d1) - strike * norm.cdf(d2))
    return float(strike * norm.cdf(-d2) - spot * norm.cdf(-d1))


def implied_vol(price: float, spot: float, strike: float, t_years: float,
                is_call: bool, lo: float = 1e-4, hi: float = 5.0) -> float:
    """Invert BS for sigma by bisection.  NaN when the price is not arbitrage-free."""
    if not (finite(price) and finite(spot) and finite(strike)) or t_years <= 0:
        return float("nan")
    intrinsic = max(0.0, (spot - strike) if is_call else (strike - spot))
    if price < intrinsic - 1e-9 or price <= 0:
        return float("nan")
    upper_bound = spot if is_call else strike
    if price >= upper_bound:
        return float("nan")
    f_lo = bs_price(spot, strike, t_years, lo, is_call) - price
    f_hi = bs_price(spot, strike, t_years, hi, is_call) - price
    if f_lo * f_hi > 0:
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = bs_price(spot, strike, t_years, mid, is_call) - price
        if abs(f_mid) < 1e-10:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))


# ---------------------------------------------------------------------------
# Live chain -> smile by delta
# ---------------------------------------------------------------------------

def parse_chain(asset: str) -> tuple[float | None, dict[int, list[tuple]]]:
    """
    Group the live option chain by DTE.

    Returns ``(spot, {dte: [(strike, iv_decimal, 'C'|'P', instrument), ...]})``.
    """
    cfg = acfg(asset)
    spot = deribit.get_index_price(cfg["index"])
    chain = deribit.get_option_chain(cfg["deribit_ccy"], "option")
    if not chain or not spot:
        return spot, {}

    prefix = cfg["deribit_prefix"] + "-"
    today = datetime.now(timezone.utc).date()
    by_dte: dict[int, list[tuple]] = {}

    for row in chain:
        instr = row.get("instrument_name") or ""
        if cfg["style"] == "linear" and not instr.startswith(prefix):
            continue
        parts = instr.split("-")
        if len(parts) < 4:
            continue
        exp_date = parse_expiry_date(parts[1])
        if exp_date is None or exp_date < today:
            continue
        strike = parse_strike(parts[2])
        mark_iv = row.get("mark_iv")
        if strike is None or mark_iv is None:
            continue
        try:
            iv_d = float(mark_iv) / 100.0        # Deribit mark_iv is percent
        except (TypeError, ValueError):
            continue
        if not finite(iv_d) or iv_d <= 0:
            continue
        opt_type = "C" if parts[-1].upper().startswith("C") else "P"
        dte = (exp_date - today).days
        by_dte.setdefault(dte, []).append((strike, iv_d, opt_type, instr))

    return spot, by_dte


def option_vols_by_dte(asset: str) -> dict[str, dict[int, float]]:
    """
    Live IV by delta bucket and DTE, in decimal. Cached for ``_VOLS_TTL``
    seconds per asset — see the comment on ``_VOLS_CACHE`` above.

    ``{"atm": {dte: iv}, "call25": {...}, "put25": {...},
       "call10": ..., "put10": ..., "call35": ..., "put35": ...}``
    """
    hit = _VOLS_CACHE.get(asset)
    if hit is not None and (time.time() - hit[0]) < _VOLS_TTL:
        return hit[1]
    out = _option_vols_by_dte_uncached(asset)
    _VOLS_CACHE[asset] = (time.time(), out)
    return out


def _option_vols_by_dte_uncached(asset: str) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {
        k: {} for k in
        ("atm", "call10", "call25", "call35", "put10", "put25", "put35")
    }
    spot, by_dte = parse_chain(asset)
    if not spot or not by_dte:
        return out

    bucket_field = {
        "deltaCall10": "call10", "deltaCall25": "call25", "deltaCall35": "call35",
        "deltaPut10": "put10", "deltaPut25": "put25", "deltaPut35": "put35",
    }

    for dte, options in by_dte.items():
        if not options:
            continue
        t_years = max(dte, 0.0) / 365.0
        # ATM = strike nearest the index price.
        atm = min(options, key=lambda x: abs(x[0] - spot))
        out["atm"][dte] = atm[1]
        if t_years <= 0:
            continue
        calls = [(s, iv) for s, iv, tp, _ in options if tp == "C"]
        puts = [(s, iv) for s, iv, tp, _ in options if tp == "P"]
        for key, field in bucket_field.items():
            target = _DELTA_TARGET[key]
            pool = calls if target > 0 else puts
            if not pool:
                continue
            is_call = target > 0
            best = min(pool, key=lambda x: abs(
                bs_delta(spot, x[0], t_years, x[1], is_call) - target))
            out[field][dte] = best[1]
    return out


def atm_iv_by_dte(asset: str) -> dict[int, float]:
    return option_vols_by_dte(asset).get("atm", {})


def iv_by_dte_for_delta(asset: str, delta_key: str) -> dict[int, float]:
    """Live IV keyed by DTE for one Amberdata-style delta key."""
    vols = option_vols_by_dte(asset)
    field = {
        "delta50": "atm",
        "deltaCall10": "call10", "deltaCall25": "call25", "deltaCall35": "call35",
        "deltaPut10": "put10", "deltaPut25": "put25", "deltaPut35": "put35",
    }.get(delta_key)
    return dict(vols.get(field, {})) if field else {}


def future_prices_by_dte(asset: str) -> dict[int, float]:
    """Mark price of each DATED future keyed by DTE (perpetual excluded)."""
    cfg = acfg(asset)
    chain = deribit.get_option_chain(cfg["deribit_ccy"], "future")
    out: dict[int, float] = {}
    if not chain:
        return out
    today = datetime.now(timezone.utc).date()
    prefix = cfg["deribit_prefix"] + "-"
    for row in chain:
        instr = (row.get("instrument_name") or "").upper()
        mark = row.get("mark_price")
        if not instr or mark is None:
            continue
        if "PERPETUAL" in instr or "-PERP" in instr:
            continue
        if cfg["style"] == "linear" and not instr.startswith(prefix.upper()):
            continue
        parts = instr.split("-")
        if len(parts) < 2:
            continue
        exp_date = parse_expiry_date(parts[1])
        if exp_date is None or exp_date < today:
            continue
        dte = (exp_date - today).days
        if dte >= 0:
            try:
                out[dte] = float(mark)
            except (TypeError, ValueError):
                continue
    return out


def listed_expiries(asset: str) -> list[int]:
    """Sorted DTEs of listed Deribit option expiries.  Falls back to [7, 30, 90]."""
    cfg = acfg(asset)
    today = datetime.now(timezone.utc).date()
    dtes: set[int] = set()

    if cfg["style"] == "linear":
        instruments = deribit.get_instruments(cfg["deribit_ccy"], "option") or []
        prefix = cfg["deribit_prefix"] + "-"
        for inst in instruments:
            name = inst.get("instrument_name") or ""
            if not name.startswith(prefix):
                continue
            parts = name.split("-")
            if len(parts) < 2:
                continue
            exp = parse_expiry_date(parts[1])
            if exp and exp >= today:
                dtes.add((exp - today).days)
    else:
        stamps = deribit.get_expirations(cfg["deribit_ccy"], "option") or []
        for s in stamps:
            try:
                d = datetime.fromtimestamp(int(s) / 1000.0, tz=timezone.utc).date()
            except (TypeError, ValueError, OSError):
                continue
            if d >= today:
                dtes.add((d - today).days)
        if not dtes:
            _, by_dte = parse_chain(asset)
            dtes = {d for d in by_dte if d >= 0}

    return sorted(dtes) if dtes else [7, 30, 90]


def resolve_dte(asset: str, target_days: int) -> int:
    """Nearest listed Deribit expiry (in DTE) to the requested tenor."""
    expiries = listed_expiries(asset)
    if not expiries:
        return int(target_days)
    return int(min(expiries, key=lambda d: abs(d - int(target_days))))


# ---------------------------------------------------------------------------
# Term-structure maths
# ---------------------------------------------------------------------------

def interp_at_dte(vals: dict[float, float], target: float,
                  flat_outside: bool = True) -> float:
    """
    Linear-in-DTE interpolation of IV.

    Mirrors the FalconX bracketing helper: exact hit wins, otherwise linear
    between the neighbours.  Outside the range it holds the nearest endpoint
    flat (``flat_outside``) or returns NaN.
    """
    cols = sorted(c for c in vals if finite(vals[c]))
    if not cols:
        return float("nan")
    t = float(target)
    for c in cols:
        if abs(c - t) < 0.01:
            return float(vals[c])
    left = [c for c in cols if c < t]
    right = [c for c in cols if c > t]
    if left and right:
        cl, cr = max(left), min(right)
        if cr == cl:
            return float(vals[cl])
        w = (t - cl) / (cr - cl)
        return float(vals[cl] + (vals[cr] - vals[cl]) * w)
    if not flat_outside:
        return float("nan")
    return float(vals[max(left)]) if left else float(vals[min(right)])


def forward_vol(v_short: float, t_short: float,
                v_long: float, t_long: float) -> float:
    """
    Forward volatility between two tenors (decimal in, decimal out).

        sigma_fwd = sqrt( (sigma_long^2 * T_long - sigma_short^2 * T_short)
                          / (T_long - T_short) )

    Negative forward variance is clamped to zero.
    """
    if not (finite(v_short) and finite(v_long)):
        return float("nan")
    if t_long <= t_short:
        return float("nan")
    fwd_var = (v_long ** 2 * t_long - v_short ** 2 * t_short) / (t_long - t_short)
    return float(np.sqrt(max(0.0, fwd_var)))


def build_forward_vol_term_series(asset: str, delta_key: str,
                                  vol_label: str) -> dict | None:
    """
    Spot and forward vol term structure for one delta bucket.

    Returns dict with ``dtes``, ``expiry_dates``, ``x_cat``, ``expiry_labels``,
    ``spot_vols`` and ``forward_vols`` (both in vol points / percent).
    """
    iv_map = iv_by_dte_for_delta(asset, delta_key)
    if not iv_map:
        return None
    vals = {float(k): float(v) for k, v in iv_map.items() if finite(v)}
    if len(vals) < 2:
        return None

    dtes_all = listed_expiries(asset) or sorted(int(c) for c in vals)
    today = datetime.now(timezone.utc).date()

    dtes: list[int] = []
    spot_vols: list[float] = []
    forward_vols: list[float] = []
    expiry_dates: list = []

    for i, dte in enumerate(dtes_all):
        v = interp_at_dte(vals, float(dte))
        if not finite(v) or v <= 0:
            continue
        t = dte / 365.0
        spot_vols.append(v * 100.0)
        if not dtes:
            fwd = v                                  # first tenor: forward == spot
        else:
            dte_prev = dtes[-1]
            v_prev = interp_at_dte(vals, float(dte_prev))
            t_prev = dte_prev / 365.0
            fwd = forward_vol(v_prev, t_prev, v, t)
            if not finite(fwd):
                fwd = v
        forward_vols.append(float(fwd) * 100.0)
        dtes.append(int(dte))
        expiry_dates.append(today + timedelta(days=int(dte)))

    if len(expiry_dates) < 2:
        return None

    x_cat = [d.strftime("%d-%b") for d in expiry_dates]
    return {
        "vol_label": vol_label,
        "dtes": dtes,
        "expiry_dates": expiry_dates,
        "x_cat": x_cat,
        "expiry_labels": tidy_labels(x_cat),
        "spot_vols": spot_vols,
        "forward_vols": forward_vols,
    }


def tidy_labels(labels: list[str], max_labels: int = 10) -> list[str]:
    """Thin a label list to at most ``max_labels`` entries, blanking the rest."""
    if not labels or len(labels) <= max_labels:
        return list(labels) if labels else []
    step = max(1, (len(labels) + max_labels - 1) // max_labels)
    return [labels[i] if i % step == 0 else "" for i in range(len(labels))]


def current_smile(asset: str, dte: int) -> dict[int, float]:
    """
    Current smile for one expiry as ``{x_position: iv_percent}`` where the x
    positions follow ``DELTA_X_MAP`` (10 = 10d call ... 90 = 10d put).
    """
    vols = option_vols_by_dte(asset)
    field_for_x = {
        10: "call10", 25: "call25", 35: "call35",
        50: "atm", 65: "put35", 75: "put25", 90: "put10",
    }
    out: dict[int, float] = {}
    for x, field in field_for_x.items():
        series = {float(k): float(v) for k, v in vols.get(field, {}).items()
                  if finite(v)}
        if not series:
            continue
        v = interp_at_dte(series, float(dte))
        if finite(v) and v > 0:
            out[x] = v * 100.0
    return out


def dvol_now(asset: str) -> float | None:
    """Latest DVOL close as a percentage.  None for assets without DVOL."""
    cfg = acfg(asset)
    if not cfg.get("has_dvol"):
        return None
    import time as _t
    end_ms = int(_t.time() * 1000)
    df = deribit.get_dvol(cfg["deribit_ccy"], resolution=60,
                          start_ms=end_ms - 6 * 3600 * 1000, end_ms=end_ms)
    if df is None or df.empty or "close" not in df:
        return None
    try:
        v = float(df["close"].dropna().iloc[-1])
    except (IndexError, ValueError, TypeError):
        return None
    return v * 100.0 if v < 5 else v


def perp_mark(asset: str) -> float | None:
    t = deribit.get_ticker(acfg(asset)["perp"])
    if not t:
        return None
    v = t.get("mark_price")
    return float(v) if finite(v) else None


def clear_cache():
    """Drop the cached per-asset vol computation (see ``_VOLS_CACHE``)."""
    _VOLS_CACHE.clear()
