"""
Time Based Realized Vol — Deribit perpetual OHLC, close-to-close +
range-based annualized RV across hedging frequencies, rolling windows,
quality controls, decision matrices, Telegram report.

Ported from the exodus-analytics "Time Based Realized Vol" page. The
original compared Binance spot/perp klines across ~28 assets; this version
is Deribit-only (project-wide constraint — see CLAUDE.md) and fetches
BTC-PERPETUAL / ETH-PERPETUAL via lib/deribit.py's tradingview endpoint,
always on the perpetual (no spot leg — Deribit's public API has no spot
market to compare against). Scope is BTC/ETH for now; more Deribit-listed
perps (SOL, HYPE) can be added once their tradingview history depth is
verified.

Core question this page answers: if you hedge every X minutes/hours, what
realized vol do you actually experience? Seven estimators (close-to-close,
Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, Bipower variation,
and a Realized Kernel / Pre-averaging proxy blend) are computed per
hedging frequency and lookback window, averaged into a Composite RV used
for long/short-gamma ranking, with an EWMA next-window forecast, data
completeness tracking, and outlier controls (winsorize/drop via robust
z-score). Crypto annualization uses 365 days, matching lib/vol_math.py
elsewhere in this app.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import concurrent.futures
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.deribit import get_tradingview_ohlc, clear_cache
from lib.constants import ASSET_CONFIG, ASSET_COLORS, PLOTLY_LAYOUT
from lib.telegram import send_message, send_photo, is_configured
from lib import fx_style

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Time Based Realized Vol",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Time Based Realized Vol")
st.caption(
    "Compare **annualized realized vol across hedging frequencies** for BTC/ETH "
    "perpetuals. Use the table and rolling charts to choose intervals for "
    "**long gamma (seek higher RV)** or **short gamma (seek lower RV)**. "
    "Crypto annualization uses **365** days."
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Focused on BTC/ETH perps for now — both have deep tradingview history on
# Deribit. SOL/HYPE could be added once their candle history depth at short
# resolutions (1m/5m) has been checked.
TBRV_ASSETS = ("BTC", "ETH")

LOOKBACK_OPTIONS = ("1d", "3d", "7d", "14d", "21d", "30d")
# 4h (resolution "240") deliberately excluded: confirmed via live testing
# that Deribit's tradingview endpoint returns no candles for BTC-PERPETUAL at
# that resolution ("No candles returned for BTC-PERPETUAL (4h)" across every
# lookback window) — not a transient outage. 2h and 12h already bracket that
# gap, so this frequency is dropped rather than worked around.
INTERVAL_OPTIONS = ("1m", "5m", "10m", "15m", "30m", "1h", "2h", "12h", "1d")

# Deribit's get_tradingview_chart_data resolution parameter: minutes as a
# string, or "1D" for daily.
INTERVAL_TO_RESOLUTION: Dict[str, str] = {
    "1m": "1", "5m": "5", "10m": "10", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "12h": "720", "1d": "1D",
}

# Long hedge intervals whose last completed bar can be many hours stale (the
# close-to-close return only updates at the daily/12h boundary). For these,
# optionally keep the in-progress Deribit candle so the latest price
# snapshot is reflected in realized vol instead of waiting for bar close.
LIVE_INTERVALS = frozenset({"12h", "1d"})

MATRIX_METRIC_OPTIONS = {
    "Composite RV (%)": "Composite RV (%)",
    "Close-to-close RV (%)": "Latest ann. RV (%)",
    "Parkinson RV (%)": "Parkinson RV (%)",
    "Garman-Klass RV (%)": "Garman-Klass RV (%)",
    "Rogers-Satchell RV (%)": "Rogers-Satchell RV (%)",
    "Yang-Zhang RV (%)": "Yang-Zhang RV (%)",
    "Bipower RV (%)": "Bipower RV (%)",
    "RK/PA RV (%)": "RK/PA RV (%)",
}

# Deribit's tradingview endpoint isn't documented with a hard per-call candle
# cap; chunk conservatively (same order of magnitude as Binance's 1000-candle
# cap) so a 1m/30d request (43,200 candles) doesn't risk a silently truncated
# single response.
_MAX_BARS_PER_CALL = 1000

_CACHE_SCHEMA_VERSION = "v1_deribit_perp"
_SNAP_KEY = f"_tbrv_last_good_snapshots_{_CACHE_SCHEMA_VERSION}"
_UI_CACHE_KEY = f"_tbrv_ui_fetch_cache_{_CACHE_SCHEMA_VERSION}"
_UI_CACHE_TTL_SEC = 60.0


def _show(fig: go.Figure, **kwargs) -> None:
    """Standard render: watermark + shared theme, matching this app's other pages."""
    fx_style.add_watermark(fig)
    st.plotly_chart(fx_style.apply_theme(fig), width="stretch", **kwargs)


# ---------------------------------------------------------------------------
# Interval / lookback helpers
# ---------------------------------------------------------------------------

def interval_to_ms(interval: str) -> int:
    m = {
        "1m": 60_000, "5m": 300_000, "10m": 600_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
        "12h": 43_200_000, "1d": 86_400_000,
    }
    if interval not in m:
        raise ValueError(f"Unknown interval: {interval}")
    return m[interval]


def lookback_to_ms(lookback: str) -> int:
    m = {
        "1d": 24 * 3_600_000, "3d": 3 * 24 * 3_600_000, "7d": 7 * 24 * 3_600_000,
        "14d": 14 * 24 * 3_600_000, "21d": 21 * 24 * 3_600_000, "30d": 30 * 24 * 3_600_000,
    }
    if lookback not in m:
        raise ValueError(f"Unknown lookback: {lookback}")
    return m[lookback]


def rolling_window_ms_for_lookback(lookback: str) -> int:
    """Default rolling chart window in ms — scales with the selected lookback."""
    m = {
        "1d": 4 * 3_600_000, "3d": 12 * 3_600_000, "7d": 24 * 3_600_000,
        "14d": 48 * 3_600_000, "21d": 72 * 3_600_000, "30d": 96 * 3_600_000,
    }
    return m.get(lookback, 4 * 3_600_000)


def sort_intervals_short_to_long(intervals: List[str]) -> List[str]:
    return sorted(intervals, key=lambda x: interval_to_ms(x))


def sort_lookbacks_short_to_long(lookbacks: List[str]) -> List[str]:
    return sorted(lookbacks, key=lambda x: lookback_to_ms(x))


def expected_bar_count(start_ms: int, end_ms: int, bar_ms: int) -> int:
    """Expected candles with open_time in [start_ms, end_ms)."""
    if bar_ms <= 0 or end_ms <= start_ms:
        return 0
    return int((end_ms - start_ms) // bar_ms)


def floor_to_bar_ms(ts_ms: int, bar_ms: int) -> int:
    return (ts_ms // bar_ms) * bar_ms


def should_use_live_bar(interval: str, enabled: bool) -> bool:
    """Whether to keep the in-progress (not-yet-closed) candle for this interval."""
    return bool(enabled) and interval in LIVE_INTERVALS


# ---------------------------------------------------------------------------
# Small shared utilities
# ---------------------------------------------------------------------------

def contrast_text_color(value: float, vmin: float, vmax: float) -> str:
    if not np.isfinite(value) or not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return "black"
    norm = (value - vmin) / (vmax - vmin)
    if norm <= 0.18 or norm >= 0.82:
        return "white"
    return "black"


def build_heatmap_annotations(
    x_vals: List[Any], y_vals: List[Any], z: np.ndarray, decimals: int = 2,
) -> List[Dict[str, Any]]:
    z_arr = np.asarray(z, dtype=float)
    finite = z_arr[np.isfinite(z_arr)]
    vmin, vmax = (float(np.min(finite)), float(np.max(finite))) if len(finite) else (0.0, 1.0)
    out: List[Dict[str, Any]] = []
    for yi, y in enumerate(y_vals):
        for xi, x in enumerate(x_vals):
            val = z_arr[yi, xi]
            # Blank cells mean the estimator was undefined for that frequency
            # (insufficient sample or a duplicated/forward-filled series), not zero.
            label = f"{val:.{decimals}f}" if np.isfinite(val) else "N/A"
            out.append(dict(
                x=x, y=y, text=label, showarrow=False,
                font=dict(color=contrast_text_color(float(val), vmin, vmax), size=11),
            ))
    return out


def safe_rank_int(series: pd.Series, ascending: bool) -> pd.Series:
    """Rank with nullable integer output — keeps NA ranks as <NA>."""
    return series.rank(ascending=ascending, method="min").astype("Int64")


def safe_pick_row(df: pd.DataFrame, metric: str, ascending: bool) -> pd.Series:
    """Best row by metric, ignoring non-finite values when possible."""
    if df.empty:
        return pd.Series(dtype=object)
    s = pd.to_numeric(df[metric], errors="coerce")
    valid = df[s.notna()].copy()
    if not valid.empty:
        return valid.sort_values(metric, ascending=ascending).iloc[0]
    return df.iloc[0]


# ---------------------------------------------------------------------------
# Data fetching — Deribit tradingview OHLC, paginated + resilient
# ---------------------------------------------------------------------------

def _resolution_bar_ms(resolution: str) -> int:
    if resolution == "1D":
        return 86_400_000
    return int(resolution) * 60_000


def _build_chunk_plan(start_ms: int, end_ms: int, bar_ms: int) -> List[Tuple[int, int]]:
    """Time-range chunk boundaries covering [start_ms, end_ms) — each chunk is
    an independent request, so unlike Binance-style cursor pagination these
    don't need to be issued in order."""
    chunk_span_ms = _MAX_BARS_PER_CALL * bar_ms
    plan: List[Tuple[int, int]] = []
    cursor = int(start_ms)
    while cursor < end_ms:
        chunk_end = min(cursor + chunk_span_ms, end_ms)
        plan.append((cursor, chunk_end))
        cursor = chunk_end
    return plan


def _fetch_chunks_parallel(
    chunk_specs: List[Tuple[str, str, int, int]], max_workers: int = 16,
) -> Dict[Tuple[str, str, int, int], Optional[pd.DataFrame]]:
    """
    Fetch every (instrument, resolution, start_ms, end_ms) chunk across every
    hedging frequency in one shared thread pool. Each chunk is an independent
    Deribit request — lib.deribit.py's own rate limiter (a module-level lock,
    10 req/s) already serializes actual request starts across threads, so
    this doesn't relax that; it just stops paying one request's network
    latency, then the next's, in sequence. A cold load with all 10
    frequencies selected at a 30d lookback is ~68 chunk requests — sequential,
    that's ~20-25s of pure network wait; pooled, it's bound by the shared
    rate limiter at ~7s.
    """
    results: Dict[Tuple[str, str, int, int], Optional[pd.DataFrame]] = {}
    if not chunk_specs:
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(chunk_specs))) as ex:
        future_map = {ex.submit(get_tradingview_ohlc, *spec): spec for spec in chunk_specs}
        for fut in concurrent.futures.as_completed(future_map):
            spec = future_map[fut]
            try:
                results[spec] = fut.result()
            except Exception:
                results[spec] = None
    return results


def _assemble_klines(frames: List[Optional[pd.DataFrame]], bar_ms: int) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    valid = [f for f in frames if f is not None and not f.empty]
    if not valid:
        return pd.DataFrame(columns=cols)
    out = pd.concat(valid, ignore_index=True)
    out = out.rename(columns={"timestamp": "open_time_dt"})
    # Cast explicitly to millisecond resolution before extracting the epoch
    # integer: pandas' default resolution for `pd.to_datetime(..., unit="ms")`
    # has changed across versions (datetime64[ns] historically, datetime64[ms]
    # in newer pandas), and `.astype("int64")` returns whatever the *current*
    # resolution's epoch unit is — forcing "datetime64[ms]" first keeps this
    # correct regardless of which pandas resolves at runtime.
    out["open_time"] = out["open_time_dt"].to_numpy().astype("datetime64[ms]").astype("int64")
    out["close_time"] = out["open_time"] + bar_ms - 1
    out = out.drop(columns=["open_time_dt"])
    out = out.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time").reset_index(drop=True)
    return out[cols]


@dataclass
class FetchResult:
    df: pd.DataFrame
    from_cache_fallback: bool
    error_message: Optional[str]
    fetched_at_utc: datetime
    resolved_symbol: str


def _snapshots() -> Dict[str, Any]:
    if _SNAP_KEY not in st.session_state:
        st.session_state[_SNAP_KEY] = {}
    return st.session_state[_SNAP_KEY]


def _cache_key(asset: str, lookback: str, interval: str) -> str:
    return f"{_CACHE_SCHEMA_VERSION}|{asset.upper()}|{lookback}|{interval}"


def load_klines_for_intervals(
    asset: str, lookback: str, intervals: List[str],
    force_refresh: bool = False, include_live_bar_fn=None,
) -> Dict[str, FetchResult]:
    """
    Fetch klines for several hedging frequencies at once, with resilience: on
    total failure for a given (asset, lookback, interval), fall back to the
    last successfully-fetched snapshot for that combo rather than failing the
    whole comparison. Every frequency's Deribit chunk requests are dispatched
    into one shared thread pool (see `_fetch_chunks_parallel`) so the cold-load
    wait is the slowest frequency's chunk count under the shared rate limit,
    not the sum of all frequencies' chunk counts run one after another.
    """
    cfg = ASSET_CONFIG[asset]
    instrument = cfg["perp"]
    now_ms = int(time.time() * 1000)
    now_utc = datetime.now(timezone.utc)
    snaps = _snapshots()
    if _UI_CACHE_KEY not in st.session_state:
        st.session_state[_UI_CACHE_KEY] = {}
    ui_cache: Dict[str, Any] = st.session_state[_UI_CACHE_KEY]

    results: Dict[str, FetchResult] = {}
    pending: Dict[str, Dict[str, Any]] = {}
    chunk_specs: List[Tuple[str, str, int, int]] = []

    for interval in intervals:
        resolution = INTERVAL_TO_RESOLUTION[interval]
        bar_ms = interval_to_ms(interval)
        lb_ms = lookback_to_ms(lookback)
        end_ms = floor_to_bar_ms(now_ms, bar_ms)
        start_ms = end_ms - lb_ms
        include_live = bool(include_live_bar_fn(interval)) if include_live_bar_fn else False
        fetch_end_ms = now_ms if include_live else end_ms
        filter_end_ms = (end_ms + bar_ms) if include_live else end_ms

        key = _cache_key(asset, lookback, interval)
        if include_live:
            key = f"{key}|live"

        if force_refresh and key in snaps:
            del snaps[key]
        if force_refresh and key in ui_cache:
            del ui_cache[key]
        if not force_refresh:
            ent = ui_cache.get(key)
            if ent and (time.time() - float(ent.get("ts", 0))) < _UI_CACHE_TTL_SEC:
                results[interval] = FetchResult(
                    df=ent["df"].copy(), from_cache_fallback=bool(ent.get("from_cache_fallback", False)),
                    error_message=ent.get("error_message"),
                    fetched_at_utc=ent.get("fetched_at_utc", now_utc), resolved_symbol=instrument,
                )
                continue

        plan = _build_chunk_plan(start_ms, fetch_end_ms, bar_ms)
        specs = [(instrument, resolution, c0, c1) for c0, c1 in plan]
        pending[interval] = {
            "specs": specs, "bar_ms": bar_ms, "start_ms": start_ms, "filter_end_ms": filter_end_ms, "key": key,
        }
        chunk_specs.extend(specs)

    fetched = _fetch_chunks_parallel(chunk_specs) if chunk_specs else {}

    for interval, plan in pending.items():
        key = plan["key"]
        try:
            df = _assemble_klines([fetched.get(s) for s in plan["specs"]], plan["bar_ms"])
            if df.empty:
                raise RuntimeError(f"No candles returned for {instrument} ({interval})")
            df = df[(df["open_time"] >= plan["start_ms"]) & (df["open_time"] < plan["filter_end_ms"])].copy()
            if df.empty:
                raise RuntimeError(f"No candles in requested window for {instrument} ({interval})")
            snaps[key] = {"df": df.copy(), "fetched_at": now_utc}
            res = FetchResult(df=df, from_cache_fallback=False, error_message=None,
                               fetched_at_utc=now_utc, resolved_symbol=instrument)
            ui_cache[key] = {
                "df": df.copy(), "ts": time.time(), "from_cache_fallback": False,
                "error_message": None, "fetched_at_utc": now_utc,
            }
        except Exception as e:
            prev = snaps.get(key)
            if prev and isinstance(prev.get("df"), pd.DataFrame) and not prev["df"].empty:
                res = FetchResult(
                    df=prev["df"].copy(), from_cache_fallback=True, error_message=str(e),
                    fetched_at_utc=prev.get("fetched_at", now_utc), resolved_symbol=instrument,
                )
                ui_cache[key] = {
                    "df": res.df.copy(), "ts": time.time(), "from_cache_fallback": True,
                    "error_message": res.error_message, "fetched_at_utc": res.fetched_at_utc,
                }
            else:
                res = FetchResult(df=pd.DataFrame(), from_cache_fallback=False, error_message=str(e),
                                   fetched_at_utc=now_utc, resolved_symbol=instrument)
        results[interval] = res

    return results


def load_or_fetch_klines(
    asset: str, lookback: str, interval: str,
    force_refresh: bool = False, include_live_bar: bool = False,
) -> FetchResult:
    """Single-frequency convenience wrapper around `load_klines_for_intervals`
    (used by the suspicious-interval recompute guardrail, which only ever
    needs to refetch one frequency at a time)."""
    res = load_klines_for_intervals(
        asset, lookback, [interval], force_refresh=force_refresh,
        include_live_bar_fn=lambda _iv: include_live_bar,
    )
    return res[interval]


# ---------------------------------------------------------------------------
# Math: gaps, returns, outliers, annualized vol (exchange-agnostic)
# ---------------------------------------------------------------------------

def fill_gaps_and_bad_closes(df: pd.DataFrame, bar_ms: int) -> Tuple[pd.DataFrame, int, int]:
    """Forward-fill missing bars on the time grid; treat close <= 0 as invalid and ffill."""
    if df.empty or len(df) < 2:
        return df.copy(), 0, 0
    d = df.sort_values("open_time").copy()
    n_zero_rep = int((d["close"] <= 0).sum())
    d["close"] = d["close"].where(d["close"] > 0)
    if bar_ms > 0:
        d["open_time"] = (pd.to_numeric(d["open_time"], errors="coerce") // bar_ms) * bar_ms
    d = d.dropna(subset=["open_time"])
    d["open_time"] = d["open_time"].astype(np.int64)
    d = d.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time")
    if len(d) < 2:
        return d.reset_index(drop=True), 0, n_zero_rep
    first_t = int(d["open_time"].iloc[0])
    last_t = int(d["open_time"].iloc[-1])
    origin = floor_to_bar_ms(first_t, bar_ms)
    full_index = np.arange(origin, last_t + 1, bar_ms, dtype=np.int64)
    base = pd.DataFrame({"open_time": full_index})
    cols = [c for c in ("open_time", "open", "high", "low", "close", "volume", "close_time") if c in d.columns]
    merged = base.merge(d[cols], on="open_time", how="left")
    missing_before_ffill = int(merged["close"].isna().sum())
    merged["close"] = merged["close"].ffill()
    merged = merged.dropna(subset=["close"])
    if "high" in merged.columns:
        merged["high"] = merged["high"].fillna(merged["close"])
        merged["low"] = merged["low"].fillna(merged["close"])
        merged["open"] = merged["open"].fillna(merged["close"])
    return merged, missing_before_ffill, n_zero_rep


def log_returns(closes: np.ndarray) -> np.ndarray:
    x = np.asarray(closes, dtype=float)
    if len(x) < 2:
        return np.array([])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(x[1:] / x[:-1])
    return np.where(np.isfinite(r), r, np.nan)


def sample_std(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan")
    return float(np.std(v, ddof=1))


# Share of exact-zero finite returns above which the close series is treated
# as duplicated/forward-filled (degenerate) rather than a genuinely quiet market.
_DEGENERATE_ZERO_RETURN_RATIO = 0.95


def returns_are_degenerate(returns: np.ndarray) -> bool:
    """
    A forward-filled (gap-filled) close series produces runs of *exactly*
    zero log returns. An extreme share of them (or zero sample std) means
    the return-based estimators would read ~0 from duplicated data, not
    genuinely low realized vol — such series should report N/A, not 0.00.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return False
    n_zero = int(np.sum(np.abs(r) < 1e-12))
    if n_zero / len(r) >= _DEGENERATE_ZERO_RETURN_RATIO:
        return True
    return float(np.std(r, ddof=1)) <= 1e-12


def periods_per_year(interval: str) -> float:
    bar_ms = interval_to_ms(interval)
    year_ms = 365.0 * 24 * 3600 * 1000
    return year_ms / float(bar_ms)


def annualize_vol(std: float, interval: str) -> float:
    if not np.isfinite(std):
        return float("nan")
    return std * sqrt(periods_per_year(interval))


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if window <= 0 or n < window:
        return out
    for i in range(window - 1, n):
        w = arr[i - window + 1: i + 1]
        w = w[np.isfinite(w)]
        if len(w) > 0:
            out[i] = float(np.mean(w))
    return out


def _rolling_from_var_contrib(var_contrib: np.ndarray, window: int, ppy: float) -> np.ndarray:
    rm = _rolling_mean(var_contrib, window)
    out = np.full(len(rm), np.nan)
    mask = np.isfinite(rm) & (rm >= 0)
    out[mask] = np.sqrt(rm[mask] * ppy)
    return out


def rolling_annualized_vol(returns: np.ndarray, window: int, interval: str) -> np.ndarray:
    """Returns array aligned to `returns`; first window-1 entries are nan."""
    n = len(returns)
    out = np.full(n, np.nan)
    if window < 2 or n < window:
        return out
    ppy = periods_per_year(interval)
    for i in range(window - 1, n):
        w = returns[i - window + 1: i + 1]
        w = w[np.isfinite(w)]
        if len(w) < 2:
            continue
        out[i] = float(np.std(w, ddof=1)) * sqrt(ppy)
    return out


def apply_outlier_policy(
    returns: np.ndarray, mode: Literal["none", "winsorize", "drop"], z_threshold: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    r = np.asarray(returns, dtype=float)
    mask = np.isfinite(r)
    stats = {"n_total": int(len(r)), "n_finite": int(mask.sum()), "n_filtered": 0, "mode": mode}
    if mode == "none" or len(r) < 4:
        return r, stats
    med = float(np.nanmedian(r))
    mad = float(np.nanmedian(np.abs(r - med)))
    sigma = 1.4826 * mad if mad > 1e-12 else float(np.nanstd(r))
    if not np.isfinite(sigma) or sigma <= 0:
        return r, stats
    z = np.abs((r - med) / sigma)
    outlier = mask & (z > z_threshold)
    stats["n_filtered"] = int(outlier.sum())
    if mode == "drop":
        r2 = r.copy()
        r2[outlier] = np.nan
        return r2, stats
    if mode == "winsorize":
        lo = med - z_threshold * sigma
        hi = med + z_threshold * sigma
        r2 = np.clip(r, lo, hi)
        r2[~mask] = np.nan
        return r2, stats
    return r, stats


def compute_advanced_estimators(dfc: pd.DataFrame, interval: str, roll_win: int) -> Dict[str, Any]:
    """Compute all 7 realized-vol estimators + rolling annualized series."""
    ppy = periods_per_year(interval)
    eps = 1e-12

    o = pd.to_numeric(dfc["open"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(dfc["high"], errors="coerce").to_numpy(dtype=float)
    l = pd.to_numeric(dfc["low"], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(dfc["close"], errors="coerce").to_numpy(dtype=float)
    t_bars = pd.to_datetime(dfc["open_time"].values, unit="ms", utc=True)
    t_ret = pd.to_datetime(dfc["open_time"].values[1:], unit="ms", utc=True)

    valid_ohlc = (o > eps) & (h > eps) & (l > eps) & (c > eps)

    # Close-to-close
    r = log_returns(c)
    close_var_contrib = np.where(np.isfinite(r), r ** 2, np.nan)
    returns_degenerate = returns_are_degenerate(r)

    # Parkinson (HL)
    hl = np.full(len(c), np.nan)
    idx = valid_ohlc & (h >= l)
    hl[idx] = np.log(h[idx] / np.maximum(l[idx], eps))
    park_var = (hl ** 2) / (4.0 * np.log(2.0))

    # Garman-Klass (OHLC)
    oc = np.full(len(c), np.nan)
    oc[valid_ohlc] = np.log(c[valid_ohlc] / np.maximum(o[valid_ohlc], eps))
    gk_var = 0.5 * (hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (oc ** 2)
    gk_var = np.where(gk_var >= 0, gk_var, np.nan)

    # Rogers-Satchell (OHLC, drift robust)
    rs_var = np.full(len(c), np.nan)
    rs_var[valid_ohlc] = (
        np.log(h[valid_ohlc] / np.maximum(c[valid_ohlc], eps)) * np.log(h[valid_ohlc] / np.maximum(o[valid_ohlc], eps))
        + np.log(l[valid_ohlc] / np.maximum(c[valid_ohlc], eps)) * np.log(l[valid_ohlc] / np.maximum(o[valid_ohlc], eps))
    )
    rs_var = np.where(rs_var >= 0, rs_var, np.nan)

    # Yang-Zhang (overnight/open jump + close-open + RS)
    n = len(c)
    yz_var = np.full(n, np.nan)
    if n >= 3:
        oc_jump = np.log(np.maximum(o[1:], eps) / np.maximum(c[:-1], eps))
        co_ret = np.log(np.maximum(c, eps) / np.maximum(o, eps))
        sigma_o2 = np.nanvar(oc_jump, ddof=1) if np.isfinite(oc_jump).sum() >= 2 else np.nan
        sigma_c2 = np.nanvar(co_ret, ddof=1) if np.isfinite(co_ret).sum() >= 2 else np.nan
        sigma_rs = np.nanmean(rs_var)
        if np.isfinite(sigma_o2) and np.isfinite(sigma_c2) and np.isfinite(sigma_rs):
            k = 0.34 / (1.34 + (n + 1.0) / (max(n - 1.0, 1.0)))
            yz = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs
            if yz >= 0:
                yz_var[:] = yz

    # Bipower variation (continuous component proxy, returns-based)
    mu1 = np.sqrt(2.0 / np.pi)
    bp_var = np.full(len(r), np.nan)
    if len(r) >= 2:
        abs_r = np.abs(r)
        bp_var[1:] = (1.0 / (mu1 ** 2)) * abs_r[1:] * abs_r[:-1]

    # Realized kernel / pre-averaging proxy (returns-based)
    rk_var = np.nan
    pa_var = np.nan
    rkpa_var = np.full(len(r), np.nan)
    r_f = r[np.isfinite(r)]
    if len(r_f) >= 5:
        x = r_f - np.nanmean(r_f)
        n_r = len(x)
        h_bw = max(1, min(20, int(np.sqrt(n_r))))
        gamma0 = float(np.mean(x * x))
        rk = gamma0
        for lag in range(1, h_bw + 1):
            cov = float(np.mean(x[lag:] * x[:-lag]))
            weight = 1.0 - lag / (h_bw + 1.0)
            rk += 2.0 * weight * cov
        rk_var = max(rk, 0.0)

        m = max(2, int(np.sqrt(n_r)))
        g = np.array([min(j / m, 1.0 - j / m) for j in range(1, m)], dtype=float)
        if len(g) >= 1:
            n_pre = n_r - m
            if n_pre > 0:
                # Vectorized pre-averaging: each pre[i] = sum(g * x[i+1:i+m]), a
                # sliding dot-product of x against g — equivalent to the original
                # per-i Python loop but computed as one matrix-vector product.
                windows = np.lib.stride_tricks.sliding_window_view(x[1:], m - 1)
                pre = windows[:n_pre] @ g
            else:
                pre = np.asarray([], dtype=float)
            psi2 = float(np.sum(g * g) / m)
            if psi2 > 0 and len(pre) > 0:
                pa_var = max(float(np.sum(pre * pre) / (len(pre) * psi2 * m)), 0.0)
        if np.isfinite(rk_var) and np.isfinite(pa_var):
            rkpa_var[:] = 0.5 * (rk_var + pa_var)
        elif np.isfinite(rk_var):
            rkpa_var[:] = rk_var
        elif np.isfinite(pa_var):
            rkpa_var[:] = pa_var

    def _ann_from_contrib(vc: np.ndarray) -> float:
        m_ = np.nanmean(vc) if len(vc) else np.nan
        if not np.isfinite(m_) or m_ < 0:
            return float("nan")
        return float(np.sqrt(m_ * ppy) * 100.0)

    latest = {
        "close": _ann_from_contrib(close_var_contrib),
        "parkinson": _ann_from_contrib(park_var),
        "garman_klass": _ann_from_contrib(gk_var),
        "rogers_satchell": _ann_from_contrib(rs_var),
        "yang_zhang": _ann_from_contrib(yz_var),
        "bipower": _ann_from_contrib(bp_var),
        "rk_pa": _ann_from_contrib(rkpa_var),
    }
    if returns_degenerate:
        # Return-driven estimators are meaningless on a duplicated close
        # series; per-bar OHLC range estimators (Parkinson/GK/RS) are unaffected.
        for k_ in ("close", "yang_zhang", "bipower", "rk_pa"):
            latest[k_] = float("nan")
    finite_latest = [v for v in latest.values() if np.isfinite(v)]
    latest["composite"] = float(np.mean(finite_latest)) if finite_latest else float("nan")

    roll = {
        "close": rolling_annualized_vol(r, roll_win, interval),
        "parkinson": _rolling_from_var_contrib(park_var, roll_win, ppy),
        "garman_klass": _rolling_from_var_contrib(gk_var, roll_win, ppy),
        "rogers_satchell": _rolling_from_var_contrib(rs_var, roll_win, ppy),
        "yang_zhang": _rolling_from_var_contrib(yz_var, roll_win, ppy),
        "bipower": _rolling_from_var_contrib(bp_var, roll_win, ppy),
        "rk_pa": _rolling_from_var_contrib(rkpa_var, roll_win, ppy),
    }

    return {
        "latest_pct": latest,
        "rolling": roll,
        "degenerate_returns": bool(returns_degenerate),
        "time_by_method": {
            "close": t_ret, "parkinson": t_bars, "garman_klass": t_bars,
            "rogers_satchell": t_bars, "yang_zhang": t_bars, "bipower": t_ret, "rk_pa": t_ret,
        },
    }


def estimate_next_rv_forecast_pct(rolling_series: np.ndarray) -> Tuple[float, float]:
    vals = rolling_series[np.isfinite(rolling_series)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[-1] * 100.0), 0.0
    alpha = 0.35
    ewma = vals[0]
    for v in vals[1:]:
        ewma = alpha * v + (1.0 - alpha) * ewma
    resid = vals - ewma
    sigma = float(np.nanstd(resid, ddof=1)) if len(vals) > 2 else float(np.nanstd(vals))
    return float(ewma * 100.0), float(sigma * 100.0)


# ---------------------------------------------------------------------------
# Per-interval analysis
# ---------------------------------------------------------------------------

def slice_df_for_lookback(df: pd.DataFrame, lookback: str, interval: str, include_live_bar: bool = False) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    bar_ms = interval_to_ms(interval)
    lb_ms = lookback_to_ms(lookback)
    now_ms = int(time.time() * 1000)
    end_ms = floor_to_bar_ms(now_ms, bar_ms)
    start_ms = end_ms - lb_ms
    upper_ms = (end_ms + bar_ms) if include_live_bar else end_ms
    return df[(df["open_time"] >= start_ms) & (df["open_time"] < upper_ms)].copy()


def analyze_interval_from_df(
    df_in: pd.DataFrame, lookback: str, interval: str,
    outlier_mode: Literal["none", "winsorize", "drop"], z_threshold: float, min_returns: int,
    fetch_meta: Optional[Dict[str, Any]] = None, include_live_bar: bool = False,
) -> Dict[str, Any]:
    bar_ms = interval_to_ms(interval)
    lb_ms = lookback_to_ms(lookback)
    now_ms = int(time.time() * 1000)
    end_ms = floor_to_bar_ms(now_ms, bar_ms)
    start_ms = end_ms - lb_ms
    exp_n = expected_bar_count(start_ms, end_ms, bar_ms)
    df = slice_df_for_lookback(df_in, lookback, interval, include_live_bar=include_live_bar)

    nan_row = {
        "headline_pct": float("nan"), "composite_pct": float("nan"), "parkinson_pct": float("nan"),
        "garman_klass_pct": float("nan"), "rogers_satchell_pct": float("nan"), "yang_zhang_pct": float("nan"),
        "bipower_pct": float("nan"), "rk_pa_pct": float("nan"), "rolling_mean_pct": float("nan"),
        "rolling_median_pct": float("nan"), "rolling_p90_pct": float("nan"),
        "forecast_next_pct": float("nan"), "forecast_sigma_pct": float("nan"),
    }

    if df.empty:
        err = "No data in requested lookback window"
        if fetch_meta and fetch_meta.get("error_message"):
            err = str(fetch_meta.get("error_message"))
        return {
            "interval": interval, "error": err, "expected": exp_n, "received": 0,
            "completeness_pct": 0.0,
            "from_cache_fallback": bool((fetch_meta or {}).get("from_cache_fallback", False)),
            "fetched_at_utc": (fetch_meta or {}).get("fetched_at_utc", datetime.now(timezone.utc)),
            "resolved_symbol": str((fetch_meta or {}).get("resolved_symbol", "")),
            **nan_row,
        }

    dfc, n_gap_fill, n_zero_rep = fill_gaps_and_bad_closes(df, bar_ms)
    closes = dfc["close"].to_numpy(dtype=float)
    lr = log_returns(closes)
    lr_proc, ostats = apply_outlier_policy(lr, outlier_mode, z_threshold)
    finite_lr = lr_proc[np.isfinite(lr_proc)]
    std_full = sample_std(finite_lr)
    ann_full = annualize_vol(std_full, interval)
    headline_pct = float(ann_full * 100.0) if np.isfinite(ann_full) else float("nan")

    rw_ms = rolling_window_ms_for_lookback(lookback)
    roll_win = max(2, int(rw_ms // bar_ms))
    est = compute_advanced_estimators(dfc, interval, roll_win)
    if est.get("degenerate_returns"):
        headline_pct = float("nan")
    rv_series = est["rolling"]["close"]
    rv_f = rv_series[np.isfinite(rv_series)]
    rolling_mean = float(np.nanmean(rv_f) * 100.0) if len(rv_f) else float("nan")
    rolling_p90 = float(np.nanpercentile(rv_f, 90) * 100.0) if len(rv_f) else float("nan")
    rolling_median = float(np.nanmedian(rv_f) * 100.0) if len(rv_f) else float("nan")
    forecast_next, forecast_sigma = estimate_next_rv_forecast_pct(rv_series)

    # Completed bars only, so an in-progress live bar can't push completeness above 100%.
    n_completed = int((df["open_time"] < end_ms).sum())
    live_bar_used = bool(include_live_bar and (int(df["open_time"].max()) >= end_ms))

    return {
        "interval": interval, "error": None, "expected": exp_n, "received": n_completed,
        "completeness_pct": (100.0 * n_completed / exp_n) if exp_n else 0.0,
        "live_bar_used": live_bar_used,
        "from_cache_fallback": bool((fetch_meta or {}).get("from_cache_fallback", False)),
        "fetched_at_utc": (fetch_meta or {}).get("fetched_at_utc", datetime.now(timezone.utc)),
        "resolved_symbol": str((fetch_meta or {}).get("resolved_symbol", "")),
        "gap_fill": n_gap_fill, "zero_replacements": n_zero_rep, "outlier_stats": ostats,
        "min_returns_pass": len(finite_lr) >= int(min_returns),
        "headline_pct": headline_pct,
        "parkinson_pct": est["latest_pct"]["parkinson"],
        "garman_klass_pct": est["latest_pct"]["garman_klass"],
        "rogers_satchell_pct": est["latest_pct"]["rogers_satchell"],
        "yang_zhang_pct": est["latest_pct"]["yang_zhang"],
        "bipower_pct": est["latest_pct"]["bipower"],
        "rk_pa_pct": est["latest_pct"]["rk_pa"],
        "composite_pct": est["latest_pct"]["composite"],
        "degenerate_returns": bool(est.get("degenerate_returns", False)),
        "rolling_mean_pct": rolling_mean, "rolling_median_pct": rolling_median, "rolling_p90_pct": rolling_p90,
        "forecast_next_pct": forecast_next, "forecast_sigma_pct": forecast_sigma,
        "rolling_vol_pct": rv_series,
        "rolling_time": pd.to_datetime(dfc["open_time"].values[1:], unit="ms", utc=True),
        "price_time": pd.to_datetime(dfc["open_time"].values[1:], unit="ms", utc=True),
        "price_series": closes[1:],
        "rolling_by_method": est["rolling"],
        "rolling_time_by_method": est["time_by_method"],
        "returns_processed": lr_proc,
    }


def maybe_recompute_suspicious_interval(
    asset: str, lookback: str, interval: str, analysis: Dict[str, Any],
    outlier_mode: Literal["none", "winsorize", "drop"], z_threshold: float, min_returns: int,
) -> Dict[str, Any]:
    """
    Guardrail against a rare corrupted slice: a collapsed/duplicated close
    series, a non-finite composite, or a composite implausibly small relative
    to close-to-close. Forces a fresh exact-lookback refetch/recompute when
    triggered — defense-in-depth against a genuine upstream data gap.
    """
    if analysis.get("error"):
        return analysis
    comp = pd.to_numeric(pd.Series([analysis.get("composite_pct")]), errors="coerce").iloc[0]
    close_cc = pd.to_numeric(pd.Series([analysis.get("headline_pct")]), errors="coerce").iloc[0]
    suspicious = (
        bool(analysis.get("degenerate_returns"))
        or (not np.isfinite(comp))
        or (np.isfinite(close_cc) and np.isfinite(comp) and comp < 0.35 * close_cc)
    )
    if not suspicious:
        return analysis

    fr = load_or_fetch_klines(asset, lookback, interval, force_refresh=True)
    return analyze_interval_from_df(
        fr.df, lookback=lookback, interval=interval,
        outlier_mode=outlier_mode, z_threshold=z_threshold, min_returns=min_returns,
        fetch_meta={
            "from_cache_fallback": fr.from_cache_fallback,
            "fetched_at_utc": fr.fetched_at_utc,
            "error_message": fr.error_message,
            "resolved_symbol": fr.resolved_symbol,
        },
    )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def style_summary_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def _na_white(col: pd.Series) -> np.ndarray:
        return np.where(col.isna(), "background-color: white; color: black;", "")

    def _contrast_col(col: pd.Series) -> np.ndarray:
        vals = pd.to_numeric(col, errors="coerce")
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            return np.array([""] * len(col), dtype=object)
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        out = []
        for v in vals:
            out.append("" if not np.isfinite(v) else f"color: {contrast_text_color(float(v), vmin, vmax)}; font-weight: 600;")
        return np.array(out, dtype=object)

    rv_cols = [
        "Latest ann. RV (%)", "Composite RV (%)", "Parkinson RV (%)", "Garman-Klass RV (%)",
        "Rogers-Satchell RV (%)", "Yang-Zhang RV (%)", "Bipower RV (%)", "RK/PA RV (%)",
        "Rolling mean RV (%)", "Rolling p90 RV (%)",
    ]
    return (
        df.style
        .format({
            **{c: "{:.2f}" for c in rv_cols},
            "Forecast next RV (%)": "{:.2f}", "Forecast sigma (%)": "{:.2f}",
            "Completeness (%)": "{:.1f}",
        }, na_rep="N/A")
        .background_gradient(subset=rv_cols, cmap="RdYlGn")
        .background_gradient(subset=["Completeness (%)"], cmap="Blues")
        .background_gradient(subset=["Long-gamma rank (higher RV)"], cmap="Greens_r")
        .background_gradient(subset=["Short-gamma rank (lower RV)"], cmap="Oranges_r")
        .apply(_contrast_col, axis=0, subset=rv_cols + ["Completeness (%)", "Long-gamma rank (higher RV)", "Short-gamma rank (lower RV)"])
        .apply(_na_white, axis=0)
        .set_properties(
            subset=rv_cols + ["Forecast next RV (%)", "Forecast sigma (%)", "Completeness (%)",
                               "Long-gamma rank (higher RV)", "Short-gamma rank (lower RV)"],
            **{"font-weight": "600"},
        )
    )


def decision_table_display_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["Expected candles", "Received candles", "Used cache fallback"]
    existing = [c for c in drop_cols if c in summary_df.columns]
    return summary_df.drop(columns=existing)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def fig_interval_bar(summary_df: pd.DataFrame, asset: str, lookback: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary_df["Interval"], y=summary_df["Latest ann. RV (%)"],
        text=summary_df["Latest ann. RV (%)"].map(lambda x: f"{x:.1f}%"),
        textposition="outside", name="Latest ann. RV",
        marker_color=ASSET_COLORS.get(asset, "#1f77b4"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=430,
        title=f"{asset} — realized vol by hedging interval ({lookback})",
        xaxis_title="Hedging frequency", yaxis_title="Annualized RV (%)",
    )
    return fig


def fig_completeness_bar(summary_df: pd.DataFrame, asset: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=summary_df["Interval"], y=summary_df["Expected candles"],
                          name="Expected", marker_color="#8fa3bf", opacity=0.7))
    fig.add_trace(go.Bar(x=summary_df["Interval"], y=summary_df["Received candles"],
                          name="Received", marker_color="#1f77b4", opacity=0.9))
    fig.update_layout(
        **PLOTLY_LAYOUT, barmode="group", height=420,
        title=f"{asset} — data completeness by frequency",
        xaxis_title="Hedging frequency", yaxis_title="Candle count",
    )
    return fig


def fig_rolling_multi(analyses: List[Dict[str, Any]], asset: str, lookback: str) -> go.Figure:
    fig = go.Figure()
    marker_symbol_map = {
        "1m": "circle", "5m": "square", "10m": "diamond", "15m": "triangle-up",
        "30m": "triangle-down", "1h": "cross", "2h": "x",
        "12h": "hexagon", "1d": "pentagon",
    }
    for item in analyses:
        if item.get("error"):
            continue
        t = fx_style.to_local(np.asarray(item["rolling_time"]))
        y = np.asarray(item["rolling_vol_pct"], dtype=float) * 100.0
        mask = np.isfinite(y)
        t, y = np.asarray(t)[mask], y[mask]
        if len(y) == 0:
            continue
        interval_label = str(item["interval"])
        interval_ms = interval_to_ms(interval_label)
        snap_stride = max(1, int((30 * 60 * 1000) / max(interval_ms, 1)))
        fig.add_trace(go.Scatter(
            x=t, y=y, mode="lines", name=interval_label, legendgroup=interval_label,
            line=dict(width=2.6),
        ))
        fig.add_trace(go.Scatter(
            x=t[::snap_stride], y=y[::snap_stride], mode="markers",
            name=f"{interval_label} hedges", showlegend=False, legendgroup=interval_label,
            marker=dict(size=8, opacity=1.0, symbol=marker_symbol_map.get(interval_label, "circle"),
                        line=dict(width=1.2, color="white")),
            hovertemplate=f"{interval_label} hedge point<br>%{{x}}<br>RV: %{{y:.2f}}%<extra></extra>",
        ))

    # Overlay close price on a secondary axis (prefer 1h for readability).
    overlay_item = None
    for p in ("1h", "30m", "15m", "5m", "1m"):
        by_interval = {str(i.get("interval")): i for i in analyses if not i.get("error")}
        if p in by_interval:
            overlay_item = by_interval[p]
            break
    if overlay_item is None:
        valid = [i for i in analyses if not i.get("error")]
        if valid:
            overlay_item = sorted(valid, key=lambda i: interval_to_ms(str(i.get("interval", "1h"))))[0]

    if overlay_item is not None:
        pt = fx_style.to_local(np.asarray(overlay_item.get("price_time", [])))
        pv = np.asarray(overlay_item.get("price_series", []), dtype=float)
        pmask = np.isfinite(pv)
        pt, pv = np.asarray(pt)[pmask], pv[pmask]
        if len(pv) > 0:
            fig.add_trace(go.Scatter(
                x=pt, y=pv, mode="lines", name=f"{asset}-PERPETUAL",
                line=dict(width=2.2, color=ASSET_COLORS.get(asset, "#111111")), opacity=0.85,
                yaxis="y2", hovertemplate="Price context<br>%{x}<br>%{y:.2f}<extra></extra>",
            ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{asset} — rolling annualized RV across hedging frequencies ({lookback})",
        xaxis_title=fx_style.XAXIS_TIME, yaxis=dict(title="Annualized RV (%)"),
        yaxis2=dict(title="Price (USD)", overlaying="y", side="right", showgrid=False),
        xaxis_rangeslider_visible=False, height=680,
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, x=0),
    )
    return fig


def fig_rolling_distribution_box(analyses: List[Dict[str, Any]], asset: str, lookback: str) -> go.Figure:
    fig = go.Figure()
    for item in analyses:
        if item.get("error"):
            continue
        vals = item["rolling_vol_pct"] * 100.0
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        fig.add_trace(go.Box(y=vals, name=item["interval"], boxmean=True))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=470,
        title=f"{asset} — rolling RV distribution by hedging frequency ({lookback})",
        xaxis_title="Hedging frequency", yaxis_title="Annualized RV (%)",
    )
    return fig


def fig_metrics_heatmap(summary_df: pd.DataFrame, asset: str, lookback: str) -> go.Figure:
    cols = ["Latest ann. RV (%)", "Rolling mean RV (%)", "Rolling p90 RV (%)", "Forecast next RV (%)", "Completeness (%)"]
    mat = summary_df[cols].to_numpy(dtype=float).T
    x_vals = list(summary_df["Interval"])
    fig = go.Figure(data=go.Heatmap(z=mat, x=x_vals, y=cols, colorscale="RdYlGn", colorbar=dict(title="Value")))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=430,
        title=f"{asset} — metric heatmap by hedging frequency ({lookback})",
        xaxis_title="Hedging frequency", yaxis_title="Metric",
        annotations=build_heatmap_annotations(x_vals, cols, mat, decimals=2),
    )
    return fig


def fig_interval_scatter(summary_df: pd.DataFrame, asset: str, lookback: str) -> go.Figure:
    x_minutes = summary_df["Interval"].map(lambda s: interval_to_ms(s) / 60_000.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_minutes, y=summary_df["Latest ann. RV (%)"], mode="lines+markers+text",
        text=summary_df["Interval"], textposition="top center", name="Latest ann. RV",
        marker=dict(color=ASSET_COLORS.get(asset, "#1f77b4")),
        line=dict(color=ASSET_COLORS.get(asset, "#1f77b4")),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=430,
        title=f"{asset} — RV vs hedge interval length ({lookback})",
        xaxis=dict(title="Minutes between hedges (log scale)", type="log"),
        yaxis=dict(title="Annualized RV (%)"),
    )
    return fig


def fig_forecast_bar(summary_df: pd.DataFrame, asset: str, lookback: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=summary_df["Interval"], y=summary_df["Forecast next RV (%)"],
                          name="Forecast next RV", marker_color="#9467bd"))
    fig.add_trace(go.Scatter(x=summary_df["Interval"], y=summary_df["Latest ann. RV (%)"],
                              mode="lines+markers", name="Current RV",
                              marker=dict(size=7), line=dict(width=1.2, color="#ff7f0e"), yaxis="y2"))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=440,
        title=f"{asset} — forecast next-window RV by frequency ({lookback})",
        yaxis=dict(title="Forecast next RV (%)"),
        yaxis2=dict(title="Current RV (%)", overlaying="y", side="right"),
    )
    return fig


def fig_cross_lookback_heatmap(
    combined_rows: pd.DataFrame, asset: str,
    metric_col: str = "Latest ann. RV (%)", metric_label: str = "Latest ann. RV (%)",
) -> go.Figure:
    if combined_rows.empty:
        return go.Figure()
    if metric_col not in combined_rows.columns:
        metric_col, metric_label = "Latest ann. RV (%)", "Latest ann. RV (%)"
    pivot = combined_rows.pivot(index="Interval", columns="Lookback", values=metric_col)
    lookback_order = sort_lookbacks_short_to_long(list(pivot.columns))
    pivot = pivot.reindex(index=sort_intervals_short_to_long(list(pivot.index)), columns=lookback_order)
    z = pivot.to_numpy(dtype=float)
    x_vals, y_vals = list(pivot.columns), list(pivot.index)
    fig = go.Figure(data=go.Heatmap(z=z, x=x_vals, y=y_vals, colorscale="RdYlGn", colorbar=dict(title="Ann. RV (%)")))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=460,
        title=f"{asset} — {metric_label} across lookbacks and hedging frequencies",
        xaxis_title="Lookback window", yaxis_title="Hedging frequency",
        yaxis=dict(autorange="reversed"),
        annotations=build_heatmap_annotations(x_vals, y_vals, z, decimals=2),
    )
    return fig


def fig_decision_matrix_3d(
    combined_rows: pd.DataFrame, asset: str,
    metric_col: str = "Composite RV (%)", metric_label: str = "Composite RV (%)",
) -> go.Figure:
    """Interactive 3D decision matrix: x=lookback, y=hedging frequency, z=selected metric."""
    if combined_rows.empty:
        return go.Figure()
    if metric_col not in combined_rows.columns:
        metric_col, metric_label = "Composite RV (%)", "Composite RV (%)"

    lookback_order = sort_lookbacks_short_to_long(list(combined_rows["Lookback"].dropna().unique()))
    interval_order = sort_intervals_short_to_long(list(combined_rows["Interval"].dropna().unique()))
    pivot = combined_rows.pivot(index="Interval", columns="Lookback", values=metric_col).reindex(
        index=interval_order, columns=lookback_order)
    z = pivot.to_numpy(dtype=float)
    x = np.arange(len(lookback_order), dtype=float)
    y = np.arange(len(interval_order), dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x, y=y, z=z, colorscale="RdYlGn", colorbar=dict(title=metric_label),
        showscale=True, opacity=0.9, hovertemplate=f"{metric_label}: " + "%{z:.2f}%<extra></extra>",
    ))

    def _fmt(v: Any, suffix: str = "") -> str:
        try:
            fv = float(v)
            if np.isfinite(fv):
                return f"{fv:.2f}{suffix}"
        except Exception:
            pass
        return "—"

    row_map = {(str(r["Lookback"]), str(r["Interval"])): r for _, r in combined_rows.iterrows()}
    xs, ys, zs, hover = [], [], [], []
    for yi, iv in enumerate(interval_order):
        for xi, lb in enumerate(lookback_order):
            v = z[yi, xi]
            if not np.isfinite(v):
                continue
            row = row_map.get((str(lb), str(iv)))
            h = f"Lookback: {lb}<br>Hedging frequency: {iv}<br>{metric_label}: {_fmt(v, '%')}<br>"
            if row is not None:
                h += (
                    f"Composite RV: {_fmt(row.get('Composite RV (%)'), '%')}<br>"
                    f"Close-close RV: {_fmt(row.get('Latest ann. RV (%)'), '%')}<br>"
                    f"Forecast next RV: {_fmt(row.get('Forecast next RV (%)'), '%')}<br>"
                    f"Completeness: {_fmt(row.get('Completeness (%)'), '%')}<br>"
                    f"Long-gamma rank: {row.get('Long-gamma rank (higher RV)', '—')}<br>"
                    f"Short-gamma rank: {row.get('Short-gamma rank (lower RV)', '—')}"
                )
            xs.append(float(xi)); ys.append(float(yi)); zs.append(float(v)); hover.append(h)
    if xs:
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers+text", text=[f"{v:.1f}" for v in zs],
            textposition="top center", marker=dict(size=4, color=zs, colorscale="RdYlGn", showscale=False),
            hovertext=hover, hovertemplate="%{hovertext}<extra></extra>", name="Matrix points",
        ))

    fig.update_layout(
        title=f"{asset} — 3D decision matrix (lookback × hedging frequency × {metric_label})",
        scene=dict(
            xaxis=dict(title="Lookback", tickmode="array", tickvals=list(range(len(lookback_order))), ticktext=lookback_order),
            yaxis=dict(title="Hedging frequency", tickmode="array", tickvals=list(range(len(interval_order))), ticktext=interval_order),
            zaxis=dict(title=metric_label), aspectmode="cube",
        ),
        height=980, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def render_decision_matrix_3d_png(
    combined_rows: pd.DataFrame, asset: str,
    metric_col: str = "Composite RV (%)", metric_label: str = "Composite RV (%)",
    width: int = 1280, height: int = 860,
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Render the 3D decision matrix as a static PNG via matplotlib for Telegram.
    Plotly's Surface/Scatter3d traces need WebGL, which headless Kaleido
    can't compile on a GPU-less host ("gl-shader: Error compiling vertex
    shader") — matplotlib's mplot3d renders on CPU via the Agg backend instead.
    """
    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
    except Exception as e:
        return None, f"matplotlib 3D backend unavailable: {e}"

    if combined_rows.empty:
        return None, "No data for 3D decision matrix"
    if metric_col not in combined_rows.columns:
        metric_col, metric_label = "Composite RV (%)", "Composite RV (%)"

    lookback_order = sort_lookbacks_short_to_long(list(combined_rows["Lookback"].dropna().unique()))
    interval_order = sort_intervals_short_to_long(list(combined_rows["Interval"].dropna().unique()))
    pivot = combined_rows.pivot(index="Interval", columns="Lookback", values=metric_col).reindex(
        index=interval_order, columns=lookback_order)
    z = pivot.to_numpy(dtype=float)
    if z.size == 0:
        return None, "Empty matrix for 3D decision matrix"

    nx, ny = len(lookback_order), len(interval_order)
    grid_x, grid_y = np.meshgrid(np.arange(nx), np.arange(ny))
    finite = z[np.isfinite(z)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1.0

    try:
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        z_plot = np.where(np.isfinite(z), z, np.nan)
        surf = ax.plot_surface(
            grid_x, grid_y, z_plot, cmap="RdYlGn", vmin=vmin, vmax=vmax,
            edgecolor="0.35", linewidth=0.3, alpha=0.92, antialiased=True,
            rcount=max(ny, 2), ccount=max(nx, 2),
        )
        for yi in range(ny):
            for xi in range(nx):
                v = z[yi, xi]
                if not np.isfinite(v):
                    continue
                ax.scatter(xi, yi, v, color="#111111", s=10, depthshade=False)
                ax.text(xi, yi, v, f"{v:.0f}", fontsize=6.5, ha="center", va="bottom", color="#111111")
        ax.set_xticks(list(np.arange(nx))); ax.set_xticklabels(lookback_order, fontsize=8)
        ax.set_yticks(list(np.arange(ny))); ax.set_yticklabels(interval_order, fontsize=8)
        ax.set_xlabel("Lookback", fontsize=9, labelpad=8)
        ax.set_ylabel("Hedging frequency", fontsize=9, labelpad=10)
        ax.set_zlabel(metric_label, fontsize=9, labelpad=6)
        ax.set_title(f"{asset} — 3D decision matrix\n(lookback × hedging frequency × {metric_label})", fontsize=11)
        ax.view_init(elev=24, azim=-58)
        fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08, label=metric_label)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        data = buf.getvalue()
        return (data, None) if data else (None, "3D render produced empty bytes")
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        return None, f"3D render failed: {e}"


def fig_decision_matrix_2d(
    combined_rows: pd.DataFrame, asset: str,
    metric_col: str = "Composite RV (%)", metric_label: str = "Composite RV (%)",
) -> go.Figure:
    """Static-image fallback for the 3D decision matrix if matplotlib is unavailable."""
    if combined_rows.empty:
        return go.Figure()
    if metric_col not in combined_rows.columns:
        metric_col, metric_label = "Composite RV (%)", "Composite RV (%)"
    lookback_order = sort_lookbacks_short_to_long(list(combined_rows["Lookback"].dropna().unique()))
    interval_order = sort_intervals_short_to_long(list(combined_rows["Interval"].dropna().unique()))
    pivot = combined_rows.pivot(index="Interval", columns="Lookback", values=metric_col).reindex(
        index=interval_order, columns=lookback_order)
    z = pivot.to_numpy(dtype=float)
    fig = go.Figure(data=go.Heatmap(z=z, x=lookback_order, y=interval_order, colorscale="RdYlGn",
                                     colorbar=dict(title=metric_label)))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=520,
        title=f"{asset} — decision matrix (lookback × hedging frequency × {metric_label})",
        xaxis_title="Lookback window", yaxis_title="Hedging frequency",
        yaxis=dict(autorange="reversed"),
        annotations=build_heatmap_annotations(lookback_order, interval_order, z, decimals=2),
    )
    return fig


def fig_estimator_heatmap(summary_df: pd.DataFrame, asset: str, lookback: str) -> go.Figure:
    cols = ["Latest ann. RV (%)", "Parkinson RV (%)", "Garman-Klass RV (%)", "Rogers-Satchell RV (%)",
            "Yang-Zhang RV (%)", "Bipower RV (%)", "RK/PA RV (%)", "Composite RV (%)"]
    mat = summary_df[cols].to_numpy(dtype=float).T
    x_vals = list(summary_df["Interval"])
    fig = go.Figure(data=go.Heatmap(z=mat, x=x_vals, y=cols, colorscale="RdYlGn", colorbar=dict(title="Ann. RV (%)")))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=520,
        title=f"{asset} — estimator comparison heatmap ({lookback})",
        xaxis_title="Hedging frequency", yaxis_title="Estimator",
        annotations=build_heatmap_annotations(x_vals, cols, mat, decimals=2),
    )
    return fig


def fig_estimator_grouped_bars(summary_df: pd.DataFrame, asset: str, lookback: str) -> go.Figure:
    methods = [
        ("Close-close", "Latest ann. RV (%)"), ("Parkinson", "Parkinson RV (%)"),
        ("Garman-Klass", "Garman-Klass RV (%)"), ("Rogers-Satchell", "Rogers-Satchell RV (%)"),
        ("Yang-Zhang", "Yang-Zhang RV (%)"), ("Bipower", "Bipower RV (%)"), ("RK/PA", "RK/PA RV (%)"),
    ]
    fig = go.Figure()
    for name, col in methods:
        fig.add_trace(go.Bar(x=summary_df["Interval"], y=summary_df[col], name=name))
    fig.update_layout(
        **PLOTLY_LAYOUT, barmode="group", height=500,
        title=f"{asset} — realized vol by estimator and frequency ({lookback})",
        xaxis_title="Hedging frequency", yaxis_title="Annualized RV (%)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0),
    )
    return fig


def fig_rolling_methods_for_interval(analysis: Dict[str, Any], asset: str, lookback: str) -> go.Figure:
    method_labels = {
        "close": "Close-close", "parkinson": "Parkinson", "garman_klass": "Garman-Klass",
        "rogers_satchell": "Rogers-Satchell", "yang_zhang": "Yang-Zhang",
        "bipower": "Bipower", "rk_pa": "RK/PA",
    }
    fig = go.Figure()
    rolling = analysis.get("rolling_by_method", {})
    times = analysis.get("rolling_time_by_method", {})
    for key in ("close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang", "bipower", "rk_pa"):
        if key not in rolling:
            continue
        y = np.asarray(rolling[key], dtype=float) * 100.0
        t = times.get(key)
        if t is None or len(y) == 0:
            continue
        fig.add_trace(go.Scatter(x=fx_style.to_local(np.asarray(t)), y=y, mode="lines",
                                  name=method_labels.get(key, key), line=dict(width=1.1)))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=500,
        title=f"{asset} — rolling RV by estimator at {analysis.get('interval')} hedge interval ({lookback})",
        xaxis_title=fx_style.XAXIS_TIME, yaxis_title="Annualized RV (%)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Telegram report
# ---------------------------------------------------------------------------

def send_time_based_rv_report_to_telegram(
    asset: str, lookback: str, summary_df: pd.DataFrame, figures: List[Tuple[go.Figure, str]],
) -> Tuple[bool, str]:
    if summary_df.empty:
        send_message(f"⚠️ <b>Time Based Realized Vol</b>\nNo valid frequency rows for {asset} ({lookback}).")
        return False, "No rows to send"
    metric_col = "Composite RV (%)" if "Composite RV (%)" in summary_df.columns else "Latest ann. RV (%)"
    top = safe_pick_row(summary_df, metric_col, ascending=False)
    low = safe_pick_row(summary_df, metric_col, ascending=True)
    send_message(
        f"📊 <b>Time Based Realized Vol</b>\n"
        f"Asset: <b>{asset}</b>\n"
        f"Lookback: {lookback}\n"
        f"Best for long gamma (highest composite RV): <b>{top['Interval']}</b> ({top[metric_col]:.2f}%)\n"
        f"Best for short gamma (lowest composite RV): <b>{low['Interval']}</b> ({low[metric_col]:.2f}%)\n"
        f"Time: {fx_style.local_now():%Y-%m-%d %H:%M} {fx_style.DISPLAY_TZ_LABEL}"
    )
    ok = True
    for fig, name in figures:
        if fig is None:
            continue
        png = fx_style.fig_to_png(fig)
        if png:
            ok = send_photo(png, caption=name) and ok
        else:
            ok = False
    return ok, "Report sent to Telegram" if ok else "One or more charts failed to send"


def send_decision_matrix_3d_to_telegram(
    combined_rows: pd.DataFrame, asset: str, metric_col: str, metric_label: str, caption: str,
) -> bool:
    png_bytes, err = render_decision_matrix_3d_png(combined_rows, asset, metric_col=metric_col, metric_label=metric_label)
    if err or not png_bytes:
        fig_2d = fig_decision_matrix_2d(combined_rows, asset, metric_col=metric_col, metric_label=metric_label)
        png_2d = fx_style.fig_to_png(fig_2d)
        return bool(png_2d) and send_photo(png_2d, caption=caption)
    return send_photo(png_bytes, caption=caption)


# ---------------------------------------------------------------------------
# Explanatory expanders
# ---------------------------------------------------------------------------

with st.expander("How to interpret this page", expanded=False):
    st.markdown(
        "- **Core question:** if you hedge every `X` minutes/hours, what realized vol do you experience?\n"
        "- **Higher realized vol** generally favors **long gamma** books (more movement captured).\n"
        "- **Lower realized vol** generally favors **short gamma** books (less movement to fund).\n"
        "- **Latest ann. RV (%)** is the most recent full-window estimate for that frequency/lookback.\n"
        "- **Rolling mean / p90 RV** show typical level vs stressed high-vol regime.\n"
        "- **Forecast next RV (%)** is an EWMA-based near-term directional estimate, not a guarantee.\n"
        "- **Completeness (%)** tells you data reliability; lower completeness means lower confidence.\n"
        "- **Advanced estimators** are included: Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, "
        "Bipower variation, and a Realized Kernel / Pre-averaging proxy blend.\n"
        "- **Composite RV (%)** is the cross-estimator mean and is used for robust ranking/recommendations."
    )

with st.expander("Metric glossary (detailed)", expanded=False):
    glossary = pd.DataFrame([
        {"Metric": "Latest ann. RV (%)", "Definition": "Annualized close-to-close realized volatility over the selected lookback and frequency.",
         "How to interpret": "Higher means more observed movement between hedge times. Favors long gamma; hurts short gamma."},
        {"Metric": "Composite RV (%)", "Definition": "Average of active estimators (close, Parkinson, GK, RS, YZ, Bipower, RK/PA) for robustness.",
         "How to interpret": "Primary ranking metric. Reduces dependence on any single estimator's assumptions."},
        {"Metric": "Rolling mean RV (%)", "Definition": "Mean of rolling annualized RV series within the lookback.",
         "How to interpret": "Baseline regime level. Good for typical expected hedging environment."},
        {"Metric": "Rolling p90 RV (%)", "Definition": "90th percentile of rolling annualized RV values.",
         "How to interpret": "Stress/high-vol regime indicator. Useful for worst-decile planning."},
        {"Metric": "Forecast next RV (%)", "Definition": "EWMA projection of near-term RV from rolling series.",
         "How to interpret": "Directional guide only. Use with sigma/confidence and not as certainty."},
        {"Metric": "Forecast sigma (%)", "Definition": "Residual variability around EWMA forecast.",
         "How to interpret": "Higher means noisier forecast; lower confidence in next-window ranking."},
        {"Metric": "Completeness (%)", "Definition": "Received candles divided by expected candles for the window.",
         "How to interpret": "Data-quality confidence. Lower completeness can distort estimator comparison."},
        {"Metric": "Outliers addressed", "Definition": "Number of returns clipped/dropped under selected outlier mode.",
         "How to interpret": "High counts mean distribution tails are heavily modified by preprocessing."},
        {"Metric": "Long-gamma rank", "Definition": "Rank by Composite RV descending.",
         "How to interpret": "Rank 1 is best if objective is maximizing realized movement capture."},
        {"Metric": "Short-gamma rank", "Definition": "Rank by Composite RV ascending.",
         "How to interpret": "Rank 1 is best if objective is minimizing realized movement paid out."},
    ])
    st.dataframe(glossary, width="stretch", hide_index=True)

with st.expander("Estimator glossary (assumptions and caveats)", expanded=False):
    estimator_glossary = pd.DataFrame([
        {"Estimator": "Close-to-close", "Inputs": "Close prices", "Strength": "Simple, widely understood baseline.",
         "Caveat": "Misses intrabar range information."},
        {"Estimator": "Parkinson", "Inputs": "High/Low", "Strength": "Efficient when intrabar range is informative.",
         "Caveat": "Can understate jump-heavy regimes."},
        {"Estimator": "Garman-Klass", "Inputs": "OHLC", "Strength": "Often lower variance than close-close under stable assumptions.",
         "Caveat": "Can be biased with drift/jumps."},
        {"Estimator": "Rogers-Satchell", "Inputs": "OHLC", "Strength": "More drift-robust than GK.",
         "Caveat": "Still sensitive to data quality and jump behavior."},
        {"Estimator": "Yang-Zhang", "Inputs": "OHLC + open jump component", "Strength": "Balanced composite estimator in many settings.",
         "Caveat": "In 24/7 crypto, overnight interpretation is less literal."},
        {"Estimator": "Bipower variation", "Inputs": "Returns", "Strength": "Continuous-variation proxy; useful for jump diagnostics.",
         "Caveat": "Not a complete jump model by itself."},
        {"Estimator": "RK/PA (proxy blend)", "Inputs": "High-frequency returns", "Strength": "More noise-robust at very high frequency.",
         "Caveat": "Proxy implementation; not a full academic realized-kernel stack."},
    ])
    st.dataframe(estimator_glossary, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Hedging comparison setup")
    prev_asset = str(st.session_state.get("_tbrv_asset_pref", "BTC")).upper()
    asset_idx = TBRV_ASSETS.index(prev_asset) if prev_asset in TBRV_ASSETS else 0
    asset = st.selectbox("Asset", TBRV_ASSETS, index=asset_idx)
    st.session_state["_tbrv_asset_pref"] = asset
    st.caption("Overall summaries include all lookbacks.")

    compare_intervals = st.multiselect(
        "Hedging frequencies to compare", INTERVAL_OPTIONS, default=list(INTERVAL_OPTIONS),
    )
    use_live_bar = st.checkbox(
        "Use latest price for in-progress 12h/1d bar", value=True,
        help=(
            "For 12h and 1d frequencies, include the current not-yet-closed candle "
            "so realized vol reflects the latest price instead of freezing until "
            "the next bar close."
        ),
    )
    objective = st.radio(
        "Recommendation objective",
        options=["Long gamma (maximize RV)", "Short gamma (minimize RV)"], index=0,
    )
    matrix_metric_label = st.selectbox(
        "3D matrix realized-vol type", options=list(MATRIX_METRIC_OPTIONS.keys()), index=0,
        help="Controls which realized-vol estimator is plotted on the 3D decision matrix z-axis.",
    )
    matrix_metric_col = MATRIX_METRIC_OPTIONS[matrix_metric_label]

    st.markdown("---")
    st.subheader("Quality controls")
    outlier_mode = st.selectbox(
        "Outlier handling", ["none", "winsorize", "drop"], index=0,
        help="Uses robust z-score (median + MAD). Winsorize clips extremes; drop sets them to NaN.",
    )
    z_threshold = st.slider("Robust z threshold", 2.0, 12.0, 6.0, 0.5)
    min_returns = st.number_input(
        "Minimum returns for headline RV", min_value=5, max_value=500, value=30,
        help="If fewer finite returns remain after filtering, headline RV shows a warning.",
    )
    st.markdown("---")
    refresh = st.button("Refresh data", width="stretch")

if not compare_intervals:
    st.warning("Select at least one hedging frequency.")
    st.stop()
compare_intervals = sort_intervals_short_to_long(list(compare_intervals))
lookbacks_to_run = list(LOOKBACK_OPTIONS)

col_a, col_b = st.columns(2)
with col_a:
    hard_refresh = st.button("🔄 Hard refresh (clear cache)", width="stretch")
with col_b:
    send_tg = st.button(
        "📤 Send to Telegram", width="stretch", type="primary", disabled=not is_configured(),
    )
if not is_configured():
    st.caption("Telegram not configured — set credentials in secrets.toml or environment to enable sending.")

if hard_refresh:
    try:
        st.cache_data.clear()
    except Exception:
        pass
    clear_cache()
    st.session_state.pop(_SNAP_KEY, None)
    st.session_state.pop(_UI_CACHE_KEY, None)
    st.rerun()

st.markdown("### Detailed charts lookbacks")
st.caption("Select one or more lookbacks to render in the detailed section below.")
detail_lookbacks = st.multiselect(
    "Choose detailed lookbacks", LOOKBACK_OPTIONS, default=["7d"], label_visibility="collapsed",
)
if not detail_lookbacks:
    st.warning("Select at least one detailed charts lookback.")
    st.stop()

# ---------------------------------------------------------------------------
# Data loading — fetch once per interval at max lookback, slice for each lookback
# ---------------------------------------------------------------------------

with st.spinner("Loading comparison dataset from Deribit across selected frequencies…"):
    lookback_data: Dict[str, Dict[str, Any]] = {}
    lookback_errors: Dict[str, List[str]] = {}
    max_lookback = max(lookbacks_to_run, key=lookback_to_ms)
    base_fetch: Dict[str, FetchResult] = load_klines_for_intervals(
        asset, max_lookback, compare_intervals, force_refresh=refresh,
        include_live_bar_fn=lambda iv: should_use_live_bar(iv, use_live_bar),
    )

    for lb in lookbacks_to_run:
        analyses = []
        for iv in compare_intervals:
            fr = base_fetch[iv]
            analyses.append(analyze_interval_from_df(
                fr.df, lookback=lb, interval=iv,
                outlier_mode=outlier_mode, z_threshold=float(z_threshold), min_returns=int(min_returns),
                include_live_bar=should_use_live_bar(iv, use_live_bar),
                fetch_meta={
                    "from_cache_fallback": fr.from_cache_fallback,
                    "fetched_at_utc": fr.fetched_at_utc,
                    "error_message": fr.error_message,
                    "resolved_symbol": fr.resolved_symbol,
                },
            ))
            analyses[-1] = maybe_recompute_suspicious_interval(
                asset, lb, iv, analyses[-1], outlier_mode, float(z_threshold), int(min_returns),
            )
        ok_analyses = sorted([x for x in analyses if not x.get("error")], key=lambda x: interval_to_ms(x["interval"]))
        bad_analyses = [x for x in analyses if x.get("error")]
        lookback_errors[lb] = [f"{x['interval']}: {x['error']}" for x in bad_analyses]
        if not ok_analyses:
            continue
        summary_df = pd.DataFrame([
            {
                "Lookback": lb, "Interval": x["interval"],
                "Latest ann. RV (%)": x["headline_pct"], "Composite RV (%)": x["composite_pct"],
                "Parkinson RV (%)": x["parkinson_pct"], "Garman-Klass RV (%)": x["garman_klass_pct"],
                "Rogers-Satchell RV (%)": x["rogers_satchell_pct"], "Yang-Zhang RV (%)": x["yang_zhang_pct"],
                "Bipower RV (%)": x["bipower_pct"], "RK/PA RV (%)": x["rk_pa_pct"],
                "Rolling mean RV (%)": x["rolling_mean_pct"], "Rolling p90 RV (%)": x["rolling_p90_pct"],
                "Forecast next RV (%)": x["forecast_next_pct"], "Forecast sigma (%)": x["forecast_sigma_pct"],
                "Completeness (%)": x["completeness_pct"], "Expected candles": x["expected"],
                "Received candles": x["received"],
                "Used cache fallback": "Yes" if x["from_cache_fallback"] else "No",
                "Outliers addressed": int(x["outlier_stats"].get("n_filtered", 0)),
                "Gap/ffill bars": int(x["gap_fill"]),
                "Min sample pass": "Yes" if x["min_returns_pass"] else "No",
            }
            for x in ok_analyses
        ])
        summary_df["Long-gamma rank (higher RV)"] = safe_rank_int(summary_df["Composite RV (%)"], ascending=False)
        summary_df["Short-gamma rank (lower RV)"] = safe_rank_int(summary_df["Composite RV (%)"], ascending=True)
        lookback_data[lb] = {"analyses": ok_analyses, "summary": summary_df}

if not lookback_data:
    st.error("No successful datasets for any lookback window. Deribit may be unreachable right now.")
    st.stop()

_live_intervals = [iv for iv in compare_intervals if should_use_live_bar(iv, use_live_bar)]
if _live_intervals:
    st.caption(
        "🟢 Live bar enabled for " + ", ".join(_live_intervals) +
        ": the in-progress candle uses the latest price so realized vol "
        "tracks current moves instead of freezing until the next bar close."
    )

for lb in lookbacks_to_run:
    errs = lookback_errors.get(lb) or []
    if errs:
        st.warning(f"{lb} had excluded frequencies: {'; '.join(errs)}")

# ---------------------------------------------------------------------------
# All-lookback overview
# ---------------------------------------------------------------------------

all_rows = pd.concat([v["summary"] for v in lookback_data.values()], ignore_index=True)
fig_cross = fig_cross_lookback_heatmap(all_rows, asset, metric_col=matrix_metric_col, metric_label=matrix_metric_label)
fig_decision_3d = fig_decision_matrix_3d(all_rows, asset, metric_col=matrix_metric_col, metric_label=matrix_metric_label)

rec_rows = []
for lb in lookbacks_to_run:
    if lb not in lookback_data:
        continue
    sdf = lookback_data[lb]["summary"]
    best_for_long = safe_pick_row(sdf, "Composite RV (%)", ascending=False)
    best_for_short = safe_pick_row(sdf, "Composite RV (%)", ascending=True)
    best_forecast = safe_pick_row(sdf, "Forecast next RV (%)", ascending=False)
    if objective == "Long gamma (maximize RV)":
        rec, reason = best_for_long, "highest composite RV across estimators"
    else:
        rec, reason = best_for_short, "lowest composite RV across estimators"
    rec_rows.append({
        "Lookback": lb, "Recommended interval": rec["Interval"], "Reason": reason,
        "Current RV (%)": rec["Latest ann. RV (%)"], "Composite RV (%)": rec["Composite RV (%)"],
        "Forecast next RV (%)": rec["Forecast next RV (%)"],
        "Long-gamma best": f"{best_for_long['Interval']} ({best_for_long['Composite RV (%)']:.2f}%)",
        "Short-gamma best": f"{best_for_short['Interval']} ({best_for_short['Composite RV (%)']:.2f}%)",
        "Forecast leader": f"{best_forecast['Interval']} ({best_forecast['Forecast next RV (%)']:.2f}%)",
    })

rec_df = pd.DataFrame(rec_rows)
st.markdown("### All-lookback overview")
_show(fig_cross)
st.caption(
    f"Heatmap reading: each cell is {matrix_metric_label} for a specific "
    "lookback + hedging frequency. Compare across rows (frequency effect) and columns (lookback effect)."
)
_show(fig_decision_3d)
st.caption(
    f"3D decision matrix: x=lookback, y=hedging frequency, z={matrix_metric_label}. "
    "Hover any point for detailed context (composite, forecast, completeness, and ranks)."
)
st.markdown("### Recommendations by lookback")
st.dataframe(rec_df, width="stretch", hide_index=True)
st.caption(
    "Recommendation table: choose row by lookback, then use `Recommended interval` based on selected objective. "
    "`Forecast leader` highlights the highest next-window forecasted RV."
)

# ---------------------------------------------------------------------------
# Detailed per-lookback sections
# ---------------------------------------------------------------------------

detail_candidates = [x for x in sort_lookbacks_short_to_long(list(detail_lookbacks)) if x in lookback_data]
if not detail_candidates:
    detail_candidates = [x for x in lookbacks_to_run if x in lookback_data]

for lb in detail_candidates:
    summary_df = lookback_data[lb]["summary"]
    ok_analyses = lookback_data[lb]["analyses"]

    st.markdown(f"## Detailed view: {lb}")
    st.markdown(f"#### Decision table ({lb})")
    st.caption(
        "Color guide: warmer colors in RV columns = higher vol; rank columns directly map to "
        "long-gamma (higher is better) and short-gamma (lower is better) decisions."
    )
    st.dataframe(style_summary_table(decision_table_display_df(summary_df)), width="stretch", hide_index=True)

    fig_bar = fig_interval_bar(summary_df, asset, lb)
    fig_roll_multi = fig_rolling_multi(ok_analyses, asset, lb)
    fig_box = fig_rolling_distribution_box(ok_analyses, asset, lb)
    fig_comp = fig_completeness_bar(summary_df, asset)
    fig_heat = fig_metrics_heatmap(summary_df, asset, lb)
    fig_scatter = fig_interval_scatter(summary_df, asset, lb)
    fig_forecast = fig_forecast_bar(summary_df, asset, lb)
    fig_est_heat = fig_estimator_heatmap(summary_df, asset, lb)
    fig_est_bar = fig_estimator_grouped_bars(summary_df, asset, lb)

    if objective == "Long gamma (maximize RV)":
        rec_interval = safe_pick_row(summary_df, "Composite RV (%)", ascending=False)["Interval"]
    else:
        rec_interval = safe_pick_row(summary_df, "Composite RV (%)", ascending=True)["Interval"]
    rec_analysis = next((x for x in ok_analyses if x["interval"] == rec_interval), ok_analyses[0])
    fig_roll_methods = fig_rolling_methods_for_interval(rec_analysis, asset, lb)

    _show(fig_roll_multi)
    st.caption(
        "Rolling RV lines: use this to evaluate stability. A frequency with high average RV but highly erratic "
        "swings may be less desirable for operational hedging than a slightly lower but stable alternative."
    )
    c_a, c_b = st.columns(2)
    with c_a:
        _show(fig_bar)
        st.caption("Bar chart: direct cross-frequency ranking of current realized vol (shortest to longest hedge interval).")
    with c_b:
        _show(fig_box)
        st.caption("Box plot: distribution of rolling RV; median and spread reveal consistency vs tail-risk behavior.")

    _show(fig_heat)
    st.caption("Metrics heatmap: quick multi-metric scan for trade-offs (RV level, forecast, completeness).")
    c_c, c_d = st.columns(2)
    with c_c:
        _show(fig_scatter)
        st.caption("Scatter (log x-axis): shows how RV changes as hedge interval length increases.")
    with c_d:
        _show(fig_forecast)
        st.caption("Forecast chart: bars are next-window EWMA forecast; line is current RV for context.")
    _show(fig_comp)
    st.caption("Completeness chart: received vs expected candles per interval; persistent gaps reduce confidence in comparisons.")

    st.markdown("#### Advanced estimator sections")
    st.caption(
        "These charts feature Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, Bipower variation, and RK/PA. "
        "Use them to cross-check whether frequency recommendations are robust across estimator families."
    )
    _show(fig_est_heat)
    st.caption(
        "Cells marked **N/A** are not zero: the estimator was undefined for that "
        "frequency — either too few bars for its minimum sample (e.g. RK/PA needs "
        "≥5 returns) or a duplicated/forward-filled close series. Return-based "
        "estimators (Close-to-close, Bipower, RK/PA, Yang-Zhang) require more "
        "intraday observations than the per-bar OHLC range estimators."
    )
    c_e, c_f = st.columns(2)
    with c_e:
        _show(fig_est_bar)
        st.caption("Grouped bars compare all estimator RV levels per hedging frequency.")
    with c_f:
        _show(fig_roll_methods)
        st.caption(
            f"Rolling estimator comparison at recommended interval ({rec_interval}) for this lookback. "
            "Divergence between lines indicates model sensitivity."
        )

st.markdown("### Method diagnostics")
diag = pd.DataFrame([
    {"Method": "Close-to-close", "Status": "Active", "Current implementation": "All selected frequencies",
     "Notes": "Log returns, sample std, sqrt(periods/year); 365d year."},
    {"Method": "Parkinson", "Status": "Active", "Current implementation": "All selected frequencies",
     "Notes": "High/low range estimator. Efficient in diffusion-like regimes; can understate jumps."},
    {"Method": "Garman-Klass", "Status": "Active", "Current implementation": "All selected frequencies",
     "Notes": "OHLC estimator with lower variance potential than close-close in stable regimes."},
    {"Method": "Rogers-Satchell", "Status": "Active", "Current implementation": "All selected frequencies",
     "Notes": "OHLC drift-robust estimator; useful in trending markets."},
    {"Method": "Yang-Zhang", "Status": "Active", "Current implementation": "All selected frequencies",
     "Notes": "Composite of open jump, close-open variance, and RS term. Robust all-rounder."},
    {"Method": "Bipower variation", "Status": "Active", "Current implementation": "All selected frequencies",
     "Notes": "Continuous variation proxy; helps separate jump-driven realized vol."},
    {"Method": "Realized Kernel / Pre-averaging", "Status": "Active (proxy blend)", "Current implementation": "All selected frequencies",
     "Notes": "Noise-robust proxy using kernel + pre-averaging blend for high-frequency data."},
])
st.dataframe(diag, width="stretch", hide_index=True)
st.caption(
    "Estimator note: all listed estimators are active. Use composite RV and cross-estimator charts "
    "to avoid overfitting a hedge decision to any single estimator."
)

# ---------------------------------------------------------------------------
# Telegram send
# ---------------------------------------------------------------------------

if send_tg:
    with st.spinner("Sending report to Telegram…"):
        ok = True
        send_message(
            "📊 <b>Time Based Realized Vol — Overview</b>\n"
            f"Asset: <b>{asset}</b>\n"
            f"3D matrix metric: <b>{matrix_metric_label}</b>\n"
            f"Detailed lookbacks selected: <b>{', '.join(detail_candidates)}</b>"
        )
        png_cross = fx_style.fig_to_png(fig_cross)
        ok = (bool(png_cross) and send_photo(png_cross, caption=f"TB RV — {asset} cross-lookback heatmap")) and ok
        ok = send_decision_matrix_3d_to_telegram(
            all_rows, asset, metric_col=matrix_metric_col, metric_label=matrix_metric_label,
            caption=f"TB RV — {asset} 3D decision matrix",
        ) and ok

        if not rec_df.empty:
            send_message(
                "📋 <b>Overview recommendations (selected detailed lookbacks)</b>\n" +
                "\n".join(
                    f"{row['Lookback']}: {row['Recommended interval']} "
                    f"(Composite {row['Composite RV (%)']:.2f}%, Forecast {row['Forecast next RV (%)']:.2f}%)"
                    for _, row in rec_df[rec_df["Lookback"].isin(detail_candidates)].iterrows()
                )
            )

        for lb in [x for x in detail_candidates if x in lookback_data]:
            sdf = lookback_data[lb]["summary"]
            an = lookback_data[lb]["analyses"]
            sub_ok, _ = send_time_based_rv_report_to_telegram(
                asset, lb, sdf,
                [
                    (fig_rolling_multi(an, asset, lb), f"TB RV — {asset} {lb} rolling comparison"),
                    (fig_interval_bar(sdf, asset, lb), f"TB RV — {asset} {lb} interval ranking"),
                    (fig_rolling_distribution_box(an, asset, lb), f"TB RV — {asset} {lb} rolling distribution"),
                    (fig_metrics_heatmap(sdf, asset, lb), f"TB RV — {asset} {lb} metric heatmap"),
                    (fig_forecast_bar(sdf, asset, lb), f"TB RV — {asset} {lb} forecast"),
                    (fig_completeness_bar(sdf, asset), f"TB RV — {asset} {lb} completeness"),
                    (fig_estimator_heatmap(sdf, asset, lb), f"TB RV — {asset} {lb} estimator heatmap"),
                    (fig_estimator_grouped_bars(sdf, asset, lb), f"TB RV — {asset} {lb} estimator grouped bars"),
                ],
            )
            ok = ok and sub_ok
    if ok:
        st.success("Report sent to Telegram")
    else:
        st.error("One or more lookback reports failed to send")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.caption(
    f"Data: Deribit perpetual OHLC ({ASSET_CONFIG[asset]['perp']}) | Bars 1m–1d, lookback 1d–30d | "
    f"Realized-vol annualization uses 365 days | "
    f"Last refresh: {fx_style.local_now():%H:%M:%S} {fx_style.DISPLAY_TZ_LABEL}"
)
