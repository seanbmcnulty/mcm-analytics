"""
MCM Analytics — Volatility regime identification: data, indicators, and the
breakout-probability model behind pages/07_Regime_Identifier.py.

This is a from-scratch, Deribit-only port of exodus-analytics's
Regime_Identifier.py (analytics_frontend/streamlit/pages/Regime_Identifier.py,
~4500 lines). Every formula in the probability model, the calibration
pipeline, the shock/squeeze detectors, and the advanced-indicator functions
below is carried over unchanged from that source — only the data-fetch layer
and the calibration *persistence* mechanism differ, both deliberately:

- **Data**: exodus fetched OHLCV from Binance (ccxt) and ATM IV from Deribit's
  book-summary endpoint directly. This app has no Binance dependency at all
  (Deribit-only architecture — see CLAUDE.md) and already has purpose-built
  Deribit history/surface infrastructure, so `fetch_perp_ohlc()` below reads
  perp candles via `lib.deribit.get_tradingview_ohlc()` and 30d ATM IV via
  `lib.surface.option_vols_by_dte()` instead.
- **Assets**: BTC/ETH only (not SOL) — matches this page's DVOL-dependent
  sibling, Spot_Vol_Correlation, and was a deliberate scope decision (see
  CLAUDE.md session log) rather than a data-availability constraint; Deribit
  does list SOL options and this module would work for it if that scope
  changes later.
- **Calibration**: exodus persisted fitted calibration params to a JSON file
  next to the script and re-fit when that file was >24h stale. Streamlit
  Community Cloud's filesystem is ephemeral (see CLAUDE.md) — a written file
  vanishes on the next redeploy — so persistence is replaced with an
  `st.cache_data`-backed in-session fit (`get_calibration`): still shared
  across users/reruns for its TTL (avoiding refitting on every page load),
  but recomputed from scratch each time that TTL lapses rather than loaded
  from disk. This was a deliberate choice, not a workaround — see CLAUDE.md.
"""

from __future__ import annotations

import time
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

from lib import deribit, surface
from lib.constants import TTL_SLOW, TTL_DAILY_EXTERNAL

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


# ============================================================================
# CONSTANTS (verbatim from exodus, BTC/ETH subset)
# ============================================================================

REGIME_ASSETS = ["BTC", "ETH"]

PERP_INSTRUMENT = {"BTC": "BTC-PERPETUAL", "ETH": "ETH-PERPETUAL"}

# Asset-specific volatility thresholds (annualized RV %)
VOL_THRESHOLDS = {
    "BTC": {"low": 40.0, "high": 60.0},
    "ETH": {"low": 50.0, "high": 75.0},
}
LOW_VOL_THRESHOLD = 40.0
HIGH_VOL_THRESHOLD = 60.0

# Asset-specific shock thresholds (daily move %)
SHOCK_THRESHOLDS_PCT = {"BTC": 4.0, "ETH": 5.0}
HALF_LIFE_DAYS = 6.6  # Volatility shock half-life
BREAKOUT_PROBABILITY_THRESHOLD = 60  # Days in low-vol regime
# Asset-specific volume squeeze: volume < X% of 30-day average
SQUEEZE_THRESHOLDS = {"BTC": 0.5, "ETH": 0.5}
SHOCK_VOLUME_SECTION_DAYS = 90  # Shock & volume charts: last 3 months

CALIBRATION_HORIZON_DAYS = 3
CALIBRATION_BACKTEST_STEP = 3
CALIBRATION_MIN_SAMPLES = 30
ROLLING_CALIBRATION_DAYS = 365  # Rolling window used when fitting calibration


# ============================================================================
# DATA FETCHING
# ============================================================================

class RegimeDataError(RuntimeError):
    """Raised rather than returning an empty frame, so a transient outage is
    not pinned in st.cache_data for the full TTL (see lib/history.py and
    pages/06 for the same pattern in this codebase)."""


@st.cache_data(ttl=TTL_SLOW, max_entries=8, show_spinner=False)
def _fetch_perp_ohlc_cached(asset: str, days_back: int) -> pd.DataFrame:
    instrument = PERP_INSTRUMENT.get(asset)
    if instrument is None:
        raise RegimeDataError(f"Unsupported asset: {asset}")
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days_back * 86400 * 1000
    df = deribit.get_tradingview_ohlc(instrument, "1D", start_ms, end_ms)
    if df is None or df.empty:
        raise RegimeDataError(f"No candles returned for {instrument} (1D)")
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["open", "high", "low", "close", "volume"]].copy()


def fetch_perp_ohlc(asset: str, days_back: int) -> tuple[pd.DataFrame | None, str | None]:
    """Daily perp OHLCV for `asset` over the trailing `days_back` days.
    Returns (df, error_message) — df is None on failure."""
    try:
        return _fetch_perp_ohlc_cached(asset, days_back).copy(), None
    except RegimeDataError as exc:
        return None, str(exc)
    except Exception as exc:  # network/parse errors from the Deribit client
        return None, str(exc)


def get_deribit_atm_iv_30d(asset: str) -> tuple[float | None, str | None]:
    """30d ATM implied vol (decimal) interpolated from the live option chain.
    Returns (iv_decimal, error_reason) — mirrors exodus's Deribit-direct
    version but reuses this app's existing surface machinery (BS delta
    search + DTE interpolation) instead of a crude moneyness/DTE filter."""
    try:
        vols = surface.option_vols_by_dte(asset)
    except Exception as exc:
        return None, str(exc)
    atm = (vols or {}).get("atm") or {}
    if not atm:
        return None, "no ATM quotes in live chain"
    try:
        iv = surface.interp_at_dte(atm, 30)
    except Exception as exc:
        return None, str(exc)
    if iv is None or not np.isfinite(iv) or iv <= 0:
        return None, "interpolation failed"
    return float(iv), None


# ============================================================================
# VOLATILITY CALCULATIONS
# ============================================================================

def calculate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["daily_return"] = df["close"].pct_change()
    return df


def calculate_realized_volatility(df: pd.DataFrame, window_days: int = 30) -> pd.Series:
    if "log_returns" not in df.columns:
        df = calculate_log_returns(df)
    rolling_std = df["log_returns"].rolling(window=window_days, min_periods=1).std()
    return rolling_std * np.sqrt(365) * 100


def calculate_multiple_volatility_windows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for w in (7, 14, 21, 30, 90):
        df[f"rv_{w}d"] = calculate_realized_volatility(df, window_days=w)
    return df


# ============================================================================
# GARCH
# ============================================================================

def fit_garch_model(returns: pd.Series, p: int = 1, q: int = 1, dist: str = "t"):
    if not ARCH_AVAILABLE or len(returns) < 100:
        return None
    try:
        returns_clean = returns.dropna()
        if len(returns_clean) < 100:
            return None
        model = arch_model(returns_clean * 100, vol="Garch", p=p, q=q, dist=dist)
        return model.fit(disp="off", show_warning=False)
    except Exception:
        return None


def calculate_garch_conditional_volatility(df: pd.DataFrame, window: int = 252, step: int = 5) -> pd.Series:
    if "log_returns" not in df.columns:
        df = calculate_log_returns(df)
    if not ARCH_AVAILABLE or len(df) < window + 50:
        return pd.Series(index=df.index, dtype=float)
    garch_vol = pd.Series(index=df.index, dtype=float)
    last_value = None
    for i in range(window, len(df), step):
        try:
            model = fit_garch_model(df["log_returns"].iloc[i - window:i + 1])
            if model is not None:
                forecast = model.forecast(horizon=1)
                cond_vol = np.sqrt(forecast.variance.values[-1, 0]) * np.sqrt(365)
                garch_vol.iloc[i] = cond_vol
                last_value = cond_vol
            elif last_value is not None:
                garch_vol.iloc[i] = last_value
        except Exception:
            if last_value is not None:
                garch_vol.iloc[i] = last_value
            continue
    return garch_vol.ffill()


def calculate_garch_persistence(df: pd.DataFrame, window: int = 252, step: int = 5) -> pd.Series:
    if "log_returns" not in df.columns:
        df = calculate_log_returns(df)
    if not ARCH_AVAILABLE or len(df) < window + 50:
        return pd.Series(index=df.index, dtype=float)
    persistence = pd.Series(index=df.index, dtype=float)
    last_value = None
    for i in range(window, len(df), step):
        try:
            model = fit_garch_model(df["log_returns"].iloc[i - window:i + 1])
            if model is not None and hasattr(model, "params"):
                params = model.params
                if "alpha[1]" in params.index and "beta[1]" in params.index:
                    val = params["alpha[1]"] + params["beta[1]"]
                    persistence.iloc[i] = val
                    last_value = val
            elif last_value is not None:
                persistence.iloc[i] = last_value
        except Exception:
            if last_value is not None:
                persistence.iloc[i] = last_value
            continue
    return persistence.ffill()


# ============================================================================
# ADVANCED INDICATORS (compression, early-warning, microstructure)
# ============================================================================

def calculate_volatility_of_volatility(df: pd.DataFrame, window: int = 30) -> pd.Series:
    if "rv_30d" not in df.columns:
        df = calculate_multiple_volatility_windows(df)
    return df["rv_30d"].rolling(window=window, min_periods=10).std()


def calculate_term_structure_slope(df: pd.DataFrame) -> pd.Series:
    if "rv_7d" not in df.columns or "rv_90d" not in df.columns:
        df = calculate_multiple_volatility_windows(df)
    return df["rv_7d"] - df["rv_90d"]


def calculate_volatility_momentum(df: pd.DataFrame, windows=(7, 30, 90)) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        col = f"rv_{window}d"
        if col not in df.columns:
            df[col] = calculate_realized_volatility(df, window_days=window)
        df[f"{col}_momentum"] = df[col].pct_change(periods=7)
        df[f"{col}_acceleration"] = df[f"{col}_momentum"].diff()
    return df


def calculate_bollinger_band_width(df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.Series:
    rolling_mean = df["close"].rolling(window=window).mean()
    rolling_std = df["close"].rolling(window=window).std()
    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std
    return (upper - lower) / rolling_mean * 100


def calculate_atr_compression_ratio(df: pd.DataFrame, window: int = 14, lookback: int = 252) -> pd.Series:
    d = df.copy()
    d["tr1"] = d["high"] - d["low"]
    d["tr2"] = (d["high"] - d["close"].shift(1)).abs()
    d["tr3"] = (d["low"] - d["close"].shift(1)).abs()
    d["tr"] = d[["tr1", "tr2", "tr3"]].max(axis=1)
    atr = d["tr"].rolling(window=window).mean()
    atr_max = atr.rolling(window=lookback).max()
    atr_min = atr.rolling(window=lookback).min()
    atr_range = atr_max - atr_min
    return (atr - atr_min) / (atr_range + 1e-10)


def calculate_volatility_percentile_rank(df: pd.DataFrame, window: int = 252) -> pd.Series:
    if "rv_30d" not in df.columns:
        df = calculate_multiple_volatility_windows(df)
    rv = df["rv_30d"]
    pct = pd.Series(index=df.index, dtype=float)
    for i in range(window, len(df)):
        window_data = rv.iloc[i - window:i + 1]
        current = rv.iloc[i]
        if len(window_data) > 0 and not pd.isna(current):
            pct.iloc[i] = (window_data <= current).sum() / len(window_data) * 100
        else:
            pct.iloc[i] = 50
    return pct


def calculate_volatility_acceleration(df: pd.DataFrame) -> pd.Series:
    if "rv_30d" not in df.columns:
        df = calculate_multiple_volatility_windows(df)
    return df["rv_30d"].diff().diff()


def calculate_cross_timeframe_divergence(df: pd.DataFrame) -> pd.Series:
    if "rv_7d" not in df.columns or "rv_90d" not in df.columns:
        df = calculate_multiple_volatility_windows(df)
    rv_7d_mean = df["rv_7d"].rolling(window=90).mean()
    rv_90d_mean = df["rv_90d"].rolling(window=90).mean()
    rv_7d_norm = (df["rv_7d"] - rv_7d_mean) / (rv_7d_mean + 1e-10)
    rv_90d_norm = (df["rv_90d"] - rv_90d_mean) / (rv_90d_mean + 1e-10)
    return rv_7d_norm - rv_90d_norm


def calculate_regime_transition_momentum(df: pd.DataFrame, asset: str = "BTC") -> pd.Series:
    if "regime" not in df.columns:
        df = add_regime_classification(df, asset=asset)
    regime_changes = (df["regime"] != df["regime"].shift(1)).astype(int)
    return regime_changes.rolling(window=30).sum()


def calculate_price_range_compression(df: pd.DataFrame, window: int = 20, lookback: int = 252) -> pd.Series:
    daily_range = (df["high"] - df["low"]) / df["close"]
    return daily_range.rolling(window=lookback).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50, raw=False
    )


def calculate_volume_volatility_correlation(df: pd.DataFrame, window: int = 60) -> pd.Series:
    if "rv_30d" not in df.columns:
        df = calculate_multiple_volatility_windows(df)
    return df["volume"].rolling(window=window).corr(df["rv_30d"])


def detect_volume_spikes(df: pd.DataFrame, window: int = 20, threshold_multiplier: float = 2.0) -> pd.Series:
    avg_volume = df["volume"].rolling(window=window, min_periods=5).mean()
    return (df["volume"] > avg_volume * threshold_multiplier).astype(int)


def calculate_price_momentum(df: pd.DataFrame, short_window: int = 7, long_window: int = 30) -> pd.Series:
    returns = df["close"].pct_change()
    short_m = returns.rolling(window=short_window).mean() * np.sqrt(short_window) * 100
    long_m = returns.rolling(window=long_window).mean() * np.sqrt(long_window) * 100
    momentum = (short_m - long_m).abs()
    momentum = momentum.rolling(window=min(252, len(df))).apply(
        lambda x: min(100, max(0, (x.iloc[-1] - x.quantile(0.1)) / max(x.quantile(0.9) - x.quantile(0.1), 0.01) * 100))
        if len(x) > 0 else 50,
        raw=False,
    )
    return momentum.fillna(0)


def calculate_garch_volatility_forecast_signal(df: pd.DataFrame, forecast_horizon: int = 5) -> pd.Series:
    if "garch_conditional_vol" not in df.columns:
        return pd.Series(0, index=df.index)
    vol_trend = df["garch_conditional_vol"].diff(forecast_horizon)
    signal = vol_trend.rolling(window=min(252, len(df))).apply(
        lambda x: (x.iloc[-1] - x.mean()) / max(x.std(), 0.01) if len(x) > 0 and x.std() > 0 else 0, raw=False
    )
    return signal.fillna(0)


def calculate_all_advanced_indicators(df: pd.DataFrame, asset: str = "BTC") -> pd.DataFrame:
    """Adds every advanced-indicator column used by calculate_breakout_probability
    and the chart builders. Mirrors exodus's calculate_all_advanced_indicators."""
    if "rv_30d" not in df.columns:
        df = calculate_multiple_volatility_windows(df)
    if "log_returns" not in df.columns:
        df = calculate_log_returns(df)

    if ARCH_AVAILABLE and len(df) > 300:
        try:
            df["garch_conditional_vol"] = calculate_garch_conditional_volatility(df, step=5)
            df["garch_persistence"] = calculate_garch_persistence(df, step=5)
        except Exception:
            pass

    df["vol_of_vol"] = calculate_volatility_of_volatility(df)
    for w in (7, 14, 30, 60, 90):
        col = f"rv_{w}d"
        if col not in df.columns:
            df[col] = calculate_realized_volatility(df, window_days=w)
    df["term_structure_slope"] = calculate_term_structure_slope(df)
    df = calculate_volatility_momentum(df)

    df["bb_width"] = calculate_bollinger_band_width(df)
    df["atr_compression"] = calculate_atr_compression_ratio(df)
    df["vol_percentile_rank"] = calculate_volatility_percentile_rank(df)

    df["vol_acceleration"] = calculate_volatility_acceleration(df)
    df["cross_timeframe_divergence"] = calculate_cross_timeframe_divergence(df)
    df["regime_transition_momentum"] = calculate_regime_transition_momentum(df, asset)

    df["range_compression"] = calculate_price_range_compression(df)
    df["volume_vol_correlation"] = calculate_volume_volatility_correlation(df)

    df["volume_spikes"] = detect_volume_spikes(df)
    df["price_momentum"] = calculate_price_momentum(df)
    df["garch_forecast_signal"] = calculate_garch_volatility_forecast_signal(df)

    return df


# ============================================================================
# REGIME CLASSIFICATION
# ============================================================================

def classify_regime(volatility: float, low_threshold: float | None = None, high_threshold: float | None = None) -> str:
    if pd.isna(volatility):
        return "Unknown"
    low_t = low_threshold if low_threshold is not None else LOW_VOL_THRESHOLD
    high_t = high_threshold if high_threshold is not None else HIGH_VOL_THRESHOLD
    if volatility < low_t:
        return "Low"
    if volatility <= high_t:
        return "Moderate"
    return "High"


def add_regime_classification(df: pd.DataFrame, asset: str = "BTC", rv_col: str = "rv_30d") -> pd.DataFrame:
    df = df.copy()
    thresholds = VOL_THRESHOLDS.get(asset, {"low": LOW_VOL_THRESHOLD, "high": HIGH_VOL_THRESHOLD})
    if rv_col not in df.columns:
        rv_col = "rv_30d"
    df["regime"] = df[rv_col].apply(lambda v: classify_regime(v, thresholds["low"], thresholds["high"]))
    return df


def count_consecutive_days_in_regime(df: pd.DataFrame, regime_col: str = "regime") -> int:
    if len(df) == 0:
        return 0
    current = df[regime_col].iloc[-1]
    count = 0
    for i in range(len(df) - 1, -1, -1):
        if df[regime_col].iloc[i] == current:
            count += 1
        else:
            break
    return count


def calculate_days_in_regime_series(df: pd.DataFrame, regime_col: str = "regime") -> pd.Series:
    if len(df) == 0:
        return pd.Series(dtype=int, index=df.index)
    out = pd.Series(index=df.index, dtype=int)
    for i in range(len(df)):
        current = df[regime_col].iloc[i]
        count = 0
        for j in range(i, -1, -1):
            if df[regime_col].iloc[j] == current:
                count += 1
            else:
                break
        out.iloc[i] = count
    return out


def get_historical_regime_spell_lengths(df: pd.DataFrame, regime_col: str = "regime") -> dict[str, list[int]]:
    """Completed historical spell lengths (in days), grouped by regime label.

    A "spell" is a maximal run of consecutive days in the same regime. The
    spell still in progress at the end of the series is deliberately
    excluded -- its final length is unknown (censored), and including it as
    if it were a completed spell would understate how long spells of that
    type can actually run."""
    if df is None or len(df) == 0 or regime_col not in df.columns:
        return {}
    regimes = df[regime_col].tolist()
    if not regimes:
        return {}
    spells: dict[str, list[int]] = {}
    run_regime, run_len = regimes[0], 1
    for r in regimes[1:]:
        if r == run_regime:
            run_len += 1
        else:
            spells.setdefault(run_regime, []).append(run_len)
            run_regime, run_len = r, 1
    # run_regime/run_len is the in-progress spell at the end -- not appended.
    return spells


MIN_SPELLS_FOR_DURATION_FORECAST = 3


def forecast_regime_duration(df: pd.DataFrame, current_regime: str, days_in_regime: int,
                              regime_col: str = "regime") -> dict:
    """Empirical forecast for how much longer the *current* regime spell is
    likely to run, from this asset's own completed historical spells of the
    same regime label.

    Survivor-conditioned: only past spells that reached at least
    ``days_in_regime`` are used to estimate what happens next -- a spell
    that historically ended on day 10 says nothing about a spell that has
    already run 40 days. This is the same idea as a Kaplan-Meier-style
    conditional residual life, done directly on the small empirical sample
    rather than fitting a parametric survival curve (there usually isn't
    enough history per asset/regime to justify one).

    Returns a dict; ``n_spells`` is always present. Below
    ``MIN_SPELLS_FOR_DURATION_FORECAST`` completed spells of this regime
    type, only ``{"n_spells": n}`` is returned -- too little history to say
    anything. Otherwise also includes: ``n_survivors`` (historical spells
    that lasted at least as long as the current one), ``median_total_duration``
    (unconditional, for context), ``median_remaining``/``mean_remaining``/
    ``p25_remaining``/``p75_remaining`` (days), and ``prob_ends_within_7d``.
    When every historical spell of this type ended before reaching the
    current length (``n_survivors == 0``), the current spell is already
    unusually long for this asset/regime -- remaining-duration stats are
    reported as 0 and ``prob_ends_within_7d`` as 1.0 to signal "expect this
    to end imminently," rather than extrapolating past the observed range."""
    spells = get_historical_regime_spell_lengths(df, regime_col=regime_col)
    lengths = spells.get(current_regime, [])
    n_spells = len(lengths)
    if n_spells < MIN_SPELLS_FOR_DURATION_FORECAST:
        return {"n_spells": n_spells}

    lengths_arr = np.array(lengths, dtype=float)
    days_in_regime = max(0, int(days_in_regime))
    survivors = lengths_arr[lengths_arr >= days_in_regime]
    n_survivors = int(len(survivors))
    if n_survivors == 0:
        median_remaining = mean_remaining = p25_remaining = p75_remaining = 0.0
        prob_ends_within_7d = 1.0
    else:
        remaining = survivors - days_in_regime
        median_remaining = float(np.median(remaining))
        mean_remaining = float(np.mean(remaining))
        p25_remaining = float(np.percentile(remaining, 25))
        p75_remaining = float(np.percentile(remaining, 75))
        prob_ends_within_7d = float(np.mean(remaining <= 7))

    return {
        "n_spells": n_spells,
        "n_survivors": n_survivors,
        "median_total_duration": float(np.median(lengths_arr)),
        "median_remaining": median_remaining,
        "mean_remaining": mean_remaining,
        "p25_remaining": p25_remaining,
        "p75_remaining": p75_remaining,
        "prob_ends_within_7d": prob_ends_within_7d,
        "current_days": days_in_regime,
    }


# ============================================================================
# SHOCK & SQUEEZE DETECTION
# ============================================================================

def detect_major_shocks(df: pd.DataFrame, asset: str = "BTC", threshold: float | None = None) -> pd.DataFrame:
    if "daily_return" not in df.columns:
        df = calculate_log_returns(df)
    if threshold is None:
        threshold = SHOCK_THRESHOLDS_PCT.get(asset, 4.0) / 100.0
    shocks = df[df["daily_return"].abs() > threshold].copy()
    shocks["shock_magnitude"] = shocks["daily_return"] * 100
    return shocks


def calculate_shock_persistence(shock_date, current_date, half_life_days: float = HALF_LIFE_DAYS) -> float:
    days_since = (current_date - shock_date).days
    stabilization_days = 2 * half_life_days
    return max(0.0, stabilization_days - days_since)


def get_latest_shock_info(df: pd.DataFrame, asset: str = "BTC"):
    if df.empty:
        return None, None, None
    shocks = detect_major_shocks(df, asset=asset)
    if len(shocks) == 0:
        return None, None, None
    latest = shocks.iloc[-1]
    shock_date = latest.name
    return shock_date, latest["shock_magnitude"], calculate_shock_persistence(shock_date, df.index[-1])


def detect_squeeze_periods(df: pd.DataFrame, asset: str = "BTC", threshold: float | None = None, window: int = 30) -> pd.DataFrame:
    df = df.copy()
    if threshold is None:
        threshold = SQUEEZE_THRESHOLDS.get(asset, 0.5)
    df["volume_30d_avg"] = df["volume"].rolling(window=window, min_periods=1).mean()
    df["is_squeeze"] = df["volume"] < (df["volume_30d_avg"] * threshold)
    return df


# ============================================================================
# BREAKOUT PROBABILITY MODEL (verbatim port of exodus's weighted composite)
# ============================================================================

def _combine_probability_components(prob_components: list[tuple[str, float, float]]) -> float:
    """Combine weighted components and apply a strength boost. Returns 0-100."""
    if not prob_components:
        return 0
    total_weight = sum(w for _, _, w in prob_components)
    base_prob = sum(s * w for _, s, w in prob_components) / max(total_weight, 0.01)
    strong = sum(1 for _, s, _ in prob_components if s > 70)
    very_strong = sum(1 for _, s, _ in prob_components if s > 85)
    boost = 1.25 if very_strong >= 3 else (1.15 if very_strong >= 2 else (1.10 if strong >= 4 else 1.0))
    return min(100, max(0, base_prob * boost))


def calculate_breakout_probability(
    df: pd.DataFrame,
    days_in_regime: int,
    regime: str,
    lookback_days: int = 730,
    asset: str = "BTC",
    return_components: bool = False,
    learned_weights: dict | None = None,
):
    """Regime-dependent composite score, 0-100:
    - In **Low**: P(breakout to Moderate/High) — 16 weighted factors.
    - In **Moderate/High**: P(return to Low) — 7 weighted factors.

    If `return_components=True`, returns the raw [(name, score), ...] list
    used to fit calibration weights. If `learned_weights` is supplied (from
    `get_calibration`), combines components via a fitted logistic instead of
    the fixed-weight fallback in `_combine_probability_components`.
    """
    if df.empty or len(df) < 30:
        return [] if return_components else 0

    lookback_date = df.index[-1] - pd.Timedelta(days=lookback_days)
    historical_df = df[df.index >= lookback_date].copy()
    if len(historical_df) < 30:
        historical_df = df.copy()

    current_rv_7d = historical_df["rv_7d"].iloc[-1] if "rv_7d" in historical_df.columns else 0
    current_rv_30d = historical_df["rv_30d"].iloc[-1] if "rv_30d" in historical_df.columns else 0
    current_rv_90d = historical_df["rv_90d"].iloc[-1] if "rv_90d" in historical_df.columns else 0
    thresholds = VOL_THRESHOLDS.get(asset, {"low": LOW_VOL_THRESHOLD, "high": HIGH_VOL_THRESHOLD})
    low_thresh = thresholds["low"]

    # ----- Return-to-Low probability (currently Moderate or High) -----
    if regime in ("Moderate", "High"):
        comp = []
        to_low = sum(
            1 for i in range(len(historical_df) - 1)
            if historical_df["regime"].iloc[i] in ("Moderate", "High") and historical_df["regime"].iloc[i + 1] == "Low"
        )
        from_elevated = sum(1 for i in range(len(historical_df) - 1) if historical_df["regime"].iloc[i] in ("Moderate", "High"))
        if from_elevated > 0:
            comp.append(("hist_to_low", min(100, (to_low / from_elevated) * 150), 0.20))
        elevated_durations = []
        start = None
        for date, row in historical_df.iterrows():
            r = row.get("regime", "Unknown")
            if r in ("Moderate", "High"):
                if start is None:
                    start = date
            else:
                if start is not None:
                    elevated_durations.append((date - start).days)
                start = None
        if start is not None:
            elevated_durations.append((historical_df.index[-1] - start).days)
        if elevated_durations:
            avg_high = np.mean(elevated_durations)
            p75 = np.percentile(elevated_durations, 75) if len(elevated_durations) > 1 else avg_high
            if days_in_regime >= p75:
                dur_score = min(100, 40 + (days_in_regime - p75) / max(p75 * 0.1, 1) * 30)
            elif days_in_regime >= avg_high:
                dur_score = 25 + (days_in_regime - avg_high) / max(p75 - avg_high, 1) * 15
            else:
                dur_score = max(0, 10 * (days_in_regime / max(avg_high, 1)))
            comp.append(("days_elevated", dur_score, 0.18))
        if not pd.isna(current_rv_30d) and current_rv_30d >= low_thresh:
            distance_above = current_rv_30d - low_thresh
            comp.append(("proximity_to_low", max(0, min(100, 100 - distance_above * 2)), 0.15))
        if len(historical_df) >= 14:
            trend = historical_df["rv_30d"].iloc[-14:].diff().mean()
            comp.append(("vol_decline", min(100, 50 + abs(trend) * 20) if trend < 0 else max(0, 30 - trend * 10), 0.12))
        if not pd.isna(current_rv_7d) and not pd.isna(current_rv_30d) and current_rv_30d > 0:
            ratio = current_rv_7d / current_rv_30d
            comp.append(("vol_decompression", min(100, 40 + (1 - ratio) * 60) if ratio < 1 else max(0, 50 - (ratio - 1) * 30), 0.10))
        if "garch_persistence" in df.columns:
            p = df["garch_persistence"].iloc[-1]
            if not pd.isna(p):
                comp.append(("garch_low_pers", max(0, min(100, (1 - p) * 120)), 0.10))
        if "garch_forecast_signal" in df.columns:
            f = df["garch_forecast_signal"].iloc[-1]
            if not pd.isna(f) and f < 1:
                comp.append(("garch_decline", min(100, max(0, (1 - f) * 80)), 0.08))
        if len(historical_df) >= 60:
            trans = sum(1 for i in range(len(historical_df) - 1) if historical_df["regime"].iloc[i] != historical_df["regime"].iloc[i + 1])
            comp.append(("transitions", min(100, trans / (len(historical_df) / 30) * 20), 0.07))
        if return_components:
            return [(n, s) for n, s, _ in comp]
        key = "return_to_low"
        if learned_weights and key in learned_weights and isinstance(learned_weights[key], dict):
            w = learned_weights[key]
            try:
                from scipy.special import expit
                logit = w.get("intercept", 0) + sum((w.get("coef") or {}).get(n, 0) * (s / 100.0) for n, s in comp)
                return min(100, max(0, 100.0 * expit(logit)))
            except Exception:
                pass
        return _combine_probability_components(comp)

    # ----- Breakout probability (currently Low) -----
    prob_components = []

    if regime == "Low":
        regime_periods = []
        current_period_start = None
        current_regime_type = None
        for date, row in historical_df.iterrows():
            reg = row.get("regime", "Unknown")
            if reg != current_regime_type:
                if current_regime_type == "Low" and current_period_start:
                    regime_periods.append((date - current_period_start).days)
                current_regime_type = reg
                current_period_start = date
        if current_regime_type == "Low" and current_period_start:
            regime_periods.append((historical_df.index[-1] - current_period_start).days)
        if regime_periods:
            avg_duration = np.mean(regime_periods) if regime_periods else 60
            p75_duration = np.percentile(regime_periods, 75) if len(regime_periods) > 1 else avg_duration
            if days_in_regime > p75_duration:
                duration_score = min(100, 50 + (days_in_regime - p75_duration) / max(p75_duration * 0.05, 1) * 50)
            elif days_in_regime > avg_duration:
                duration_score = 30 + (days_in_regime - avg_duration) / max(p75_duration - avg_duration, 1) * 20
            else:
                duration_score = max(0, 15 * (days_in_regime / max(avg_duration, 1)))
            prob_components.append(("duration", duration_score, 0.18))

    if not pd.isna(current_rv_7d) and not pd.isna(current_rv_30d) and not pd.isna(current_rv_90d):
        compression_ratio = current_rv_7d / max(current_rv_30d, 1)
        compression_score = 0
        if current_rv_7d < current_rv_30d < current_rv_90d:
            compression_score = min(100, 60 + (1 - compression_ratio) ** 2 * 40)
        elif current_rv_7d < current_rv_30d:
            compression_score = 35 + (1 - compression_ratio) * 45
        prob_components.append(("compression", compression_score, 0.15))

    if "is_squeeze" in historical_df.columns:
        recent_squeeze = historical_df["is_squeeze"].iloc[-min(10, len(historical_df)):].sum()
        squeeze_score = min(100, recent_squeeze / 10 * 100) if recent_squeeze > 0 else 0
        prob_components.append(("squeeze", squeeze_score, 0.10))

    if len(historical_df) >= 14:
        recent_vol_trend = historical_df["rv_30d"].iloc[-14:].diff().mean()
        vol_trend_score = min(100, abs(recent_vol_trend) * 25) if recent_vol_trend < -0.5 else 0
        prob_components.append(("vol_trend", vol_trend_score, 0.08))

    if regime == "Low" and not pd.isna(current_rv_30d):
        distance_to_threshold = low_thresh - current_rv_30d
        if distance_to_threshold > 0:
            proximity_score = min(100, max(0, (15 - distance_to_threshold) / 15 * 100) ** 1.5)
            prob_components.append(("proximity", proximity_score, 0.06))

    if len(historical_df) >= 60:
        transitions = 0
        prev_regime = None
        for reg in historical_df["regime"]:
            if prev_regime and reg != prev_regime:
                transitions += 1
            prev_regime = reg
        transition_rate = transitions / (len(historical_df) / 30)
        prob_components.append(("transitions", min(100, transition_rate * 25), 0.04))

    if "garch_persistence" in df.columns:
        current_persistence = df["garch_persistence"].iloc[-1]
        if not pd.isna(current_persistence):
            if current_persistence > 0.96:
                persistence_score = min(100, 70 + (current_persistence - 0.96) / 0.04 * 30)
            elif current_persistence > 0.92:
                persistence_score = 40 + (current_persistence - 0.92) / 0.04 * 30
            else:
                persistence_score = max(0, (current_persistence - 0.85) / 0.07 * 40)
            prob_components.append(("garch_persistence", persistence_score, 0.12))

    if "term_structure_slope" in df.columns:
        current_slope = df["term_structure_slope"].iloc[-1]
        if not pd.isna(current_slope):
            if current_slope < -0.5:
                slope_score = min(100, 60 + abs(current_slope) * 8)
            elif current_slope < 0:
                slope_score = 30 + abs(current_slope) * 6
            else:
                slope_score = max(0, 20 - current_slope * 10)
            prob_components.append(("term_structure", slope_score, 0.10))

    if "vol_percentile_rank" in df.columns:
        current_percentile = df["vol_percentile_rank"].iloc[-1]
        if not pd.isna(current_percentile):
            if current_percentile < 10:
                percentile_score = min(100, 70 + (10 - current_percentile) / 10 * 30)
            elif current_percentile < 25:
                percentile_score = 40 + (25 - current_percentile) / 15 * 30
            else:
                percentile_score = max(0, (30 - current_percentile) / 30 * 40)
            prob_components.append(("vol_percentile", percentile_score, 0.08))

    if "bb_width" in df.columns:
        current_bb_width = df["bb_width"].iloc[-1]
        if not pd.isna(current_bb_width):
            bb_percentile = df["bb_width"].rolling(window=min(252, len(df))).apply(
                lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50, raw=False
            ).iloc[-1]
            if bb_percentile < 20:
                bb_score = min(100, 50 + (20 - bb_percentile) / 20 * 50)
            else:
                bb_score = max(0, (30 - bb_percentile) / 30 * 50)
            prob_components.append(("bb_width", bb_score, 0.05))

    if "vol_acceleration" in df.columns:
        current_accel = df["vol_acceleration"].iloc[-1]
        if not pd.isna(current_accel):
            if current_accel > 0.5:
                accel_score = min(100, 60 + current_accel * 8)
            elif current_accel > 0:
                accel_score = 30 + current_accel * 6
            else:
                accel_score = max(0, 20 + current_accel * 5)
            prob_components.append(("vol_acceleration", accel_score, 0.08))

    if "cross_timeframe_divergence" in df.columns:
        current_divergence = df["cross_timeframe_divergence"].iloc[-1]
        if not pd.isna(current_divergence):
            abs_div = abs(current_divergence)
            div_score = min(100, 50 + abs_div * 60) if abs_div > 0.3 else max(0, abs_div * 50)
            prob_components.append(("divergence", div_score, 0.05))

    if "range_compression" in df.columns:
        current_range_comp = df["range_compression"].iloc[-1]
        if not pd.isna(current_range_comp):
            if current_range_comp < 15:
                range_score = min(100, 60 + (15 - current_range_comp) / 15 * 40)
            elif current_range_comp < 30:
                range_score = 30 + (30 - current_range_comp) / 15 * 30
            else:
                range_score = max(0, (30 - current_range_comp) / 30 * 30)
            prob_components.append(("range_compression", range_score, 0.05))

    if "volume_spikes" in df.columns:
        recent_spikes = df["volume_spikes"].iloc[-min(14, len(df)):].sum()
        prob_components.append(("volume_spikes", min(100, recent_spikes / 14 * 100), 0.08))

    if "price_momentum" in df.columns:
        current_momentum = df["price_momentum"].iloc[-1]
        if not pd.isna(current_momentum):
            prob_components.append(("price_momentum", min(100, current_momentum), 0.07))

    if "garch_forecast_signal" in df.columns:
        current_forecast = df["garch_forecast_signal"].iloc[-1]
        if not pd.isna(current_forecast):
            if current_forecast > 1.0:
                forecast_score = min(100, 60 + current_forecast * 10)
            elif current_forecast > 0:
                forecast_score = 30 + current_forecast * 30
            else:
                forecast_score = max(0, 20 + current_forecast * 10)
            prob_components.append(("garch_forecast", forecast_score, 0.06))

    if return_components:
        return [(n, s) for n, s, _ in prob_components]
    key = "breakout"
    if learned_weights and key in learned_weights and isinstance(learned_weights[key], dict):
        w = learned_weights[key]
        try:
            from scipy.special import expit
            logit = w.get("intercept", 0) + sum((w.get("coef") or {}).get(n, 0) * (s / 100.0) for n, s in prob_components)
            return min(100, max(0, 100.0 * expit(logit)))
        except Exception:
            pass
    return _combine_probability_components(prob_components)


def calculate_breakout_probability_series(
    df: pd.DataFrame, lookback_days: int = 730, asset: str = "BTC", step: int | None = None, learned_weights: dict | None = None
) -> pd.Series:
    """Breakout/return-to-low probability at every point in time, computed
    from data available up to that point only (no look-ahead). Only
    evaluated every `step` days for performance, then linearly interpolated —
    identical approach to exodus's calculate_breakout_probability_series."""
    if df.empty or len(df) < 30:
        return pd.Series(dtype=float, index=df.index)
    if step is None:
        step = CALIBRATION_BACKTEST_STEP

    breakout_probs = pd.Series(index=df.index, dtype=float)
    min_start_idx = min(60, len(df) // 10)
    if "regime" not in df.columns:
        df = add_regime_classification(df, asset=asset)

    calculated_indices, calculated_values = [], []
    for i in range(min_start_idx, len(df), step):
        historical_df = df.iloc[:i + 1]
        if len(historical_df) < 30:
            continue
        days_in_regime = count_consecutive_days_in_regime(historical_df)
        current_regime = historical_df["regime"].iloc[-1]
        prob = calculate_breakout_probability(
            historical_df, days_in_regime, current_regime,
            lookback_days=lookback_days, asset=asset, learned_weights=learned_weights,
        )
        calculated_indices.append(i)
        calculated_values.append(prob)
        breakout_probs.iloc[i] = prob

    if len(calculated_indices) > 1:
        last_idx, last_val = calculated_indices[-1], calculated_values[-1]
        breakout_probs.iloc[last_idx:] = last_val
        for j in range(len(calculated_indices) - 1):
            start_idx, end_idx = calculated_indices[j], calculated_indices[j + 1]
            start_val, end_val = calculated_values[j], calculated_values[j + 1]
            for k in range(start_idx + 1, end_idx):
                alpha = (k - start_idx) / (end_idx - start_idx)
                breakout_probs.iloc[k] = start_val + alpha * (end_val - start_val)

    first_val = calculated_values[0] if calculated_values else 0
    breakout_probs.iloc[:min_start_idx] = first_val
    return breakout_probs


# ============================================================================
# CALIBRATION — fit in-session, TTL-cached (no disk persistence)
#
# exodus persisted fitted (a, b) logistic params + component weights to a
# JSON file next to the script, refitting when that file was >24h stale.
# On Streamlit Community Cloud the filesystem is ephemeral (CLAUDE.md), so a
# written file is gone on the next redeploy — there is nothing durable to
# refresh. Per the 2026-09-04 port decision, calibration is instead fit
# on demand and cached via st.cache_data for TTL_DAILY_EXTERNAL (1h): still
# shared across reruns/users within that window (avoiding refitting on every
# page load, which is the expensive part — see _backtest_component_outcomes),
# but recomputed from scratch, from whatever data is live, once the cache
# entry expires. The math (backtest -> logistic fit) is untouched from
# exodus; only *where the fitted params live* changed.
# ============================================================================

def _backtest_component_outcomes(df: pd.DataFrame, asset: str, horizon_days: int = CALIBRATION_HORIZON_DAYS, step: int = CALIBRATION_BACKTEST_STEP):
    """Backtest collecting (component_scores_list, outcome) pairs, used to
    fit per-component logistic weights."""
    if "regime" not in df.columns or len(df) < horizon_days + 60:
        return [], []
    breakout_records, return_to_low_records = [], []
    for i in range(60, len(df) - horizon_days, step):
        hist = df.iloc[: i + 1]
        regime = hist["regime"].iloc[-1]
        days_in_reg = count_consecutive_days_in_regime(hist)
        comp_list = calculate_breakout_probability(
            hist, days_in_reg, regime, lookback_days=min(730, len(hist)), asset=asset, return_components=True
        )
        if not isinstance(comp_list, list):
            continue
        future = df.iloc[i + 1: i + 1 + horizon_days]
        if regime == "Low":
            outcome = 1 if (future["regime"] == "Moderate").any() or (future["regime"] == "High").any() else 0
            breakout_records.append((comp_list, outcome))
        elif regime in ("Moderate", "High"):
            outcome = 1 if (future["regime"] == "Low").any() else 0
            return_to_low_records.append((comp_list, outcome))
    return breakout_records, return_to_low_records


def _fit_component_weights(records) -> dict | None:
    """Fit a logistic regression on (component_scores, outcome). Returns
    {intercept, coef: {name: float}} or None if there isn't enough data."""
    if not records or len(records) < CALIBRATION_MIN_SAMPLES:
        return None
    try:
        from scipy.special import expit
        from scipy.optimize import minimize
        all_names = sorted(set(n for comp_list, _ in records for n, _ in comp_list))
        if not all_names:
            return None
        X = np.zeros((len(records), len(all_names)))
        y = np.array([out for _, out in records], dtype=float)
        for i, (comp_list, _) in enumerate(records):
            d = dict(comp_list)
            for j, name in enumerate(all_names):
                X[i, j] = d.get(name, 0) / 100.0
        if np.sum(y) < 5 or np.sum(1 - y) < 5:
            return None

        def negloglik(params):
            b0, b = params[0], params[1:]
            p = np.clip(expit(b0 + X.dot(b)), 1e-6, 1 - 1e-6)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        res = minimize(negloglik, np.zeros(len(all_names) + 1), method="L-BFGS-B")
        if not res.success:
            return None
        coef = {all_names[j]: float(res.x[j + 1]) for j in range(len(all_names))}
        return {"intercept": float(res.x[0]), "coef": coef}
    except Exception:
        return None


def _backtest_raw_scores_and_outcomes(df: pd.DataFrame, asset: str, horizon_days: int = CALIBRATION_HORIZON_DAYS, step: int = CALIBRATION_BACKTEST_STEP, learned_weights: dict | None = None):
    """For each stepped date, the raw 0-100 score and whether the event
    happened within horizon_days. Returns (breakout_records, return_to_low_records),
    each a list of (raw_score, outcome 0/1)."""
    if "regime" not in df.columns or len(df) < horizon_days + 60:
        return [], []
    breakout_records, return_to_low_records = [], []
    for i in range(60, len(df) - horizon_days, step):
        hist = df.iloc[: i + 1]
        regime = hist["regime"].iloc[-1]
        days_in_reg = count_consecutive_days_in_regime(hist)
        raw = calculate_breakout_probability(
            hist, days_in_reg, regime, lookback_days=min(730, len(hist)), asset=asset, learned_weights=learned_weights
        )
        if isinstance(raw, list):
            continue
        future = df.iloc[i + 1: i + 1 + horizon_days]
        if regime == "Low":
            outcome = 1 if (future["regime"] == "Moderate").any() or (future["regime"] == "High").any() else 0
            breakout_records.append((raw, outcome))
        elif regime in ("Moderate", "High"):
            outcome = 1 if (future["regime"] == "Low").any() else 0
            return_to_low_records.append((raw, outcome))
    return breakout_records, return_to_low_records


def _fit_calibration_logistic(records):
    """Fit P(outcome) = 1 / (1 + exp(-(a*raw/100 + b))). Returns (a, b) or None."""
    if not records or len(records) < CALIBRATION_MIN_SAMPLES:
        return None
    raw = np.array([r[0] for r in records])
    y = np.array([r[1] for r in records])
    if np.sum(y) < 5 or np.sum(1 - y) < 5:
        return None
    try:
        from scipy.optimize import minimize
        from scipy.special import expit

        def negloglik(params):
            a, b = params
            p = np.clip(expit(a * (raw / 100.0) + b), 1e-6, 1 - 1e-6)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        res = minimize(negloglik, [1.0, 0.0], method="Nelder-Mead")
        if res.success:
            return (float(res.x[0]), float(res.x[1]))
    except Exception:
        pass
    return None


def calibrate_probability(raw_score: float, params: dict, regime: str) -> float:
    """Map a raw 0-100 score to a calibrated probability using the fitted
    logistic for the applicable branch (`breakout` when regime == Low, else
    `return_to_low`). Falls back to the raw score when no fit is available."""
    key = "breakout" if regime == "Low" else "return_to_low"
    p = (params or {}).get(key)
    if p is None or len(p) != 2:
        return raw_score
    a, b = p
    try:
        from scipy.special import expit
        return min(100, max(0, 100.0 * expit(a * (raw_score / 100.0) + b)))
    except Exception:
        return raw_score


@st.cache_data(ttl=TTL_DAILY_EXTERNAL, max_entries=4, show_spinner=False)
def _fit_calibration_cached(asset: str, _df: pd.DataFrame, as_of: str) -> dict:
    """Backtest -> component weights -> second-stage logistic fit, run once
    per TTL window per asset. `_df` (underscore prefix) is excluded from
    Streamlit's cache-key hash; `as_of` (the data's last timestamp) is the
    actual cache key alongside `asset`, so a materially newer dataset still
    busts the cache within the TTL window rather than serving a stale fit."""
    df_cal = _df
    if len(df_cal) > ROLLING_CALIBRATION_DAYS:
        cutoff = df_cal.index[-1] - pd.Timedelta(days=ROLLING_CALIBRATION_DAYS)
        df_cal = df_cal.loc[df_cal.index >= cutoff]

    bo_recs, rtl_recs = _backtest_component_outcomes(df_cal, asset)
    bw = _fit_component_weights(bo_recs)
    rtlw = _fit_component_weights(rtl_recs)
    learned_for_backtest = {}
    if bw:
        learned_for_backtest["breakout"] = bw
    if rtlw:
        learned_for_backtest["return_to_low"] = rtlw

    breakout_recs_cal, rtl_recs_cal = _backtest_raw_scores_and_outcomes(
        df_cal, asset, learned_weights=learned_for_backtest or None
    )
    bp = _fit_calibration_logistic(breakout_recs_cal)
    rtl = _fit_calibration_logistic(rtl_recs_cal)

    return {
        "breakout": bp,
        "return_to_low": rtl,
        "breakout_weights": bw,
        "return_to_low_weights": rtlw,
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "n_breakout": len(breakout_recs_cal),
        "n_return_to_low": len(rtl_recs_cal),
    }


def get_calibration(asset: str, df: pd.DataFrame) -> dict:
    """In-session calibration for `asset`: fitted (a, b) logistic params for
    both branches plus learned per-component weights, cached for
    TTL_DAILY_EXTERNAL. Returns {} when there isn't enough history yet
    (CALIBRATION_MIN_SAMPLES outcomes per branch) — callers should fall back
    to the uncalibrated raw score, which `calibrate_probability` does
    automatically when `params` is empty."""
    if df is None or len(df) < CALIBRATION_HORIZON_DAYS + 100:
        return {}
    as_of = str(df.index[-1])
    try:
        return _fit_calibration_cached(asset, df, as_of)
    except Exception:
        return {}


def learned_weights_from_calibration(cal: dict) -> dict | None:
    """calibration dict -> the `learned_weights` shape calculate_breakout_probability expects."""
    if not cal:
        return None
    weights = {}
    if cal.get("breakout_weights"):
        weights["breakout"] = cal["breakout_weights"]
    if cal.get("return_to_low_weights"):
        weights["return_to_low"] = cal["return_to_low_weights"]
    return weights or None


# ============================================================================
# FORWARD PROBABILITIES, EXPECTED MOVE, CROSS-ASSET STANCE
# ============================================================================

def _build_transition_matrix(transitions, regimes):
    matrix = np.zeros((3, 3))
    for from_r, to_r in transitions:
        if from_r in regimes and to_r in regimes:
            matrix[regimes.index(from_r), regimes.index(to_r)] += 1
    for i in range(3):
        if matrix[i, :].sum() == 0:
            matrix[i, i] = 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums


def get_forward_regime_probs_3d(df: pd.DataFrame, asset: str = "BTC", days_in_regime: int | None = None) -> dict | None:
    """P(Low), P(Moderate), P(High) three days forward via a Markov transition
    matrix raised to the 3rd power. Duration-dependent: uses a different
    matrix for short (<30d) vs long Low-vol spells, since a 5-day-old Low
    regime and a 90-day-old one have historically very different odds."""
    if df is None or len(df) == 0 or "regime" not in df.columns:
        return None
    regimes = ["Low", "Moderate", "High"]
    days_series = calculate_days_in_regime_series(df, regime_col="regime")
    transitions_short_low, transitions_long_low, transitions_elevated = [], [], []
    prev_regime, prev_days = None, 0
    for i in range(len(df)):
        regime = df["regime"].iloc[i]
        days_here = int(days_series.iloc[i]) if i < len(days_series) else 0
        if prev_regime is not None and regime != prev_regime:
            t = (prev_regime, regime)
            if prev_regime == "Low":
                (transitions_short_low if prev_days < 30 else transitions_long_low).append(t)
            else:
                transitions_elevated.append(t)
        prev_regime, prev_days = regime, days_here
    current_regime = df["regime"].iloc[-1]
    if current_regime not in regimes:
        current_regime = "Moderate"
    if days_in_regime is None:
        days_in_regime = int(days_series.iloc[-1]) if len(days_series) > 0 else 0
    if current_regime == "Low":
        trans = transitions_long_low if days_in_regime >= 30 else transitions_short_low
        if not trans:
            trans = transitions_short_low + transitions_long_low
    else:
        trans = transitions_elevated
    if not trans:
        trans = transitions_short_low + transitions_long_low + transitions_elevated
    if not trans:
        return {r: (1.0 if r == current_regime else 0.0) for r in regimes}
    P = _build_transition_matrix(trans, regimes)
    try:
        P3 = np.linalg.matrix_power(P, 3)
    except Exception:
        return {r: (1.0 if r == current_regime else 0.0) for r in regimes}
    row = P3[regimes.index(current_regime)]
    return {regimes[i]: float(np.clip(row[i], 0.0, 1.0)) for i in range(3)}


def get_expected_move_by_regime_dict(df: pd.DataFrame, asset: str = "BTC", horizon: int = 3) -> dict:
    if "regime" not in df.columns:
        df = add_regime_classification(df, asset=asset)
    if "daily_return" not in df.columns:
        df = calculate_log_returns(df)
    fwd_abs = df["daily_return"].abs().rolling(horizon).sum().shift(-horizon)
    by_regime = df.groupby(df["regime"]).apply(lambda g: fwd_abs.reindex(g.index).mean() * 100)
    by_regime = by_regime.dropna()
    return by_regime.to_dict() if not by_regime.empty else {}


def get_calibration_outcome_count(breakout_recs, rtl_recs) -> tuple[int, int]:
    return (len(breakout_recs) if breakout_recs else 0, len(rtl_recs) if rtl_recs else 0)


def get_cross_asset_regime_stance(rv_summary: dict) -> list[tuple]:
    """rv_summary: {symbol: 30d RV or None} -> [(asset, regime, rv, stance), ...]."""
    rows = []
    for sym, rv in (rv_summary or {}).items():
        if rv is None:
            rows.append((sym, "—", None, "—"))
            continue
        th = VOL_THRESHOLDS.get(sym, {"low": LOW_VOL_THRESHOLD, "high": HIGH_VOL_THRESHOLD})
        if rv < th["low"]:
            regime, stance = "Low", "Sell vol"
        elif rv <= th["high"]:
            regime, stance = "Moderate", "Neutral / selective"
        else:
            regime, stance = "High", "Reduce short vol"
        rows.append((sym, regime, rv, stance))
    return rows


# ============================================================================
# STRATEGIC PLAYBOOK
# ============================================================================

def get_strategy_recommendation(current_regime: str, current_rv: float, days_in_regime: int, breakout_prob: float, shock_info: tuple, asset: str = "BTC") -> list[dict]:
    recommendations = []
    thresholds = VOL_THRESHOLDS.get(asset, {"low": LOW_VOL_THRESHOLD, "high": HIGH_VOL_THRESHOLD})

    if current_regime == "Low":
        if current_rv < thresholds["low"]:
            recommendations.append({
                "title": "Strategy: Pivot to Carry/Yield",
                "message": "The 'Compression Spring' is loading. Consider Iron Condors and yield-seeking strategies.",
                "priority": "high" if days_in_regime > BREAKOUT_PROBABILITY_THRESHOLD else "medium",
            })
        if breakout_prob > 70:
            recommendations.append({
                "title": "High Breakout Probability",
                "message": f"Market has been in Low Vol regime for {days_in_regime} days. Historical average is 45-60 days. Prepare for potential regime shift.",
                "priority": "high",
            })
    elif current_regime == "High":
        recommendations.append({
            "title": "Strategy: Pivot to Momentum/Trend Following",
            "message": "The 'Squeeze' has triggered. Do not sell volatility. Focus on trend-following strategies.",
            "priority": "high",
        })

    if current_regime in ("Moderate", "High") and breakout_prob > 60:
        recommendations.append({
            "title": "Elevated Return-to-Low Probability",
            "message": f"Probability of returning to Low vol is {breakout_prob:.0f}%. Volatility may mean-revert; consider positioning for compression.",
            "priority": "medium",
        })

    if shock_info[0] is not None:
        shock_date, shock_mag, persistence_days = shock_info
        if persistence_days > 0:
            recommendations.append({
                "title": "Volatility Shock Detected",
                "message": f"Major shock ({shock_mag:.1f}%) on {shock_date.strftime('%Y-%m-%d')}. Elevated volatility expected for {persistence_days:.0f} more days (6.6-day half-life).",
                "priority": "medium",
            })
    return recommendations


def get_options_positioning(current_regime: str, days_in_regime: int, breakout_prob: float, shock_info: tuple, term_structure_slope: float | None, asset: str = "BTC") -> list[tuple[str, str]]:
    bullets = []
    if current_regime == "Low":
        bullets.append(("Vega", "Reduce short vega; consider flattening before breakout." if breakout_prob >= 70 else "Favor selling vol (IV often rich vs RV in Low)."))
    elif current_regime == "High":
        bullets.append(("Vega", "Avoid or reduce short vol; consider long vol or delta-neutral trend."))
    else:
        bullets.append(("Vega", "Selective premium sell; size smaller than in Low."))

    if current_regime == "Low" and (breakout_prob >= 70 or days_in_regime > BREAKOUT_PROBABILITY_THRESHOLD):
        bullets.append(("Gamma", "Reduce short gamma; breakout risk elevated."))
    elif current_regime == "High":
        bullets.append(("Gamma", "Avoid large short gamma; prefer defined-risk or long vol."))

    inverted = term_structure_slope is not None and term_structure_slope < 0
    if inverted or (current_regime == "Low" and breakout_prob >= 70):
        bullets.append(("Tenor", "Prefer shorter-dated (7-14d) to limit re-pricing if regime shifts."))
    elif current_regime == "Low" and breakout_prob < 50:
        bullets.append(("Tenor", "Can extend to 3-7d in stable Low."))

    if current_regime == "Low":
        bullets.append(("Tactic", "Iron condors / strangles; take profit or tighten when breakout prob > 70%."))
    elif current_regime == "High":
        bullets.append(("Tactic", "No new short vol; consider long straddles/strangles or trend options."))
    if current_regime in ("Moderate", "High") and breakout_prob > 60:
        bullets.append(("Tactic", "Add short vol into spikes; target mean reversion to Low."))
    if shock_info[0] and shock_info[2] > 0:
        bullets.append(("Tactic", f"Recent shock: elevated vol for ~{shock_info[2]:.0f} more days (half-life 6.6d)."))
    return bullets


# ============================================================================
# TOP-LEVEL ORCHESTRATION
# ============================================================================

@st.cache_data(ttl=TTL_SLOW, max_entries=8, show_spinner=False)
def _get_processed_df_cached(asset: str, days_back: int) -> pd.DataFrame:
    df, err = fetch_perp_ohlc(asset, days_back)
    if err or df is None or df.empty:
        raise RegimeDataError(err or "No data received")
    df = calculate_log_returns(df)
    df = calculate_multiple_volatility_windows(df)
    df = add_regime_classification(df, asset=asset)
    df = detect_squeeze_periods(df, asset=asset)
    df = calculate_all_advanced_indicators(df, asset=asset)
    return df


@st.cache_data(ttl=TTL_SLOW, max_entries=8, show_spinner=False)
def _get_rv_only_df_cached(asset: str, days_back: int) -> pd.DataFrame:
    """Cheap RV-only frame for the top-of-page cross-asset strip — skips
    GARCH/compression/early-warning indicators (calculate_all_advanced_indicators
    is the expensive part of get_processed_df) since the strip only needs
    rv_30d. Mirrors exodus's _get_processed_rv_df_for_cross."""
    df, err = fetch_perp_ohlc(asset, days_back)
    if err or df is None or df.empty:
        raise RegimeDataError(err or "No data received")
    df = calculate_log_returns(df)
    df = calculate_multiple_volatility_windows(df)
    return df


@st.cache_data(ttl=TTL_SLOW, max_entries=4, show_spinner=False)
def get_cross_asset_rv_summary(days_back: int = 60) -> dict:
    """{asset: latest 30d RV or None} across REGIME_ASSETS, for the
    always-visible top-of-page strip. Raises only when every asset fails, so
    a partial outage still shows what did resolve."""
    values = {}
    for sym in REGIME_ASSETS:
        try:
            df = _get_rv_only_df_cached(sym, days_back)
            values[sym] = float(df["rv_30d"].iloc[-1]) if "rv_30d" in df.columns and len(df) else None
        except Exception:
            values[sym] = None
    if all(v is None for v in values.values()):
        raise RegimeDataError("no Deribit perp data available for BTC or ETH")
    return values


def get_cross_asset_rv_frames(days_back: int) -> dict[str, pd.DataFrame]:
    """{asset: rv-only df} across REGIME_ASSETS, for the full cross-asset
    30d-RV comparison chart. Assets that fail to fetch are omitted."""
    out = {}
    for sym in REGIME_ASSETS:
        try:
            out[sym] = _get_rv_only_df_cached(sym, days_back)
        except Exception:
            continue
    return out


def get_processed_df(asset: str, days_back: int) -> tuple[pd.DataFrame | None, str | None]:
    """Fetch + fully process one asset's data (returns, RV windows, regime,
    squeeze flags, GARCH/compression/early-warning/microstructure indicators).
    Returns (df, error_message)."""
    try:
        return _get_processed_df_cached(asset, days_back).copy(), None
    except RegimeDataError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, str(exc)


def compute_snapshot(asset: str, df: pd.DataFrame, rv_window_days: int = 30) -> dict:
    """Assemble everything a dashboard render or a Telegram report needs for
    one asset from an already-processed df: current regime/RV, the blended
    (raw-calibrated + Markov-P^3) transition probability, shock info, term
    structure slope, and the single shared calibration backtest used by both
    the calibration-reliability chart and the forward-outcome chart.

    This centralizes logic that exodus duplicated between render_asset_dashboard
    and _build_ccy_data_for_telegram — same computation, one call site."""
    rv_col = f"rv_{rv_window_days}d" if f"rv_{rv_window_days}d" in df.columns else "rv_30d"
    if rv_window_days != 30:
        df = add_regime_classification(df, asset=asset, rv_col=rv_col)

    current_rv = df[rv_col].iloc[-1] if not pd.isna(df[rv_col].iloc[-1]) else 0
    current_rv = float(np.clip(np.nan_to_num(current_rv, nan=0.0, posinf=200.0, neginf=0.0), 0.0, 200.0))
    current_regime = df["regime"].iloc[-1] if df["regime"].iloc[-1] != "Unknown" else "Moderate"
    days_in_regime = count_consecutive_days_in_regime(df)
    lookback_days = min(730, len(df))

    cal = get_calibration(asset, df)
    learned_weights = learned_weights_from_calibration(cal)

    raw_prob = calculate_breakout_probability(
        df, days_in_regime, current_regime, lookback_days=lookback_days, asset=asset, learned_weights=learned_weights
    )
    breakout_prob = calibrate_probability(raw_prob, cal, current_regime)
    breakout_prob = float(np.clip(np.nan_to_num(breakout_prob, nan=0.0, posinf=100.0, neginf=0.0), 0.0, 100.0))

    fwd_probs = get_forward_regime_probs_3d(df, asset=asset, days_in_regime=days_in_regime)
    if fwd_probs:
        prob_from_p3 = (
            100.0 * (fwd_probs.get("Moderate", 0) + fwd_probs.get("High", 0))
            if current_regime == "Low" else 100.0 * fwd_probs.get("Low", 0)
        )
        breakout_prob = float(np.clip(0.5 * breakout_prob + 0.5 * prob_from_p3, 0.0, 100.0))

    shock_raw = get_latest_shock_info(df, asset=asset)
    if isinstance(shock_raw, (tuple, list)) and len(shock_raw) >= 3 and shock_raw[0] is not None:
        shock_info = (shock_raw[0], shock_raw[1] if shock_raw[1] is not None else 0.0, shock_raw[2] if shock_raw[2] is not None else 0)
    else:
        shock_info = (None, 0.0, 0)

    term_structure_slope = (
        df["term_structure_slope"].iloc[-1]
        if "term_structure_slope" in df.columns and len(df) > 0 and not pd.isna(df["term_structure_slope"].iloc[-1])
        else None
    )

    backtest_breakout, backtest_rtl = _backtest_raw_scores_and_outcomes(df, asset, learned_weights=learned_weights)
    duration_forecast = forecast_regime_duration(df, current_regime, days_in_regime)

    rv_pct = np.nan
    if len(df) >= 60:
        try:
            rv_pct = df[rv_col].rolling(min(252, len(df)), min_periods=60).apply(
                lambda x: (x.iloc[-1] >= x).mean() * 100 if len(x) > 0 else np.nan, raw=False
            ).iloc[-1]
        except Exception:
            rv_pct = np.nan

    return {
        "df": df,
        "rv_col": rv_col,
        "rv_window_days": rv_window_days,
        "current_rv": current_rv,
        "current_regime": current_regime,
        "days_in_regime": days_in_regime,
        "breakout_prob": breakout_prob,
        "shock_info": shock_info,
        "fwd_probs": fwd_probs,
        "term_structure_slope": term_structure_slope,
        "calibration": cal,
        "learned_weights": learned_weights,
        "backtest_breakout": backtest_breakout,
        "backtest_rtl": backtest_rtl,
        "rv_percentile": rv_pct,
        "duration_forecast": duration_forecast,
    }
