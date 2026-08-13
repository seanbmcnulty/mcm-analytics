"""
MCM Analytics — Volatility mathematics and realized vol estimators.

Shared computation functions used across multiple pages.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Realized Volatility Estimators
# ---------------------------------------------------------------------------

def close_to_close_vol(prices: pd.Series, window: int = 30) -> pd.Series:
    """Standard close-to-close realized volatility (annualized)."""
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window).std() * np.sqrt(365)


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 30) -> pd.Series:
    """Parkinson (1980) high-low range estimator (annualized)."""
    hl = np.log(high / low) ** 2
    factor = 1 / (4 * np.log(2))
    return np.sqrt(hl.rolling(window).mean() * factor * 365)


def garman_klass_vol(open_: pd.Series, high: pd.Series, low: pd.Series,
                     close: pd.Series, window: int = 30) -> pd.Series:
    """Garman-Klass (1980) OHLC volatility estimator (annualized)."""
    hl = 0.5 * np.log(high / low) ** 2
    co = -(2 * np.log(2) - 1) * np.log(close / open_) ** 2
    gk = hl + co
    return np.sqrt(gk.rolling(window).mean() * 365)


def yang_zhang_vol(open_: pd.Series, high: pd.Series, low: pd.Series,
                   close: pd.Series, window: int = 30) -> pd.Series:
    """Yang-Zhang (2000) volatility estimator (annualized)."""
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    log_oc = np.log(open_ / close.shift(1))
    log_co = np.log(close / open_)
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)

    # Overnight variance
    sigma_o = log_oc.rolling(n).var()
    # Close-to-open variance
    sigma_c = log_co.rolling(n).var()
    # Rogers-Satchell
    rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(n).mean()

    sigma2 = sigma_o + k * sigma_c + (1 - k) * rs
    return np.sqrt(np.maximum(sigma2, 0) * 365)


def rogers_satchell_vol(open_: pd.Series, high: pd.Series, low: pd.Series,
                        close: pd.Series, window: int = 30) -> pd.Series:
    """Rogers-Satchell volatility estimator (annualized)."""
    log_ho = np.log(high / open_)
    log_hc = np.log(high / close)
    log_lo = np.log(low / open_)
    log_lc = np.log(low / close)
    rs = (log_ho * log_hc + log_lo * log_lc).rolling(window).mean()
    return np.sqrt(np.maximum(rs, 0) * 365)


RV_ESTIMATORS = {
    "Close-to-Close": close_to_close_vol,
    "Parkinson": parkinson_vol,
    "Garman-Klass": garman_klass_vol,
    "Yang-Zhang": yang_zhang_vol,
    "Rogers-Satchell": rogers_satchell_vol,
}


def compute_rv_matrix(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """
    Compute all RV estimators for given OHLC data across multiple windows.
    df must have columns: open, high, low, close
    Returns DataFrame with columns: estimator, window, value
    """
    if windows is None:
        windows = [7, 14, 30, 60, 90]

    results = []
    for name, func in RV_ESTIMATORS.items():
        for w in windows:
            if name == "Close-to-Close":
                series = func(df["close"], window=w)
            elif name == "Parkinson":
                series = func(df["high"], df["low"], window=w)
            else:
                series = func(df["open"], df["high"], df["low"], df["close"], window=w)
            latest = series.iloc[-1] if len(series) > 0 else np.nan
            results.append({"estimator": name, "window": w, "value": latest * 100})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Black-Scholes Helpers
# ---------------------------------------------------------------------------

def bs_delta(spot: float, strike: float, tte: float, vol: float,
             option_type: str = "C", r: float = 0.0) -> float:
    """Black-Scholes delta for a European option."""
    if tte <= 0 or vol <= 0:
        return 0.0
    d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * tte) / (vol * np.sqrt(tte))
    if option_type == "C":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def bs_price(spot: float, strike: float, tte: float, vol: float,
             option_type: str = "C", r: float = 0.0) -> float:
    """Black-Scholes price for a European option."""
    if tte <= 0:
        if option_type == "C":
            return max(spot - strike, 0)
        return max(strike - spot, 0)
    if vol <= 0:
        vol = 0.001

    d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * tte) / (vol * np.sqrt(tte))
    d2 = d1 - vol * np.sqrt(tte)

    if option_type == "C":
        return spot * norm.cdf(d1) - strike * np.exp(-r * tte) * norm.cdf(d2)
    return strike * np.exp(-r * tte) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def bs_gamma(spot: float, strike: float, tte: float, vol: float,
             r: float = 0.0) -> float:
    """Black-Scholes gamma."""
    if tte <= 0 or vol <= 0:
        return 0.0
    d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * tte) / (vol * np.sqrt(tte))
    return norm.pdf(d1) / (spot * vol * np.sqrt(tte))


def bs_vega(spot: float, strike: float, tte: float, vol: float,
            r: float = 0.0) -> float:
    """Black-Scholes vega (per 1% vol move)."""
    if tte <= 0 or vol <= 0:
        return 0.0
    d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * tte) / (vol * np.sqrt(tte))
    return spot * norm.pdf(d1) * np.sqrt(tte) * 0.01


def bs_theta(spot: float, strike: float, tte: float, vol: float,
             option_type: str = "C", r: float = 0.0) -> float:
    """Black-Scholes theta (per day)."""
    if tte <= 0 or vol <= 0:
        return 0.0
    d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * tte) / (vol * np.sqrt(tte))
    d2 = d1 - vol * np.sqrt(tte)
    term1 = -(spot * norm.pdf(d1) * vol) / (2 * np.sqrt(tte))
    if option_type == "C":
        term2 = -r * strike * np.exp(-r * tte) * norm.cdf(d2)
    else:
        term2 = r * strike * np.exp(-r * tte) * norm.cdf(-d2)
    return (term1 + term2) / 365
