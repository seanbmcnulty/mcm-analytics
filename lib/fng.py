"""
Crypto Fear & Greed Index client (alternative.me) — MCM Analytics.

This is the one deliberate exception to the app's "Deribit-only" data
policy: Deribit has no fear/greed-style sentiment index, and alternative.me's
Crypto Fear & Greed Index is the standard free, no-auth, crypto-specific
source for it (daily values back to 2018-02-01). Kept in its own module,
same shape as lib/deribit.py (plain functions, retry, no Streamlit import),
so it's independently testable and easy to swap out later if needed.

API docs: https://alternative.me/crypto/fear-and-greed-index/
Endpoint: https://api.alternative.me/fng/?limit=N&format=json
  limit=0 returns full history. Each row: value (0-100, string),
  value_classification (str), timestamp (unix seconds, string).
"""

import time
from typing import Any

import requests
import pandas as pd

BASE_URL = "https://api.alternative.me/fng/"
DEFAULT_TIMEOUT = 15  # seconds

# Standard alternative.me classification bucket edges (value <= edge).
# Exposed so callers can use them as slider defaults.
DEFAULT_THRESHOLDS = {
    "extreme_fear": 25,   # value < 25          -> Extreme Fear
    "fear": 45,            # 25 <= value < 45     -> Fear
    "greed": 55,            # 45 <= value <= 55    -> Neutral, > 55 starts Greed
    "extreme_greed": 75,   # 55 < value <= 75     -> Greed, > 75 -> Extreme Greed
}

BUCKET_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

# Suggested contrarian signal direction per bucket: +1 = lean long,
# -1 = lean short, 0 = no lean. Extreme buckets carry the strongest lean.
BUCKET_SIGNAL = {
    "Extreme Fear": 1.0,
    "Fear": 0.5,
    "Neutral": 0.0,
    "Greed": -0.5,
    "Extreme Greed": -1.0,
}

BUCKET_COLORS = {
    "Extreme Fear": "#C0392B",
    "Fear": "#E67E22",
    "Neutral": "#888888",
    "Greed": "#27AE60",
    "Extreme Greed": "#1E8449",
}


def classify(value: float, thresholds: dict | None = None) -> str:
    """Bucket a 0-100 F&G value into one of BUCKET_ORDER using edge
    thresholds (defaults to alternative.me's own convention)."""
    t = thresholds or DEFAULT_THRESHOLDS
    if value is None or pd.isna(value):
        return "Unknown"
    if value < t["extreme_fear"]:
        return "Extreme Fear"
    if value < t["fear"]:
        return "Fear"
    if value <= t["greed"]:
        return "Neutral"
    if value <= t["extreme_greed"]:
        return "Greed"
    return "Extreme Greed"


def _request(params: dict, retries: int = 3) -> dict | None:
    """GET the alternative.me endpoint with basic retry/backoff. No
    in-process cache here on purpose — the page layer owns caching via
    st.cache_data (same division of labor as lib/deribit.py's `_request`
    for TTL-cached calls, kept simple since this is a single daily-moving
    series, not a hot path)."""
    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            return None
        except (requests.RequestException, ValueError):
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            continue
    return None


def get_fng_history(limit: int = 0) -> pd.DataFrame | None:
    """
    Fetch Fear & Greed Index history.

    limit=0 fetches the full history (back to 2018-02-01). Returns a
    DataFrame sorted ascending by date with columns:
      timestamp (UTC, tz-naive, midnight-aligned), value (float 0-100),
      classification (str, alternative.me's own label).
    Returns None on total failure (network unreachable, bad response
    shape) so callers can show a clear "data unavailable" state rather
    than silently rendering an empty page.
    """
    payload = _request({"limit": limit, "format": "json"})
    if not payload or "data" not in payload or not payload["data"]:
        return None

    rows = payload["data"]
    try:
        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["timestamp"] = pd.to_datetime(
            pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True
        ).dt.tz_convert(None)
        df = df.rename(columns={"value_classification": "classification"})
        df = df[["timestamp", "value", "classification"]].dropna(subset=["value", "timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        # Normalize to midnight so it joins cleanly against daily OHLC dates.
        df["timestamp"] = df["timestamp"].dt.normalize()
        df = df.drop_duplicates(subset="timestamp", keep="last")
        return df
    except Exception:
        return None
