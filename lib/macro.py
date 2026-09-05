"""
Macro Event Impact — data/math library for pages/10_Macro_Event_Impact.py.

Full port of exodus-analytics's EMCI_vFALC.py math engine (surprise
z-scores, multi-timeframe price reaction, path-dependency diagnostics,
realized vol, implied-move/move-ratio, and Vol Crush) onto this app's
Deribit-only data layer. Read in full before porting (2545 lines), per the
"verify, don't assume" precedent from pages 06/07/08 — see CLAUDE.md.

Adapted, not verbatim:
- exodus fetched IV from Amberdata's delta-surfaces API (per-DTE ATM IV,
  1W/1M term structure) and price from Binance minute klines. This app has
  no Amberdata access, so every IV metric here uses the *real* Deribit DVOL
  index (``lib.deribit.get_dvol``) instead of a reconstruction — DVOL is
  Deribit's own published vol index, a strictly better and simpler
  substitute than rebuilding Amberdata's per-tenor delta surfaces from
  ``lib/history.py``'s estimation chain at arbitrary past timestamps. Price
  comes from Deribit's own perpetual (``lib.deribit.get_tradingview_ohlc``
  at 1-minute resolution), matching every other page in this app.
- **BTC/ETH only** — exodus's own ``ASSETS`` list was already BTC/ETH-only
  (DVOL only exists for these two on Deribit), so this is not a new scope
  narrowing, just carrying forward the same constraint pages 07/08 needed
  to have explained to them.
- The bundled calendar here (``data/macro_events_calendar.csv``) is this
  app's simpler pre-existing format (date, event, actual, consensus, prior,
  currency) — exodus's calendar additionally tracked a release *time* and a
  ``consensus_basis`` provenance tag (survey median vs. model nowcast vs.
  prior-release random walk) so surprises measured different ways were
  never pooled into one z-score. Reproducing that provenance tracking would
  mean re-curating years of historical consensus data from scratch, which
  is out of scope here — so z-scores below are computed per event type only
  (one basis, implicitly "whatever `consensus` in the CSV means"), and the
  caller should treat that as a known simplification, not silently assume
  parity with exodus's more careful version. Release *time* is recovered
  instead via ``RELEASE_TIME_ET`` (below) — these US macro releases keep a
  small number of fixed, well-known ET release times, so a per-event-type
  lookup (DST-aware ET->UTC) restores the minute-level T0 exodus had from
  its own explicit ``time`` column.
- Dropped as out of portable scope (no data source, or not a metric):
  Amberdata's 1W/1M ATM term-structure change chart, the Wikipedia
  hover-context enrichment, the Bloomberg-style multi-span reaction chart,
  the event-spider chart, and the event-timeline-candles chart. Kept
  everything that measures the actual price/vol reaction to a release,
  which is the substance of the page.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import sqrt

import numpy as np
import pandas as pd
import streamlit as st

from lib import deribit
from lib.constants import ASSET_CONFIG, TTL_MEDIUM, TTL_SLOW

try:
    import zoneinfo
    ET = zoneinfo.ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - zoneinfo is stdlib on py>=3.9
    import pytz
    ET = pytz.timezone("America/New_York")

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MACRO_ASSETS = [a for a in ("BTC", "ETH") if ASSET_CONFIG.get(a, {}).get("has_dvol")]

TIMEFRAME_OPTIONS = ["1m", "5m", "15m", "1h", "4h", "24h"]
TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "24h": 1440}
MINUTES_PER_DAY = 24 * 60

# Vol Crush horizons (hours post-release)
VOL_CRUSH_HOURS = [1, 4, 24, 48, 72]

# Well-known fixed ET release times for these recurring US macro prints.
# Falls back to the 8:30am slot (CPI/NFP/PCE/PPI/GDP/Retail Sales are all
# released then) for any event type not explicitly listed.
RELEASE_TIME_ET = {
    "FOMC Rate Decision": "14:00",
}
DEFAULT_RELEASE_TIME_ET = "08:30"

MIN_SURPRISE_SAMPLES_FOR_STD = 3


# ---------------------------------------------------------------------------
# Calendar loading + surprise z-scores
# ---------------------------------------------------------------------------

def load_macro_calendar(csv_path) -> pd.DataFrame | None:
    """Load the bundled macro events calendar CSV and attach a release
    timestamp in UTC (DST-aware ET -> UTC via RELEASE_TIME_ET)."""
    from pathlib import Path
    path = Path(csv_path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        required = {"date", "event", "actual", "consensus", "prior", "currency"}
        if not required.issubset(set(df.columns)):
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        release_time = df["event"].map(lambda e: RELEASE_TIME_ET.get(e, DEFAULT_RELEASE_TIME_ET))
        dt_naive = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d") + " " + release_time, format="%Y-%m-%d %H:%M")
        df["release_time_utc"] = dt_naive.dt.tz_localize(ET, ambiguous="infer", nonexistent="shift_forward").dt.tz_convert(UTC)
        df = df.sort_values("date", ascending=False).reset_index(drop=True)
        return df
    except Exception:
        return None


def compute_surprise_zscores(df: pd.DataFrame, zscore_years: int | None = None) -> pd.DataFrame:
    """Add surprise/surprise_zscore columns. Z is computed per event type,
    against an expanding standard deviation (so early history isn't scored
    against a full-sample std it couldn't have known at the time), falling
    back to the full-sample std once there's enough of it, and reporting
    z=0 (not NaN) for a genuinely zero surprise even when std has collapsed
    to zero (e.g. a run of FOMC decisions that all matched consensus) —
    mirrors exodus's ``compute_surprise_zscore``, vectorized per group.

    ``zscore_years``, if given, restricts the standard-deviation reference
    window to the trailing N years so one outsized historical surprise (a
    payrolls collapse, a shock hike) doesn't permanently compress every
    later z-score — falls back to the full history if that window is too
    thin (<``MIN_SURPRISE_SAMPLES_FOR_STD`` rows) to say anything."""
    df = df.copy()
    df["actual_num"] = pd.to_numeric(df["actual"], errors="coerce")
    df["consensus_num"] = pd.to_numeric(df["consensus"], errors="coerce")
    df["prior_num"] = pd.to_numeric(df["prior"], errors="coerce")
    df["surprise"] = df["actual_num"] - df["consensus_num"]

    ref = df
    if zscore_years is not None:
        cutoff = pd.Timestamp.now(tz=UTC) - pd.Timedelta(days=int(365.25 * zscore_years))
        windowed = df[df["release_time_utc"] >= cutoff]
        if windowed.groupby("event")["surprise"].count().max(default=0) >= MIN_SURPRISE_SAMPLES_FOR_STD:
            ref = windowed

    df = df.sort_values("date")
    ref_by_event = {evt: g.sort_values("date")["surprise"] for evt, g in ref.groupby("event")}

    zscores = []
    for _, row in df.iterrows():
        evt, surprise = row["event"], row["surprise"]
        hist = ref_by_event.get(evt)
        if hist is None or pd.isna(surprise):
            zscores.append(np.nan)
            continue
        prior_vals = hist[hist.index <= row.name] if row.name in hist.index else hist
        std = prior_vals.expanding().std().iloc[-1] if len(prior_vals) else np.nan
        if pd.isna(std) or std == 0:
            std = hist.std()
        if pd.isna(std) or std == 0:
            zscores.append(0.0 if surprise == 0 else np.nan)
        else:
            zscores.append(float(surprise / std))
    df["surprise_zscore"] = zscores
    return df.sort_values("date", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-event data fetch (Deribit perp 1m OHLC + hourly DVOL, real history)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=TTL_MEDIUM, max_entries=64, show_spinner=False)
def fetch_event_ohlc(asset: str, event_ts_ms: int, window_before_h: int, window_after_h: int) -> pd.DataFrame | None:
    """1-minute perp OHLC in [T0-window_before_h, T0+window_after_h]."""
    cfg = ASSET_CONFIG.get(asset)
    if not cfg:
        return None
    start_ms = event_ts_ms - window_before_h * 3600 * 1000
    end_ms = event_ts_ms + window_after_h * 3600 * 1000
    df = deribit.get_tradingview_ohlc(cfg["perp"], "1", start_ms, end_ms)
    if df is None or df.empty:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


@st.cache_data(ttl=TTL_SLOW, max_entries=64, show_spinner=False)
def fetch_event_dvol(asset: str, event_ts_ms: int, window_before_h: int, window_after_h: int) -> pd.Series | None:
    """Hourly real Deribit DVOL index in [T0-window_before_h-1, T0+window_after_h]
    (padded an extra hour back so a T-1h baseline lookup always has a candidate)."""
    cfg = ASSET_CONFIG.get(asset)
    if not cfg or not cfg.get("has_dvol"):
        return None
    start_ms = event_ts_ms - (window_before_h + 1) * 3600 * 1000
    end_ms = event_ts_ms + window_after_h * 3600 * 1000
    df = deribit.get_dvol(cfg["deribit_ccy"], resolution="60", start_ms=start_ms, end_ms=end_ms)
    if df is None or df.empty:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")["close"].sort_index()


# ---------------------------------------------------------------------------
# Path / reaction math — direct ports of exodus's per-event functions
# ---------------------------------------------------------------------------

def _align_ts(index: pd.Index, ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if not isinstance(index, pd.DatetimeIndex):
        return ts
    if index.tz is None:
        return ts.tz_convert(UTC).tz_localize(None) if ts.tz is not None else ts
    return ts.tz_localize(index.tz) if ts.tz is None else ts.tz_convert(index.tz)


def compute_pct_changes(ohlc: pd.DataFrame | None, release_ts, windows_minutes: dict) -> dict:
    """Price % change from T0 to T+window, for each named window."""
    if ohlc is None or ohlc.empty:
        return {k: np.nan for k in windows_minutes}
    idx = ohlc.index
    release_cmp = _align_ts(idx, release_ts)
    pre = np.asarray(idx <= release_cmp)
    if not pre.any():
        return {k: np.nan for k in windows_minutes}
    price_t0 = float(ohlc.iloc[pre].iloc[-1]["close"])
    if price_t0 <= 0:
        return {k: np.nan for k in windows_minutes}
    out = {}
    for name, w_min in windows_minutes.items():
        end_cmp = _align_ts(idx, release_ts + timedelta(minutes=w_min))
        pos = idx.searchsorted(end_cmp, side="left")
        if pos >= len(ohlc):
            out[name] = np.nan
            continue
        price_end = float(ohlc.iloc[pos]["close"])
        out[name] = (price_end - price_t0) / price_t0 * 100
    return out


def compute_max_up_down(ohlc: pd.DataFrame | None, release_ts, window_minutes: int) -> tuple[float, float, float]:
    """Max up %, max down %, range % within [T0, T0+window]."""
    if ohlc is None or ohlc.empty:
        return np.nan, np.nan, np.nan
    idx = ohlc.index
    release_cmp = _align_ts(idx, release_ts)
    end_cmp = _align_ts(idx, release_ts + timedelta(minutes=window_minutes))
    mask = np.asarray((idx >= release_cmp) & (idx <= end_cmp))
    seg = ohlc.iloc[mask]
    if seg.empty:
        return np.nan, np.nan, np.nan
    pre_arr = np.asarray(idx <= release_cmp)
    price_t0 = ohlc.iloc[pre_arr].iloc[-1]["close"] if np.any(pre_arr) else seg["close"].iloc[0]
    if price_t0 is None or price_t0 <= 0 or not np.isfinite(price_t0):
        return np.nan, np.nan, np.nan
    high, low = seg["high"].max(), seg["low"].min()
    return (float((high - price_t0) / price_t0 * 100),
            float((low - price_t0) / price_t0 * 100),
            float((high - low) / price_t0 * 100))


def compute_pre_event_drift(ohlc: pd.DataFrame | None, release_ts, window_minutes: int = 240) -> float:
    """% change from T-window to T0 (default T-4h -> T0)."""
    if ohlc is None or ohlc.empty:
        return np.nan
    idx = ohlc.index
    release_cmp = _align_ts(idx, release_ts)
    start_cmp = _align_ts(idx, release_ts - timedelta(minutes=window_minutes))
    seg = ohlc.iloc[np.asarray((idx >= start_cmp) & (idx <= release_cmp))]
    if len(seg) < 2 or seg["close"].iloc[0] <= 0:
        return np.nan
    return float((seg["close"].iloc[-1] - seg["close"].iloc[0]) / seg["close"].iloc[0] * 100)


def compute_path_dependency_details(ohlc: pd.DataFrame | None, release_ts, window_minutes: int) -> dict:
    """Within [T0, T0+window]: which extreme (high/low) came first, how long
    each took to arrive, and the larger of the two excursions."""
    empty = {"first_extreme": np.nan, "time_to_high_min": np.nan, "time_to_low_min": np.nan, "max_excursion_pct": np.nan}
    if ohlc is None or ohlc.empty:
        return empty
    idx = ohlc.index
    release_cmp = _align_ts(idx, release_ts)
    end_cmp = _align_ts(idx, release_ts + timedelta(minutes=window_minutes))
    seg = ohlc.iloc[np.asarray((idx >= release_cmp) & (idx <= end_cmp))]
    pre = np.asarray(idx <= release_cmp)
    if seg.empty or not pre.any():
        return empty
    price_t0 = float(ohlc.iloc[pre].iloc[-1]["close"])
    if price_t0 <= 0:
        return empty
    high_val, low_val = float(seg["high"].max()), float(seg["low"].min())
    ts_high, ts_low = seg["high"].idxmax(), seg["low"].idxmin()
    time_to_high = (pd.Timestamp(ts_high) - pd.Timestamp(release_cmp)) / np.timedelta64(1, "m")
    time_to_low = (pd.Timestamp(ts_low) - pd.Timestamp(release_cmp)) / np.timedelta64(1, "m")
    if pd.isna(time_to_high) or pd.isna(time_to_low):
        first_extreme = np.nan
    elif time_to_high < time_to_low:
        first_extreme = "Up first"
    elif time_to_low < time_to_high:
        first_extreme = "Down first"
    else:
        first_extreme = "Same bar"
    max_up = (high_val - price_t0) / price_t0 * 100
    max_down = (low_val - price_t0) / price_t0 * 100
    return {
        "first_extreme": first_extreme,
        "time_to_high_min": float(time_to_high),
        "time_to_low_min": float(time_to_low),
        "max_excursion_pct": float(max(abs(max_up), abs(max_down))),
    }


def compute_realized_vol_window(ohlc: pd.DataFrame | None, release_ts, window_minutes: int) -> float:
    """Annualized close-to-close realized vol (%) within [T0, T0+window],
    assuming 1-minute bars (this module always fetches at that resolution)."""
    if ohlc is None or ohlc.empty:
        return np.nan
    idx = ohlc.index
    release_cmp = _align_ts(idx, release_ts)
    end_cmp = _align_ts(idx, release_ts + timedelta(minutes=window_minutes))
    seg = ohlc.iloc[np.asarray((idx >= release_cmp) & (idx <= end_cmp))]
    closes = seg["close"].astype(float).replace(0, np.nan).dropna() if not seg.empty else pd.Series(dtype=float)
    if len(closes) < 3:
        return np.nan
    log_ret = np.log(closes).diff().dropna()
    if len(log_ret) < 2:
        return np.nan
    bars_per_year = 365 * 24 * 60  # 1-minute bars
    return float(log_ret.std(ddof=1) * np.sqrt(bars_per_year) * 100)


def implied_move_pct(dvol_pct: float | None, window_minutes: int) -> float:
    """Scale an annualized DVOL level (vol points, e.g. 55.0 = 55%) down to
    an implied move over a shorter window: daily_vol = DVOL/sqrt(365);
    implied_move = daily_vol * sqrt(window/1440) * 100."""
    if dvol_pct is None or not np.isfinite(dvol_pct) or dvol_pct <= 0:
        return np.nan
    daily_vol = (dvol_pct / 100.0) / sqrt(365)
    return daily_vol * sqrt(window_minutes / MINUTES_PER_DAY) * 100


def move_ratio(actual_pct: float, implied_pct: float) -> float:
    if implied_pct is None or not np.isfinite(implied_pct) or implied_pct == 0:
        return np.nan
    return actual_pct / implied_pct


def _nearest_dvol(dvol_series: pd.Series | None, target_ts, max_gap_hours: float = 3.0) -> float | None:
    """Nearest DVOL sample to target_ts, rejected if more than max_gap_hours away
    (avoids silently matching a far-off point when history doesn't reach back)."""
    if dvol_series is None or dvol_series.empty:
        return None
    idx = dvol_series.index
    pos = idx.get_indexer([pd.Timestamp(target_ts)], method="nearest")[0]
    if pos < 0:
        return None
    found = idx[pos]
    if abs((found - pd.Timestamp(target_ts)).total_seconds()) > max_gap_hours * 3600:
        return None
    return float(dvol_series.iloc[pos])


def compute_dvol_crush(dvol_series: pd.Series | None, release_ts) -> dict:
    """DVOL level ~1h pre-release, and % crush (or expansion, if negative) at
    each of VOL_CRUSH_HOURS relative to that baseline."""
    before = _nearest_dvol(dvol_series, pd.Timestamp(release_ts) - timedelta(hours=1))
    out = {"before": before}
    for h in VOL_CRUSH_HOURS:
        if before is None or before <= 0:
            out[h] = np.nan
            continue
        after = _nearest_dvol(dvol_series, pd.Timestamp(release_ts) + timedelta(hours=h))
        out[h] = float((before - after) / before * 100) if after is not None else np.nan
    return out


# ---------------------------------------------------------------------------
# Impact table assembly
# ---------------------------------------------------------------------------

def build_impact_table(events: pd.DataFrame, asset: str, timeframe: str, window_minutes: int,
                        window_before_h: int, window_after_h: int) -> pd.DataFrame:
    """One row per past event: surprise/z-score plus every price/vol reaction
    metric, built from real Deribit data fetched per event."""
    rows = []
    for _, row in events.iterrows():
        t0 = row["release_time_utc"]
        t0_ms = int(pd.Timestamp(t0).timestamp() * 1000)
        ohlc = fetch_event_ohlc(asset, t0_ms, window_before_h, window_after_h)
        dvol = fetch_event_dvol(asset, t0_ms, window_before_h, window_after_h)

        pct = compute_pct_changes(ohlc, t0, TIMEFRAME_MINUTES)
        actual_pct = pct.get(timeframe, np.nan)
        crush = compute_dvol_crush(dvol, t0)
        dvol_before = crush.get("before")
        impl = implied_move_pct(dvol_before, window_minutes)
        mr = move_ratio(actual_pct, impl)
        max_up, max_down, range_pct = compute_max_up_down(ohlc, t0, window_minutes)
        pre_drift = compute_pre_event_drift(ohlc, t0, 240)
        path = compute_path_dependency_details(ohlc, t0, window_minutes)
        rv_window = compute_realized_vol_window(ohlc, t0, window_minutes)

        surprise = row.get("surprise")
        if pd.isna(surprise):
            expectation_bucket = "No expectations data"
        elif surprise > 0:
            expectation_bucket = "Above expectations"
        elif surprise < 0:
            expectation_bucket = "Below expectations"
        else:
            expectation_bucket = "In line"
        if pd.isna(surprise) or pd.isna(actual_pct) or surprise == 0 or actual_pct == 0:
            response_alignment = "N/A"
        else:
            response_alignment = "Aligned" if np.sign(surprise) == np.sign(actual_pct) else "Diverged"

        rows.append({
            "date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
            "event": row["event"],
            "actual": row.get("actual_num"),
            "consensus": row.get("consensus_num"),
            "surprise": surprise,
            "expectation_bucket": expectation_bucket,
            "response_alignment": response_alignment,
            "z_score": row.get("surprise_zscore"),
            "dvol_before": dvol_before,
            "implied_move_pct": impl,
            "actual_move_pct": actual_pct,
            "move_ratio": mr,
            "dvol_crush_1h": crush.get(1),
            "dvol_crush_4h": crush.get(4),
            "dvol_crush_24h": crush.get(24),
            "dvol_crush_48h": crush.get(48),
            "dvol_crush_72h": crush.get(72),
            "max_up_pct": max_up,
            "max_down_pct": max_down,
            "range_pct": range_pct,
            "pre_event_drift_pct": pre_drift,
            "first_extreme": path["first_extreme"],
            "time_to_high_min": path["time_to_high_min"],
            "time_to_low_min": path["time_to_low_min"],
            "max_excursion_pct": path["max_excursion_pct"],
            "realized_vol_window_pct": rv_window,
            "rv_dvol_ratio": (rv_window / dvol_before) if (pd.notna(rv_window) and dvol_before) else np.nan,
            "has_ohlc": ohlc is not None and not ohlc.empty,
            "has_dvol": dvol is not None and not dvol.empty,
        })
    return pd.DataFrame(rows)


def summary_kpis(df: pd.DataFrame) -> dict:
    """High-level summary stats for the top of the page."""
    if df.empty:
        return {}
    aligned_mask = df["response_alignment"].isin(["Aligned", "Diverged"])
    mean_t_high = df["time_to_high_min"].mean()
    mean_t_low = df["time_to_low_min"].mean()
    return {
        "events": int(len(df)),
        "avg_abs_actual": float(df["actual_move_pct"].abs().mean()),
        "avg_implied": float(df["implied_move_pct"].mean()),
        "priced_outside_pct": float((df["move_ratio"] > 1).mean() * 100) if df["move_ratio"].notna().any() else np.nan,
        "aligned_pct": float((df.loc[aligned_mask, "response_alignment"] == "Aligned").mean() * 100) if aligned_mask.any() else np.nan,
        "up_first_pct": float((df["first_extreme"] == "Up first").mean() * 100),
        "down_first_pct": float((df["first_extreme"] == "Down first").mean() * 100),
        "first_touch": ("High first" if pd.notna(mean_t_high) and pd.notna(mean_t_low) and mean_t_high < mean_t_low else "Low first")
                       if pd.notna(mean_t_high) or pd.notna(mean_t_low) else "—",
    }


def expectations_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """How the market reacted, grouped by expectation bucket."""
    valid = df[df["expectation_bucket"] != "No expectations data"]
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby("expectation_bucket", as_index=False)
        .agg(events=("date", "count"), mean_surprise=("surprise", "mean"),
             mean_actual_move_pct=("actual_move_pct", "mean"), mean_move_ratio=("move_ratio", "mean"),
             aligned_pct=("response_alignment", lambda s: (s == "Aligned").mean() * 100))
        .sort_values("events", ascending=False)
    )
