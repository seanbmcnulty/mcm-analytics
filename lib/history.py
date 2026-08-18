"""
MCM Analytics — historical volatility reconstruction from Deribit only.

The FalconX bot sourced every historical vol series from Amberdata's
constant-maturity delta surface.  Deribit's public API serves no option-IV
history, so this module rebuilds an equivalent from what Deribit *does* expose,
using a three-tier preference chain:

  1. **Snapshot store** - hourly surface snapshots this app has recorded itself
     (``data/snapshots``).  True history, but only covers the period since the
     app started running.
  2. **DVOL-anchored reconstruction** - Deribit's DVOL index is a genuine,
     continuous hourly series of 30-day ATM implied vol (BTC and ETH only).
     The current surface shape is re-levelled through time by the ratio
     ``DVOL(t) / DVOL(now)``.  Shape is held fixed, level is real.
  3. **Realized-vol proxy** - for assets with no DVOL (SOL, HYPE), the same
     re-levelling driven by a rolling Parkinson RV ratio from perp candles.

Anything produced by tiers 2 or 3 is flagged ``estimated=True`` so the UI can
label it honestly rather than passing it off as recorded history.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lib import deribit, surface
from lib.constants import ASSET_CONFIG

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"

# Tenors we persist / reconstruct, in days.
SNAPSHOT_DTES = [1, 7, 14, 30, 60, 90, 180, 365]
_FIELDS = ("atm", "call10", "call25", "call35", "put10", "put25", "put35")

_DELTA_FIELD = {
    "delta50": "atm",
    "deltaCall10": "call10", "deltaCall25": "call25", "deltaCall35": "call35",
    "deltaPut10": "put10", "deltaPut25": "put25", "deltaPut35": "put35",
}


# ---------------------------------------------------------------------------
# Perp candles (replaces the Binance klines the FalconX bot used)
# ---------------------------------------------------------------------------

def perp_ohlc(asset: str, days: int = 120,
              resolution: str = "1D") -> pd.DataFrame | None:
    """
    OHLC candles for the asset's perpetual, indexed by UTC timestamp.

    ``resolution`` follows Deribit: "1D", "60" (hourly), "15", ...
    """
    cfg = ASSET_CONFIG.get(asset, ASSET_CONFIG["BTC"])
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(days * 24 * 3600 * 1000)
    df = deribit.get_tradingview_ohlc(cfg["perp"], resolution, start_ms, end_ms)
    if df is None or df.empty:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for c in ("open", "high", "low", "close"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def parkinson_rv(ohlc: pd.DataFrame, window: int,
                 periods_per_year: float = 365.0) -> pd.Series:
    """
    Parkinson high/low realized volatility, in percent.

        RV = sqrt( periods_per_year * mean_window( ln(H/L)^2 / (4 ln 2) ) ) * 100
    """
    if ohlc is None or ohlc.empty or "high" not in ohlc or "low" not in ohlc:
        return pd.Series(dtype=float)
    ratio = ohlc["high"] / ohlc["low"].replace(0, np.nan)
    var_hl = np.log(ratio) ** 2 / (4 * np.log(2))
    rolled = var_hl.rolling(window, min_periods=window).mean()
    return np.sqrt(periods_per_year * rolled) * 100.0


def latest_rv(asset: str) -> dict[str, float]:
    """Latest Parkinson RV for the windows the FalconX header showed."""
    out: dict[str, float] = {}
    daily = perp_ohlc(asset, days=120, resolution="1D")
    if daily is not None and not daily.empty:
        for key, win in (("3d", 3), ("7d", 7), ("30d", 30), ("90d", 90)):
            s = parkinson_rv(daily, win).dropna()
            if not s.empty:
                out[key] = float(s.iloc[-1])
    hourly = perp_ohlc(asset, days=3, resolution="60")
    if hourly is not None and not hourly.empty:
        s = parkinson_rv(hourly, 24, periods_per_year=365 * 24).dropna()
        if not s.empty:
            out["24hr"] = float(s.iloc[-1])
    return out


# ---------------------------------------------------------------------------
# Snapshot store
# ---------------------------------------------------------------------------

def _snapshot_path(asset: str) -> Path:
    return SNAPSHOT_DIR / f"{asset}_surface.csv"


def record_snapshot(asset: str, min_interval_s: float = 1800.0) -> bool:
    """
    Append the current surface to the snapshot store.

    No-ops when the last row is younger than ``min_interval_s``.  Returns True
    when a row was written.  Best-effort: on a read-only or ephemeral
    filesystem (Streamlit Community Cloud) this silently does nothing.
    """
    try:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = _snapshot_path(asset)
        now = datetime.now(timezone.utc)
        if path.exists():
            try:
                tail = pd.read_csv(path).tail(1)
                if not tail.empty:
                    last = pd.to_datetime(tail["timestamp"].iloc[0], utc=True)
                    if (now - last).total_seconds() < min_interval_s:
                        return False
            except Exception:
                pass

        vols = surface.option_vols_by_dte(asset)
        if not vols.get("atm"):
            return False
        row: dict[str, object] = {"timestamp": now.isoformat()}
        for field in _FIELDS:
            series = {float(k): float(v) for k, v in vols.get(field, {}).items()
                      if surface.finite(v)}
            for dte in SNAPSHOT_DTES:
                v = surface.interp_at_dte(series, float(dte)) if series else np.nan
                row[f"{field}_{dte}"] = round(float(v), 6) if surface.finite(v) else ""
        header = not path.exists()
        pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)
        return True
    except Exception:
        return False


def load_snapshots(asset: str, days: int) -> pd.DataFrame | None:
    """Recorded snapshots for the last ``days``, indexed by UTC timestamp."""
    path = _snapshot_path(asset)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or "timestamp" not in df:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        df = df[df.index >= cutoff]
        return df if len(df) >= 2 else None
    except Exception:
        return None


def _surface_from_snapshots(asset: str, delta_key: str,
                            days: int) -> pd.DataFrame | None:
    """Snapshot store -> wide frame (index = time, columns = DTE, decimal IV)."""
    snaps = load_snapshots(asset, days)
    if snaps is None:
        return None
    field = _DELTA_FIELD.get(delta_key)
    if not field:
        return None
    cols = {}
    for dte in SNAPSHOT_DTES:
        name = f"{field}_{dte}"
        if name in snaps.columns:
            s = pd.to_numeric(snaps[name], errors="coerce")
            if s.notna().sum() >= 2:
                cols[dte] = s
    if len(cols) < 2:
        return None
    out = pd.DataFrame(cols)
    out.columns = [int(c) for c in out.columns]
    return out.sort_index()


# ---------------------------------------------------------------------------
# Level driver: DVOL, else realized vol
# ---------------------------------------------------------------------------

def dvol_history(asset: str, days: int = 90,
                 resolution: str = "60") -> pd.Series | None:
    """DVOL close history in vol points, indexed by UTC timestamp."""
    cfg = ASSET_CONFIG.get(asset, ASSET_CONFIG["BTC"])
    if not cfg.get("has_dvol"):
        return None
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(days * 24 * 3600 * 1000)
    df = deribit.get_dvol(cfg["deribit_ccy"], resolution=resolution,
                          start_ms=start_ms, end_ms=end_ms)
    if df is None or df.empty:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    s = pd.to_numeric(df.set_index("timestamp")["close"], errors="coerce").dropna()
    if s.empty:
        return None
    if s.median() < 5:                       # tolerate a decimal-shaped feed
        s = s * 100.0
    s.index = s.index.floor("min")           # kill sub-minute drift between calls
    return s[~s.index.duplicated(keep="last")].sort_index()


def _rv_level_series(asset: str, days: int) -> pd.Series | None:
    """Rolling 7-day Parkinson RV, used as the level driver when DVOL is absent."""
    hourly = perp_ohlc(asset, days=days + 10, resolution="60")
    if hourly is None or hourly.empty:
        return None
    rv = parkinson_rv(hourly, 24 * 7, periods_per_year=365 * 24).dropna()
    if rv.empty:
        return None
    rv.index = rv.index.floor("min")
    rv = rv[~rv.index.duplicated(keep="last")]
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rv = rv[rv.index >= cutoff]
    return rv if len(rv) >= 2 else None


_DRIVER_CACHE: dict[tuple[str, int], tuple[float, tuple]] = {}
_DRIVER_TTL = 300.0


def level_driver(asset: str, days: int = 90) -> tuple[pd.Series | None, str]:
    """
    The series used to re-level the current surface through time.

    Cached per (asset, days) so every delta bucket shares one identical index —
    without this each bucket would fetch its own copy and the timestamps would
    not line up when the series are differenced.

    Returns ``(series_in_vol_points, source_label)``.
    """
    key = (asset, int(days))
    hit = _DRIVER_CACHE.get(key)
    if hit is not None and (time.time() - hit[0]) < _DRIVER_TTL:
        return hit[1]

    result = _compute_level_driver(asset, days)
    _DRIVER_CACHE[key] = (time.time(), result)
    return result


def _compute_level_driver(asset: str, days: int) -> tuple[pd.Series | None, str]:
    dv = dvol_history(asset, days=days, resolution="60")
    if dv is not None and len(dv) >= 2:
        return dv, "DVOL"
    rv = _rv_level_series(asset, days)
    if rv is not None:
        return rv, "RV"
    return None, "none"


# ---------------------------------------------------------------------------
# The public surface-history entry point
# ---------------------------------------------------------------------------

def surface_history(asset: str, delta_key: str = "delta50",
                    days: int = 90) -> tuple[pd.DataFrame | None, bool, str]:
    """
    Historical IV surface: index = UTC timestamp, columns = DTE, values decimal IV.

    Returns ``(frame, estimated, source_label)``.  ``estimated`` is True when
    the frame was reconstructed by re-levelling rather than recorded.
    """
    recorded = _surface_from_snapshots(asset, delta_key, days)
    if recorded is not None and len(recorded) >= 8:
        return recorded, False, "recorded snapshots"

    live = surface.iv_by_dte_for_delta(asset, delta_key)
    live = {float(k): float(v) for k, v in live.items() if surface.finite(v)}
    if len(live) < 2:
        return None, True, "none"

    driver, label = level_driver(asset, days=days)
    if driver is None or len(driver) < 2:
        return None, True, "none"

    cur_level = float(driver.iloc[-1])
    if not np.isfinite(cur_level) or cur_level <= 0:
        return None, True, "none"
    ratio = (driver / cur_level).clip(lower=0.2, upper=5.0)

    dtes = sorted({int(round(d)) for d in live} | set(SNAPSHOT_DTES))
    dtes = [d for d in dtes if min(live) - 1 <= d <= max(live) + 1]
    if len(dtes) < 2:
        dtes = sorted(int(round(d)) for d in live)

    cols = {}
    for d in dtes:
        base = surface.interp_at_dte(live, float(d))
        if surface.finite(base) and base > 0:
            cols[int(d)] = ratio * float(base)
    if len(cols) < 2:
        return None, True, "none"

    frame = pd.DataFrame(cols).sort_index()
    if recorded is not None:
        # Splice: recorded rows win wherever we actually have them.
        frame = frame[~frame.index.isin(recorded.index)]
        shared = [c for c in recorded.columns if c in frame.columns]
        if shared:
            frame = pd.concat([frame[shared], recorded[shared]]).sort_index()
    return frame, True, f"{label}-scaled"


def surface_row_at(frame: pd.DataFrame, when: datetime) -> pd.Series | None:
    """Row of ``frame`` nearest to ``when`` (index must be tz-aware UTC)."""
    if frame is None or frame.empty:
        return None
    try:
        idx = frame.index.get_indexer([pd.Timestamp(when)], method="nearest")[0]
    except Exception:
        return None
    idx = max(0, min(int(idx), len(frame) - 1))
    return frame.iloc[idx]


def surface_row_near(frame: pd.DataFrame,
                     when: datetime) -> tuple[pd.Series | None, pd.Timestamp | None]:
    """
    Same nearest-match lookup as ``surface_row_at``, but also returns the
    timestamp the row actually came from.

    ``get_indexer(..., method="nearest")`` always returns *some* row, even
    when the frame doesn't reach back anywhere near ``when`` (e.g. asking for
    a month ago when only three days of history exist). Callers that label
    the result with a fixed period ("1 Month Ago") need the real timestamp
    to relabel honestly and to de-duplicate requests that all resolve to the
    same earliest row.
    """
    if frame is None or frame.empty:
        return None, None
    try:
        idx = frame.index.get_indexer([pd.Timestamp(when)], method="nearest")[0]
    except Exception:
        return None, None
    idx = max(0, min(int(idx), len(frame) - 1))
    return frame.iloc[idx], frame.index[idx]


def iv_series_at_dte(asset: str, dte: int, delta_key: str = "delta50",
                     days: int = 30) -> tuple[pd.Series | None, bool, str]:
    """Historical IV (vol points) for one tenor of one delta bucket."""
    frame, estimated, src = surface_history(asset, delta_key, days=days)
    if frame is None or frame.empty:
        return None, estimated, src
    cols = sorted(float(c) for c in frame.columns)
    target = float(dte)
    exact = [c for c in cols if abs(c - target) < 0.01]
    if exact:
        s = frame[int(exact[0]) if exact[0] == int(exact[0]) else exact[0]]
    else:
        left = [c for c in cols if c < target]
        right = [c for c in cols if c > target]
        if left and right:
            cl, cr = max(left), min(right)
            w = (target - cl) / (cr - cl)
            s = frame[int(cl)] + (frame[int(cr)] - frame[int(cl)]) * w
        else:
            near = min(cols, key=lambda c: abs(c - target))
            s = frame[int(near)]
    s = pd.to_numeric(s, errors="coerce").dropna() * 100.0
    return (s if len(s) >= 2 else None), estimated, src
