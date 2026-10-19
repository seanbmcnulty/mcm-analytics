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

Snapshot durability: the *local* copy of ``data/snapshots`` only covers what
this process has recorded since it started (Streamlit Community Cloud wipes
it on every redeploy). A GitHub Actions cron job records the same snapshots
independently and commits them to a separate ``snapshots`` branch — deliberately
not the branch Streamlit Cloud watches, so hourly recording commits don't
trigger app redeploys (see .github/workflows/record_snapshots.yml and
scripts/record_snapshot.py). ``load_snapshots`` merges both sources, so the
app always sees the fuller of the two.
"""

from __future__ import annotations

import io
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from lib import deribit, surface
from lib.constants import ASSET_CONFIG

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"

# Durable snapshot history lives on a branch Streamlit Cloud does NOT watch,
# so the hourly recorder commit doesn't cause app redeploys.
SNAPSHOT_REPO = "seanbmcnulty/mcm-analytics"
SNAPSHOT_BRANCH = "snapshots"
_REMOTE_SNAPSHOT_BASE = (
    f"https://raw.githubusercontent.com/{SNAPSHOT_REPO}/{SNAPSHOT_BRANCH}/data/snapshots"
)
_REMOTE_TIMEOUT = 6.0
# Short waits: this runs inside a Streamlit request, so keep the worst case small.
_REMOTE_BACKOFF_S = (1.0, 2.0, 0.0)  # 3 attempts
_REMOTE_SNAPSHOT_CACHE: dict[str, tuple[float, "pd.DataFrame | None"]] = {}
_REMOTE_SNAPSHOT_TTL = 300.0

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

SOURCE_COL = "source"
SOURCE_LIVE, SOURCE_SEED, SOURCE_BACKFILL = "live", "seed", "backfill"
_SEED_CUTOFF = pd.Timestamp("2026-09-27", tz="UTC")


def infer_source(ts_text: str) -> str:
    """
    Best guess for a legacy row with no ``source``: the seeded rows carry an
    exact-midnight timestamp with no fractional seconds; live rows always have
    microseconds from ``datetime.now().isoformat()``.
    """
    try:
        t = pd.Timestamp(ts_text)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    except Exception:
        return SOURCE_LIVE
    if "." not in str(ts_text) and t == t.floor("D") and t < _SEED_CUTOFF:
        return SOURCE_SEED
    return SOURCE_LIVE


def ensure_source_column(path: Path) -> bool:
    """
    Add a ``source`` column to a legacy snapshot CSV (in place), inferring
    seed vs live per row. Fills blanks in an existing column the same way.
    Returns True if the file was rewritten. Never raises.
    """
    try:
        if not path.exists():
            return False
        df = pd.read_csv(path, dtype={"timestamp": str})
        if df.empty or "timestamp" not in df:
            return False
        if SOURCE_COL in df.columns and df[SOURCE_COL].notna().all():
            return False
        inferred = df["timestamp"].map(infer_source)
        if SOURCE_COL in df.columns:
            df[SOURCE_COL] = df[SOURCE_COL].where(df[SOURCE_COL].notna(), inferred)
        else:
            df[SOURCE_COL] = inferred
        df.to_csv(path, index=False)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Retention + storage format
# ---------------------------------------------------------------------------
# The recorder runs every ~15 min. Retention (approved 2026-10-16):
#   * rows younger than FULL_RES_DAYS are kept at full (15-min) resolution;
#   * older rows are downsampled to one row per OLD_RES bin (UTC hour), keeping
#     the row closest to the top of the hour (earliest in the bin), with its
#     ``source`` tag untouched;
#   * nothing is ever deleted beyond that.
# Set FULL_RES_DAYS = None to keep every row forever.
# scripts/snapshot_tools.py validation applies exactly the same rule: the only
# published rows allowed to disappear are the non-kept rows of >FULL_RES_DAYS bins.
FULL_RES_DAYS: int | None = 90
OLD_RES = "1h"
_BEYOND_EXPIRY_TOL_DAYS = 5.0  # leave a tenor blank if it's this far past the last listed expiry

# Storage format: CSV until a file passes PARQUET_THRESHOLD_BYTES, then it is
# converted (once) to ``{asset}_surface.parquet`` and the CSV is removed.
# Readers prefer parquet when present, so old CSV-only stores keep working.
PARQUET_THRESHOLD_BYTES = 20 * 1024 * 1024


def _parquet_path(asset: str, base: Path | None = None) -> Path:
    return (base or SNAPSHOT_DIR) / f"{asset}_surface.parquet"


def _csv_path(asset: str, base: Path | None = None) -> Path:
    return (base or SNAPSHOT_DIR) / f"{asset}_surface.csv"


def store_path(asset: str, base: Path | None = None) -> Path | None:
    """The file that currently holds ``asset``'s history (parquet wins), or None."""
    for p in (_parquet_path(asset, base), _csv_path(asset, base)):
        if p.exists():
            return p
    return None


def read_store(asset: str, base: Path | None = None) -> pd.DataFrame | None:
    """Raw rows (timestamp kept as ISO text) from parquet or CSV."""
    p = store_path(asset, base)
    if p is None:
        return None
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, dtype={"timestamp": str, SOURCE_COL: str})
    return df


def write_store(asset: str, df: pd.DataFrame, base: Path | None = None) -> Path:
    """
    Write ``df`` (timestamp as ISO text) in the right format: parquet if a
    parquet store already exists or the CSV would pass the threshold.
    """
    base = base or SNAPSHOT_DIR
    base.mkdir(parents=True, exist_ok=True)
    pq, csv = _parquet_path(asset, base), _csv_path(asset, base)
    df = df.copy()
    df["timestamp"] = df["timestamp"].astype(str)
    if SOURCE_COL in df:
        df[SOURCE_COL] = df[SOURCE_COL].astype(object).where(df[SOURCE_COL].notna(), None)
    if not pq.exists():
        text = df.to_csv(index=False)
        if len(text.encode()) < PARQUET_THRESHOLD_BYTES:
            csv.write_text(text)
            return csv
    num = [c for c in df.columns if c not in ("timestamp", SOURCE_COL)]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df.to_parquet(pq, index=False)
    if csv.exists():
        csv.unlink()
    return pq


def downsample_keep_mask(ts: pd.Series, now: datetime | None = None,
                         full_res_days: int | None = None,
                         old_res: str | None = None) -> pd.Series:
    """
    Boolean mask (aligned to ``ts``) of rows the retention rule keeps.
    Rows in a bin wholly older than the cutoff keep only the earliest row of
    the bin (closest to the top of the hour). Deterministic + idempotent.
    """
    full_res_days = FULL_RES_DAYS if full_res_days is None else full_res_days
    old_res = old_res or OLD_RES
    t = pd.to_datetime(ts, utc=True, format="ISO8601", errors="coerce")
    keep = pd.Series(True, index=ts.index)
    if full_res_days is None:
        return keep
    now = now or datetime.now(timezone.utc)
    cutoff = pd.Timestamp(now - timedelta(days=full_res_days)).floor(old_res)
    old = t.notna() & (t < cutoff)
    if not old.any():
        return keep
    bins = t[old].dt.floor(old_res)
    first = t[old].groupby(bins).transform("min")
    # numpy assignment: a boolean-Series setitem upcasts to object on pandas 2.1
    vals = keep.to_numpy(dtype=bool).copy()
    vals[old.to_numpy()] = ((t[old] == first) & ~t[old].duplicated()).to_numpy(dtype=bool)
    return pd.Series(vals, index=ts.index)


def apply_retention(asset: str, base: Path | None = None,
                    now: datetime | None = None) -> int:
    """Downsample >FULL_RES_DAYS rows in place; return rows removed. Never raises."""
    try:
        df = read_store(asset, base)
        if df is None or df.empty or "timestamp" not in df:
            return 0
        keep = downsample_keep_mask(df["timestamp"], now=now)
        removed = int((~keep).sum())
        if removed:
            write_store(asset, df[keep.values], base)
        return removed
    except Exception:
        return 0


def maybe_convert_to_parquet(asset: str, base: Path | None = None) -> bool:
    """Convert a CSV store to parquet once it passes the size threshold."""
    csv = _csv_path(asset, base)
    if _parquet_path(asset, base).exists() or not csv.exists():
        return False
    if csv.stat().st_size < PARQUET_THRESHOLD_BYTES:
        return False
    write_store(asset, pd.read_csv(csv, dtype={"timestamp": str, SOURCE_COL: str}), base)
    return True


def record_snapshot_status(asset: str, min_interval_s: float = 1800.0,
                           slot_minutes: int | None = None) -> str:
    """
    Append the current surface to the snapshot store and say what happened:

      "recorded"  a row was written
      "skipped"   the last row is younger than ``min_interval_s``, or (when
                  ``slot_minutes`` is set) already falls in the current UTC
                  slot (e.g. :00/:15/:30/:45 for 15)
      "no_data"   Deribit returned no usable surface (worth retrying)
      "error"     anything else went wrong (e.g. read-only filesystem)

    Tenors beyond the longest listed expiry are written blank rather than
    held flat (SOL/HYPE list nothing past ~3 months, so a flat-held 180d/365d
    value is just a copy of the last expiry and carries no information).
    Upgrades a legacy file to include the ``source`` column before appending,
    then applies retention and converts to parquet past the size threshold.
    """
    try:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = store_path(asset) or _csv_path(asset)
        now = datetime.now(timezone.utc)
        if path.exists():
            try:
                if path.suffix == ".parquet":
                    last_txt = pd.read_parquet(path, columns=["timestamp"])["timestamp"].iloc[-1]
                else:
                    last_txt = pd.read_csv(path, usecols=["timestamp"])["timestamp"].iloc[-1]
                last = pd.to_datetime(last_txt, utc=True, format="ISO8601")
                if (now - last).total_seconds() < min_interval_s:
                    return "skipped"
                if slot_minutes and in_same_slot(last, now, slot_minutes):
                    return "skipped"
            except Exception:
                pass

        vols = surface.option_vols_by_dte(asset)
        if not vols.get("atm"):
            return "no_data"
        listed = [float(k) for k, v in vols.get("atm", {}).items() if surface.finite(v)]
        max_listed = max(listed) if listed else 0.0
        row: dict[str, object] = {"timestamp": now.isoformat()}
        for field in _FIELDS:
            series = {float(k): float(v) for k, v in vols.get(field, {}).items()
                      if surface.finite(v)}
            for dte in SNAPSHOT_DTES:
                if dte > max_listed + _BEYOND_EXPIRY_TOL_DAYS:
                    v = np.nan
                else:
                    v = surface.interp_at_dte(series, float(dte)) if series else np.nan
                row[f"{field}_{dte}"] = round(float(v), 6) if surface.finite(v) else ""
        row[SOURCE_COL] = SOURCE_LIVE
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            new = pd.DataFrame([row]).reindex(columns=list(df.columns))
            write_store(asset, pd.concat([df, new], ignore_index=True))
        else:
            ensure_source_column(path)
            if path.exists():
                # Match the existing header exactly so an append can never shift columns.
                cols = list(pd.read_csv(path, nrows=0).columns)
                frame = pd.DataFrame([row]).reindex(columns=cols)
                frame.to_csv(path, mode="a", header=False, index=False)
            else:
                pd.DataFrame([row]).to_csv(path, index=False)
        apply_retention(asset)
        maybe_convert_to_parquet(asset)
        return "recorded"
    except Exception:
        return "error"


def in_same_slot(a, b, slot_minutes: int) -> bool:
    """True when UTC timestamps ``a`` and ``b`` fall in the same ``slot_minutes`` bucket."""
    fa = pd.Timestamp(a).tz_convert("UTC").floor(f"{int(slot_minutes)}min")
    fb = pd.Timestamp(b).tz_convert("UTC").floor(f"{int(slot_minutes)}min")
    return fa == fb


def record_snapshot(asset: str, min_interval_s: float = 1800.0) -> bool:
    """Back-compat wrapper: True when a row was written."""
    return record_snapshot_status(asset, min_interval_s) == "recorded"


def _parse_snapshot_csv(text) -> pd.DataFrame | None:
    """Parse CSV text, or an already-loaded raw frame (parquet), into the indexed form."""
    try:
        df = text.copy() if isinstance(text, pd.DataFrame) else pd.read_csv(io.StringIO(text))
        if df.empty or "timestamp" not in df:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True,
                                          format="ISO8601", errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        if SOURCE_COL not in df.columns:
            df[SOURCE_COL] = SOURCE_LIVE
        df[SOURCE_COL] = df[SOURCE_COL].fillna(SOURCE_LIVE).astype(str)
        return df if not df.empty else None
    except Exception:
        return None


# Parsed local store, keyed by (path, mtime, size): surface_history() runs
# ~80 times per /vol_run (11 expiries x 7 delta buckets), and re-reading and
# re-parsing the CSV on every call made a cold /vol_run ~3x slower locally.
_LOCAL_SNAPSHOT_CACHE: dict[str, tuple[tuple, "pd.DataFrame | None"]] = {}
# Last merged local+remote frame per asset (reused while neither input changes).
_MERGED_SNAPSHOT_CACHE: dict[str, tuple[object, object, "pd.DataFrame"]] = {}


def _load_local_snapshots_raw(asset: str) -> pd.DataFrame | None:
    try:
        p = store_path(asset)
        if p is None:
            return None
        st = p.stat()
        sig = (str(p), st.st_mtime_ns, st.st_size)
        hit = _LOCAL_SNAPSHOT_CACHE.get(asset)
        if hit is not None and hit[0] == sig:
            return hit[1]
        raw = read_store(asset)
        df = None if raw is None else _parse_snapshot_csv(raw)
        _LOCAL_SNAPSHOT_CACHE[asset] = (sig, df)
        return df
    except Exception:
        return None


def _fetch_remote_snapshots(asset: str) -> pd.DataFrame | None:
    """
    Durable snapshot history from the GitHub Actions recorder (``snapshots``
    branch). Cached in-process for a few minutes; best-effort — any network
    hiccup, missing branch, or malformed response just yields None so this
    never breaks the app, it only means less history than we'd like.
    """
    hit = _REMOTE_SNAPSHOT_CACHE.get(asset)
    if hit is not None and (time.time() - hit[0]) < _REMOTE_SNAPSHOT_TTL:
        return hit[1]
    frame = None
    # Prefer parquet (written once a store passes the size threshold); a 404
    # there just means the store is still CSV.
    for ext in ("parquet", "csv"):
        url = f"{_REMOTE_SNAPSHOT_BASE}/{asset}_surface.{ext}"
        missing = False
        for attempt, wait in enumerate(_REMOTE_BACKOFF_S):
            try:
                resp = requests.get(url, timeout=_REMOTE_TIMEOUT)
                if resp.status_code == 200 and resp.content:
                    if ext == "parquet":
                        frame = _parse_snapshot_csv(pd.read_parquet(io.BytesIO(resp.content)))
                    else:
                        frame = _parse_snapshot_csv(resp.text)
                    break
                if resp.status_code < 500 and resp.status_code != 429:
                    missing = True
                    break  # 404 etc.: retrying won't help
            except Exception:
                frame = None
            if attempt < len(_REMOTE_BACKOFF_S) - 1:
                time.sleep(wait)
        if frame is not None or not missing:
            break
    _REMOTE_SNAPSHOT_CACHE[asset] = (time.time(), frame)
    return frame


def load_snapshots(asset: str, days: int) -> pd.DataFrame | None:
    """
    Recorded snapshots for the last ``days``, indexed by UTC timestamp.

    Merges the local file (whatever this process has recorded itself since
    it started — ephemeral on Streamlit Community Cloud) with the durable
    history the GitHub Actions recorder has committed to the ``snapshots``
    branch. Rows present in both win from the local copy, since it's always
    at least as fresh as the last hourly commit.
    """
    local = _load_local_snapshots_raw(asset)
    remote = _fetch_remote_snapshots(asset)
    if local is None and remote is None:
        return None
    if local is None:
        df = remote
    elif remote is None:
        df = local
    else:
        hit = _MERGED_SNAPSHOT_CACHE.get(asset)
        if hit is not None and hit[0] is local and hit[1] is remote:
            df = hit[2]
        else:
            combined = pd.concat([remote, local])
            df = combined[~combined.index.duplicated(keep="last")].sort_index()
            _MERGED_SNAPSHOT_CACHE[asset] = (local, remote, df)
    # Boolean indexing below always returns a new frame, so callers never
    # share (or mutate) the cached ones.
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    df = df[df.index >= cutoff]
    return df if len(df) >= 2 else None


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

def _dvol_resolution_for_days(days: float) -> str:
    """
    Coarsest Deribit ``get_volatility_index_data`` resolution that still
    covers ``days`` of history without hitting the API's ~1000-candle cap,
    preferring the finest resolution that fits.
    """
    total_seconds = max(float(days), 0.01) * 86400.0
    for res in (3600, 43200, 86400):
        if total_seconds / res <= 1000:
            return str(res)
    return "86400"


def dvol_history(asset: str, days: int = 90,
                 resolution: str | None = None) -> pd.Series | None:
    """
    DVOL close history in vol points, indexed by UTC timestamp.

    ``resolution`` is Deribit's ``get_volatility_index_data`` bar width in
    *seconds* (60, 3600, 43200, 86400) -- NOT minutes, despite the name
    matching ``perp_ohlc``'s minutes-based convention. Passing "60" here
    used to be read by Deribit as 60 *seconds* (1-minute bars), and the API
    caps a single request at ~1000 candles, so any ``days`` window wider
    than about 16.7 hours silently came back covering only its most recent
    sliver instead of the full requested span -- the level driver behind
    every DVOL-scaled reconstruction (intraday and multi-day alike) was
    quietly missing most of its history. Leave ``resolution`` unset to
    auto-pick the coarsest bar width that still covers the full window.
    """
    cfg = ASSET_CONFIG.get(asset, ASSET_CONFIG["BTC"])
    if not cfg.get("has_dvol"):
        return None
    if resolution is None:
        resolution = _dvol_resolution_for_days(days)
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


# ---------------------------------------------------------------------------
# Intraday densification: 15-minute path anchored to recorded snapshots
# ---------------------------------------------------------------------------

INTRADAY_STEP = "15min"
_DVOL_FINE_CACHE: dict[tuple[str, int], tuple[float, pd.Series | None]] = {}
_DVOL_FINE_TTL = 120.0


def dvol_fine(asset: str, hours: int = 24) -> pd.Series | None:
    """
    DVOL on a regular ``INTRADAY_STEP`` grid (UTC) covering the last ``hours``.

    Built from Deribit's 1-minute bars (resolution "60" = 60 *seconds*),
    fetched in 12h chunks so each request stays under the ~1000-candle cap,
    then averaged into 15-minute buckets. Request boundaries are floored to
    the minute so the HTTP cache can actually hit.
    """
    cfg = ASSET_CONFIG.get(asset, ASSET_CONFIG["BTC"])
    if not cfg.get("has_dvol"):
        return None
    key = (asset, int(hours))
    hit = _DVOL_FINE_CACHE.get(key)
    if hit is not None and (time.time() - hit[0]) < _DVOL_FINE_TTL:
        return hit[1]

    end_ms = (int(time.time() * 1000) // 60000) * 60000
    start_ms = end_ms - int(hours) * 3600 * 1000
    chunk = 12 * 3600 * 1000
    parts = []
    t = start_ms
    while t < end_ms:
        e = min(t + chunk, end_ms)
        df = deribit.get_dvol(cfg["deribit_ccy"], resolution="60",
                              start_ms=t, end_ms=e)
        if df is not None and not df.empty:
            parts.append(df)
        t = e
    out = None
    if parts:
        df = pd.concat(parts)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        s = pd.to_numeric(df.set_index("timestamp")["close"],
                          errors="coerce").dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if not s.empty:
            if s.median() < 5:
                s = s * 100.0
            s = s.resample(INTRADAY_STEP).mean().dropna()
            out = s if len(s) >= 4 else None
    _DVOL_FINE_CACHE[key] = (time.time(), out)
    return out


def _secs(idx: pd.DatetimeIndex) -> np.ndarray:
    """Seconds since epoch as floats -- immune to ns/us index-unit differences."""
    idx = pd.DatetimeIndex(idx)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return np.asarray((idx - pd.Timestamp("1970-01-01", tz="UTC"))
                      / pd.Timedelta(seconds=1), dtype=float)


def densify_intraday(asset: str, recorded: pd.Series,
                     hours: int = 24) -> pd.Series | None:
    """
    Dense 15-minute version of a sparse recorded series (vol points).

    The recorded snapshots are ground truth but arrive every few hours, so
    plotting them alone is jumpy. DVOL is a real 1-minute series, so between
    snapshots the path follows DVOL's shape:

        dense(t) = DVOL(t) * r(t)

    where ``r`` is the recorded/DVOL ratio at each snapshot, linearly
    interpolated in time (held flat outside the first/last snapshot). The
    result passes exactly through every recorded point. Returns None when
    DVOL is unavailable (SOL/HYPE) so callers keep the raw series.
    """
    if recorded is None:
        return None
    rec = recorded.dropna()
    rec = rec[~rec.index.duplicated(keep="last")].sort_index()
    if len(rec) < 2:
        return None
    dv = dvol_fine(asset, hours=hours)
    if dv is None:
        return None

    t_dv, t_rec = _secs(dv.index), _secs(rec.index)
    dv_at = np.interp(t_rec, t_dv, dv.values)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_pts = rec.values / dv_at
    ok = np.isfinite(r_pts) & (dv_at > 0)
    if ok.sum() < 2:
        return None
    r_grid = np.interp(t_dv, t_rec[ok], r_pts[ok])
    dense = pd.Series(dv.values * r_grid, index=dv.index)

    # Drop grid points that sit within half a step of a real snapshot so the
    # recorded value isn't shadowed by a near-duplicate model value.
    half = pd.Timedelta(INTRADAY_STEP) / 2
    gaps = np.abs(t_dv[:, None] - t_rec[None, :]).min(axis=1)
    dense = dense[gaps > half.total_seconds()]
    return pd.concat([dense, rec]).sort_index()


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
    dv = dvol_history(asset, days=days)
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
                    days: int = 90,
                    recorded_only: bool = False) -> tuple[pd.DataFrame | None, bool, str]:
    """
    Historical IV surface: index = UTC timestamp, columns = DTE, values decimal IV.

    Returns ``(frame, estimated, source_label)``.  ``estimated`` is True when
    the frame was reconstructed by re-levelling rather than recorded.
    """
    recorded = _surface_from_snapshots(asset, delta_key, days)
    if recorded is not None and len(recorded) >= 8:
        return recorded, False, "recorded snapshots"
    if recorded_only:
        # Intraday callers densify the (possibly sparse) recorded rows
        # themselves, so don't splice an hourly reconstruction onto them.
        if recorded is not None and len(recorded) >= 2:
            return recorded, False, "recorded snapshots"
        return None, True, "none"

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
        # Splice: recorded rows are ground truth wherever we have them, so
        # the reconstruction should only ever extend the line *before* the
        # earliest recorded snapshot or *after* the latest one -- never
        # compete with a real row from roughly the same moment.
        #
        # The old check dropped reconstructed rows only on an *exact*
        # timestamp match (`frame.index.isin(recorded.index)`), which almost
        # never fired: reconstructed timestamps land on DVOL's on-the-hour
        # bars while recorded snapshots land at whatever irregular second
        # the recorder happened to run. So both a reconstructed point and a
        # real recorded point -- each a plausible but different IV reading
        # for nearly the same time -- ended up plotted side by side. That
        # interleaving is what produced the zig-zagging/spiking intraday
        # charts (worsened by the spline line shape overshooting between
        # them). Restricting the reconstruction to outside recorded's
        # covered span removes the overlap entirely.
        lo, hi = recorded.index.min(), recorded.index.max()
        frame = frame[(frame.index < lo) | (frame.index > hi)]
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
                     days: int = 30,
                     recorded_only: bool = False) -> tuple[pd.Series | None, bool, str]:
    """Historical IV (vol points) for one tenor of one delta bucket."""
    frame, estimated, src = surface_history(asset, delta_key, days=days,
                                            recorded_only=recorded_only)
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


def clear_cache() -> None:
    """Drop the in-memory level-driver and remote-snapshot caches."""
    _DRIVER_CACHE.clear()
    _DVOL_FINE_CACHE.clear()
    _REMOTE_SNAPSHOT_CACHE.clear()
    _LOCAL_SNAPSHOT_CACHE.clear()
    _MERGED_SNAPSHOT_CACHE.clear()
