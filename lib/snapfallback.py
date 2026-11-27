"""
Delayed-data fallback for when the live Deribit API is unreachable from the
app's server (e.g. Cloudflare 403 "bot check" on Streamlit Cloud's IPs).

The GitHub Actions recorder ("Record vol snapshots", every 15 min) publishes
real Deribit data to the ``snapshots`` branch:

* ``data/snapshots/{ASSET}_expiry.csv`` — per-listed-expiry ATM / 10d / 25d IV
* ``data/snapshots/market.csv``         — index price, perp mark, funding,
  perp OI and option-book totals per asset (written by
  scripts/record_snapshot.py from this PR on)

When a live call fails, callers here serve the latest recorded row instead and
mark the current thread as "using snapshot data" so the UI (and any chart sent
to Telegram) can say so: ``Delayed — Deribit snapshot as of 20:15 SGT (7 min
old)``. Nothing is ever synthesized: fields the snapshots don't carry stay
blank.
"""
from __future__ import annotations

import io
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

SGT = timezone(timedelta(hours=8))
MARKET_FILE = "market.csv"
_TTL = 300.0
_TIMEOUT = 6.0
# Snapshots older than this aren't used (the recorder itself would have to be
# down for hours); the page stays blank with a note instead.
MAX_AGE_H = 6.0

_tls = threading.local()
_market_cache: tuple[float, pd.DataFrame | None] | None = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Usage tracking (per thread: MCM Bot runs commands on worker threads)
# ---------------------------------------------------------------------------

def begin() -> None:
    """Start tracking snapshot use for the work about to run on this thread."""
    _tls.used = None


def mark(ts) -> None:
    """Record that data as of ``ts`` (UTC timestamp) was served from snapshots."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    prev = getattr(_tls, "used", None)
    _tls.used = ts if prev is None else min(prev, ts)


def used():
    """Oldest snapshot timestamp served on this thread since ``begin()``, or None."""
    return getattr(_tls, "used", None)


def deribit_blocked() -> bool:
    """True when the most recent live Deribit call failed (or the breaker is open)."""
    from lib import deribit
    st = deribit.last_status()
    return st.get("ok") is False or bool(st.get("breaker_open"))


def as_of_text(ts, now: datetime | None = None) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    now = now or datetime.now(timezone.utc)
    age = max(0, int((pd.Timestamp(now) - ts).total_seconds() // 60))
    return f"{ts.tz_convert(SGT).strftime('%H:%M')} SGT ({age} min old)"


def label(ts, now: datetime | None = None) -> str:
    """Short label for chart titles / captions."""
    return f"Delayed — Deribit snapshot as of {as_of_text(ts, now)}"


def banner(ts, reason: str | None = None, now: datetime | None = None) -> str:
    out = label(ts, now)
    if reason:
        out += f"; live Deribit blocked: {reason}"
    return out


def stamp_title(fig, ts) -> None:
    """Prefix a plotly figure's title with the delayed label (in place)."""
    try:
        old = fig.layout.title.text or ""
        tag = f"⚠ {label(ts)}"
        if tag not in old:
            fig.update_layout(title_text=f"{tag}<br>{old}" if old else tag)
    except Exception:  # noqa: BLE001 - never break a chart over its label
        pass


# ---------------------------------------------------------------------------
# Market snapshot (prices / funding / OI / option activity)
# ---------------------------------------------------------------------------

def _remote_base() -> str:
    from lib import history
    return history._REMOTE_SNAPSHOT_BASE


def _fetch_market(get=None) -> pd.DataFrame | None:
    get = get or (lambda url: requests.get(url, timeout=_TIMEOUT))
    try:
        resp = get(f"{_remote_base()}/{MARKET_FILE}")
        if resp.status_code != 200 or not resp.text:
            return None
        df = pd.read_csv(io.StringIO(resp.text))
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601", errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        return df if not df.empty else None
    except Exception:  # noqa: BLE001
        return None


def market_frame(get=None) -> pd.DataFrame | None:
    global _market_cache
    with _lock:
        if _market_cache is not None and time.time() - _market_cache[0] < _TTL:
            return _market_cache[1]
    df = _fetch_market(get)
    with _lock:
        _market_cache = (time.time(), df)
    return df


def latest_market(asset: str, now: datetime | None = None) -> tuple[dict, pd.Timestamp | None]:
    """Latest recorded market fields for ``asset`` (``index``, ``perp_mark``,
    ``funding``, ``perp_oi``, ``opt_oi``, ``opt_volume``, ``opt_count``) and
    their timestamp. Empty dict when there is no recent row."""
    df = market_frame()
    if df is None:
        return {}, None
    now = now or datetime.now(timezone.utc)
    col = f"{asset}_index"
    if col not in df:
        return {}, None
    rows = df[df[col].notna()]
    if rows.empty:
        return {}, None
    row = rows.iloc[-1]
    ts = row["timestamp"]
    if (pd.Timestamp(now) - ts) > pd.Timedelta(hours=MAX_AGE_H):
        return {}, None
    out = {}
    for f in ("index", "perp_mark", "funding", "perp_oi", "opt_oi", "opt_volume", "opt_count"):
        v = row.get(f"{asset}_{f}")
        if v is not None and pd.notna(v):
            out[f] = float(v)
    return out, ts


def market_value(asset: str, field: str) -> float | None:
    """One market field from the latest snapshot, marking the thread as delayed."""
    vals, ts = latest_market(asset)
    v = vals.get(field)
    if v is not None:
        mark(ts)
    return v


# ---------------------------------------------------------------------------
# Vol surface (per listed expiry)
# ---------------------------------------------------------------------------

def latest_expiry_vols(asset: str, now: datetime | None = None):
    """``({field: {dte: iv}}, ts)`` from the latest per-expiry snapshot, with
    the same field names as ``surface.option_vols_by_dte`` (35d buckets are not
    recorded per expiry and stay empty). ``({}, None)`` when unavailable."""
    from lib import history
    df = history.load_expiry_snapshots(asset)
    if df is None or df.empty:
        return {}, None
    now = now or datetime.now(timezone.utc)
    ts = df["timestamp"].max()
    if (pd.Timestamp(now) - pd.Timestamp(ts)) > pd.Timedelta(hours=MAX_AGE_H):
        return {}, None
    sub = df[df["timestamp"] == ts]
    today = now.date()
    out: dict[str, dict[int, float]] = {k: {} for k in
                                        ("atm", "call10", "call25", "call35", "put10", "put25", "put35")}
    for _, r in sub.iterrows():
        d = (r["expiry"] - today).days
        if d < 0:
            continue
        for f in history.EXPIRY_FIELDS:
            v = r.get(f)
            if v is not None and pd.notna(v) and float(v) > 0:
                out[f][int(d)] = float(v)
    if not out["atm"]:
        return {}, None
    return out, pd.Timestamp(ts)


def reset_cache() -> None:
    global _market_cache
    with _lock:
        _market_cache = None
