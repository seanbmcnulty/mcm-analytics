#!/usr/bin/env python3
"""
Standalone vol-surface snapshot recorder.

Runs independently of the Streamlit app (no streamlit import, no Streamlit
Cloud runtime needed) so it can be driven by a GitHub Actions cron job and
keep recording even while the app itself is asleep, crashed, or mid-redeploy.

Streamlit Community Cloud's filesystem is ephemeral — anything the running
app writes to data/snapshots/*.csv is lost on the next redeploy. This script
is meant to be run on a schedule from CI, where its writes get committed back
to the repo (see .github/workflows/record_snapshots.yml), so the history
survives redeploys instead of resetting to zero each time.

Usage:
    python scripts/record_snapshot.py
    python scripts/record_snapshot.py --assets BTC ETH
    python scripts/record_snapshot.py --min-interval 0   # force a row even if recent
    python scripts/record_snapshot.py --slot-minutes 15 --min-interval 120   # one row per 15-min UTC slot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lib import deribit, history, surface
from lib.constants import ASSET_CONFIG, ASSETS

MARKET_PATH = history.SNAPSHOT_DIR / "market.csv"
MARKET_RETENTION_DAYS = 30
MARKET_FIELDS = ("index", "perp_mark", "funding", "perp_oi", "opt_oi", "opt_volume", "opt_count", "dvol")


# Waits between attempts for an asset whose surface came back empty or
# errored (Deribit hiccup). 3 attempts total; the last entry is only used if
# --attempts is raised.
RETRY_BACKOFF_S = (2.0, 5.0, 15.0)


def record_with_retry(asset: str, min_interval: float, attempts: int = 3,
                      slot_minutes: int | None = None,
                      sleep=time.sleep) -> str:
    status = "error"
    for attempt in range(attempts):
        try:
            status = history.record_snapshot_status(asset, min_interval_s=min_interval,
                                                    slot_minutes=slot_minutes)
        except Exception as exc:  # defensive: record_snapshot_status shouldn't raise
            print(f"[{asset}] attempt {attempt + 1}: {exc}", file=sys.stderr)
            status = "error"
        if status in ("recorded", "skipped"):
            return status
        if attempt < attempts - 1:
            wait = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]
            print(f"[{asset}] {status} on attempt {attempt + 1}; retrying in {wait:.0f}s")
            surface._VOLS_CACHE.pop(asset, None)  # don't get the cached empty result back
            sleep(wait)
    return status


def market_row(asset: str) -> dict:
    """Real Deribit market fields for one asset (blank where Deribit gave nothing).
    Used by the app's delayed-data fallback (lib/snapfallback.py) when the live
    API is unreachable from Streamlit Cloud."""
    cfg = ASSET_CONFIG[asset]
    out: dict[str, float] = {}
    px = deribit.get_index_price(cfg["index"])
    if px:
        out["index"] = float(px)
    t = deribit.get_ticker(cfg["perp"]) or {}
    for src, dst in (("mark_price", "perp_mark"), ("current_funding", "funding"),
                     ("open_interest", "perp_oi")):
        if t.get(src) is not None:
            out[dst] = float(t[src])
    chain = deribit.get_option_chain(cfg["deribit_ccy"], "option")
    if chain:
        mine = [c for c in chain if str(c.get("instrument_name", "")).startswith(cfg["deribit_prefix"])]
        out["opt_oi"] = float(sum(c.get("open_interest") or 0 for c in mine))
        out["opt_volume"] = float(sum(c.get("volume") or 0 for c in mine))
        out["opt_count"] = float(len(mine))
    try:
        dv = surface.dvol_now(asset)
    except Exception:  # noqa: BLE001
        dv = None
    if dv is not None:
        out["dvol"] = float(dv)
    return out


def record_market(assets, slot_minutes: int | None = 15, path=None, now=None,
                  row_fn=market_row) -> str:
    """Append one row of market fields for all assets to market.csv.
    Returns recorded / skipped / no_data / error. Never raises."""
    path = path or MARKET_PATH
    try:
        from datetime import datetime, timezone
        now = now or datetime.now(timezone.utc)
        old = None
        if path.exists():
            old = pd.read_csv(path)
            if not old.empty and slot_minutes:
                last = pd.to_datetime(old["timestamp"].iloc[-1], utc=True, format="ISO8601")
                if history.in_same_slot(last, now, slot_minutes):
                    return "skipped"
        row: dict[str, object] = {"timestamp": pd.Timestamp(now).isoformat()}
        for a in assets:
            for k, v in row_fn(a).items():
                row[f"{a}_{k}"] = v
        if len(row) == 1:
            return "no_data"
        new = pd.DataFrame([row])
        df = new if old is None or old.empty else pd.concat([old, new], ignore_index=True)
        ts = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601", errors="coerce")
        df = df[ts >= pd.Timestamp(now) - pd.Timedelta(days=MARKET_RETENTION_DAYS)]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return "recorded"
    except Exception as exc:  # noqa: BLE001
        print(f"[market] error: {exc}", file=sys.stderr)
        return "error"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", default=ASSETS,
                        help=f"Assets to snapshot (default: {ASSETS})")
    parser.add_argument("--min-interval", type=float, default=1800.0,
                        help="Skip an asset if its last row is younger than "
                             "this many seconds (default: 1800 = 30 min)")
    parser.add_argument("--slot-minutes", type=int, default=None,
                        help="Slot-based skip: record at most one row per UTC slot of this "
                             "many minutes (15 -> :00/:15/:30/:45). Combined with --min-interval.")
    parser.add_argument("--attempts", type=int, default=3,
                        help="Attempts per asset when Deribit returns nothing (default 3)")
    args = parser.parse_args()

    failed = []
    wrote_any = False
    for asset in args.assets:
        status = record_with_retry(asset, args.min_interval, args.attempts,
                                   slot_minutes=args.slot_minutes)
        print(f"[{asset}] {status}")
        wrote_any = wrote_any or status == "recorded"
        if status not in ("recorded", "skipped"):
            failed.append(asset)

    # Market fields (prices / funding / OI / option activity) for the app's
    # delayed fallback. Best-effort: never fails the run.
    print(f"[market] {record_market(args.assets, slot_minutes=args.slot_minutes)}")

    print("Done." + (" New rows written." if wrote_any else " No new rows."))
    if failed:
        # Exit 2 = partial failure. The workflow still publishes the assets
        # that did record, then fails the run so the Telegram alert fires.
        print(f"FAILED assets after retries: {' '.join(failed)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
