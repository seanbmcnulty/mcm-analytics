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

from lib import history, surface
from lib.constants import ASSETS


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

    print("Done." + (" New rows written." if wrote_any else " No new rows."))
    if failed:
        # Exit 2 = partial failure. The workflow still publishes the assets
        # that did record, then fails the run so the Telegram alert fires.
        print(f"FAILED assets after retries: {' '.join(failed)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
