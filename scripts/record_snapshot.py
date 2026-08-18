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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import history
from lib.constants import ASSETS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", default=ASSETS,
                        help=f"Assets to snapshot (default: {ASSETS})")
    parser.add_argument("--min-interval", type=float, default=1800.0,
                        help="Skip an asset if its last row is younger than "
                             "this many seconds (default: 1800 = 30 min)")
    args = parser.parse_args()

    wrote_any = False
    for asset in args.assets:
        try:
            wrote = history.record_snapshot(asset, min_interval_s=args.min_interval)
        except Exception as exc:
            print(f"[{asset}] ERROR: {exc}", file=sys.stderr)
            continue
        status = "recorded" if wrote else "skipped (too recent, or no live data)"
        print(f"[{asset}] {status}")
        wrote_any = wrote_any or wrote

    # Exit 0 either way — "nothing new to record" is not a failure, it just
    # means the workflow's commit step will find no changes to push.
    print("Done." + (" New rows written." if wrote_any else " No new rows."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
