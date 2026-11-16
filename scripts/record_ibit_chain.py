#!/usr/bin/env python3
"""
Record one post-close IBIT option-chain snapshot.

The Cboe delayed chain is a snapshot, not a tape.  This script stores that
snapshot once per US session so the IBIT page can diff open interest.  It
writes only ``data/snapshots/IBIT_chain.parquet`` (real rows, never a
backfill) and is meant to run from ``.github/workflows/record_ibit_chain.yml``,
which publishes to the existing ``snapshots`` branch without touching the
vol-surface files.

Exit 0: recorded, already stored, weekend/holiday, or before the 16:45 ET gate.
Exit 2: the chain was stale, empty, or the fetch failed.  The file is left
as it was.

    python scripts/record_ibit_chain.py
    python scripts/record_ibit_chain.py validate --published DIR --candidate DIR
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import ibit  # noqa: E402

# A session that did not need a write, or got one.  Anything else is a failed
# close and the workflow must not publish.
_GRACEFUL = {"recorded", "skipped", "closed", "too_early"}


def _fetch():
    now = datetime.now(timezone.utc)
    payload = ibit.fetch_raw("IBIT")
    return ibit.parse_cboe(payload, now)


def cmd_record() -> int:
    now = datetime.now(timezone.utc)
    status = ibit.record_chain_snapshot(now, _fetch)
    print(f"[IBIT] {status}")
    decision, session = ibit.recording_decision(now)
    if decision == "ok" and session is not None and not ibit.holiday_list_covers(session):
        print(f"WARNING: NYSE holiday list covers {min(ibit.HOLIDAY_YEARS)}-"
              f"{max(ibit.HOLIDAY_YEARS)} only; {session.isoformat()} was treated as a trading day.",
              file=sys.stderr)
    if status == "closed":
        print("US cash session is closed (weekend or NYSE holiday); nothing stored.")
    elif status == "too_early":
        print("Before 16:45 America/New_York; the delayed feed cannot see the close yet.")
    elif status == "skipped":
        print("This session is already stored.")
    elif status == "stale":
        print("Cboe chain is not this session; refusing to label it as today's close.",
              file=sys.stderr)
    if status not in _GRACEFUL:
        return 2
    return 0


def cmd_validate(published: str, candidate: str, max_new_sessions: int) -> int:
    try:
        old = ibit.read_chain_store(published)
    except Exception as exc:  # unreadable published history must not be replaced
        print(f"::error::published IBIT chain unreadable: {exc}")
        return 1
    try:
        new = ibit.read_chain_store(candidate)
    except Exception as exc:
        print(f"::error::candidate IBIT chain unreadable: {exc}")
        return 1
    errs = ibit.validate_chain_history(old, new, max_new_sessions=max_new_sessions)
    n_old = 0 if old is None else len(old)
    n_new = 0 if new is None else len(new)
    if errs:
        for e in errs:
            print(f"::error::IBIT chain: {e}")
        print(f"[IBIT] published={n_old} candidate={n_new} FAIL")
        return 1
    print(f"[IBIT] published={n_old} candidate={n_new} OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("record", help="append today's close if the US session has ended")
    val = sub.add_parser("validate", help="fail if the candidate would drop or rewrite history")
    val.add_argument("--published", required=True)
    val.add_argument("--candidate", required=True)
    val.add_argument("--max-new-sessions", type=int, default=1)
    args = parser.parse_args(argv)
    if args.cmd == "validate":
        return cmd_validate(args.published, args.candidate, args.max_new_sessions)
    return cmd_record()


if __name__ == "__main__":
    raise SystemExit(main())
