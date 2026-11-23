"""Rebuild the Ecoinometrics CSVs that the page falls back to.

Usage:
  python tools/refresh_ecoinometrics_data.py [--datasets all|etf_flows|fred|fed_tone|strategy]

Used by ``.github/workflows/update_ecoinometrics_data.yml``. Exit 0 when at
least one requested dataset succeeds, 1 when every requested dataset fails.
Files are written only for the datasets that succeeded, so a later ``git add``
can commit a partial refresh.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import ecoinometrics as eco
from lib import eco_more
from lib.eco_refresh import (DISPATCH_DATASETS, build_etf_flows, build_fed_tone, build_fred_cache,
                             build_strategy_funding)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=os.environ.get("ECO_DATASETS", "all"),
        choices=DISPATCH_DATASETS,
    )
    parser.add_argument("--fed-from", type=int, default=eco.FED_TONE_START_YEAR)
    args = parser.parse_args(argv)
    selected = ["etf_flows", "fred", "fed_tone", "strategy"] if args.datasets == "all" else [args.datasets]
    failed = 0
    for name in selected:
        if name == "etf_flows":
            result = build_etf_flows(log=print)
            ok = result.ok
        elif name == "fred":
            results = build_fred_cache(log=print)
            ok = any(item.ok for item in results.values())
            if not ok:
                print("FRED: every series failed")
            # Treasury curve fallback for the yield-curve tab (FRED, then home.treasury.gov).
            curve = eco_more.build_yield_curve_csv(log=print)
            ok = ok or curve.ok
        elif name == "strategy":
            result = build_strategy_funding(log=print)
            ok = result.ok
        else:
            result = build_fed_tone(args.fed_from, log=print)
            ok = result.ok
            print(result.message)
        if not ok:
            failed += 1
            print(f"fail {name}")
        else:
            print(f"ok {name}")
    return 1 if failed == len(selected) else 0


if __name__ == "__main__":
    sys.exit(main())
