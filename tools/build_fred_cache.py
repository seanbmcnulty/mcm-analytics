"""Cache FRED series used by pages/13 into data/fred_<ID>.csv.

Streamlit Cloud often times out reaching fred.stlouisfed.org.
``lib.ecoinometrics.fred_series`` falls back to these files. The same function
runs from the Ecoinometrics page's Refresh FRED button and from
``.github/workflows/update_ecoinometrics_data.yml``.

Usage: python tools/build_fred_cache.py
Exit 0 when at least one series was fetched, 1 when every series failed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eco_refresh import build_fred_cache


def main(argv=None) -> int:
    results = build_fred_cache(log=print)
    bad = sum(not result.ok for result in results.values())
    return 1 if results and bad == len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
