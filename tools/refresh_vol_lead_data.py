"""Rebuild the VIX-vs-DVOL page's committed CSVs (data/vl_*.csv).

Usage: python tools/refresh_vol_lead_data.py [--datasets all|dvol,cboe,yahoo,fred]
Used by .github/workflows/update_vol_lead_data.yml. A failed source keeps its
old CSV and makes the exit code 1 only when every source failed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import vol_lead_data as vld  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all")
    a = ap.parse_args(argv)
    keys = list(vld.DATASETS) if a.datasets == "all" else [k.strip() for k in a.datasets.split(",")]
    ok = 0
    for k in keys:
        res = vld.refresh(k)
        print(res.message)
        ok += int(res.ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
