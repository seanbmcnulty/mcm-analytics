#!/usr/bin/env python3
"""Merge our market.csv rows onto the published one (union by timestamp).

Usage: python scripts/merge_market.py <published_dir> <incoming_dir>
Writes <published_dir>/market.csv. Used by record_snapshots.yml when the
snapshots branch moved under us. Missing files are fine (no-op).
"""
import sys
from pathlib import Path

import pandas as pd


def merge(base_dir: Path, incoming_dir: Path) -> int:
    inc = incoming_dir / "market.csv"
    if not inc.exists():
        return 0
    base = base_dir / "market.csv"
    frames = [pd.read_csv(p) for p in (base, inc) if p.exists()]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("timestamp", keep="last")
    ts = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601", errors="coerce")
    df = df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts").drop(columns="_ts")
    df.to_csv(base, index=False)
    return len(df)


if __name__ == "__main__":
    print(f"market rows: {merge(Path(sys.argv[1]), Path(sys.argv[2]))}")
