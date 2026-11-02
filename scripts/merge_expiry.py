#!/usr/bin/env python3
"""Merge incoming *_expiry.csv rows onto the published copies (union, dedupe on timestamp+expiry)."""
import sys
from pathlib import Path
import pandas as pd

base, incoming = Path(sys.argv[1]), Path(sys.argv[2])
for inc in incoming.glob("*_expiry.csv"):
    dst = base / inc.name
    frames = [pd.read_csv(f, dtype=str) for f in (dst, inc) if f.exists()]
    df = pd.concat(frames).drop_duplicates(["timestamp", "expiry"], keep="last")
    df = df.sort_values(["timestamp", "expiry"])
    df.to_csv(dst, index=False)
    print(f"merged {inc.name}: {len(df)} rows")
