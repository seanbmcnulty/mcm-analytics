#!/usr/bin/env python3
"""
OPTIONAL: fill hourly gaps in the BTC/ETH snapshot CSVs from Deribit's DVOL
index (public/get_volatility_index_data, resolution 3600 = hourly).

Deribit serves no option-IV history, so a gap row is reconstructed the same
way lib/history.py's tier-2 does: take the last live row before the gap and
scale every IV column by DVOL(t) / DVOL(t_prev). Shape held fixed, level real.
Every such row is tagged source=backfill so it can be told apart from live
recordings. SOL/HYPE have no DVOL and are left alone.

    python tools/backfill_dvol.py --dir data/snapshots            # dry run
    python tools/backfill_dvol.py --dir data/snapshots --write    # apply

Run it on the local merged files BEFORE publish_snapshot_history.bat (the
publish merge keeps any timestamps the branch doesn't already have).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import deribit, history  # noqa: E402

GAP_HOURS = 2.0
MIN_SEP_MIN = 40      # don't synthesize a row this close to a real one
CHUNK_H = 900         # stay under the API's ~1000-candle cap


def dvol_hourly(cur: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    parts = []
    t = start
    while t < end:
        t2 = min(t + pd.Timedelta(hours=CHUNK_H), end)
        df = deribit.get_dvol(cur, "3600", int(t.timestamp() * 1000), int(t2.timestamp() * 1000))
        if df is not None and not df.empty:
            parts.append(df)
        t = t2
        time.sleep(0.2)
    if not parts:
        return pd.Series(dtype=float)
    df = pd.concat(parts)
    s = pd.Series(pd.to_numeric(df["close"], errors="coerce").values,
                  index=pd.to_datetime(df["timestamp"], utc=True).dt.as_unit("us"))
    return s[~s.index.duplicated()].sort_index().dropna()


def backfill_asset(path: Path, cur: str, write: bool) -> int:
    df = pd.read_csv(path, dtype={"timestamp": str, history.SOURCE_COL: str})
    if history.SOURCE_COL not in df:
        df[history.SOURCE_COL] = df["timestamp"].map(history.infer_source)
    df["_ts"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601").dt.as_unit("us")
    ivcols = [c for c in df.columns if c not in ("timestamp", "_ts", history.SOURCE_COL)]
    live = df[df[history.SOURCE_COL] == history.SOURCE_LIVE].sort_values("_ts")
    if len(live) < 2:
        return 0
    dv = dvol_hourly(cur, live["_ts"].min().floor("h"), live["_ts"].max().ceil("h"))
    if dv.empty:
        print(f"[{cur}] no DVOL data"); return 0
    all_ts = df["_ts"].sort_values()
    new_rows = []
    for (_, a), (_, b) in zip(live.iterrows(), live.iloc[1:].iterrows()):
        if (b["_ts"] - a["_ts"]).total_seconds() / 3600 <= GAP_HOURS:
            continue
        base = dv.asof(a["_ts"])
        if not base or pd.isna(base):
            continue
        for h in pd.date_range(a["_ts"].ceil("h"), b["_ts"], freq="h", inclusive="left"):
            near = (all_ts - h).abs().min()
            if near < pd.Timedelta(minutes=MIN_SEP_MIN) or h not in dv.index:
                continue
            ratio = float(dv[h]) / float(base)
            row = {"timestamp": h.isoformat(), history.SOURCE_COL: history.SOURCE_BACKFILL}
            for c in ivcols:
                v = pd.to_numeric(a[c], errors="coerce")
                row[c] = "" if pd.isna(v) else round(float(v) * ratio, 6)
            new_rows.append(row)
    print(f"[{cur}] {len(new_rows)} backfill rows for gaps > {GAP_HOURS}h")
    if write and new_rows:
        out = pd.concat([df.drop(columns="_ts"), pd.DataFrame(new_rows)], ignore_index=True)
        out["_ts"] = pd.to_datetime(out["timestamp"], utc=True, format="ISO8601")
        out = out.sort_values("_ts").drop(columns="_ts")
        out[[c for c in df.columns if c != "_ts"]].to_csv(path, index=False)
    return len(new_rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=str(history.SNAPSHOT_DIR))
    p.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
    p.add_argument("--write", action="store_true")
    a = p.parse_args()
    for cur in a.assets:
        path = Path(a.dir) / f"{cur}_surface.csv"
        if path.exists():
            backfill_asset(path, cur, a.write)
    if not a.write:
        print("Dry run - re-run with --write to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
