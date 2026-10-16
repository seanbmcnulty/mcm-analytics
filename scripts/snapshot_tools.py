#!/usr/bin/env python3
"""
Snapshot-store maintenance used by the GitHub Actions workflows and by
publish_snapshot_history.bat. Pure pandas, no Deribit calls.

  validate --published DIR --candidate DIR   fail (exit 1) if the candidate
        CSVs would lose or corrupt published history. Run before every push.
  summary  --published DIR --candidate DIR   markdown health report (stdout;
        the workflow appends it to $GITHUB_STEP_SUMMARY).
  stale    --dir DIR | --url-base URL [--max-age-hours 2]
        exit 1 and print a message if any asset's newest row is too old.
  merge    --base DIR --incoming DIR         union incoming rows into base
        (append-only: existing rows win on duplicate timestamps).
  tag-sources --dir DIR                      add/fill the ``source`` column.
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import history  # noqa: E402
from lib.constants import ASSETS  # noqa: E402

IV_MIN, IV_MAX = 0.05, 5.0
CORE_COLS = ("atm_7", "atm_30")       # must be present on every new row
GAP_HOURS = 2.0


def _name(asset: str) -> str:
    return f"{asset}_surface.csv"


def read_csv(path_or_text, is_text: bool = False) -> pd.DataFrame | None:
    try:
        src = io.StringIO(path_or_text) if is_text else path_or_text
        if not is_text and not Path(src).exists():
            return None
        df = pd.read_csv(src, dtype={"timestamp": str, history.SOURCE_COL: str})
    except Exception:
        return None
    if "timestamp" not in df:
        return None
    df["_ts"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601", errors="coerce")
    return df


def _iv_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("timestamp", "_ts", history.SOURCE_COL)]


# --------------------------------------------------------------------------- validate

def validate_asset(old: pd.DataFrame | None, new: pd.DataFrame | None,
                   now: datetime | None = None, max_added: int | None = 1,
                   allow_insert: bool = False) -> list[str]:
    """Return a list of problems (empty = OK to publish)."""
    now = now or datetime.now(timezone.utc)
    if new is None or new.empty:
        return ["candidate file missing or unreadable"] if old is not None else []
    errs: list[str] = []
    if new["_ts"].isna().any():
        errs.append(f"{int(new['_ts'].isna().sum())} unparseable timestamps")
    if not new["_ts"].dropna().is_monotonic_increasing:
        errs.append("timestamps out of order")
    if new["_ts"].duplicated().any():
        errs.append("duplicate timestamps")

    old_cols = [c for c in (old.columns if old is not None else []) if c != "_ts"]
    new_cols = [c for c in new.columns if c != "_ts"]
    added_rows = new
    if old is not None and not old.empty:
        if new_cols != old_cols and new_cols != old_cols + [history.SOURCE_COL]:
            errs.append(f"header changed ({len(old_cols)} -> {len(new_cols)} columns)")
        old_ts, new_ts = set(old["_ts"].dropna()), set(new["_ts"].dropna())
        dropped = sorted(old_ts - new_ts)
        if dropped:
            # Only allowed via thinning: old rows, and their day must still have a row.
            thin_cut = now - timedelta(days=history._THIN_AFTER_DAYS)
            new_days = {t.floor("D") for t in new_ts}
            bad = [t for t in dropped if t >= thin_cut or t.floor("D") not in new_days]
            if bad:
                errs.append(f"{len(bad)} published rows would be lost (e.g. {bad[0].isoformat()})")
        added_rows = new[~new["_ts"].isin(old_ts)]
        if max_added is not None and len(added_rows) > max_added:
            errs.append(f"{len(added_rows)} rows added (max {max_added})")
        if not allow_insert and len(added_rows) and added_rows["_ts"].min() < old["_ts"].max():
            errs.append("new row older than the newest published row")

    if len(added_rows):
        ivs = added_rows[_iv_cols(added_rows)].apply(pd.to_numeric, errors="coerce")
        for c in CORE_COLS:
            if c in ivs and ivs[c].isna().any():
                errs.append(f"new row has blank {c}")
        vals = ivs.stack()
        out = vals[(vals < IV_MIN) | (vals > IV_MAX)]
        if len(out):
            errs.append(f"{len(out)} IVs outside {IV_MIN}..{IV_MAX} (e.g. {out.index[0][1]}={out.iloc[0]})")
    return errs


def cmd_validate(a) -> int:
    ok = True
    for asset in a.assets:
        old = read_csv(Path(a.published) / _name(asset))
        new = read_csv(Path(a.candidate) / _name(asset))
        errs = validate_asset(old, new, max_added=None if a.allow_insert else a.max_added,
                              allow_insert=a.allow_insert)
        n_old = 0 if old is None else len(old)
        n_new = 0 if new is None else len(new)
        if errs:
            ok = False
            for e in errs:
                print(f"::error::{asset}: {e}")
        print(f"[{asset}] published={n_old} candidate={n_new} {'OK' if not errs else 'FAIL'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- summary / stale

def gaps(df: pd.DataFrame, hours: float = 24, now: datetime | None = None,
         gap_h: float = GAP_HOURS) -> list[tuple[pd.Timestamp, float]]:
    now = now or datetime.now(timezone.utc)
    ts = df["_ts"].dropna().sort_values()
    ts = ts[ts >= now - timedelta(hours=hours)]
    pts = list(ts) + [pd.Timestamp(now)]
    out = []
    for a, b in zip(pts, pts[1:]):
        h = (b - a).total_seconds() / 3600
        if h > gap_h:
            out.append((a, h))
    return out


def cmd_summary(a) -> int:
    now = datetime.now(timezone.utc)
    lines = ["## Snapshot health", "",
             "| Asset | Rows | Added | Newest (UTC) | Age | Gaps >2h (24h) | Rows 24h |",
             "|---|---|---|---|---|---|---|"]
    for asset in a.assets:
        old = read_csv(Path(a.published) / _name(asset)) if a.published else None
        new = read_csv(Path(a.candidate) / _name(asset))
        if new is None or new.empty:
            lines.append(f"| {asset} | - | - | missing | - | - | - |")
            continue
        added = len(new) - (len(old) if old is not None else 0)
        newest = new["_ts"].max()
        age_h = (now - newest).total_seconds() / 3600
        g = gaps(new, now=now)
        n24 = int((new["_ts"] >= now - timedelta(hours=24)).sum())
        gtxt = ", ".join(f"{t:%m-%d %H:%M} ({h:.1f}h)" for t, h in g) or "none"
        flag = " :warning:" if age_h > a.max_age_hours else ""
        lines.append(f"| {asset} | {len(new)} | {added:+d} | {newest:%Y-%m-%d %H:%M} | "
                     f"{age_h:.1f}h{flag} | {gtxt} | {n24} |")
    print("\n".join(lines))
    return 0


def newest_rows(a) -> dict[str, pd.Timestamp | None]:
    out = {}
    for asset in a.assets:
        if a.dir:
            df = read_csv(Path(a.dir) / _name(asset))
        else:
            import requests
            df = None
            for _ in range(3):
                try:
                    r = requests.get(f"{a.url_base.rstrip('/')}/{_name(asset)}", timeout=15)
                    if r.status_code == 200:
                        df = read_csv(r.text, is_text=True)
                        break
                except Exception:
                    pass
        out[asset] = None if df is None or df.empty else df["_ts"].max()
    return out


def cmd_stale(a) -> int:
    now = datetime.now(timezone.utc)
    stale = []
    for asset, t in newest_rows(a).items():
        if t is None:
            stale.append(f"{asset}: no data")
        else:
            age = (now - t).total_seconds() / 3600
            print(f"[{asset}] newest {t.isoformat()} age {age:.1f}h")
            if age > a.max_age_hours:
                stale.append(f"{asset}: newest row {age:.1f}h old ({t:%Y-%m-%d %H:%M} UTC)")
    if stale:
        print("STALE: " + "; ".join(stale))
        return 1
    return 0


# --------------------------------------------------------------------------- merge / tag

def merge_frames(base: pd.DataFrame | None, incoming: pd.DataFrame | None) -> pd.DataFrame:
    frames = [f for f in (base, incoming) if f is not None and not f.empty]
    if not frames:
        raise ValueError("nothing to merge")
    cols: list[str] = []
    for f in frames:
        for c in f.columns:
            if c not in cols and c != "_ts":
                cols.append(c)
    if history.SOURCE_COL in cols:  # keep source last
        cols.remove(history.SOURCE_COL)
    cols.append(history.SOURCE_COL)
    df = pd.concat(frames, ignore_index=True)  # base first -> base wins
    df = df.dropna(subset=["_ts"])
    df = df[~df["_ts"].duplicated(keep="first")].sort_values("_ts")
    if history.SOURCE_COL not in df:
        df[history.SOURCE_COL] = None
    df[history.SOURCE_COL] = df[history.SOURCE_COL].where(
        df[history.SOURCE_COL].notna(), df["timestamp"].map(history.infer_source))
    return df[cols + ["_ts"]]


def cmd_merge(a) -> int:
    for asset in a.assets:
        bp, ip = Path(a.base) / _name(asset), Path(a.incoming) / _name(asset)
        base, inc = read_csv(bp), read_csv(ip)
        if base is None and inc is None:
            continue
        out = merge_frames(base, inc)
        if base is not None and len(out) < len(base):
            print(f"::error::{asset}: merge would shrink the file"); return 1
        bp.parent.mkdir(parents=True, exist_ok=True)
        out.drop(columns="_ts").to_csv(bp, index=False)
        print(f"[{asset}] base={0 if base is None else len(base)} "
              f"incoming={0 if inc is None else len(inc)} merged={len(out)} "
              f"{out['_ts'].min():%Y-%m-%d} .. {out['_ts'].max():%Y-%m-%d %H:%M}")
    return 0


def cmd_tag(a) -> int:
    for asset in a.assets:
        p = Path(a.dir) / _name(asset)
        changed = history.ensure_source_column(p)
        df = read_csv(p)
        counts = df[history.SOURCE_COL].value_counts().to_dict() if df is not None else {}
        print(f"[{asset}] {'updated' if changed else 'unchanged'} {counts}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "summary", "stale", "merge", "tag-sources"):
        s = sub.add_parser(name)
        s.add_argument("--assets", nargs="+", default=ASSETS)
        s.add_argument("--max-age-hours", type=float, default=2.0)
        if name in ("validate", "summary"):
            s.add_argument("--published", default=None)
            s.add_argument("--candidate", required=True)
            s.add_argument("--max-added", type=int, default=1)
            s.add_argument("--allow-insert", action="store_true",
                           help="one-off merges: allow any number of rows, inserted anywhere")
        if name == "stale":
            g = s.add_mutually_exclusive_group(required=True)
            g.add_argument("--dir"); g.add_argument("--url-base")
        if name == "merge":
            s.add_argument("--base", required=True); s.add_argument("--incoming", required=True)
        if name == "tag-sources":
            s.add_argument("--dir", required=True)
    a = p.parse_args(argv)
    return {"validate": cmd_validate, "summary": cmd_summary, "stale": cmd_stale,
            "merge": cmd_merge, "tag-sources": cmd_tag}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
