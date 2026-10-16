"""Offline checks for the snapshot validation/merge tooling and source tagging."""
import io
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from lib import history
from scripts import snapshot_tools as st

HDR = "timestamp,atm_7,atm_30,atm_365"


def _df(text):
    return st.read_csv(text, is_text=True)


def _csv(rows, source=True):
    head = HDR + (",source" if source else "")
    return "\n".join([head] + rows) + "\n"


def test_infer_source():
    assert history.infer_source("2026-09-01T00:00:00+00:00") == "seed"
    assert history.infer_source("2026-09-01T00:00:00.123+00:00") == "live"
    assert history.infer_source("2026-10-10T00:00:00+00:00") == "live"   # after seed era
    assert history.infer_source("2026-09-01T05:00:00+00:00") == "live"


def test_validate_accepts_one_new_row_and_source_upgrade():
    old = _df(_csv(["2026-10-15T01:00:00.1+00:00,0.5,0.5,0.5"], source=False))
    new = _df(_csv(["2026-10-15T01:00:00.1+00:00,0.5,0.5,0.5,live",
                    "2026-10-15T02:00:00.1+00:00,0.5,0.5,,live"]))
    assert st.validate_asset(old, new) == []


def test_validate_rejects_lost_rows_bad_iv_and_blank_core():
    old = _df(_csv(["2026-10-15T01:00:00.1+00:00,0.5,0.5,0.5,live",
                    "2026-10-15T02:00:00.1+00:00,0.5,0.5,0.5,live"]))
    shrunk = _df(_csv(["2026-10-15T02:00:00.1+00:00,0.5,0.5,0.5,live"]))
    assert any("lost" in e for e in st.validate_asset(old, shrunk))
    bad = _df(_csv(["2026-10-15T01:00:00.1+00:00,0.5,0.5,0.5,live",
                    "2026-10-15T02:00:00.1+00:00,0.5,0.5,0.5,live",
                    "2026-10-15T03:00:00.1+00:00,9.0,,0.5,live"]))
    errs = st.validate_asset(old, bad)
    assert any("outside" in e for e in errs) and any("blank atm_30" in e for e in errs)


def _rows(start, n, step_min=15, val="0.5"):
    t0 = pd.Timestamp(start)
    return [f"{(t0 + pd.Timedelta(minutes=step_min * i)).isoformat()},{val},{val},{val},live"
            for i in range(n)]


NOW = datetime(2027, 3, 1, 12, tzinfo=timezone.utc)   # 2026-11-01 is ~120d old here


def test_retention_defaults():
    assert history.FULL_RES_DAYS == 90 and history.OLD_RES == "1h"


def test_downsample_keeps_first_row_per_old_hour_and_all_recent_rows():
    old = _rows("2026-11-01T00:07:00.1+00:00", 8)            # 2 old hours x 4 rows
    recent = _rows("2027-02-28T00:07:00.1+00:00", 8)          # inside 90d
    df = _df(_csv(old + recent))
    keep = history.downsample_keep_mask(df["timestamp"], now=NOW)
    kept = df[keep.values]["timestamp"].tolist()
    assert kept[:2] == ["2026-11-01T00:07:00.100000+00:00", "2026-11-01T01:07:00.100000+00:00"]
    assert len(kept) == 2 + 8
    # idempotent
    again = history.downsample_keep_mask(df[keep.values]["timestamp"], now=NOW)
    assert again.all()


def test_apply_retention_preserves_source_and_validation_accepts_it():
    with tempfile.TemporaryDirectory() as d:
        rows = _rows("2026-11-01T00:00:00.1+00:00", 8) + _rows("2027-02-28T00:00:00.1+00:00", 4)
        rows[0] = rows[0].replace(",live", ",seed")
        p = Path(d) / "X_surface.csv"
        p.write_text(_csv(rows))
        before = st.read_asset(d, "X")
        removed = history.apply_retention("X", Path(d), now=NOW)
        after = st.read_asset(d, "X")
        assert removed == 6 and len(after) == 6
        assert after["source"].iloc[0] == "seed"
        assert st.validate_asset(before, after, now=NOW, max_added=None) == []


def test_validation_rejects_any_other_loss():
    old_rows = _rows("2026-11-01T00:00:00.1+00:00", 4) + _rows("2027-02-28T00:00:00.1+00:00", 4)
    old = _df(_csv(old_rows))
    # dropping the KEPT (first) row of an old hour is a loss
    wrong = _df(_csv(old_rows[1:2] + old_rows[4:]))
    assert any("lost" in e for e in st.validate_asset(old, wrong, now=NOW, max_added=None))
    # dropping a recent (<90d) row is a loss
    recent_gap = _df(_csv(old_rows[:1] + old_rows[4:7]))
    assert any("lost" in e for e in st.validate_asset(old, recent_gap, now=NOW, max_added=None))
    # the exact downsample is fine
    ok = _df(_csv(old_rows[:1] + old_rows[4:]))
    assert st.validate_asset(old, ok, now=NOW, max_added=None) == []


def test_parquet_switch_is_dormant_below_threshold_and_round_trips():
    with tempfile.TemporaryDirectory() as d:
        df = _df(_csv(_rows("2027-02-28T00:00:00.1+00:00", 10))).drop(columns="_ts")
        p = history.write_store("X", df, Path(d))
        assert p.suffix == ".csv" and not history._parquet_path("X", Path(d)).exists()
        prev = history.PARQUET_THRESHOLD_BYTES
        history.PARQUET_THRESHOLD_BYTES = 10
        try:
            assert history.maybe_convert_to_parquet("X", Path(d))
        finally:
            history.PARQUET_THRESHOLD_BYTES = prev
        assert history.store_path("X", Path(d)).suffix == ".parquet"
        assert not (Path(d) / "X_surface.csv").exists()
        back = st.read_asset(d, "X")
        assert back["timestamp"].tolist() == df["timestamp"].tolist()
        assert back["source"].tolist() == ["live"] * 10
        parsed = history._parse_snapshot_csv(history.read_store("X", Path(d)))
        assert len(parsed) == 10 and float(parsed["atm_7"].iloc[0]) == 0.5
        # once parquet, further writes stay parquet even when small
        assert history.write_store("X", df, Path(d)).suffix == ".parquet"


def test_interp_frame_matches_scalar():
    import numpy as np
    from lib import surface
    rng = np.random.default_rng(0)
    f = pd.DataFrame(rng.uniform(.2, .9, (300, 8)), columns=[1, 7, 14, 30, 60, 90, 180, 365])
    f[f > 0.75] = np.nan
    for t in (0.5, 7, 7.005, 45, 365, 500):
        for fo in (True, False):
            vec = surface.interp_frame_at_dte(f, t, fo)
            ref = [surface.interp_at_dte({float(c): float(r[c]) for c in r.index
                                          if surface.finite(r[c])}, t, fo) for _, r in f.iterrows()]
            assert np.allclose(vec, ref, equal_nan=True)


def test_merge_is_append_only_and_tags():
    base = _df(_csv(["2026-09-01T00:00:00+00:00,0.4,0.4,0.4,seed",
                     "2026-10-15T02:00:00.1+00:00,0.5,0.5,0.5,live"]))
    inc = _df(_csv(["2026-08-30T00:00:00+00:00,0.3,0.3,0.3",
                    "2026-10-15T02:00:00.1+00:00,0.9,0.9,0.9",
                    "2026-10-15T03:00:00.1+00:00,0.6,0.6,0.6"], source=False))
    out = st.merge_frames(base, inc)
    assert len(out) == 4 and list(out.columns)[-2:] == ["source", "_ts"]
    assert float(out.loc[out["timestamp"].str.startswith("2026-10-15T02"), "atm_7"].iloc[0]) == 0.5
    assert out["source"].tolist() == ["seed", "seed", "live", "live"]
    assert st.validate_asset(base, out, max_added=None, allow_insert=True) == []


def test_ensure_source_column_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "X_surface.csv"
        p.write_text(_csv(["2026-09-01T00:00:00+00:00,0.4,0.4,0.4",
                           "2026-10-15T02:00:00.1+00:00,0.5,0.5,0.5"], source=False))
        assert history.ensure_source_column(p)
        df = pd.read_csv(p)
        assert df["source"].tolist() == ["seed", "live"]
        assert not history.ensure_source_column(p)
        parsed = history._parse_snapshot_csv(p.read_text())
        assert "source" in parsed.columns and len(parsed) == 2


def test_gaps():
    now = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
    df = _df(_csv(["2026-10-15T01:00:00.1+00:00,0.5,0.5,0.5,live",
                   "2026-10-15T02:00:00.1+00:00,0.5,0.5,0.5,live",
                   "2026-10-15T08:00:00.1+00:00,0.5,0.5,0.5,live"]))
    g = st.gaps(df, now=now)
    assert [round(h) for _, h in g] == [6, 4]
