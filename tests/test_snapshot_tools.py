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


def _with_thinning(days=60):
    import contextlib

    @contextlib.contextmanager
    def cm():
        prev = history.THIN_AFTER_DAYS
        history.THIN_AFTER_DAYS = days
        try:
            yield
        finally:
            history.THIN_AFTER_DAYS = prev
    return cm()


def test_thinning_disabled_by_default_and_no_row_may_be_lost():
    assert history.THIN_AFTER_DAYS is None
    now = datetime(2026, 12, 31, tzinfo=timezone.utc)
    old = _df(_csv(["2026-09-01T01:00:00.1+00:00,0.5,0.5,0.5,live",
                    "2026-09-01T02:00:00.1+00:00,0.5,0.5,0.5,live"]))
    new = _df(_csv(["2026-09-01T01:00:00.1+00:00,0.5,0.5,0.5,live"]))
    assert any("lost" in e for e in st.validate_asset(old, new, now=now))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "X_surface.csv"
        rows = [f"2026-08-{1 + i // 24:02d}T{i % 24:02d}:00:00.1+00:00,0.5,0.5,0.5,live"
                for i in range(24 * 30)]
        p.write_text(_csv(rows))
        history._thin_snapshot_file(p)
        assert len(pd.read_csv(p)) == len(rows)


def test_validate_allows_thinning_of_old_rows():
    with _with_thinning():
        _validate_allows_thinning_of_old_rows()


def _validate_allows_thinning_of_old_rows():
    now = datetime(2026, 12, 31, tzinfo=timezone.utc)
    old = _df(_csv(["2026-09-01T01:00:00.1+00:00,0.5,0.5,0.5,live",
                    "2026-09-01T02:00:00.1+00:00,0.5,0.5,0.5,live"]))
    new = _df(_csv(["2026-09-01T01:00:00.1+00:00,0.5,0.5,0.5,live"]))
    assert st.validate_asset(old, new, now=now) == []


def test_validate_tiered_thinning():
    with _with_thinning():
        _validate_tiered_thinning()


def _validate_tiered_thinning():
    now = datetime(2026, 12, 31, 12, tzinfo=timezone.utc)
    # 20 days old: one row per hour must survive, extra rows in the hour may go
    old = _df(_csv(["2026-12-11T01:00:00.1+00:00,0.5,0.5,0.5,live",
                    "2026-12-11T01:15:00.1+00:00,0.5,0.5,0.5,live"]))
    ok = _df(_csv(["2026-12-11T01:00:00.1+00:00,0.5,0.5,0.5,live"]))
    assert st.validate_asset(old, ok, now=now) == []
    # but dropping a whole hour in that tier is a loss
    two = _df(_csv(["2026-12-11T01:00:00.1+00:00,0.5,0.5,0.5,live",
                    "2026-12-11T02:00:00.1+00:00,0.5,0.5,0.5,live"]))
    assert any("lost" in e for e in st.validate_asset(two, ok, now=now))
    # inside the 7-day full-resolution window nothing may be dropped
    recent = _df(_csv(["2026-12-29T01:00:00.1+00:00,0.5,0.5,0.5,live",
                       "2026-12-29T01:15:00.1+00:00,0.5,0.5,0.5,live"]))
    rec_ok = _df(_csv(["2026-12-29T01:00:00.1+00:00,0.5,0.5,0.5,live"]))
    assert any("lost" in e for e in st.validate_asset(recent, rec_ok, now=now))


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
