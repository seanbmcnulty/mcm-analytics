"""365d snapshot tenor: listed interpolation vs modest forward-variance extrap."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lib import history, surface, cmd_vol


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def test_parse_format_tenor_flags():
    assert history.format_tenor_flags({365: "flat_fwd"}) == "365:flat_fwd"
    assert history.format_tenor_flags({180: "flat_vol", 365: "flat_fwd"}) == (
        "180:flat_vol;365:flat_fwd"
    )
    assert history.parse_tenor_flags("365:flat_fwd") == {365: "flat_fwd"}
    assert history.parse_tenor_flags("180:flat_vol;365:flat_fwd")[365] == "flat_fwd"
    assert history.parse_tenor_flags("") == {}
    assert history.parse_tenor_flags(None) == {}
    assert history.parse_tenor_flags(float("nan")) == {}


def test_listed_tenor_is_not_extrapolated():
    # Longest expiry 370d: 365d is interpolated, not extrapolated.
    series = {30: 0.40, 90: 0.38, 180: 0.37, 370: 0.36}
    v, method = history.snapshot_tenor_iv(series, 365)
    want = surface.interp_at_dte(series, 365.0)
    assert method == history.METHOD_LISTED
    assert abs(v - want) < 1e-12


def test_short_curve_stays_blank():
    """SOL/HYPE-like 90d curve must not invent a 180d or 365d pillar."""
    series = {7: 0.55, 30: 0.53, 90: 0.52}
    v, method = history.snapshot_tenor_iv(series, 365)
    assert method is None and _nan(v)
    v, method = history.snapshot_tenor_iv(series, 180)
    assert method is None and _nan(v)
    v, method = history.snapshot_tenor_iv(series, 30)
    assert method == history.METHOD_LISTED and abs(v - 0.53) < 1e-12


def test_flat_forward_variance_365():
    """BTC-like: last two expiries 180d/300d, extend that forward to 365d."""
    series = {30: 0.36, 90: 0.37, 180: 0.38, 300: 0.40}
    v, method = history.snapshot_tenor_iv(series, 365)
    assert method == history.METHOD_FLAT_FWD
    t1, t2, t = 180 / 365.0, 300 / 365.0, 365 / 365.0
    fwd = surface.forward_vol(0.38, t1, 0.40, t2)
    want = math.sqrt((0.40 ** 2 * t2 + fwd ** 2 * (t - t2)) / t)
    assert abs(v - want) < 1e-12
    # Modest: between the last print and a naive flat-vol hold.
    assert 0.40 < v < 0.42


def test_single_expiry_uses_flat_vol_when_coverage_allows():
    series = {300: 0.40}
    v, method = history.snapshot_tenor_iv(series, 365)
    assert method == history.METHOD_FLAT_VOL
    assert abs(v - 0.40) < 1e-12


def test_coverage_threshold_at_boundary():
    # Just under 0.75 * 365 ≈ 273.75 -> blank; just over -> filled.
    under = {7: 0.4, 270: 0.4}
    v, method = history.snapshot_tenor_iv(under, 365)
    assert method is None and _nan(v)
    over = {7: 0.4, 280: 0.4}
    v, method = history.snapshot_tenor_iv(over, 365)
    assert method in (history.METHOD_FLAT_FWD, history.METHOD_FLAT_VOL)
    assert abs(v - 0.40) < 1e-12


def test_record_snapshot_writes_365_and_flags(tmp_path=None):
    import tempfile
    vols = {
        "atm": {7: 0.50, 30: 0.48, 90: 0.45, 180: 0.44, 300: 0.43},
        "call10": {7: 0.51, 30: 0.49, 90: 0.46, 180: 0.45, 300: 0.44},
        "call25": {7: 0.50, 30: 0.48, 90: 0.45, 180: 0.44, 300: 0.43},
        "call35": {7: 0.50, 30: 0.48, 90: 0.45, 180: 0.44, 300: 0.43},
        "put10": {7: 0.55, 30: 0.52, 90: 0.48, 180: 0.46, 300: 0.45},
        "put25": {7: 0.53, 30: 0.50, 90: 0.47, 180: 0.45, 300: 0.44},
        "put35": {7: 0.52, 30: 0.49, 90: 0.46, 180: 0.445, 300: 0.435},
    }

    def run(asset, surface_map, dest):
        with patch.object(history, "SNAPSHOT_DIR", Path(dest)), \
             patch.object(surface, "option_vols_by_dte", lambda a: surface_map):
            return history.record_snapshot_status(asset, min_interval_s=0)

    with tempfile.TemporaryDirectory() as dest:
        dest = Path(dest)
        status = run("BTC", vols, dest)
        assert status == "recorded", status
        path = dest / "BTC_surface.csv"
        assert path.exists()
        df = pd.read_csv(path)
        assert history.TENOR_FLAGS_COL in df.columns
        row = df.iloc[-1]
        assert float(row["atm_365"]) > 0
        flags = history.parse_tenor_flags(row[history.TENOR_FLAGS_COL])
        assert flags.get(365) == history.METHOD_FLAT_FWD
        assert 180 not in flags
        short = {k: {7: 0.5, 30: 0.5, 90: 0.5} for k in vols}
        status = run("SOL", short, dest)
        assert status == "recorded"
        sol = pd.read_csv(dest / "SOL_surface.csv")
        val = sol.iloc[-1]["atm_365"]
        assert pd.isna(val) or str(val).strip() == ""
        assert history.parse_tenor_flags(sol.iloc[-1].get(history.TENOR_FLAGS_COL)) == {}


def test_validate_allows_tenor_flags_column():
    from scripts import snapshot_tools as st
    old = st.read_csv(
        "timestamp,atm_7,atm_30,atm_365,source\n"
        "2026-11-22T01:00:00.1+00:00,0.5,0.5,,live\n",
        is_text=True,
    )
    new = st.read_csv(
        "timestamp,atm_7,atm_30,atm_365,source,tenor_flags\n"
        "2026-11-22T01:00:00.1+00:00,0.5,0.5,,live,\n"
        "2026-11-22T02:00:00.1+00:00,0.5,0.5,0.48,live,365:flat_fwd\n",
        is_text=True,
    )
    assert st.validate_asset(old, new) == []


def test_extrap_mask_marks_only_beyond_last_listed_pillar():
    grid = {1: 0.29, 7: 0.32, 30: 0.37, 90: 0.373, 180: 0.383, 365: 0.40}
    flags = {365: "flat_fwd"}
    dtes = [1, 30, 90, 180, 215, 306, 365]
    mask = cmd_vol._extrap_point_mask(dtes, grid, flags)
    assert mask == [False, False, False, False, True, True, True]


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception:
                fails += 1
                print("FAIL", name)
                traceback.print_exc()
    raise SystemExit(1 if fails else 0)
