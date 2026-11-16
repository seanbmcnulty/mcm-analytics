"""Lookback lines on the vol term structure span every listed expiry.

Per-expiry history only has contracts that were actually listed at the
lookback. The chart must still plot those tenors (marked estimated) instead
of dropping them, while a recorded expiry keeps its own IV.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

from lib import cmd_vol, history, surface


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def test_recorded_expiry_is_not_replaced_by_the_grid():
    """An expiry that was listed keeps its print even when the grid disagrees."""
    exact = history.ExactMap({5: 0.80, 12: 0.32, 40: 0.36})
    grid = {0.0: 0.50, 7.0: 0.40, 23.0: 0.40}
    dtes = [1, 4, 5, 12, 19, 40]
    ys, flags = cmd_vol._merge_lookback_iv(dtes, exact, grid)
    assert abs(ys[2] - 0.80) < 1e-12 and flags[2] is False
    assert abs(ys[3] - 0.32) < 1e-12 and flags[3] is False
    assert abs(ys[5] - 0.36) < 1e-12 and flags[5] is False


def test_interior_gap_interpolates_recorded_neighbours_not_the_grid():
    """A missing daily between two recorded expiries must not use the coarse grid.

    The 1/7-day pillars pull a single missing daily (19-Nov on the 24h line)
    several vol points away from the prints beside it. Interpolate those
    prints instead.
    """
    exact = history.ExactMap({3: 0.52, 5: 0.54, 12: 0.32, 40: 0.36})
    grid = {0.0: 0.10, 7.0: 0.10}  # would be wildly wrong if used inside
    ys, flags = cmd_vol._merge_lookback_iv([3, 4, 5, 19, 40], exact, grid)
    assert flags[1] is True and abs(ys[1] - 0.53) < 1e-12  # midway 3 -> 5
    # dte 19 sits 7/28 of the way from 12 to 40.
    assert flags[3] is True and abs(ys[3] - (0.32 + 0.25 * 0.04)) < 1e-12
    assert flags[0] is False and flags[2] is False and flags[4] is False


def test_short_end_uses_shifted_fixed_tenor_grid():
    """Tenors shorter than anything recorded fall back to the fixed-tenor grid."""
    now = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
    # Snapshot a week earlier. Columns are DTE *then*; _row_to_map re-keys them
    # onto today's DTE for the same expiry date (1 -> -6 dropped, 7 -> 0, ...).
    row = pd.Series(
        {1: 0.90, 7: 0.50, 14: 0.40, 30: 0.40},
        name=pd.Timestamp(now - timedelta(days=7)),
    )
    grid = cmd_vol._row_to_map(row, now)
    assert 0.0 in grid and abs(grid[0.0] - 0.50) < 1e-12
    assert all(k >= 0 for k in grid)
    exact = history.ExactMap({5: 0.80})  # nothing shorter than 5 DTE was listed
    ys, flags = cmd_vol._merge_lookback_iv([1, 4, 5], exact, grid)
    want = 0.50 + (1.0 / 7.0) * (0.40 - 0.50)
    assert flags[0] is True and abs(ys[0] - want) < 1e-12
    assert flags[1] is True  # still short of the first recorded expiry
    assert abs(ys[1] - (0.50 + (4.0 / 7.0) * (0.40 - 0.50))) < 1e-12
    assert flags[2] is False and abs(ys[2] - 0.80) < 1e-12


def test_no_grid_leaves_the_short_end_blank():
    """No synthetic backfill when the fixed-tenor history isn't there either."""
    exact = history.ExactMap({5: 0.80, 12: 0.32})
    ys, flags = cmd_vol._merge_lookback_iv([1, 5, 8, 12], exact, None)
    assert _nan(ys[0]) and flags[0] is False
    assert abs(ys[1] - 0.80) < 1e-12 and flags[1] is False
    # 8 DTE is between two recorded expiries, so it is still estimated.
    assert flags[2] is True and abs(ys[2] - (0.80 + (3.0 / 7.0) * (0.32 - 0.80))) < 1e-12
    assert flags[3] is False


def test_full_grid_when_nothing_was_recorded():
    grid = {1.0: 0.25, 7.0: 0.25, 30.0: 0.25, 90.0: 0.25}
    ys, flags = cmd_vol._merge_lookback_iv([1, 5, 30, 90], None, grid)
    assert all(abs(y - 0.25) < 1e-12 for y in ys)
    assert all(flags)


def _finite_ys(trace):
    out = []
    for y in list(trace.y):
        if y is None:
            out.append(None)
            continue
        y = float(y)
        out.append(None if math.isnan(y) else y)
    return out


def _symbols(trace):
    sym = trace.marker.symbol
    if sym is None:
        return []
    if isinstance(sym, str):
        return [sym]
    return list(sym)


def _plotly_keeps_trace_data() -> bool:
    """False under the no-op plotly stub used when the real package is absent."""
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=["a"], y=[2.0], name="n"))
        data = list(getattr(fig, "data", []) or [])
        return bool(data) and list(data[0].y) == [2.0]
    except Exception:
        return False


_NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
_DTES = [1, 2, 3, 4, 5, 12, 19, 40]


def _history_frame():
    idx = pd.date_range(_NOW - timedelta(days=34), _NOW, freq="12h", tz="UTC")
    # Flat 30% grid. After the DTE shift every lookback interpolates to 0.30,
    # so a test can tell a grid fill from a recorded print.
    return pd.DataFrame({d: 0.30 for d in (1, 7, 14, 30, 60, 90, 180, 365)}, index=idx)


def _exact_for(when: datetime):
    """Per-expiry ATM. 24h misses one interior daily; 1w misses the front and one interior."""
    age = (_NOW - when).total_seconds() / 3600.0
    if abs(age - 24) < 5:
        # Missing 4 DTE — sits between 3 and 5.
        return {"atm": history.ExactMap({1: 0.50, 2: 0.51, 3: 0.52, 5: 0.54,
                                          12: 0.45, 19: 0.44, 40: 0.42})}
    if abs(age - 24 * 7) < 8:
        # Nothing inside 5 DTE was listed. 19 DTE sits between 12 and 40.
        return {"atm": history.ExactMap({5: 0.80, 12: 0.32, 40: 0.36})}
    return None  # 1 month: per-expiry history does not reach


def _render(asset="BTC"):
    live = {float(d): 0.40 + d / 1000.0 for d in _DTES}
    frame = _history_frame()

    def surface_history(asset_name, delta_key="delta50", days=35, **kwargs):
        return frame, False, "recorded snapshots"

    def expiry_snapshot_near(asset_name, when, tol_hours, now=None):
        return _exact_for(when)

    with patch.object(cmd_vol, "_now", lambda: _NOW), \
         patch.object(surface, "listed_expiries", lambda a: list(_DTES)), \
         patch.object(surface, "iv_by_dte_for_delta", lambda a, k: dict(live)), \
         patch.object(surface, "option_vols_by_dte", lambda a: {
             "atm": dict(live), "call10": dict(live), "put10": dict(live),
             "call25": dict(live), "put25": dict(live),
             "call35": dict(live), "put35": dict(live),
         }), \
         patch.object(history, "surface_history", surface_history), \
         patch.object(history, "expiry_snapshot_near", expiry_snapshot_near):
        return cmd_vol.cmd_vol_term_structure(asset)


def test_lookback_lines_span_every_listed_expiry():
    if not _plotly_keeps_trace_data():
        print("skip test_lookback_lines_span_every_listed_expiry (plotly stub)")
        return
    for asset in ("BTC", "ETH"):
        fig, _df, text = _render(asset)
        assert fig is not None and text is None, text
        by_name = {tr.name: tr for tr in fig.data}
        assert set(by_name) >= {"Current", "24h Ago", "1 Week Ago", "1 Month Ago (est.)"}
        # The month line is the smoothed grid, not a relabelled partial.
        assert "1 Month Ago" not in by_name
        x = list(by_name["Current"].x)
        assert len(x) == len(_DTES)
        for name in ("Current", "24h Ago", "1 Week Ago", "1 Month Ago (est.)"):
            ys = _finite_ys(by_name[name])
            assert len(ys) == len(x)
            assert all(y is not None for y in ys), (asset, name, ys)
            assert list(by_name[name].x) == x

        week = _finite_ys(by_name["1 Week Ago"])
        # Recorded prints are kept (0.80 / 0.32 / 0.36 -> vol points).
        assert abs(week[4] - 80.0) < 1e-6  # 5 DTE
        assert abs(week[5] - 32.0) < 1e-6  # 12 DTE
        assert abs(week[7] - 36.0) < 1e-6  # 40 DTE
        # 19 DTE is interior: between 12 (32) and 40 (36), not the 30% grid.
        assert abs(week[6] - 33.0) < 1e-6
        # Front tenors were not listed: fixed-tenor grid, which is a flat 30.
        for i in range(4):
            assert abs(week[i] - 30.0) < 1e-6, week

        day = _finite_ys(by_name["24h Ago"])
        assert abs(day[2] - 52.0) < 1e-6 and abs(day[4] - 54.0) < 1e-6
        assert abs(day[3] - 53.0) < 1e-6  # missing 4 DTE, midway, not the 30% grid

        month = _finite_ys(by_name["1 Month Ago (est.)"])
        assert all(abs(y - 30.0) < 1e-6 for y in month)

        # Hollow marker only on the estimated points of a mixed line.
        week_sym = _symbols(by_name["1 Week Ago"])
        assert week_sym[0] == "square-open" and week_sym[4] == "square"
        assert week_sym[6] == "square-open" and week_sym[7] == "square"
        day_sym = _symbols(by_name["24h Ago"])
        assert day_sym[3] == "diamond-open"
        assert all(s == "diamond" for i, s in enumerate(day_sym) if i != 3)
        assert _symbols(by_name["1 Month Ago (est.)"]) == ["triangle-up"]

        week_cd = list(by_name["1 Week Ago"].customdata)
        assert "(est.)" in str(week_cd[0]) and "(est.)" not in str(week_cd[4])
        day_cd = list(by_name["24h Ago"].customdata)
        assert "(est.)" in str(day_cd[3]) and "(est.)" not in str(day_cd[0])

        # finalize() inserts <br> (and may break on the hyphen in "16-Nov").
        title = re.sub(r"<br\s*/?>", " ", fig.layout.title.text)
        title = re.sub(r"<[^>]+>", "", title)
        title = re.sub(r"\s+", " ", title)
        assert "1 Month Ago" in title
        assert "24h Ago (19-Nov)" in title
        assert "1 Week Ago (16-Nov, 17-Nov, 18-Nov, 19-Nov, 04-Dec)" in title
        assert "Open markers are (est.)" in title


def test_full_grid_fallback_keeps_honest_elapsed_label():
    """A month lookback whose history only reaches ~10 days is not labelled 1 Month."""
    if not _plotly_keeps_trace_data():
        print("skip test_full_grid_fallback_keeps_honest_elapsed_label (plotly stub)")
        return
    idx = pd.date_range(_NOW - timedelta(days=10), _NOW, freq="12h", tz="UTC")
    frame = pd.DataFrame({d: 0.22 for d in (1, 7, 14, 30, 60)}, index=idx)

    def surface_history(*_a, **_k):
        return frame, False, "recorded snapshots"

    live = {float(d): 0.40 for d in _DTES}
    with patch.object(cmd_vol, "_now", lambda: _NOW), \
         patch.object(surface, "listed_expiries", lambda a: list(_DTES)), \
         patch.object(surface, "iv_by_dte_for_delta", lambda a, k: dict(live)), \
         patch.object(history, "surface_history", surface_history), \
         patch.object(history, "expiry_snapshot_near", lambda *a, **k: None):
        fig, _df, text = cmd_vol.cmd_vol_term_structure("SOL")
    assert fig is not None and text is None
    names = [tr.name for tr in fig.data]
    assert "1 Month Ago (est.)" not in names
    assert "1 Month Ago" not in names
    est = [n for n in names if n.endswith("(est.)")]
    assert est, names
    # ~10 days, not a month. _fmt_ago rounds hours/24.
    assert est[-1].startswith("10d Ago") or est[-1].startswith("11d Ago"), est
    ys = _finite_ys([tr for tr in fig.data if tr.name == est[-1]][0])
    assert len(ys) == len(_DTES) and all(y is not None for y in ys)


def test_skew_and_butterfly_already_span_every_expiry():
    """Skew and butterfly interpolate the fixed-tenor grid, so they already cover
    every listed expiry. Locked here so a later ExactMap port doesn't reopen the gap.
    """
    if not _plotly_keeps_trace_data():
        print("skip test_skew_and_butterfly_already_span_every_expiry (plotly stub)")
        return
    frame = _history_frame()
    live = {float(d): 0.40 + d / 500.0 for d in _DTES}

    def surface_history(*_a, **_k):
        return frame, False, "recorded snapshots"

    vols = {k: dict(live) for k in
            ("atm", "call10", "put10", "call25", "put25", "call35", "put35")}
    # Puts a touch richer so skew/butterfly aren't identically zero.
    vols["put25"] = {d: v + 0.02 for d, v in live.items()}
    vols["put10"] = {d: v + 0.04 for d, v in live.items()}
    with patch.object(cmd_vol, "_now", lambda: _NOW), \
         patch.object(surface, "listed_expiries", lambda a: list(_DTES)), \
         patch.object(surface, "option_vols_by_dte", lambda a: vols), \
         patch.object(history, "surface_history", surface_history):
        skew, _df, stext = cmd_vol.cmd_skew_term_structure("BTC")
        fly, _df2, ftext = cmd_vol.cmd_butterfly_term_structure("ETH")
    assert skew is not None and stext is None
    assert fly is not None and ftext is None
    for fig, kind in ((skew, "skew"), (fly, "butterfly")):
        assert len(fig.data) >= 4, kind  # current + 3 lookbacks
        xs = list(fig.data[0].x)
        assert len(xs) == len(_DTES)
        for tr in fig.data:
            ys = _finite_ys(tr)
            assert list(tr.x) == xs, (kind, tr.name)
            assert all(y is not None for y in ys), (kind, tr.name, ys)


if __name__ == "__main__":
    import traceback
    _fails = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print("ok  ", _name)
            except Exception:
                _fails += 1
                print("FAIL", _name)
                traceback.print_exc()
    raise SystemExit(1 if _fails else 0)
