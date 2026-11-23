"""Fear & Greed equity-curve regressions.

The live page crashed with KeyError: "['fwd_1'] not in index" when the
sidebar "Forward-return horizons" multiselect did not include 1. The
equity curve always compounds next-day returns, so fwd_1 must be built
even if 1-day is not a selected stats horizon. These tests use a small
constructed OHLC/F&G frame with known closes — they check the transform,
not a made-up market history.
"""
from __future__ import annotations

import pandas as pd

from lib.fng import (
    DEFAULT_THRESHOLDS,
    add_bucket,
    add_forward_returns,
    compute_equity_curve,
    merge_price_fng,
)


def _sample_merged() -> pd.DataFrame:
    # Four consecutive days: extreme fear, fear, greed, extreme greed.
    # Closes: 100 -> 110 -> 99 -> 108, so fwd_1 is +10%, -10%, +9.090...%
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "close": [100.0, 110.0, 99.0, 108.0],
        "value": [10, 30, 70, 90],
        "classification": ["Extreme Fear", "Fear", "Greed", "Extreme Greed"],
    })


def test_add_forward_returns_always_includes_fwd_1():
    """The page crash: selecting only 3d/7d/14d used to omit fwd_1."""
    out = add_forward_returns(_sample_merged(), horizons=[3, 7, 14])
    assert "fwd_1" in out.columns
    assert "fwd_3" in out.columns
    assert "fwd_7" in out.columns
    assert list(out["fwd_1"].iloc[:3]) == [
        110.0 / 100.0 - 1,
        99.0 / 110.0 - 1,
        108.0 / 99.0 - 1,
    ]
    assert pd.isna(out["fwd_1"].iloc[-1])
    # 3-day return uses the last close; later horizons have no future close.
    assert out["fwd_3"].iloc[0] == 108.0 / 100.0 - 1
    assert out["fwd_7"].isna().all()


def test_equity_curve_when_1d_horizon_not_selected():
    """compute_equity_curve must not KeyError if fwd_1 was never requested."""
    merged = add_bucket(_sample_merged(), DEFAULT_THRESHOLDS)
    merged = add_forward_returns(merged, horizons=[3, 7, 14])
    equity = compute_equity_curve(merged)
    assert not equity.empty
    assert list(equity.columns) == [
        "date", "signal", "fwd_1", "strat_ret", "strategy_equity", "buyhold_equity",
    ]
    # Extreme Fear (+1) on day 1, then Fear (+0.5) on day 2, Greed (-0.5) on day 3.
    assert list(equity["signal"]) == [1.0, 0.5, -0.5]
    fwd = [0.10, 99.0 / 110.0 - 1, 108.0 / 99.0 - 1]
    assert abs(equity["fwd_1"].iloc[0] - fwd[0]) < 1e-12
    assert abs(equity["fwd_1"].iloc[1] - fwd[1]) < 1e-12
    assert abs(equity["fwd_1"].iloc[2] - fwd[2]) < 1e-12
    # Strategy return is signal * next-day close-to-close, compounded — real
    # returns from the input closes, not a synthetic series.
    expected_strat = 1.0
    expected_bh = 1.0
    for sig, r in zip([1.0, 0.5, -0.5], fwd):
        expected_strat *= 1 + sig * r
        expected_bh *= 1 + r
    assert abs(equity["strategy_equity"].iloc[-1] - expected_strat) < 1e-12
    assert abs(equity["buyhold_equity"].iloc[-1] - expected_bh) < 1e-12
    # Buy & hold must end at last/first close.
    assert abs(equity["buyhold_equity"].iloc[-1] - 108.0 / 100.0) < 1e-12


def test_compute_equity_curve_derives_fwd_1_from_close():
    """Even a frame that never went through add_forward_returns should work."""
    merged = add_bucket(_sample_merged(), DEFAULT_THRESHOLDS)
    assert "fwd_1" not in merged.columns
    equity = compute_equity_curve(merged)
    assert not equity.empty
    assert abs(equity["fwd_1"].iloc[0] - 0.10) < 1e-12
    assert abs(equity["buyhold_equity"].iloc[-1] - 108.0 / 100.0) < 1e-12


def test_equity_curve_empty_when_only_one_day():
    """A single overlapping date has no realizable next-day return."""
    one = add_bucket(_sample_merged().iloc[:1], DEFAULT_THRESHOLDS)
    one = add_forward_returns(one, horizons=[7])
    equity = compute_equity_curve(one)
    assert equity.empty


def test_merge_price_fng_joins_on_utc_calendar_date():
    """Deribit daily candles arrive at 08:00 UTC; F&G is midnight-aligned."""
    price = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02 08:00:00", "2024-01-03 08:00:00"]),
        "close": [200.0, 210.0],
    })
    fng = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "value": [20, 40, 60],
        "classification": ["Extreme Fear", "Fear", "Greed"],
    })
    merged = merge_price_fng(price, fng)
    assert len(merged) == 2
    assert list(merged["close"]) == [200.0, 210.0]
    assert list(merged["value"]) == [20, 40]
