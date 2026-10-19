"""Basis run: fractional day-count and history on the same basis as the live value."""
import os
import sys
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import cmd_market, deribit, surface  # noqa: E402


def test_days_to_expiry_is_fractional_to_0800_utc():
    now = datetime(2026, 10, 19, 22, 0, tzinfo=timezone.utc)
    assert abs(surface.days_to_expiry(date(2026, 10, 20), now) - 10 / 24) < 1e-12
    assert abs(surface.frac_dte(1, now) - 10 / 24) < 1e-12
    assert surface.days_to_expiry(date(2026, 10, 19), now) < 0


def test_annualise_matches_formula_and_guards():
    assert abs(cmd_market._annualise(0.01, 90) - ((1.01 ** (365 / 90)) - 1) * 100) < 1e-12
    assert np.isnan(cmd_market._annualise(0.01, 0))
    assert np.isnan(cmd_market._annualise(-1.5, 30))


def test_history_uses_hourly_index_and_fractional_days(monkeypatch):
    hours = pd.date_range("2026-10-01", periods=48, freq="h", tz="UTC")
    # future is flat 1% above a flat index at every hour
    fut = pd.DataFrame({"timestamp": hours.tz_localize(None) - pd.Timedelta(hours=1),
                        "close": 101.0})
    idx = pd.Series(100.0, index=hours)
    monkeypatch.setattr(cmd_market, "_future_instrument", lambda a, d: "BTC-6NOV26")
    monkeypatch.setattr(deribit, "get_tradingview_ohlc", lambda *a, **k: fut)
    out = cmd_market._basis_apr_history("BTC", 18, 30, index_hourly=idx)
    assert out is not None and len(out) == 48
    t = hours[0]
    left = (pd.Timestamp(2026, 11, 6, 8, tz="UTC") - t).total_seconds() / 86400
    assert abs(out.loc[t] - ((1.01 ** (365 / left)) - 1) * 100) < 1e-9
    assert out.iloc[-1] > out.iloc[0]  # same basis over less time -> higher APR
