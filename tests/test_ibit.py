"""Offline tests for lib/ibit.py (Cboe IBIT chain parser + analytics)."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib import ibit as L  # noqa: E402

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


def _payload(spot=50.0, iv0=0.40):
    opts = []
    for ymd, dte_sig in (("261120", 1.0), ("270115", 1.2)):
        for k in range(30, 71, 2):
            for cp in "CP":
                skew = 0.002 * (50 - k) if cp == "P" else 0.0005 * (k - 50)
                iv = max(iv0 + skew, 0.1)
                opts.append(dict(option=f"IBIT{ymd}{cp}{k * 1000:08d}", bid=1.0, ask=1.2, iv=iv,
                                 open_interest=100.0 + k, volume=10.0 * (k % 5)))
    opts.append(dict(option="IBIT261120C00020000", bid=0, ask=0, iv=0.0, open_interest=5, volume=0))
    opts.append(dict(option="garbage", bid=1, ask=1, iv=.3, open_interest=1, volume=1))
    return {"timestamp": "2026-10-21 08:00:00",
            "data": {"current_price": spot, "iv30": 40.1, "prev_day_close": 49.0,
                     "price_change_percent": 1.0, "volume": 1, "options": opts}}


def test_parse_occ_and_meta():
    df, meta = L.parse_cboe(_payload(), NOW)
    assert meta["spot"] == 50.0 and meta["n_expiries"] == 2
    assert set(df["type"]) == {"C", "P"} and df["strike"].between(20, 70).all()
    assert (df["mult"] == 100).all()
    assert abs(df["oi_usd"].iloc[0] - df["oi"].iloc[0] * 100 * 50) < 1e-6
    # expiry is 16:00 New York (EST in Nov -> 21:00 UTC)
    assert df["expiry"].min().hour == 21
    # zero-IV / no-quote rows are not usable for surfaces
    assert not df[df["strike"] == 20]["ok"].any()


def test_greeks_and_metrics():
    df, _ = L.parse_cboe(_payload(), NOW)
    c = df[(df.type == "C") & (df.strike == 50) & df.ok].iloc[0]
    assert 0.45 < c["delta"] < 0.6 and c["gamma"] > 0
    m = L.expiry_metrics(df)
    assert len(m) == 2
    # built-in put skew -> negative risk reversal, ATM ~ 40%
    assert (m["rr25"] < 0).all() and abs(m["atm"].iloc[0] - 0.40) < 0.02
    t = L.at_tenors(m, ["atm"], [60])
    assert np.isfinite(t["atm"].iloc[0])
    assert np.isnan(L.at_tenors(m, ["atm"], [5000])["atm"].iloc[0])  # no extrapolation


def test_positioning_and_density():
    df, _ = L.parse_cboe(_payload(), NOW)
    mp = L.max_pain(df)
    assert len(mp) == 2 and mp["max_pain"].between(30, 70).all()
    g = L.gex_by_strike(df)
    assert g["gex"].abs().sum() > 0
    ob = L.oi_by_tenor(df)
    assert abs((ob.call_pct + ob.put_pct).sum() - 100) < 1e-6
    om = L.oi_by_moneyness(df)
    assert abs((om.C_pct + om.P_pct).sum() - 100) < 1e-6
    g1 = df[df.expiry == df.expiry.min()]
    dens = L.implied_density(g1)
    assert dens is not None
    x, f, _ = dens
    assert abs(np.trapezoid(f, x) - 1) < 1e-6 if hasattr(np, "trapezoid") else True
    tp = L.tail_probs(x, f, 0.10)
    assert 0 < tp["down"] < 1 and 0 < tp["up"] < 1


def test_surface_grid_masks_outside_range():
    df, _ = L.parse_cboe(_payload(), NOW)
    kg = np.linspace(-0.9, 0.9, 19)
    z = L.surface_grid(df, kg)
    assert z.shape[0] == 2 and z.isna().to_numpy().any() and z.notna().to_numpy().any()
