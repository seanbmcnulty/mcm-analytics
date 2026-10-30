"""Offline tests for lib/venue_vol.py (Deribit vs OKX term-structure comparison maths)."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib import ibit, venue_vol as V  # noqa: E402

NOW = datetime(2026, 10, 30, 0, 0, tzinfo=timezone.utc)


def _chain(atm_by_dte, skew=-0.05, spot=100.0):
    rows = []
    for dte, atm in atm_by_dte.items():
        exp = pd.Timestamp(NOW) + pd.Timedelta(days=dte)
        for k in np.arange(40, 181, 2.0):
            x = np.log(k / spot)
            for cp in "CP":
                rows.append(dict(venue="T", expiry=exp, type=cp, strike=float(k), fwd=spot, spot=spot,
                                 iv=atm + skew * x + 0.6 * x * x, bid=1.0, ask=1.1, volume=0, oi=0, mult=1))
    df = ibit._finish(pd.DataFrame(rows), NOW)
    df["ok"] = df["iv"].between(0.05, 5.0)
    return df[ibit.COLS]


def test_flat_term_structure_forward_equals_spot_vol():
    d = _chain({10: 0.50, 30: 0.50, 90: 0.50})
    r = V.compare(d, d, [10, 30, 90])
    assert np.allclose(r["grid"]["atm_d"], 0.50, atol=2e-3)
    assert np.allclose(r["fwd"]["fwd_d"], 0.50, atol=3e-3)
    assert np.allclose(r["grid"]["atm_diff"].fillna(0), 0, atol=1e-12)


def test_forward_vol_from_total_variance():
    d = _chain({30: 0.40, 90: 0.50})
    r = V.compare(d, d, [30, 90])
    exp = np.sqrt((0.50 ** 2 * 90 - 0.40 ** 2 * 30) / 60)
    assert abs(r["fwd"]["fwd_d"].iloc[0] - exp) < 4e-3


def test_rr_fly_signs_and_venue_diff():
    d, o = _chain({30: 0.50, 90: 0.50}, skew=-0.05), _chain({30: 0.52, 90: 0.52}, skew=-0.10)
    g = V.compare(d, o, [30, 60])["grid"]
    assert (g["rr25_d"] < 0).all() and (g["rr25_diff"] < 0).all()      # OKX more put-skewed
    assert (g["fly10_d"] > 0).all()                                    # smile convexity
    assert (g["atm_diff"] > 0.015).all()
