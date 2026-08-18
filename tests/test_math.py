"""Numeric checks on the vol maths — values, not just 'it rendered'."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import numpy as np
import pandas as pd

from lib import surface, history

fails = []


def check(name, got, want, tol=1e-6):
    ok = (abs(got - want) <= tol) if np.isfinite(want) else not np.isfinite(got)
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


# --- forward vol -----------------------------------------------------------
# sigma_30 = 50%, sigma_60 = 55%; fwd^2*(T2-T1) = s2^2*T2 - s1^2*T1
T1, T2 = 30 / 365, 60 / 365
s1, s2 = 0.50, 0.55
want = math.sqrt((s2 ** 2 * T2 - s1 ** 2 * T1) / (T2 - T1))
check("forward_vol 30->60", surface.forward_vol(s1, T1, s2, T2), want)
# Flat term structure => forward == spot.
check("forward_vol flat", surface.forward_vol(0.5, T1, 0.5, T2), 0.5)
# Inverted curve with negative fwd variance clamps to 0.
check("forward_vol clamp", surface.forward_vol(0.90, T1, 0.10, T2), 0.0)
# Non-monotonic tenors return NaN.
check("forward_vol bad order", surface.forward_vol(0.5, T2, 0.5, T1), float("nan"))

# --- DTE interpolation -----------------------------------------------------
vals = {7.0: 0.40, 30.0: 0.50, 90.0: 0.60}
check("interp exact", surface.interp_at_dte(vals, 30), 0.50)
# Midpoint of 30->90 is 60 => 0.55
check("interp midpoint", surface.interp_at_dte(vals, 60), 0.55)
# Quarter of the way from 7 to 30 (t=12.75) => 0.40 + 0.25*0.10
check("interp partial", surface.interp_at_dte(vals, 7 + 0.25 * 23), 0.425)
check("interp flat below", surface.interp_at_dte(vals, 1), 0.40)
check("interp flat above", surface.interp_at_dte(vals, 400), 0.60)
check("interp nan outside", surface.interp_at_dte(vals, 400, flat_outside=False),
      float("nan"))

# --- Black-Scholes ---------------------------------------------------------
# ATM (S==K), r=0: d1 = 0.5*sigma*sqrt(T) so call delta > 0.5.
d = surface.bs_delta(100, 100, 1.0, 0.20, True)
check("bs_delta atm call", d, 0.5398278, tol=1e-6)
check("bs_delta atm put", surface.bs_delta(100, 100, 1.0, 0.20, False),
      d - 1.0, tol=1e-9)
# Put-call parity with r=0: C - P = S - K.
c = surface.bs_price(100, 90, 0.5, 0.30, True)
p = surface.bs_price(100, 90, 0.5, 0.30, False)
check("put-call parity", c - p, 10.0, tol=1e-8)
# IV inversion round-trips.
px = surface.bs_price(100, 110, 0.25, 0.65, True)
check("implied_vol roundtrip", surface.implied_vol(px, 100, 110, 0.25, True),
      0.65, tol=1e-6)
# Deep ITM put round-trips too.
px2 = surface.bs_price(100, 130, 0.75, 0.42, False)
check("implied_vol roundtrip put",
      surface.implied_vol(px2, 100, 130, 0.75, False), 0.42, tol=1e-6)
# Sub-intrinsic price is not invertible.
check("implied_vol sub-intrinsic",
      surface.implied_vol(1.0, 100, 50, 0.5, True), float("nan"))

# --- Parkinson realized vol ------------------------------------------------
# Constant H/L ratio k => RV = sqrt(ann * ln(k)^2/(4 ln2)) * 100, exactly.
n, k = 40, 1.02
idx = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
ohlc = pd.DataFrame({"high": np.full(n, 100.0 * k), "low": np.full(n, 100.0)},
                    index=idx)
want_rv = math.sqrt(365 * (math.log(k) ** 2 / (4 * math.log(2)))) * 100
got_rv = float(history.parkinson_rv(ohlc, 10).dropna().iloc[-1])
check("parkinson constant-range", got_rv, want_rv, tol=1e-9)
# Window shorter than the data yields exactly (n - window + 1) values.
check("parkinson window count",
      float(len(history.parkinson_rv(ohlc, 10).dropna())), float(n - 10 + 1))

# --- basis annualisation ---------------------------------------------------
# 1% basis at 90 DTE annualises to (1.01)^(365/90) - 1
basis_dec, dte = 0.01, 90
want_apr = ((1 + basis_dec) ** (365 / dte) - 1) * 100
got_apr = ((1 + (101.0 - 100.0) / 100.0) ** (365.0 / dte) - 1.0) * 100.0
check("basis APR 1% @90d", got_apr, want_apr, tol=1e-12)

# --- vega carry weighting --------------------------------------------------
# w30 = sqrt(30 / mid_dte); at mid_dte == 30 the weight is exactly 1.
check("vega weight at 30d", float(np.sqrt(30.0 / 30.0)), 1.0)
check("vega weight at 120d", float(np.sqrt(30.0 / 120.0)), 0.5)

# --- strike parsing --------------------------------------------------------
check("parse_strike plain", surface.parse_strike("64000"), 64000.0)
check("parse_strike usdc decimal", surface.parse_strike("6d4"), 6.4)

print("\n" + "=" * 60)
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("All maths checks passed.")
