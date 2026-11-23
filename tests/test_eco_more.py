"""Five charts added 2026-11-23 (lib/eco_more.py, lib/eco_more_charts.py). No network."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from lib import eco_more as em
from lib import eco_more_charts as ec

FRED_TEXT = (
    "observation_date,DGS3MO,DGS6MO,DGS1,DGS2,DGS3,DGS5,DGS7,DGS10,DGS20,DGS30\n"
    "2025-11-18,3.9,3.8,3.7,3.6,3.6,3.7,3.9,4.1,4.7,4.7\n"
    "2025-11-19,3.95,3.80,3.68,3.58,3.59,3.71,3.90,4.13,4.71,4.75\n"
    "2026-11-18,4.34,4.43,4.62,4.80,4.84,4.89,4.96,5.07,5.44,5.41\n"
    "2026-11-19,4.34,4.42,4.62,4.76,4.81,4.86,4.94,5.04,5.43,5.40\n"
    "2026-11-20,4.30,,,,,,,,,\n"
)
TREASURY_TEXT = (
    'Date,"1 Mo","1.5 Month","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"\n'
    "11/20/2026,4.19,4.26,4.28,4.32,4.32,4.38,4.51,4.63,4.65,4.71,4.80,4.92,5.34,5.31\n"
    "11/19/2026,4.20,4.27,4.29,4.34,4.35,4.42,4.62,4.76,4.81,4.86,4.94,5.04,5.43,5.40\n"
)


class _Resp:
    def __init__(self, text="", content=b"", status=200):
        self.text, self.content, self.status_code = text, content, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_fred_multi_and_curve_compare_uses_full_curves_only():
    df = em.parse_fred_multi(FRED_TEXT)
    assert list(df.columns) == ["date", *em.YIELD_LABELS]
    cc = em.curve_compare(df)
    # 20 Nov 2026 has only 3M, so the latest *full* curve is 19 Nov.
    assert cc.now_date == pd.Timestamp("2026-11-19") and cc.ago_date == pd.Timestamp("2025-11-19")
    t = cc.table.set_index("tenor")
    assert abs(t.loc["2Y", "change"] - 1.18) < 1e-9
    assert abs(cc.avg_change - t["change"].mean()) < 1e-12
    fig = ec.figure_yield_curve(cc, "FRED")
    assert len(fig.data) == 2


def test_curve_compare_refuses_a_gap_instead_of_borrowing_an_old_curve():
    df = em.parse_fred_multi(FRED_TEXT)
    assert em.curve_compare(df[df["date"] >= "2026-01-01"]) is None


def test_treasury_parse_maps_columns():
    df = em.parse_treasury_csv(TREASURY_TEXT)
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2026-11-19", "2026-11-20"]
    assert df.iloc[-1]["3M"] == 4.32 and df.iloc[-1]["30Y"] == 5.31


def test_yield_curve_falls_back_to_treasury_then_fails_cleanly():
    calls = []

    def get(url, **kw):
        calls.append(url)
        if "fred" in url:
            raise TimeoutError("read timed out")
        return _Resp(TREASURY_TEXT)

    r = em.fetch_yield_curve_result(get=get, now="2026-11-23", allow_csv=False)
    assert r.ok and r.source == "live" and "treasury" in r.origin and "FRED unavailable" in r.message
    assert len(calls) == 4      # one FRED try, then three Treasury years

    def dead(url, **kw):
        raise TimeoutError("down")

    r = em.fetch_yield_curve_result(get=dead, now="2026-11-23", allow_csv=False)
    assert not r.ok and "FRED" in r.message and "Treasury" in r.message and r.df.empty


def test_flow_dots_month_total():
    fb = pd.DataFrame({"date": pd.to_datetime(["2026-10-30", "2026-11-02", "2026-11-03", "2026-11-04"]),
                       "flow": [500.0, -100.0, 40.0, -300.0]})
    d, st = em.flow_dots(fb)
    assert list(d["in_month"]) == [False, True, True, True]
    assert st["total"] == -360.0 and st["inflow_days"] == 1 and st["outflow_days"] == 2 and st["sessions"] == 3
    fig = ec.figure_flow_dots(d, st)
    assert len(fig.data) == 2


def _gld_xlsx(rows) -> bytes:
    df = pd.DataFrame(rows, columns=["Date", "Closing Price", "NAV/Share at 10:30am NYT",
                                     "Total Net Asset Value in the Trust"])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        pd.DataFrame({"x": ["disclaimer"]}).to_excel(w, sheet_name="Disclaimer", index=False)
        df.to_excel(w, sheet_name=em.GLD_SHEET, index=False)
    return buf.getvalue()


def test_gld_parse_and_flows_skip_holidays():
    content = _gld_xlsx([
        ["09-Jan-2024", 190.0, 190.0, 190.0 * 100_000_000],
        ["10-Jan-2024", 191.0, 191.0, 191.0 * 100_200_000],
        ["11-Jan-2024", " HOLIDAY", " HOLIDAY", " HOLIDAY"],
        ["12-Jan-2024", 192.0, 192.0, 192.0 * 100_100_000],
    ])
    g = em.parse_gld_archive(content)
    assert list(g["date"].dt.day) == [9, 10, 12]
    assert list(g["shares"]) == [100_000_000, 100_200_000, 100_100_000]
    f = em.gld_flows_usd(g)
    assert list(f["flow_usd"]) == [200_000 * 191.0, -100_000 * 192.0]
    r = em.fetch_gld_result(get=lambda url, **kw: _Resp(content=content), now="2026-11-23")
    assert r.ok and len(r.df) == 3
    bad = em.fetch_gld_result(get=lambda url, **kw: _Resp(status=403), now="2026-11-23")
    assert not bad.ok and "403" in bad.message


def test_cumulative_flows_start_after_approval():
    btc = pd.DataFrame({"date": pd.to_datetime(["2024-01-10", "2024-01-11", "2024-01-12"]),
                        "flow": [999.0, 655.0, 200.0]})          # US$ m
    gold = pd.DataFrame({"date": pd.to_datetime(["2024-01-11", "2024-01-12"]), "flow_usd": [1e9, -5e8]})
    c = em.cumulative_flows(btc, gold)
    b = c[c["series"] == "btc"]["cum_bn"].tolist()
    assert np.allclose(b, [0.655, 0.855])                         # 10 Jan itself is excluded
    assert np.allclose(c[c["series"] == "gold"]["cum_bn"], [1.0, 0.5])
    assert len(ec.figure_btc_gold_flows(c, None).data) == 2
    only_btc = em.cumulative_flows(btc, None)
    assert set(only_btc["series"]) == {"btc"}


def test_ytd_returns_base_is_previous_year_close():
    idx = pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-02", "2026-06-01"])
    s = pd.Series([90.0, 100.0, 110.0, 80.0], index=idx)
    late = pd.Series([50.0, 55.0], index=pd.to_datetime(["2026-01-05", "2026-02-01"]))  # no 2025 close
    r, info = em.ytd_returns({"A": s, "B": late}, 2026)
    assert set(info) == {"A"}
    assert abs(info["A"]["ret"] + 20.0) < 1e-9 and info["A"]["base_date"] == pd.Timestamp("2025-12-31")
    a = r[r["name"] == "A"]
    assert a["ret"].iloc[0] == 0.0 and abs(a["ret"].iloc[1] - 10.0) < 1e-9
    assert len(ec.figure_ytd_returns(r, info, 2026).data) == 1


def test_flow_scenarios_terciles_and_determinism():
    rng = np.random.default_rng(1)
    n = 300
    e = rng.normal(0, 1000, n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = -100 + 0.4 * y[i - 1] + e[i]
    fb = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=n), "flow": y})
    a = em.simulate_flow_scenarios(fb, n_paths=3000)
    b = em.simulate_flow_scenarios(fb, n_paths=3000)
    assert a.finals == b.finals                                   # fixed seed
    assert a.finals["downside"] < a.finals["continuation"] < a.finals["upside"]
    assert a.cutoffs[0] < a.finals["continuation"] < a.cutoffs[1]
    assert len(a.future_dates) == 30 and a.future_dates[0] > fb["date"].iloc[-1]
    for name in em.SCENARIO_NAMES:
        band = a.bands[name]
        assert (band["p5"] <= band["median"]).all() and (band["median"] <= band["p95"]).all()
    # one step ahead, 29 of the 30 actual sessions are still in the window
    first = a.bands["continuation"]["median"].iloc[0]
    kept = fb["flow"].iloc[-29:].sum()
    assert abs(first - kept) < 5 * 1000
    fig = ec.figure_flow_scenarios(a)
    assert len(fig.data) == 1 + 3 * 3


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
