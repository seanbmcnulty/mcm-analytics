"""Nasdaq down-day buckets, drawdown detection, the 30-session flow line and
the Strategy 8-K parser (fixture: the 9 Nov 2026 filing). No network."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from lib import eco_charts as charts
from lib import eco_refresh
from lib import ecoinometrics as eco
from lib import strategy_8k as s8

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mstr_8k_20261109.htm"


def test_fred_series_include_nasdaq100():
    assert "NASDAQ100" in eco.FRED_SERIES and "NASDAQCOM" in eco.FRED_SERIES
    assert "strategy" in eco_refresh.DISPATCH_DATASETS


def test_nasdaq_bucket_edges():
    assert eco.nasdaq_bucket(0.0) is None
    assert eco.nasdaq_bucket(0.5) is None
    assert eco.nasdaq_bucket(-0.1) == "−2% to 0%"
    assert eco.nasdaq_bucket(-2.0) == "−2% to 0%"      # each bucket includes its lower bound
    assert eco.nasdaq_bucket(-2.01) == "−4% to −2%"
    assert eco.nasdaq_bucket(-4.0) == "−4% to −2%"
    assert eco.nasdaq_bucket(-4.01) == "−6% to −4%"
    assert eco.nasdaq_bucket(-6.5) == "−8% to −6%"
    assert eco.nasdaq_bucket(-12.0) == "Below −8%"
    assert eco.nasdaq_bucket(float("nan")) is None


def test_btc_on_nasdaq_down_days_same_date_rule():
    ndx = pd.DataFrame({"date": pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]),
                        "value": [100.0, 97.0, 97.97, 92.0]})
    days = pd.date_range("2024-01-04", "2024-01-10", freq="D")
    close = pd.Series([10, 10, 10, 10, 11, 11, 9.9], index=days, dtype=float)  # 7 Jan=10, 8 Jan=11 -> +10%
    out = eco.btc_on_nasdaq_down_days(ndx, close, "2024-01-01")
    # 8 Jan: NDX -3% (Fri->Mon), BTC 7->8 Jan +10%. 9 Jan is an up day. 10 Jan: NDX ~-6.1%, BTC -10%.
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-08", "2024-01-10"]
    assert out.iloc[0]["bucket"] == "−4% to −2%" and abs(out.iloc[0]["btc_ret"] - 10.0) < 1e-9
    assert out.iloc[1]["bucket"] == "−8% to −6%" and abs(out.iloc[1]["btc_ret"] + 10.0) < 1e-9
    # A missing previous BTC bar drops the day instead of carrying a close.
    out2 = eco.btc_on_nasdaq_down_days(ndx, close.drop(pd.Timestamp("2024-01-07")), "2024-01-01")
    assert list(out2["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-10"]
    st = eco.bucket_stats(out)
    assert list(st["bucket"]) == ["−8% to −6%", "−4% to −2%"] and list(st["n"]) == [1, 1]
    fig = charts.figure_nasdaq_down_days(out, st)
    assert len(fig.data) == 2


def test_daily_closes_later_frame_wins_and_drops_bad():
    a = pd.DataFrame({"timestamp": pd.to_datetime(["2018-07-31", "2018-08-01"]), "close": [1.0, 2.0]})
    b = pd.DataFrame({"timestamp": pd.to_datetime(["2018-08-01 08:00", "2018-08-02 08:00"]), "close": [3.0, 0.0]})
    s = eco.daily_closes(a, b)
    assert list(s.values) == [1.0, 3.0]
    assert eco.daily_closes(None, None).empty


def test_drawdown_episodes():
    idx = pd.date_range("2013-12-01", periods=12, freq="D")
    vals = [100, 120, 90, 60, 80, 121, 130, 110, 125, 131, 100, 70]
    dd = eco.drawdown_episodes(pd.Series(vals, index=idx, dtype=float), 0.20, active_from="2013-01-01")
    # 120 -> 60 (-50%) recovered at 121 after 4 days; 130->110 is only -15%, dropped; 131 -> 70 ongoing.
    assert len(dd) == 2
    first, cur = dd.iloc[0], dd.iloc[1]
    assert first["peak"] == 120 and first["trough"] == 60 and abs(first["depth"] + 0.5) < 1e-12
    assert first["days"] == 4 and not first["ongoing"]
    assert cur["ongoing"] and cur["peak"] == 131 and cur["days"] == 2 and pd.isna(cur["recovered"])
    # active_from drops episodes that recovered before it.
    late = eco.drawdown_episodes(pd.Series(vals, index=idx, dtype=float), 0.20, active_from="2013-12-07")
    assert len(late) == 1 and late.iloc[0]["ongoing"]
    # a lower threshold brings back the 15% dip
    assert len(eco.drawdown_episodes(pd.Series(vals, index=idx, dtype=float), 0.10, active_from="2013-01-01")) == 3


def test_drawdown_trend_and_figure():
    dd = pd.DataFrame({"peak_date": pd.to_datetime(["2014-01-01", "2016-01-01", "2018-01-01", "2020-01-01"]),
                       "peak": [1, 1, 1, 1], "trough_date": pd.to_datetime(["2014-02-01"] * 4), "trough": [1] * 4,
                       "recovered": pd.to_datetime(["2014-03-01", "2016-06-01", "2019-01-01", pd.NaT]),
                       "days": [10, 100, 1000, 400], "depth": [-0.2, -0.4, -0.6, -0.5],
                       "ongoing": [False, False, False, True]})
    a, b = eco.drawdown_trend(dd)
    assert abs(b + 20) < 1e-9 and abs(a - 0) < 1e-9   # -20 pts per decade, through 0 at 1 day
    fig = charts.figure_drawdowns(dd, (a, b), 0.2)
    names = [t.name for t in fig.data]
    assert "Current (ongoing)" in names and "Trend" in names
    assert fig.layout.xaxis.type == "log"


def test_rolling_flow_with_price():
    dates = pd.date_range("2025-01-01", periods=35, freq="D")
    flows = pd.DataFrame({"date": dates, "flow": np.arange(35, dtype=float)})
    px = pd.DataFrame({"timestamp": dates + pd.Timedelta(hours=8), "close": np.linspace(50_000, 60_000, 35)})
    out = eco.rolling_flow_with_price(flows, px, 30)
    assert out["roll"].iloc[:29].isna().all()
    assert out["roll"].iloc[29] == sum(range(30))
    assert out["close"].iloc[0] == 50_000
    fig = charts.figure_rolling_flow_price(out, 30)
    assert any(getattr(t.marker, "colorscale", None) for t in fig.data if t.mode == "markers")


def test_sma_rows_trading_days():
    d = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=250), "value": np.arange(250, dtype=float)})
    out = eco.sma_rows(d, 200)
    assert out["sma"].iloc[:199].isna().all() and out["sma"].iloc[199] == np.mean(range(200))
    fig = charts.figure_nasdaq_sma(out, "2024-06-01", "2024-12-31")
    assert len(fig.data) == 2


def test_strategy_8k_parser_fixture():
    f = s8.parse_8k_text(s8.html_to_text(FIXTURE.read_bytes()))
    assert f.period_start == pd.Timestamp("2026-11-02") and f.period_end == pd.Timestamp("2026-11-08")
    assert f.sources == {"MSTR ATM": 283.8}
    assert f.uses == {"BTC purchases": 81.9, "STRC buyback": 188.3, "USD Cash": 13.6}
    assert f.btc_bought == 1036 and f.btc_avg_price == 79012 and f.btc_holdings == 850364
    assert f.usd_reserve_b == 4.77 and f.usd_cash_m == 146.2
    # round trip through the CSV shape
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "strategy_funding.csv"
        s8.to_frame(f).to_csv(path, index=False)
        r = s8.load_result(live=False, csv_path=path)
        assert r.ok and r.source == "csv"
        g = s8.from_frame(r.df)
        assert g.sources == f.sources and g.uses == f.uses and g.btc_holdings == f.btc_holdings
    fig = charts.figure_strategy_funding(f)
    assert len(fig.data) == 1 and len(fig.data[0].x) == 4


def test_strategy_8k_parser_rejects_other_8k():
    try:
        s8.parse_8k_text("Item 2.02 Results of Operations and Financial Condition. Quarterly results.")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_strategy_fetch_latest_with_fake_edgar():
    html = FIXTURE.read_bytes()

    class R:
        def __init__(self, payload=None, content=b""):
            self._p, self.content, self.status_code = payload, content, 200
        def raise_for_status(self):
            pass
        def json(self):
            return self._p

    sub = {"filings": {"recent": {
        "form": ["10-Q", "8-K", "8-K"], "filingDate": ["2026-11-10", "2026-11-09", "2026-11-02"],
        "accessionNumber": ["x", "0001193125-26-452693", "y"], "primaryDocument": ["q.htm", "mstr-20261109.htm", "z.htm"],
        "items": ["", "7.01,8.01", "7.01,8.01"]}}}
    calls = []

    def get(url):
        calls.append(url)
        return R(sub) if url == s8.SUBMISSIONS_URL else R(content=html)

    f = s8.fetch_latest(get)
    assert f.accession == "0001193125-26-452693" and f.filed == "2026-11-09"
    assert calls[1].endswith("/1050446/000119312526452693/mstr-20261109.htm")

    def broken(url):
        raise OSError("offline")
    r = s8.load_result(get=broken, allow_csv=False)
    assert not r.ok and "offline" in r.message
