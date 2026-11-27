"""Offline tests for lib/dailyshot.py and lib/dailyshot_charts.py (no network)."""
import numpy as np
import pandas as pd

from lib import dailyshot as ds


def _s(vals, start="2025-01-01", freq="B"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq=freq), dtype=float)


def test_chart_list_shape():
    from lib import dailyshot_charts as dc
    keys = [c.key for c in ds.CHARTS]
    assert len(keys) == len(set(keys)) and 35 <= len(keys) <= 60
    for c in ds.CHARTS:
        assert c.section in ds.SECTIONS, c.key
        assert c.kind in dc.KINDS, c.key
        assert c.title and c.caption
    order = [ds.SECTIONS.index(s) for s, _ in ds.sections()]
    assert order == sorted(order)
    assert sum(len(v) for _, v in ds.sections()) == len(ds.CHARTS)
    assert ds.anchor("Emerging Markets") == "ds-emerging-markets"


def test_symbols_and_ids():
    syms = ds.yahoo_symbols()
    assert "BTC-USD" in syms and "^GSPC" in syms and "10Y" not in syms
    assert len(syms) == len(set(syms))
    assert "CPIAUCSL" in ds.fred_ids()


def test_maths():
    s = pd.Series([100.0, 110.0, 99.0], index=pd.to_datetime(["2025-12-31", "2026-01-02", "2026-01-05"]))
    assert np.isclose(ds.period_return(s, "YTD"), -1)
    m = pd.Series(np.arange(100, 125, dtype=float), index=pd.date_range("2024-01-01", periods=25, freq="MS"))
    assert np.isclose(ds.yoy(m).iloc[0], 12.0)
    gap = m.drop(pd.Timestamp("2024-02-01"))
    assert pd.Timestamp("2025-02-01") not in ds.yoy(gap).index   # missing base month stays missing
    dd = ds.drawdown(_s([10, 12, 9, 12, 6]))
    assert np.isclose(dd.iloc[2], -25) and np.isclose(dd.iloc[-1], -50) and dd.iloc[1] == 0
    df = pd.DataFrame({"a": _s([10, 20, 30])})
    assert ds.rebase(df)["a"].tolist() == [100, 200, 300]
    w = ds.window(_s(np.arange(800.0), freq="D"), 1)
    assert 365 <= len(w) <= 366
    a = _s(np.exp(np.cumsum(np.random.default_rng(0).normal(0, .01, 200))))
    assert np.allclose(ds.rolling_corr(a, a, 20), 1)
    rv = ds.realised_vol(_s(100 * np.exp(np.arange(60) * 0.01), freq="D"))
    assert np.allclose(rv.dropna(), 0, atol=1e-6)


def test_parsers_and_merge():
    got = ds.parse_fred_csv("observation_date,UMCSENT\n2026-09-01,48.1\n2026-10-01,.\n")
    assert got["UMCSENT"].tolist() == [48.1]
    csv = pd.DataFrame({"UMCSENT": [47.0, 48.0]}, index=pd.to_datetime(["2026-08-01", "2026-09-01"]))
    df, src = ds.merge_live_csv({"UMCSENT": got["UMCSENT"]}, csv, ["UMCSENT", "X"])
    assert src == {"UMCSENT": "live+csv", "X": "none"} and df["UMCSENT"].tolist() == [47.0, 48.1]
    payload = {"chart": {"result": [{"timestamp": [1764000000, 1764086400, 1764172800],
                                     "meta": {"exchangeTimezoneName": "America/New_York"},
                                     "indicators": {"quote": [{"close": [1.0, None, 3.0]}]}}]}}
    assert ds.parse_yahoo_chart(payload, "X").tolist() == [1.0, 3.0]
    t = ds.parse_treasury_csv('Date,"1 Mo","2 Yr","10 Yr"\n11/25/2026,4.27,4.61,4.85\n11/24/2026,4.2,4.6,\n')
    assert t["10Y"].dropna().tolist() == [4.85] and t.index[0] == pd.Timestamp("2026-11-24")
    d = ds.parse_dvol({"result": {"data": [[1795737600000, 50, 51, 49, 48.58]]}}, "BTC")
    assert d.iloc[0] == 48.58


def test_offline_loaders_fall_back_to_csv():
    pr = ds.load_prices(live=False)
    assert set(pr.source.values()) <= {"csv", "none"}
    assert pr.source.get("^GSPC") == "csv"


def test_every_chart_draws_from_committed_data():
    """What Cloud shows when every live source fails."""
    from lib import dailyshot_charts as dc
    ctx = dc.Ctx(ds.load_prices(live=False), ds.load_fred(live=False), ds.load_yields(live=False),
                 ds.load_dvol(live=False))
    for c in ds.CHARTS:
        fig, info = dc.build(c, ctx)
        assert fig.data, c.key
        assert isinstance(info["as_of"], pd.Timestamp) and "committed CSV" in info["source"], c.key
