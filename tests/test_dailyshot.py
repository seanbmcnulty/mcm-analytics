"""Offline tests for lib/dailyshot.py and lib/dailyshot_charts.py (no network)."""
import numpy as np
import pandas as pd

from lib import dailyshot as ds


def _s(vals, start="2025-01-01", freq="B"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq=freq), dtype=float)


def test_registry_shape():
    from lib import dailyshot_charts
    keys = ds.registry_keys()
    assert keys and len(keys) == len(set(keys))
    for c in ds.REGISTRY:
        assert c.key in dailyshot_charts.BUILDERS
        assert c.issue_url.startswith("https://thedailyshot.com/")
        assert set(c.series) <= set(ds.YAHOO_SYMBOLS)
        assert set(c.fred) <= set(ds.FRED_SERIES)
    dates = [pd.Timestamp(c.issue_date) for c in ds.REGISTRY]
    assert dates == sorted(dates, reverse=True), "registry must be newest first"
    assert set(dailyshot_charts.BUILDERS) == set(keys)


def test_issue_keys():
    c = ds.REGISTRY[0]
    assert ds.has_issue(c.issue_date, "  " + c.issue_title.replace(" ", "  ") + " ")
    assert not ds.has_issue("1999-01-01", c.issue_title)
    by = ds.charts_by_issue()
    assert sum(len(v) for v in by.values()) == len(ds.REGISTRY)
    assert list(by)[0].startswith(max(x.issue_date for x in ds.REGISTRY))


def test_rebase_and_ytd():
    df = pd.DataFrame({"a": _s([10, 20, 30]), "b": _s([5, 5, 10])})
    rb = ds.rebase(df)
    assert rb["a"].tolist() == [100, 200, 300] and rb["b"].iloc[-1] == 200
    rb2 = ds.rebase(df, df.index[1])
    assert rb2["a"].iloc[1] == 100
    s = pd.Series([100.0, 110.0, 99.0], index=pd.to_datetime(["2025-12-31", "2026-01-02", "2026-01-05"]))
    y = ds.ytd(s)
    assert np.isclose(y.iloc[0], 10) and np.isclose(y.iloc[-1], -1)


def test_defensive_ratio_and_period_returns():
    n = 30
    df = pd.DataFrame({c: _s(np.full(n, 10.0)) for c in ds.DEFENSIVES + ds.CYCLICALS})
    for c in ds.DEFENSIVES:
        df[c] = _s(np.linspace(10, 13, n))
    d = ds.defensive_ratio(df, window=5)
    assert (d["ratio"].diff().dropna() > 0).all() and (d["chg"].dropna() > 0).all()
    r = ds.period_returns(df[["XLP", "XLK"]], df.index[0], df.index[-1])
    assert np.isclose(r["XLP"], 30) and np.isclose(r["XLK"], 0)


def test_post_event_returns_and_yoy():
    s = _s([100, 101, 103, 99, 105], start="2026-11-02", freq="D")
    r = ds.post_event_returns(s, ["2026-11-03"])
    assert r["session"].iloc[0] == pd.Timestamp("2026-11-04")
    assert np.isclose(r["pct"].iloc[0], (103 / 101 - 1) * 100)
    m = pd.Series(np.arange(100, 125, dtype=float), index=pd.date_range("2024-01-01", periods=25, freq="MS"))
    assert np.isclose(ds.yoy(m).iloc[0], 12.0)


def test_highest_since():
    s = _s([1, 9, 2, 3, 2, 8, 2])
    hs = ds.highest_since(s, s.index[4])
    assert hs["peak"] == 8 and hs["since"] == s.index[1]
    assert ds.highest_since(s, s.index[1])["since"] is None


def test_parse_fred_and_merge():
    got = ds.parse_fred_csv("observation_date,UMCSENT\n2026-09-01,48.1\n2026-10-01,.\n")
    assert got["UMCSENT"].tolist() == [48.1]
    live = {"UMCSENT": got["UMCSENT"]}
    csv = pd.DataFrame({"UMCSENT": [47.0, 48.0]}, index=pd.to_datetime(["2026-08-01", "2026-09-01"]))
    df, src = ds.merge_live_csv(live, csv, ["UMCSENT", "CPIAUCSL"])
    assert src == {"UMCSENT": "live+csv", "CPIAUCSL": "none"}
    assert df["UMCSENT"].tolist() == [47.0, 48.1]


def test_parse_yahoo_chart():
    payload = {"chart": {"result": [{"timestamp": [1764000000, 1764086400, 1764172800],
                                     "meta": {"exchangeTimezoneName": "America/New_York"},
                                     "indicators": {"quote": [{"close": [1.0, None, 3.0]}]}}]}}
    s = ds.parse_yahoo_chart(payload, "X")
    assert s.tolist() == [1.0, 3.0]


def test_all_builders_on_committed_data():
    """Every chart draws from the committed CSVs alone (what Cloud uses when live fails)."""
    from lib import dailyshot_charts as dc
    pr, fr = ds.load_prices(live=False), ds.load_fred(live=False)
    yc = pd.read_csv(ds.DATA_DIR / "yield_curve.csv", parse_dates=["date"])
    ctx = dc.Ctx(pr, fr, yc, "data/yield_curve.csv")
    for c in ds.REGISTRY:
        fig, stats = dc.BUILDERS[c.key](ctx)
        assert fig.data, c.key
        assert stats, c.key


def test_layout_follows_issue_sections():
    lay = ds.layout()
    dates = [pd.Timestamp(x["date"]) for x in lay]
    assert dates == sorted(dates, reverse=True)
    for c in ds.REGISTRY:
        assert c.section in ds.issue_sections(c.issue_date), (c.key, c.section)
    for x in lay:
        order = [x["all_sections"].index(s) for s in x["sections"]]
        assert order == sorted(order)
    assert sum(len(v) for x in lay for v in x["sections"].values()) == len(ds.REGISTRY)
    assert ds.anchor("2026-11-24", "Emerging Markets") == "ds-2026-11-24-emerging-markets"
    done = {x["date"] for x in lay}
    assert not done & {d for d, *_ in ds.NOT_REBUILT}
