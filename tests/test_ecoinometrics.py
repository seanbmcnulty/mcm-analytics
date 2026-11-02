import pandas as pd
from lib import ecoinometrics as e


def test_streaks_and_rolling():
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=7),
                       "flow": [10, 5, -3, -4, 0, 8, 2]})
    s = e.flow_streaks(df)
    assert list(s["days"]) == [2, 2, 2] and list(s["total"]) == [15, -7, 10]
    assert e.rolling_flow(df, 3).iloc[2] == 12


def test_parse_farside():
    html = ("<table><tr><th>Date</th><th>IBIT</th><th>Total</th></tr>"
            "<tr><td>02 Jan 2026</td><td>1.0</td><td>(5.5)</td></tr>"
            "<tr><td>05 Jan 2026</td><td>-</td><td>1,200.0</td></tr></table>")
    d = e.parse_farside_html(html)
    assert list(d["flow"]) == [-5.5, 1200.0]


def test_hack_returns():
    px = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=20), "close": range(100, 120)})
    h = pd.DataFrame({"date": pd.to_datetime(["2025-01-03", "2024-01-01"]), "exchange": ["a", "b"]})
    r = e.hack_event_returns(px, h, 5)
    assert len(r) == 1 and abs(r["ret"][0] - 5 / 102) < 1e-9


def test_tone():
    assert e.tone_score("inflation is elevated and we may hike")["score"] > 0
    assert e.tone_score("we could cut as the labor market is softening")["score"] < 0


def test_new_helpers():
    import numpy as np
    d = pd.date_range("2026-01-01", periods=300)
    px = pd.DataFrame({"timestamp": d, "close": np.linspace(100, 200, 300)})
    g = e.sma_gap(px, 200)
    assert g["gap"].iloc[-1] > 0 and g["sma"].iloc[:199].isna().all()
    fl = pd.DataFrame({"date": d[:3], "flow": [100.0, -50.0, 10.0]})
    b = e.flows_to_btc(fl, px)
    assert abs(b["flow"][0] - 100e6 / 100) < 1e-6
    sp = e.streak_paths(pd.DataFrame({"date": d[:4], "flow": [1, 2, -1, -3]}))
    assert list(sp["cum"]) == [1, 3, -1, -4]
