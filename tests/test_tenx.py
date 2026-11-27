"""Offline tests for lib/tenx.py (no network)."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from lib import tenx


def _series(vals, start="2025-01-01", freq="D"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq=freq), dtype=float)


def test_registry_shape():
    keys = tenx.registry_keys()
    assert len(keys) == len(set(keys)) and keys
    from lib import tenx_charts
    for c in tenx.REGISTRY:
        assert c.key in tenx_charts.BUILDERS
        assert c.post_url.startswith("https://update.10xresearch.com/p/")
        pd.Timestamp(c.post_date)
        assert set(c.series) <= set(tenx.YAHOO_SYMBOLS)
    dates = [pd.Timestamp(c.post_date) for c in tenx.REGISTRY]
    assert dates == sorted(dates, reverse=True), "registry must be newest first"


def test_rsi_bounds_and_extremes():
    up = tenx.rsi(_series(np.arange(1, 60)))
    assert up.iloc[:14].isna().all()
    assert np.isclose(up.dropna(), 100).all()
    down = tenx.rsi(_series(np.arange(60, 1, -1)))
    assert np.isclose(down.dropna(), 0).all()
    rng = np.random.default_rng(1)
    r = tenx.rsi(_series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))))).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_matches_wilder_reference():
    # Classic Wilder / StockCharts example: the first 14-day RSI is 70.53, then 66.32.
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
              45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64]
    r = tenx.rsi(_series(closes))
    assert abs(r.iloc[14] - 70.53) < 0.15  # textbook closes are rounded to 2 dp
    assert abs(r.iloc[15] - 66.32) < 0.15


def test_rsi_frame_signal_and_view():
    # Steady fall into oversold, then a bounce: one buy reversal on the turn.
    vals = list(np.linspace(100, 60, 40)) + list(np.linspace(60.5, 66, 8))
    s = _series(vals)
    df, st = tenx.rsi_frame(s, n=14, ma=5)
    sig = df.index[df["signal"]]
    assert len(sig) == 1
    assert sig[0] >= s.index[40]
    assert df.loc[sig[0], "rsi"] <= 30 and df.loc[sig[0], "rsi"] > df.loc[sig[0], "ma"]
    assert st["view"] in {"bullish", "oversold", "neutral", "overbought", "bearish"}
    assert tenx.rsi_view(26.4, 0.8) == "bullish"
    assert tenx.rsi_view(26.4, -0.8) == "oversold"
    assert tenx.rsi_view(75, -1) == "bearish"
    assert tenx.rsi_view(50, 1) == "neutral"


def test_relative_return_simple():
    idx = pd.date_range("2024-01-01", "2025-06-30", freq="D")
    a = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    b = pd.Series(100.0, index=idx)
    rel = tenx.relative_return(a, b, 365)
    assert rel[rel.index < idx[0] + pd.Timedelta(days=365)].isna().all()
    d = pd.Timestamp("2025-06-30")
    exp = (a[d] / a[d - pd.Timedelta(days=365)] - 1) * 100
    assert np.isclose(rel[d], exp)


def test_fit_channel_contains_prices():
    idx = pd.date_range("2025-01-01", periods=300, freq="D")
    rng = np.random.default_rng(3)
    trend = 5000 - 3 * np.arange(300)
    s = pd.Series(trend + rng.normal(0, 40, 300), index=idx)
    s.iloc[0] = 5200  # make the start the high
    ch = tenx.fit_channel(s)
    assert ch is not None and ch.start == idx[0]
    assert -4 < ch.slope_per_day < -2
    for d, v in s.items():
        assert ch.at(d, "lower") - 1e-6 <= v <= ch.at(d, "upper") + 1e-6
    assert 0 <= ch.position(idx[-1], s.iloc[-1]) <= 1


def test_pair_frame_uses_other_dates():
    btc = _series(np.linspace(40000, 60000, 200))
    coin = btc[btc.index.dayofweek < 5] / 300
    df, st = tenx.pair_frame(btc, coin, corr_days=60, chg_days=30)
    assert (df.index.dayofweek < 5).all()
    assert st["corr"] > 0.99 or np.isnan(st["corr"]) is False
    assert np.isclose(st["btc_chg"], st["other_chg"])


def test_parse_yahoo_and_merge_csv():
    ts = [int(pd.Timestamp(d, tz="America/New_York").timestamp()) + 16 * 3600
          for d in ("2026-11-23", "2026-11-24", "2026-11-25")]
    payload = {"chart": {"result": [{"timestamp": ts, "meta": {"exchangeTimezoneName": "America/New_York"},
                                     "indicators": {"quote": [{"close": [10.0, None, 12.0]}]}}]}}
    s = tenx.parse_yahoo_chart(json.loads(json.dumps(payload)), "COIN")
    assert list(s.index.strftime("%Y-%m-%d")) == ["2026-11-23", "2026-11-25"]  # None dropped, not filled
    csv = pd.DataFrame({"BTC": [1.0, 2.0], "GOLD": [3.0, np.nan], "COIN": [9.0, 11.0]},
                       index=pd.to_datetime(["2026-11-20", "2026-11-23"]))
    csv.index.name = "date"
    df, src = tenx.merge_live_csv({"COIN": s}, csv)
    assert src == {"BTC": "csv", "GOLD": "csv", "COIN": "live+csv"}
    assert df.loc["2026-11-23", "COIN"] == 10.0   # live wins
    assert df.loc["2026-11-20", "COIN"] == 9.0    # csv fills a date live lacked
    df2, src2 = tenx.merge_live_csv({}, csv.drop(columns=["GOLD"]))
    assert src2["GOLD"] == "none"


def test_load_prices_offline_from_csv():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / tenx.PRICES_CSV
        pd.DataFrame({"date": ["2026-01-02", "2026-01-05"], "BTC": [1, 2], "GOLD": [3, 4], "COIN": [5, 6]}).to_csv(p, index=False)
        pr = tenx.load_prices(data_dir=Path(d), live=False)
        assert pr.source == {"BTC": "csv", "GOLD": "csv", "COIN": "csv"}
        assert len(pr.df) == 2


def test_committed_csv_is_real_and_recent():
    df = tenx.load_prices_csv()
    assert set(tenx.YAHOO_SYMBOLS) <= set(df.columns)
    assert df.index.min() <= pd.Timestamp("2023-12-01")
    for c in tenx.YAHOO_SYMBOLS:
        assert df[c].dropna().gt(0).all()


def test_builders_draw_from_committed_csv():
    from lib import tenx_charts
    pr = tenx.load_prices(live=False)
    for c in tenx.REGISTRY:
        fig, st = tenx_charts.BUILDERS[c.key](pr)
        assert len(fig.data) >= 2 and st


def test_registry_keyed_by_post():
    by = tenx.charts_by_post()
    assert "2026-11-27 | Bitcoin and Gold: The Thanksgiving Charts" in by
    assert tenx.has_post("2026-11-27", "Bitcoin  and Gold: The Thanksgiving Charts")
    assert not tenx.has_post("2026-11-26", "Bitcoin and Gold: The Thanksgiving Charts")
    pairs = [(c.post_id, c.exhibit) for c in tenx.REGISTRY]
    assert len(pairs) == len(set(pairs)), "same exhibit from the same post registered twice"
    for pid, charts in by.items():
        assert len({c.post_url for c in charts}) == 1


def test_anchor_and_charts_grouped_by_post():
    pid = "2026-11-27 | Bitcoin and Gold: The Thanksgiving Charts"
    assert tenx.anchor(pid) == "tenx-2026-11-27-bitcoin-and-gold-the-thanksgiving-charts"
    assert tenx.anchor(pid, "BTC daily RSI").endswith("btc-daily-rsi")
    by = tenx.charts_by_post()
    posts = list(by)
    dates = [pd.Timestamp(by[p][0].post_date) for p in posts]
    assert dates == sorted(dates, reverse=True)
    keys = [c.key for charts in by.values() for c in charts]
    assert set(keys) == set(tenx.registry_keys()) and len(keys) == len(tenx.REGISTRY)


def test_white_theme_and_scroll_page():
    from lib import tenx_charts
    pr = tenx.load_prices(live=False)
    fig, _st = tenx_charts.BUILDERS[tenx.REGISTRY[0].key](pr)
    layout = fig.layout
    assert layout.paper_bgcolor == "#FFFFFF"
    assert layout.plot_bgcolor == "#FFFFFF"
    assert tenx_charts.BG == "#FFFFFF" and tenx_charts.INK == "#1A1A1A"
    assert tenx_charts.GRID == "#E6E6E6"
    src = (Path(__file__).resolve().parent.parent / "pages" / "15_10x_Research.py").read_text(encoding="utf-8")
    assert "st.tabs(" not in src
    assert 'with st.expander("Refresh data", expanded=False):' in src
    assert "#### Contents" in src
    assert "tenx.charts_by_post()" in src
    assert "tenx.anchor(" in src
    for c in tenx.REGISTRY:
        assert c.key in tenx_charts.BUILDERS
