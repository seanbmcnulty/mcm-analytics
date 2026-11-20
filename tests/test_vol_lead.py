"""lib/vol_lead.py + lib/vol_lead_data.py: analysis maths and parsers.

Synthetic series here are test fixtures only; the page uses real data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from lib import vol_lead as vl
from lib import vol_lead_data as vld


def _days(n):
    return pd.bdate_range("2021-03-24", periods=n)


def _lead_pair(n=900, k=2, beta=0.8, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = beta * np.roll(x, k) + 0.5 * rng.normal(size=n)
    y[:k] = 0
    idx = _days(n)
    return pd.Series(x, idx), pd.Series(y, idx)


def _panel(n=800, seed=3, lead=True):
    rng = np.random.default_rng(seed)
    idx = _days(n)
    dvix = rng.normal(0, 1, n)
    vix = 20 + np.cumsum(dvix) * 0.3
    vix = np.clip(vix, 9, None)
    move = 100 + np.cumsum(rng.normal(0, 2, n)) * 0.3
    dd = (1.5 * np.roll(dvix, 1) if lead else 0) + rng.normal(0, 1, n)
    dvol = 55 + np.cumsum(dd) * 0.5 + 0.5 * (vix - 20)
    return pd.DataFrame({"VIX": vix, "MOVE": move, "VIX3M": vix + 1, "VIX9D": vix - 0.5, "VVIX": 100 + vix,
                         "DXY": 100 + np.cumsum(rng.normal(0, .3, n)) * .1, "NDX": 10000 * np.exp(np.cumsum(rng.normal(0, .01, n))),
                         "DVOL_BTC": dvol, "DVOL_ETH": dvol * 1.2}, index=idx)


def test_cross_corr_finds_lead():
    x, y = _lead_pair(k=2)
    cc = vl.cross_corr(x, y, 5)
    best = cc.loc[cc["corr"].abs().idxmax()]
    assert int(best["lag"]) == 2 and best["corr"] > 0.6
    assert abs(cc.loc[cc["lag"] == -2, "corr"].iloc[0]) < 0.15
    assert np.isclose(cc["band"].iloc[0], 1.96 / np.sqrt(cc["n"].iloc[0]))


def test_granger_direction():
    x, y = _lead_pair(k=1)
    fwd = vl.granger(y, x, 3)
    rev = vl.granger(x, y, 3)
    assert fwd["p_value"].max() < 1e-6
    assert rev["p_value"].min() > 0.01


def test_ols_hac_matches_ols_coef_and_hac_wider_with_autocorr():
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=n)
    e = np.convolve(rng.normal(size=n + 10), np.ones(10) / 3, "valid")[:n]   # MA(9) errors
    y = 1.0 + 2.0 * x + e
    X = np.column_stack([np.ones(n), x])
    r = vl.ols_hac(y, X, lags=0)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(r.coef, b)
    r10 = vl.ols_hac(y, X, lags=10)
    assert r10.se[0] > r.se[0] * 1.5           # serial correlation -> wider intercept se
    assert 0 < r.r2 < 1


def test_expanding_forecasts_no_lookahead():
    idx = _days(400)
    y = pd.Series(np.arange(400, dtype=float), idx)
    X = pd.DataFrame({"a": np.arange(400, dtype=float)}, idx)
    f = vl.expanding_forecasts(y, X, h=5, min_train=100, refit_every=1)
    assert f.iloc[:104].isna().all() and f.iloc[104:].notna().all()
    # a later-only change to y must not move earlier forecasts
    y2 = y.copy()
    y2.iloc[300:] += 1000
    f2 = vl.expanding_forecasts(y2, X, h=5, min_train=100, refit_every=1)
    assert np.allclose(f.iloc[105:305], f2.iloc[105:305])


def test_oos_r2_and_clark_west():
    rng = np.random.default_rng(5)
    n = 500
    sig = rng.normal(size=n)
    y = pd.Series(sig + 0.3 * rng.normal(size=n))
    good, base = pd.Series(sig), pd.Series(np.zeros(n))
    r2, m = vl.oos_r2(y, good, base)
    assert r2 > 0.8 and m == n
    assert vl.clark_west(y, base, good, 1) < 0.001
    assert vl.oos_r2(y, base, base)[0] == 0.0


def test_predictive_regression_and_ranking_detect_planted_lead():
    df = vl.asset_frame(vl.align_panel(_panel(lead=True)), "BTC")
    r = vl.predictive_regression(df, "VIX", 1, min_train=200)
    assert r["ok"] and r["t"]["dVIX"] > 5 and r["oos_r2_vs_base"] > 0.2 and r["cw_p"] < 0.01
    rank = vl.ranking_table(df, ["VIX", "MOVE"], (1, 5), min_train=200)
    top1 = rank[rank["h"] == 1].sort_values("oos_r2_gain_pct").iloc[-1]
    assert top1["predictor"] == "VIX"
    lines = vl.verdict("BTC", vl.cross_corr(vl.change(df, "VIX"), df["DVOL"].diff()),
                       vl.granger(df["DVOL"].diff(), vl.change(df, "VIX")),
                       vl.granger(vl.change(df, "VIX"), df["DVOL"].diff()), rank)
    assert any("Lead: VIX changes lead" in ln for ln in lines)
    assert any(ln.startswith("Bottom line: VIX adds") for ln in lines)


def test_verdict_says_no_power_without_lead():
    df = vl.asset_frame(vl.align_panel(_panel(lead=False, seed=11)), "BTC")
    rank = vl.ranking_table(df, ["MOVE"], (1,), min_train=200)
    lines = vl.verdict("BTC", vl.cross_corr(vl.change(df, "MOVE"), df["DVOL"].diff()),
                       vl.granger(df["DVOL"].diff(), vl.change(df, "MOVE")),
                       vl.granger(vl.change(df, "MOVE"), df["DVOL"].diff()), rank)
    assert any("no TradFi gauge" in ln or "adds the most" in ln for ln in lines)


def test_align_panel_calendar_and_features():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")   # includes a weekend
    raw = pd.DataFrame({"DVOL_BTC": np.arange(10.0) + 50, "VIX": [15, 16, 17, 18, 19, np.nan, np.nan, 20, 21, 22],
                        "MOVE": [100, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 105, 106, 107],
                        "VIX3M": 18.0, "VIX9D": 14.0}, index=idx)
    p = vl.align_panel(raw, ffill_limit=3)
    assert list(p.index) == [d for d, v in zip(idx, raw["VIX"]) if np.isfinite(v)]
    # Monday change spans Friday -> Monday (weekend DVOL move included)
    df = vl.asset_frame(p, "BTC")
    mon = pd.Timestamp("2024-01-08")
    assert df["DVOL"].diff().loc[mon] == 3.0
    assert np.isnan(p.loc["2024-01-05", "MOVE"])          # ffill limited to 3 rows
    assert p["TS_SLOPE"].iloc[0] == 3.0 and p["VIX9D_GAP"].iloc[0] == -1.0
    assert np.isclose(df["RATIO_VIX"].iloc[0], 50 / 15)


def test_events_and_paths():
    idx = _days(400)
    lvl = pd.Series(20.0 + np.sin(np.arange(400) / 7.0) * 0.2, idx)
    lvl.iloc[300] += 8      # spike, then stays
    lvl.iloc[301:] += 8
    lvl.iloc[305] += 0.5
    ev = vl.find_events(lvl, 2.0, 25.0, min_gap=10)
    assert len(ev) == 1 and ev["pos"].iloc[0] == 300
    dv = pd.Series(np.zeros(400), idx)
    dv.iloc[302:] = 4.0
    p = vl.event_path(dv, ev, pre=2, post=5).set_index("offset")
    assert p.loc[0, "mean"] == 0 and p.loc[2, "mean"] == 4 and p.loc[5, "n"] == 1


def test_rolling_helpers():
    s = pd.Series(np.arange(300, dtype=float))
    assert vl.rolling_pct(s, 100).iloc[-1] > 99
    assert vl.rolling_z(s, 100).iloc[-1] > 1.5
    rb = vl.rolling_beta_corr(2 * s.diff(), s.diff() + np.sin(s), 30)
    assert rb["beta"].dropna().between(0, 3).all()


def test_rich_cheap_score_and_signal():
    df = vl.asset_frame(vl.align_panel(_panel(n=900, seed=7)), "BTC")
    rc = vl.rich_cheap(df)
    assert {"fair", "resid", "resid_z", "score", "score_pct", "ratio_move_z"} <= set(rc.columns)
    last = rc.dropna(subset=["score"]).iloc[-1]
    comps = [last[c] for c in rc.attrs["components"]]
    assert np.isclose(last["score"], np.mean(comps))
    assert np.isclose(last["resid"], df["DVOL"].iloc[-1] - last["fair"])
    head = vl.rich_cheap_headline("BTC", rc)
    assert head.startswith("BTC vol is") and ("rich" in head or "cheap" in head)
    # planted mean reversion: forward DVOL change = -score
    sc = rc["score"].dropna()
    dv = pd.Series(np.cumsum(np.r_[0, -sc.values[:-1]]), sc.index)
    q = vl.score_quintiles(sc, dv, (1,))
    assert q.iloc[0]["fwd_dvol_mean"] > q.iloc[-1]["fwd_dvol_mean"]
    sig = vl.score_signal(sc, dv, (1, 5), min_train=200)
    assert (sig["slope"] < 0).all() and (sig["t"] < -5).all() and (sig["oos_r2_pct"] > 50).all()


def test_fair_value_recovers_linear_model():
    idx = _days(400)
    rng = np.random.default_rng(2)
    vix = pd.Series(20 + rng.normal(0, 3, 400), idx)
    move = pd.Series(100 + rng.normal(0, 10, 400), idx)
    df = pd.DataFrame({"DVOL": 10 + 2 * vix + 0.1 * move, "VIX": vix, "MOVE": move})
    fv = vl.fair_value(df)
    assert np.allclose(fv["fair"].dropna(), df["DVOL"].loc[fv["fair"].dropna().index], atol=1e-6)


def test_ordinal():
    assert [vl.ordinal(x) for x in (1, 2, 3, 4, 11, 12, 13, 21, 22, 99.6)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "100th"]


# -------------------------------------------------------------- data parsers
def test_snap_to_ny_close_dst_and_partial_day():
    def ms(s):
        return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)
    # winter (EST, UTC-5): 16:00 NY = 21:00 UTC -> candle starting 20:00 UTC
    win = [ms("2026-01-15 18:00") + i * 3600_000 for i in range(6)]
    # summer (EDT, UTC-4): 16:00 NY = 20:00 UTC -> candle starting 19:00 UTC
    summ = [ms("2026-07-15 17:00") + i * 3600_000 for i in range(6)]
    h = pd.DataFrame({"ts": win + summ, "close": list(range(6)) + list(range(10, 16))})
    s = vld.snap_to_ny_close(h)
    assert s.loc["2026-01-15"] == 2 and s.loc["2026-07-15"] == 12
    part = pd.DataFrame({"ts": [ms("2026-01-16 10:00")], "close": [1.0]})   # 05:00 NY: not a close yet
    assert vld.snap_to_ny_close(part).empty


def test_parse_cboe_and_yahoo():
    s = vld.parse_cboe_csv("DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2026,1,2,0.5,17.5\n01/05/2026,1,2,0.5,18\n", "VIX")
    assert list(s.values) == [17.5, 18.0] and s.index[0] == pd.Timestamp("2026-01-02")
    v = vld.parse_cboe_csv("DATE,VVIX\n01/02/2026,90.1\n", "VVIX")
    assert v.iloc[0] == 90.1
    t0 = int(pd.Timestamp("2026-01-02 14:30", tz="UTC").timestamp())
    payload = {"chart": {"result": [{"meta": {"exchangeTimezoneName": "America/New_York"},
                                     "timestamp": [t0, t0 + 86400 * 3],
                                     "indicators": {"quote": [{"close": [95.5, None]}]}}]}}
    y = vld.parse_yahoo_chart(payload, "MOVE")
    assert len(y) == 1 and y.index[0] == pd.Timestamp("2026-01-02")


def test_refresh_failure_keeps_csv(tmp=None):
    import tempfile
    d = Path(tempfile.mkdtemp())
    pd.DataFrame({"date": ["2026-01-02"], "VIX": [17.0]}).to_csv(d / "vl_cboe.csv", index=False)

    def boom(*a, **k):
        raise RuntimeError("offline")
    r = vld.refresh("cboe", data_dir=d, get=boom)
    assert not r.ok and "refresh failed" in r.message
    assert vld.load_csv("cboe", d).ok


def test_committed_csvs_load():
    for k in ("dvol", "cboe", "yahoo"):
        r = vld.load_csv(k)
        assert r.ok and len(r.df) > 500, k


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
