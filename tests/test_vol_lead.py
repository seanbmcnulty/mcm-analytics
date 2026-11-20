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


# -------------------------------------------------------------- intraday
from lib import vol_lead_intraday as vi  # noqa: E402

B5 = pd.Timedelta(minutes=5)


def _sessions_index(days=("2026-01-12", "2026-01-13", "2026-01-14"), first="09:35", last="16:00"):
    out = []
    for d in days:
        rng = pd.date_range(f"{d} {first}", f"{d} {last}", freq="5min", tz="America/New_York")
        out.append(rng.tz_convert("UTC"))
    return out[0].append(out[1:])


def test_bar_change_and_lags_never_cross_overnight():
    idx = _sessions_index()
    s = pd.Series(np.arange(len(idx), dtype=float), idx)
    d = vi.bar_change(s, B5)
    firsts = [t for t in idx if t.tz_convert("America/New_York").strftime("%H:%M") == "09:35"]
    assert all(np.isnan(d.loc[t]) for t in firsts)
    assert d.dropna().eq(1.0).all()
    lag = vi.lag_series(s, 1, B5)
    assert all(np.isnan(lag.loc[t]) for t in firsts)
    lead = vi.lag_series(s, -1, B5)          # one bar later
    lasts = [t for t in idx if t.tz_convert("America/New_York").strftime("%H:%M") == "16:00"]
    assert all(np.isnan(lead.loc[t]) for t in lasts)


def test_intraday_cross_corr_and_granger_find_10min_lead():
    days = [str(d.date()) for d in pd.bdate_range("2026-01-05", periods=30)]
    idx = _sessions_index(days)
    rng = np.random.default_rng(4)
    dx = pd.Series(rng.normal(size=len(idx)), idx)
    dy = 0.7 * vi.lag_series(dx, 2, B5).fillna(0) + 0.5 * pd.Series(rng.normal(size=len(idx)), idx)
    cc = vi.cross_corr(dx, dy, B5, 6)
    best = cc.loc[cc["corr"].idxmax()]
    assert int(best["minutes"]) == 10 and best["corr"] > 0.6
    g = vi.granger(dy, dx, B5, 3)
    gr = vi.granger(dx, dy, B5, 3)
    assert g.loc[g["lag"] >= 2, "p_value"].max() < 1e-6 and gr["p_value"].min() > 0.001


def test_dvol_at_obs_uses_candle_end_only():
    t0 = pd.Timestamp("2026-01-12 14:30", tz="UTC")
    candles = pd.DataFrame({"ts": [int((t0 + pd.Timedelta(minutes=i)).timestamp() * 1000) for i in range(6)],
                            "open": 0, "high": 0, "low": 0, "close": [10, 11, 12, 13, 14, 15]})
    obs = pd.DatetimeIndex([t0 + pd.Timedelta(minutes=5), t0 + pd.Timedelta(minutes=10)])
    v = vld.dvol_at_obs(candles, pd.Timedelta(minutes=1), obs)
    assert v.iloc[0] == 14 and np.isnan(v.iloc[1])     # no later value, no fill


def test_cash_session_mask_dst():
    winter = pd.DatetimeIndex(["2026-01-12 14:30", "2026-01-12 14:35", "2026-01-12 21:00", "2026-01-12 21:05"], tz="UTC")
    summer = pd.DatetimeIndex(["2026-07-13 13:35", "2026-07-13 20:00", "2026-07-13 20:05"], tz="UTC")
    assert list(vld.cash_session_mask(winter)) == [False, True, True, False]
    assert list(vld.cash_session_mask(summer)) == [True, True, False]
    sat = pd.DatetimeIndex(["2026-01-17 15:00"], tz="UTC")
    assert not vld.cash_session_mask(sat)[0]


def test_parse_yahoo_intraday_bar_end_and_future_drop():
    t0 = int(pd.Timestamp("2026-01-12 14:30", tz="UTC").timestamp())
    payload = {"chart": {"result": [{"timestamp": [t0, t0 + 300, t0 + 600],
                                     "indicators": {"quote": [{"close": [20.0, None, 21.0]}]}}]}}
    s = vld.parse_yahoo_intraday(payload, B5, now=pd.Timestamp("2026-01-12 14:42", tz="UTC"))
    assert list(s.index) == [pd.Timestamp("2026-01-12 14:35", tz="UTC")] and s.iloc[0] == 20.0


class _Resp:
    def __init__(self, j):
        self._j = j

    def raise_for_status(self):
        pass

    def json(self):
        return self._j


def test_build_intraday_fetches_only_missing_sessions():
    days = ["2026-01-12", "2026-01-13", "2026-01-14"]
    idx = _sessions_index(days)
    starts = [int((t - B5).timestamp()) for t in idx]
    calls = {"deribit": 0}

    def get(url, params=None, headers=None, timeout=None):
        if "yahoo" in url:
            return _Resp({"chart": {"result": [{"timestamp": starts,
                                                "indicators": {"quote": [{"close": [20.0] * len(starts)}]}}]}})
        calls["deribit"] += 1
        a, b = params["start_timestamp"], params["end_timestamp"]
        ts = list(range(a - a % 60000, b, 60000))
        return _Resp({"result": {"data": [[t, 1, 1, 1, 50.0] for t in ts]}})

    now = pd.Timestamp("2026-01-14 22:00", tz="UTC")
    full = vld.build_intraday("5m", None, get=get, now=now, pause=0)
    assert calls["deribit"] == 6 and full["DVOL_BTC"].notna().all() and len(full) == len(idx)
    calls["deribit"] = 0
    again = vld.build_intraday("5m", full, get=get, now=now, pause=0)
    assert calls["deribit"] == 2          # only the latest stored session, both assets
    assert len(again) == len(full)


def test_session_predict_uses_only_post_cut_targets():
    days = [str(d.date()) for d in pd.bdate_range("2025-06-02", periods=120)]
    rows = []
    rng = np.random.default_rng(9)
    for d in days:
        dv_m = rng.normal()
        dvix = rng.normal()
        for h in range(10, 17):
            t = pd.Timestamp(f"{d} {h}:00", tz="America/New_York").tz_convert("UTC")
            vix = 20 + (dvix if h >= 12 else 0)
            dvol = 50 + (dv_m if h >= 12 else 0) + (2.0 * dvix if h == 16 else 0)   # afternoon follows the morning VIX
            rows.append((t, vix, dvol, dvol))
    h1 = pd.DataFrame(rows, columns=["obs", "VIX", "DVOL_BTC", "DVOL_ETH"]).set_index("obs")
    sp = vi.session_predict(h1, "BTC")
    r = sp.iloc[0]
    assert r["target"].startswith("rest of session") and abs(r["beta_vix"] - 2.0) < 0.05 and r["oos_r2_pct"] > 90


def test_intraday_events_same_session():
    idx = _sessions_index([str(d.date()) for d in pd.bdate_range("2026-01-05", periods=15)])
    vix = pd.Series(20.0, idx) + np.sin(np.arange(len(idx))) * 0.01
    t_ev = idx[-30]
    vix.loc[vix.index >= t_ev] += 3
    dvol = pd.Series(50.0, idx)
    dvol.loc[dvol.index >= t_ev + 2 * B5] += 1
    ev = vi.find_moves(vi.bar_change(vix, B5), 3.0, lookback=200)
    assert list(ev["obs"]) == [t_ev]
    p = vi.event_path(dvol, ev, B5, pre=2, post=4).set_index("minutes")
    assert p.loc[0, "mean"] == 0 and p.loc[10, "mean"] == 1


def test_short_window_rich_cheap_and_intraday_fair():
    df = vl.asset_frame(vl.align_panel(_panel(n=400, seed=12)), "BTC")
    rc = vl.rich_cheap(df, window=63, long_window=126)
    coef = rc.attrs["coef"]
    assert set(coef) == {"const", "VIX", "MOVE"}
    last = rc.dropna(subset=["fair"]).index[-1]
    assert np.isclose(rc.loc[last, "fair"], coef["const"] + coef["VIX"] * df.loc[last, "VIX"] + coef["MOVE"] * df.loc[last, "MOVE"])
    assert 0 <= vl.window_pct(rc["score"].iloc[-126:]) <= 100
    day = pd.DataFrame({"VIX": [20.0, 21.0], "DVOL_BTC": [50.0, 52.0]})
    fv = vi.intraday_fair(day, "BTC", {"const": 10.0, "VIX": 2.0, "MOVE": 0.1}, 100.0)
    assert list(fv["fair"]) == [60.0, 62.0] and list(fv["resid"]) == [-10.0, -10.0]


def test_committed_intraday_csvs_load():
    for k in ("5m", "1h"):
        d = vld.load_intraday(k)
        assert len(d) > 1000 and d.index.tz is not None, k
        assert vld.cash_session_mask(d.index, vld.INTRADAY[k]["first"], vld.INTRADAY[k]["last"]).all()


# ------------------------------------------------------------------ explain / charts (readability layer)
def _cc(corrs, band=0.05, unit="min"):
    lags = list(range(-3, 4))
    return pd.DataFrame({"lag": lags, "minutes": [5 * l for l in lags], "corr": corrs, "n": 1000, "band": band})


def test_leadlag_takeaway_needs_unbroken_run_from_first_lag():
    from lib import vol_lead_explain as vx
    # lead at 5 and 10 min, gap at 15: peak 5 min, run ends at 10
    cc = _cc([0.0, 0.0, 0.0, 0.2, 0.15, 0.08, 0.01])
    s = vx.leadlag_stats(cc, "min")
    assert s["peak_x"] == 5 and s["sig_x"] == [5.0, 10.0] and s["isolated_x"] == []
    cc2 = _cc([0.0, 0.0, 0.0, 0.2, 0.15, 0.0, 0.09])  # 15 min bar is isolated
    s2 = vx.leadlag_stats(cc2, "min")
    assert s2["sig_x"] == [5.0] and s2["isolated_x"] == [15.0]
    t2 = vx.leadlag_takeaway(cc2, "min", "BTC")
    assert "about 5 min" in t2 and "corr +0.01" not in t2 and "Isolated significant bar at 15 min" in t2
    # no lag-1 lead -> "no meaningful lead" even with a significant bar further out
    cc3 = _cc([0.0, 0.0, 0.0, 0.3, 0.01, 0.0, 0.09])
    t3 = vx.leadlag_takeaway(cc3, "min", "ETH")
    assert t3.startswith("No meaningful VIX lead") and "Same-bar" not in t3.split(".")[0]


def test_granger_and_signal_tables_use_plain_verdicts():
    from lib import vol_lead_explain as vx
    gr = pd.DataFrame({"lag": [1, 2], "minutes": [5, 10], "F": [9.0, 1.0], "p_value": [0.0004, 0.4], "n": [100, 99]})
    rev = gr.assign(p_value=[0.03, 0.2])
    t = vx.granger_table(gr, rev, "min")
    assert list(t.columns) == ["Past moves included", "Past VIX helps predict DVOL?", "Past DVOL helps predict VIX?",
                               "Observations"]
    assert t.iloc[0, 1] == "✅ yes (p<0.001)" and t.iloc[1, 1] == "— no (p=0.40)" and t.iloc[0, 2] == "✅ yes (p=0.030)"
    assert "both ways" in vx.granger_takeaway(gr, rev, "min", "BTC")
    sig = pd.DataFrame({"target": ["DVOL", "DVOL"], "h": [5, 20], "slope": [-1.5, 0.2], "t": [-2.5, 0.4],
                        "r2": [0.1, 0.0], "oos_r2_pct": [8.0, -1.0], "cw_p": [0.01, 0.6], "n": 200, "n_oos": 100})
    sp = vx.signal_plain(sig)
    assert sp.iloc[0]["Rich → falls? (vols per +1 sd)"].startswith("✅") and sp.iloc[1]["Beats naive forecast?"].startswith("—")
    assert vx.signal_takeaway(sig, "BTC").startswith("Yes")


def test_ranking_plain_sorted_by_usefulness():
    from lib import vol_lead_explain as vx
    rows = []
    for lab, g, p in (("VIX", -1.0, 0.5), ("DXY", 0.6, 0.01), ("MOVE", 0.9, 0.2)):
        for h in (1, 5):
            rows.append({"predictor": lab, "label": lab, "h": h, "oos_r2_gain_pct": g if h == 5 else -0.5, "cw_p": p})
    rank = pd.DataFrame(rows)
    t = vx.ranking_plain(rank)
    assert list(t["Predictor"]) == ["DXY", "MOVE", "VIX"]  # significant first, then by gain
    assert t.iloc[0]["Improves forecast?"] == "✅ yes, 5d" and t.iloc[1]["Improves forecast?"] == "— no"
    m = vx.ranking_sig_mask(rank, t)
    assert m.loc[0, "5d gain %"] and not m.loc[1, "5d gain %"] and not m.loc[0, "1d gain %"]
    assert "DXY" in vx.ranking_takeaway(rank, "BTC") and "VIX itself never helps" in vx.ranking_takeaway(rank, "BTC")


def test_verdict_bullets_are_short_and_computed():
    from lib import vol_lead_explain as vx
    df = vl.asset_frame(vl.align_panel(_panel(n=400, seed=12)), "BTC")
    rc = vl.rich_cheap(df, window=63, long_window=126)
    cc5 = _cc([0.0, 0.0, 0.0, 0.2, 0.15, 0.01, 0.0], band=0.03)
    ccd = vl.cross_corr(vl.change(df, "VIX"), df["DVOL"].diff(), 5)
    b = vx.verdict_bullets("BTC", "1 year", cc5=cc5, cc_daily=ccd, rc=rc, pct=37.0)
    assert 2 <= len(b) <= 4
    assert "about 5 min" in b[0] and "corr +0.15" in b[0]
    last = rc.dropna(subset=["score"]).iloc[-1]
    assert f"{abs(last['resid']):.1f} vols" in b[-1] and "37th percentile of the last 1 year" in b[-1]


def test_event_takeaway_flags_insignificant_moves():
    from lib import vol_lead_explain as vx
    p = pd.DataFrame({"offset": [0, 1, 2], "minutes": [0, 60, 120], "mean": [0.5, 0.2, 0.3],
                      "lo": [0.1, -0.1, -0.2], "hi": [0.9, 0.5, 0.8], "n": [10, 10, 10]})
    t = vx.event_takeaway(p, "VIX jumps", "BTC", (0, 120), "min")
    assert t == "After 10 VIX jumps, BTC DVOL averaged +0.5 vols in the jump bar, +0.3 vols after 2h (not significant)."


def test_charts_mark_noise_band_peak_and_sides():
    from lib import vol_lead_charts as vlc
    cc = _cc([0.0, 0.0, 0.0, 0.2, 0.15, 0.01, 0.0])
    f = vlc.intraday_cross_corr(cc, "BTC", "min")
    names = {t.name for t in f.data}
    assert "VIX leads (significant)" in names and "Noise (inside band)" in names
    texts = " ".join(a.text for a in f.layout.annotations)
    assert "peak: VIX leads by 5 min" in texts and "DVOL moves first" in texts and "VIX moves first" in texts
    assert any(s.type == "rect" for s in f.layout.shapes)  # shaded noise band
    g = vlc.rich_cheap_gauge(19.0, -0.98, -0.5, "BTC", "the last 1 year")
    assert "cheap" in g.layout.title.text and "19th percentile" in g.layout.title.text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
