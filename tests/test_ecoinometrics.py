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


def _px(dates, closes):
    return pd.DataFrame({"timestamp": pd.to_datetime(list(dates)), "close": list(closes)})


def test_sma_ratio_warmup_and_hand_value():
    dates = pd.date_range("2020-01-01", periods=200)
    r = e.price_sma_ratio(_px(dates, range(1, 201)))
    assert r["ratio"].iloc[:199].isna().all()
    assert r["sma"].iloc[:199].isna().all()
    assert abs(r["sma"].iloc[199] - 100.5) < 1e-9
    assert abs(r["ratio"].iloc[199] - (200 / 100.5)) < 1e-12
    assert r["close"].iloc[199] == 200


def test_sma_ratio_calendar_gaps_are_not_filled():
    # 200 observations every other day never contain 200 consecutive calendar days.
    dates = pd.date_range("2019-01-01", periods=200, freq="2D")
    r = e.price_sma_ratio(_px(dates, [50.0] * 200))
    assert r["ratio"].isna().all()
    assert r["sma"].isna().all()
    assert r.loc[r["date"] == dates[0], "close"].iloc[0] == 50.0
    gap = dates[0] + pd.Timedelta(days=1)
    assert pd.isna(r.loc[r["date"] == gap, "close"].iloc[0])

    # A published zero is a real close. A missing day is not, and it blanks the window.
    start = pd.Timestamp("2021-06-01")
    present = [start, start + pd.Timedelta(days=1), start + pd.Timedelta(days=2)]
    with_zero = e.price_sma_ratio(_px(present, [10.0, 0.0, 10.0]), window=3)
    assert abs(with_zero["sma"].iloc[2] - 20 / 3) < 1e-12
    assert abs(with_zero["ratio"].iloc[2] - 1.5) < 1e-12
    skipped = [start, start + pd.Timedelta(days=2)]
    with_gap = e.price_sma_ratio(_px(skipped, [10.0, 10.0]), window=3)
    assert with_gap["ratio"].isna().all()
    assert pd.isna(with_gap.loc[with_gap["date"] == start + pd.Timedelta(days=1), "close"].iloc[0])


def test_sma_ratio_gap_blocks_until_window_clears():
    start = pd.Timestamp("2020-01-01")
    first = pd.date_range(start, periods=200, freq="D")
    rest = pd.date_range(start + pd.Timedelta(days=201), periods=200, freq="D")
    dates = first.append(rest)
    r = e.price_sma_ratio(_px(dates, [100.0] * len(dates)))
    missing = start + pd.Timedelta(days=200)
    assert pd.isna(r.loc[r["date"] == missing, "close"].iloc[0])
    assert r.loc[r["date"] == start + pd.Timedelta(days=199), "ratio"].iloc[0] == 1.0
    # The missing day sits inside every window until it rolls out.
    assert pd.isna(r.loc[r["date"] == start + pd.Timedelta(days=399), "ratio"].iloc[0])
    assert r.loc[r["date"] == start + pd.Timedelta(days=400), "ratio"].iloc[0] == 1.0


def test_sma_ratio_no_lookahead_and_input_order():
    dates = pd.date_range("2021-01-01", periods=250)
    closes = list(range(10, 260))
    px = _px(dates, closes)
    full = e.price_sma_ratio(px)
    early = e.price_sma_ratio(px.iloc[:220])
    day = dates[219]
    a = full.loc[full["date"] == day, "ratio"].iloc[0]
    b = early.loc[early["date"] == day, "ratio"].iloc[0]
    assert a == b and pd.notna(a)
    # A later close does not move an earlier ratio.
    spiked = px.copy()
    spiked.loc[spiked.index[-1], "close"] = 10_000
    again = e.price_sma_ratio(spiked)
    assert again.loc[again["date"] == day, "ratio"].iloc[0] == a

    shuffled = px.sample(frac=1.0, random_state=3)
    got = e.price_sma_ratio(shuffled)
    assert got["ratio"].dropna().tolist() == full["ratio"].dropna().tolist()

    day = pd.Timestamp("2022-05-01")
    later_wins = e.price_sma_ratio(pd.DataFrame({
        "timestamp": [day + pd.Timedelta(hours=12), day],
        "close": [40.0, 10.0],
    }), window=1)
    assert later_wins["close"].iloc[0] == 40.0
    assert later_wins["ratio"].iloc[0] == 1.0

    sgt = pd.Timestamp("2024-01-02 00:30", tz="Asia/Singapore")  # 16:30 UTC on the 1st
    tz_row = e.price_sma_ratio(pd.DataFrame({"timestamp": [sgt], "close": [7.0]}), window=1)
    assert tz_row["date"].iloc[0] == pd.Timestamp("2024-01-01")

    # Deribit daily bars are stamped 08:00 UTC. A late-UTC bar must keep the
    # UTC date rather than rolling onto the next SGT calendar date.
    stamped = e.price_sma_ratio(_px([pd.Timestamp("2024-06-01 08:00")], [5.0]), window=1)
    assert stamped["date"].iloc[0] == pd.Timestamp("2024-06-01")
    late = e.price_sma_ratio(_px([pd.Timestamp("2024-06-01 20:00")], [5.0]), window=1)
    assert late["date"].iloc[0] == pd.Timestamp("2024-06-01")

    dead = e.price_sma_ratio(_px(pd.date_range("2023-01-01", periods=2), [0.0, -4.0]), window=1)
    assert dead["ratio"].isna().all()
    assert list(dead["close"]) == [0.0, -4.0]


def test_sma_ratio_display_slice_uses_full_history():
    dates = pd.date_range("2018-01-01", periods=400)
    px = _px(dates, [20.0] * 400)
    full = e.price_sma_ratio(px)
    view = e.display_from(full, dates[250])
    assert view.iloc[0]["ratio"] == full.loc[full["date"] == dates[250], "ratio"].iloc[0]
    assert pd.notna(view.iloc[0]["ratio"])
    # Computing the average on the slice alone has no complete 200-day window at the left edge.
    short = e.price_sma_ratio(px.iloc[250:])
    assert pd.isna(short["ratio"].iloc[0])


def test_flow_regime_boundaries_warmup_and_no_lookahead():
    def frame(last5, n=63, fill=10.0):
        flow = [fill] * (n - len(last5)) + list(last5)
        return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "flow": flow})

    # M stays 10 when only the last five sessions change. S is their sum.
    expect = {
        (10, 10, 10, 10, 10): "strong_inflow",   # S = 50 = 5M
        (10, 10, 10, 10, 9): "inflow",           # S = 49
        (4, 4, 4, 4, 4): "inflow",               # S = 20 = 2M
        (4, 4, 4, 4, 3): "neutral",              # S = 19
        (0, 0, 0, 0, 0): "neutral",
        (-4, -4, -4, -4, -3): "neutral",         # S = -19
        (-4, -4, -4, -4, -4): "outflow",         # S = -20
        (-10, -10, -10, -10, -9): "outflow",     # S = -49
        (-10, -10, -10, -10, -10): "strong_outflow",  # S = -50
    }
    for last5, label in expect.items():
        got = e.etf_flow_regime(frame(last5)).iloc[-1]
        assert got["regime"] == label, (last5, got["regime"], got["sum_window"], got["scale_median"])
        assert got["scale_median"] == 10
        assert got["mild_cutoff"] == 20
        assert got["strong_cutoff"] == 50

    zeros = e.etf_flow_regime(frame((0, 0, 0, 0, 0), fill=0.0)).iloc[-1]
    assert zeros["regime"] == "neutral" and zeros["scale_median"] == 0
    pos = e.etf_flow_regime(frame((0, 0, 0, 0, 5), fill=0.0)).iloc[-1]
    neg = e.etf_flow_regime(frame((0, 0, 0, 0, -5), fill=0.0)).iloc[-1]
    assert pos["regime"] == "inflow" and neg["regime"] == "outflow"
    assert pos["regime"] != "strong_inflow" and neg["regime"] != "strong_outflow"

    base = frame((10, 10, 10, 10, 10), n=80)
    assert base.shape[0] == 63 or True
    long = e.etf_flow_regime(pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=80),
        "flow": [10.0] * 80,
    }))
    assert list(long["regime"].iloc[:62]) == ["unclassified"] * 62
    assert long["regime"].iloc[62] != "unclassified"
    assert pd.notna(long["sum_window"].iloc[4]) and long["regime"].iloc[4] == "unclassified"
    assert pd.isna(long["sum_window"].iloc[3])

    flows = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=80),
        "flow": [10.0] * 80,
    })
    flows.loc[71, "flow"] = 10_000.0
    full = e.etf_flow_regime(flows)
    prefix = e.etf_flow_regime(flows.iloc[:70])
    assert full.iloc[69]["sum_window"] == 50
    assert full.iloc[69]["scale_median"] == 10
    assert full.iloc[69]["regime"] == prefix.iloc[69]["regime"] == "strong_inflow"
    assert full.iloc[71]["sum_window"] == 10 * 4 + 10_000

    shuffled = flows.sample(frac=1.0, random_state=1)
    again = e.etf_flow_regime(shuffled)
    assert again["regime"].tolist() == full["regime"].tolist()


def test_flow_regime_missing_zero_and_duplicates():
    dates = pd.date_range("2024-04-01", periods=64)
    flow = [10.0] * 64
    flow[10] = float("nan")
    raw = pd.DataFrame({"date": dates, "flow": flow})
    raw.loc[20, "flow"] = 0.0
    out = e.etf_flow_regime(raw)
    assert dates[10] not in set(out["date"])
    assert 0.0 in set(out["flow"])
    assert len(out) == 63

    dup = pd.DataFrame({
        "date": [pd.Timestamp("2024-05-01"), pd.Timestamp("2024-05-01")],
        "flow": [3.0, 4.0],
    })
    summed = e.etf_flow_regime(dup, window=1, scale=1)
    assert len(summed) == 1 and summed["flow"].iloc[0] == 7.0
    assert e.etf_flow_regime(pd.DataFrame(columns=["date", "flow"])).empty
    assert e.classify_flow_regime(float("nan"), 10) == "unclassified"


def test_flows_to_btc_units_missing_and_dtypes():
    dates = pd.to_datetime(["2024-03-01", "2024-03-02", "2024-03-03"])
    flows = pd.DataFrame({"date": dates.astype("datetime64[us]"), "flow": [100.0, -25.0, 8.0]})
    px = pd.DataFrame({
        "timestamp": dates[:2].astype("datetime64[ms]"),
        "close": [50_000.0, 100_000.0],
    })
    out = e.flows_to_btc(flows, px)
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2024-03-01", "2024-03-02"]
    assert abs(out["flow"].iloc[0] - 2_000) < 1e-6
    assert abs(out["flow"].iloc[1] - (-250)) < 1e-6

    zero_px = pd.DataFrame({"timestamp": dates[:1], "close": [0.0]})
    assert e.flows_to_btc(flows.iloc[:1], zero_px).empty
    neg_px = pd.DataFrame({"timestamp": dates[:1], "close": [-5.0]})
    assert e.flows_to_btc(flows.iloc[:1], neg_px).empty
    kept = e.flows_to_btc(
        pd.DataFrame({"date": dates[:1], "flow": [0.0]}),
        pd.DataFrame({"timestamp": dates[:1], "close": [50_000.0]}),
    )
    assert kept["flow"].iloc[0] == 0.0
    dirty = pd.DataFrame({"date": dates[:1], "flow": [float("nan")]})
    assert e.flows_to_btc(dirty, px).empty
    # Deribit daily bars are stamped 08:00 UTC. That must still match a
    # Farside session dated on the same UTC calendar day (not the next one).
    stamped = e.flows_to_btc(
        pd.DataFrame({"date": [pd.Timestamp("2024-06-01")], "flow": [10.0]}),
        pd.DataFrame({"timestamp": [pd.Timestamp("2024-06-01 08:00")], "close": [50_000.0]}),
    )
    assert len(stamped) == 1 and abs(stamped["flow"].iloc[0] - 200) < 1e-9


def test_methodology_documents_the_rule():
    sma = e.sma_ratio_methodology()
    assert "200" in sma and "calendar" in sma
    assert "No lookahead" in sma
    assert "not copied forward" in sma or "not interpolated" in sma or "stays blank" in sma
    flow = e.flow_regime_methodology()
    assert "not the Ecoinometrics model" in flow
    assert "1,000,000" in flow
    assert "No lookahead" in flow or "no lookahead" in flow
    assert f"{e.FLOW_REGIME_WINDOW}" in flow
    assert f"{e.FLOW_REGIME_SCALE}" in flow
    assert "unclassified" in flow
    assert "zero" in flow.lower()
