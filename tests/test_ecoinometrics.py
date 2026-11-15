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
    assert bool(r["complete"][0]) and abs(r["partial"][0] - r["ret"][0]) < 1e-12


def test_hack_anchor_is_event_date_bar():
    # Deribit 1D stamps are 08:00 UTC. A late-evening hack stays on that bar's
    # close (the next morning's settlement), not the previous bar and not the
    # next calendar date's bar.
    px = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-10-28 08:00", "2026-10-29 08:00", "2026-10-30 08:00",
            "2026-11-03 08:00", "2026-11-12 08:00",
        ]),
        "close": [200.0, 100.0, 80.0, 90.0, 110.0],
    })
    h = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-29", "2026-10-28"]),
        "exchange": ["Binance", "Earlier"],
        "loss_usd_m": [6340.0, 10.0],
    })
    r = e.hack_event_returns(px, h, 5).set_index("exchange")
    assert abs(r.loc["Binance", "ret"] - (90 / 100 - 1)) < 1e-12
    assert bool(r.loc["Binance", "complete"])
    # 28 Oct + 5 days is 2 Nov, missing, while a later bar (12 Nov) exists: a hole.
    assert "Earlier" not in r.index

    # 16:30 UTC is 00:30 the next day in Singapore. The bar still keys to the UTC date.
    sgt = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2026-10-30 00:30", tz="Asia/Singapore"),
            pd.Timestamp("2026-11-03 16:00", tz="Asia/Singapore"),
        ],
        "close": [100.0, 125.0],
    })
    anchored = e.hack_event_returns(sgt, h.iloc[:1], 5)
    assert len(anchored) == 1 and abs(anchored["ret"].iloc[0] - 0.25) < 1e-12

    # No close after the anchor: nothing to plot, and not a zero return.
    lone = pd.DataFrame({"timestamp": [pd.Timestamp("2026-10-29 08:00")], "close": [100.0]})
    assert e.hack_event_returns(lone, h.iloc[:1], 5).empty


def test_hack_incomplete_horizon_excluded_from_average():
    px = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-10-29 08:00", "2026-11-03 08:00", "2026-11-12 08:00",
        ]),
        "close": [100.0, 90.0, 110.0],
    })
    h = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-29"]),
        "exchange": ["Binance"],
        "loss_usd_m": [6340.0],
    })
    full = e.hack_event_returns(px, h, 5).iloc[0]
    assert bool(full["complete"]) and abs(full["ret"] - (90 / 100 - 1)) < 1e-12

    pending = e.hack_event_returns(px, h, 30).iloc[0]
    assert pending["complete"] == False
    assert pd.isna(pending["ret"])
    assert abs(pending["partial"] - (110 / 100 - 1)) < 1e-12
    assert int(pending["elapsed"]) == (pd.Timestamp("2026-11-12") - pd.Timestamp("2026-10-29")).days
    # mean() skips NaN, so the partial return cannot move the average.
    mixed = e.hack_event_returns(px, pd.concat([h, h.assign(exchange="copy")], ignore_index=True), 5)
    # second copy is the same complete event; a 30d call of the same prices is the pending case
    assert abs(mixed["ret"].mean() - (90 / 100 - 1)) < 1e-12
    assert pd.isna(e.hack_event_returns(px, h, 30)["ret"].mean())


def test_hack_csv_includes_2026_bitget_and_binance():
    h = e.load_hacks()
    bitget = h[(h["exchange"] == "Bitget") & (h["date"] == pd.Timestamp("2026-09-24"))].iloc[0]
    binance = h[(h["exchange"] == "Binance") & (h["date"] == pd.Timestamp("2026-10-29"))].iloc[0]
    assert abs(float(bitget["loss_usd_m"]) - 387.5) < 1e-9
    assert float(binance["loss_usd_m"]) == 6340
    assert "18:31" in str(bitget["note"]) and "387.5" in str(bitget["note"])
    assert "20:05" in str(binance["note"]) and "6.34" in str(binance["note"])

    # Price history ends 15 Nov 2026. Bitget's 30-day window closes 24 Oct; Binance's closes 28 Nov.
    days = pd.date_range("2026-09-01 08:00", "2026-11-15 08:00", freq="D")
    px = pd.DataFrame({"timestamp": days, "close": 1000.0 + (days - days[0]).days.astype(float)})
    r = e.hack_event_returns(px, h, 30).set_index("exchange")
    assert bool(r.loc["Bitget", "complete"]) and pd.notna(r.loc["Bitget", "ret"])
    assert r.loc["Binance", "complete"] == False and pd.isna(r.loc["Binance", "ret"])
    assert int(r.loc["Binance", "elapsed"]) == (pd.Timestamp("2026-11-15") - pd.Timestamp("2026-10-29")).days
    assert r["ret"].mean() == r.loc[r["complete"].astype(bool), "ret"].mean()


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
