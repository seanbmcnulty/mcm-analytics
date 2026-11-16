"""OI-change math, last-print side, unusual-activity flags, snapshot no-wipe."""
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib import ibit as L  # noqa: E402

UTC = timezone.utc
NOW = datetime(2026, 11, 16, 18, 0, tzinfo=UTC)


def _chain(rows):
    """rows: (option, type, strike, oi, volume, last, bid, ask, delta, expiry)."""
    out = []
    for opt, typ, k, oi, vol, last, bid, ask, delta, exp in rows:
        out.append(dict(
            option=opt, type=typ, strike=k, oi=oi, volume=vol, last=last, bid=bid,
            ask=ask, delta=delta, expiry=pd.Timestamp(exp), spot=50.0, mid=(bid + ask) / 2,
            mult=100, iv=0.40, dte=10.0, m=k / 50.0))
    return pd.DataFrame(out)


def _store_row(session, option, oi, volume=0.0, strike=50.0, typ="C",
               expiry="2026-12-18T21:00:00Z", delta=0.4, last=1.0):
    return dict(session_date=session, recorded_at=f"{session}T21:50:00+00:00",
                option=option, expiry=expiry, type=typ, strike=strike, spot=50.0,
                bid=0.9, ask=1.1, last=last, iv=0.4, volume=volume, oi=oi, delta=delta)


def test_oi_change_math_and_expired_are_not_closes():
    asof = datetime(2026, 11, 16, 21, 0, tzinfo=UTC)
    current = _chain([
        ("A", "C", 50, 140, 10, 1.2, 1.0, 1.2, 0.4, "2026-12-18T21:00:00Z"),
        ("D", "P", 40, 7, 7, 0.5, 0.4, 0.6, -0.3, "2026-12-18T21:00:00Z"),
    ])
    prior = pd.DataFrame([
        _store_row("2026-11-13", "A", 100, strike=50, typ="C"),
        _store_row("2026-11-13", "B", 50, strike=45, typ="P", delta=-0.4),
        _store_row("2026-11-13", "C", 20, strike=55, typ="C",
                   expiry="2026-11-13T21:00:00Z"),
    ])
    ch = L.oi_change(current, prior, asof=asof)
    by = ch.set_index("option")
    assert by.loc["A", "oi_change"] == 40 and by.loc["A", "status"] == "increase"
    assert by.loc["B", "oi_change"] == -50 and by.loc["B", "status"] == "closed"
    assert by.loc["C", "oi_change"] == -20 and by.loc["C", "status"] == "expired"
    assert by.loc["D", "oi_change"] == 7 and by.loc["D", "status"] == "new"
    assert by.loc["A", "oi_change_usd"] == 40 * 100 * 50
    up, down = L.largest_oi_moves(ch, n=10)
    assert list(up["option"]) == ["A", "D"]
    assert list(down["option"]) == ["B"]          # expiry drop is not a close
    assert "C" not in set(up["option"]) | set(down["option"])


def test_aggressor_classification():
    last = [1.20, 1.25, 1.00, 0.95, 1.10, 1.00, 0.0, 1.10, 1.05]
    bid =  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.0, 1.10]
    ask =  [1.20, 1.20, 1.20, 1.20, 1.20, 1.00, 1.20, 1.20, 1.00]
    got = list(L.classify_aggressor(last, bid, ask))
    assert got == [
        "bought",       # at the ask
        "bought",       # above the ask
        "sold",         # at the bid
        "sold",         # below the bid
        "ambiguous",    # inside the spread
        "ambiguous",    # locked bid == ask == last
        "unknown",      # no last
        "unknown",      # no bid
        "unknown",      # crossed market
    ]


def test_unusual_activity_flags_premium_and_delta():
    df = _chain([
        # volume > prior OI and OI rose
        ("OPEN", "C", 50, 110, 150, 2.5, 2.4, 2.6, 0.40, "2026-12-18T21:00:00Z"),
        # volume below prior, OI fell
        ("QUIET", "C", 50, 90, 10, 2.5, 2.4, 2.6, 0.40, "2026-12-18T21:00:00Z"),
        # small volume but OI rose
        ("ROSE", "P", 40, 130, 10, 2.5, 2.4, 2.6, -0.25, "2026-12-18T21:00:00Z"),
        # volume > prior even though OI fell
        ("TRADED", "C", 55, 80, 150, 2.5, 2.4, 2.6, 0.20, "2026-12-18T21:00:00Z"),
        # new listing, prior treated as 0
        ("NEW", "C", 60, 5, 80, 2.5, 2.4, 2.6, 0.10, "2026-12-18T21:00:00Z"),
        # dust, dropped by the premium floor
        ("DUST", "C", 50, 10, 100, 0.01, 0.01, 0.02, 0.50, "2026-12-18T21:00:00Z"),
    ])
    prior = pd.DataFrame([
        _store_row("2026-11-13", "OPEN", 100),
        _store_row("2026-11-13", "QUIET", 100),
        _store_row("2026-11-13", "ROSE", 100),
        _store_row("2026-11-13", "TRADED", 100),
        _store_row("2026-11-13", "DUST", 0),
    ])
    screen = L.unusual_activity(df, prior, min_premium=1_000, min_contracts=1)
    flags = screen.set_index("option")["likely_opening"].to_dict()
    assert flags["OPEN"] and flags["ROSE"] and flags["TRADED"] and flags["NEW"]
    assert not flags["QUIET"]
    assert "DUST" not in flags          # 100 × 0.01 × 100 = $100
    assert "DUST" not in L.unusual_activity(df, prior).set_index("option").index  # default $25k floor
    open_row = screen.set_index("option").loc["OPEN"]
    assert open_row["premium"] == 150 * 2.5 * 100
    assert open_row["delta_shares"] == 150 * 0.40 * 100
    assert open_row["delta_usd"] == 150 * 0.40 * 100 * 50
    assert list(screen["premium"]) == sorted(screen["premium"], reverse=True)

    no_prior = L.unusual_activity(df, None, min_premium=0, min_contracts=0).set_index("option")
    assert bool(no_prior.loc["OPEN", "likely_opening"])      # 150 > 110
    assert not bool(no_prior.loc["QUIET", "likely_opening"])  # 10 < 90


def test_lean_nets_use_last_print_only():
    df = _chain([
        ("C_BUY", "C", 50, 10, 10, 1.20, 1.00, 1.20, 0.50, "2026-12-18T21:00:00Z"),
        ("C_MID", "C", 50, 10, 4, 1.10, 1.00, 1.20, 0.50, "2026-12-18T21:00:00Z"),
        ("P_SELL", "P", 40, 10, 5, 0.40, 0.40, 0.60, -0.40, "2026-12-18T21:00:00Z"),
    ])
    table = L.lean_table(df).set_index("side")
    # bought call: +10 * 1.20 * 100; mid ignored; sold put: -5 * 0.40 * 100
    assert table.loc["Calls", "net_premium"] == 10 * 1.20 * 100
    assert table.loc["Calls", "volume_ambiguous"] == 4
    assert table.loc["Puts", "net_premium"] == -5 * 0.40 * 100
    assert table.loc["Puts", "net_delta_shares"] == -1 * 5 * (-0.40) * 100
    assert table.loc["Total", "net_premium"] == (
        table.loc["Calls", "net_premium"] + table.loc["Puts", "net_premium"])
    # ambiguous volume is in the gross, not the net
    assert table.loc["Total", "gross_premium"] > table.loc["Total", "net_premium"]


def test_close_gate_follows_new_york_across_dst():
    winter = datetime(2026, 1, 2, 21, 45, tzinfo=UTC)       # Friday 16:45 EST
    summer = datetime(2026, 7, 2, 21, 45, tzinfo=UTC)       # Thursday 17:45 EDT
    early_summer = datetime(2026, 7, 2, 20, 40, tzinfo=UTC)  # 16:40 EDT, before the gate
    assert L.recording_decision(winter) == ("ok", date(2026, 1, 2))
    assert L.recording_decision(summer) == ("ok", date(2026, 7, 2))
    assert L.recording_decision(early_summer)[0] == "too_early"
    assert L.recording_decision(datetime(2026, 1, 2, 21, 44, tzinfo=UTC))[0] == "too_early"
    assert L.recording_decision(datetime(2026, 7, 3, 21, 45, tzinfo=UTC))[0] == "closed"
    assert L.recording_decision(datetime(2026, 11, 14, 21, 45, tzinfo=UTC))[0] == "closed"
    assert L.recording_decision(datetime(2026, 11, 26, 21, 45, tzinfo=UTC))[0] == "closed"
    # day after Thanksgiving is an early close, still a session
    assert L.recording_decision(datetime(2026, 11, 27, 21, 45, tzinfo=UTC))[0] == "ok"
    # Sunday 00:20 ET still belongs to Friday's session
    assert L.session_date_of(datetime(2026, 11, 15, 5, 20, tzinfo=UTC)) == date(2026, 11, 13)


def test_deribit_window_falls_back_when_the_public_tape_misses_the_session():
    start = datetime(2026, 11, 13, 14, 30, tzinfo=UTC)
    end = datetime(2026, 11, 13, 21, 15, tzinfo=UTC)
    now = datetime(2026, 11, 15, 23, 50, tzinfo=UTC)
    a, b, note = L.deribit_compare_window(start, end, now, True)
    assert (a, b) == (start, end) and note is None
    a, b, note = L.deribit_compare_window(start, end, now, False)
    assert b == now and (b - a).total_seconds() == 24 * 3600
    assert note and "last 24 hours" in note


def test_history_notice_collects_until_a_prior_close_exists():
    none = L.history_notice(None, "2026-11-16")
    assert none["ready"] is False and none["since"] is None
    assert none["text"].startswith("Collecting history")
    one = pd.DataFrame([_store_row("2026-11-16", "A", 1)])
    collecting = L.history_notice(one, "2026-11-16")
    assert collecting["ready"] is False
    assert "Collecting history since 2026-11-16" in collecting["text"]
    prior_only = pd.DataFrame([_store_row("2026-11-13", "A", 1)])
    ready = L.history_notice(prior_only, "2026-11-16")
    assert ready["ready"] is True and ready["prior"] == "2026-11-13"
    assert "versus the 2026-11-13 close" in ready["text"]
    day, rows = L.prior_chain(prior_only, "2026-11-16")
    assert day == "2026-11-13" and list(rows["option"]) == ["A"]
    assert L.prior_chain(one, "2026-11-16") == (None, None)


def test_chain_append_keeps_history_and_rejects_a_wipe():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        day1 = pd.DataFrame([
            _store_row("2026-11-12", "A", 10, delta=np.nan),
            _store_row("2026-11-12", "B", 4),
        ])
        day2 = pd.DataFrame([
            _store_row("2026-11-13", "A", 99),   # must not overwrite the 12th
            _store_row("2026-11-13", "C", 1),
        ])
        L.append_chain_snapshot(day1, base)
        L.append_chain_snapshot(day2, base)
        got = L.read_chain_store(base)
        keys = set(zip(got["session_date"], got["option"]))
        assert keys == {("2026-11-12", "A"), ("2026-11-12", "B"),
                        ("2026-11-13", "A"), ("2026-11-13", "C")}
        a = got[(got.session_date == "2026-11-12") & (got.option == "A")].iloc[0]
        assert a["oi"] == 10 and np.isnan(a["delta"])
        assert L.validate_chain_history(got, got) == []
        lost = got[got.session_date != "2026-11-12"]
        assert any("lost" in e for e in L.validate_chain_history(got, lost))
        tampered = got.copy()
        tampered.loc[tampered.option == "A", "oi"] = 999
        assert any("oi" in e for e in L.validate_chain_history(got, tampered))
        extra = pd.concat([got, pd.DataFrame([_store_row("2026-11-16", "A", 1),
                                              _store_row("2026-11-17", "A", 1)])],
                          ignore_index=True)
        assert any("new sessions" in e for e in L.validate_chain_history(got, extra))
        backfill = pd.concat([got, pd.DataFrame([_store_row("2026-11-10", "Z", 1)])],
                             ignore_index=True)
        assert any("older" in e for e in L.validate_chain_history(got, backfill))
        assert any("lost" in e or "empty" in e for e in L.validate_chain_history(got, got.iloc[0:0]))
        try:
            L.append_chain_snapshot(got.iloc[0:0], base)
            raise AssertionError("empty write should have been refused")
        except ValueError:
            pass
        assert len(L.read_chain_store(base)) == 4
        # a clean one-day append is publishable
        plus = pd.concat([got, pd.DataFrame([_store_row("2026-11-16", "A", 3)])], ignore_index=True)
        assert L.validate_chain_history(got, plus) == []


def test_record_does_not_wipe_history_or_surface_files():
    import tempfile
    calls = {"n": 0}

    def chain(oi, snap):
        df = _chain([
            ("A", "C", 50, oi, 10, 1.2, 1.0, 1.2, 0.4, "2026-12-18T21:00:00Z"),
        ])
        return df, {"snapshot_ny": pd.Timestamp(snap)}

    def fetch_fri():
        calls["n"] += 1
        return chain(10, "2026-11-13 16:50:00-05:00")

    def fetch_mon():
        calls["n"] += 1
        return chain(80, "2026-11-16 16:50:00-05:00")

    def boom():
        raise AssertionError("fetch should not run")

    friday = datetime(2026, 11, 13, 21, 50, tzinfo=UTC)   # 16:50 EST
    monday = datetime(2026, 11, 16, 21, 50, tzinfo=UTC)
    holiday = datetime(2026, 7, 3, 21, 50, tzinfo=UTC)
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        surface = base / "BTC_surface.parquet"
        surface.write_bytes(b"surface-history-must-survive")
        assert L.record_chain_snapshot(friday, fetch_fri, base) == "recorded"
        assert calls["n"] == 1
        first = L.read_chain_store(base)
        assert L.record_chain_snapshot(friday, boom, base) == "skipped"
        assert calls["n"] == 1
        pd.testing.assert_frame_equal(first, L.read_chain_store(base))
        assert L.record_chain_snapshot(holiday, boom, base) == "closed"
        assert L.record_chain_snapshot(datetime(2026, 11, 14, 21, 50, tzinfo=UTC), boom, base) == "closed"

        def empty():
            calls["n"] += 1
            df, meta = chain(80, "2026-11-16 16:50:00-05:00")
            return df.iloc[0:0], meta

        def bad():
            calls["n"] += 1
            raise RuntimeError("cboe down")

        def stale():
            calls["n"] += 1
            return chain(80, "2026-11-13 16:50:00-05:00")

        assert L.record_chain_snapshot(monday, empty, base) == "no_data"
        assert L.record_chain_snapshot(monday, bad, base) == "error"
        assert L.record_chain_snapshot(monday, stale, base) == "stale"
        assert len(L.read_chain_store(base)) == len(first)
        assert L.record_chain_snapshot(monday, fetch_mon, base) == "recorded"
        got = L.read_chain_store(base)
        assert set(got["session_date"]) == {"2026-11-13", "2026-11-16"}
        old_oi = got.loc[got.session_date == "2026-11-13", "oi"].iloc[0]
        assert old_oi == first["oi"].iloc[0] == 10
        assert surface.read_bytes() == b"surface-history-must-survive"
        # publishing the result would pass validation; dropping Friday would not
        assert L.validate_chain_history(first, got) == []
        assert any("lost" in e for e in L.validate_chain_history(got, got[got.session_date == "2026-11-16"]))


_HOLDINGS = """iShares Bitcoin Trust ETF
Fund Holdings as of,"Nov 13, 2026"
Inception Date,"Jan 05, 2024"
Shares Outstanding,"1,387,960,000.00"
Stock,"-"

Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Quantity,Market Currency,Accrual Date
"BTC","BITCOIN","-","Alternative","53,780,114,424.43","100.00","53,780,114,424.43","783,448.30480","BTC","-"
"USD","USD CASH","-","Cash","26,683.47","0.00","26,683.47","26,683.47000","USD","-"
"""


def test_ishares_ratio_and_deribit_equivalent_size():
    parsed = L.parse_ishares_holdings(_HOLDINGS)
    assert parsed["asof"] == "2026-11-13" and parsed["fallback"] is False
    assert abs(parsed["btc_per_share"] - (783_448.30480 / 1_387_960_000)) < 1e-15
    assert abs(L.IBIT_BTC_PER_SHARE_FALLBACK - parsed["btc_per_share"]) < 1e-15
    assert L.IBIT_BTC_PER_SHARE_ASOF == "2026-11-13"
    bps = parsed["btc_per_share"]
    # 1 contract = 100 shares. 1 Deribit BTC option = 1 BTC.
    assert abs(L.contracts_to_btc(1, bps) - 100 * bps) < 1e-15
    assert abs(L.ibit_btc_equivalent(250, bps) - 250 * bps) < 1e-12
    # strike in USD/share ÷ BTC-per-share = USD per BTC
    assert abs(L.strike_btc_equivalent(38.94, bps) - 38.94 / bps) < 1e-6
    assert np.isnan(L.strike_btc_equivalent(40, 0))
    # sqrt size, floored and capped, and the BTC divisor keeps the same pixels
    assert abs(float(L.bubble_marker_size(1_000)) - np.sqrt(1_000) / 4) < 1e-12
    assert float(L.bubble_marker_size(0)) == L.BUBBLE_SIZE_MIN
    assert float(L.bubble_marker_size(1e12)) == L.BUBBLE_SIZE_MAX
    for n in L.BUBBLE_KEY_CONTRACTS:
        same = float(L.bubble_marker_size(L.contracts_to_btc(n, bps), L.bubble_size_divisor("btc", bps)))
        assert abs(same - float(L.bubble_marker_size(n))) < 1e-9
    key = L.bubble_size_key(bps)
    assert [k["contracts"] for k in key] == [1_000, 5_000, 20_000]
    assert key[0]["size"] < key[1]["size"] < key[2]["size"]
    try:
        L.parse_ishares_holdings("not a holdings file")
        raise AssertionError("bad file should fail")
    except ValueError:
        pass


def test_ratio_falls_back_when_ishares_cannot_be_fetched():
    from unittest.mock import patch
    L.clear_cache()
    with patch("lib.ibit.requests.get", side_effect=RuntimeError("down")):
        got = L.btc_per_share_ratio()
    assert got["fallback"] is True and got["asof"] == "2026-11-13"
    assert abs(got["btc_per_share"] - L.IBIT_BTC_PER_SHARE_FALLBACK) < 1e-15
    L.clear_cache()

    class _Resp:
        status_code = 200
        text = _HOLDINGS

    with patch("lib.ibit.requests.get", return_value=_Resp()):
        live = L.btc_per_share_ratio()
    assert live["fallback"] is False and abs(live["btc_per_share"] - L.IBIT_BTC_PER_SHARE_FALLBACK) < 1e-15
    L.clear_cache()


def test_snapshot_rows_come_from_the_parsed_chain():
    payload = {"timestamp": "2026-11-13 16:50:00",
               "data": {"current_price": 50.0, "options": [
                   dict(option="IBIT261218C00050000", bid=1.0, ask=1.2, iv=0.4,
                        open_interest=12, volume=3, last_trade_price=1.2,
                        bid_size=10, ask_size=8)]}}
    df, _ = L.parse_cboe(payload, datetime(2026, 11, 13, 21, 50, tzinfo=UTC))
    assert df["option"].iloc[0] == "IBIT261218C00050000"
    assert df["last"].iloc[0] == 1.2
    rows = L.snapshot_rows_from_chain(df, date(2026, 11, 13),
                                     datetime(2026, 11, 13, 21, 50, tzinfo=UTC))
    assert len(rows) == 1 and rows["oi"].iloc[0] == 12 and rows["session_date"].iloc[0] == "2026-11-13"
    assert rows["option"].iloc[0] == "IBIT261218C00050000"
