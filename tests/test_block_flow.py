"""Offline tests for lib.block_flow / lib.derive / lib.paradex / lib.okx / lib.bybit / lib.bullish parsing + math."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm

from lib import binance, block_flow, bullish, bybit, derive, okx, paradex


def _bs(S, K, T, sig, call):
    d1 = (np.log(S / K) + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T))
    c = S * norm.cdf(d1) - K * norm.cdf(d1 - sig * np.sqrt(T))
    return c if call else c - S + K


def test_implied_vol_round_trip():
    S, K, T = 80000.0, np.array([70000.0, 80000.0, 95000.0]), np.full(3, 30 / 365.25)
    for call in (True, False):
        px = _bs(S, K, T, 0.55, call)
        iv = block_flow.implied_vol(px, S, K, T, call)
        assert np.allclose(iv, 0.55, atol=1e-4)


def test_implied_vol_rejects_arbitrage_prices():
    iv = block_flow.implied_vol([0.0, 90000.0], 80000.0, 80000.0, 0.1, True)
    assert np.isnan(iv).all()


def test_parse_derive_and_paradex_names():
    base, exp, strike, cp = derive.parse_option("BTC-20261010-86000-C")
    assert (base, strike, cp) == ("BTC", 86000.0, "C") and exp == pd.Timestamp("2026-10-10")
    base, exp, strike, cp = paradex.parse_option("ETH-USD-23OCT26-2200-P")
    assert (base, strike, cp) == ("ETH", 2200.0, "P") and exp == pd.Timestamp("2026-10-23")
    assert paradex.parse_option("BTC-USD-PERP")[3] is None
    assert paradex.parse_option("NEAR-USD-25DEC26-6_8-P")[2] == 6.8


def test_derive_normalize_keeps_taker_row_and_flags_rfq():
    rows = [
        {"trade_id": "a", "instrument_name": "BTC-20261127-95000-C", "timestamp": 1791000000000,
         "trade_price": "1684", "trade_amount": "4", "mark_price": "1700", "index_price": "85000",
         "direction": "sell", "liquidity_role": "maker", "quote_id": "q", "rfq_id": "r"},
        {"trade_id": "a", "instrument_name": "BTC-20261127-95000-C", "timestamp": 1791000000000,
         "trade_price": "1684", "trade_amount": "4", "mark_price": "1700", "index_price": "85000",
         "direction": "buy", "liquidity_role": "taker", "quote_id": "q", "rfq_id": "r"},
        {"trade_id": "b", "instrument_name": "BTC-20261127-90000-P", "timestamp": 1791000001000,
         "trade_price": "10", "trade_amount": "1", "mark_price": "10", "index_price": "85000",
         "direction": "buy", "liquidity_role": "taker", "quote_id": None, "rfq_id": None},
    ]
    df = derive._normalize(rows, "BTC")
    assert len(df) == 2
    a = df[df.trade_id == "a"].iloc[0]
    assert a.direction == "buy" and a.amount == 4 and a.is_block and a.block_id == "r"
    assert not df[df.trade_id == "b"].iloc[0].is_block


def test_enrich_signs_and_premium():
    rows = [
        {"trade_id": "a", "instrument_name": "BTC-20261127-95000-C", "timestamp": 1791000000000,
         "trade_price": "1684", "trade_amount": "4", "mark_price": "1700", "index_price": "85000",
         "direction": "buy", "liquidity_role": "taker", "quote_id": "q", "rfq_id": "r"},
        {"trade_id": "b", "instrument_name": "BTC-20261127-95000-C", "timestamp": 1791000001000,
         "trade_price": "1684", "trade_amount": "4", "mark_price": "1700", "index_price": "85000",
         "direction": "sell", "liquidity_role": "taker", "quote_id": "q", "rfq_id": "r2"},
    ]
    e = block_flow.enrich(derive._normalize(rows, "BTC"), 85000.0, "BTC", True)
    buy, sell = e[e.direction == "buy"].iloc[0], e[e.direction == "sell"].iloc[0]
    assert buy.premium_usd == 4 * 1684 and sell.premium_usd == -4 * 1684
    assert buy.dollar_delta > 0 > sell.dollar_delta
    assert buy.dollar_vega > 0 > sell.dollar_vega


def test_derive_sub_dollar_strikes_use_underscore_decimals():
    assert derive.parse_option("PUMP-20261225-0_007-C")[2] == 0.007
    assert derive.parse_option("LIT-20261023-3_4-P")[2] == 3.4


def test_flow_summary_normalises_deribit_usd_premium():
    from lib import flow_summary
    rows = [{"trade_id": "1", "instrument_name": "BTC-26MAR27-80000-C", "timestamp": 1791000000000,
             "amount": 20.0, "direction": "buy", "price": 0.05, "mark_price": 0.05, "index_price": 80000.0, "iv": 50.0},
            {"trade_id": "2", "instrument_name": "BTC-26MAR27-80000-C", "timestamp": 1791000001000,
             "amount": 1.0, "direction": "buy", "price": 0.05, "mark_price": 0.05, "index_price": 80000.0, "iv": 50.0}]
    df = flow_summary._normalize_deribit(rows, "BTC", 12.5)   # second trade is below the block threshold
    assert len(df) == 1 and df.iloc[0].price == 0.05 * 80000.0 and df.iloc[0].abs_amount == 20.0


# --- OKX -------------------------------------------------------------------
def test_okx_parse_option():
    base, exp, strike, cp = okx.parse_option("BTC-USD-261106-89000-C")
    assert (base, strike, cp) == ("BTC", 89000.0, "C") and exp == pd.Timestamp("2026-11-06")
    assert okx.parse_option("BTC-USD-SWAP")[3] is None
    assert okx.parse_option("BTC-USDT")[3] is None


def test_okx_normalize_converts_contracts_and_coin_prices_to_usd():
    tape = [{"blockTdId": "9", "groupId": "", "strategy": "CALL_SPREAD", "inverse": False, "cTime": "1792557758577",
             "legs": [{"instId": "BTC-USD-261127-76000-P", "side": "sell", "sz": "300", "px": "0.0156", "tradeId": "t1"},
                      {"instId": "BTC-USD-261127-70000-P", "side": "buy", "sz": "300", "px": "0.0050", "tradeId": "t2"},
                      {"instId": "BTC-USD-SWAP", "side": "buy", "sz": "10", "px": "82000", "tradeId": "t3"}]}]
    lookup = {"t1": {"idxPx": "80000", "markPx": "0.0150", "fillVol": "0.40"}}   # t2 missed the join
    k = pd.DataFrame({"ts_ms": [1792557000000], "open": [79000.0], "close": [79500.0]})
    df = okx._normalize(tape, "BTC", {"BTC-USD": (0.01, False)}, lookup, k, 0.0)
    assert len(df) == 2                                    # swap leg dropped
    a = df[df.trade_id == "t1"].iloc[0]
    assert a.direction == "sell" and a.amount == -3.0 and a.abs_amount == 3.0   # 300 contracts * 0.01 BTC
    assert abs(a.price - 0.0156 * 80000) < 1e-9 and abs(a.mark_price - 0.015 * 80000) < 1e-9
    assert abs(a.iv - 40.0) < 1e-9 and a.block_id == "9" and a.is_block
    b = df[df.trade_id == "t2"].iloc[0]
    assert b.index_price == 79000.0 and abs(b.price - 0.005 * 79000.0) < 1e-9 and np.isnan(b.iv)   # candle fallback
    e = block_flow.enrich(df, 80000.0, "BTC", True)
    assert abs(e[e.trade_id == "t1"].iloc[0].premium_usd - (-3.0 * 0.0156 * 80000)) < 1e-6


def test_okx_um_family_is_linear_usd_with_its_own_contract_size():
    assert okx.parse_option("ETH-USD_UM-261030-2200-P")[:1] == ("ETH",) and okx.family("ETH-USD_UM-261030-2200-P") == "ETH-USD_UM"
    assert okx.family("ETH-USD-261030-2200-P") == "ETH-USD"
    tape = [{"blockTdId": "7", "groupId": "", "cTime": "1792562124420", "legs": [
        {"instId": "ETH-USD_UM-261030-2200-P", "side": "buy", "sz": "1999", "px": "7.6", "tradeId": "u1"},
        {"instId": "ETH-USD-261030-2200-P", "side": "buy", "sz": "10", "px": "0.003", "tradeId": "u2"}]}]
    lookup = {"u1": {"idxPx": "2450", "markPx": "7.26", "fillVol": "0.42"}, "u2": {"idxPx": "2450"}}
    specs = {"ETH-USD_UM": (0.01, True), "ETH-USD": (0.1, False)}
    df = okx._normalize(tape, "ETH", specs, lookup, pd.DataFrame(columns=["ts_ms", "open", "close"]), 0.0)
    um, inv = df[df.trade_id == "u1"].iloc[0], df[df.trade_id == "u2"].iloc[0]
    assert abs(um.abs_amount - 19.99) < 1e-9 and um.price == 7.6 and um.mark_price == 7.26      # linear: already USD
    assert abs(inv.abs_amount - 1.0) < 1e-9 and abs(inv.price - 0.003 * 2450) < 1e-9             # inverse: coin * index


def test_default_min_sizes_are_each_exchange_minimum_order():
    assert okx.DEFAULT_MIN_SIZES == {"BTC": 0.01, "ETH": 0.01, "SOL": 0.1, "XAU": 0.01}
    assert bybit.DEFAULT_MIN_SIZES["BTC"] == 0.01 and bybit.DEFAULT_MIN_SIZES["ETH"] == 0.1
    assert bullish.DEFAULT_MIN_SIZES == {"BTC": 0.0005}
    assert all(v == 0.01 for v in binance.DEFAULT_MIN_SIZES.values()) and "BTC" in binance.DEFAULT_MIN_SIZES


def test_okx_feed_status_reports_hedge_legs_and_join_gaps():
    txt = okx.feed_status({"tape_rows": 5, "option_legs": 7, "non_option_legs": 2, "joined_legs": 6, "total_legs": 7})
    assert "2 futures/swap/spot hedge legs" in txt and "6/7" in txt
    assert "no block trades" in okx.feed_status({"tape_rows": 0})


# --- Bybit -----------------------------------------------------------------
def test_bybit_parse_option_and_normalize():
    base, exp, strike, cp = bybit.parse_option("BTC-22OCT26-82500-C-USDT")
    assert (base, strike, cp) == ("BTC", 82500.0, "C") and exp == pd.Timestamp("2026-10-22")
    assert bybit.parse_option("XRP-30OCT26-1.35-C-USDT")[2] == 1.35
    rows = [{"execId": "a", "symbol": "XRP-30OCT26-1.4-C-USDT", "price": "0.0267", "size": "10000", "side": "Buy",
             "time": "1792508385302", "isBlockTrade": True, "mP": "0.02676897", "iP": "1.34434622",
             "mIv": "0.5518", "iv": "0.5507"},
            {"execId": "b", "symbol": "XRP-30OCT26-1.35-C-USDT", "price": "0.044", "size": "10000", "side": "Sell",
             "time": "1792508385302", "isBlockTrade": True, "mP": "0.0435", "iP": "1.34434622",
             "mIv": "0.52", "iv": "0"}]
    df = bybit._normalize(rows, "XRP")
    assert len(df) == 2 and df.block_id.nunique() == 1                       # same-ms legs form one package
    assert df[df.trade_id == "a"].iloc[0].amount == 10000 and df[df.trade_id == "b"].iloc[0].amount == -10000
    assert abs(df[df.trade_id == "a"].iloc[0].iv - 55.07) < 1e-9 and np.isnan(df[df.trade_id == "b"].iloc[0].iv)


def test_bybit_store_keeps_only_blocks_and_feed_status_explains_horizon():
    bybit._STORE.clear(); bybit._COVERAGE.clear()
    now = 1792567000000
    rows = [{"execId": "x", "symbol": "BTC-22OCT26-82500-C-USDT", "time": str(now - 1000), "isBlockTrade": True},
            {"execId": "y", "symbol": "BTC-22OCT26-82500-C-USDT", "time": str(now - 2000), "isBlockTrade": False}]
    bybit._store_blocks("BTC", rows, now)
    assert list(bybit._STORE["BTC"]) == ["x"] and bybit._COVERAGE["BTC"]["oldest_ms"] == now - 2000
    meta = {"coverage": bybit._COVERAGE, "start_ms": now - 24 * 3600_000, "end_ms": now, "core": ["BTC"]}
    assert "1,000" in bybit.feed_status(meta)
    meta["start_ms"] = now - 1500
    assert bybit.feed_status(meta) is None
    bybit._STORE.clear(); bybit._COVERAGE.clear()


# --- Bullish ---------------------------------------------------------------
def test_bullish_parse_and_normalize_flags_blocks_by_size():
    base, exp, strike, cp = bullish.parse_option("BTC-USDC-20261021-76000-C")
    assert (base, strike, cp) == ("BTC", 76000.0, "C") and exp == pd.Timestamp("2026-10-21")
    rows = [{"symbol": "BTC-USDC-20261030-85000-C", "tradeId": "1", "price": "880.0000", "quantity": "3.0",
             "side": "SELL", "isTaker": True, "createdAtTimestamp": "1792566822888"},
            {"symbol": "BTC-USDC-20261030-85000-C", "tradeId": "2", "price": "880.0000", "quantity": "0.5",
             "side": "BUY", "isTaker": True, "createdAtTimestamp": "1792566822999"}]
    k = pd.DataFrame({"ts_ms": [1792566600000], "open": [82700.0], "close": [82710.0]})
    df = bullish._normalize(rows, "BTC", k)
    assert df.is_block.all()                     # default min size is Bullish's 0.0005 BTC minimum order: every print
    assert df.iloc[0].index_price == 82700.0 and df[df.trade_id == "1"].iloc[0].amount == -3.0


def test_flow_summary_applies_default_min_sizes_to_bullish():
    from lib import flow_summary
    rows = [{"symbol": "BTC-USDC-20261030-85000-C", "tradeId": str(i), "price": "880", "quantity": q, "side": "BUY",
             "isTaker": True, "createdAtTimestamp": str(1792566822888 + i)} for i, q in enumerate(["3.0", "0.5"])]
    orig = bullish.fetch_all
    bullish.VENUE.fetch_all = staticmethod(lambda s, e, a: {"frames": {"BTC": bullish._normalize(rows, "BTC", pd.DataFrame())},
                                                            "meta": {"listed": ["BTC"]}})
    bullish.VENUE.default_min_sizes = {"BTC": 1.0}
    try:
        p = flow_summary._venue_blocks(bullish.VENUE, 0, 1)
    finally:
        bullish.VENUE.fetch_all = staticmethod(orig)
        del bullish.VENUE.default_min_sizes
    assert len(p["frames"]["BTC"]) == 1 and p["frames"]["BTC"].iloc[0].abs_amount == 3.0


# --- Binance ---------------------------------------------------------------
def test_binance_parse_and_normalize():
    base, exp, strike, cp = binance.parse_option("BTC-261225-85000-C")
    assert (base, strike, cp) == ("BTC", 85000.0, "C") and exp == pd.Timestamp("2026-12-25")
    rows = [{"id": 1, "tradeId": 709, "symbol": "ETH-261030-2400-C", "price": "109.6", "qty": "0.5", "quoteQty": "54.8",
             "side": -1, "time": 1792549024832},
            {"id": 2, "tradeId": 197, "symbol": "ETH-261030-2425-C", "price": "39.2", "qty": "0.5", "quoteQty": "19.6",
             "side": 1, "time": 1792549024832},
            {"id": 3, "tradeId": 5, "symbol": "BTC-261022-82000-C", "price": "570", "qty": "2", "quoteQty": "1140",
             "side": -1, "time": 1792549024999}]
    k = pd.DataFrame({"ts_ms": [1792548000000], "open": [2400.0], "close": [2410.0]})
    df = binance._normalize(rows, "ETH", k)
    assert len(df) == 2 and df.block_id.nunique() == 1                      # same-ms legs form one package
    s, b = df[df.trade_id == "1"].iloc[0], df[df.trade_id == "2"].iloc[0]
    assert s.amount == -0.5 and b.amount == 0.5 and s.index_price == 2400.0 and np.isnan(s.iv)
    e = block_flow.enrich(df, 2400.0, "ETH", False)
    assert abs(e[e.trade_id == "1"].iloc[0].premium_usd - (-0.5 * 109.6)) < 1e-9


def test_binance_geoblock_raises_clear_error():
    class R:
        status_code = 451
        text = "Service unavailable from a restricted location"
        def json(self): raise ValueError
    orig = binance._session.get
    binance._session.get = lambda *a, **k: R()
    try:
        binance.fetch_block_tape(0, 1)
        raise AssertionError("expected PermissionError")
    except PermissionError as e:
        assert "restricted" in str(e).lower() and "BINANCE_EAPI_BASE" in str(e)
    finally:
        binance._session.get = orig
