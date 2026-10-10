"""Offline tests for lib.block_flow / lib.derive / lib.paradex parsing + math."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm

from lib import block_flow, derive, paradex


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
