"""Cache-key bucketing for windowed Deribit endpoints (no network)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import deribit  # noqa: E402


def test_window_bucketed_to_ttl():
    p = {"instrument_name": "BTC-PERPETUAL", "resolution": "60",
         "start_timestamp": 1_000_123_456, "end_timestamp": 1_000_999_999}
    out = deribit._bucket_window("get_tradingview_chart_data", p, 300)
    assert out["start_timestamp"] % 300_000 == 0
    assert out["end_timestamp"] % 300_000 == 0
    assert out["start_timestamp"] <= p["start_timestamp"]
    assert out["end_timestamp"] >= p["end_timestamp"]
    assert p["end_timestamp"] == 1_000_999_999  # input not mutated


def test_nearby_calls_share_key():
    base = {"instrument_name": "BTC-PERPETUAL", "resolution": "1D"}
    a = deribit._bucket_window("get_tradingview_chart_data",
                               {**base, "start_timestamp": 10_000_001, "end_timestamp": 90_000_001}, 600)
    b = deribit._bucket_window("get_tradingview_chart_data",
                               {**base, "start_timestamp": 10_000_900, "end_timestamp": 90_000_900}, 600)
    assert a == b


def test_minimum_bucket_and_other_endpoints_untouched():
    p = {"start_timestamp": 61_000, "end_timestamp": 119_000}
    out = deribit._bucket_window("get_volatility_index_data", p, 5)
    assert out["start_timestamp"] == 60_000 and out["end_timestamp"] == 120_000
    q = {"instrument_name": "X", "start_timestamp": 61_000, "end_timestamp": 119_000}
    assert deribit._bucket_window("get_last_trades_by_instrument_and_time", q, 300) == q
