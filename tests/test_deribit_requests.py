"""Deribit client: ticker endpoint name, single-flight, funding fetch count (no network)."""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import deribit  # noqa: E402


def test_get_ticker_uses_public_ticker_endpoint(monkeypatch):
    seen = {}

    def fake_request(endpoint, params=None, ttl=60.0, retries=3):
        seen["endpoint"] = endpoint
        return {"mark_price": 1.0}

    monkeypatch.setattr(deribit, "_request", fake_request)
    assert deribit.get_ticker("BTC-PERPETUAL") == {"mark_price": 1.0}
    assert seen["endpoint"] == "ticker"  # "get_ticker" does not exist (HTTP 400)


class _Resp:
    status_code = 200

    def json(self):
        return {"result": {"ok": True}}


class _Session:
    def __init__(self, counter):
        self.counter = counter

    def get(self, url, params=None, timeout=None):
        with self.counter["lock"]:
            self.counter["n"] += 1
        time.sleep(0.2)
        return _Resp()


def test_concurrent_identical_requests_hit_network_once(monkeypatch):
    counter = {"n": 0, "lock": threading.Lock()}
    monkeypatch.setattr(deribit, "_session", lambda: _Session(counter))
    deribit.clear_cache()
    out = []
    threads = [threading.Thread(target=lambda: out.append(
        deribit._request("get_index_price", {"index_name": "zz_test"}, ttl=30)))
        for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert counter["n"] == 1
    assert out == [{"ok": True}] * 4
    deribit.clear_cache()


def test_clear_cache_bumps_generation():
    g = deribit.cache_generation()
    deribit.clear_cache()
    assert deribit.cache_generation() == g + 1


def test_funding_plot_makes_one_history_call(monkeypatch):
    import pandas as pd
    from lib import cmd_market

    calls = []

    def fake_hist(instrument_name, start_ms=None, end_ms=None):
        calls.append((start_ms, end_ms))
        ts = pd.date_range(end=pd.Timestamp.now(tz="UTC").tz_localize(None),
                           periods=24 * 30, freq="h")
        return pd.DataFrame({"timestamp": ts, "interest_1h": 1e-6})

    monkeypatch.setattr(cmd_market.deribit, "get_funding_history", fake_hist)
    fig, df, text = cmd_market.cmd_plot_funding_rates("BTC")
    assert fig is not None and text is None
    assert len(calls) == 1
