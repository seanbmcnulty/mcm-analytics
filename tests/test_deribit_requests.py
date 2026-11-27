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


class _ErrResp:
    def __init__(self, code, text="", headers=None, body=None):
        self.status_code, self.text, self.headers, self._body = code, text, headers or {}, body

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


def test_describe_response_names_cloudflare_block_and_jsonrpc_error():
    r = _ErrResp(403, "<html><title>Access denied | deribit.com used Cloudflare</title>Error 1020</html>",
                 {"server": "cloudflare"})
    assert deribit._describe_response(r) == "HTTP 403 · Cloudflare access denied"
    r = _ErrResp(400, body={"error": {"message": "Method not found", "code": -32601}})
    assert deribit._describe_response(r) == "HTTP 400 · Method not found (code -32601)"


def test_session_sends_descriptive_user_agent():
    s = deribit._session()
    assert "mcm-analytics" in s.headers["User-Agent"]


def test_breaker_fails_fast_after_network_errors(monkeypatch):
    import requests
    calls = {"n": 0}

    class _Dead:
        def get(self, url, params=None, timeout=None):
            calls["n"] += 1
            raise requests.ConnectTimeout("timed out")

    monkeypatch.setattr(deribit, "_session", lambda: _Dead())
    monkeypatch.setattr(deribit, "RETRY_BACKOFF_S", (0.0, 0.0, 0.0))
    deribit.clear_cache()
    assert deribit._request("get_index_price", {"index_name": "zz_a"}, ttl=30) is None
    assert calls["n"] == 3
    st = deribit.last_status()
    assert st["ok"] is False and st["breaker_open"] and "ConnectTimeout" in st["reason"]
    t0 = time.time()
    assert deribit._request("get_index_price", {"index_name": "zz_b"}, ttl=30) is None
    assert calls["n"] == 3 and time.time() - t0 < 0.5  # no new network call
    deribit.clear_cache()  # refresh closes the breaker
    assert not deribit.last_status()["breaker_open"]


def test_probe_reports_http_status(monkeypatch):
    class _Blocked:
        def get(self, url, params=None, timeout=None):
            return _ErrResp(403, "<title>Attention Required! | Cloudflare</title>", {"server": "cloudflare"})

    monkeypatch.setattr(deribit, "_session", lambda: _Blocked())
    ok, reason, px = deribit.probe()
    assert (ok, px) == (False, None)
    assert reason == "HTTP 403 · Cloudflare bot check"
