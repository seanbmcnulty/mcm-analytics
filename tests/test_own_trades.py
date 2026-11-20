"""Offline tests for lib.own_trades (OKX / Paradex own block trades, mocked HTTP) and the
won-vs-market helpers in lib.block_flow.  No network, no real credentials."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from lib import block_flow, own_trades

T0 = 1_795_000_000_000          # ms
OKX_CREDS = ("k-test", "s-test", "p-test")
PDX_TOKEN = "a.b.c"


class _Resp:
    def __init__(self, j, status=200):
        self._j, self.status_code = j, status

    def json(self):
        return self._j


class _FakeSession:
    def __init__(self, router):
        self.router, self.calls = router, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return self.router(url, params or {})


def _with_session(router, fn):
    old = own_trades._session
    own_trades._session = fake = _FakeSession(router)
    old_gap = dict(own_trades._OKX_GAP)
    own_trades._OKX_GAP.clear()
    try:
        return fn(), fake
    finally:
        own_trades._session = old
        own_trades._OKX_GAP.update(old_gap)


def _okx_router(url, params):
    if "/rfq/trades" in url:
        return _Resp({"code": "0", "data": [
            {"blockTdId": "B1", "cTime": str(T0 + 1000), "mTraderCode": "ME", "tTraderCode": "X1", "isSuccessful": True},
            {"blockTdId": "B2", "cTime": str(T0 + 2000), "mTraderCode": "ME", "tTraderCode": "X2", "isSuccessful": True},
            {"blockTdId": "B3", "cTime": str(T0 + 3000), "mTraderCode": "MM", "tTraderCode": "ME", "isSuccessful": True},
            {"blockTdId": "B4", "cTime": str(T0 + 4000), "mTraderCode": "ME", "tTraderCode": "X3", "isSuccessful": False},
        ]})
    if "/rfq/rfqs" in url:
        return _Resp({"code": "0", "data": [
            {"rfqId": "r1", "cTime": str(T0 + 10), "state": "filled", "traderCode": "X1"},
            {"rfqId": "r2", "cTime": str(T0 + 20), "state": "traded_away", "traderCode": "X4"},
            {"rfqId": "r3", "cTime": str(T0 + 30), "state": "traded_away", "traderCode": ""},
            {"rfqId": "r4", "cTime": str(T0 + 40), "state": "expired", "traderCode": "X5"},
            {"rfqId": "r5", "cTime": str(T0 + 50), "state": "filled", "traderCode": "ME"},
            {"rfqId": "r0", "cTime": str(T0 - 99_000), "state": "filled", "traderCode": "X1"},  # before window
        ]})
    raise AssertionError(url)


def test_okx_own_roles_and_win_rate():
    out, fake = _with_session(_okx_router, lambda: own_trades.okx_fetch_own(T0, T0 + 60_000, creds=OKX_CREDS))
    assert out["configured"] and out["error"] is None
    assert out["own"] == {"B1": "maker", "B2": "maker", "B3": "taker"}      # failed B4 dropped
    assert out["n_won"] == 2 and out["n_blocks"] == 3
    assert out["rfq"] == {"won": 1, "lost": 2, "no_trade": 1, "own_rfqs": 1, "total": 4}
    # read-only: only GETs to the two RFQ history endpoints, signed
    assert all(("/rfq/trades" in u or "/rfq/rfqs" in u) for u, _, _ in fake.calls)
    h = fake.calls[0][2]
    assert h["OK-ACCESS-KEY"] == "k-test" and h["OK-ACCESS-SIGN"] and h["OK-ACCESS-TIMESTAMP"].endswith("Z")


def test_okx_api_error_is_reported_without_secrets():
    def router(url, params):
        return _Resp({"code": "50113", "msg": "Invalid Sign"})
    out, _ = _with_session(router, lambda: own_trades.okx_fetch_own(T0, T0 + 1, creds=OKX_CREDS))
    assert out["configured"] and "50113" in out["error"]
    assert not any(s in out["error"] for s in OKX_CREDS)
    assert out["own"] == {}


def _pdx_router(url, params):
    if url.endswith("/account"):
        return _Resp({"account": "0xME"})
    if url.endswith("/block-trades"):
        if params.get("cursor") == "p2":
            return _Resp({"results": [
                {"block_id": "P9", "status": "COMPLETED", "created_at": T0 - 500_000, "trades": {}},
            ], "next": None})
        return _Resp({"results": [
            {"block_id": "P1", "status": "COMPLETED", "created_at": T0 + 3000,
             "trades": {"BTC-USD-27NOV26-90000-C": {"maker_account": "0xME", "taker_account": "0xT"}}},
            {"block_id": "P2", "status": "COMPLETED", "created_at": T0 + 2000,
             "trades": {"x": {"maker_account": "0xM", "taker_account": "0xME"}}},
            {"block_id": "P3", "status": "FAILED", "created_at": T0 + 1000,
             "trades": {"x": {"maker_account": "0xME", "taker_account": "0xT"}}},
        ], "next": "p2"})
    raise AssertionError(url)


def test_paradex_own_roles_window_and_paging():
    out, fake = _with_session(_pdx_router, lambda: own_trades.paradex_fetch_own(T0, T0 + 60_000, token=PDX_TOKEN))
    assert out["error"] is None
    assert out["own"] == {"P1": "maker", "P2": "taker"}    # failed + out-of-window dropped
    assert out["n_won"] == 1 and out["rfq"] is None
    assert any(p.get("cursor") == "p2" for _, p, _ in fake.calls)
    assert all(h.get("Authorization") == "Bearer a.b.c" for _, _, h in fake.calls)


def test_missing_credentials_degrade_to_public_only():
    import os
    saved = {k: os.environ.pop(k, None) for k in ("OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE",
                                                  "PARADEX_API_KEY")}
    try:
        assert own_trades.okx_credentials() is None and own_trades.paradex_credentials() is None
        assert own_trades.okx_fetch_own(T0, T0 + 1)["configured"] is False
        assert own_trades.paradex_fetch_own(T0, T0 + 1)["configured"] is False
        os.environ["PARADEX_API_KEY"] = "YOUR_PARADEX_READONLY_TOKEN"     # placeholder is not a key
        assert own_trades.paradex_credentials() is None
        os.environ["PARADEX_API_KEY"] = "x.y.z"
        assert own_trades.paradex_credentials() == "x.y.z"
    finally:
        os.environ.pop("PARADEX_API_KEY", None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _frame():
    ts = pd.Timestamp("2026-11-20 10:00", tz="Asia/Singapore")
    df = pd.DataFrame({
        "trade_id": ["t1", "t2", "t3", "t4"],
        "timestamp": [ts, ts, ts + pd.Timedelta(minutes=5), ts + pd.Timedelta(minutes=9)],
        "instrument_name": ["BTC-USD-27NOV26-90000-C"] * 4, "strike": [90000.0] * 4,
        "expiry": [pd.Timestamp("2026-11-27")] * 3 + [pd.Timestamp("2027-03-26")],
        "option_type": ["C"] * 4, "direction": ["buy", "sell", "buy", "sell"],
        "abs_amount": [1.0, 1.0, 2.0, 5.0], "price": [1000.0, 1000.0, 1100.0, 3000.0],
        "index_price": [90000.0] * 4, "mark_price": [float("nan")] * 4, "iv": [float("nan")] * 4,
        "block_id": ["B1", "B1", "B2", "B9"], "is_block": [True] * 4,
    })
    df["amount"] = df["abs_amount"].where(df["direction"] == "buy", -df["abs_amount"])
    return block_flow.enrich(df, 90000.0, "BTC", False)


def test_tag_own_summary_packages_and_chart():
    d = block_flow.tag_own(_frame(), {"B1": "maker", "B2": "taker"})
    assert d["ours"].tolist() == [True, True, True, False]
    assert d["won"].tolist() == [True, True, False, False]
    s = block_flow.own_summary({"BTC": d})
    assert (s["blocks"], s["won"], s["ours"]) == (3, 1, 2)
    assert abs(s["premium_won"] - 2000.0) < 1e-6 and 0 < s["premium_share"] < 1
    pk = block_flow.build_packages(d)
    assert sorted(pk["Won"].tolist()) == ["", "ours (taker)", "🏆 won"]
    assert "Won" not in block_flow.build_packages(_frame()).columns      # public-only: no column
    fig = block_flow.plot_won_vs_market(d, "BTC", "OKX")
    assert fig.layout.title.text == "BTC OKX Blocks Won vs Market (1/3 blocks, 10% of premium)"
    sc = block_flow.add_won_overlay(block_flow.plot_scatter(d, None, None, "BTC", "Index"), d)
    assert sc.data[-1].name == "Won by us"
    assert block_flow.plot_won_vs_market(_frame(), "BTC", "OKX").layout.title.text.endswith("No Data")


def test_venues_expose_fetch_own():
    from lib import okx, paradex, deribit  # noqa: F401
    assert callable(okx.VENUE.fetch_own) and callable(paradex.VENUE.fetch_own)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
