"""Delayed snapshot fallback when live Deribit is unreachable (no network)."""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from lib import commands, deribit, history, snapfallback, surface  # noqa: E402

NOW = datetime(2026, 11, 27, 12, 30, tzinfo=timezone.utc)
TS = pd.Timestamp("2026-11-27T12:15:00+00:00")


@pytest.fixture(autouse=True)
def _reset():
    snapfallback.reset_cache()
    surface.clear_cache()
    yield
    deribit._set_status(True, "test reset")
    deribit.reset_breaker()
    snapfallback.reset_cache()
    surface.clear_cache()


def _expiry_frame():
    return pd.DataFrame({
        "timestamp": [TS - pd.Timedelta(minutes=15)] * 1 + [TS] * 3,
        "expiry": [pd.Timestamp("2026-11-28").date(), pd.Timestamp("2026-11-28").date(),
                   pd.Timestamp("2026-12-04").date(), pd.Timestamp("2026-11-20").date()],
        "atm": [0.40, 0.46, 0.45, 0.50], "call10": [0.5, 0.53, 0.45, 0.5],
        "call25": [0.4, 0.47, 0.44, 0.5], "put10": [0.6, 0.52, 0.56, 0.5],
        "put25": [0.5, 0.47, 0.49, 0.5],
    })


def test_latest_expiry_vols_uses_latest_row_and_drops_expired(monkeypatch):
    monkeypatch.setattr(history, "load_expiry_snapshots", lambda a: _expiry_frame())
    vols, ts = snapfallback.latest_expiry_vols("BTC", now=NOW)
    assert ts == TS
    assert vols["atm"] == {1: 0.46, 7: 0.45}       # 20-Nov expired, older row ignored
    assert vols["call35"] == {} and vols["put35"] == {}  # not recorded per expiry: blank


def test_latest_expiry_vols_refuses_stale_snapshots(monkeypatch):
    monkeypatch.setattr(history, "load_expiry_snapshots", lambda a: _expiry_frame())
    later = datetime(2026, 11, 28, 0, 0, tzinfo=timezone.utc)
    assert snapfallback.latest_expiry_vols("BTC", now=later) == ({}, None)


class _Resp:
    def __init__(self, text, code=200):
        self.text, self.status_code = text, code


def test_latest_market_parses_csv():
    csv = ("timestamp,BTC_index,BTC_funding,BTC_opt_oi,ETH_index\n"
           "2026-11-27T12:00:00+00:00,61000,0.0001,300000,1700\n"
           "2026-11-27T12:15:00+00:00,61500,,310000,\n")
    snapfallback.market_frame(get=lambda url: _Resp(csv))
    vals, ts = snapfallback.latest_market("BTC", now=NOW)
    assert ts == TS and vals == {"index": 61500.0, "opt_oi": 310000.0}  # blank funding stays blank
    vals, ts = snapfallback.latest_market("ETH", now=NOW)
    assert vals["index"] == 1700.0 and ts == pd.Timestamp("2026-11-27T12:00:00+00:00")
    assert snapfallback.latest_market("SOL", now=NOW) == ({}, None)


def test_label_text():
    assert snapfallback.label(TS, NOW) == "Delayed — Deribit snapshot as of 20:15 SGT (15 min old)"
    assert snapfallback.banner(TS, "HTTP 403", NOW).endswith("; live Deribit blocked: HTTP 403")


def test_option_vols_fall_back_only_when_blocked(monkeypatch):
    monkeypatch.setattr(surface, "_option_vols_by_dte_uncached", lambda a: {"atm": {}})
    monkeypatch.setattr(snapfallback, "latest_expiry_vols", lambda a: ({"atm": {1: 0.46}}, TS))
    deribit._set_status(True, "ok")
    assert surface.option_vols_by_dte("BTC") == {"atm": {}}
    surface.clear_cache()
    deribit._set_status(False, "HTTP 403")
    snapfallback.begin()
    assert surface.option_vols_by_dte("BTC") == {"atm": {1: 0.46}}
    assert snapfallback.used() == TS
    snapfallback.begin()
    surface.option_vols_by_dte("BTC")  # cache hit still marks the thread
    assert snapfallback.used() == TS


def test_run_command_stamps_delayed_label_on_titles(monkeypatch):
    def handler(asset, **kw):
        snapfallback.mark(TS)
        return go.Figure(layout={"title": {"text": "BTC Vol Term Structure"}}), None, None

    monkeypatch.setitem(commands.HANDLERS, "vol_term_structure", handler)
    fig, df, text = commands.run_command("BTC", "/vol_term_structure")
    assert fig.layout.title.text.startswith("⚠ Delayed — Deribit snapshot as of 20:15 SGT")
    assert fig.layout.title.text.endswith("<br>BTC Vol Term Structure")
    assert "Delayed" in text


def test_run_command_live_is_untouched(monkeypatch):
    monkeypatch.setitem(commands.HANDLERS, "vol_term_structure",
                        lambda asset, **kw: (go.Figure(layout={"title": {"text": "T"}}), None, "x"))
    fig, df, text = commands.run_command("BTC", "/vol_term_structure")
    assert fig.layout.title.text == "T" and text == "x"


def test_record_market_appends_skips_same_slot_and_merges(tmp_path):
    import merge_market
    import record_snapshot as rs
    rows = {"BTC": {"index": 61000.0, "funding": 0.0}, "ETH": {}}
    p = tmp_path / "market.csv"
    assert rs.record_market(["BTC", "ETH"], path=p, now=NOW, row_fn=rows.get) == "recorded"
    assert rs.record_market(["BTC", "ETH"], path=p, now=NOW, row_fn=rows.get) == "skipped"
    later = datetime(2026, 11, 27, 12, 45, tzinfo=timezone.utc)
    assert rs.record_market(["BTC"], path=p, now=later, row_fn=rows.get) == "recorded"
    df = pd.read_csv(p)
    assert list(df["BTC_index"]) == [61000.0, 61000.0] and "ETH_index" not in df
    assert rs.record_market(["ETH"], path=tmp_path / "x.csv", now=NOW, row_fn=rows.get) == "no_data"
    pub = tmp_path / "pub"
    pub.mkdir()
    pd.DataFrame({"timestamp": ["2026-11-27T12:00:00+00:00"], "BTC_index": [60000.0]}).to_csv(
        pub / "market.csv", index=False)
    assert merge_market.merge(pub, tmp_path) == 3
