"""Refresh controls for the Ecoinometrics page. No network."""
from __future__ import annotations

import ast
import builtins
import os
import pickle
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from lib import ecoinometrics as eco
from lib import eco_refresh

ROOT = Path(__file__).resolve().parent.parent
NOW = pd.Timestamp("2026-01-15 00:30", tz="UTC")
ASOF = pd.Timestamp("2026-01-14")


@contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _reset_lock():
    with eco_refresh._guard:
        eco_refresh._running.clear()


class _Resp:
    def __init__(self, text="", content=b"", status=200, json_body=None):
        self.text = text
        self.content = content
        self.status_code = status
        self._json = {} if json_body is None else json_body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise eco.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


def _frame(rows=1):
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-14"]),
        "flow": [5.0],
    }).iloc[:rows]


def test_format_sgt_and_status_lines():
    assert eco_refresh.format_sgt(NOW) == "15 Jan 2026 08:30 SGT"
    assert eco_refresh.format_sgt(None) == "—"
    assert eco_refresh.format_as_of(ASOF) == "14 Jan 2026"
    live = eco.FetchResult(
        _frame(), "live", "https://farside.co.uk/btc/", NOW, ASOF, True, "Farside 1 rows.",
    )
    line = eco_refresh.format_result_line("ETF flows", live)
    assert line == (
        "ETF flows — live (https://farside.co.uk/btc/) — "
        "last updated 15 Jan 2026 08:30 SGT — through 14 Jan 2026 — "
        "1 rows — Farside 1 rows."
    )
    failed = eco.FetchResult(
        pd.DataFrame(columns=["date", "flow"]), "none", "data/etf_flows.csv",
        NOW, None, False, "Farside unavailable (HTTP 403). CSV fallback was not used.",
    )
    previous = eco.FetchResult(
        _frame(), "csv", "data/etf_flows.csv",
        pd.Timestamp("2026-01-01", tz="UTC"), ASOF, True, "using csv",
    )
    fail_line = eco_refresh.format_failure_line("ETF flows", failed, previous)
    assert "refresh failed" in fail_line
    assert "CSV fallback (data/etf_flows.csv)" in fail_line
    assert "01 Jan 2026 08:00 SGT" in fail_line
    shown = {"etf_flows": previous}
    attempts = {"etf_flows": failed}
    details = eco_refresh.detail_lines(shown, attempts)
    assert details[0] == fail_line
    caption = eco_refresh.summary_caption(shown, attempts)
    assert caption.startswith("ETF flows: refresh failed")
    ok_caption = eco_refresh.summary_caption({"etf_flows": live}, {})
    assert "ETF flows: live, updated 15 Jan 2026 08:30 SGT" in ok_caption
    assert "FRED: not loaded" in ok_caption


def test_etf_live_csv_and_refresh_does_not_hide_a_failed_scrape():
    html = (
        "<table><tr><th>Date</th><th>Total</th></tr>"
        "<tr><td>02 Jan 2026</td><td>(5.5)</td></tr></table>"
    )

    def ok_get(url, headers=None, timeout=None, params=None):
        return _Resp(text=html)

    with patched(eco.requests, "get", ok_get):
        result = eco.fetch_etf_flows_result(now=NOW)
        frame = eco.fetch_etf_flows()
    assert result.ok and result.source == "live"
    assert result.origin == eco.FARSIDE_URLS[0]
    assert list(frame["flow"]) == [-5.5]

    def bad_get(*_a, **_k):
        raise RuntimeError("blocked")

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        csv_path = folder / "etf_flows.csv"
        csv_path.write_text("date,flow\n2026-01-02,-5.5\n")
        os.utime(csv_path, (1_700_000_000, 1_700_000_000))
        with patched(eco.requests, "get", bad_get), patched(eco, "DATA_DIR", folder):
            fallback = eco.fetch_etf_flows_result(now=NOW)
            refresh = eco_refresh.build_etf_flows(data_dir=folder, now=NOW)
        assert fallback.ok and fallback.source == "csv"
        assert fallback.fetched_at == pd.Timestamp(1_700_000_000, unit="s", tz="UTC")
        assert not refresh.ok
        assert "CSV fallback was not used" in refresh.message
        assert csv_path.read_text().count("\n") == 2


def test_fred_live_fallback_and_partial_cache_write():
    def live_get(url, headers=None, timeout=None, params=None):
        sid = params["id"]
        return _Resp(text=f"observation_date,{sid}\n2024-01-02,1.5\n")

    with patched(eco.requests, "get", live_get):
        result = eco.fred_series_result("DFF", now=NOW)
        frame = eco.fred_series("DFF")
    assert result.ok and result.source == "live"
    assert list(frame["value"]) == [1.5]

    def bad_get(*_a, **_k):
        raise eco.requests.RequestException("timeout")

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "fred_DFF.csv").write_text("date,value\n2013-01-01,0.09\n2013-01-02,0.17\n")
        with patched(eco.requests, "get", bad_get), patched(eco, "DATA_DIR", folder):
            fallback = eco.fred_series_result("DFF", retries=1, now=NOW)
            live_only = eco.fred_series_result("DFF", retries=1, live_only=True, now=NOW)
        assert fallback.ok and fallback.source == "csv" and fallback.rows == 2
        assert not live_only.ok and "live only" in live_only.message

        def fetcher(sid):
            if sid == "DGS10":
                return eco.FetchResult(
                    pd.DataFrame(columns=["date", "value"]), "none", "x", NOW, None, False, "timeout",
                )
            df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "value": [1.25]})
            return eco.FetchResult(df, "live", f"https://fred.example/{sid}", NOW, ASOF, True, f"FRED {sid}: 1 rows.")

        def rude(_i, _n, _label):
            raise RuntimeError("progress widget failed")

        results = eco_refresh.build_fred_cache(
            write=True, data_dir=folder, fetcher=fetcher, on_progress=rude,
        )
        assert results["DFF"].ok and not results["DGS10"].ok
        assert (folder / "fred_DFF.csv").exists()
        assert not (folder / "fred_DGS10.csv").exists()
        back = pd.read_csv(folder / "fred_DFF.csv")
        assert len(back) == 1 and list(back["date"]) == ["2020-01-01"]
        assert "Wrote fred_DFF.csv (1 rows)" in results["DFF"].message
        level, text = eco_refresh.fred_banner(results)
        assert level == "warning" and "DGS10" in text


def test_fed_tone_merges_and_does_not_clobber_on_failure():
    calls = []

    def get(url, timeout):
        calls.append(url)
        if url.endswith(".pdf"):
            return _Resp(content=b"CHAIR POWELL. inflation is elevated and we may hike")
        return _Resp(text="see presconf20190130.htm")

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        path = folder / "fed_tone.csv"
        path.write_text(
            "date,hawk,dove,words,score,net_per_1k\n"
            "2019-03-20,1,1,10,0.0,0.0\n"
        )
        before = path.read_bytes()
        untouched = eco_refresh.build_fed_tone(
            2019, data_dir=folder, get=lambda url, timeout: _Resp(text="<html></html>"),
            extract_text=lambda content: content.decode(), now=NOW,
        )
        assert not untouched.ok
        assert "left untouched" in untouched.message
        assert path.read_bytes() == before

        merged = eco_refresh.build_fed_tone(
            2019, data_dir=folder, get=get, extract_text=lambda content: content.decode(), now=NOW,
        )
        assert merged.ok and merged.source == "live"
        assert "idx" in merged.df.columns and "avg6" in merged.df.columns
        back = pd.read_csv(path, parse_dates=["date"])
        assert len(back) == 2
        dates = set(pd.to_datetime(back["date"]).dt.strftime("%Y-%m-%d"))
        assert dates == {"2019-01-30", "2019-03-20"}
        assert "Wrote fed_tone.csv (2 rows)" in merged.message

    pages = []

    def count_pages(url, timeout):
        pages.append(url)
        return _Resp(text="")

    eco_refresh.discover_fomc_press_conferences(
        2019, count_pages, now=pd.Timestamp("2026-06-01", tz="UTC"),
    )
    calendar = [url for url in pages if url.endswith("fomccalendars.htm")]
    assert calendar and calendar.count(calendar[0]) == len(calendar) == 1


def test_missing_pypdf_does_not_fetch():
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf" or name.startswith("pypdf."):
            raise ImportError("no pypdf")
        return real_import(name, *args, **kwargs)

    called = []

    def get(*_a, **_k):
        called.append(1)
        raise AssertionError("network")

    builtins.__import__ = fake_import
    try:
        result = eco_refresh.build_fed_tone(get=get)
    finally:
        builtins.__import__ = real_import
    assert not result.ok
    assert "pypdf" in result.message
    assert called == []


def test_local_files_do_not_hit_the_network():
    def boom(*_a, **_k):
        raise AssertionError("network")

    with patched(eco.requests, "get", boom):
        tone = eco.load_fed_tone_result()
        hacks = eco.load_hacks_result()
    assert tone.ok and tone.source == "csv" and {"idx", "avg6"} <= set(tone.df.columns)
    assert hacks.ok and hacks.source == "bundled" and hacks.rows >= 1
    assert "exchange_hacks.csv" in hacks.origin


def test_bitstamp_and_deribit_results():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2015-01-01"]), "close": [250.0]})
    with patched(eco, "fetch_bitstamp_daily", lambda *a, **k: frame):
        ok = eco.fetch_bitstamp_daily_result(now=NOW)
    assert ok.ok and ok.source == "live" and "In-session only" in ok.message
    with patched(eco, "fetch_bitstamp_daily", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope"))):
        bad = eco.fetch_bitstamp_daily_result(now=NOW)
    assert not bad.ok and "nope" in bad.message

    start = "2020-01-01"
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)

    def get_ohlc(name, resolution, _start, _end):
        assert name == "BTC-PERPETUAL" and str(resolution) == "1D"
        return pd.DataFrame({
            "timestamp": pd.to_datetime(["2020-01-01 08:00"]),
            "open": [1], "high": [1], "low": [1], "close": [100.0], "volume": [1],
        })

    got = eco.fetch_deribit_btc_daily(get_ohlc, start=start, now_ms=start_ms + 86_400_000, now=NOW)
    assert got.ok and got.source == "live" and got.rows == 1
    assert abs(got.df["close"].iloc[0] - 100) < 1e-9
    end_ms = int(pd.Timestamp("2020-01-05", tz="UTC").timestamp() * 1000)
    empty = eco.fetch_deribit_btc_daily(lambda *a, **k: None, start=start, now_ms=end_ms)
    assert not empty.ok and "No BTC-PERP" in empty.message
    failed = eco.fetch_deribit_btc_daily(
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")),
        start=start, now_ms=end_ms,
    )
    assert not failed.ok and "down" in failed.message


def test_lock_blocks_a_second_refresh_and_releases_on_error():
    _reset_lock()
    try:
        assert eco_refresh.try_acquire("etf_flows")
        calls = []
        busy = eco_refresh.run_guarded("all", lambda: calls.append("ran"))
        assert busy.busy and calls == []
        eco_refresh.release("etf_flows")
        ran = eco_refresh.run_guarded("fred", lambda: "done")
        assert not ran.busy and ran.value == "done"
        assert not eco_refresh.refresh_busy()

        def explode():
            raise RuntimeError("boom")

        try:
            eco_refresh.run_guarded("fed_tone", explode)
            raise AssertionError("expected the error to propagate")
        except RuntimeError as ex:
            assert "boom" in str(ex)
        assert not eco_refresh.refresh_busy()
    finally:
        _reset_lock()


def test_dispatch_without_token_does_not_call_github_and_redacts_the_token():
    token = "ghp_" + "x" * 24
    posted = []

    def post(url, headers=None, json=None, timeout=None):
        posted.append({"url": url, "headers": headers, "json": json})
        return _Resp(status=204)

    missing = eco_refresh.dispatch_data_refresh(token="")
    assert not missing.ok and not posted
    assert "dispatch_token" in missing.message
    assert "Actions: Read and write" in missing.message
    assert eco_refresh.WORKFLOW_FILE in missing.message

    queued = eco_refresh.dispatch_data_refresh(token=token, post=post)
    assert queued.ok and queued.status_code == 204
    assert posted[0]["json"] == {"ref": "main", "inputs": {"datasets": "all"}}
    assert token not in queued.message
    assert "Authorization" not in queued.message
    assert eco_refresh.WORKFLOW_FILE in posted[0]["url"]

    def rejected(url, headers=None, json=None, timeout=None):
        return _Resp(status=401, json_body={"message": f"bad credentials {token}"})

    denied = eco_refresh.dispatch_data_refresh(token=token, post=rejected)
    assert not denied.ok and denied.status_code == 401
    assert token not in denied.message
    assert "[redacted]" in denied.message

    def explode(url, headers=None, json=None, timeout=None):
        raise RuntimeError(f"socket reset {token}")

    blown = eco_refresh.dispatch_data_refresh(token=token, post=explode)
    assert not blown.ok and token not in blown.message and "[redacted]" in blown.message

    missing_workflow = eco_refresh.dispatch_data_refresh(
        token=token,
        post=lambda url, **_k: _Resp(status=404, json_body={"message": "Not Found"}),
    )
    assert not missing_workflow.ok and "default branch" in missing_workflow.message
    unknown = eco_refresh.dispatch_data_refresh(datasets="btc", token=token, post=post)
    assert not unknown.ok and len(posted) == 1


def test_token_placeholder_and_config_file():
    assert eco_refresh._token_usable("YOUR_TOKEN") == ""
    assert eco_refresh._token_usable("  ") == ""
    assert eco_refresh._token_usable(None) == ""
    assert eco_refresh._token_usable("ghp_realvalue") == "ghp_realvalue"

    old_env = os.environ.pop("GITHUB_DISPATCH_TOKEN", None)
    home = eco_refresh.Path.home
    try:
        with tempfile.TemporaryDirectory() as tmp:
            eco_refresh.Path.home = classmethod(lambda cls: Path(tmp))
            cfg = Path(tmp) / ".config" / "mcm-analytics"
            cfg.mkdir(parents=True)
            (cfg / "secrets.toml").write_text('[github]\ndispatch_token = "YOUR_TOKEN"\n')
            # A configured Streamlit secret wins over the file. The placeholder
            # must never be the answer either way.
            got = eco_refresh.load_dispatch_token()
            assert got != "YOUR_TOKEN"
            (cfg / "secrets.toml").write_text('[github]\ndispatch_token = "ghp_fromfile_123456"\n')
            got = eco_refresh.load_dispatch_token()
            assert got in ("", "ghp_fromfile_123456") or got.startswith("ghp_") or got.startswith("github_pat_")
            if got == "ghp_fromfile_123456":
                assert "YOUR_" not in got
    finally:
        eco_refresh.Path.home = home
        if old_env is not None:
            os.environ["GITHUB_DISPATCH_TOKEN"] = old_env


def test_fetch_result_pickles_and_clear_cache_is_safe():
    result = eco.FetchResult(_frame(), "live", "https://example", NOW, ASOF, True, "ok")
    again = pickle.loads(pickle.dumps(result))
    assert again.ok and again.rows == 1 and again.source == "live"
    eco_refresh.clear_cache()


def test_tools_and_workflow_share_the_builders():
    fred = (ROOT / "tools" / "build_fred_cache.py").read_text(encoding="utf-8")
    tone = (ROOT / "tools" / "build_fed_tone.py").read_text(encoding="utf-8")
    refresh = (ROOT / "tools" / "refresh_ecoinometrics_data.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / eco_refresh.WORKFLOW_FILE).read_text(encoding="utf-8")
    cache = (ROOT / "lib" / "cache.py").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "from lib.eco_refresh import build_fred_cache" in fred
    assert "PdfReader" not in tone
    assert "build_fed_tone" in tone
    assert "build_etf_flows" in refresh and "build_fred_cache" in refresh
    assert "workflow_dispatch:" in workflow
    assert "refresh_ecoinometrics_data.py" in workflow
    assert "secrets." not in workflow
    assert "data/etf_flows.csv" in workflow and "data/fed_tone.csv" in workflow
    assert "eco_refresh.clear_cache" in cache
    assert "pypdf" in requirements


def test_page_builders_run_only_from_the_refresh_handler():
    tree = ast.parse((ROOT / "pages" / "13_Ecoinometrics_Charts.py").read_text(encoding="utf-8"))
    banned = {"build_etf_flows", "build_fred_cache", "build_fed_tone", "dispatch_data_refresh"}

    class Builders(ast.NodeVisitor):
        def __init__(self):
            self.stack = []
            self.offenders = []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            name = None
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if name in banned and "_on_refresh" not in self.stack:
                self.offenders.append((name, node.lineno))
            self.generic_visit(node)

    visitor = Builders()
    visitor.visit(tree)
    assert visitor.offenders == []

    class HandlerCalls(ast.NodeVisitor):
        def __init__(self):
            self.in_if = 0
            self.bare = []

        def visit_If(self, node):
            self.in_if += 1
            self.generic_visit(node)
            self.in_if -= 1

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "_on_refresh" and self.in_if == 0:
                self.bare.append(node.lineno)
            self.generic_visit(node)

    calls = HandlerCalls()
    calls.visit(tree)
    assert calls.bare == []


def test_import_does_not_start_a_refresh():
    assert eco_refresh.refresh_busy() is False
