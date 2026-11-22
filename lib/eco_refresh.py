"""
Rebuild the Ecoinometrics page's external datasets.

The CLI tools (``tools/build_fred_cache.py``, ``tools/build_fed_tone.py``,
``tools/refresh_ecoinometrics_data.py``) and the Streamlit page call the same
functions. Nothing here runs on import, and nothing here fetches on its own:
the page calls a builder only from a button.

Streamlit Community Cloud's disk is ephemeral and the app cannot push to git.
A successful builder updates the in-memory ``FetchResult`` the page stores in
``st.session_state`` and, when the disk is writable, the CSV under ``data/``.
That file survives only until the next redeploy. ``dispatch_data_refresh``
optionally queues ``.github/workflows/update_ecoinometrics_data.yml``, which
commits the CSVs, and only when a token is configured.

``clear_cache`` drops those session frames. The in-process lock that stops two
refreshes from running at once is not a cache and is left alone.

This module must stay importable without plotly or streamlit: the weekly
GitHub Action (``update_ecoinometrics_data.yml``) installs only requests,
pandas, lxml and pypdf. Do not import ``lib.fx_style`` here; it pulls in
plotly at module level.
"""
from __future__ import annotations

import io
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from lib import ecoinometrics as eco

GITHUB_REPO = "seanbmcnulty/mcm-analytics"
WORKFLOW_FILE = "update_ecoinometrics_data.yml"
DISPATCH_DATASETS = ("all", "etf_flows", "fred", "fed_tone", "strategy")
FRED_SERIES = eco.FRED_SERIES
FED_BASE = "https://www.federalreserve.gov"
_PRESCONF = re.compile(r"presconf(\d{8})\.(?:htm|pdf)")
TONE_COLUMNS = ["date", "hawk", "dove", "words", "score", "net_per_1k"]

SESSION_KEYS = ("eco_good", "eco_attempt", "eco_shown")
BUSY_MESSAGE = "A refresh is already running. Wait for it to finish before starting another."

SOURCE_LABELS = {
    "live": "live",
    "csv": "CSV fallback",
    "bundled": "bundled CSV",
    "mixed": "mixed",
    "none": "unavailable",
}
DATASET_LABELS = {
    "etf_flows": "ETF flows",
    "fed_tone": "Fed tone",
    "bitstamp": "Bitstamp BTC",
    "deribit_btc": "Deribit BTC-PERP",
    "hacks": "Exchange hacks",
    "strategy": "Strategy 8-K",
}

NO_TOKEN_MESSAGE = (
    "No GitHub token configured, so the committed CSVs were not rebuilt. "
    "In-session refresh still works without it. "
    "To enable this button, set `[github] dispatch_token` in Streamlit secrets "
    "(or `~/.config/mcm-analytics/secrets.toml`, or env `GITHUB_DISPATCH_TOKEN`) "
    "to a fine-grained personal access token for seanbmcnulty/mcm-analytics "
    "with Actions: Read and write. Contents can stay at No access: "
    "the workflow's own GITHUB_TOKEN commits the CSVs. "
    "The workflow file is `.github/workflows/update_ecoinometrics_data.yml`."
)
DISPATCH_OK_MESSAGE = (
    "Queued the GitHub Action update_ecoinometrics_data.yml on main. "
    "When the run finishes it commits data/fred_*.csv, data/fed_tone.csv, and "
    "data/etf_flows.csv. That push redeploys Streamlit Community Cloud. "
    "This button does not change the charts in front of you; use the dataset "
    "refresh buttons for that."
)

_guard = threading.Lock()
_running: set[str] = set()


@dataclass
class GuardResult:
    busy: bool
    value: object = None


@dataclass
class DispatchResult:
    ok: bool
    message: str
    status_code: int | None = None


def try_acquire(name: str) -> bool:
    """Return False when any refresh is already in progress.

    One name is held at a time, process-wide, so a second click or a second
    Streamlit session cannot start another Farside, FRED, or Fed-tone rebuild
    while the first is still inside this process. ``dispatch`` takes the same
    lock: it is a short HTTP call, and it should not race a CSV rewrite.
    """
    with _guard:
        if _running:
            return False
        _running.add(name)
        return True


def release(name: str) -> None:
    with _guard:
        _running.discard(name)


def refresh_busy() -> bool:
    with _guard:
        return bool(_running)


def run_guarded(name: str, fn):
    """Run ``fn`` while ``name`` is held. A conflicting hold does not call ``fn``."""
    if not try_acquire(name):
        return GuardResult(True, None)
    try:
        return GuardResult(False, fn())
    finally:
        release(name)


def clear_cache() -> None:
    """Drop in-session Ecoinometrics frames. Safe outside a Streamlit runtime."""
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        if get_script_run_ctx(suppress_warning=True) is None:
            return
    except Exception:
        return
    try:
        import streamlit as st
        state = st.session_state
    except Exception:
        return
    for key in SESSION_KEYS:
        try:
            del state[key]
        except Exception:
            try:
                state.pop(key, None)
            except Exception:
                pass


def _dir(data_dir: Path | None) -> Path:
    return Path(data_dir) if data_dir is not None else eco.DATA_DIR


def _log(log, message: str) -> None:
    if log is not None:
        log(message)


def _progress(callback, index: int, total: int, label: str) -> None:
    """Progress is best-effort. A UI callback must not abort a download."""
    if callback is None:
        return
    try:
        callback(index, total, label)
    except Exception:
        pass


def write_dataset_csv(df: pd.DataFrame, path: Path) -> str:
    """Write ``df`` and read it back. Never raises.

    Dates are ``YYYY-MM-DD`` so a later ``parse_dates`` read sees the same
    number of rows. A short count on the way back is reported and the file
    is still left in place for inspection.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ready = df.copy()
        if "date" in ready.columns:
            ready["date"] = pd.to_datetime(ready["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        ready.to_csv(path, index=False)
        back = pd.read_csv(path)
        if len(back) != len(ready):
            return (
                f"Wrote {path.name} but read back {len(back)} rows, expected {len(ready)}."
            )
        return (
            f"Wrote {path.name} ({len(ready)} rows) on this machine; "
            "a redeploy drops it unless the GitHub Action commits it."
        )
    except Exception as ex:
        return f"In-session only; could not write {path.name} ({ex})."


def build_etf_flows(*, write: bool = True, data_dir: Path | None = None, timeout: int = 20,
                    now=None, log=None) -> eco.FetchResult:
    """Scrape Farside. Does not fall back to the CSV: a failed scrape is a failure."""
    result = eco.fetch_etf_flows_result(timeout=timeout, live=True, allow_csv=False, now=now)
    if not result.ok:
        _log(log, result.message)
        return result
    note = "In-session only (file write skipped)."
    if write:
        note = write_dataset_csv(result.df[["date", "flow"]], _dir(data_dir) / "etf_flows.csv")
    result.message = f"{result.message} {note}"
    _log(log, result.message)
    return result


def build_fred_cache(series=FRED_SERIES, *, write: bool = True, data_dir: Path | None = None,
                     fetcher=None, now=None, log=None, on_progress=None) -> dict[str, eco.FetchResult]:
    """Fetch each FRED series live and, on success, rewrite ``data/fred_<ID>.csv``.

    One series failing leaves the others in the returned dict and does not
    delete an existing CSV for the series that failed.
    """
    fetch = fetcher or (lambda sid: eco.fred_series_result(sid, live_only=True, now=now))
    folder = _dir(data_dir)
    out: dict[str, eco.FetchResult] = {}
    total = len(tuple(series))
    for index, sid in enumerate(tuple(series), start=1):
        _progress(on_progress, index - 1, total, f"FRED {sid}")
        try:
            result = fetch(sid)
        except Exception as ex:
            result = eco._fail(
                ["date", "value"], f"data/fred_{sid}.csv", eco._now(now),
                f"FRED {sid} unavailable ({ex})",
            )
        if result.ok and write:
            try:
                note = write_dataset_csv(result.df[["date", "value"]], folder / f"fred_{sid}.csv")
            except Exception as ex:
                note = f"In-session only; could not write fred_{sid}.csv ({ex})."
            result.message = f"{result.message} {note}"
        _log(log, f"{sid} {result.message}")
        out[sid] = result
    _progress(on_progress, total, total, "FRED done")
    return out


def build_strategy_funding(*, write: bool = True, data_dir: Path | None = None, get=None,
                           now=None, log=None) -> eco.FetchResult:
    """Parse Strategy's latest weekly 8-K from SEC EDGAR. No CSV fallback on refresh.

    On success, rewrites ``data/strategy_funding.csv`` (the page's fallback).
    """
    from lib import strategy_8k
    result = strategy_8k.load_result(live=True, allow_csv=False, get=get, now=now)
    if not result.ok:
        _log(log, result.message)
        return result
    note = "In-session only (file write skipped)."
    if write:
        note = write_dataset_csv(result.df, _dir(data_dir) / "strategy_funding.csv")
    result.message = f"{result.message} {note}"
    _log(log, result.message)
    return result


def _default_get(url: str, timeout: int):
    response = requests.get(url, headers=eco.UA, timeout=timeout)
    response.raise_for_status()
    return response


def _pdf_extractor():
    """Return ``(extract, error_message)``. Imports pypdf only when called."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None, (
            "Fed tone refresh needs pypdf (pip install pypdf). "
            "data/fed_tone.csv was left untouched."
        )

    def extract(content: bytes) -> str:
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return extract, None


def discover_fomc_press_conferences(first_year: int, get, now=None) -> tuple[dict[str, str], list[str]]:
    """Map ``YYYYMMDD`` to the press-conference PDF url. One fetch per calendar page."""
    this_year = eco._now(now).year
    pages: dict[str, None] = {}
    for year in range(int(first_year), this_year + 1):
        if year >= this_year - 4:
            path = "/monetarypolicy/fomccalendars.htm"
        else:
            path = f"/monetarypolicy/fomchistorical{year}.htm"
        pages.setdefault(FED_BASE + path, None)
    urls: dict[str, str] = {}
    failures: list[str] = []
    for page_url in pages:
        try:
            html = get(page_url, 30).text
        except Exception as ex:
            failures.append(f"{page_url} ({ex})")
            continue
        for match in _PRESCONF.finditer(html):
            stamp = match.group(1)
            if int(stamp[:4]) < int(first_year):
                continue
            urls[stamp] = f"{FED_BASE}/mediacenter/files/FOMCpresconf{stamp}.pdf"
    return urls, failures


def _read_tone_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["date"])
    keep = [column for column in TONE_COLUMNS if column in frame.columns]
    return frame[keep]


def build_fed_tone(first_year: int = eco.FED_TONE_START_YEAR, *, write: bool = True,
                   merge_existing: bool = True, data_dir: Path | None = None,
                   get=None, extract_text=None, now=None, log=None,
                   on_progress=None) -> eco.FetchResult:
    """Score FOMC press-conference PDFs and merge them into ``data/fed_tone.csv``.

    Dates that are not rescored keep their previous row, so a partial download
    does not truncate the bundled history. No new transcripts at all leaves the
    file untouched and returns ``ok=False``.
    """
    fetched_at = eco._now(now)
    origin = "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf*.pdf"
    folder = _dir(data_dir)
    path = folder / "fed_tone.csv"
    if extract_text is None:
        extract_text, import_error = _pdf_extractor()
        if import_error:
            return eco.FetchResult(
                eco._empty(["date", "score", "idx", "avg6"]), "none", origin,
                fetched_at, None, False, import_error,
            )
    getter = get or _default_get
    urls, page_failures = discover_fomc_press_conferences(first_year, getter, now=now)
    rows = []
    pdf_failures: list[str] = []
    ordered = sorted(urls.items())
    total = len(ordered)
    for index, (stamp, url) in enumerate(ordered, start=1):
        _progress(on_progress, index - 1, total, f"Fed transcript {stamp}")
        try:
            payload = getter(url, 60).content
            text = extract_text(payload)
            scored = eco.tone_score(eco.chair_text(text))
            rows.append({"date": pd.to_datetime(stamp), **scored})
            _log(log, f"{stamp} {scored['score']}")
        except Exception as ex:
            pdf_failures.append(f"{stamp} ({ex})")
            _log(log, f"fail {stamp} {ex}")
    _progress(on_progress, total, total, "Fed tone done")
    failures = page_failures + pdf_failures
    fail_note = ""
    if failures:
        shown = ", ".join(failures[:8])
        if len(failures) > 8:
            shown += f", +{len(failures) - 8} more"
        fail_note = f" Failures: {shown}."

    if not rows:
        existing = _read_tone_csv(path)
        indexed = eco.with_tone_index(existing) if existing is not None else eco._empty(
            ["date", "score", "idx", "avg6"])
        return eco.FetchResult(
            indexed, "csv" if existing is not None else "none", "data/fed_tone.csv",
            eco.file_mtime(path) or fetched_at,
            eco._as_of(indexed, "date") if existing is not None else None,
            False,
            "No transcripts fetched; existing data/fed_tone.csv left untouched." + fail_note,
        )

    fresh = pd.DataFrame(rows)
    fresh["date"] = pd.to_datetime(fresh["date"])
    for column in TONE_COLUMNS:
        if column not in fresh.columns:
            fresh[column] = pd.NA
    fresh = fresh[TONE_COLUMNS].sort_values("date")
    if merge_existing:
        existing = _read_tone_csv(path)
        if existing is not None and len(existing):
            fresh = pd.concat([existing, fresh], ignore_index=True)
            fresh["date"] = pd.to_datetime(fresh["date"])
            fresh = fresh.drop_duplicates("date", keep="last").sort_values("date")
            for column in TONE_COLUMNS:
                if column not in fresh.columns:
                    fresh[column] = pd.NA
            fresh = fresh[TONE_COLUMNS]
    note = "In-session only (file write skipped)."
    if write:
        note = write_dataset_csv(fresh[TONE_COLUMNS], path)
    indexed = eco.with_tone_index(fresh)
    message = (
        f"Scored {len(rows)} transcripts; {len(indexed)} meetings in the merged file. "
        f"{note}{fail_note}"
    )
    _log(log, message)
    return eco.FetchResult(
        indexed, "live", origin, fetched_at, eco._as_of(indexed, "date"), True, message,
    )


def _token_usable(value) -> str:
    if not isinstance(value, str):
        return ""
    token = value.strip()
    if not token or token.startswith("YOUR_"):
        return ""
    return token


def load_dispatch_token() -> str:
    """Streamlit secrets, then the user config file, then the environment.

    Empty, and any ``YOUR_...`` placeholder, count as not configured. The
    token is not logged.
    """
    import os

    try:
        import streamlit as st
        secrets = st.secrets
        token = _token_usable(secrets.get("github", {}).get("dispatch_token", ""))
        if token:
            return token
        token = _token_usable(secrets.get("GITHUB_DISPATCH_TOKEN", ""))
        if token:
            return token
    except Exception:
        pass

    config_path = Path.home() / ".config" / "mcm-analytics" / "secrets.toml"
    if config_path.exists():
        try:
            import toml
            data = toml.load(config_path)
            github = data.get("github", {}) if isinstance(data, dict) else {}
            token = _token_usable(github.get("dispatch_token", "") if isinstance(github, dict) else "")
            if token:
                return token
        except Exception:
            pass

    return _token_usable(os.environ.get("GITHUB_DISPATCH_TOKEN", ""))


def _redact(text: str, token: str) -> str:
    if not text:
        return ""
    if token and len(token) >= 8:
        text = text.replace(token, "[redacted]")
    return text


def dispatch_data_refresh(datasets: str = "all", *, token: str | None = None, post=None,
                          ) -> DispatchResult:
    """Queue the CSV rebuild workflow. No token and no network call when unset.

    HTTP 204 is GitHub's success for workflow_dispatch. Any other outcome is a
    message the page can show; the token is removed from that message.
    """
    if datasets not in DISPATCH_DATASETS:
        return DispatchResult(False, f"Unknown dataset {datasets!r}.", None)
    if token is None:
        token = load_dispatch_token()
    if not token:
        return DispatchResult(False, NO_TOKEN_MESSAGE, None)
    sender = post or requests.post
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    try:
        response = sender(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
            json={"ref": "main", "inputs": {"datasets": datasets}},
            timeout=20,
        )
    except Exception as ex:
        return DispatchResult(
            False,
            f"GitHub dispatch failed: {_redact(str(ex), token)}. In-session data was not changed.",
            None,
        )
    code = getattr(response, "status_code", None)
    if code == 204:
        return DispatchResult(True, DISPATCH_OK_MESSAGE, 204)
    body = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            body = str(payload.get("message") or "")
    except Exception:
        body = ""
    body = _redact(body, token)[:300]
    hint = ""
    if code == 404:
        hint = (
            " The workflow must be on the default branch (main) before dispatch works."
        )
    return DispatchResult(
        False,
        f"GitHub dispatch failed (HTTP {code}). {body}{hint} In-session data was not changed.",
        code,
    )


def _to_sgt(ts):
    """Convert a single timestamp to Asia/Singapore (tz-naive)."""
    if ts is None:
        return ts
    try:
        t = pd.Timestamp(ts)
        if t.tz is None:
            t = t.tz_localize("UTC")
        return t.tz_convert("Asia/Singapore").tz_localize(None)
    except Exception:
        return pd.Timestamp(ts)


def format_sgt(ts) -> str:
    """Format a timestamp in Asia/Singapore."""
    if ts is None:
        return "—"
    try:
        if pd.isna(ts):
            return "—"
    except TypeError:
        return "—"
    local = _to_sgt(ts)
    if local is None:
        return "—"
    try:
        if pd.isna(local):
            return "—"
    except TypeError:
        pass
    return f"{pd.Timestamp(local):%d %b %Y %H:%M} SGT"


def format_as_of(ts) -> str:
    """Calendar date of the latest observation, in UTC, without moving the session day."""
    if ts is None:
        return "—"
    try:
        if pd.isna(ts):
            return "—"
    except TypeError:
        return "—"
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return f"{stamp:%d %b %Y}"


def source_phrase(result: eco.FetchResult) -> str:
    label = SOURCE_LABELS.get(result.source, result.source)
    if result.origin:
        return f"{label} ({result.origin})"
    return label


def format_result_line(label: str, result: eco.FetchResult) -> str:
    return (
        f"{label} — {source_phrase(result)} — "
        f"last updated {format_sgt(result.fetched_at)} — "
        f"through {format_as_of(result.as_of)} — "
        f"{result.rows} rows — {result.message}"
    )


def format_failure_line(label: str, attempt: eco.FetchResult, shown: eco.FetchResult | None) -> str:
    line = f"{label} — refresh failed — {attempt.message}"
    if shown is not None and shown.ok:
        line += (
            f" Showing previous {source_phrase(shown)}, "
            f"last updated {format_sgt(shown.fetched_at)}, "
            f"through {format_as_of(shown.as_of)}."
        )
    return line


def _pair(key: str, shown: dict | None, attempts: dict | None):
    attempt = None if not attempts else attempts.get(key)
    show = None if not shown else shown.get(key)
    return attempt, show


def _detail_line(label: str, attempt, show) -> str:
    if attempt is not None and not attempt.ok:
        return format_failure_line(label, attempt, show if show is not attempt else None)
    current = show if show is not None else attempt
    if current is None:
        return f"{label} — not loaded yet"
    return format_result_line(label, current)


def _short(label: str, attempt, show) -> str:
    if attempt is not None and not attempt.ok:
        return f"{label}: refresh failed"
    current = show if show is not None else attempt
    if current is None:
        return f"{label}: not loaded"
    src = SOURCE_LABELS.get(current.source, current.source)
    return f"{label}: {src}, updated {format_sgt(current.fetched_at)}"


def _fred_members(shown, attempts) -> list[tuple[str, object, object]]:
    rows = []
    for sid in FRED_SERIES:
        rows.append((sid, *_pair(f"fred:{sid}", shown, attempts)))
    return rows


def _fred_summary(shown, attempts) -> tuple[str, str]:
    """Return ``(short, detail)`` for the four FRED series as one dataset."""
    members = _fred_members(shown, attempts)
    loaded = []
    failed = []
    for sid, attempt, show in members:
        if attempt is not None and not attempt.ok:
            failed.append((sid, attempt, show))
        current = show if show is not None else attempt
        if current is not None and current.ok and not (attempt is not None and not attempt.ok):
            loaded.append((sid, current))
        elif attempt is not None and not attempt.ok and show is not None and show.ok:
            loaded.append((sid, show))
    if not loaded and not failed and all(attempt is None and show is None for _, attempt, show in members):
        return "FRED: not loaded", "FRED — not loaded yet"
    sources = {item.source for _, item in loaded}
    if failed and not loaded:
        source = "unavailable"
    elif len(sources) == 1:
        source = SOURCE_LABELS.get(sources.pop(), "mixed")
    elif sources:
        source = "mixed"
    else:
        source = "unavailable"
    times = [item.fetched_at for _, item in loaded if item.fetched_at is not None]
    as_ofs = [item.as_of for _, item in loaded if item.as_of is not None]
    when = format_sgt(max(times)) if times else "—"
    through = format_as_of(max(as_ofs)) if as_ofs else "—"
    n_ok = len(loaded)
    n = len(members)
    fail_bits = ", ".join(sid for sid, _, _ in failed)
    if failed:
        short = f"FRED: {source}, {n_ok}/{n} series, refresh failed ({fail_bits})"
        detail = (
            f"FRED — {source} — {n_ok}/{n} series — "
            f"last updated {when} — through {through} — "
            f"failed: {fail_bits}"
        )
    else:
        short = f"FRED: {source}, {n_ok}/{n} series, updated {when}"
        detail = (
            f"FRED — {source} — {n_ok}/{n} series — "
            f"last updated {when} — through {through}"
        )
    return short, detail


def summary_caption(shown: dict | None, attempts: dict | None) -> str:
    """One line for the top of the page. Times are SGT."""
    bits = [_short("ETF flows", *_pair("etf_flows", shown, attempts))]
    bits.append(_fred_summary(shown, attempts)[0])
    for key in ("fed_tone", "bitstamp", "deribit_btc", "hacks", "strategy"):
        bits.append(_short(DATASET_LABELS[key], *_pair(key, shown, attempts)))
    return " · ".join(bits)


def detail_lines(shown: dict | None, attempts: dict | None) -> list[str]:
    """Per-dataset lines: source, last updated in SGT, through-date, and the latest message."""
    lines = [_detail_line("ETF flows", *_pair("etf_flows", shown, attempts))]
    lines.append(_fred_summary(shown, attempts)[1])
    for sid, attempt, show in _fred_members(shown, attempts):
        lines.append(_detail_line(f"FRED {sid}", attempt, show))
    for key in ("fed_tone", "bitstamp", "deribit_btc", "hacks", "strategy"):
        lines.append(_detail_line(DATASET_LABELS[key], *_pair(key, shown, attempts)))
    return lines


def banner(label: str, result: eco.FetchResult) -> tuple[str, str]:
    """``('success'|'error', message)`` for a just-finished refresh."""
    if result.ok:
        return "success", f"{label}: {result.message}"
    return "error", f"{label}: {result.message}"


def fred_banner(results: dict[str, eco.FetchResult]) -> tuple[str, str]:
    ok = [sid for sid, result in results.items() if result.ok]
    bad = [f"{sid}: {result.message}" for sid, result in results.items() if not result.ok]
    if bad and ok:
        return "warning", f"FRED: {len(ok)}/{len(results)} series refreshed. " + " ".join(bad)
    if bad:
        return "error", "FRED refresh failed. " + " ".join(bad)
    return "success", f"FRED: {len(ok)}/{len(results)} series refreshed."
