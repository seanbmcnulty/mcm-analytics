"""
Ecoinometrics-style charts — pure data/analytics helpers (no Streamlit).

DELIBERATE EXCEPTIONS to "Deribit-only" (requested by the user 2026-11-01):
  * US spot BTC ETF daily flows  -> Farside (scraped HTML) or data/etf_flows.csv
  * Fed press-conference tone    -> data/fed_tone.csv built by tools/build_fed_tone.py
    from federalreserve.gov transcripts (dictionary score, NOT Ecoinometrics' index).
    The page can rebuild it in-session; .github/workflows/update_ecoinometrics_data.yml
    commits a fresh CSV when someone dispatches it.
BTC prices for the hack study come from Deribit (lib/deribit.py) in the page.

Charts replicated (Ecoinometrics, 2026-10-30 issue):
  1. ETF flow streaks: cumulative flow over consecutive same-sign trading days,
     plus rolling 30-trading-day net flow.
  2. BTC N-day return after major exchange hacks (data/exchange_hacks.csv).
  3. Fed hawk/dove tone per FOMC press conference + 6-meeting average.

Added 2026-11-09 (inspired by the 6 Nov 2026 issue, not a copy of it):
  4. Close / 200-calendar-day SMA. See ``price_sma_ratio``.
  5. Daily ETF flow in BTC, coloured by ``etf_flow_regime``. That regime is
     MCM's own rule. It is not the Ecoinometrics model, which is unpublished.
"""
from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FARSIDE_URLS = ("https://farside.co.uk/btc/", "https://farside.co.uk/bitcoin-etf-flow-all-data/")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_SERIES = ("DFF", "NASDAQCOM", "CPIAUCSL", "DGS10")
# Hack events before Deribit's perp history (Aug 2018) use Bitstamp spot.
BITSTAMP_START = "2014-12-01"
BITSTAMP_END = "2018-08-02"
BITSTAMP_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
DERIBIT_BTC_DAILY_START = "2018-08-01"
# Bundled data/fed_tone.csv starts in 2019. The in-app rebuild uses this so a
# refresh does not quietly drop the early meetings. The CLI still defaults to
# 2022 when no year is passed, and merges into the existing file.
FED_TONE_START_YEAR = 2019
UA = {"User-Agent": "Mozilla/5.0 (compatible; mcm-analytics)"}


@dataclass
class FetchResult:
    """One dataset load, with enough provenance for the refresh panel.

    ``source`` is ``live`` (upstream just responded), ``csv`` (committed or
    locally written fallback), ``bundled`` (hand-curated file, no live feed),
    or ``none``. ``fetched_at`` is timezone-aware UTC: the fetch time for a
    live load, or the file mtime for a disk load. ``as_of`` is the latest
    observation, not the fetch time.
    """

    df: pd.DataFrame
    source: str
    origin: str
    fetched_at: pd.Timestamp | None
    as_of: pd.Timestamp | None
    ok: bool
    message: str

    @property
    def rows(self) -> int:
        if self.df is None:
            return 0
        return int(len(self.df))


def _now(now) -> pd.Timestamp:
    if now is None:
        return pd.Timestamp.now(tz="UTC")
    ts = pd.Timestamp(now)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def file_mtime(path: Path) -> pd.Timestamp | None:
    """File modification time as timezone-aware UTC, or None if it cannot be read."""
    try:
        return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
    except OSError:
        return None


def _as_of(df: pd.DataFrame | None, column: str) -> pd.Timestamp | None:
    if df is None or len(df) == 0 or column not in df.columns:
        return None
    values = pd.to_datetime(df[column], errors="coerce", utc=True).dropna()
    if values.empty:
        return None
    return pd.Timestamp(values.max())


def _empty(columns) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _fail(columns, origin: str, fetched_at, message: str) -> FetchResult:
    return FetchResult(_empty(columns), "none", origin, fetched_at, None, False, message)

# ---------------------------------------------------------------- ETF flows
def _num(x) -> float:
    s = str(x).strip().replace(",", "")
    if s in ("", "-", "nan", "NaN"):
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -v if neg else v


def parse_farside_html(html: str) -> pd.DataFrame:
    """Farside BTC table -> DataFrame[date, flow] (US$ m, total of all ETFs)."""
    for t in pd.read_html(io.StringIO(html)):
        cols = [str(c).strip().lower() for c in t.columns.get_level_values(-1)]
        t.columns = cols
        if "date" in cols and "total" in cols:
            d = pd.to_datetime(t["date"], format="%d %b %Y", errors="coerce")
            out = pd.DataFrame({"date": d, "flow": t["total"].map(_num)})
            return out.dropna().sort_values("date").reset_index(drop=True)
    raise ValueError("no Farside flow table found")


def fetch_etf_flows_result(timeout: int = 20, *, live: bool = True, allow_csv: bool = True,
                           now=None) -> FetchResult:
    """Farside totals, with provenance.

    A refresh passes ``allow_csv=False`` so a dead scrape is a failure instead
    of a silent re-read of the fallback file. Page loads pass the defaults and
    keep today's behaviour: live first, then ``data/etf_flows.csv``.
    """
    fetched_at = _now(now)
    err = None
    if live:
        for url in FARSIDE_URLS:
            try:
                r = requests.get(url, headers=UA, timeout=timeout)
                r.raise_for_status()
                df = parse_farside_html(r.text)
                return FetchResult(
                    df, "live", url, fetched_at, _as_of(df, "date"), True,
                    f"Farside {len(df)} rows.",
                )
            except Exception as e:  # blocked / format change
                err = e
    path = DATA_DIR / "etf_flows.csv"
    if not allow_csv:
        return _fail(
            ["date", "flow"], "data/etf_flows.csv", fetched_at,
            f"Farside unavailable ({err}). CSV fallback was not used.",
        )
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df = df[["date", "flow"]].sort_values("date").reset_index(drop=True)
        except Exception as e:
            return _fail(
                ["date", "flow"], "data/etf_flows.csv", file_mtime(path) or fetched_at,
                f"Farside unavailable ({err}) and data/etf_flows.csv could not be read ({e})",
            )
        return FetchResult(
            df, "csv", "data/etf_flows.csv", file_mtime(path) or fetched_at,
            _as_of(df, "date"), True,
            f"Farside unavailable ({err}); using data/etf_flows.csv.",
        )
    return _fail(
        ["date", "flow"], "data/etf_flows.csv", fetched_at,
        f"Farside unavailable ({err}) and no data/etf_flows.csv",
    )


def fetch_etf_flows(timeout: int = 20) -> pd.DataFrame:
    """Try Farside; fall back to data/etf_flows.csv (date,flow in US$ m)."""
    result = fetch_etf_flows_result(timeout=timeout)
    if not result.ok:
        raise RuntimeError(result.message)
    return result.df


def flow_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive same-sign trading days. Zero-flow days are dropped."""
    d = df[df["flow"] != 0].reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(columns=["start", "end", "days", "total"])
    sign = np.sign(d["flow"])
    grp = (sign != sign.shift()).cumsum()
    g = d.groupby(grp)
    return pd.DataFrame({
        "start": g["date"].first(), "end": g["date"].last(),
        "days": g["flow"].size(), "total": g["flow"].sum(),
    }).reset_index(drop=True)


def rolling_flow(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """Rolling net flow over `window` trading days."""
    return df.set_index("date")["flow"].rolling(window).sum()


# ----------------------------------------------------------- external helpers
def _read_fred_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.dropna().reset_index(drop=True)


def fred_series_result(series_id: str, timeout: int = 15, start: str = "2013-01-01", retries: int = 2,
                       live_only: bool = False, now=None) -> FetchResult:
    """FRED public CSV (no key), with provenance.

    Falls back to ``data/fred_<ID>.csv`` unless ``live_only``. Those files are
    rewritten by ``lib.eco_refresh.build_fred_cache`` and by
    ``.github/workflows/update_ecoinometrics_data.yml``. Streamlit Cloud often
    times out reaching FRED, which is why the fallback exists.
    """
    fetched_at = _now(now)
    origin = f"{FRED_URL}?id={series_id}"
    err = None
    response = None
    for _ in range(retries):
        try:
            response = requests.get(
                FRED_URL, params={"id": series_id, "cosd": start}, headers=UA, timeout=timeout,
            )
            response.raise_for_status()
            break
        except requests.RequestException as e:
            err = e
            response = None
    if response is not None:
        try:
            df = pd.read_csv(io.StringIO(response.text))
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna().reset_index(drop=True)
            return FetchResult(
                df, "live", origin, fetched_at, _as_of(df, "date"), True,
                f"FRED {series_id}: {len(df)} rows.",
            )
        except Exception as e:
            err = e
            response = None
    path = DATA_DIR / f"fred_{series_id}.csv"
    if live_only or not path.exists():
        why = f"FRED {series_id} unavailable ({err})"
        if live_only:
            why += " (live only)"
        else:
            why += f" and no data/fred_{series_id}.csv"
        return _fail(["date", "value"], f"data/fred_{series_id}.csv", fetched_at, why)
    try:
        df = _read_fred_csv(path)
    except Exception as e:
        return _fail(
            ["date", "value"], f"data/fred_{series_id}.csv", file_mtime(path) or fetched_at,
            f"FRED {series_id} live fetch failed ({err}) and data/fred_{series_id}.csv could not be read ({e})",
        )
    return FetchResult(
        df, "csv", f"data/fred_{series_id}.csv", file_mtime(path) or fetched_at,
        _as_of(df, "date"), True,
        f"FRED {series_id} live fetch failed ({err}); using data/fred_{series_id}.csv.",
    )


def fred_series(series_id: str, timeout: int = 15, start: str = "2013-01-01", retries: int = 2,
                live_only: bool = False) -> pd.DataFrame:
    """FRED public CSV (no key). Returns DataFrame[date, value].

    Falls back to data/fred_<ID>.csv (kept fresh by tools/build_fred_cache.py
    and .github/workflows/update_ecoinometrics_data.yml) because Streamlit
    Cloud frequently times out reaching FRED.
    """
    result = fred_series_result(
        series_id, timeout=timeout, start=start, retries=retries, live_only=live_only,
    )
    if not result.ok:
        raise RuntimeError(result.message)
    return result.df


def fetch_bitstamp_daily(start: str, end: str, timeout: int = 20) -> pd.DataFrame:
    """Daily BTC/USD closes from Bitstamp (history to 2011) -> DataFrame[timestamp, close].
    Used only for hack events before Deribit's perp history (Aug 2018)."""
    out, cur, end_ts = [], int(pd.Timestamp(start, tz="UTC").timestamp()), int(pd.Timestamp(end, tz="UTC").timestamp())
    while cur < end_ts:
        r = requests.get(BITSTAMP_URL, headers=UA, timeout=timeout,
                         params={"step": 86400, "limit": 1000, "start": cur})
        r.raise_for_status()
        rows = r.json()["data"]["ohlc"]
        if not rows:
            break
        out += rows
        nxt = int(rows[-1]["timestamp"]) + 86400
        if nxt <= cur:
            break
        cur = nxt
    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df["close"] = df["close"].astype(float)
    return df[["timestamp", "close"]].drop_duplicates("timestamp")


def fetch_bitstamp_daily_result(start: str = BITSTAMP_START, end: str = BITSTAMP_END,
                                timeout: int = 20, now=None) -> FetchResult:
    """Bitstamp daily closes for the pre-2018 hack window. No CSV fallback."""
    fetched_at = _now(now)
    try:
        df = fetch_bitstamp_daily(start, end, timeout=timeout)
    except Exception as ex:
        return _fail(
            ["timestamp", "close"], BITSTAMP_URL, fetched_at,
            f"Bitstamp unavailable ({ex})",
        )
    if df is None or len(df) == 0:
        return _fail(
            ["timestamp", "close"], BITSTAMP_URL, fetched_at,
            "Bitstamp returned no daily closes.",
        )
    return FetchResult(
        df, "live", BITSTAMP_URL, fetched_at, _as_of(df, "timestamp"), True,
        f"Bitstamp {len(df)} daily closes ({start} to {end}). In-session only; there is no CSV fallback.",
    )


def fetch_deribit_btc_daily(get_ohlc, start: str = DERIBIT_BTC_DAILY_START,
                            now_ms: int | None = None, now=None) -> FetchResult:
    """Paginate Deribit BTC-PERPETUAL daily closes.

    ``get_ohlc(instrument, resolution, start_ms, end_ms)`` is
    ``lib.deribit.get_tradingview_ohlc``. One thousand daily bars per call,
    matching the page's previous loop. A failure leaves the caller on whatever
    frame it already had; this function does not write a file.
    """
    fetched_at = _now(now)
    origin = "Deribit tradingview BTC-PERPETUAL 1D"
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end = int(time.time() * 1000) if now_ms is None else int(now_ms)
    out = []
    cur = start_ms
    try:
        while cur < end:
            nxt = min(cur + 1000 * 86_400_000, end)
            df = get_ohlc("BTC-PERPETUAL", "1D", cur, nxt)
            if df is not None and len(df):
                out.append(df)
            cur = nxt
    except Exception as ex:
        return _fail(["timestamp", "close"], origin, fetched_at, f"Deribit BTC-PERP fetch failed: {ex}")
    if not out:
        return _fail(["timestamp", "close"], origin, fetched_at, "No BTC-PERP daily candles returned.")
    px = pd.concat(out).drop_duplicates("timestamp")[["timestamp", "close"]]
    return FetchResult(
        px, "live", origin, fetched_at, _as_of(px, "timestamp"), True,
        f"Deribit {len(px)} daily closes. In-session only.",
    )


def _utc_calendar_date(series: pd.Series) -> pd.Series:
    """UTC calendar date as naive datetime64[ns].

    Tz-aware values are converted to UTC first. Naive values are already UTC
    (Deribit tradingview stamps, and Farside dates stored as midnight).
    The unit is pinned so a later join cannot miss on us-vs-ns dtypes.
    """
    t = pd.to_datetime(series, errors="coerce")
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    return t.dt.normalize().astype("datetime64[ns]")


def flows_to_btc(flows: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Convert Farside US$ millions to BTC at that UTC day's close.

    ``px`` is ``DataFrame[timestamp, close]``. Alignment is the UTC calendar
    date of ``timestamp`` (naive timestamps are UTC). A flow session with no
    strictly positive finite close on that date is dropped: there is no
    previous-close carry and no fill. A flow of zero is kept. Non-finite
    flows are dropped. Duplicate price timestamps on one date keep the first
    close (same rule this helper has always used).

    BTC = (US$ millions) × 1,000,000 / close.
    """
    empty = pd.DataFrame(columns=["date", "flow"])
    if flows is None or px is None or len(flows) == 0 or len(px) == 0:
        return empty
    p = pd.DataFrame({
        "date": _utc_calendar_date(px["timestamp"]),
        "close": pd.to_numeric(px["close"], errors="coerce"),
    }).dropna(subset=["date"])
    p = p[np.isfinite(p["close"].to_numpy()) & (p["close"].to_numpy() > 0)]
    if p.empty:
        return empty
    p = p.drop_duplicates("date", keep="first").set_index("date")["close"]
    d = pd.DataFrame({
        "date": _utc_calendar_date(flows["date"]),
        "flow": pd.to_numeric(flows["flow"], errors="coerce"),
    }).dropna(subset=["date"])
    d = d[np.isfinite(d["flow"].to_numpy())]
    if d.empty:
        return empty
    d["close"] = d["date"].map(p)
    d = d.dropna(subset=["close"])
    if d.empty:
        return empty
    d["flow"] = d["flow"] * 1e6 / d["close"]
    return d[["date", "flow"]].reset_index(drop=True)


def streak_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day cumulative flow within each streak: DataFrame[streak, date, cum, sign]."""
    d = df[df["flow"] != 0].reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(columns=["streak", "date", "cum", "sign"])
    sign = np.sign(d["flow"])
    d["streak"] = (sign != sign.shift()).cumsum()
    d["cum"] = d.groupby("streak")["flow"].cumsum()
    d["sign"] = sign
    return d[["streak", "date", "cum", "sign"]]


def sma_gap(px: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    d = px.assign(date=pd.to_datetime(px["timestamp"]).dt.normalize()).set_index("date")[["close"]]
    d["sma"] = d["close"].rolling(window).mean()
    d["gap"] = d["close"] / d["sma"] - 1
    return d.reset_index()


# Close / trailing calendar-day SMA. Separate from sma_gap, which averages the
# last `window` *observations* and so treats a gappy series as continuous.
SMA_CALENDAR_DAYS = 200


def price_sma_ratio(px: pd.DataFrame, window: int = SMA_CALENDAR_DAYS) -> pd.DataFrame:
    """Close divided by its trailing ``window``-calendar-day simple average.

    ``px`` is ``DataFrame[timestamp, close]``. Naive timestamps are UTC.
    The series is reindexed onto every UTC calendar day from the first
    observation to the last. A missing day stays missing: nothing is
    forward-filled, back-filled, interpolated, or invented.

    The average on date T is the mean of the closes on the ``window`` days
    ending at T, and only when every one of those days has a real finite
    close. The first ``window - 1`` days are warmup. A window that contains
    a gap has no average and no ratio, even if ``window`` observations
    exist somewhere in a longer span. Duplicate timestamps on one UTC date
    keep the later timestamp's close. A non-positive average (non-positive
    closes) yields no ratio; the close itself is left as reported.

    No lookahead: date T uses only closes dated on or before T.

    Returns columns ``date, close, sma, ratio`` for every calendar day in
    the span, including warmup and gap rows (those have a null ratio).
    """
    cols = ["date", "close", "sma", "ratio"]
    if window < 1:
        raise ValueError("window must be >= 1")
    if px is None or len(px) == 0:
        return pd.DataFrame(columns=cols)
    d = pd.DataFrame({
        "date": _utc_calendar_date(px["timestamp"]),
        "ts": pd.to_datetime(px["timestamp"], errors="coerce", utc=True),
        "close": pd.to_numeric(px["close"], errors="coerce"),
    }).dropna(subset=["date"])
    d = d[np.isfinite(d["close"].to_numpy())]
    if d.empty:
        return pd.DataFrame(columns=cols)
    d = d.sort_values(["date", "ts"]).groupby("date", as_index=False).tail(1)
    d = d.sort_values("date")
    full = pd.date_range(d["date"].iloc[0], d["date"].iloc[-1], freq="D").astype("datetime64[ns]")
    s = d.set_index("date")["close"].reindex(full)
    # min_periods=window on a calendar-day index: one missing day in the
    # window drops the count below `window`, so the mean stays null.
    # pandas skips NaN inside the average only when the count still qualifies;
    # with min_periods equal to the window length, that never happens.
    sma = s.rolling(window, min_periods=window).mean()
    ratio = s / sma
    ratio = ratio.where(sma > 0)
    return pd.DataFrame({
        "date": full,
        "close": s.to_numpy(),
        "sma": sma.to_numpy(),
        "ratio": ratio.to_numpy(),
    })


def sma_ratio_methodology() -> str:
    """User-facing description of ``price_sma_ratio``. Kept next to the code."""
    w = SMA_CALENDAR_DAYS
    return f"""**What is plotted.** The daily Deribit BTC-PERPETUAL close divided by its trailing {w}-calendar-day simple moving average. Each session is coloured by that day's BTC close (navy when the close is low in the window, red when it is high). The colour scale stretches across the closes in the displayed window only. It does not enter the ratio.

**Price series.** Deribit BTC-PERPETUAL daily closes from the tradingview endpoint, the same series as the "BTC vs 200DMA" tab. Naive timestamps are UTC. Bitstamp spot is not spliced on: mixing spot into the perpetual would put two markets inside one average. History starts in August 2018, so the first ratio is about {w} calendar days later, once a complete window exists.

**Window and warmup.** The average on date T is the arithmetic mean of the closes on the {w} UTC calendar days from T through the preceding {w - 1} days, including T. The first {w - 1} calendar days are warmup and have no ratio. A ratio is emitted only when all {w} days have a real close.

**Missing data.** A day with no close stays blank. Neighbouring closes are not copied forward or back, and the gap is not interpolated. A {w}-day window that contains a blank day has no average and no ratio, even if {w} closes exist over a longer span. Two bars on the same UTC date keep the later bar's close. A non-positive close produces no ratio.

**No lookahead.** The average on date T uses only closes dated on or before T.

**Timezone.** The session date is the UTC calendar date of the bar timestamp. Deribit daily tradingview bars are stamped at 08:00 UTC, which is 16:00 SGT on that same calendar date, so the date on the axis matches both clocks. A bar is not rolled onto the next SGT date. Clocks elsewhere on this page use SGT via the shared timezone helpers.
"""


# ------------------------------------------------------------------ ETF regime
# Independent approximation. The Ecoinometrics regime formula is unpublished;
# these constants were not fitted to their chart.
# A US session week is 5 sessions. The scale is the trailing median absolute
# flow over 63 sessions (a quarter of a 252-session year, and odd so the
# median is a single observation). "Mild" is two typical days of net flow
# over that week, so one ordinary print stays neutral. "Strong" is five
# typical days of one-way net flow.
FLOW_REGIME_WINDOW = 5
FLOW_REGIME_SCALE = 63
FLOW_REGIME_MILD = 2.0
FLOW_REGIME_STRONG = 5.0
FLOW_REGIMES = (
    "strong_inflow",
    "inflow",
    "neutral",
    "outflow",
    "strong_outflow",
    "unclassified",
)


def classify_flow_regime(sum_window: float, scale_median: float,
                         mild: float = FLOW_REGIME_MILD,
                         strong: float = FLOW_REGIME_STRONG) -> str:
    """Label one session from its trailing sum and trailing median |flow|.

    Both inputs must already be causal (computed from sessions on or before
    this one). A null input is ``unclassified`` (warmup), which is distinct
    from ``neutral``. When the scale median is 0, "strong" is not used.
    """
    if pd.isna(sum_window) or pd.isna(scale_median):
        return "unclassified"
    if not (strong >= mild > 0):
        raise ValueError("need strong >= mild > 0")
    if scale_median == 0:
        if sum_window > 0:
            return "inflow"
        if sum_window < 0:
            return "outflow"
        return "neutral"
    if sum_window >= strong * scale_median:
        return "strong_inflow"
    if sum_window <= -strong * scale_median:
        return "strong_outflow"
    if sum_window >= mild * scale_median:
        return "inflow"
    if sum_window <= -mild * scale_median:
        return "outflow"
    return "neutral"


def etf_flow_regime(df: pd.DataFrame,
                    window: int = FLOW_REGIME_WINDOW,
                    scale: int = FLOW_REGIME_SCALE,
                    mild: float = FLOW_REGIME_MILD,
                    strong: float = FLOW_REGIME_STRONG) -> pd.DataFrame:
    """Causal regime for a daily BTC-flow series.

    ``df`` has columns ``date`` and ``flow`` (BTC, already converted).
    Rows with a null date or a non-finite flow are dropped and are not
    treated as zero. A flow of zero is a real session and stays in both
    windows. Duplicate dates are summed into one session.

    ``sum_window`` is the trailing sum of the last ``window`` remaining
    sessions (this session included). ``scale_median`` is the trailing
    median of absolute flow over the last ``scale`` remaining sessions.
    Both are right-aligned, so session t does not see a later session.
    Because missing sessions were dropped rather than filled with zero,
    the window is "last N observations", which can cover more than N
    calendar days when the tape has a hole.

    The regime stays ``unclassified`` until both windows are full. With
    the default scale of 63, the 63rd aligned session is the first label.

    Also returns ``mild_cutoff`` and ``strong_cutoff`` (± multiples of the
    scale median) so a reader can check the label by hand. When the scale
    median is 0 those cutoffs are 0 and the zero-scale rule applies instead.
    """
    cols = ["date", "flow", "sum_window", "scale_median",
            "mild_cutoff", "strong_cutoff", "regime"]
    if window < 1 or scale < 1:
        raise ValueError("window and scale must be >= 1")
    if not (strong >= mild > 0):
        raise ValueError("need strong >= mild > 0")
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)
    d = pd.DataFrame({
        "date": _utc_calendar_date(df["date"]),
        "flow": pd.to_numeric(df["flow"], errors="coerce"),
    }).dropna(subset=["date"])
    d = d[np.isfinite(d["flow"].to_numpy())]
    if d.empty:
        return pd.DataFrame(columns=cols)
    d = d.groupby("date", as_index=False)["flow"].sum().sort_values("date")
    flow = d["flow"]
    # center=False is the pandas default: the window ends at this row.
    d["sum_window"] = flow.rolling(window, min_periods=window).sum()
    d["scale_median"] = flow.abs().rolling(scale, min_periods=scale).median()
    d["mild_cutoff"] = d["scale_median"] * mild
    d["strong_cutoff"] = d["scale_median"] * strong
    d["regime"] = [
        classify_flow_regime(s, m, mild=mild, strong=strong)
        for s, m in zip(d["sum_window"], d["scale_median"])
    ]
    return d.reset_index(drop=True)


def flow_regime_methodology() -> str:
    """User-facing description of ``etf_flow_regime``. Kept next to the code."""
    w, s = FLOW_REGIME_WINDOW, FLOW_REGIME_SCALE
    mild, strong = FLOW_REGIME_MILD, FLOW_REGIME_STRONG
    return f"""**This is not the Ecoinometrics model.** Their flow-regime formula is unpublished. The colours on this tab are MCM's independent trailing-flow regime, recomputed from public data. The cutoffs were not fitted to their chart.

**What is plotted.** One bar per US session: the Farside "Total" of all US spot BTC ETFs, converted to BTC. The bar colour is the regime as of that session. The bar height is that session's own flow, not the multi-session sum that sets the colour.

**Unit conversion.** Farside totals are US$ millions. BTC flow = US$ millions × 1,000,000 / the Deribit BTC-PERPETUAL close on the same UTC calendar date. The close is the perpetual, not a spot index, so the unit is perp-denominated BTC. The ETF streak tab uses this same conversion. A session with no strictly positive finite close that day is dropped. The previous close is not carried forward. A published flow of zero stays zero BTC.

**Regime, with no lookahead.** Let F be the session's BTC flow. S is the sum of F over this session and the {w - 1} aligned sessions before it ({w} sessions, one US trading week). M is the median of absolute F over this session and the {s - 1} aligned sessions before it. {s} is a quarter of a 252-session year, and it is odd so the median is one observation rather than the average of two. Both windows end on the session being labelled and contain only sessions dated on or before it.

- Strong inflow when S is greater than or equal to {strong:g} × M
- Inflow when S is greater than or equal to {mild:g} × M and below the strong cutoff
- Neutral when S is strictly inside ±{mild:g} × M
- Outflow when S is less than or equal to −{mild:g} × M and above the strong-outflow cutoff
- Strong outflow when S is less than or equal to −{strong:g} × M

Touching {strong:g} × M is strong. Touching {mild:g} × M is the mild band. A sum of exactly zero is neutral.

**Why these cutoffs.** {mild:g} × M is about two typical days of net flow over the week, so a single ordinary print stays neutral. {strong:g} × M is about a full week of typical one-way flow. The {s}-session median moves slowly and sets the scale. The {w}-session sum moves quickly and sets the state.

**Zero scale.** When M is 0, every session in the scale window printed zero. A positive S is then inflow, a negative S is outflow, and a zero S is neutral. Strong is not used, because there is no non-zero typical day to scale it against.

**Warmup.** S needs {w} sessions and M needs {s}. Until both exist the session is unclassified, shown in grey, and is not called neutral. The first labelled session is aligned session number {s}.

**Missing data.** A missing Farside total, a non-finite flow, or a missing price is dropped before either window. It is not written in as a zero, and it does not count as a session. The {w}-session window is then the last {w} sessions that remain, so a hole in the tape can make those sessions span more than {w} calendar days. A published zero is a real session and does count. Two flow rows on the same date are summed.
"""


def display_from(df: pd.DataFrame, start) -> pd.DataFrame:
    """Rows with ``date`` on or after ``start`` (UTC calendar date).

    Call this only after the ratio or the regime has been computed on the
    full history. Slicing first would treat the left edge of the window as
    warmup and would drop real closes from the 200-day average.
    """
    if df is None or len(df) == 0:
        return df
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_convert("UTC").tz_localize(None)
    start_ts = pd.Timestamp(start_ts).normalize()
    out = df.loc[pd.to_datetime(df["date"]) >= start_ts].reset_index(drop=True)
    return out


def rolling_corr(btc: pd.DataFrame, other: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """Rolling correlation of daily log returns, on `other`'s trading days. other: [date, value]."""
    b = btc.assign(date=pd.to_datetime(btc["timestamp"]).dt.normalize()).set_index("date")["close"]
    o = other.set_index("date")["value"]
    j = pd.concat([b.rename("btc"), o.rename("oth")], axis=1, join="inner").dropna()
    r = np.log(j).diff().dropna()
    c = r["btc"].rolling(window).corr(r["oth"]).dropna()
    return c.rename("corr").reset_index()

# ------------------------------------------------------------- Exchange hacks
def load_hacks_result(now=None) -> FetchResult:
    """Hand-curated hack list. There is no live feed; a refresh re-reads the file."""
    path = DATA_DIR / "exchange_hacks.csv"
    origin = "data/exchange_hacks.csv"
    fetched_at = file_mtime(path) or _now(now)
    if not path.exists():
        return _fail(["date", "exchange", "loss_usd_m", "note"], origin, fetched_at,
                     "data/exchange_hacks.csv is missing.")
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception as ex:
        return _fail(["date", "exchange", "loss_usd_m", "note"], origin, fetched_at,
                     f"data/exchange_hacks.csv could not be read ({ex})")
    return FetchResult(
        df, "bundled", origin, fetched_at, _as_of(df, "date"), True,
        "Hand-curated list in data/exchange_hacks.csv. There is no live feed.",
    )


def load_hacks() -> pd.DataFrame:
    result = load_hacks_result()
    if not result.ok:
        raise RuntimeError(result.message)
    return result.df


def hack_event_returns(prices: pd.DataFrame, hacks: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Return from close of hack day to close `horizon` days later.
    prices: DataFrame[timestamp, close] daily. Events outside price history dropped."""
    px = prices.assign(d=pd.to_datetime(prices["timestamp"]).dt.normalize()).set_index("d")["close"]
    rows = []
    for _, h in hacks.iterrows():
        d0, d1 = h["date"].normalize(), h["date"].normalize() + pd.Timedelta(days=horizon)
        if d0 in px.index and d1 in px.index:
            rows.append({**h.to_dict(), "ret": px[d1] / px[d0] - 1})
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ Fed tone
HAWK = [r"inflation", r"restrictive", r"tighten", r"\bhike", r"raise (the )?(rate|policy)",
        r"upside risk", r"elevated", r"persistent", r"overheat", r"strong(er)? (growth|labor|demand)",
        r"further (firming|increase)", r"above (our )?(2 percent|target)"]
DOVE = [r"\bcut", r"easing", r"accommodat", r"downside risk", r"weak(er|ening)? (labor|growth|demand)",
        r"softening", r"slowing", r"lower (the )?(rate|policy)", r"disinflation", r"unemployment (has )?(risen|rising)",
        r"restrain(t|ing) (growth|activity)"]
_H = [re.compile(p, re.I) for p in HAWK]
_D = [re.compile(p, re.I) for p in DOVE]


def chair_text(full: str) -> str:
    """Keep only the Chair's speaking turns (heuristic on 'CHAIR NAME.' labels)."""
    parts = re.split(r"\n\s*(?=[A-Z][A-Za-z.' -]{2,40}\.\s)", full)
    keep = [p for p in parts if re.match(r"\s*(CHAIR|Chair)\b", p)]
    return "\n".join(keep) if keep else full


def tone_score(text: str) -> dict:
    h = sum(len(p.findall(text)) for p in _H)
    d = sum(len(p.findall(text)) for p in _D)
    n = len(text.split())
    return {"hawk": h, "dove": d, "words": n,
            "score": (h - d) / (h + d) if h + d else 0.0,
            "net_per_1k": 1000 * (h - d) / n if n else 0.0}


def with_tone_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add the chart's 0-centred index and six-meeting average. Does not fetch."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "score", "idx", "avg6"])
    out = df.sort_values("date").copy()
    out["idx"] = out["score"] * 100
    out["avg6"] = out["idx"].rolling(6, min_periods=1).mean()
    return out.reset_index(drop=True)


def load_fed_tone_result(now=None) -> FetchResult:
    """Read ``data/fed_tone.csv`` only. Transcript downloads happen in ``build_fed_tone``."""
    path = DATA_DIR / "fed_tone.csv"
    origin = "data/fed_tone.csv"
    fetched_at = file_mtime(path) or _now(now)
    if not path.exists():
        return _fail(
            ["date", "score", "idx", "avg6"], origin, fetched_at,
            "No data/fed_tone.csv yet.",
        )
    try:
        raw = pd.read_csv(path, parse_dates=["date"])
    except Exception as ex:
        return _fail(
            ["date", "score", "idx", "avg6"], origin, fetched_at,
            f"data/fed_tone.csv could not be read ({ex})",
        )
    df = with_tone_index(raw)
    return FetchResult(
        df, "csv", origin, fetched_at, _as_of(df, "date"), True,
        f"Fed tone from data/fed_tone.csv ({len(df)} meetings).",
    )


def load_fed_tone() -> pd.DataFrame:
    result = load_fed_tone_result()
    if not result.ok:
        return pd.DataFrame(columns=["date", "score", "idx", "avg6"])
    return result.df


def add_fed_funds(ft: pd.DataFrame, dff: pd.DataFrame) -> pd.DataFrame:
    """Attach the effective fed funds rate (FRED DFF) on/before each meeting date."""
    m = pd.merge_asof(ft.sort_values("date"), dff.rename(columns={"value": "ffr"}).sort_values("date"), on="date")
    return m
