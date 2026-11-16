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
FRED_SERIES = ("DFF", "NASDAQCOM", "CPIAUCSL", "DGS10", "NASDAQ100")
# Hack events before Deribit's perp history (Aug 2018) use Bitstamp spot.
# 2013 start so the drawdown study sees the Nov/Dec 2013 peak and the
# Nasdaq down-day study has all of 2014. The earliest hack is Jan 2015.
BITSTAMP_START = "2013-01-01"
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


def _hack_bar_dates(series: pd.Series) -> pd.Series:
    """UTC calendar date of each timestamp, as naive midnight.

    Naive values are already UTC (Deribit tradingview stamps, Bitstamp daily
    stamps, and the dates in ``exchange_hacks.csv``). Tz-aware values are
    converted to UTC before the date is taken, so a bar at 20:00 UTC does not
    roll onto the next Asia/Singapore calendar date.
    """
    t = pd.to_datetime(series, errors="coerce")
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    return t.dt.normalize()


def hack_event_returns(prices: pd.DataFrame, hacks: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Forward return from the event-date daily close over ``horizon`` calendar days.

    ``prices`` is ``DataFrame[timestamp, close]``, one row per daily bar. The
    bar is keyed by the UTC calendar date of its timestamp. The start price is
    the **close of the bar stamped on the event date**, not the next calendar
    date's bar and not the previous day's close.

    That matches a late-UTC-evening hack. Deribit 1D bars are stamped at
    08:00 UTC and close at 08:00 UTC the next morning, so a theft at 20:05 UTC
    (Binance, 29 Oct 2026) or 18:31 UTC (Bitget, 24 Sep 2026) falls inside the
    bar stamped that date. Its close is the first daily settlement after the
    theft, about 12 hours later. Bitstamp daily bars (pre-2018 only) are
    stamped at 00:00 UTC and close at the end of that UTC date, so the same
    rule is the first settlement after an evening event, a few hours later.
    Rolling the anchor forward an extra day would skip that settlement.

    The end price is the close of the bar stamped ``horizon`` calendar days
    after the event date. That exact date has to be in the series.

    * Both bars present: ``complete`` is True and ``ret`` is the full-window
      return. ``partial`` equals ``ret``.
    * The end date is after the last bar (the window is not finished):
      ``complete`` is False, ``ret`` is NaN so it cannot enter an average,
      and ``partial`` is the return from the anchor close to the latest bar
      strictly after the anchor. ``elapsed`` is how many calendar days that
      partial window actually covers. The chart draws these as open circles.
    * The end date is missing while a later bar exists (a hole, not an
      unfinished window): the event is dropped. A nearby close is not used
      as a stand-in for the horizon.
    * No anchor bar, a non-finite or zero anchor close, or no close after
      the anchor: the event is dropped.

    A still-forming last bar is used as-is once it is in the series. It only
    becomes the end of a *complete* window when the horizon date is that bar.
    """
    if prices is None or hacks is None or len(prices) == 0 or len(hacks) == 0:
        return pd.DataFrame()
    ts = pd.to_datetime(prices["timestamp"], utc=True, errors="coerce")
    bars = pd.DataFrame({
        "d": _hack_bar_dates(prices["timestamp"]),
        "close": pd.to_numeric(prices["close"], errors="coerce"),
        "ts": ts,
    }).dropna(subset=["d"])
    bars = bars[np.isfinite(bars["close"].to_numpy())]
    if bars.empty:
        return pd.DataFrame()
    bars = bars.sort_values("ts").drop_duplicates("d", keep="last")
    px = bars.set_index("d")["close"].sort_index()
    last = px.index.max()
    horizon = int(horizon)
    rows = []
    for rec, d0 in zip(hacks.to_dict("records"), _hack_bar_dates(hacks["date"])):
        if pd.isna(d0) or d0 not in px.index:
            continue
        start = float(px.loc[d0])
        if not np.isfinite(start) or start == 0:
            continue
        after = px.loc[px.index > d0]
        if after.empty:
            continue
        d1 = d0 + pd.Timedelta(days=horizon)
        if d1 in px.index and np.isfinite(float(px.loc[d1])):
            ret = float(px.loc[d1]) / start - 1.0
            rec.update(ret=ret, partial=ret, complete=True, elapsed=horizon, anchor=d0)
        elif d1 > last:
            end_dt = after.index.max()
            rec.update(
                ret=np.nan,
                partial=float(after.loc[end_dt]) / start - 1.0,
                complete=False,
                elapsed=int((end_dt - d0).days),
                anchor=d0,
            )
        else:
            continue
        rows.append(rec)
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


# ------------------------------------------- added 2026-11-16 (13 Nov issue)
# Nasdaq panel, 30-session flow line, BTC on Nasdaq down days, drawdown
# duration vs depth. Strategy's 8-K breakdown lives in lib/strategy_8k.py.
def sma_rows(series: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """FRED-style ``DataFrame[date, value]`` plus a ``window``-observation SMA.

    FRED equity indices only print on trading days, so 200 rows is a
    200-trading-day average (about 40 calendar weeks). Blank until full.
    """
    d = series[["date", "value"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
    d["sma"] = d["value"].rolling(window, min_periods=window).mean()
    d["gap"] = d["value"] / d["sma"] - 1
    return d


def rolling_flow_with_price(flows_btc: pd.DataFrame, px: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """``flows_to_btc`` output -> DataFrame[date, flow, roll, close].

    ``roll`` is the sum of the last ``window`` aligned Farside sessions
    (blank until ``window`` sessions exist). ``close`` is the Deribit close on
    that UTC date, the same close the dollar flow was converted at.
    """
    cols = ["date", "flow", "roll", "close"]
    if flows_btc is None or px is None or len(flows_btc) == 0 or len(px) == 0:
        return pd.DataFrame(columns=cols)
    d = flows_btc[["date", "flow"]].copy()
    d["date"] = _utc_calendar_date(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    d["roll"] = d["flow"].rolling(window, min_periods=window).sum()
    p = pd.DataFrame({"date": _utc_calendar_date(px["timestamp"]),
                      "close": pd.to_numeric(px["close"], errors="coerce")})
    p = p.dropna().drop_duplicates("date", keep="first").set_index("date")["close"]
    d["close"] = d["date"].map(p)
    return d[cols]


def daily_closes(*frames: pd.DataFrame) -> pd.Series:
    """Splice ``DataFrame[timestamp, close]`` frames into one close per UTC date.

    Later frames win on an overlapping date (pass Bitstamp first, Deribit
    second). Non-finite or non-positive closes are dropped, never filled.
    """
    parts = []
    for rank, f in enumerate(frames):
        if f is None or len(f) == 0:
            continue
        parts.append(pd.DataFrame({
            "date": _utc_calendar_date(f["timestamp"]),
            "close": pd.to_numeric(f["close"], errors="coerce"),
            "rank": rank,
        }))
    if not parts:
        return pd.Series(dtype=float)
    d = pd.concat(parts).dropna(subset=["date"])
    d = d[np.isfinite(d["close"].to_numpy()) & (d["close"].to_numpy() > 0)]
    d = d.sort_values(["date", "rank"]).drop_duplicates("date", keep="last")
    return d.set_index("date")["close"].sort_index()


NASDAQ_BUCKETS = ((-np.inf, -8.0), (-8.0, -6.0), (-6.0, -4.0), (-4.0, -2.0), (-2.0, 0.0))
NASDAQ_BUCKET_LABELS = ("Below −8%", "−8% to −6%", "−6% to −4%", "−4% to −2%", "−2% to 0%")


def nasdaq_bucket(ret_pct: float) -> str | None:
    """Bucket label for a Nasdaq daily % return. ``lo <= r < hi``; 0% and up is None."""
    if ret_pct is None or not np.isfinite(ret_pct) or ret_pct >= 0:
        return None
    for (lo, hi), label in zip(NASDAQ_BUCKETS, NASDAQ_BUCKET_LABELS):
        if lo <= ret_pct < hi:
            return label
    return None


def btc_on_nasdaq_down_days(ndx: pd.DataFrame, btc_close: pd.Series, start: str = "2014-01-01") -> pd.DataFrame:
    """Same-date BTC daily return on every Nasdaq-100 down day.

    ``ndx`` is FRED ``DataFrame[date, value]``; its return is close-to-close
    over consecutive FRED prints (Friday to Monday over a weekend).
    ``btc_close`` is one close per UTC date (``daily_closes``). The BTC return
    for Nasdaq date D is close(D) / close(D - 1 calendar day) - 1, using the
    bar stamped D. Both of those bars must exist; nothing is carried.
    """
    cols = ["date", "ndx_ret", "btc_ret", "bucket"]
    if ndx is None or len(ndx) == 0 or btc_close is None or len(btc_close) == 0:
        return pd.DataFrame(columns=cols)
    n = ndx[["date", "value"]].copy()
    n["date"] = pd.to_datetime(n["date"]).dt.normalize()
    n = n.dropna().sort_values("date").drop_duplicates("date")
    n["ndx_ret"] = n["value"].pct_change() * 100
    n = n[n["date"] >= pd.Timestamp(start)]
    prev = btc_close.copy()
    prev.index = prev.index + pd.Timedelta(days=1)
    n["btc_ret"] = (n["date"].map(btc_close) / n["date"].map(prev) - 1) * 100
    n["bucket"] = n["ndx_ret"].map(nasdaq_bucket)
    n = n.dropna(subset=["ndx_ret", "btc_ret", "bucket"])
    return n[cols].reset_index(drop=True)


def bucket_stats(df: pd.DataFrame) -> pd.DataFrame:
    """n, median and mean BTC return per bucket, in bucket order (empty buckets dropped)."""
    rows = []
    for label in NASDAQ_BUCKET_LABELS:
        g = df.loc[df["bucket"] == label, "btc_ret"] if len(df) else pd.Series(dtype=float)
        if len(g):
            rows.append({"bucket": label, "n": int(len(g)), "median": float(g.median()),
                         "mean": float(g.mean()), "share_down": float((g < 0).mean())})
    return pd.DataFrame(rows, columns=["bucket", "n", "median", "mean", "share_down"])


DRAWDOWN_FROM = "2014-01-01"


def drawdown_episodes(close: pd.Series, min_depth: float = 0.20, active_from: str = DRAWDOWN_FROM) -> pd.DataFrame:
    """Peak-to-recovery drawdowns of a daily close series.

    An episode starts at a running all-time-high close (``peak``) when the
    next close is below it, and ends on the first later close at or above
    that peak (``recovered``). ``depth`` is the lowest close in between over
    the peak, minus one (negative). ``days`` is peak to recovery in calendar
    days; for the open episode it is peak to the latest close and
    ``ongoing`` is True. Episodes shallower than ``min_depth`` are dropped,
    as are episodes that recovered before ``active_from``.
    """
    cols = ["peak_date", "peak", "trough_date", "trough", "recovered", "days", "depth", "ongoing"]
    s = close.dropna().sort_index()
    s = s[s > 0]
    if len(s) < 2:
        return pd.DataFrame(columns=cols)
    vals, dates = s.to_numpy(dtype=float), s.index
    rows = []
    peak_i, trough_i, in_dd = 0, 0, False
    for i in range(1, len(vals)):
        if vals[i] >= vals[peak_i]:
            if in_dd:
                rows.append((peak_i, trough_i, i))
                in_dd = False
            peak_i = i
        else:
            if not in_dd:
                in_dd, trough_i = True, i
            elif vals[i] < vals[trough_i]:
                trough_i = i
    if in_dd:
        rows.append((peak_i, trough_i, None))
    out = []
    last = dates[-1]
    for p, t, r in rows:
        depth = vals[t] / vals[p] - 1
        end = dates[r] if r is not None else last
        out.append({
            "peak_date": dates[p], "peak": vals[p], "trough_date": dates[t], "trough": vals[t],
            "recovered": dates[r] if r is not None else pd.NaT,
            "days": int((end - dates[p]).days), "depth": float(depth), "ongoing": r is None,
        })
    df = pd.DataFrame(out, columns=cols)
    df = df[df["depth"] <= -abs(min_depth)]
    keep = df["ongoing"] | (pd.to_datetime(df["recovered"]) >= pd.Timestamp(active_from))
    return df[keep].reset_index(drop=True)


def drawdown_trend(dd: pd.DataFrame) -> tuple[float, float] | None:
    """Least-squares depth (%) = a + b * log10(days) on finished episodes. None if fewer than 3."""
    done = dd[~dd["ongoing"].astype(bool)] if len(dd) else dd
    if len(done) < 3:
        return None
    x = np.log10(done["days"].clip(lower=1).to_numpy(dtype=float))
    y = done["depth"].to_numpy(dtype=float) * 100
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)
