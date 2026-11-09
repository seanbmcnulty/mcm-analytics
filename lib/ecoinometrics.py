"""
Ecoinometrics-style charts — pure data/analytics helpers (no Streamlit).

DELIBERATE EXCEPTIONS to "Deribit-only" (requested by the user 2026-11-01):
  * US spot BTC ETF daily flows  -> Farside (scraped HTML) or data/etf_flows.csv
  * Fed press-conference tone    -> data/fed_tone.csv built by tools/build_fed_tone.py
    from federalreserve.gov transcripts (dictionary score, NOT Ecoinometrics' index)
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
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FARSIDE_URLS = ("https://farside.co.uk/btc/", "https://farside.co.uk/bitcoin-etf-flow-all-data/")
UA = {"User-Agent": "Mozilla/5.0 (compatible; mcm-analytics)"}

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


def fetch_etf_flows(timeout: int = 20) -> pd.DataFrame:
    """Try Farside; fall back to data/etf_flows.csv (date,flow in US$ m)."""
    err = None
    for url in FARSIDE_URLS:
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return parse_farside_html(r.text)
        except Exception as e:  # blocked / format change
            err = e
    p = DATA_DIR / "etf_flows.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"])
        return df[["date", "flow"]].sort_values("date").reset_index(drop=True)
    raise RuntimeError(f"Farside unavailable ({err}) and no data/etf_flows.csv")


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
def fred_series(series_id: str, timeout: int = 15, start: str = "2013-01-01", retries: int = 2,
                live_only: bool = False) -> pd.DataFrame:
    """FRED public CSV (no key). Returns DataFrame[date, value].
    Falls back to data/fred_<ID>.csv (kept fresh by tools/build_fred_cache.py + GH Action)
    because Streamlit Cloud frequently times out reaching FRED."""
    err = None
    for _ in range(retries):
        try:
            r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                             params={"id": series_id, "cosd": start}, headers=UA, timeout=timeout)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            err = e
    else:
        p = DATA_DIR / f"fred_{series_id}.csv"
        if live_only or not p.exists():
            raise err
        df = pd.read_csv(p, parse_dates=["date"])
        return df.dropna().reset_index(drop=True)
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def fetch_bitstamp_daily(start: str, end: str, timeout: int = 20) -> pd.DataFrame:
    """Daily BTC/USD closes from Bitstamp (history to 2011) -> DataFrame[timestamp, close].
    Used only for hack events before Deribit's perp history (Aug 2018)."""
    out, cur, end_ts = [], int(pd.Timestamp(start, tz="UTC").timestamp()), int(pd.Timestamp(end, tz="UTC").timestamp())
    while cur < end_ts:
        r = requests.get("https://www.bitstamp.net/api/v2/ohlc/btcusd/", headers=UA, timeout=timeout,
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
def load_hacks() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "exchange_hacks.csv", parse_dates=["date"])


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


def load_fed_tone() -> pd.DataFrame:
    p = DATA_DIR / "fed_tone.csv"
    if not p.exists():
        return pd.DataFrame(columns=["date", "score", "idx", "avg6"])
    df = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
    df["idx"] = df["score"] * 100
    df["avg6"] = df["idx"].rolling(6, min_periods=1).mean()
    return df.reset_index(drop=True)


def add_fed_funds(ft: pd.DataFrame, dff: pd.DataFrame) -> pd.DataFrame:
    """Attach the effective fed funds rate (FRED DFF) on/before each meeting date."""
    m = pd.merge_asof(ft.sort_values("date"), dff.rename(columns={"value": "ffr"}).sort_values("date"), on="date")
    return m
