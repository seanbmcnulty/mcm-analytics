"""The Daily Shot chart collection: data, maths and the chart registry. No Streamlit here.

Sean asked (27 Nov 2026) for our own copy of the charts in The Daily Shot
(https://thedailyshot.com/, Lev Borodovsky's macro newsletter; issues come
from Editor@thedailyshot.com when subscribed). Issues are posted on the site
every US business day, but the body is behind the subscriber login: the
public page carries only the headline (the lead story) and the section list.
So each chart here rebuilds the issue's headline story (or a section item
named in the public excerpt) from free public data. Nothing of The Daily
Shot's is stored in this public repo: each registry entry carries just the
issue headline, date and link, and our own one-line description.

Adding a chart from a new issue
-------------------------------
1. Check ``has_issue(date, title)`` so a re-run never adds an issue twice
   (key "YYYY-MM-DD | headline", the issue date as posted, US Eastern).
2. Write the maths here (pure functions on pandas objects) and a builder in
   ``lib.dailyshot_charts`` (``BUILDERS[key]``).
3. New Yahoo symbol -> ``YAHOO_SYMBOLS`` (+ ``SYMBOL_LABELS``); new FRED id ->
   ``FRED_SERIES``. Then refresh the committed fallbacks:
   ``python3 -c "from lib import dailyshot as d; d.write_prices_csv(); d.write_fred_csv()"``
4. Put a ``DailyShotChart`` at the TOP of ``REGISTRY`` (newest first), with
   ``section`` set to the issue section it belongs to, and add the issue's
   public section list (the "•" line in the excerpt, e.g. from
   https://thedailyshot.com/wp-json/wp/v2/posts?per_page=5&_fields=date,title,link,excerpt)
   to ``ISSUE_SECTIONS``. The page is one scrolling newsletter: issues newest
   first, sections in the issue's order, charts stacked under each section.

Data
----
* Yahoo Finance daily closes (chart API, as ``lib.tenx``), live, with the
  committed ``data/dailyshot_prices.csv`` (real Yahoo closes) for anything
  live misses.
* FRED monthly / quarterly series, live with one short attempt (FRED times
  out from Streamlit Cloud), else ``data/dailyshot_fred.csv`` (real FRED
  values).
* Treasury yields via ``lib.eco_more.fetch_yield_curve_result`` (FRED, then
  home.treasury.gov, then ``data/yield_curve.csv``).
Nothing is filled or invented: a missing day stays missing.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_CSV = "dailyshot_prices.csv"
FRED_CSV = "dailyshot_fred.csv"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_UA = {"User-Agent": "Mozilla/5.0 (compatible; mcm-analytics)"}  # FRED stalls on a browser UA
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}
SHORT_START = "2024-01-01"
SECTOR_START = "1998-12-22"   # first SPDR sector ETF bar
DOW_START = "1992-01-02"      # first ^DJI bar on Yahoo

# column -> (Yahoo symbol, history start)
YAHOO_SYMBOLS = {
    "SPX": ("^GSPC", SHORT_START), "IXIC": ("^IXIC", SHORT_START), "NDX": ("^NDX", SHORT_START),
    "NVDA": ("NVDA", SHORT_START), "ANT": ("ANT", "2026-11-01"),
    "XLP": ("XLP", SECTOR_START), "XLU": ("XLU", SECTOR_START), "XLV": ("XLV", SECTOR_START),
    "XLK": ("XLK", SECTOR_START), "XLY": ("XLY", SECTOR_START), "XLF": ("XLF", SECTOR_START),
    "XLI": ("XLI", SECTOR_START), "XLE": ("XLE", SHORT_START), "XLB": ("XLB", SHORT_START),
    "XLRE": ("XLRE", SHORT_START), "XLC": ("XLC", SHORT_START),
    "DJI": ("^DJI", DOW_START),
    "EWZ": ("EWZ", SHORT_START), "BVSP": ("^BVSP", SHORT_START), "USDBRL": ("BRL=X", SHORT_START),
}
SYMBOL_LABELS = {
    "SPX": "S&P 500", "IXIC": "Nasdaq Composite", "NDX": "Nasdaq 100", "NVDA": "Nvidia",
    "ANT": "Anthropic (ANT)", "XLP": "Staples (XLP)", "XLU": "Utilities (XLU)", "XLV": "Health care (XLV)",
    "XLK": "Technology (XLK)", "XLY": "Discretionary (XLY)", "XLF": "Financials (XLF)",
    "XLI": "Industrials (XLI)", "XLE": "Energy (XLE)", "XLB": "Materials (XLB)",
    "XLRE": "Real estate (XLRE)", "XLC": "Communications (XLC)", "DJI": "Dow Jones Industrial Average",
    "EWZ": "iShares MSCI Brazil (EWZ)", "BVSP": "Ibovespa", "USDBRL": "USD/BRL",
}
FRED_SERIES = {
    "UMCSENT": "University of Michigan consumer sentiment",
    "A191RL1Q225SBEA": "Real GDP, % q/q annualised (BEA)",
    "CPIAUCSL": "CPI, all items (BLS)",
    "CPILFESL": "Core CPI, ex food and energy (BLS)",
}
DEFENSIVES = ("XLP", "XLU", "XLV")
CYCLICALS = ("XLK", "XLY", "XLF", "XLI")
SECTORS = ("XLK", "XLC", "XLY", "XLF", "XLI", "XLB", "XLE", "XLRE", "XLP", "XLU", "XLV")
# US midterm election days (Tuesday) that ^DJI history covers
MIDTERMS = ("1994-11-08", "1998-11-03", "2002-11-05", "2006-11-07", "2010-11-02",
            "2014-11-04", "2018-11-06", "2022-11-08", "2026-11-03")


# ------------------------------------------------------------------ registry
@dataclass(frozen=True)
class DailyShotChart:
    key: str               # stable id, maps to a builder in lib.dailyshot_charts
    tab: str               # short tab label
    section: str           # Daily Shot section the chart belongs to
    issue_title: str       # the issue's public headline
    issue_date: str        # issue date as posted (US Eastern), YYYY-MM-DD
    issue_url: str
    summary: str           # our own one-line description
    series: tuple = field(default_factory=tuple)   # Yahoo columns it needs
    fred: tuple = field(default_factory=tuple)     # FRED ids it needs
    yields: bool = False                           # needs the Treasury curve

    @property
    def issue_id(self) -> str:
        return issue_id(self.issue_date, self.issue_title)


_U = "https://thedailyshot.com/2026/"
REGISTRY: tuple[DailyShotChart, ...] = (
    DailyShotChart(
        "nvda_rebound", "Nvidia rebound", "Equities",
        "Nvidia’s rebound lifts the broader market", "2026-11-24",
        _U + "11/24/nvidias-rebound-lifts-the-broader-market/",
        "Nvidia and the S&P 500 since September, rebased to 100, with the latest day's move in each.",
        series=("NVDA", "SPX")),
    DailyShotChart(
        "nasdaq_ytd", "Nasdaq YTD", "Equities",
        "The Nasdaq erased all its gains for the year", "2026-11-23",
        _U + "11/23/the-nasdaq-erased-all-its-gains-for-the-year/",
        "Nasdaq Composite and Nasdaq 100 year-to-date return: the 2026 gain is gone.",
        series=("IXIC", "NDX")),
    DailyShotChart(
        "umich", "UMich sentiment", "United States",
        "The Nasdaq erased all its gains for the year", "2026-11-23",
        _U + "11/23/the-nasdaq-erased-all-its-gains-for-the-year/",
        "University of Michigan consumer sentiment since 1978 (the issue opens with the November "
        "reading; FRED posts the index with a lag, so the latest month may be missing).",
        fred=("UMCSENT",)),
    DailyShotChart(
        "anthropic_ipo", "Anthropic IPO", "Equities",
        "Anthropic’s record-breaking IPO fails to lift markets", "2026-11-20",
        _U + "11/20/anthropics-record-breaking-ipo-fails-to-lift-markets/",
        "Anthropic (ANT) since its Nasdaq debut against the S&P 500 and Nasdaq 100, rebased to the day "
        "before the listing.",
        series=("ANT", "SPX", "NDX")),
    DailyShotChart(
        "defensive_rotation", "Defensive rotation", "Equities",
        "US equities experience their sharpest defensive rotation in 25 years", "2026-11-17",
        _U + "11/17/us-equities-experience-their-sharpest-defensive-rotation-in-25-years/",
        "Defensive sectors (staples, utilities, health care) relative to cyclicals (tech, discretionary, "
        "financials, industrials), equal-weighted SPDR ETFs, and the ratio's 1-month change since 1999.",
        series=DEFENSIVES + CYCLICALS),
    DailyShotChart(
        "cpi_2y", "Inflation & 2Y", "United States",
        "Hotter-than-expected inflation fuels Fed tightening expectations", "2026-11-11",
        _U + "11/11/hotter-than-expected-inflation-fuels-fed-tightening-expectations/",
        "Headline and core CPI, % y/y, against the 2-year Treasury yield, the most Fed-sensitive "
        "point on the curve.",
        fred=("CPIAUCSL", "CPILFESL"), yields=True),
    DailyShotChart(
        "dow_midterm", "Dow after midterms", "Equities",
        "The Dow posted its best post-midterm election day since 1982", "2026-11-05",
        _U + "11/05/the-dow-posted-its-best-post-midterm-election-day-since-1982/",
        "The Dow's % change on the session after each US midterm election (Yahoo history starts in 1992, "
        "so 1994 onward).",
        series=("DJI",)),
    DailyShotChart(
        "sector_week", "Sector week", "Equities",
        "Defensive sectors outperform as technology shares come under pressure", "2026-11-02",
        _U + "11/02/defensive-sectors-outperform-as-technology-shares-come-under-pressure/",
        "S&P 500 sector ETF returns over the week before the issue (23 to 30 Oct 2026 closes).",
        series=SECTORS),
    DailyShotChart(
        "gdp", "US GDP", "United States",
        "US GDP growth remained robust in the third quarter", "2026-10-30",
        _U + "10/30/us-gdp-growth-remained-robust-in-the-third-quarter/",
        "US real GDP growth, % quarter on quarter annualised, since 2019 (BEA via FRED).",
        fred=("A191RL1Q225SBEA",)),
    DailyShotChart(
        "brazil", "Brazil after Lula", "Emerging Markets",
        "Brazil’s markets plunge following Lula’s surprise election victory", "2026-10-27",
        _U + "10/27/brazils-markets-plunge-following-lulas-surprise-election-victory/",
        "Ibovespa, the EWZ ETF and the real (inverted USD/BRL, up = stronger real) around the 25 Oct "
        "2026 run-off, rebased to the last close before it.",
        series=("BVSP", "EWZ", "USDBRL")),
    DailyShotChart(
        "tech_treasuries", "Tech rout & Treasuries", "Rates",
        "The rout in tech stocks sparks safe-haven demand for Treasuries", "2026-10-23",
        _U + "10/23/the-rout-in-tech-stocks-sparks-safe-haven-demand-for-treasuries/",
        "Nasdaq 100 against the 10-year Treasury yield since September 2026: yields fell as tech sold off.",
        series=("NDX",), yields=True),
)


# Each issue's section list, in the newsletter's order, from the public excerpt on
# thedailyshot.com (section names only). The page stacks our charts under these
# headers, so a chart's ``section`` must be one of its issue's sections.
_S = lambda s: tuple(x.strip() for x in s.split("•") if x.strip())
ISSUE_SECTIONS = {
    "2026-11-24": _S("Administrative Update • United States • Canada • United Kingdom • Euro Area • Europe • "
                     "Japan • Asia-Pacific • Emerging Markets • Equities • Rates • Credit • Commodities • Energy • "
                     "Alternatives • Cryptocurrency • Global Developments • Food for Thought"),
    "2026-11-23": _S("United States • Canada • United Kingdom • Euro Area • Japan • Asia-Pacific • China • India • "
                     "Emerging Markets • Equities • Rates • Credit • Commodities • Energy • Global Developments • "
                     "Food for Thought"),
    "2026-11-20": _S("Administrative Update • United States • Canada • United Kingdom • Euro Area • Europe • "
                     "Japan • Asia-Pacific • China • India • Emerging Markets • Equities • Rates • Commodities • "
                     "Energy • Alternatives • Cryptocurrency • Global Developments • Food for Thought"),
    "2026-11-17": _S("Administrative Update • United States • Canada • United Kingdom • Euro Area • Europe • "
                     "Japan • Asia-Pacific • China • India • Emerging Markets • Equities • Commodities • Energy • "
                     "Alternatives • Food for Thought"),
    "2026-11-11": _S("United States • Euro Area • Europe • Japan • Asia-Pacific • China • Emerging Markets • "
                     "Equities • Rates • Credit • Commodities • Energy • Alternatives • Cryptocurrency • "
                     "Global Developments • Food for Thought"),
    "2026-11-05": _S("United States • Canada • United Kingdom • Euro Area • Europe • Japan • Asia-Pacific • China • "
                     "India • Emerging Markets • Equities • Rates • Credit • Commodities • Energy • Alternatives • "
                     "Cryptocurrency • Global Developments • Food for Thought"),
    "2026-11-02": _S("United States • Canada • United Kingdom • Euro Area • Europe • Asia-Pacific • China • India • "
                     "Emerging Markets • Equities • Rates • Credit • Commodities • Energy • Global Developments • "
                     "Cryptocurrency • Food for Thought"),
    "2026-10-30": _S("Administrative Update • United States • Canada • United Kingdom • Euro Area • Europe • "
                     "Japan • Asia-Pacific • China • Emerging Markets • Equities • Rates • Credit • Energy • "
                     "Alternatives • Cryptocurrency • Food for Thought"),
    "2026-10-27": _S("Administrative Update • United States • Canada • United Kingdom • Euro Area • Europe • "
                     "Asia-Pacific • China • India • Emerging Markets • Equities • Rates • Credit • Commodities • "
                     "Energy • Global Developments • Cryptocurrency • Food for Thought"),
    "2026-10-23": _S("United States • Canada • United Kingdom • Euro Area • Europe • Japan • Asia-Pacific • China • "
                     "India • Emerging Markets • Equities • Rates • Credit • Commodities • Energy • Alternatives • "
                     "Food for Thought"),
}

# Recent issues with no chart here, and why (our own one-liners; listed at the page foot).
NOT_REBUILT = (
    ("2026-11-25", "Conference Board consumer confidence jumps to its highest since January 2025",
     _U + "11/25/conference-board-consumer-confidence-jumps-to-its-highest-since-january-2025/",
     "Conference Board data are proprietary."),
    ("2026-11-19", "Pending home sales fell to another record low",
     _U + "11/19/pending-home-sales-fell-to-another-record-low/", "NAR pending home sales are proprietary."),
    ("2026-11-18", "Fund managers slash technology exposure by the most since 2006",
     _U + "11/18/fund-managers-slash-technology-exposure-by-the-most-since-2006/",
     "BofA fund manager survey is proprietary."),
    ("2026-11-16", "China’s economy falters with broad-based weakness",
     _U + "11/16/chinas-economy-falters-with-broad-based-weakness/",
     "China NBS activity data have no reliable free feed."),
    ("2026-11-13", "US existing home sales fell to their lowest level since 2010",
     _U + "11/13/us-existing-home-sales-fell-to-their-lowest-level-since-2010/",
     "NAR existing home sales: FRED carries only the latest year."),
    ("2026-11-12", "US economic data continue to surprise to the upside",
     _U + "11/12/us-economic-data-continue-to-surprise-to-the-upside/",
     "Citi economic surprise index is proprietary."),
    ("2026-11-10", "China’s year-to-date trade surplus surpasses $1 trillion",
     _U + "11/10/chinas-year-to-date-trade-surplus-surpasses-1-trillion/",
     "China customs trade data have no reliable free feed."),
    ("2026-11-09", "Semiconductor stocks trade at their steepest discount to the broader market since 2018",
     _U + "11/09/semiconductor-stocks-trade-at-their-steepest-discount-to-the-broader-market-since-2018/",
     "Forward P/E valuations are proprietary (LSEG / FactSet)."),
    ("2026-11-06", "The Nasdaq 100 suffered a record $2.3 trillion loss in market value",
     _U + "11/06/the-nasdaq-100-suffered-a-record-2-3-trillion-loss-in-market-value/",
     "Index market-cap history is not in free data."),
)


def issue_sections(issue_date: str) -> tuple:
    return ISSUE_SECTIONS.get(f"{pd.Timestamp(issue_date):%Y-%m-%d}", ())


def layout() -> list[dict]:
    """The page's newsletter layout: issues newest first, each with its sections in the
    issue's own order (only sections we have charts for) and the charts under each."""
    out = []
    for iid, charts in charts_by_issue().items():
        c0 = charts[0]
        order = list(issue_sections(c0.issue_date))
        secs: dict[str, list] = {}
        for c in sorted(charts, key=lambda c: order.index(c.section) if c.section in order else len(order)):
            secs.setdefault(c.section, []).append(c)
        out.append({"issue_id": iid, "date": c0.issue_date, "title": c0.issue_title, "url": c0.issue_url,
                    "all_sections": tuple(order), "sections": secs})
    return out


def anchor(*parts) -> str:
    """Streamlit header anchor: lowercase, a-z0-9 and dashes only."""
    import re
    return "ds-" + re.sub(r"[^a-z0-9]+", "-", "-".join(str(p) for p in parts).lower()).strip("-")


def issue_id(issue_date: str, issue_title: str) -> str:
    """Registry key for one issue: "YYYY-MM-DD | headline" (whitespace-normalised)."""
    return f"{pd.Timestamp(issue_date):%Y-%m-%d} | {' '.join(str(issue_title).split())}"


def registry_keys() -> list[str]:
    return [c.key for c in REGISTRY]


def charts_by_issue() -> dict[str, list[DailyShotChart]]:
    """issue_id -> its charts, issues newest first (registry order kept within an issue)."""
    out: dict[str, list[DailyShotChart]] = {}
    for c in sorted(REGISTRY, key=lambda c: c.issue_date, reverse=True):
        out.setdefault(c.issue_id, []).append(c)
    return out


def has_issue(issue_date: str, issue_title: str) -> bool:
    return issue_id(issue_date, issue_title) in charts_by_issue()


# ---------------------------------------------------------------------- data
@dataclass
class Prices:
    df: pd.DataFrame      # index date (naive), one column per YAHOO_SYMBOLS key
    source: dict          # column -> "live" | "csv" | "live+csv" | "none"
    errors: list


@dataclass
class Fred:
    df: pd.DataFrame      # index date, one column per FRED id
    source: dict
    errors: list


def parse_yahoo_chart(payload: dict, name: str) -> pd.Series:
    """Yahoo v8 chart JSON -> Series of daily closes on the exchange's calendar date."""
    res = (payload.get("chart") or {}).get("result") or []
    if not res:
        raise ValueError(f"Yahoo {name}: empty result")
    res = res[0]
    ts = res.get("timestamp") or []
    closes = ((res.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    if len(ts) != len(closes):
        raise ValueError(f"Yahoo {name}: {len(ts)} timestamps vs {len(closes)} closes")
    tz = (res.get("meta") or {}).get("exchangeTimezoneName") or "America/New_York"
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    s = pd.Series(pd.to_numeric(pd.Series(closes, dtype="float64"), errors="coerce").values,
                  index=idx, name=name)
    s = s[np.isfinite(s.to_numpy()) & (s.to_numpy() > 0)]
    s = s[~s.index.duplicated(keep="last")]
    s.index.name = "date"
    return s.sort_index()


def fetch_yahoo(sym: str, name: str, start: str, get=None, timeout: int = 20) -> pd.Series:
    get = get or requests.get
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(time.time()) + 86400
    r = get(YAHOO_URL.format(sym=sym), params={"period1": p1, "period2": p2, "interval": "1d"},
            headers=UA, timeout=timeout)
    r.raise_for_status()
    return parse_yahoo_chart(r.json(), name)


def _read_csv(name: str, cols, data_dir: Path | None) -> pd.DataFrame:
    p = (data_dir or DATA_DIR) / name
    if not p.exists():
        return pd.DataFrame(columns=list(cols)).rename_axis("date")
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def merge_live_csv(live: dict, csv: pd.DataFrame, names) -> tuple[pd.DataFrame, dict]:
    """Live values win on a shared date; the CSV supplies dates live did not return."""
    cols, source = {}, {}
    for name in names:
        lv = live.get(name)
        cv = csv[name].dropna() if name in csv.columns else pd.Series(dtype=float)
        if lv is not None and len(lv):
            extra = cv[~cv.index.isin(lv.index)]
            s = pd.concat([lv, extra]).sort_index()
            source[name] = "live+csv" if len(extra) else "live"
        elif len(cv):
            s, source[name] = cv, "csv"
        else:
            s, source[name] = pd.Series(dtype=float), "none"
        cols[name] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df, source


def load_prices(get=None, data_dir: Path | None = None, live: bool = True) -> Prices:
    live_cols, errors = {}, []
    if live:
        for name, (sym, start) in YAHOO_SYMBOLS.items():
            try:
                live_cols[name] = fetch_yahoo(sym, name, start, get=get)
            except Exception as ex:
                errors.append(f"{sym}: {ex}")
    df, source = merge_live_csv(live_cols, _read_csv(PRICES_CSV, YAHOO_SYMBOLS, data_dir), YAHOO_SYMBOLS)
    return Prices(df, source, errors)


def parse_fred_csv(text: str) -> dict[str, pd.Series]:
    """fredgraph.csv (one or several ids) -> {id: Series}; '.' and blanks dropped."""
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    out = {}
    for c in df.columns[1:]:
        s = pd.to_numeric(df[c], errors="coerce")
        s.index = df[date_col]
        s = s.dropna()
        s.index.name = "date"
        out[c] = s.rename(c)
    return out


def load_fred(get=None, data_dir: Path | None = None, live: bool = True, timeout: int = 8) -> Fred:
    """One request per id (FRED's multi-id CSV is slow). The first failure stops the live
    attempts, so a dead FRED (as from Streamlit Cloud) costs one ``timeout``, not four."""
    live_cols, errors = {}, []
    if live:
        for sid in FRED_SERIES:
            try:
                r = (get or requests.get)(FRED_URL, params={"id": sid}, headers=FRED_UA, timeout=timeout)
                r.raise_for_status()
                got = parse_fred_csv(r.text)
                if sid not in got or got[sid].empty:
                    raise ValueError("no rows")
                live_cols[sid] = got[sid]
            except Exception as ex:
                errors.append(f"FRED {sid}: {type(ex).__name__}")
                break
    df, source = merge_live_csv(live_cols, _read_csv(FRED_CSV, FRED_SERIES, data_dir), FRED_SERIES)
    return Fred(df, source, errors)


def write_prices_csv(data_dir: Path | None = None, get=None) -> Path:
    """Refresh the committed Yahoo fallback (real closes only, today's partial bar dropped)."""
    pr = load_prices(get=get, data_dir=data_dir)
    if pr.errors:
        raise RuntimeError("; ".join(pr.errors))
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    p = (data_dir or DATA_DIR) / PRICES_CSV
    pr.df[pr.df.index < today].astype(float).round(4).to_csv(p, date_format="%Y-%m-%d", float_format="%.4f")
    return p


def write_fred_csv(data_dir: Path | None = None, get=None) -> Path:
    fr = load_fred(get=get, data_dir=data_dir, timeout=30)
    if fr.errors:
        raise RuntimeError("; ".join(fr.errors))
    df = fr.df[fr.df.index >= pd.Timestamp("1978-01-01")]
    p = (data_dir or DATA_DIR) / FRED_CSV
    df.to_csv(p, date_format="%Y-%m-%d", float_format="%.3f")
    return p


# --------------------------------------------------------------------- maths
def rebase(df: pd.DataFrame, base_date=None) -> pd.DataFrame:
    """Each column / its value on the last date <= base_date (or its first value), x100."""
    out = {}
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        b = s[s.index <= pd.Timestamp(base_date)] if base_date is not None else s.iloc[:0]
        base = b.iloc[-1] if len(b) else s.iloc[0]
        out[c] = df[c] / base * 100
    return pd.DataFrame(out)


def ytd(s: pd.Series, year: int | None = None) -> pd.Series:
    """% return vs the last close of the prior year, for ``year`` (default: latest year in s)."""
    s = s.dropna()
    year = year or s.index[-1].year
    prior = s[s.index.year < year]
    if prior.empty:
        raise ValueError("no prior-year close")
    cur = s[s.index.year == year]
    return (cur / prior.iloc[-1] - 1) * 100


def defensive_ratio(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Equal-weight defensives / cyclicals (each ETF rebased to 1 at the first common date)
    and the ratio's ``window``-session % change."""
    cols = list(DEFENSIVES + CYCLICALS)
    d = df[cols].dropna()
    if d.empty:
        return pd.DataFrame(columns=["ratio", "chg"])
    d = d / d.iloc[0]
    ratio = d[list(DEFENSIVES)].mean(axis=1) / d[list(CYCLICALS)].mean(axis=1)
    return pd.DataFrame({"ratio": ratio, "chg": (ratio / ratio.shift(window) - 1) * 100})


def yoy(s: pd.Series, periods: int = 12) -> pd.Series:
    s = s.dropna()
    return ((s / s.shift(periods) - 1) * 100).dropna()


def post_event_returns(s: pd.Series, event_dates) -> pd.DataFrame:
    """% change on the first session after each event date (close vs prior close)."""
    s = s.dropna()
    rows = []
    for d in event_dates:
        d = pd.Timestamp(d)
        after = s.index[s.index > d]
        if not len(after):
            continue
        day = after[0]
        prev = s[s.index < day]
        if prev.empty:
            continue
        rows.append({"election": d, "session": day, "pct": (s[day] / prev.iloc[-1] - 1) * 100})
    return pd.DataFrame(rows, columns=["election", "session", "pct"])


def period_returns(df: pd.DataFrame, start, end) -> pd.Series:
    """% change from the last close <= start to the last close <= end, per column."""
    out = {}
    for c in df.columns:
        s = df[c].dropna()
        a, b = s[s.index <= pd.Timestamp(start)], s[s.index <= pd.Timestamp(end)]
        if len(a) and len(b):
            out[c] = (b.iloc[-1] / a.iloc[-1] - 1) * 100
    return pd.Series(out, dtype=float).sort_values()


def highest_since(s: pd.Series, window_start, window_end=None) -> dict:
    """The peak of ``s`` inside [window_start, window_end] and the last earlier date that was
    higher ("highest since ..."); ``since`` is None when the peak is a record."""
    s = s.dropna()
    w = s[(s.index >= pd.Timestamp(window_start))]
    if window_end is not None:
        w = w[w.index <= pd.Timestamp(window_end)]
    if w.empty:
        raise ValueError("empty window")
    peak_date, peak = w.idxmax(), w.max()
    before = s[(s.index < w.index[0]) & (s > peak)]
    return {"peak": float(peak), "peak_date": peak_date,
            "since": before.index[-1] if len(before) else None}
