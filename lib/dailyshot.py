"""Daily Shot-style chart book: data loaders, maths and the chart list. No Streamlit here.

Sean asked (27 Nov 2026) for a standing set of charts laid out like The Daily
Shot newsletter (https://thedailyshot.com/): one long page, the newsletter's
section names in its order, several charts under each, refreshed on demand.
It is not tied to any issue and holds none of The Daily Shot's text or
images: every chart is our own, drawn from free public data on a rolling
window that ends at the latest data point.

Data (all real, nothing filled or invented; a missing day stays missing)
----
* Yahoo Finance daily closes, fetched in parallel (one request per symbol,
  short timeout), with ``data/dailyshot_prices.csv`` (real Yahoo closes) for
  any symbol or date live does not return.
* FRED series, fetched in parallel with a short timeout (FRED times out from
  Streamlit Cloud), with ``data/dailyshot_fred.csv`` (real FRED values).
* US Treasury daily par yields from home.treasury.gov, with
  ``data/dailyshot_yields.csv``.
* Deribit DVOL (BTC, ETH) daily, with ``data/vl_dvol.csv``.

Refresh the committed fallbacks:
``python3 -c "from lib import dailyshot as d; d.write_fallbacks()"``

Adding a chart: append a ``Chart`` to ``CHARTS`` with an existing ``kind``
(see ``lib.dailyshot_charts.KINDS``) and a ``section`` from ``SECTIONS``; any
new Yahoo symbol / FRED id is picked up from the chart args automatically
(``yahoo_symbols()``, ``fred_ids()``). Re-run ``write_fallbacks()``.
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_CSV, FRED_CSV, YIELDS_CSV, DVOL_CSV = ("dailyshot_prices.csv", "dailyshot_fred.csv",
                                              "dailyshot_yields.csv", "vl_dvol.csv")
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TREASURY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
                "&field_tdr_date_value={year}&page&_format=csv")
DERIBIT_DVOL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}
FRED_UA = {"User-Agent": "Mozilla/5.0 (compatible; mcm-analytics)"}   # FRED stalls on a browser UA
YAHOO_RANGE = "5y"
TIMEOUT = 8
FRED_TIMEOUT = 6
TENORS = {"1 Mo": "1M", "3 Mo": "3M", "6 Mo": "6M", "1 Yr": "1Y", "2 Yr": "2Y", "3 Yr": "3Y",
          "5 Yr": "5Y", "7 Yr": "7Y", "10 Yr": "10Y", "20 Yr": "20Y", "30 Yr": "30Y"}

# The Daily Shot's section names, in the order the newsletter runs them.
SECTIONS = ("United States", "Canada", "United Kingdom", "Euro Area", "Japan", "Asia-Pacific", "China",
            "India", "Emerging Markets", "Equities", "Rates", "Credit", "Commodities", "Energy",
            "Alternatives", "Cryptocurrency", "Global Developments")

SECTORS = [("XLK", "Tech"), ("XLC", "Comms"), ("XLY", "Discretionary"), ("XLF", "Financials"),
           ("XLI", "Industrials"), ("XLB", "Materials"), ("XLE", "Energy"), ("XLRE", "Real estate"),
           ("XLP", "Staples"), ("XLU", "Utilities"), ("XLV", "Health care")]


@dataclass(frozen=True)
class Chart:
    key: str
    section: str
    title: str
    kind: str
    caption: str               # our own one-liner (source and as-of are added by the page)
    args: dict = field(default_factory=dict)


C = Chart
CHARTS: tuple[Chart, ...] = (
    # ---------------------------------------------------------------- United States
    C("us_cpi", "United States", "US inflation (% y/y)", "fred",
      "Headline and core CPI, year on year.",
      {"series": [("CPIAUCSL", "CPI"), ("CPILFESL", "Core CPI")], "transform": "yoy", "years": 6, "suffix": "%"}),
    C("us_payrolls", "United States", "US nonfarm payrolls, monthly change (k)", "fred",
      "Monthly change in nonfarm payrolls, thousands.",
      {"series": [("PAYEMS", "Payrolls change")], "transform": "diff", "years": 3, "bar": True}),
    C("us_unrate", "United States", "US unemployment rate (%)", "fred", "Civilian unemployment rate.",
      {"series": [("UNRATE", "Unemployment rate")], "years": 10, "suffix": "%"}),
    C("us_claims", "United States", "US initial jobless claims (k)", "fred",
      "Weekly initial unemployment claims, thousands, with a 4-week average.",
      {"series": [("ICSA", "Initial claims")], "years": 3, "scale": 0.001, "ma": 4}),
    C("us_umich", "United States", "University of Michigan consumer sentiment", "fred",
      "Consumer sentiment index (FRED posts it with a one-month lag).",
      {"series": [("UMCSENT", "Sentiment")], "years": 25}),
    C("us_gdp", "United States", "US real GDP growth (% q/q annualised)", "fred",
      "Real GDP, quarter on quarter, seasonally adjusted annual rate.",
      {"series": [("A191RL1Q225SBEA", "Real GDP")], "years": 6, "bar": True, "suffix": "%", "clip": (-10, 15)}),
    C("us_mortgage", "United States", "30-year mortgage rate (%)", "fred", "Freddie Mac 30-year fixed rate, weekly.",
      {"series": [("MORTGAGE30US", "30y mortgage")], "years": 5, "suffix": "%"}),
    # ----------------------------------------------------------------------- Canada
    C("ca_tsx_cad", "Canada", "TSX and the Canadian dollar", "dual",
      "S&P/TSX Composite (left) and USD/CAD on an inverted right axis (up = stronger loonie).",
      {"left": ("^GSPTSE", "S&P/TSX"), "right": ("CAD=X", "USD/CAD"), "years": 2, "invert_right": True}),
    # --------------------------------------------------------------- United Kingdom
    C("uk_ftse_gbp", "United Kingdom", "FTSE 100 and sterling", "dual", "FTSE 100 (left) and GBP/USD (right).",
      {"left": ("^FTSE", "FTSE 100"), "right": ("GBPUSD=X", "GBP/USD"), "years": 2}),
    # -------------------------------------------------------------------- Euro Area
    C("ea_stoxx_eur", "Euro Area", "Euro Stoxx 50 and the euro", "dual", "Euro Stoxx 50 (left) and EUR/USD (right).",
      {"left": ("^STOXX50E", "Euro Stoxx 50"), "right": ("EURUSD=X", "EUR/USD"), "years": 2}),
    C("ea_ecb", "Euro Area", "ECB deposit facility rate (%)", "fred", "The ECB's deposit rate.",
      {"series": [("ECBDFR", "Deposit rate")], "years": 10, "suffix": "%", "step": True}),
    # ------------------------------------------------------------------------ Japan
    C("jp_nikkei_jpy", "Japan", "Nikkei 225 and the yen", "dual",
      "Nikkei 225 (left) and USD/JPY on an inverted right axis (up = stronger yen).",
      {"left": ("^N225", "Nikkei 225"), "right": ("JPY=X", "USD/JPY"), "years": 2, "invert_right": True}),
    C("jp_jgb", "Japan", "Japan 10-year government bond yield (%)", "fred", "OECD long-term rate for Japan, monthly.",
      {"series": [("IRLTLT01JPM156N", "JGB 10y")], "years": 10, "suffix": "%"}),
    # ----------------------------------------------------------------- Asia-Pacific
    C("ap_markets", "Asia-Pacific", "Korea, Taiwan and Australia", "lines",
      "KOSPI, iShares MSCI Taiwan and AUD/USD, rebased to 100 a year ago.",
      {"syms": [("^KS11", "KOSPI"), ("EWT", "Taiwan (EWT)"), ("AUDUSD=X", "AUD/USD")], "years": 1, "rebase": True}),
    # ------------------------------------------------------------------------ China
    C("cn_equities", "China", "Chinese equities", "lines",
      "Shanghai Composite, Hang Seng and FXI, rebased to 100 a year ago.",
      {"syms": [("000001.SS", "Shanghai Comp."), ("^HSI", "Hang Seng"), ("FXI", "FXI (USD)")], "years": 1,
       "rebase": True}),
    C("cn_cny", "China", "USD/CNY", "lines", "Onshore yuan per dollar (up = weaker yuan).",
      {"syms": [("CNY=X", "USD/CNY")], "years": 2}),
    # ------------------------------------------------------------------------ India
    C("in_nifty_inr", "India", "Nifty 50 and the rupee", "dual",
      "Nifty 50 (left) and USD/INR on an inverted right axis (up = stronger rupee).",
      {"left": ("^NSEI", "Nifty 50"), "right": ("INR=X", "USD/INR"), "years": 2, "invert_right": True}),
    # ------------------------------------------------------------- Emerging Markets
    C("em_vs_us", "Emerging Markets", "EM equities relative to the S&P 500", "ratio",
      "EEM / SPY price ratio with its 200-day average.",
      {"num": "EEM", "den": "SPY", "years": 5, "ma": 200}),
    C("em_countries", "Emerging Markets", "EM country ETFs, last month (%)", "returns",
      "1-month price return of iShares country ETFs (USD).",
      {"syms": [("EWZ", "Brazil"), ("EWW", "Mexico"), ("INDA", "India"), ("FXI", "China"), ("EWY", "Korea"),
                ("EWT", "Taiwan"), ("EZA", "South Africa"), ("TUR", "Turkey")], "period": "1M"}),
    C("em_fx", "Emerging Markets", "EM currencies vs the dollar", "lines",
      "Local currency per USD, rebased to 100 a year ago (up = weaker local currency).",
      {"syms": [("MXN=X", "MXN"), ("BRL=X", "BRL"), ("INR=X", "INR"), ("CNY=X", "CNY")], "years": 1, "rebase": True}),
    # --------------------------------------------------------------------- Equities
    C("eq_spx", "Equities", "S&P 500 with 50- and 200-day averages", "lines", "S&P 500 index level.",
      {"syms": [("^GSPC", "S&P 500")], "years": 2, "ma": [50, 200]}),
    C("eq_sectors_1m", "Equities", "S&P 500 sectors, last month (%)", "returns",
      "1-month price return of the SPDR sector ETFs.", {"syms": SECTORS, "period": "1M"}),
    C("eq_sectors_ytd", "Equities", "S&P 500 sectors, year to date (%)", "returns",
      "Year-to-date price return of the SPDR sector ETFs.", {"syms": SECTORS, "period": "YTD"}),
    C("eq_equal_weight", "Equities", "Equal-weight vs cap-weight S&P 500", "ratio",
      "RSP / SPY: falling = mega-caps leading.", {"num": "RSP", "den": "SPY", "years": 3, "ma": 200}),
    C("eq_small_large", "Equities", "Small caps vs large caps", "ratio", "IWM / SPY price ratio.",
      {"num": "IWM", "den": "SPY", "years": 3, "ma": 200}),
    C("eq_growth_value", "Equities", "Growth vs value", "ratio", "Russell 1000 Growth / Value (IWF / IWD).",
      {"num": "IWF", "den": "IWD", "years": 3, "ma": 200}),
    C("eq_ndx_dd", "Equities", "Nasdaq 100 drawdown from its high (%)", "drawdown",
      "Distance of the Nasdaq 100 from its running peak.", {"sym": "^NDX", "years": 5}),
    # ------------------------------------------------------------------------ Rates
    C("rt_curve", "Rates", "US Treasury yield curve", "curve",
      "Par yields now, a month ago and a year ago.", {}),
    C("rt_2y10y", "Rates", "2-year and 10-year Treasury yields (%)", "yields", "Daily par yields.",
      {"tenors": [("2Y", "2-year"), ("10Y", "10-year")], "years": 5}),
    C("rt_2s10s", "Rates", "2s10s curve spread (bp)", "yspread", "10-year minus 2-year yield.",
      {"a": "10Y", "b": "2Y", "years": 5}),
    C("rt_3m10y", "Rates", "3m-10y curve spread (bp)", "yspread", "10-year minus 3-month yield.",
      {"a": "10Y", "b": "3M", "years": 5}),
    C("rt_real", "Rates", "10-year real yield (TIPS, %)", "fred", "10-year Treasury inflation-indexed yield.",
      {"series": [("DFII10", "10y real yield")], "years": 5, "suffix": "%"}),
    # ----------------------------------------------------------------------- Credit
    C("cr_hy_oas", "Credit", "US high-yield spread (%)", "fred",
      "ICE BofA US High Yield option-adjusted spread (FRED carries 3 years).",
      {"series": [("BAMLH0A0HYM2", "HY OAS")], "years": 3, "suffix": "%"}),
    C("cr_ig_oas", "Credit", "US investment-grade spread (%)", "fred",
      "ICE BofA US Corporate option-adjusted spread (FRED carries 3 years).",
      {"series": [("BAMLC0A0CM", "IG OAS")], "years": 3, "suffix": "%"}),
    C("cr_hyg_ief", "Credit", "High yield vs Treasuries", "ratio",
      "HYG / IEF price ratio: falling = credit underperforming.", {"num": "HYG", "den": "IEF", "years": 3, "ma": 200}),
    # ------------------------------------------------------------------ Commodities
    C("cm_ytd", "Commodities", "Commodity futures, year to date (%)", "returns",
      "Front-month futures, year-to-date price change.",
      {"syms": [("GC=F", "Gold"), ("SI=F", "Silver"), ("HG=F", "Copper"), ("CL=F", "WTI"), ("BZ=F", "Brent"),
                ("NG=F", "Nat gas"), ("ZW=F", "Wheat"), ("ZC=F", "Corn"), ("ZS=F", "Soybeans")], "period": "YTD"}),
    C("cm_gold_silver", "Commodities", "Gold and silver", "lines", "Front-month futures, rebased to 100 two years ago.",
      {"syms": [("GC=F", "Gold"), ("SI=F", "Silver")], "years": 2, "rebase": True}),
    C("cm_gs_ratio", "Commodities", "Gold / silver ratio", "ratio", "Gold price over silver price.",
      {"num": "GC=F", "den": "SI=F", "years": 5, "ma": 200}),
    C("cm_copper", "Commodities", "Copper (USD/lb)", "lines", "COMEX front-month copper with 50/200-day averages.",
      {"syms": [("HG=F", "Copper")], "years": 2, "ma": [50, 200]}),
    C("cm_grains", "Commodities", "Grains", "lines", "Wheat, corn and soybean futures, rebased to 100 a year ago.",
      {"syms": [("ZW=F", "Wheat"), ("ZC=F", "Corn"), ("ZS=F", "Soybeans")], "years": 1, "rebase": True}),
    # ----------------------------------------------------------------------- Energy
    C("en_crude", "Energy", "Crude oil (USD/bbl)", "lines", "WTI and Brent front-month futures.",
      {"syms": [("CL=F", "WTI"), ("BZ=F", "Brent")], "years": 2}),
    C("en_brent_wti", "Energy", "Brent-WTI spread (USD/bbl)", "spread", "Brent minus WTI, front month.",
      {"a": "BZ=F", "b": "CL=F", "years": 2}),
    C("en_natgas", "Energy", "US natural gas (USD/MMBtu)", "lines", "Henry Hub front-month futures.",
      {"syms": [("NG=F", "Nat gas")], "years": 2, "ma": [50]}),
    C("en_gasoline", "Energy", "RBOB gasoline (USD/gal)", "lines", "NYMEX front-month gasoline futures.",
      {"syms": [("RB=F", "RBOB")], "years": 2}),
    # ----------------------------------------------------------------- Alternatives
    C("alt_reits", "Alternatives", "REITs vs the S&P 500", "ratio", "VNQ / SPY price ratio.",
      {"num": "VNQ", "den": "SPY", "years": 3, "ma": 200}),
    C("alt_private", "Alternatives", "Listed private equity and BDCs", "lines",
      "PSP (listed PE) and BIZD (BDCs) against SPY, rebased to 100 two years ago.",
      {"syms": [("PSP", "Listed PE (PSP)"), ("BIZD", "BDCs (BIZD)"), ("SPY", "S&P 500 (SPY)")], "years": 2,
       "rebase": True}),
    # --------------------------------------------------------------- Cryptocurrency
    C("cx_btc", "Cryptocurrency", "Bitcoin with 50- and 200-day averages", "lines", "BTC-USD daily close (UTC).",
      {"syms": [("BTC-USD", "Bitcoin")], "years": 2, "ma": [50, 200], "log": True}),
    C("cx_eth_btc", "Cryptocurrency", "ETH / BTC", "ratio", "Ether priced in bitcoin.",
      {"num": "ETH-USD", "den": "BTC-USD", "years": 2, "ma": 200}),
    C("cx_vol", "Cryptocurrency", "Bitcoin implied vs realised volatility (%)", "rvol_dvol",
      "Deribit DVOL (30-day implied) against 30-day realised volatility of daily closes.",
      {"coin": "BTC", "sym": "BTC-USD", "years": 1}),
    C("cx_corr", "Cryptocurrency", "Bitcoin's correlation with the Nasdaq 100", "corr",
      "60-day rolling correlation of daily returns (US trading days).",
      {"a": "BTC-USD", "b": "^NDX", "window": 60, "years": 3}),
    C("cx_dd", "Cryptocurrency", "Bitcoin drawdown from its high (%)", "drawdown",
      "Distance of BTC from its running peak.", {"sym": "BTC-USD", "years": 5}),
    C("cx_1m", "Cryptocurrency", "Crypto, last month (%)", "returns", "1-month price return.",
      {"syms": [("BTC-USD", "BTC"), ("ETH-USD", "ETH"), ("SOL-USD", "SOL"), ("XRP-USD", "XRP"), ("BNB-USD", "BNB"),
                ("DOGE-USD", "DOGE"), ("IBIT", "IBIT")], "period": "1M"}),
    # ---------------------------------------------------------- Global Developments
    C("gl_dxy", "Global Developments", "US dollar index", "lines", "ICE dollar index (DXY) with its 200-day average.",
      {"syms": [("DX-Y.NYB", "DXY")], "years": 2, "ma": [200]}),
    C("gl_world_ytd", "Global Developments", "World equity indices, year to date (%)", "returns",
      "Year-to-date return in local currency.",
      {"syms": [("^GSPC", "S&P 500"), ("^GSPTSE", "TSX"), ("^FTSE", "FTSE 100"), ("^STOXX50E", "Euro Stoxx 50"),
                ("^N225", "Nikkei"), ("^KS11", "KOSPI"), ("^HSI", "Hang Seng"), ("000001.SS", "Shanghai"),
                ("^NSEI", "Nifty 50"), ("^BVSP", "Ibovespa"), ("ACWI", "MSCI ACWI (USD)")], "period": "YTD"}),
    C("gl_vix_move", "Global Developments", "Equity and bond volatility", "dual",
      "VIX (left) and the ICE BofA MOVE index of Treasury volatility (right).",
      {"left": ("^VIX", "VIX"), "right": ("^MOVE", "MOVE"), "years": 2}),
)


def sections() -> list[tuple[str, list[Chart]]]:
    """[(section, charts)] in newsletter order, only sections that have charts."""
    return [(s, [c for c in CHARTS if c.section == s]) for s in SECTIONS if any(c.section == s for c in CHARTS)]


def anchor(*parts) -> str:
    return "ds-" + re.sub(r"[^a-z0-9]+", "-", "-".join(str(p) for p in parts).lower()).strip("-")


def yahoo_symbols() -> list[str]:
    out = []
    for c in CHARTS:
        a = c.args
        for s, _ in a.get("syms", []):
            out.append(s)
        for k in ("num", "den", "sym", "a", "b"):
            if k in a and c.kind not in ("yspread",):
                out.append(a[k])
        for k in ("left", "right"):
            if k in a:
                out.append(a[k][0])
    return list(dict.fromkeys(out))


def fred_ids() -> list[str]:
    return list(dict.fromkeys(i for c in CHARTS if c.kind == "fred" for i, _ in c.args["series"]))


# ---------------------------------------------------------------------- data
@dataclass
class Dataset:
    df: pd.DataFrame       # date index, one column per symbol / id / tenor
    source: dict           # column -> "live" | "csv" | "live+csv" | "none"
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
    s = pd.Series(pd.to_numeric(pd.Series(closes, dtype="float64"), errors="coerce").values, index=idx, name=name)
    s = s[np.isfinite(s.to_numpy()) & (s.to_numpy() > 0)]
    s = s[~s.index.duplicated(keep="last")]
    s.index.name = "date"
    return s.sort_index()


def parse_fred_csv(text: str) -> dict[str, pd.Series]:
    """fredgraph.csv -> {id: Series}; '.' and blanks dropped."""
    df = pd.read_csv(io.StringIO(text))
    dcol = df.columns[0]
    idx = pd.to_datetime(df[dcol])
    out = {}
    for c in df.columns[1:]:
        s = pd.Series(pd.to_numeric(df[c], errors="coerce").values, index=idx, name=c).dropna()
        s.index.name = "date"
        out[c] = s
    return out


def parse_treasury_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    if "Date" not in df.columns:
        raise ValueError("Treasury CSV has no Date column")
    out = pd.DataFrame({"date": pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")})
    for src, lab in TENORS.items():
        if src in df.columns:
            out[lab] = pd.to_numeric(df[src], errors="coerce")
    return out.dropna(subset=["date"]).set_index("date").sort_index()


def parse_dvol(payload: dict, name: str) -> pd.Series:
    rows = ((payload or {}).get("result") or {}).get("data") or []
    if not rows:
        raise ValueError(f"Deribit {name}: no rows")
    idx = pd.to_datetime([r[0] for r in rows], unit="ms").normalize()
    s = pd.Series([float(r[4]) for r in rows], index=idx, name=name)
    s.index.name = "date"
    return s[~s.index.duplicated(keep="last")].sort_index()


def _read_csv(name: str, data_dir: Path | None) -> pd.DataFrame:
    p = (data_dir or DATA_DIR) / name
    if not p.exists():
        return pd.DataFrame().rename_axis("date")
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


def _parallel(fn, keys, workers=12):
    """{key: result} for keys whose fn(key) succeeded; errors as 'key: ExcType'."""
    got, errors = {}, []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, k): k for k in keys}
        for f in cf.as_completed(futs):
            k = futs[f]
            try:
                got[k] = f.result()
            except Exception as e:
                errors.append(f"{k}: {type(e).__name__}")
    return got, sorted(errors)


def load_prices(get=None, data_dir=None, live=True, symbols=None) -> Dataset:
    syms = symbols or yahoo_symbols()
    get = get or requests.get

    def one(sym):
        r = get(YAHOO_URL.format(sym=sym), params={"range": YAHOO_RANGE, "interval": "1d"}, headers=UA,
                timeout=TIMEOUT)
        r.raise_for_status()
        return parse_yahoo_chart(r.json(), sym)

    live_cols, errors = _parallel(one, syms) if live else ({}, [])
    df, src = merge_live_csv(live_cols, _read_csv(PRICES_CSV, data_dir), syms)
    return Dataset(df, src, [f"Yahoo {e}" for e in errors])


def load_fred(get=None, data_dir=None, live=True, timeout=FRED_TIMEOUT) -> Dataset:
    ids = fred_ids()
    get = get or requests.get

    def one(sid):
        r = get(FRED_URL, params={"id": sid}, headers=FRED_UA, timeout=timeout)
        r.raise_for_status()
        s = parse_fred_csv(r.text).get(sid)
        if s is None or s.empty:
            raise ValueError("no rows")
        return s

    live_cols, errors = _parallel(one, ids, workers=8) if live else ({}, [])
    df, src = merge_live_csv(live_cols, _read_csv(FRED_CSV, data_dir), ids)
    err = [f"FRED unreachable ({len(errors)} of {len(ids)} series)"] if errors else []
    return Dataset(df, src, err)


def load_yields(get=None, data_dir=None, live=True, years=5, now=None) -> Dataset:
    get = get or requests.get
    y0 = (pd.Timestamp(now) if now is not None else pd.Timestamp.now()).year

    def one(year):
        r = get(TREASURY_URL.format(year=year), headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        return parse_treasury_csv(r.text)

    got, errors = _parallel(one, list(range(y0 - years, y0 + 1)), workers=6) if live else ({}, [])
    names = list(TENORS.values())
    live_cols = {}
    if got:
        ldf = pd.concat(got.values()).sort_index()
        ldf = ldf[~ldf.index.duplicated(keep="last")]
        live_cols = {n: ldf[n].dropna() for n in names if n in ldf}
    df, src = merge_live_csv(live_cols, _read_csv(YIELDS_CSV, data_dir), names)
    return Dataset(df, src, [f"Treasury {e}" for e in errors])


def load_dvol(get=None, data_dir=None, live=True) -> Dataset:
    get = get or requests.get
    end = int(time.time() * 1000)

    def one(coin):
        r = get(DERIBIT_DVOL, params={"currency": coin, "start_timestamp": end - 800 * 86_400_000,
                                      "end_timestamp": end, "resolution": "1D"}, timeout=5)
        r.raise_for_status()
        return parse_dvol(r.json(), coin)

    got, errors = _parallel(one, ["BTC", "ETH"], workers=2) if live else ({}, [])
    csv = _read_csv(DVOL_CSV, data_dir).rename(columns={"DVOL_BTC": "BTC", "DVOL_ETH": "ETH"})
    df, src = merge_live_csv(got, csv, ["BTC", "ETH"])
    return Dataset(df, src, [f"Deribit {e}" for e in errors])


def write_fallbacks(data_dir: Path | None = None) -> dict:
    """Refresh the committed fallbacks from live (real values only, today's partial bar dropped)."""
    d = data_dir or DATA_DIR
    out = {}
    pr = load_prices(data_dir=d)
    if pr.errors:
        raise RuntimeError("; ".join(pr.errors))
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    pr.df[pr.df.index < today].astype(float).round(4).to_csv(d / PRICES_CSV, date_format="%Y-%m-%d")
    fr = load_fred(data_dir=d, timeout=40)
    if fr.errors:
        raise RuntimeError("; ".join(fr.errors))
    fr.df[fr.df.index >= "1995-01-01"].to_csv(d / FRED_CSV, date_format="%Y-%m-%d", float_format="%.3f")
    yl = load_yields(data_dir=d)
    if yl.errors:
        raise RuntimeError("; ".join(yl.errors))
    yl.df.to_csv(d / YIELDS_CSV, date_format="%Y-%m-%d", float_format="%.2f")
    for name in (PRICES_CSV, FRED_CSV, YIELDS_CSV):
        out[name] = (d / name).stat().st_size
    return out


# --------------------------------------------------------------------- maths
def window(s, years: float, end=None):
    s = s.dropna() if isinstance(s, pd.Series) else s.dropna(how="all")
    if s.empty:
        return s
    end = pd.Timestamp(end) if end is not None else s.index[-1]
    return s[(s.index > end - pd.DateOffset(days=int(round(365.25 * years)))) & (s.index <= end)]


def rebase(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: df[c] / df[c].dropna().iloc[0] * 100 for c in df.columns if df[c].notna().any()})


def period_return(s: pd.Series, period: str) -> float:
    """% change over '1M' / '3M' / 'YTD' / '1Y' ending at the last close."""
    s = s.dropna()
    if s.empty:
        return np.nan
    end = s.index[-1]
    if period == "YTD":
        base = s[s.index.year < end.year]
    else:
        n = {"1M": 1, "3M": 3, "1Y": 12}[period]
        base = s[s.index <= end - pd.DateOffset(months=n)]
    return float((s.iloc[-1] / base.iloc[-1] - 1) * 100) if len(base) else np.nan


def yoy(s: pd.Series) -> pd.Series:
    """% change on the same date a year earlier (monthly or quarterly series)."""
    s = s.dropna()
    prev = s.reindex(s.index - pd.DateOffset(years=1))
    return pd.Series((s.values / prev.values - 1) * 100, index=s.index).dropna()


def drawdown(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return (s / s.cummax() - 1) * 100


def realised_vol(s: pd.Series, n: int = 30, ann: int = 365) -> pd.Series:
    r = np.log(s.dropna()).diff()
    return r.rolling(n).std() * np.sqrt(ann) * 100


def rolling_corr(a: pd.Series, b: pd.Series, n: int = 60) -> pd.Series:
    """Correlation of daily log returns on dates both have (b's calendar)."""
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    r = np.log(df).diff().dropna()
    return r.iloc[:, 0].rolling(n).corr(r.iloc[:, 1]).dropna()
