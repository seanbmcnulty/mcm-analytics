"""Strategy (MSTR) weekly funding breakdown from its Monday 8-K.

Strategy files an 8-K most Mondays with an "ATM Update" (net proceeds by
security), a "BTC Update", a "Repurchase Program Updates" table and a
"USD Reserve and USD Cash Updates" paragraph. This module finds the latest
such 8-K on SEC EDGAR and turns it into two small tables:

* sources: net proceeds raised that week by instrument (MSTR, STRF, STRC, ...)
* uses: BTC purchased, preferred/common buybacks, USD Cash / USD Reserve

SEC fair-access policy asks for a descriptive User-Agent with contact details
and at most 10 requests a second. We make two to four requests per refresh.

Live parse first; ``data/strategy_funding.csv`` (written by the in-app refresh
and the weekly GitHub Action) is the fallback. No network on import.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from lib import ecoinometrics as eco

CIK = "0001050446"
CIK_INT = int(CIK)
SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{CIK}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
SEC_UA = {
    "User-Agent": "MCM Analytics admin@mcm-analytics.app",
    "Accept-Encoding": "gzip, deflate",
}
CSV_PATH = eco.DATA_DIR / "strategy_funding.csv"
CSV_COLUMNS = ["kind", "item", "usd_m", "period_start", "period_end", "filed", "accession", "url"]
TICKERS = ("MSTR", "STRF", "STRC", "STRK", "STRD", "STRE")

_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_DATE = rf"{_MONTH}\s+\d{{1,2}},\s+\d{{4}}"
_NUM = r"(?:\d[\d,]*(?:\.\d+)?|-)"


@dataclass
class Funding:
    period_start: pd.Timestamp | None
    period_end: pd.Timestamp | None
    sources: dict = field(default_factory=dict)   # instrument -> net proceeds US$m
    uses: dict = field(default_factory=dict)      # use label -> US$m
    btc_bought: float | None = None
    btc_avg_price: float | None = None
    btc_holdings: float | None = None
    usd_reserve_b: float | None = None
    usd_cash_m: float | None = None
    filed: str = ""
    accession: str = ""
    url: str = ""

    @property
    def total_raised(self) -> float:
        return float(sum(self.sources.values()))

    @property
    def total_used(self) -> float:
        return float(sum(self.uses.values()))


def html_to_text(raw) -> str:
    """Visible text of an EDGAR inline-XBRL document, whitespace collapsed."""
    from lxml import html as lhtml
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="ignore")
    doc = lhtml.fromstring(raw)
    for bad in doc.xpath('//*[local-name()="header"]|//script|//style'
                         '|//*[@style and contains(translate(@style," ",""),"display:none")]'):
        bad.drop_tree()
    return re.sub(r"\s+", " ", doc.text_content()).strip()


def _f(s: str | None) -> float:
    if s is None:
        return 0.0
    s = s.replace(",", "").strip()
    return 0.0 if s in ("", "-") else float(s)


def _section(text: str, start: str, ends: tuple[str, ...]) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = len(text)
    for e in ends:
        k = text.find(e, i + len(start))
        if k >= 0:
            j = min(j, k)
    return text[i:j]


def _use_label(dest: str) -> str:
    d = dest.lower()
    if "bitcoin" in d:
        return "BTC purchases"
    m = re.search(r"repurchases? of (\w+) stock", d)
    if m:
        return f"{m.group(1).upper()} buyback"
    if "usd reserve" in d:
        return "USD Reserve"
    if "usd cash" in d or "cash" in d:
        return "USD Cash"
    if "dividend" in d:
        return "Dividends"
    return dest.strip().capitalize()


def parse_8k_text(text: str) -> Funding:
    """Parse the visible text of a weekly 8-K. Raises ValueError when it has no ATM/BTC update."""
    if "ATM Update" not in text and "BTC Update" not in text:
        raise ValueError("8-K has no ATM Update or BTC Update section")
    m = re.search(rf"During Period ({_DATE}) to ({_DATE})", text)
    p0 = pd.Timestamp(m.group(1)) if m else None
    p1 = pd.Timestamp(m.group(2)) if m else None
    out = Funding(p0, p1)

    atm = _section(text, "ATM Update", ("BTC Update", "Repurchase Program", "USD Reserve and USD Cash", "Item 7.01"))
    for t in TICKERS:
        # "MSTR Stock 2,051,348 $ - $ 283.8 (4) $ 18,337.3": shares, notional, net proceeds.
        r = re.search(rf"\b{t} Stock(?: \(\d+\))? ({_NUM}) \$ ({_NUM}) \$ ({_NUM})", atm)
        if r:
            net = _f(r.group(3))
            if net:
                out.sources[f"{t} ATM"] = net

    # Footnotes: "$81.9 million in net proceeds from MSTR Stock sales were used to fund bitcoin purchases"
    for amt, _src, dest in re.findall(
        r"\$(\d[\d,]*(?:\.\d+)?) (?:million|billion) in net proceeds from (\w+) Stock sales were "
        r"(?:used to fund|allocated to) ([^,.;]+?)(?:,| and |\.|;)", atm):
        label = _use_label(dest)
        out.uses[label] = out.uses.get(label, 0.0) + _f(amt)

    btc = _section(text, "BTC Update", ("Repurchase Program", "USD Reserve and USD Cash", "Item 7.01"))
    r = re.search(rf"Average Purchase Price \(\d\) ({_NUM}) \$ ({_NUM}) \$ ({_NUM}) ({_NUM}) \$ ({_NUM}) \$ ({_NUM})", btc)
    if r:
        out.btc_bought = _f(r.group(1))
        out.btc_avg_price = _f(r.group(3))
        out.btc_holdings = _f(r.group(4))
        spent = _f(r.group(2))
        # Footnotes are the primary split; the BTC table fills in when they are absent.
        if spent and "BTC purchases" not in out.uses:
            out.uses["BTC purchases"] = spent

    rep = _section(text, "Repurchase Program", ("USD Reserve and USD Cash", "Item 7.01"))
    for t in TICKERS:
        r = re.search(rf"\b{t} Stock(?: \(\d+\))? ({_NUM}) \$ ({_NUM})", rep)
        if r and _f(r.group(2)):
            label = f"{t} buyback"
            if label not in out.uses:
                out.uses[label] = _f(r.group(2))

    usd = _section(text, "USD Reserve and USD Cash", ("Item 7.01", "SIGNATURE"))
    r = re.search(r"balances of the USD Reserve and USD Cash were \$(\d[\d,.]*) (billion|million) and \$(\d[\d,.]*) (billion|million)", usd)
    if r:
        res = _f(r.group(1)) * (1 if r.group(2) == "billion" else 0.001)
        cash = _f(r.group(3)) * (1000 if r.group(4) == "billion" else 1)
        out.usd_reserve_b, out.usd_cash_m = res, cash

    if not out.sources and not out.uses:
        raise ValueError("ATM/BTC sections found but no amounts parsed (format change?)")
    return out


def to_frame(f: Funding) -> pd.DataFrame:
    rows = [("source", k, v) for k, v in f.sources.items()] + [("use", k, v) for k, v in f.uses.items()]
    for k, v in (("btc_bought", f.btc_bought), ("btc_avg_price", f.btc_avg_price),
                 ("btc_holdings", f.btc_holdings), ("usd_reserve_b", f.usd_reserve_b),
                 ("usd_cash_m", f.usd_cash_m)):
        if v is not None:
            rows.append(("stat", k, v))
    df = pd.DataFrame(rows, columns=["kind", "item", "usd_m"])
    df["period_start"] = f.period_start.strftime("%Y-%m-%d") if f.period_start is not None else ""
    df["period_end"] = f.period_end.strftime("%Y-%m-%d") if f.period_end is not None else ""
    df["filed"], df["accession"], df["url"] = f.filed, f.accession, f.url
    return df[CSV_COLUMNS]


def from_frame(df: pd.DataFrame) -> Funding:
    if df is None or df.empty:
        raise ValueError("empty Strategy funding table")
    first = df.iloc[0]
    def _ts(v):
        return pd.Timestamp(v) if isinstance(v, str) and v else None
    f = Funding(_ts(str(first["period_start"])), _ts(str(first["period_end"])),
                filed=str(first["filed"]), accession=str(first["accession"]), url=str(first["url"]))
    for rec in df.to_dict("records"):
        if rec["kind"] == "source":
            f.sources[rec["item"]] = float(rec["usd_m"])
        elif rec["kind"] == "use":
            f.uses[rec["item"]] = float(rec["usd_m"])
        elif rec["kind"] == "stat":
            setattr(f, rec["item"], float(rec["usd_m"]))
    return f


def recent_8k(get=None, limit: int = 8) -> list[dict]:
    """Newest-first 8-K filings from EDGAR's submissions JSON."""
    get = get or (lambda url: requests.get(url, headers=SEC_UA, timeout=20))
    r = get(SUBMISSIONS_URL)
    r.raise_for_status()
    rec = r.json()["filings"]["recent"]
    out = []
    for i, form in enumerate(rec["form"]):
        if form not in ("8-K", "8-K/A"):
            continue
        acc = rec["accessionNumber"][i]
        out.append({
            "filed": rec["filingDate"][i], "accession": acc, "items": rec.get("items", [""] * (i + 1))[i],
            "url": ARCHIVE_URL.format(cik=CIK_INT, acc=acc.replace("-", ""), doc=rec["primaryDocument"][i]),
        })
        if len(out) >= limit:
            break
    return out


def fetch_latest(get=None, limit: int = 6) -> Funding:
    """Latest 8-K that parses as a weekly funding update. Raises on failure."""
    get = get or (lambda url: requests.get(url, headers=SEC_UA, timeout=20))
    errors = []
    for filing in recent_8k(get, limit=limit + 4)[:limit]:
        if "8.01" not in str(filing["items"]):
            continue
        try:
            r = get(filing["url"])
            r.raise_for_status()
            f = parse_8k_text(html_to_text(r.content))
        except Exception as ex:
            errors.append(f"{filing['filed']}: {ex}")
            continue
        f.filed, f.accession, f.url = filing["filed"], filing["accession"], filing["url"]
        return f
    raise RuntimeError("No recent Strategy 8-K parsed as a weekly update. " + "; ".join(errors[:3]))


def load_result(*, live: bool = True, allow_csv: bool = True, get=None, now=None,
                csv_path: Path | None = None) -> eco.FetchResult:
    """Live EDGAR parse, then the committed CSV. ``df`` is the long ``to_frame`` table."""
    fetched_at = eco._now(now)
    path = Path(csv_path) if csv_path is not None else CSV_PATH
    err = None
    if live:
        try:
            f = fetch_latest(get)
            df = to_frame(f)
            return eco.FetchResult(df, "live", f.url, fetched_at, f.period_end, True,
                                   f"Strategy 8-K filed {f.filed} ({f.accession}).")
        except Exception as ex:
            err = ex
    if allow_csv and path.exists():
        try:
            df = pd.read_csv(path, dtype={"period_start": str, "period_end": str, "filed": str})
            f = from_frame(df)
            return eco.FetchResult(df, "csv", "data/strategy_funding.csv",
                                   eco.file_mtime(path) or fetched_at, f.period_end, True,
                                   (f"EDGAR parse unavailable ({err}); using " if err else "Using ")
                                   + f"data/strategy_funding.csv (8-K filed {f.filed}).")
        except Exception as ex:
            return eco._fail(CSV_COLUMNS, "data/strategy_funding.csv", fetched_at,
                             f"EDGAR parse failed ({err}) and the CSV could not be read ({ex}).")
    return eco._fail(CSV_COLUMNS, SUBMISSIONS_URL, fetched_at,
                     f"Strategy 8-K parse failed ({err})." + ("" if allow_csv else " CSV fallback was not used."))
