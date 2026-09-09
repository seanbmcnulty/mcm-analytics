#!/usr/bin/env python3
"""Incrementally update data/macro_events_calendar.csv with new FOMC / CPI YoY /
NFP releases since the last row on file.

This is this app's own builder, NOT a port of exodus-analytics's
tools/build_macro_calendar.py -- that script targets a different schema (CPI
m/m, midpoint-of-range FOMC actual, NY Fed SPD + Cleveland Fed nowcast
consensus tiers) for a file this app doesn't use. This app's CSV
(``date,event,actual,consensus,prior,currency``) uses:

  FOMC Rate Decision   actual = UPPER BOUND of the new target range (e.g. a
                        cut to 4.00-4.25% is recorded as 4.25), not the
                        midpoint. This matches every row hand-curated into
                        the file before this script existed.
  CPI YoY               actual = the headline (all-items) 12-month percent
                        change, not seasonally adjusted, as BLS states it
                        in the first paragraph of the release ("Over the
                        last 12 months, the all items index increased X
                        percent"). NOT the month-over-month figure.
  NFP                   actual = change in total nonfarm payrolls, in
                        thousands, as first reported.

Consensus: a real Bloomberg/Reuters survey median is not obtainable for
free. The rows already in the file up to 2026-09 carry real
researched-from-news-coverage consensus figures. This script cannot
reproduce that manually -- it fills consensus with the PRIOR release's
actual (a random-walk expectation) for CPI/NFP, clearly labelled as the
weakest tier in exactly the sense exodus's own script documents. FOMC
consensus is set equal to actual, matching the convention already
established in this file (every FOMC decision on record here was the
fully-priced-in outcome per CME FedWatch / economist surveys at the time).
If a meeting genuinely surprises markets, this will silently record it as
"expected" -- a known limitation, not a bug to chase without a real
survey-median source.

Only appends rows strictly after the latest date already in the CSV, so
re-running this is always safe and cheap.

Usage:
    tools/build_macro_calendar.py                  # update data/macro_events_calendar.csv in place
    tools/build_macro_calendar.py --dry-run         # print what would be added, write nothing
    tools/build_macro_calendar.py --csv other.csv   # target a different file
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data" / "macro_events_calendar.csv"

WEB_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
BLS_UA = "mcm-analytics/1.0 (macro events calendar updater; personal analytics dashboard)"

FOMC_CALENDARS = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FOMC_HISTORICAL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
FOMC_STATEMENT = "https://www.federalreserve.gov/newsevents/pressreleases/monetary{stamp}a.htm"
BLS_SCHEDULE = "https://www.bls.gov/schedule/{year}/home.htm"
BLS_CPI_ARCHIVE = "https://www.bls.gov/news.release/archives/cpi_{stamp}.htm"
BLS_EMPSIT_ARCHIVE = "https://www.bls.gov/news.release/archives/empsit_{stamp}.htm"

EVENT_FOMC = "FOMC Rate Decision"
EVENT_CPI = "CPI YoY"
EVENT_NFP = "NFP"
FIELDS = ["date", "event", "actual", "consensus", "prior", "currency"]

_FRACTION = r"\d+(?:-\d+/\d+)?|\d+/\d+"
_DECISION_RE = re.compile(
    r"(?:maintain|lower|rais|keep|increas|reduc)\w*\s+the\s+target\s+range\s+for\s+the\s+federal\s+funds\s+rate",
    re.I,
)
_RANGE_AFTER_RE = re.compile(
    r"target range for the federal funds rate[^.]{0,120}?(" + _FRACTION + r")\s+to\s+(" + _FRACTION + r")\s+percent",
    re.I,
)
_CPI_12MO_RE = re.compile(
    r"[Oo]ver the last 12 months,?\s+the all items index\s+"
    r"(increased|rose|declined|fell|decreased|was unchanged|remained unchanged)"
    r"(?:\s+([\d.]+)\s+percent)?",
    re.I,
)
_NFP_ANCHOR_RE = re.compile(r"[Tt]otal nonfarm payroll employment", re.S)
_NFP_PAREN_RE = re.compile(r"\(([+\-−])\s?([\d,]+)\)")
_NFP_VERB_RE = re.compile(
    r"\b(increased|rose|grew|gained|added|declined|fell|decreased|dropped|edged up|edged down)\b"
    r"\s+(?:by\s+)?([\d,]+(?:\.\d+)?)\s*(million|thousand)?",
    re.I,
)
_NEGATIVE = {"declined", "fell", "decreased", "dropped", "edged down"}

_session = None


def fetch(url: str, *, ua: str = WEB_UA, tries: int = 4, timeout: int = 45) -> Optional[str]:
    """GET with retries. Returns None on a 404 or after exhausting retries."""
    global _session
    import requests

    if _session is None:
        _session = requests.Session()
    last = None
    for attempt in range(tries):
        try:
            resp = _session.get(url, headers={"User-Agent": ua}, timeout=timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp.text
        except Exception as exc:  # noqa: BLE001 - retry anything transient
            last = exc
            time.sleep(2 * (attempt + 1))
    print(f"warning: giving up on {url}: {type(last).__name__}: {last}", file=sys.stderr)
    return None


def visible_text(html: str) -> str:
    txt = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = txt.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", txt)


def parse_fraction(token: str) -> float:
    token = token.strip()
    if "-" in token:
        whole, frac = token.split("-", 1)
        num, den = frac.split("/")
        return int(whole) + int(num) / int(den)
    if "/" in token:
        num, den = token.split("/")
        return int(num) / int(den)
    return float(token)


def round_half_up(value: float, places: int) -> float:
    quant = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# FOMC
# ---------------------------------------------------------------------------
def fomc_statement_dates(start_year: int) -> List[str]:
    stamps: set[str] = set()
    pages = [FOMC_CALENDARS] + [FOMC_HISTORICAL.format(year=y) for y in range(start_year, date.today().year + 1)]
    for url in pages:
        html = fetch(url)
        if not html:
            continue
        stamps.update(re.findall(r"monetary(\d{8})a\.htm", html))
    return sorted(s for s in stamps if int(s[:4]) >= start_year)


def collect_fomc(start_year: int) -> List[dict]:
    rows = []
    for stamp in fomc_statement_dates(start_year):
        html = fetch(FOMC_STATEMENT.format(stamp=stamp))
        if not html:
            continue
        text = visible_text(html).replace("‑", "-").replace("–", "-").replace("—", "-")
        if not _DECISION_RE.search(text):
            continue
        range_m = _RANGE_AFTER_RE.search(text)
        if not range_m:
            print(f"warning: no target range parsed from FOMC statement {stamp}", file=sys.stderr)
            continue
        upper = parse_fraction(range_m.group(2))
        rows.append({
            "date": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}",
            "event": EVENT_FOMC,
            "actual": round_half_up(upper, 2),
        })
    return rows


# ---------------------------------------------------------------------------
# BLS: schedule, then the first print out of the archived release
# ---------------------------------------------------------------------------
_SCHED_DATE_RE = re.compile(r"^[A-Z][a-z]+day, [A-Z][a-z]+ \d{1,2}, \d{4}$")
_SCHED_TIME_RE = re.compile(r"^(\d{1,2}:\d{2} [AP]M)$")
WANTED_RELEASES = {"Employment Situation", "Consumer Price Index"}


def bls_schedule(year: int) -> List[Tuple[date, str]]:
    html = fetch(BLS_SCHEDULE.format(year=year), ua=BLS_UA)
    if not html:
        return []
    packed = re.sub(r"<[^>]+>", "|", html)
    packed = re.sub(r"[|\s]*\|[|\s]*", "|", packed)
    cells = [c.strip() for c in packed.split("|")]

    out = []
    current: Optional[date] = None
    for i, cell in enumerate(cells):
        if _SCHED_DATE_RE.match(cell):
            current = datetime.strptime(cell, "%A, %B %d, %Y").date()
            continue
        if not _SCHED_TIME_RE.match(cell) or current is None or i + 1 >= len(cells):
            continue
        name = cells[i + 1]
        if name in WANTED_RELEASES:
            out.append((current, name))
    return sorted(set(out))


def parse_cpi_yoy(text: str) -> Optional[float]:
    """Headline (all-items) 12-month percent change, not seasonally adjusted."""
    m = _CPI_12MO_RE.search(text)
    if not m:
        return None
    verb = m.group(1).lower()
    if "unchanged" in verb:
        return 0.0
    if m.group(2) is None:
        return None
    value = float(m.group(2))
    return -value if verb in _NEGATIVE else value


def parse_nfp_actual(text: str) -> Optional[float]:
    anchor = _NFP_ANCHOR_RE.search(text)
    if not anchor:
        return None
    clause = text[anchor.end():anchor.end() + 160]
    paren = _NFP_PAREN_RE.search(clause)
    verb_m = _NFP_VERB_RE.search(clause)
    if paren and (not verb_m or paren.start() < verb_m.start()):
        value = int(paren.group(2).replace(",", "")) / 1000.0
        return -value if paren.group(1) in "-−" else value
    if not verb_m:
        return None
    value = float(verb_m.group(2).replace(",", ""))
    scale = (verb_m.group(3) or "").lower()
    if scale == "million":
        value *= 1000.0
    elif scale != "thousand":
        value /= 1000.0
    return -value if verb_m.group(1).lower() in _NEGATIVE else value


def collect_bls(start_year: int, end_year: int) -> List[dict]:
    rows = []
    today = date.today()
    for year in range(start_year, end_year + 1):
        for when, name in bls_schedule(year):
            if when > today:
                continue
            is_cpi = name == "Consumer Price Index"
            event = EVENT_CPI if is_cpi else EVENT_NFP
            stamp = when.strftime("%m%d%Y")
            url = (BLS_CPI_ARCHIVE if is_cpi else BLS_EMPSIT_ARCHIVE).format(stamp=stamp)
            html = fetch(url, ua=BLS_UA)
            if not html:
                # Not yet archived under this stamp (e.g. current release still
                # lives at the non-archived cpi.nr0.htm / empsit.nr0.htm URL,
                # or the release was delayed/skipped, e.g. the Oct 2025 CPI
                # report BLS cancelled outright during the government
                # shutdown). Skip rather than guess.
                continue
            text = visible_text(html)
            actual = parse_cpi_yoy(text) if is_cpi else parse_nfp_actual(text)
            if actual is None:
                print(f"warning: no actual parsed for {event} {when}", file=sys.stderr)
                continue
            rows.append({"date": when.isoformat(), "event": event, "actual": actual})
    return rows


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def load_existing(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def attach_consensus_and_prior(new_rows: List[dict], last_actual: Dict[str, float]) -> None:
    """Fill consensus/prior in place, using the running last-actual-by-event
    map (seeded from the existing CSV, updated as new rows are processed in
    date order)."""
    for row in sorted(new_rows, key=lambda r: (r["date"], r["event"])):
        event = row["event"]
        prior = last_actual.get(event)
        row["prior"] = "" if prior is None else f"{prior:g}"
        if event == EVENT_FOMC:
            # Every FOMC decision in this file's history was the fully
            # priced-in outcome per CME FedWatch / economist surveys as of
            # decision day - see module docstring. Not a real survey median.
            row["consensus"] = f"{row['actual']:g}"
        else:
            # Weakest tier: prior release's actual, i.e. a random-walk
            # expectation - real Reuters/Bloomberg consensus is not free to
            # obtain. See module docstring.
            row["consensus"] = "" if prior is None else f"{prior:g}"
        row["currency"] = "USD"
        row["actual"] = f"{row['actual']:g}"
        last_actual[event] = float(row["actual"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--start", type=int, default=2024, help="first calendar year to scan (kept small; only rows after the CSV's last date are ever written)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    existing = load_existing(args.csv)
    if not existing:
        print(f"error: {args.csv} has no existing rows to anchor off; run the full exodus-style backfill first", file=sys.stderr)
        return 1

    last_date = max(r["date"] for r in existing)
    last_actual: Dict[str, float] = {}
    for r in sorted(existing, key=lambda r: r["date"]):
        try:
            last_actual[r["event"]] = float(r["actual"])
        except (KeyError, ValueError):
            pass

    today_year = date.today().year
    print(f"fetching FOMC statements since {args.start} ...", file=sys.stderr)
    fomc_rows = collect_fomc(args.start)
    print(f"fetching BLS CPI/NFP releases since {args.start} ...", file=sys.stderr)
    bls_rows = collect_bls(args.start, today_year)

    all_rows = fomc_rows + bls_rows
    new_rows = [r for r in all_rows if r["date"] > last_date]
    if not new_rows:
        print(f"no new rows after {last_date}; nothing to do")
        return 0

    attach_consensus_and_prior(new_rows, last_actual)

    combined = existing + [{k: r.get(k, "") for k in FIELDS} for r in new_rows]
    combined.sort(key=lambda r: (r["date"], r["event"]))

    print(f"{len(new_rows)} new row(s) after {last_date}:", file=sys.stderr)
    for r in sorted(new_rows, key=lambda r: r["date"]):
        print(f"  {r['date']} {r['event']:20s} actual={r['actual']} consensus={r['consensus']} prior={r['prior']}", file=sys.stderr)

    if args.dry_run:
        print("dry run - not writing", file=sys.stderr)
        return 0

    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(combined)
    print(f"wrote {len(combined)} total rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
