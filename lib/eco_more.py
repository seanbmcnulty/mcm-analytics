"""Data and maths for the five charts added 2026-11-23 from the 15-23 Nov 2026
Ecoinometrics scan (Sean approved all five). No Streamlit here.

1. US Treasury yield curve now vs 12 months ago. FRED DGS* first (one
   multi-series CSV request), then the US Treasury par yield curve CSV from
   home.treasury.gov, then ``data/yield_curve.csv`` (a committed copy of real
   FRED/Treasury prints, refreshed by ``build_yield_curve_csv``).
2. Every day's net spot BTC ETF flow in BTC (Farside US$ / that day's close),
   current month highlighted with its month-to-date total.
3. Year-to-date price returns: BTC (Deribit close), gold (GLD closing price
   from the SPDR Gold Shares historical archive), Nasdaq-100 (FRED NASDAQ100).
4. Cumulative US$ ETF flows since the 10 Jan 2024 BTC ETF approval: Farside
   total vs GLD (daily change in shares outstanding x NAV per share). IAU
   was dropped: iShares returns its product page or an Akamai 403 instead of
   the fund history download, so there is no free daily shares series for it.
5. ARIMA scenarios for the next 30 sessions of BTC ETF flows, bucketed into
   upside / continuation / downside terciles of the 30-session outcome.

Everything is real data. Nothing is filled, carried forward, or invented:
a missing day is dropped, and a failed source is reported, not replaced.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from lib import ecoinometrics as eco

DATA_DIR = eco.DATA_DIR
UA = eco.UA

# ------------------------------------------------------------- yield curve
# (FRED id, short label, years, home.treasury.gov CSV column)
YIELD_TENORS = (
    ("DGS3MO", "3M", 0.25, "3 Mo"),
    ("DGS6MO", "6M", 0.5, "6 Mo"),
    ("DGS1", "1Y", 1.0, "1 Yr"),
    ("DGS2", "2Y", 2.0, "2 Yr"),
    ("DGS3", "3Y", 3.0, "3 Yr"),
    ("DGS5", "5Y", 5.0, "5 Yr"),
    ("DGS7", "7Y", 7.0, "7 Yr"),
    ("DGS10", "10Y", 10.0, "10 Yr"),
    ("DGS20", "20Y", 20.0, "20 Yr"),
    ("DGS30", "30Y", 30.0, "30 Yr"),
)
YIELD_IDS = tuple(t[0] for t in YIELD_TENORS)
YIELD_LABELS = tuple(t[1] for t in YIELD_TENORS)
TREASURY_CSV = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
                "&field_tdr_date_value={year}&page&_format=csv")
YIELD_CSV = "yield_curve.csv"


def parse_fred_multi(text: str, start=None) -> pd.DataFrame:
    """FRED fredgraph CSV with several ids -> DataFrame[date, 3M .. 30Y] (percent)."""
    df = pd.read_csv(io.StringIO(text))
    df = df.rename(columns={df.columns[0]: "date"})
    missing = [sid for sid in YIELD_IDS if sid not in df.columns]
    if missing:
        raise ValueError(f"FRED CSV missing {', '.join(missing)}")
    out = pd.DataFrame({"date": pd.to_datetime(df["date"], errors="coerce")})
    for sid, label, _, _ in YIELD_TENORS:
        out[label] = pd.to_numeric(df[sid], errors="coerce")
    out = out.dropna(subset=["date"])
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    out = out.dropna(subset=list(YIELD_LABELS), how="all")
    return out.sort_values("date").reset_index(drop=True)


def parse_treasury_csv(text: str) -> pd.DataFrame:
    """home.treasury.gov daily par yield curve CSV -> DataFrame[date, 3M .. 30Y]."""
    df = pd.read_csv(io.StringIO(text))
    cols = {str(c).strip(): c for c in df.columns}
    if "Date" not in cols:
        raise ValueError("Treasury CSV has no Date column")
    out = pd.DataFrame({"date": pd.to_datetime(df[cols["Date"]], format="%m/%d/%Y", errors="coerce")})
    for _, label, _, tcol in YIELD_TENORS:
        if tcol not in cols:
            raise ValueError(f"Treasury CSV has no '{tcol}' column")
        out[label] = pd.to_numeric(df[cols[tcol]], errors="coerce")
    out = out.dropna(subset=["date"]).dropna(subset=list(YIELD_LABELS), how="all")
    return out.sort_values("date").reset_index(drop=True)


def fetch_yield_curve_result(*, get=None, now=None, timeout: int = 20, allow_csv: bool = True,
                             live: bool = True) -> eco.FetchResult:
    """FRED (one request for all ten ids), then home.treasury.gov, then the committed CSV.

    One attempt per source with a hard timeout, so a dead FRED costs about
    ``timeout`` seconds, not ten series x retries.
    """
    get = get or requests.get
    fetched_at = eco._now(now)
    start = (fetched_at.tz_convert(None) - pd.DateOffset(years=2)).normalize()
    cols = ["date", *YIELD_LABELS]
    errors = []
    if live:
        try:
            r = get(eco.FRED_URL, params={"id": ",".join(YIELD_IDS)}, headers=UA, timeout=timeout)
            r.raise_for_status()
            df = parse_fred_multi(r.text, start=start)
            if df.empty:
                raise ValueError("no rows")
            return eco.FetchResult(df, "live", f"{eco.FRED_URL}?id={','.join(YIELD_IDS)}", fetched_at,
                                   eco._as_of(df, "date"), True, f"FRED DGS* {len(df)} rows.")
        except Exception as ex:
            errors.append(f"FRED: {ex}")
        try:
            year = fetched_at.year
            frames = []
            for y in (year - 2, year - 1, year):
                r = get(TREASURY_CSV.format(year=y), headers=UA, timeout=timeout)
                r.raise_for_status()
                frames.append(parse_treasury_csv(r.text))
            df = pd.concat(frames).drop_duplicates("date", keep="last").sort_values("date")
            df = df[df["date"] >= start].reset_index(drop=True)
            if df.empty:
                raise ValueError("no rows")
            return eco.FetchResult(df, "live", "home.treasury.gov daily par yield curve", fetched_at,
                                   eco._as_of(df, "date"), True,
                                   f"FRED unavailable ({errors[0]}); using US Treasury par yields, {len(df)} rows.")
        except Exception as ex:
            errors.append(f"Treasury: {ex}")
    path = DATA_DIR / YIELD_CSV
    if allow_csv and path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["date"])[cols].sort_values("date").reset_index(drop=True)
            return eco.FetchResult(df, "csv", f"data/{YIELD_CSV}", eco.file_mtime(path) or fetched_at,
                                   eco._as_of(df, "date"), True,
                                   f"Live sources failed ({'; '.join(errors) or 'not tried'}); using data/{YIELD_CSV}.")
        except Exception as ex:
            errors.append(f"data/{YIELD_CSV}: {ex}")
    return eco._fail(cols, "FRED / home.treasury.gov", fetched_at,
                     "Yield curve unavailable (" + "; ".join(errors or ["no source"]) + ").")


def build_yield_curve_csv(*, data_dir: Path | None = None, get=None, now=None, log=None) -> eco.FetchResult:
    """Live fetch only; on success rewrite ``data/yield_curve.csv`` (the page fallback)."""
    result = fetch_yield_curve_result(get=get, now=now, allow_csv=False)
    if result.ok:
        folder = Path(data_dir) if data_dir is not None else DATA_DIR
        out = result.df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(folder / YIELD_CSV, index=False)
        result.message += f" Wrote data/{YIELD_CSV}."
    if log:
        log(result.message)
    return result


@dataclass
class CurveCompare:
    table: pd.DataFrame          # tenor, years, now, ago, change (percentage points)
    now_date: pd.Timestamp
    ago_date: pd.Timestamp
    avg_change: float            # mean change across tenors, percentage points


def curve_compare(df: pd.DataFrame, years_back: int = 1) -> CurveCompare | None:
    """Latest date with every tenor printed vs the last full curve on or before
    that date minus ``years_back`` years. No interpolation, no fill."""
    if df is None or df.empty:
        return None
    full = df.dropna(subset=list(YIELD_LABELS)).sort_values("date")
    if full.empty:
        return None
    now_row = full.iloc[-1]
    target = pd.Timestamp(now_row["date"]) - pd.DateOffset(years=years_back)
    past = full[full["date"] <= target]
    if past.empty:
        return None
    ago_row = past.iloc[-1]
    if (target - pd.Timestamp(ago_row["date"])).days > 10:
        return None
    t = pd.DataFrame({
        "tenor": list(YIELD_LABELS),
        "years": [x[2] for x in YIELD_TENORS],
        "now": [float(now_row[c]) for c in YIELD_LABELS],
        "ago": [float(ago_row[c]) for c in YIELD_LABELS],
    })
    t["change"] = t["now"] - t["ago"]
    return CurveCompare(t, pd.Timestamp(now_row["date"]), pd.Timestamp(ago_row["date"]),
                        float(t["change"].mean()))


# --------------------------------------------------------- daily flow dots
def flow_dots(flows_btc: pd.DataFrame, month=None) -> tuple[pd.DataFrame, dict]:
    """``flows_to_btc`` output -> (frame with ``in_month`` flag, month stats).

    ``month`` defaults to the calendar month of the latest Farside session.
    Stats: month (Timestamp, first day), total (BTC), sessions, inflow_days,
    outflow_days, last_date.
    """
    if flows_btc is None or len(flows_btc) == 0:
        return pd.DataFrame(columns=["date", "flow", "in_month"]), {}
    d = flows_btc[["date", "flow"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    last = d["date"].iloc[-1]
    m = pd.Timestamp(month) if month is not None else pd.Timestamp(last.year, last.month, 1)
    m = pd.Timestamp(m.year, m.month, 1)
    d["in_month"] = (d["date"].dt.year == m.year) & (d["date"].dt.month == m.month)
    cur = d[d["in_month"]]
    stats = {
        "month": m,
        "total": float(cur["flow"].sum()) if len(cur) else 0.0,
        "sessions": int(len(cur)),
        "inflow_days": int((cur["flow"] > 0).sum()),
        "outflow_days": int((cur["flow"] < 0).sum()),
        "last_date": last,
    }
    return d, stats


# ------------------------------------------------------------- GLD archive
GLD_ARCHIVE_URL = "https://api.spdrgoldshares.com/api/v1/historical-archive?product=gld&exchange=NYSE&lang=en"
GLD_SHEET = "US GLD Historical Archive"


def parse_gld_archive(content: bytes) -> pd.DataFrame:
    """SPDR GLD historical archive xlsx -> DataFrame[date, close, nav_ps, total_nav, shares].

    ``shares`` is total NAV / NAV per share, which lands on whole creation
    baskets. Holiday rows (no NAV) are dropped, not filled.
    """
    sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
    raw = sheets.get(GLD_SHEET)
    if raw is None:
        raw = next((v for k, v in sheets.items() if "Archive" in str(k)), None)
    if raw is None:
        raise ValueError("GLD archive sheet not found")
    cols = {str(c).strip().lower(): c for c in raw.columns}

    def col(prefix):
        for k, c in cols.items():
            if k.startswith(prefix):
                return c
        raise ValueError(f"GLD archive has no '{prefix}' column")

    out = pd.DataFrame({
        "date": pd.to_datetime(raw[col("date")].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce"),
        "close": pd.to_numeric(raw[col("closing price")], errors="coerce"),
        "nav_ps": pd.to_numeric(raw[col("nav/share")], errors="coerce"),
        "total_nav": pd.to_numeric(raw[col("total net asset value")], errors="coerce"),
    })
    out = out.dropna(subset=["date", "nav_ps", "total_nav"])
    out = out[(out["nav_ps"] > 0) & (out["total_nav"] > 0)]
    out["shares"] = (out["total_nav"] / out["nav_ps"]).round(-3)
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def fetch_gld_result(*, get=None, now=None, timeout: int = 40) -> eco.FetchResult:
    """Live only. The archive says redistribution is not allowed, so no copy is committed."""
    get = get or requests.get
    fetched_at = eco._now(now)
    cols = ["date", "close", "nav_ps", "total_nav", "shares"]
    try:
        r = get(GLD_ARCHIVE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        r.raise_for_status()
        df = parse_gld_archive(r.content)
        if df.empty:
            raise ValueError("no rows")
        return eco.FetchResult(df, "live", "spdrgoldshares.com GLD historical archive", fetched_at,
                               eco._as_of(df, "date"), True, f"GLD archive {len(df)} rows.")
    except Exception as ex:
        return eco._fail(cols, "spdrgoldshares.com GLD historical archive", fetched_at,
                         f"GLD archive unavailable ({type(ex).__name__}: {ex}).")


def gld_flows_usd(gld: pd.DataFrame) -> pd.DataFrame:
    """Daily GLD creations/redemptions in US$: change in shares outstanding x that day's NAV/share."""
    if gld is None or len(gld) < 2:
        return pd.DataFrame(columns=["date", "flow_usd"])
    g = gld.sort_values("date").reset_index(drop=True)
    flow = g["shares"].diff() * g["nav_ps"]
    out = pd.DataFrame({"date": g["date"], "flow_usd": flow}).iloc[1:]
    return out.dropna().reset_index(drop=True)


ETF_APPROVAL = pd.Timestamp("2024-01-10")


def cumulative_flows(btc_flows_musd: pd.DataFrame | None, gold_flows_usd: pd.DataFrame | None,
                     since=ETF_APPROVAL) -> pd.DataFrame:
    """Cumulative US$ billions from the session after ``since``.

    Returns DataFrame[date, series, cum_bn]; ``series`` is ``btc`` or ``gold``.
    BTC is Farside US$ millions; gold is ``gld_flows_usd`` output.
    """
    since = pd.Timestamp(since)
    parts = []
    if btc_flows_musd is not None and len(btc_flows_musd):
        b = btc_flows_musd[["date", "flow"]].copy()
        b["date"] = pd.to_datetime(b["date"])
        b = b[b["date"] > since].sort_values("date")
        b = b[np.isfinite(pd.to_numeric(b["flow"], errors="coerce"))]
        parts.append(pd.DataFrame({"date": b["date"], "series": "btc", "cum_bn": b["flow"].cumsum() / 1e3}))
    if gold_flows_usd is not None and len(gold_flows_usd):
        g = gold_flows_usd.copy()
        g["date"] = pd.to_datetime(g["date"])
        g = g[g["date"] > since].sort_values("date")
        parts.append(pd.DataFrame({"date": g["date"], "series": "gold", "cum_bn": g["flow_usd"].cumsum() / 1e9}))
    if not parts:
        return pd.DataFrame(columns=["date", "series", "cum_bn"])
    return pd.concat(parts).reset_index(drop=True)


# ------------------------------------------------------------- YTD returns
def ytd_returns(series: dict[str, pd.Series], year: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Price return since the last close of the previous year, in percent.

    ``series`` maps a name to a close Series indexed by date. Returns
    (DataFrame[date, name, ret], {name: {"base_date", "base", "last_date", "last", "ret"}}).
    A series with no close before 1 Jan of ``year`` is left out (no proxy base).
    """
    rows, info = [], {}
    if year is None:
        latest = [pd.Timestamp(s.index.max()) for s in series.values() if s is not None and len(s)]
        if not latest:
            return pd.DataFrame(columns=["date", "name", "ret"]), {}
        year = max(latest).year
    jan1 = pd.Timestamp(year, 1, 1)
    for name, s in series.items():
        if s is None or len(s) == 0:
            continue
        s = pd.to_numeric(pd.Series(s), errors="coerce")
        s.index = pd.to_datetime(s.index).normalize()
        s = s[np.isfinite(s.to_numpy()) & (s.to_numpy() > 0)].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        before = s[s.index < jan1]
        if before.empty or (jan1 - before.index[-1]).days > 7:
            continue
        base_date, base = before.index[-1], float(before.iloc[-1])
        cur = s[(s.index >= jan1) & (s.index < pd.Timestamp(year + 1, 1, 1))]
        if cur.empty:
            continue
        r = (cur / base - 1.0) * 100.0
        rows.append(pd.DataFrame({"date": [base_date, *cur.index], "name": name, "ret": [0.0, *r.values]}))
        info[name] = {"base_date": base_date, "base": base, "last_date": cur.index[-1],
                      "last": float(cur.iloc[-1]), "ret": float(r.iloc[-1])}
    if not rows:
        return pd.DataFrame(columns=["date", "name", "ret"]), info
    return pd.concat(rows).reset_index(drop=True), info


# ---------------------------------------------------------- flow scenarios
SCENARIO_ORDERS = ((1, 0, 0), (1, 0, 1), (2, 0, 1), (0, 0, 1))
SCENARIO_NAMES = ("upside", "continuation", "downside")


@dataclass
class FlowScenarios:
    actual: pd.DataFrame                     # date, flow, roll (30-session sum, BTC)
    future_dates: list                       # next ``horizon`` weekdays
    bands: dict                              # name -> DataFrame[date, median, p5, p95]
    finals: dict                             # name -> median final 30-session sum
    cutoffs: tuple                           # (33.3rd, 66.7th) percentiles of the outcome
    order: tuple
    aic: float
    n_paths: int
    fit_sessions: int
    fit_from: pd.Timestamp
    fit_to: pd.Timestamp
    seed: int
    params: dict = field(default_factory=dict)


def simulate_flow_scenarios(flows_btc: pd.DataFrame, *, horizon: int = 30, window: int = 30,
                            n_paths: int = 10_000, fit_sessions: int = 252, seed: int = 20261123,
                            orders=SCENARIO_ORDERS) -> FlowScenarios:
    """ARIMA on daily BTC ETF flows -> tercile scenarios for the rolling 30-session flow.

    Fit: the last ``fit_sessions`` Farside sessions (BTC), each order in
    ``orders`` with a constant, lowest AIC wins. Simulate ``n_paths`` paths of
    ``horizon`` sessions from the end of the sample with Gaussian shocks.
    Each path's rolling 30-session flow is the actual sessions still inside
    the window plus its simulated sessions. Paths are split into terciles by
    their simulated total (the rolling sum ``horizon`` sessions ahead).
    """
    from statsmodels.tsa.arima.model import ARIMA
    import warnings

    if window != horizon:
        raise ValueError("window and horizon must match")
    d = flows_btc[["date", "flow"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    d["roll"] = d["flow"].rolling(window, min_periods=window).sum()
    fit = d.tail(fit_sessions)
    if len(fit) < max(60, window + 10):
        raise ValueError(f"need at least 60 sessions to fit, have {len(fit)}")
    y = fit["flow"].to_numpy(dtype=float)
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in orders:
            try:
                res = ARIMA(y, order=order, trend="c").fit()
            except Exception:
                continue
            if np.isfinite(res.aic) and (best is None or res.aic < best[1].aic):
                best = (order, res)
        if best is None:
            raise RuntimeError("no ARIMA order converged")
        order, res = best
        kw = dict(nsimulations=horizon, repetitions=n_paths, anchor="end")
        try:   # statsmodels >= 0.15 names the generator ``rng``
            sims = res.simulate(**kw, rng=np.random.default_rng(seed))
        except TypeError:
            sims = res.simulate(**kw, random_state=np.random.default_rng(seed))
        sims = np.asarray(sims)
    sims = sims.reshape(horizon, -1)[:, :n_paths]          # (horizon, paths)
    last_flows = d["flow"].to_numpy(dtype=float)[-window:]
    # rolling sum h sessions ahead: actual sessions h..window-1 plus simulated 0..h-1
    tail_sums = np.array([last_flows[h:].sum() for h in range(1, horizon + 1)])
    roll = tail_sums[:, None] + np.cumsum(sims, axis=0)     # (horizon, paths)
    outcome = roll[-1]
    q1, q2 = np.quantile(outcome, [1 / 3, 2 / 3])
    groups = {"upside": outcome > q2, "continuation": (outcome > q1) & (outcome <= q2),
              "downside": outcome <= q1}
    start = d["date"].iloc[-1] + pd.offsets.BDay(1)
    future = list(pd.bdate_range(start, periods=horizon))
    bands, finals = {}, {}
    for name, mask in groups.items():
        sub = roll[:, mask]
        bands[name] = pd.DataFrame({
            "date": future,
            "median": np.median(sub, axis=1),
            "p5": np.quantile(sub, 0.05, axis=1),
            "p95": np.quantile(sub, 0.95, axis=1),
        })
        finals[name] = float(np.median(sub[-1]))
    params = {k: float(v) for k, v in zip(res.param_names, res.params)}
    return FlowScenarios(d[["date", "flow", "roll"]], future, bands, finals, (float(q1), float(q2)),
                         tuple(order), float(res.aic), int(n_paths), int(len(fit)),
                         pd.Timestamp(fit["date"].iloc[0]), pd.Timestamp(fit["date"].iloc[-1]), seed, params)
