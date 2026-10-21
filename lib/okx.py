"""
OKX public REST client — block-trade tape for options (BTC, ETH, SOL, XAU).

No auth needed.  Used by ``pages/02_Block_Trades_-_OKX.py`` via
``lib.block_flow.render_page(okx.VENUE)``.

Public endpoints used (all GET ``https://www.okx.com/api/v5/...``, no auth):
  rfq/public-trades               the block tape: one row per block package (RFQ), each with
                                  1..n legs; ``side`` is the *taker's* direction on that leg;
                                  cursor-paged newest-first with ``endId`` (100/page, 5 req / 2 s).
                                  Published ~15 minutes after execution.
  public/block-trades?instId=     single-instrument block prints WITH fillVol (IV), markPx, idxPx,
                                  fwdPx — joined to the tape on ``tradeId`` to get mark / IV / index
                                  at the time of the trade (20 req / 2 s).
  public/instruments?uly=         contract spec (ctVal * ctMult = underlying units per contract).
  public/underlying               option underlyings that exist (BTC-USD, ETH-USD, SOL-USD, XAU-USD).
  market/history-index-candles    index OHLC for the chart overlay / index fallback.
  market/index-tickers            live index price.

Units: OKX has two option families per underlying, each with its own contract size:
* ``BTC-USD`` / ``ETH-USD`` / ``SOL-USD`` / ``XAU-USD`` - *inverse* (coin-margined): ``px`` is quoted in the
  underlying coin, so ``price_usd = px * idxPx``;
* ``BTC-USD_UM`` / ``ETH-USD_UM`` - *linear* (USD-settled, ``ctType=linear``): ``px`` is already USD.
``sz`` is in contracts: ``abs_amount = sz * ctVal * ctMult`` (checked 2026-10-21: BTC 0.01, ETH 0.1 but
ETH ``_UM`` 0.01, SOL 0.1, XAU 0.01 - read per family from ``public/instruments``, never assumed per underlying).

Limitations:
* only option legs are charted; futures / swap / spot hedge legs that ride in the same package are
  counted (``meta['non_option_legs']``) but not charted.
* the tape itself carries no mark / IV / index; those come from the per-instrument join and are
  NaN (IV backed out of price by lib.block_flow, index from candles) for any leg the join misses.
"""

from __future__ import annotations

import concurrent.futures
import re
import threading
import time
from datetime import timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

API_BASE = "https://www.okx.com/api/v5"
SGT = timezone(timedelta(hours=8))

ASSETS = ["BTC", "ETH", "SOL"]
# Default minimum size = OKX's minimum order (minSz = 1 contract) in underlying units; the tape is already
# blocks only, so this just shows everything.  ETH uses the smaller 0.01 (its _UM family) so nothing is hidden.
DEFAULT_MIN_SIZES: Dict[str, float] = {"BTC": 0.01, "ETH": 0.01, "SOL": 0.1, "XAU": 0.01}

# family -> (underlying units per contract, linear?), used only if public/instruments is unreachable.
_SPEC_FALLBACK = {"BTC-USD": (0.01, False), "ETH-USD": (0.1, False), "SOL-USD": (0.1, False),
                  "XAU-USD": (0.01, False), "BTC-USD_UM": (0.01, True), "ETH-USD_UM": (0.01, True)}
_ENRICH_CAP = 400          # max instruments looked up for mark/IV (rate-limited endpoint)

_session = requests.Session()
_gap_lock = threading.Lock()
_last_call: Dict[str, float] = {}
_GAPS = {"/rfq/public-trades": 0.45, "/public/block-trades": 0.12,
         "/market/history-index-candles": 0.25}   # seconds between calls, per endpoint (IP rate limits)
_spec_cache: Dict[str, Dict[str, Tuple[float, bool]]] = {}


def _throttle(path: str) -> None:
    gap = _GAPS.get(path, 0.0)
    if not gap:
        return
    with _gap_lock:
        wait = _last_call.get(path, 0.0) + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[path] = time.monotonic()


def _get(path: str, params: Dict, timeout: int = 30, retries: int = 4) -> Dict:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            _throttle(path)
            r = _session.get(f"{API_BASE}{path}", params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            j = r.json()
            if str(j.get("code")) in ("50011", "50061"):      # OKX "too many requests"
                raise requests.HTTPError(f"rate limited ({j.get('code')})")
            if str(j.get("code", "0")) != "0":
                raise RuntimeError(f"{j.get('code')}: {j.get('msg')}")
            return j
        except RuntimeError as e:          # API-level error: retrying won't help
            last = e
            break
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"OKX {path} failed: {last}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_OPT_RE = re.compile(r"^(?P<base>[A-Z0-9]+)-USD(?P<um>_UM)?-(?P<exp>\d{6})-(?P<strike>[0-9.]+)-(?P<cp>[CP])$")


def parse_option(inst_id: str):
    """'BTC-USD-261106-89000-C' -> (base, expiry Timestamp, strike, 'C'/'P'); non-options -> (None, NaT, nan, None)."""
    m = _OPT_RE.match(inst_id or "")
    if not m:
        return None, pd.NaT, np.nan, None
    return (m["base"], pd.to_datetime(m["exp"], format="%y%m%d", errors="coerce"),
            float(m["strike"]), m["cp"])


def family(inst_id: str) -> Optional[str]:
    """'ETH-USD_UM-261021-2050-C' -> 'ETH-USD_UM'; 'BTC-USD-261106-89000-C' -> 'BTC-USD'."""
    m = _OPT_RE.match(inst_id or "")
    return f"{m['base']}-USD{m['um'] or ''}" if m else None


def _f(x) -> float:
    try:
        return float(x) if x not in (None, "") else np.nan
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------
def fetch_block_tape(start_ms: int, end_ms: int, max_pages: int = 80) -> Tuple[List[Dict], bool]:
    """Block packages with cTime in [start_ms, end_ms], newest-first -> (rows, truncated)."""
    rows: List[Dict] = []
    params: Dict = {"limit": 100}
    truncated = True
    for _ in range(max_pages):
        data = _get("/rfq/public-trades", params).get("data", [])
        rows.extend(data)
        ids = [d["blockTdId"] for d in data if d.get("blockTdId")]
        if len(data) < 100 or not ids or min(int(d["cTime"]) for d in data) < start_ms:
            truncated = False
            break
        params = {"limit": 100, "endId": ids[-1]}
    seen, out = set(), []
    for r in rows:
        key = r.get("blockTdId") or (r.get("groupId"), r.get("cTime"))
        if key in seen or not (start_ms <= int(r["cTime"]) <= end_ms):
            continue
        seen.add(key)
        out.append(r)
    return out, truncated


def contract_specs(base: str) -> Dict[str, Tuple[float, bool]]:
    """{family: (underlying units per contract = ctVal * ctMult, is_linear)} for one underlying, from
    public/instruments (``uly`` - ``instFamily`` is rejected for SOL / XAU)."""
    if base in _spec_cache:
        return _spec_cache[base]
    out: Dict[str, Tuple[float, bool]] = {}
    ok = False
    try:
        for x in _get("/public/instruments", {"instType": "OPTION", "uly": f"{base}-USD"}).get("data", []):
            out.setdefault(x["instFamily"], (float(x["ctVal"]) * float(x["ctMult"]), x.get("ctType") == "linear"))
        ok = True
    except Exception:
        pass
    for fam, spec in _SPEC_FALLBACK.items():
        if fam.startswith(f"{base}-") and fam not in out:
            out[fam] = spec
    if ok:
        _spec_cache[base] = out
    return out


def _lookup_instrument(inst_id: str) -> List[Dict]:
    try:
        return _get("/public/block-trades", {"instId": inst_id}).get("data", [])
    except RuntimeError:
        return []


def fetch_index_candles(base: str, start_ms: int, end_ms: int, bar: str = "5m") -> pd.DataFrame:
    """Index OHLC -> DataFrame[ts_ms (bar open), open, close], oldest first."""
    rows: List[List[str]] = []
    after = int(end_ms) + 1            # `after` = records older than this ts
    for _ in range(60):
        try:
            data = _get("/market/history-index-candles",
                        {"instId": f"{base}-USD", "bar": bar, "limit": 100, "after": after}).get("data", [])
        except RuntimeError:
            break
        if not data:
            break
        rows.extend(data)
        oldest = int(data[-1][0])
        if oldest <= start_ms or oldest >= after:
            break
        after = oldest
    if not rows:
        return pd.DataFrame(columns=["ts_ms", "open", "close"])
    d = pd.DataFrame([r[:5] for r in rows], columns=["ts_ms", "open", "high", "low", "close"])
    d["ts_ms"] = d["ts_ms"].astype("int64")
    for c in ("open", "close"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return (d[["ts_ms", "open", "close"]].drop_duplicates("ts_ms")
            .sort_values("ts_ms").reset_index(drop=True))


def _normalize(tape: List[Dict], asset: str, specs: Dict[str, Tuple[float, bool]], lookup: Dict[str, Dict],
               candles: pd.DataFrame, spot: float) -> pd.DataFrame:
    recs = []
    for r in tape:
        pkg = r.get("blockTdId") or r.get("groupId") or None
        for i, leg in enumerate(r.get("legs", [])):
            base, exp, strike, cp = parse_option(leg.get("instId"))
            fam = family(leg.get("instId"))
            if base != asset or cp is None or pd.isna(exp) or fam not in specs:
                continue
            ct, linear = specs[fam]
            tid = leg.get("tradeId") or f"{pkg}-{i}"
            ex = lookup.get(leg.get("tradeId") or "", {})
            recs.append((tid, int(r["cTime"]), leg["instId"], strike, exp, cp,
                         str(leg.get("side", "")).lower(), _f(leg.get("sz")) * ct, _f(leg.get("px")),
                         _f(ex.get("idxPx")), _f(ex.get("markPx")), _f(ex.get("fillVol")), pkg, linear))
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs, columns=["trade_id", "ts_ms", "instrument_name", "strike", "expiry", "option_type",
                                     "direction", "abs_amount", "px_coin", "index_price", "mark_coin", "iv_dec",
                                     "block_id", "linear"])
    df = df[df["direction"].isin(["buy", "sell"])].sort_values("ts_ms").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()
    # Index fallback for legs the per-instrument join missed: candle open at/before the trade, else live spot.
    miss = df["index_price"].isna()
    if miss.any() and not candles.empty:
        k = candles.rename(columns={"ts_ms": "k_ms", "open": "k_px"})[["k_ms", "k_px"]]
        m = pd.merge_asof(df.loc[miss, ["ts_ms"]].sort_values("ts_ms"), k, left_on="ts_ms", right_on="k_ms",
                          direction="backward", tolerance=6 * 3_600_000)
        df.loc[miss, "index_price"] = m["k_px"].values   # df is already sorted by ts_ms
    if spot and spot > 0:
        df["index_price"] = df["index_price"].fillna(spot)
    mult = np.where(df["linear"], 1.0, df["index_price"])      # inverse: coin -> USD per 1 underlying; linear: already USD
    df["price"] = df["px_coin"] * mult
    df["mark_price"] = df["mark_coin"] * mult
    df["iv"] = df["iv_dec"] * 100.0
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(SGT)
    df["amount"] = np.where(df["direction"] == "buy", df["abs_amount"], -df["abs_amount"])
    df["is_block"] = True
    df["wallet"] = None
    df = df.dropna(subset=["price"])
    return df.drop(columns=["ts_ms", "px_coin", "mark_coin", "iv_dec", "linear"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public fetchers
# ---------------------------------------------------------------------------
def listed_bases() -> List[str]:
    """Underlyings with options (public/underlying -> ['SOL-USD', 'BTC-USD', ...])."""
    try:
        d = _get("/public/underlying", {"instType": "OPTION"}).get("data", [])
        return sorted({u.split("-")[0] for u in (d[0] if d else [])})
    except Exception:
        return []


def fetch_all(start_ms: int, end_ms: int, assets: List[str]) -> Dict:
    """-> {'frames': {base: df}, 'meta': {...}}.  Frames hold block option legs only (the tape is blocks
    only).  Every underlying that printed a block is normalised; only ``assets`` get their own tab."""
    tape, truncated = fetch_block_tape(start_ms, end_ms)
    n_opt = n_other = 0
    present: Dict[str, set] = {}
    for r in tape:
        for leg in r.get("legs", []):
            base, _, _, cp = parse_option(leg.get("instId"))
            if cp:
                n_opt += 1
                present.setdefault(base, set()).add(leg["instId"])
            else:
                n_other += 1
    listed = sorted(set(listed_bases()) | set(assets) | set(present))
    frames: Dict[str, pd.DataFrame] = {a: pd.DataFrame() for a in listed}

    # Mark / IV / index at trade time: join public/block-trades?instId= on tradeId.
    inst_ids = sorted({i for s in present.values() for i in s})
    skipped = max(0, len(inst_ids) - _ENRICH_CAP)
    lookup: Dict[str, Dict] = {}
    if inst_ids:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            for rows in ex.map(_lookup_instrument, inst_ids[:_ENRICH_CAP]):
                for x in rows:
                    lookup[x["tradeId"]] = x
    span_h = (end_ms - start_ms) / 3.6e6
    bar, k_start = ("5m", start_ms - 3_600_000) if span_h <= 48 else ("1H", start_ms - 6 * 3_600_000)
    n_leg = n_hit = 0
    for a in present:
        legs = [leg for r in tape for leg in r.get("legs", []) if parse_option(leg.get("instId"))[0] == a]
        need = [l for l in legs if (l.get("tradeId") or "") not in lookup]
        candles = (fetch_index_candles(a, k_start, end_ms, bar) if need
                   else pd.DataFrame(columns=["ts_ms", "open", "close"]))
        spot = fetch_spot(a) if need else 0.0
        frames[a] = _normalize(tape, a, contract_specs(a), lookup, candles, spot)
        n_leg += len(legs)
        n_hit += len(legs) - len(need)
    return {"frames": frames,
            "meta": {"tape_rows": len(tape), "option_legs": n_opt, "non_option_legs": n_other,
                     "listed": listed, "other_bases": sorted(b for b in present if b not in assets),
                     "truncated": truncated, "joined_legs": n_hit, "total_legs": n_leg,
                     "enrich_skipped": skipped}}


def fetch_spot(asset: str) -> float:
    try:
        d = _get("/market/index-tickers", {"instId": f"{asset}-USD"}).get("data", [])
        return float(d[0]["idxPx"]) if d else 0.0
    except Exception:
        return 0.0


def fetch_hist_spot(asset: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    span_h = (end_ms - start_ms) / 3.6e6
    k = fetch_index_candles(asset, start_ms, end_ms, "5m" if span_h <= 48 else "1H")
    if k.empty:
        return pd.DataFrame()
    ts = pd.to_datetime(k["ts_ms"], unit="ms", utc=True).dt.tz_convert(SGT)
    return pd.DataFrame({"timestamp": ts, "close": k["close"].values})


def feed_status(meta: Dict, now_ms: Optional[int] = None) -> Optional[str]:
    if meta.get("tape_rows", 0) == 0:
        return "OKX returned no block trades in this window (the public tape lags execution by ~15 minutes)."
    parts = [f"{meta['option_legs']} option legs across {meta['tape_rows']} block packages "
             f"(OKX publishes blocks ~15 min after execution)"]
    if meta.get("non_option_legs"):
        parts.append(f"{meta['non_option_legs']} futures/swap/spot hedge legs not charted")
    if meta.get("total_legs") and meta["joined_legs"] < meta["total_legs"]:
        parts.append(f"mark/IV joined for {meta['joined_legs']}/{meta['total_legs']} legs "
                     f"(the rest use IV backed out of price)")
    if meta.get("truncated"):
        parts.append("window exceeded the page-fetch cap, so the oldest blocks are missing")
    return "Window contains " + "; ".join(parts) + "."


class _Venue:
    key = "okx"
    title = "OKX"
    icon = "⚫"
    assets = ASSETS
    default_min_sizes = DEFAULT_MIN_SIZES
    has_mark = True
    has_iv = True
    perp_label = "Index"
    definition = ("**Block definition:** every print on OKX's public block-trade tape "
                  "(`GET /api/v5/rfq/public-trades`, RFQ blocks, published ~15 min after execution). "
                  "`side` is the taker's side on each leg; legs of one package share a `blockTdId`. "
                  "OKX's standard options are coin-margined, so prices are converted to USD per underlying with the index "
                  "at execution (the `_UM` USD-settled family is already USD); sizes go from contracts to underlying "
                  "units (ctVal x ctMult, per family). Default minimum size = 1 contract. Mark and IV come "
                  "from `GET /api/v5/public/block-trades` joined on trade id; legs it misses fall back to IV "
                  "backed out of price (Black-Scholes, r=0). DVOL overlay is Deribit's.")
    listed = staticmethod(listed_bases)
    fetch_all = staticmethod(fetch_all)
    fetch_spot = staticmethod(fetch_spot)
    fetch_hist_spot = staticmethod(fetch_hist_spot)
    feed_status = staticmethod(feed_status)


VENUE = _Venue()
