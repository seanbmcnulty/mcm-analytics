"""
Cross-exchange option-block-flow summary (Deribit, Derive, Paradex, OKX, Bybit, Bullish, Binance) for the Home page.

* Derive / Paradex / OKX / Bybit / Binance frames come straight from ``lib.derive`` / ``lib.paradex`` /
  ``lib.okx`` / ``lib.bybit`` / ``lib.binance`` (``fetch_all``), which already return block trades in the shared schema.
* Bullish has no block flag either, so its frames hold every option trade and ``collect`` keeps only those at/above
  ``lib.bullish.DEFAULT_MIN_SIZES`` (the exchange minimum order, i.e. all prints) (every venue's ``default_min_sizes`` is applied the same way; it is empty,
  i.e. a no-op, for the others).
* Deribit has no block flag on its public trades feed, so — exactly like
  ``pages/02_Block_Trades_-_Deribit.py`` — a "block" is a trade at or above a per-asset
  minimum size (``DERIBIT_MIN_SIZES``; keep in sync with that page's
  ``DEFAULT_MIN_SIZES``).  Coin-margined BTC/ETH prices are converted to USD with each
  trade's index price so every venue is on the same USD footing.

``collect()`` gathers all seven venues in parallel; ``summarize()`` turns the result
into a per-venue table and a per-underlying table.
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lib import binance, block_flow, bullish, bybit, deribit, derive, okx, paradex
from lib.constants import TTL_MEDIUM

SGT = timezone(timedelta(hours=8))

# Keep the first six in sync with pages/02_Block_Trades_-_Deribit.py DEFAULT_MIN_SIZES.
DERIBIT_MIN_SIZES: Dict[str, float] = {
    "BTC": 12.5, "ETH": 125.0, "SOL_USDC": 125.0, "XRP_USDC": 1000.0,
    "AVAX_USDC": 10.0, "HYPE_USDC": 10.0,
    # Listed on Deribit but not tabbed on the Deribit page (headline-only there):
    "BTC_USDC": 12.5, "ETH_USDC": 125.0, "TRX_USDC": 10000.0,
}
DERIBIT_EXTRA_MIN_SIZES: Dict[str, float] = {
    k: v for k, v in DERIBIT_MIN_SIZES.items() if k in ("BTC_USDC", "ETH_USDC", "TRX_USDC")}
_CURRENCY = {a: ("USDC" if a.endswith("_USDC") else a) for a in DERIBIT_MIN_SIZES}
_COIN_MARGINED = {"BTC", "ETH"}
VENUE_ORDER = ["Deribit", "Derive", "Paradex", "OKX", "Bybit", "Bullish", "Binance"]
_NO_MARK = {"Paradex", "Bullish", "Binance"}   # tapes with no mark price -> no Edge / Aggression


def _deribit_currency_trades(currency: str, start_ms: int, end_ms: int, max_pages: int = 20):
    """Page ``get_last_trades_by_currency_and_time`` forward; -> (rows, truncated)."""
    rows: List[dict] = []
    cursor = start_ms
    for _ in range(max_pages):
        res = deribit.get_last_trades_by_currency_and_time(
            currency, kind="option", start_ms=cursor, end_ms=end_ms,
            count=1000, sorting="asc", ttl=TTL_MEDIUM)
        trades = (res or {}).get("trades", [])
        if not trades:
            return rows, False
        rows.extend(trades)
        if not res.get("has_more"):
            return rows, False
        nxt = int(trades[-1]["timestamp"]) + 1
        if nxt <= cursor:
            return rows, False
        cursor = nxt
    return rows, True


def _normalize_deribit(rows: List[dict], asset: str, min_size: float) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("trade_id")
    df = df[df["instrument_name"].str.startswith(f"{asset}-")]
    if df.empty:
        return pd.DataFrame()
    df = df[pd.to_numeric(df["amount"], errors="coerce").abs() >= min_size].copy()
    if df.empty:
        return pd.DataFrame()
    px = pd.to_numeric(df["index_price"], errors="coerce")
    mult = px if asset in _COIN_MARGINED else 1.0   # coin-margined: price quoted in underlying
    parts = df["instrument_name"].str.extract(
        r"^[^-]*-(?P<exp>[0-9]{1,2}[A-Z]{3}[0-9]{2})-(?P<strike>[0-9.]+)-(?P<cp>[CP])$")
    out = pd.DataFrame({
        "trade_id": df["trade_id"].values,
        "timestamp": pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(SGT).values,
        "instrument_name": df["instrument_name"].values,
        "strike": pd.to_numeric(parts["strike"], errors="coerce").values,
        "expiry": pd.to_datetime(parts["exp"], format="%d%b%y", errors="coerce").values,
        "option_type": parts["cp"].values,
        "direction": df["direction"].astype(str).str.lower().values,
        "abs_amount": pd.to_numeric(df["amount"], errors="coerce").abs().values,
        "price": (pd.to_numeric(df["price"], errors="coerce") * mult).values,
        "mark_price": (pd.to_numeric(df["mark_price"], errors="coerce") * mult).values,
        "index_price": px.values,
        "iv": pd.to_numeric(df.get("iv"), errors="coerce").values,
        "is_block": True,
        "block_id": None,
    })
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(SGT)
    out["amount"] = np.where(out["direction"] == "buy", out["abs_amount"], -out["abs_amount"])
    out = out.dropna(subset=["strike", "expiry", "option_type"])
    return out.sort_values("timestamp").reset_index(drop=True)


def deribit_blocks(start_ms: int, end_ms: int,
                   min_sizes: Optional[Dict[str, float]] = None) -> Dict:
    """-> {'frames': {asset: df}, 'meta': {...}} in the shared schema (size-threshold blocks)."""
    min_sizes = min_sizes or DERIBIT_MIN_SIZES
    currencies = sorted(set(_CURRENCY[a] for a in min_sizes))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(currencies)) as ex:
        futs = {c: ex.submit(_deribit_currency_trades, c, start_ms, end_ms) for c in currencies}
        raw = {c: f.result() for c, f in futs.items()}
    frames = {a: _normalize_deribit(raw[_CURRENCY[a]][0], a, ms) for a, ms in min_sizes.items()}
    truncated = [c for c, (_, t) in raw.items() if t]
    return {"frames": frames, "meta": {"listed": list(min_sizes), "truncated": truncated}}


def _venue_blocks(venue, start_ms: int, end_ms: int) -> Dict:
    """``venue.fetch_all`` with the venue's default minimum block sizes applied (a no-op where empty)."""
    p = venue.fetch_all(start_ms, end_ms, list(venue.assets))
    mins = venue.default_min_sizes
    if mins:
        for a, df in p["frames"].items():
            if df is not None and not df.empty and mins.get(a):
                p["frames"][a] = df[df["abs_amount"] >= mins[a]].reset_index(drop=True)
    meta = p["meta"]
    if venue is bybit.VENUE:
        meta["home_note"] = bybit.feed_status(meta)
    elif meta.get("truncated"):
        meta["home_note"] = "window exceeded the page-fetch cap — figures are a lower bound"
    return p


def collect(start_ms: int) -> Dict[str, Dict]:
    """Fetch all seven venues in parallel -> {venue_title: {'frames', 'meta'}}.
    A venue that errors comes back as {'error': str}."""
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    jobs = {
        "Deribit": lambda: deribit_blocks(start_ms, end_ms),
        "Derive": lambda: _venue_blocks(derive.VENUE, start_ms, end_ms),
        "Paradex": lambda: _venue_blocks(paradex.VENUE, start_ms, end_ms),
        "OKX": lambda: _venue_blocks(okx.VENUE, start_ms, end_ms),
        "Bybit": lambda: _venue_blocks(bybit.VENUE, start_ms, end_ms),
        "Bullish": lambda: _venue_blocks(bullish.VENUE, start_ms, end_ms),
        "Binance": lambda: _venue_blocks(binance.VENUE, start_ms, end_ms),
    }
    out: Dict[str, Dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = {k: ex.submit(fn) for k, fn in jobs.items()}
        for k, f in futs.items():
            try:
                out[k] = f.result()
            except Exception as e:  # noqa: BLE001 - one venue failing must not blank the summary
                out[k] = {"error": str(e)}
    return out


def _clean(asset: str) -> str:
    """Display name: Deribit's USDC-settled BTC/ETH/etc. keep a suffix to stay distinct."""
    if asset in ("BTC_USDC", "ETH_USDC"):
        return asset.replace("_USDC", " (USDC)")
    return asset.replace("_USDC", "")


def _asset_row(venue: str, asset: str, df: pd.DataFrame, has_mark: bool) -> Optional[Dict]:
    if df is None or df.empty:
        return None
    gross = float((df["abs_amount"] * df["price"]).sum())
    call = float((df.loc[df["option_type"] == "C", "abs_amount"] * df.loc[df["option_type"] == "C", "price"]).sum())
    buy = float((df.loc[df["direction"] == "buy", "abs_amount"] * df.loc[df["direction"] == "buy", "price"]).sum())
    big = (df["abs_amount"] * df["price"]).idxmax()
    row = {
        "Exchange": venue, "Underlying": _clean(asset), "Blocks": len(df), "Gross premium $": gross,
        "Call %": call / gross if gross else np.nan, "Taker-buy %": buy / gross if gross else np.nan,
        "Net delta $": np.nan, "Net vega $": np.nan,
        "Largest $": float(df.loc[big, "abs_amount"] * df.loc[big, "price"]),
        "Largest print": f"{df.loc[big, 'instrument_name']} ({'+' if df.loc[big, 'direction'] == 'buy' else '-'}{df.loc[big, 'abs_amount']:g})",
    }
    idx = pd.to_numeric(df["index_price"], errors="coerce")
    if idx.notna().mean() >= 0.5:
        try:
            e = block_flow.enrich(df.copy(), float(idx.dropna().iloc[-1]), asset, has_mark)
            row["Net delta $"] = float(e["dollar_delta"].sum())
            row["Net vega $"] = float(e["dollar_vega"].sum())
        except Exception:
            pass
    return row


def summarize(payloads: Dict[str, Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """-> (per-venue table, per-underlying table, {venue: note})."""
    rows, notes = [], {}
    for venue, p in payloads.items():
        if "error" in p:
            notes[venue] = f"fetch failed: {p['error']}"
            continue
        has_mark = venue not in _NO_MARK
        for asset, df in p["frames"].items():
            r = _asset_row(venue, asset, df, has_mark)
            if r:
                rows.append(r)
        if p["meta"].get("home_note"):
            notes[venue] = p["meta"]["home_note"]
        elif p["meta"].get("truncated"):
            notes[venue] = ("window exceeded the 20k-trade fetch cap for " +
                            ", ".join(p["meta"]["truncated"]) + " — figures are a lower bound")
    by_asset = pd.DataFrame(rows)
    if by_asset.empty:
        return pd.DataFrame(), by_asset, notes

    def agg(g: pd.DataFrame) -> pd.Series:
        gp = g["Gross premium $"].sum()
        w = lambda c: float((g[c] * g["Gross premium $"]).sum() / gp) if gp else np.nan  # noqa: E731
        big = g.loc[g["Largest $"].idxmax()]
        covered = g.loc[g["Net vega $"].notna(), "Gross premium $"].sum()
        return pd.Series({
            "Blocks": int(g["Blocks"].sum()), "Underlyings": int(len(g)), "Gross premium $": gp,
            "Call %": w("Call %"), "Taker-buy %": w("Taker-buy %"),
            "Net delta $": g["Net delta $"].sum(min_count=1), "Net vega $": g["Net vega $"].sum(min_count=1),
            "Greeks cover": covered / gp if gp else np.nan,
            "Largest print": f"${big['Largest $']:,.0f} · {big['Underlying']} {big['Largest print']}",
        })

    by_venue = by_asset.groupby("Exchange").apply(agg, include_groups=False).reset_index()
    order = {v: i for i, v in enumerate(VENUE_ORDER)}
    by_venue = by_venue.sort_values("Exchange", key=lambda s: s.map(order)).reset_index(drop=True)
    by_asset = by_asset.sort_values(["Exchange", "Gross premium $"], ascending=[True, False]).reset_index(drop=True)
    return by_venue, by_asset, notes
