"""
Probe how much option-IV history Deribit's public API will actually give us.

Run this from a machine with internet access:

    python tools_probe_history.py BTC

It answers the two questions that decide whether the trade-tape backfill
(option 1 in HISTORICAL_DATA.md) is viable:

  1. How far back does public/get_last_trades_by_currency_and_time serve?
  2. How dense is the tape — can we get an hourly ATM / 25-delta series from it?

Nothing is written; this only reads and prints.
"""

import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests

BASE = "https://www.deribit.com/api/v2/public"
ASSET = (sys.argv[1] if len(sys.argv) > 1 else "BTC").upper()
CCY = {"BTC": "BTC", "ETH": "ETH", "SOL": "USDC", "HYPE": "USDC"}.get(ASSET, "BTC")


def get(endpoint, **params):
    r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("result")


def probe_depth():
    """Walk backwards in 1-day windows until the tape stops returning trades."""
    print(f"\n=== 1. How far back does the {ASSET} option trade tape go? ===")
    now = datetime.now(timezone.utc)
    for days_back in (1, 7, 30, 90, 180, 365, 730):
        start = now - timedelta(days=days_back)
        end = start + timedelta(hours=6)
        try:
            res = get("get_last_trades_by_currency_and_time",
                      currency=CCY, kind="option",
                      start_timestamp=int(start.timestamp() * 1000),
                      end_timestamp=int(end.timestamp() * 1000),
                      count=100, sorting="asc")
            trades = res.get("trades", []) if isinstance(res, dict) else (res or [])
            has_iv = sum(1 for t in trades if t.get("iv") or t.get("mark_iv"))
            flag = "OK " if trades else "-- "
            print(f"  {flag}{days_back:4d}d ago (6h window): {len(trades):4d} trades, "
                  f"{has_iv} with IV")
        except Exception as exc:
            print(f"  ERR {days_back:4d}d ago: {exc}")
        time.sleep(0.2)


def probe_density(days_back=7):
    """Count distinct hours covered, and trades per expiry, over a 24h window."""
    print(f"\n=== 2. Tape density {days_back}d ago (24h window) ===")
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days_back)
    end = start + timedelta(days=1)
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    trades = []
    for _ in range(30):
        res = get("get_last_trades_by_currency_and_time",
                  currency=CCY, kind="option",
                  start_timestamp=cursor, end_timestamp=end_ms,
                  count=1000, sorting="asc")
        page = res.get("trades", []) if isinstance(res, dict) else (res or [])
        if not page:
            break
        trades.extend(page)
        if isinstance(res, dict) and not res.get("has_more"):
            break
        cursor = int(page[-1]["timestamp"]) + 1
        if cursor >= end_ms:
            break
        time.sleep(0.15)

    if not trades:
        print("  No trades returned — the trade-tape backfill is not viable here.")
        return

    hours = set()
    by_expiry = defaultdict(int)
    with_iv = 0
    for t in trades:
        ts = datetime.fromtimestamp(int(t["timestamp"]) / 1000, tz=timezone.utc)
        hours.add(ts.replace(minute=0, second=0, microsecond=0))
        parts = (t.get("instrument_name") or "").split("-")
        if len(parts) >= 2:
            by_expiry[parts[1]] += 1
        if t.get("iv") or t.get("mark_iv"):
            with_iv += 1

    print(f"  total trades      : {len(trades):,}")
    print(f"  trades carrying IV: {with_iv:,} ({100*with_iv/len(trades):.1f}%)")
    print(f"  distinct hours    : {len(hours)}/24 covered")
    print(f"  distinct expiries : {len(by_expiry)}")
    print("  busiest expiries  : " + ", ".join(
        f"{k}({v})" for k, v in sorted(by_expiry.items(),
                                       key=lambda kv: -kv[1])[:6]))
    verdict = ("VIABLE — enough coverage to rebuild an hourly ATM/25d series."
               if len(hours) >= 18 and with_iv > 0.5 * len(trades)
               else "PATCHY — usable for liquid tenors only; expect gaps.")
    print(f"  verdict           : {verdict}")


def probe_dvol():
    print("\n=== 3. DVOL history depth (the current fallback) ===")
    if CCY not in ("BTC", "ETH"):
        print(f"  {ASSET} has no DVOL index (USDC-linear).")
        return
    now = datetime.now(timezone.utc)
    for days_back in (30, 90, 365, 1095):
        start = now - timedelta(days=days_back)
        try:
            res = get("get_volatility_index_data", currency=CCY,
                      start_timestamp=int(start.timestamp() * 1000),
                      end_timestamp=int((start + timedelta(days=2)).timestamp() * 1000),
                      resolution="60")
            rows = (res or {}).get("data", [])
            print(f"  {'OK ' if rows else '-- '}{days_back:5d}d ago: {len(rows)} hourly points")
        except Exception as exc:
            print(f"  ERR {days_back:5d}d ago: {exc}")
        time.sleep(0.2)


if __name__ == "__main__":
    print(f"Probing Deribit public history for {ASSET} (currency={CCY})")
    probe_depth()
    probe_density(7)
    probe_dvol()
    print("\nDone. Paste this output back to Claude to pick the backfill strategy.")
