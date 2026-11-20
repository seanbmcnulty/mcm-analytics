"""
Our own block trades on OKX and Paradex ("won vs market"), from private, read-only API calls.

Used by ``lib.block_flow.render_page`` (via each venue's ``fetch_own``) to mark which public block
prints were ours, and to show the won-vs-market tab.  Everything here is read-only: no order,
quote, cancel or transfer endpoint is ever called.

Credentials (Streamlit secrets first, then environment variables).  When missing, the pages fall
back to the public-only view:

    [okx]                          env OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE
    api_key = "..."
    api_secret = "..."
    passphrase = "..."

    [paradex]                      env PARADEX_API_KEY
    api_key = "..."                (a read-only API token, i.e. a JWT; sent as a Bearer token)

OKX (signed REST, ``OK-ACCESS-*`` headers):
  rfq/trades   block trades we were a counterparty to, with ``blockTdId`` (the same id as on the
               public ``rfq/public-trades`` tape), ``mTraderCode`` (maker = quoter) and
               ``tTraderCode`` (taker = RFQ sender).  We *won* a ticket when we were the maker.
  rfq/rfqs     RFQs sent to us (and by us) with their final ``state``: ``filled`` (a quote of ours
               traded = won) or ``traded_away`` (the RFQ traded with another maker = lost to the
               market).  This gives the win rate.  It cannot tell whether we actually quoted an RFQ
               that was traded away.
Paradex (``Authorization: Bearer <token>``):
  /v1/account        our account address.
  /v1/block-trades   our block trades (newest first) with ``block_id`` (the same id as on the public
                     ``/v1/trades?trade_type=BLOCK_TRADE`` tape) and per-leg ``maker_account`` /
                     ``taker_account``.  We won it when we were the maker.  Paradex publishes no
                     lost-RFQ information, so there is no win rate; market share only.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

OKX_BASE = "https://www.okx.com"
PARADEX_BASE = "https://api.prod.paradex.trade/v1"

_session = requests.Session()
_okx_lock = threading.Lock()
_okx_last: Dict[str, float] = {}
_OKX_GAP = {"/api/v5/rfq/rfqs": 1.05, "/api/v5/rfq/trades": 0.25}   # private RFQ endpoints: 2-5 req / 2 s


def _secret_section(name: str) -> dict:
    try:
        import streamlit as st
        sec = st.secrets.get(name, {})
        return dict(sec) if sec else {}
    except Exception:
        return {}


def _usable(v) -> str:
    v = str(v or "").strip()
    return "" if (not v or "YOUR_" in v.upper() or v.lower().startswith("paste")) else v


def okx_credentials() -> Optional[tuple]:
    s = _secret_section("okx")
    k = _usable(s.get("api_key")) or _usable(os.environ.get("OKX_API_KEY"))
    sec = _usable(s.get("api_secret")) or _usable(os.environ.get("OKX_API_SECRET"))
    p = _usable(s.get("passphrase")) or _usable(os.environ.get("OKX_API_PASSPHRASE"))
    return (k, sec, p) if (k and sec and p) else None


def paradex_credentials() -> Optional[str]:
    s = _secret_section("paradex")
    return _usable(s.get("api_key")) or _usable(os.environ.get("PARADEX_API_KEY")) or None


def _empty(configured: bool, error: Optional[str] = None) -> Dict:
    return {"configured": configured, "error": error, "own": {}, "n_blocks": 0, "n_won": 0,
            "first_ms": None, "last_ms": None, "rfq": None}


def _summarise(out: Dict) -> Dict:
    out["n_blocks"] = len(out["own"])
    out["n_won"] = sum(1 for r in out["own"].values() if r == "maker")
    return out


# ---------------------------------------------------------------------------
# OKX
# ---------------------------------------------------------------------------
def _okx_get(path: str, params: Dict, creds: tuple, retries: int = 4) -> List[Dict]:
    key, secret, passphrase = creds
    query = "&".join(f"{k}={v}" for k, v in params.items() if v not in (None, ""))
    full = path + (f"?{query}" if query else "")
    last = None
    for attempt in range(retries):
        gap = _OKX_GAP.get(path, 0.0)
        with _okx_lock:
            wait = _okx_last.get(path, 0.0) + gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _okx_last[path] = time.monotonic()
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        sig = base64.b64encode(hmac.new(secret.encode(), (ts + "GET" + full).encode(),
                                        hashlib.sha256).digest()).decode()
        try:
            r = _session.get(OKX_BASE + full, timeout=20, headers={
                "OK-ACCESS-KEY": key, "OK-ACCESS-SIGN": sig, "OK-ACCESS-TIMESTAMP": ts,
                "OK-ACCESS-PASSPHRASE": passphrase})
            j = r.json()
        except (requests.RequestException, ValueError) as e:
            last = type(e).__name__
            time.sleep(0.8 * (attempt + 1))
            continue
        code = str(j.get("code", ""))
        if code in ("50011", "50061") or r.status_code == 429:
            last = "rate limited"
            time.sleep(1.2 * (attempt + 1))
            continue
        if code != "0":
            raise RuntimeError(f"OKX {path}: {code} {str(j.get('msg', ''))[:120]}")
        return j.get("data") or []
    raise RuntimeError(f"OKX {path}: {last}")


def _okx_own_code(trades: List[Dict]) -> Optional[str]:
    """Our trader code = the one present on every trade we were part of (maker or taker)."""
    if not trades:
        return None
    common = None
    for t in trades:
        s = {t.get("mTraderCode"), t.get("tTraderCode")} - {None, ""}
        common = s if common is None else common & s
    if not common:
        return None
    if len(common) == 1:
        return next(iter(common))
    return max(common, key=lambda c: sum(t.get("mTraderCode") == c for t in trades))


def okx_fetch_own(start_ms: int, end_ms: int, creds: Optional[tuple] = None, max_pages: int = 40) -> Dict:
    creds = creds or okx_credentials()
    if not creds:
        return _empty(False)
    out = _empty(True)
    try:
        trades: List[Dict] = []
        params: Dict = {"limit": 100, "beginTs": int(start_ms), "endTs": int(end_ms)}
        for _ in range(max_pages):
            data = _okx_get("/api/v5/rfq/trades", params, creds)
            trades += data
            if len(data) < 100:
                break
            params = {**params, "endId": data[-1]["blockTdId"]}
        trades = [t for t in trades if str(t.get("isSuccessful", "true")).lower() != "false"]
        code = _okx_own_code(trades) or _okx_own_code(_okx_get("/api/v5/rfq/trades", {"limit": 100}, creds))
        for t in trades:
            bid = t.get("blockTdId")
            if not bid:
                continue
            role = "maker" if t.get("mTraderCode") == code else ("taker" if t.get("tTraderCode") == code else "maker")
            out["own"][bid] = role
        if trades:
            ts = [int(t["cTime"]) for t in trades]
            out["first_ms"], out["last_ms"] = min(ts), max(ts)

        # Win rate from the RFQs we were sent (newest first, paged with endId until older than start).
        rfqs: List[Dict] = []
        params = {"limit": 100}
        for _ in range(max_pages):
            data = _okx_get("/api/v5/rfq/rfqs", params, creds)
            rfqs += data
            if len(data) < 100 or min(int(x["cTime"]) for x in data) < start_ms:
                break
            params = {"limit": 100, "endId": data[-1]["rfqId"]}
        tally = {"won": 0, "lost": 0, "no_trade": 0, "own_rfqs": 0, "total": 0}
        for x in rfqs:
            if not (start_ms <= int(x["cTime"]) <= end_ms):
                continue
            if code and x.get("traderCode") == code:
                tally["own_rfqs"] += 1
                continue
            tally["total"] += 1
            st_ = str(x.get("state", "")).lower()
            tally["won" if st_ == "filled" else "lost" if st_ == "traded_away" else "no_trade"] += 1
        out["rfq"] = tally
    except Exception as e:  # never surface credentials: messages above carry only path + code
        out["error"] = str(e)[:200]
    return _summarise(out)


# ---------------------------------------------------------------------------
# Paradex
# ---------------------------------------------------------------------------
def _pdx_get(path: str, params: Dict, token: str, retries: int = 3) -> Dict:
    last = None
    for attempt in range(retries):
        try:
            r = _session.get(PARADEX_BASE + path, params=params, timeout=30,
                             headers={"Authorization": f"Bearer {token}"})
        except requests.RequestException as e:
            last = type(e).__name__
            time.sleep(0.6 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            last = f"HTTP {r.status_code}"
            time.sleep(0.8 * (attempt + 1))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"Paradex {path}: HTTP {r.status_code}")
        return r.json()
    raise RuntimeError(f"Paradex {path}: {last}")


def paradex_fetch_own(start_ms: int, end_ms: int, token: Optional[str] = None, max_pages: int = 60) -> Dict:
    token = token or paradex_credentials()
    if not token:
        return _empty(False)
    out = _empty(True)
    try:
        acct = _pdx_get("/account", {}, token).get("account")
        params: Dict = {"page_size": 100, "status": "COMPLETED"}
        blocks: List[Dict] = []
        for _ in range(max_pages):
            j = _pdx_get("/block-trades", params, token)
            res = j.get("results") or []
            blocks += res
            cur = j.get("next")
            if not cur or not res or min(int(b["created_at"]) for b in res) < start_ms:
                break
            params = {"page_size": 100, "status": "COMPLETED", "cursor": cur}
        ts = []
        for b in blocks:
            t = int(b.get("created_at") or 0)
            if str(b.get("status", "")).upper() != "COMPLETED" or not (start_ms <= t <= end_ms):
                continue
            legs = list((b.get("trades") or {}).values())
            roles = {"maker" if l.get("maker_account") == acct else "taker" if l.get("taker_account") == acct
                     else None for l in legs if isinstance(l, dict)} - {None}
            out["own"][b["block_id"]] = "maker" if "maker" in roles or not roles else "taker"
            ts.append(t)
        if ts:
            out["first_ms"], out["last_ms"] = min(ts), max(ts)
    except Exception as e:
        out["error"] = str(e)[:200]
    return _summarise(out)
