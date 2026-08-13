"""
Accumulator Syntax Builder — Build ACC/DCC syntax and JSON for sales flow.

Premium and strike/KO suggestions use a discrete European fixing heuristic
(KO checked only on fixing dates).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.deribit import get_index_price, get_option_chain
from lib.constants import ASSET_CONFIG, PLOTLY_LAYOUT

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Accumulator Syntax Builder",
    page_icon="🧮",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_TOKENS = ["BTC", "ETH", "SOL", "HYPE"]

DERIBIT_INDEX_MAP = {
    "BTC": "btc_usd",
    "ETH": "eth_usd",
    "SOL": "sol_usdc",
    "HYPE": "hype_usdc",
}

DERIBIT_CCY_MAP = {
    "BTC": "BTC",
    "ETH": "ETH",
    "SOL": "USDC",
    "HYPE": "USDC",
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _fmt_num(v: float, max_decimals: int = 6) -> str:
    if abs(v - round(v)) < 1e-12:
        return str(int(round(v)))
    return f"{v:.{max_decimals}f}".rstrip("0").rstrip(".")


def _fmt_readable(v: float, decimals: int = 2) -> str:
    return f"{v:,.{decimals}f}"


def _fmt_sheet_number(v: float, decimals: int = 2) -> str:
    if abs(v) >= 1000:
        return _fmt_readable(v, decimals)
    return f"{v:.{decimals}f}"


def _cell_to_float(v, default: float = 0.0) -> float:
    if v is None:
        return default
    s = str(v).strip().replace(",", "")
    if s == "":
        return default
    return float(s)


def _extract_charge_bps_from_va(token: str) -> float:
    m = re.match(r"^va([^/]+)/([^/]+)/([^/]+)$", token.strip(), re.IGNORECASE)
    if not m:
        return 0.0
    try:
        return float(m.group(3))
    except Exception:
        return 0.0


def estimate_charge_usd(total_usd_notional: float, charge_bps: float) -> float:
    return total_usd_notional * (charge_bps / 10000.0)


def resolve_total_size_units(
    size_value: float,
    size_mode: str,
    weekly_usd_notional: float,
    spot: float,
    fixing_count: int,
) -> float:
    if size_value > 0:
        if size_mode == "USD":
            return (size_value / spot) if spot > 0 else 0.0
        return size_value
    if spot > 0 and weekly_usd_notional > 0 and fixing_count > 0:
        return (weekly_usd_notional / spot) * fixing_count
    return 0.0


# ---------------------------------------------------------------------------
# Date / token helpers
# ---------------------------------------------------------------------------

def _token_date(d: date) -> str:
    day = str(int(d.strftime("%d")))
    mon = d.strftime("%b")
    yy = d.strftime("%y")
    return f"{day}{mon}{yy}"


def _add_month(d: date) -> date:
    year = d.year + (1 if d.month == 12 else 0)
    month = 1 if d.month == 12 else d.month + 1
    last_day = pd.Timestamp(year=year, month=month, day=1).days_in_month
    day = min(d.day, int(last_day))
    return date(year, month, day)


def build_fixing_schedule(start_d: date, end_d: date, freq: str) -> list[date]:
    if end_d < start_d:
        return []
    out: list[date] = []
    cur = start_d
    while cur <= end_d:
        out.append(cur)
        cur = cur + timedelta(days=1) if freq == "D" else cur + timedelta(days=7)
    if not out or out[-1] != end_d:
        out.append(end_d)
    return out


# ---------------------------------------------------------------------------
# Level helpers
# ---------------------------------------------------------------------------

def _parse_level_token(tok: str) -> tuple[str, float]:
    tok = tok.strip()
    if tok.endswith("%"):
        return ("pct", float(tok[:-1]))
    return ("abs", float(tok))


def level_to_abs(level_mode: str, level_val: float, spot: float) -> float:
    if level_mode == "pct":
        return spot * (1.0 + level_val / 100.0)
    return level_val


def level_to_token(level_mode: str, level_val: float) -> str:
    if level_mode == "pct":
        return f"{_fmt_num(level_val, 4)}%"
    return _fmt_num(level_val, 4)


# ---------------------------------------------------------------------------
# Syntax parsing and building
# ---------------------------------------------------------------------------

def build_current_syntax(mods: list[str]) -> str:
    size_for_syntax = resolve_total_size_units(
        st.session_state.acc_size,
        st.session_state.acc_size_mode,
        st.session_state.acc_weekly_usd_notional,
        st.session_state.acc_spot,
        len(st.session_state.acc_fixings),
    )
    core = (
        f"{_fmt_num(size_for_syntax, 6)}{st.session_state.acc_contract} {st.session_state.acc_token} "
        f"{st.session_state.acc_freq}-{_token_date(st.session_state.acc_start_date)}-{_token_date(st.session_state.acc_end_date)} "
        f"strike{level_to_token(st.session_state.acc_strike_mode, st.session_state.acc_strike_val)} "
        f"ko{level_to_token(st.session_state.acc_ko_mode, st.session_state.acc_ko_val)} "
        f"g{_fmt_num(st.session_state.acc_gearing, 4)}"
    )
    return core if not mods else f"{core} {' '.join(mods)}"


def parse_accumulator_syntax(text: str) -> Optional[dict]:
    patt = re.compile(
        r"^(?:(\d+(?:\.\d+)?)(ACC|DCC)|(ACC|DCC))\s+([A-Z0-9_]+)\s+([WD])-([^- ]+)-([^- ]+)\s+strike([^\s]+)\s+ko([^\s]+)\s+g([^\s]+)(?:\s+(.*))?$",
        re.IGNORECASE,
    )
    m = patt.match(text.strip())
    if not m:
        return None
    size_pref, contract_attached, contract_plain, token, freq, sdt, edt, strike_tok, ko_tok, g_tok, mod_rest = m.groups()
    contract = (contract_attached or contract_plain or "").upper()
    try:
        sd = datetime.strptime(sdt, "%d%b%y").date()
        ed = datetime.strptime(edt, "%d%b%y").date()
    except ValueError:
        return None
    smode, sval = _parse_level_token(strike_tok)
    kmode, kval = _parse_level_token(ko_tok)
    modifiers = [] if not mod_rest else [x for x in mod_rest.split(" ") if x]
    return {
        "contract": contract.upper(),
        "token": token.upper(),
        "freq": freq.upper(),
        "start_date": sd,
        "end_date": ed,
        "strike_mode": smode,
        "strike_val": sval,
        "ko_mode": kmode,
        "ko_val": kval,
        "gearing": float(g_tok),
        "size": float(size_pref) if size_pref is not None else None,
        "modifiers": modifiers,
    }


def serialize_modifiers() -> list[str]:
    mods: list[str] = []
    spot_kind = st.session_state.acc_spot_mod_kind
    spot_val = st.session_state.acc_spot_mod_value
    if spot_kind != "none" and spot_val > 0:
        mods.append(f"{spot_kind}{_fmt_num(spot_val, 2)}")
    if st.session_state.acc_vol_mod_raw.strip():
        mods.append(st.session_state.acc_vol_mod_raw.strip())
    if st.session_state.acc_bps_mod_raw.strip():
        mods.append(st.session_state.acc_bps_mod_raw.strip())
    if st.session_state.acc_notional_target > 0:
        mods.append(f"N{_fmt_num(st.session_state.acc_notional_target, 2)}")
    if st.session_state.acc_qty_mult > 0 and abs(st.session_state.acc_qty_mult - 1.0) > 1e-12:
        mods.append(f"x{_fmt_num(st.session_state.acc_qty_mult, 4)}")
    if st.session_state.acc_yield_target_enable:
        mods.append(f"y{_fmt_num(st.session_state.acc_yield_target, 4)}")
    pricer = st.session_state.acc_pricer_choice
    if pricer == "p_qty":
        mods.append(f"p{int(st.session_state.acc_pricer_qty)}")
    elif pricer in {"p", "w", "ww", "o"}:
        mods.append(pricer)
    if st.session_state.acc_corr_enable:
        mods.append(f"r{int(round(st.session_state.acc_corr_value * 100))}")
    if st.session_state.acc_use_vdflt:
        mods.append("vdflt")
    if st.session_state.acc_use_asof:
        mods.append("aof" + st.session_state.acc_asof_date.strftime("%y%m%d"))
    if st.session_state.acc_extra_modifiers.strip():
        extras = [x for x in st.session_state.acc_extra_modifiers.split(" ") if x]
        mods.extend(extras)
    return mods


def modifiers_validation(mods: list[str]) -> list[str]:
    errs: list[str] = []
    token_ok = re.compile(r"^[A-Za-z][A-Za-z0-9_/\.\-]*$")
    for m in mods:
        if not token_ok.match(m):
            errs.append(f"Invalid modifier token: {m}")
        if "_" in m and "/" not in m:
            errs.append(f"Modifier has '_' but missing per-leg '/' groups: {m}")
    return errs


# ---------------------------------------------------------------------------
# Market data (Deribit public API)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=20, show_spinner=False)
def fetch_deribit_spot(token: str) -> Optional[float]:
    idx = DERIBIT_INDEX_MAP.get(token.upper())
    if not idx:
        return None
    return get_index_price(idx)


@st.cache_data(ttl=45, show_spinner=False)
def fetch_deribit_iv_points(token: str) -> list[tuple[int, float]]:
    ccy = DERIBIT_CCY_MAP.get(token.upper())
    if not ccy:
        return []
    chain = get_option_chain(ccy, "option")
    if not chain:
        return []

    today = datetime.now(timezone.utc).date()
    bucket: dict[int, list[tuple[float, float]]] = {}
    patt = re.compile(r"^[A-Z0-9_]+-(\d{1,2}[A-Za-z]{3}\d{2,4})-([\d.d]+)-([CP])$")
    for row in chain:
        iv = row.get("mark_iv")
        name = row.get("instrument_name", "")
        if iv is None:
            continue
        m = patt.match(name)
        if not m:
            continue
        try:
            exp = datetime.strptime(m.group(1), "%d%b%y").date()
            dte = (exp - today).days
            if dte <= 0:
                continue
            bucket.setdefault(dte, []).append(float(iv) / 100.0)
        except Exception:
            continue

    points: list[tuple[int, float]] = []
    for dte, vals in bucket.items():
        points.append((dte, float(np.nanmedian(vals))))
    points.sort(key=lambda x: x[0])
    return points


def iv_for_days(days: int, iv_points: list[tuple[int, float]], fallback_iv: float) -> float:
    if not iv_points:
        return fallback_iv
    xs = np.array([p[0] for p in iv_points], dtype=float)
    ys = np.array([p[1] for p in iv_points], dtype=float)
    return float(np.interp(float(days), xs, ys, left=ys[0], right=ys[-1]))


# ---------------------------------------------------------------------------
# Pricing math
# ---------------------------------------------------------------------------

def _prob_le(spot: float, vol: float, t: float, x: float) -> float:
    if t <= 0 or vol <= 1e-8:
        return 1.0 if spot <= x else 0.0
    s = vol * math.sqrt(t)
    m = math.log(spot) - 0.5 * s * s
    z = (math.log(x) - m) / s
    return _norm_cdf(z)


def _e_s_le(spot: float, vol: float, t: float, x: float) -> float:
    if t <= 0 or vol <= 1e-8:
        return spot if spot <= x else 0.0
    s = vol * math.sqrt(t)
    m = math.log(spot) - 0.5 * s * s
    z = (math.log(x) - m - s * s) / s
    return spot * _norm_cdf(z)


def expected_payoff_acc(spot: float, strike: float, ko: float, gearing: float, vol: float, t: float) -> float:
    p_le_k = _prob_le(spot, vol, t, strike)
    p_le_ko = _prob_le(spot, vol, t, ko)
    e_le_k = _e_s_le(spot, vol, t, strike)
    e_le_ko = _e_s_le(spot, vol, t, ko)
    ev_low = gearing * (e_le_k - strike * p_le_k)
    ev_mid = (e_le_ko - e_le_k) - strike * (p_le_ko - p_le_k)
    return ev_low + ev_mid


def expected_payoff_dcc(spot: float, strike: float, ko: float, gearing: float, vol: float, t: float) -> float:
    p_le_ko = _prob_le(spot, vol, t, ko)
    p_le_k = _prob_le(spot, vol, t, strike)
    e_le_ko = _e_s_le(spot, vol, t, ko)
    e_le_k = _e_s_le(spot, vol, t, strike)
    p_ge_k = 1.0 - p_le_k
    e_ge_k = max(spot - e_le_k, 0.0)
    ev_high = gearing * (strike * p_ge_k - e_ge_k)
    ev_mid = strike * (p_le_k - p_le_ko) - (e_le_k - e_le_ko)
    return ev_mid + ev_high


def estimate_accumulator_premium(
    contract: str,
    spot: float,
    strike_abs: float,
    ko_abs: float,
    gearing: float,
    total_size_units: float,
    fixing_dates: list[date],
    iv_points: list[tuple[int, float]],
    fallback_iv: float,
) -> tuple[float, list[dict]]:
    if spot <= 0 or total_size_units <= 0 or not fixing_dates:
        return (0.0, [])
    units_per_fixing = total_size_units / max(len(fixing_dates), 1)
    today = datetime.now(timezone.utc).date()
    total = 0.0
    rows: list[dict] = []
    for fx in fixing_dates:
        dte = max((fx - today).days, 1)
        t = dte / 365.0
        iv = iv_for_days(dte, iv_points, fallback_iv)
        if contract == "ACC":
            ev_unit = expected_payoff_acc(spot, strike_abs, ko_abs, gearing, iv, t)
        else:
            ev_unit = expected_payoff_dcc(spot, strike_abs, ko_abs, gearing, iv, t)
        usd = ev_unit * units_per_fixing
        total += usd
        rows.append({
            "fixing_date": fx.isoformat(),
            "dte": dte,
            "iv": iv,
            "size_units": units_per_fixing,
            "expected_usd": usd,
        })
    return (total, rows)


# ---------------------------------------------------------------------------
# Sensitivity + solver
# ---------------------------------------------------------------------------

def compute_sensitivity_curves(
    contract: str, spot: float, strike_mode: str, strike_val: float,
    ko_mode: str, ko_val: float, gearing: float, total_size_units: float,
    fixing_dates: list[date], iv_points: list[tuple[int, float]], fallback_iv: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strike_offsets = np.arange(-3.0, 3.01, 0.5)
    ko_offsets = np.arange(-3.0, 3.01, 0.5)
    strike_rows = []
    ko_rows = []

    for off in strike_offsets:
        if strike_mode == "pct":
            s_val = strike_val + off
            strike_abs = level_to_abs("pct", s_val, spot)
        else:
            s_val = strike_val + (off / 100.0) * spot
            strike_abs = s_val
        ko_abs = level_to_abs(ko_mode, ko_val, spot)
        prem, _ = estimate_accumulator_premium(
            contract, spot, strike_abs, ko_abs, gearing, total_size_units,
            fixing_dates, iv_points, fallback_iv,
        )
        strike_rows.append({"offset_pct_pts": off, "strike_abs": strike_abs, "premium_usd": prem})

    for off in ko_offsets:
        if ko_mode == "pct":
            k_val = ko_val + off
            ko_abs = level_to_abs("pct", k_val, spot)
        else:
            k_val = ko_val + (off / 100.0) * spot
            ko_abs = k_val
        strike_abs = level_to_abs(strike_mode, strike_val, spot)
        prem, _ = estimate_accumulator_premium(
            contract, spot, strike_abs, ko_abs, gearing, total_size_units,
            fixing_dates, iv_points, fallback_iv,
        )
        ko_rows.append({"offset_pct_pts": off, "ko_abs": ko_abs, "premium_usd": prem})

    return (pd.DataFrame(strike_rows), pd.DataFrame(ko_rows))


def solve_candidates(
    contract: str, spot: float, fixing_dates: list[date], target_premium: float,
    total_size_units: float, iv_points: list[tuple[int, float]], fallback_iv: float,
    strike_min_pct: float, strike_max_pct: float, ko_min_pct: float, ko_max_pct: float,
    step_pct: float, g_values: list[float],
) -> pd.DataFrame:
    out = []
    strike_grid = np.arange(strike_min_pct, strike_max_pct + 0.5 * step_pct, step_pct)
    ko_grid = np.arange(ko_min_pct, ko_max_pct + 0.5 * step_pct, step_pct)
    for g in g_values:
        for s_pct in strike_grid:
            strike_abs = spot * (1.0 + s_pct / 100.0)
            for k_pct in ko_grid:
                if contract == "ACC" and k_pct <= s_pct:
                    continue
                if contract == "DCC" and k_pct >= s_pct:
                    continue
                ko_abs = spot * (1.0 + k_pct / 100.0)
                prem, _ = estimate_accumulator_premium(
                    contract, spot, strike_abs, ko_abs, g, total_size_units,
                    fixing_dates, iv_points, fallback_iv,
                )
                out.append({
                    "strike_pct": s_pct, "ko_pct": k_pct, "gearing": g,
                    "premium_usd_est": prem, "target_error_usd": abs(prem - target_premium),
                })
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values("target_error_usd").head(25).reset_index(drop=True)


# ---------------------------------------------------------------------------
# State init
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "acc_contract": "ACC",
        "acc_size": 0.0,
        "acc_size_mode": "TOKEN",
        "acc_token": "BTC",
        "acc_freq": "W",
        "acc_start_date": date.today() + timedelta(days=1),
        "acc_end_date": date.today() + timedelta(days=91),
        "acc_spot": 95000.0,
        "acc_strike_mode": "pct",
        "acc_strike_val": -8.5,
        "acc_ko_mode": "pct",
        "acc_ko_val": 8.0,
        "acc_gearing": 2.0,
        "acc_weekly_usd_notional": 2_000_000.0,
        "acc_target_premium_usd": -60_000.0,
        "acc_fallback_iv_pct": 55.0,
        "acc_fixings": [],
        "acc_syntax_input": "",
        "acc_spot_mod_kind": "none",
        "acc_spot_mod_value": 0.0,
        "acc_vol_mod_raw": "",
        "acc_bps_mod_raw": "",
        "acc_notional_target": 0.0,
        "acc_qty_mult": 1.0,
        "acc_yield_target_enable": False,
        "acc_yield_target": 0.0,
        "acc_pricer_choice": "none",
        "acc_pricer_qty": 100,
        "acc_corr_enable": False,
        "acc_corr_value": 0.5,
        "acc_use_vdflt": False,
        "acc_use_asof": False,
        "acc_asof_date": date.today(),
        "acc_extra_modifiers": "",
        "acc_default_charge_token": "va0/0/2",
        "acc_charge_bps_override": 2.0,
        "acc_zero_premium_after_charge": True,
        "acc_solver_strike_min_pct": -15.0,
        "acc_solver_strike_max_pct": -3.0,
        "acc_solver_ko_min_pct": 5.0,
        "acc_solver_ko_max_pct": 15.0,
        "acc_solver_step_pct": 0.5,
        "acc_solver_g_mode": "fixed",
        "acc_solver_g_min": 1.0,
        "acc_solver_g_max": 3.0,
        "acc_solver_g_step": 0.5,
        "acc_var_strike_step_pct": 1.0,
        "acc_var_ko_step_pct": 2.0,
        "acc_var_gearing_step": 0.0,
        "acc_base_setup": None,
        "acc_varieties_df": pd.DataFrame(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if not st.session_state.acc_fixings:
        st.session_state.acc_fixings = build_fixing_schedule(
            st.session_state.acc_start_date,
            st.session_state.acc_end_date,
            st.session_state.acc_freq,
        )


def _apply_pending_updates() -> None:
    updates = st.session_state.pop("acc_pending_updates", None)
    if isinstance(updates, dict):
        for k, v in updates.items():
            st.session_state[k] = v


def _queue_updates_and_rerun(updates: dict, flash_msg: str) -> None:
    queued = st.session_state.get("acc_pending_updates", {})
    if not isinstance(queued, dict):
        queued = {}
    queued.update(updates)
    st.session_state["acc_pending_updates"] = queued
    st.session_state["acc_flash_msg"] = flash_msg
    st.rerun()


# ---------------------------------------------------------------------------
# Sheet helpers
# ---------------------------------------------------------------------------

def _input_sheet_rows() -> list[dict]:
    return [
        {"Field": "Contract", "Value": st.session_state.acc_contract, "Hint": "ACC or DCC"},
        {"Field": "Token", "Value": st.session_state.acc_token, "Hint": "BTC/ETH/SOL/HYPE"},
        {"Field": "Frequency", "Value": st.session_state.acc_freq, "Hint": "W (weekly) or D (daily)"},
        {"Field": "Size Mode", "Value": st.session_state.acc_size_mode, "Hint": "TOKEN or USD"},
        {"Field": "Accumulator Size", "Value": _fmt_sheet_number(st.session_state.acc_size, 4), "Hint": "Total size"},
        {"Field": "Start Date", "Value": st.session_state.acc_start_date.isoformat(), "Hint": "YYYY-MM-DD"},
        {"Field": "End Date", "Value": st.session_state.acc_end_date.isoformat(), "Hint": "YYYY-MM-DD"},
        {"Field": "Spot Ref", "Value": _fmt_sheet_number(st.session_state.acc_spot, 2), "Hint": "Spot reference"},
        {"Field": "Strike Mode", "Value": st.session_state.acc_strike_mode, "Hint": "pct or abs"},
        {"Field": "Strike Value", "Value": _fmt_sheet_number(st.session_state.acc_strike_val, 6), "Hint": "% or absolute"},
        {"Field": "KO Mode", "Value": st.session_state.acc_ko_mode, "Hint": "pct or abs"},
        {"Field": "KO Value", "Value": _fmt_sheet_number(st.session_state.acc_ko_val, 6), "Hint": "% or absolute"},
        {"Field": "Gearing", "Value": _fmt_sheet_number(st.session_state.acc_gearing, 4), "Hint": "g modifier"},
        {"Field": "Weekly USD Notional", "Value": _fmt_sheet_number(st.session_state.acc_weekly_usd_notional, 2), "Hint": "Weekly ticket"},
        {"Field": "Target Premium USD", "Value": _fmt_sheet_number(st.session_state.acc_target_premium_usd, 2), "Hint": "Solver target"},
        {"Field": "Fallback IV %", "Value": _fmt_sheet_number(st.session_state.acc_fallback_iv_pct, 2), "Hint": "When IV missing"},
    ]


def _parse_input_sheet(df: pd.DataFrame) -> tuple[dict, list[str]]:
    errs: list[str] = []
    m = {str(r["Field"]): r["Value"] for _, r in df.iterrows()}
    updates: dict = {}
    try:
        contract = str(m["Contract"]).upper().strip()
        if contract not in {"ACC", "DCC"}:
            errs.append("Contract must be ACC or DCC")
        else:
            updates["acc_contract"] = contract
        token = str(m["Token"]).upper().strip()
        if token not in set(SUPPORTED_TOKENS):
            errs.append(f"Token must be one of {SUPPORTED_TOKENS}")
        else:
            updates["acc_token"] = token
        freq = str(m["Frequency"]).upper().strip()
        if freq not in {"W", "D"}:
            errs.append("Frequency must be W or D")
        else:
            updates["acc_freq"] = freq
        size_mode = str(m["Size Mode"]).upper().strip()
        if size_mode not in {"TOKEN", "USD"}:
            errs.append("Size Mode must be TOKEN or USD")
        else:
            updates["acc_size_mode"] = size_mode
        updates["acc_size"] = _cell_to_float(m["Accumulator Size"])
        updates["acc_start_date"] = pd.to_datetime(m["Start Date"]).date()
        updates["acc_end_date"] = pd.to_datetime(m["End Date"]).date()
        updates["acc_spot"] = _cell_to_float(m["Spot Ref"])
        strike_mode = str(m["Strike Mode"]).strip().lower()
        if strike_mode not in {"pct", "abs"}:
            errs.append("Strike Mode must be pct or abs")
        else:
            updates["acc_strike_mode"] = strike_mode
        updates["acc_strike_val"] = _cell_to_float(m["Strike Value"])
        ko_mode = str(m["KO Mode"]).strip().lower()
        if ko_mode not in {"pct", "abs"}:
            errs.append("KO Mode must be pct or abs")
        else:
            updates["acc_ko_mode"] = ko_mode
        updates["acc_ko_val"] = _cell_to_float(m["KO Value"])
        updates["acc_gearing"] = _cell_to_float(m["Gearing"])
        updates["acc_weekly_usd_notional"] = _cell_to_float(m["Weekly USD Notional"])
        updates["acc_target_premium_usd"] = _cell_to_float(m["Target Premium USD"])
        updates["acc_fallback_iv_pct"] = _cell_to_float(m["Fallback IV %"])
    except Exception as e:
        errs.append(f"Sheet parse error: {e}")
    return updates, errs


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

_init_state()
_apply_pending_updates()

st.title("Accumulator Syntax Builder")
st.caption(
    "Build ACC/DCC syntax and JSON for sales flow. Premium uses a discrete "
    "European fixing heuristic (KO checked only on fixing dates)."
)
if st.session_state.get("acc_flash_msg"):
    st.success(st.session_state.pop("acc_flash_msg"))

# Syntax input
with st.expander("Paste existing ACC/DCC syntax", expanded=False):
    st.text_input("Syntax", key="acc_syntax_input",
                  placeholder="ACC BTC W-10Jan25-9Apr25 strike-10% ko10% g2 s78000 v5")

# Live preview
_top_preview_mods = serialize_modifiers()
if st.session_state.acc_default_charge_token and st.session_state.acc_default_charge_token not in _top_preview_mods:
    _top_preview_mods.append(st.session_state.acc_default_charge_token)
st.subheader("Syntax (Live Preview)")
st.code(build_current_syntax(_top_preview_mods))

# Deal sheet
st.subheader("Deal Sheet")
sheet_df = pd.DataFrame(_input_sheet_rows())
sheet_edited = st.data_editor(
    sheet_df, use_container_width=True, hide_index=True, key="acc_input_sheet_editor",
    column_config={
        "Field": st.column_config.TextColumn(disabled=True),
        "Value": st.column_config.TextColumn(),
        "Hint": st.column_config.TextColumn(disabled=True),
    },
)

# Contract inputs
c1, c2 = st.columns([1.25, 1.0], gap="large")
with c1:
    st.subheader("Contract Inputs")
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        st.selectbox("Contract", ["ACC", "DCC"], key="acc_contract")
    with col_b:
        st.selectbox("Token", SUPPORTED_TOKENS, key="acc_token")
    with col_c:
        st.radio("Frequency", ["W", "D"], horizontal=True, key="acc_freq")
    with col_d:
        st.number_input("Accumulator size", min_value=0.0, key="acc_size", format="%.6f")
    with col_e:
        st.selectbox("Size mode", ["TOKEN", "USD"], key="acc_size_mode")

    d1, d2, d3 = st.columns([1, 1, 1.2])
    with d1:
        st.date_input("Start date", key="acc_start_date")
    with d2:
        st.date_input("End date", key="acc_end_date")
    with d3:
        st.number_input("Spot ref", min_value=0.0, step=100.0, key="acc_spot", format="%.2f")

    s1, s2, s3 = st.columns(3)
    with s1:
        st.radio("Strike input", ["pct", "abs"], horizontal=True, key="acc_strike_mode")
        st.number_input("Strike value", key="acc_strike_val", format="%.6f")
    with s2:
        st.radio("KO input", ["pct", "abs"], horizontal=True, key="acc_ko_mode")
        st.number_input("KO value", key="acc_ko_val", format="%.6f")
    with s3:
        st.number_input("Gearing (g)", min_value=0.1, max_value=10.0, step=0.1, key="acc_gearing")
        st.number_input("Weekly USD notional", min_value=0.0, step=10000.0, key="acc_weekly_usd_notional", format="%.2f")
        st.number_input("Target premium USD", step=1000.0, key="acc_target_premium_usd", format="%.2f")

with c2:
    st.subheader("Market Context")
    st.number_input("Fallback IV %", min_value=1.0, max_value=300.0, key="acc_fallback_iv_pct", step=0.5)
    iv_points = fetch_deribit_iv_points(st.session_state.acc_token)
    if iv_points:
        iv_df = pd.DataFrame(iv_points, columns=["DTE", "ATM_IV"])
        iv_df["ATM_IV"] = iv_df["ATM_IV"] * 100.0
        st.dataframe(iv_df.head(12), use_container_width=True, hide_index=True)
        st.caption("Deribit ATM approximation from option summaries.")
    else:
        st.info("No Deribit IV curve available; solver uses fallback IV.")

# Fixing schedule
st.subheader("Fixing Schedule")
fx_df = pd.DataFrame({"fixing_date": pd.to_datetime(st.session_state.acc_fixings)})
edited = st.data_editor(fx_df, num_rows="dynamic", use_container_width=True, key="acc_fixing_editor")
st.caption(f"Fixing count: {len(st.session_state.acc_fixings)}")

# Optional modifiers
st.subheader("Optional Modifiers")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.selectbox("Spot mod", ["none", "s", "ss", "sd"], key="acc_spot_mod_kind")
    st.number_input("Spot mod value", min_value=0.0, key="acc_spot_mod_value", format="%.2f")
    st.text_input("Vol mod raw", key="acc_vol_mod_raw")
with m2:
    st.text_input("Bps mod raw", key="acc_bps_mod_raw")
    st.number_input("N target", min_value=0.0, key="acc_notional_target", format="%.2f")
    st.number_input("x multiplier", min_value=0.0, key="acc_qty_mult", format="%.4f")
with m3:
    st.checkbox("Enable y target", key="acc_yield_target_enable")
    st.number_input("y target", key="acc_yield_target", format="%.4f")
    st.selectbox("Pricer selector", ["none", "p", "p_qty", "w", "ww", "o"], key="acc_pricer_choice")
    if st.session_state.acc_pricer_choice == "p_qty":
        st.number_input("p quantity", min_value=1, key="acc_pricer_qty", step=1)
with m4:
    st.checkbox("Enable correlation r", key="acc_corr_enable")
    st.number_input("r value (0-1)", min_value=0.0, max_value=1.0, key="acc_corr_value", step=0.01)
    st.checkbox("Use vdflt", key="acc_use_vdflt")
    st.checkbox("Use aof date", key="acc_use_asof")
    if st.session_state.acc_use_asof:
        st.date_input("As-of date", key="acc_asof_date")
st.text_input("Extra modifiers (space-separated)", key="acc_extra_modifiers")

# Solver parameters
st.subheader("Target Premium Solver")
sv1, sv2, sv3, sv4 = st.columns(4)
with sv1:
    st.number_input("Strike min %", key="acc_solver_strike_min_pct", format="%.2f")
    st.number_input("Strike max %", key="acc_solver_strike_max_pct", format="%.2f")
with sv2:
    st.number_input("KO min %", key="acc_solver_ko_min_pct", format="%.2f")
    st.number_input("KO max %", key="acc_solver_ko_max_pct", format="%.2f")
with sv3:
    st.number_input("Grid step %", min_value=0.1, key="acc_solver_step_pct", format="%.2f")
    st.radio("Gearing mode", ["fixed", "range"], horizontal=True, key="acc_solver_g_mode")
with sv4:
    st.number_input("G min", min_value=0.1, key="acc_solver_g_min", format="%.2f")
    st.number_input("G max", min_value=0.1, key="acc_solver_g_max", format="%.2f")
    st.number_input("G step", min_value=0.1, key="acc_solver_g_step", format="%.2f")
    st.number_input("Charge bps", min_value=0.0, key="acc_charge_bps_override", format="%.2f")

# Compute derived values
charge_bps = st.session_state.acc_charge_bps_override
if st.session_state.acc_default_charge_token:
    charge_bps = _extract_charge_bps_from_va(st.session_state.acc_default_charge_token) or charge_bps
total_size_units = resolve_total_size_units(
    st.session_state.acc_size, st.session_state.acc_size_mode,
    st.session_state.acc_weekly_usd_notional, st.session_state.acc_spot,
    len(st.session_state.acc_fixings),
)
per_fixing_size_units = total_size_units / max(len(st.session_state.acc_fixings), 1)
total_usd_notional = (
    st.session_state.acc_size if st.session_state.acc_size_mode == "USD"
    else total_size_units * st.session_state.acc_spot
)
charge_usd = estimate_charge_usd(total_usd_notional, charge_bps)
if st.session_state.acc_zero_premium_after_charge:
    solver_target_premium = -charge_usd
else:
    solver_target_premium = st.session_state.acc_target_premium_usd

st.caption(
    f"Total size: {_fmt_readable(total_size_units, 4)} units | "
    f"Per fixing: {_fmt_readable(per_fixing_size_units, 4)} units | "
    f"Notional: {_fmt_readable(total_usd_notional, 2)} USD"
)

# Variety generator
st.subheader("Variety Generator")
vg1, vg2, vg3 = st.columns(3)
with vg1:
    st.number_input("Strike step (pct pts)", min_value=0.0, key="acc_var_strike_step_pct", format="%.2f")
with vg2:
    st.number_input("KO step (pct pts)", min_value=0.0, key="acc_var_ko_step_pct", format="%.2f")
with vg3:
    st.number_input("Gearing step", min_value=0.0, key="acc_var_gearing_step", format="%.2f")

if st.session_state.acc_solver_g_mode == "fixed":
    g_grid = [float(st.session_state.acc_gearing)]
else:
    g_grid = list(np.arange(
        st.session_state.acc_solver_g_min,
        st.session_state.acc_solver_g_max + 0.5 * st.session_state.acc_solver_g_step,
        st.session_state.acc_solver_g_step,
    ))

# Workflow actions
with st.container(border=True):
    st.subheader("Workflow Actions")
    act0, act1, act2, act3, act4, act5 = st.columns(6)
    with act0:
        do_apply_sheet = st.button("Apply Sheet", use_container_width=True)
    with act1:
        do_parse = st.button("Parse Syntax", use_container_width=True)
    with act2:
        do_pull_spot = st.button("Pull Spot", use_container_width=True)
    with act3:
        do_regen_schedule = st.button("Regen Schedule", use_container_width=True)
    with act4:
        do_apply_schedule = st.button("Apply Schedule", use_container_width=True)
    with act5:
        do_run_solver = st.button("Run Solver", type="primary", use_container_width=True)

    act6, act7, act8, act9 = st.columns(4)
    with act6:
        do_apply_best = st.button("Apply Best", use_container_width=True)
    with act7:
        do_set_base = st.button("Set as Base", use_container_width=True)
    with act8:
        do_make_varieties = st.button("Generate Varieties", use_container_width=True)
    with act9:
        do_shift_fwd = st.button("Shift Fixings +1", use_container_width=True)

# Action handlers
if do_apply_sheet:
    updates, errors = _parse_input_sheet(sheet_edited)
    if errors:
        for err in errors:
            st.error(err)
    else:
        updates["acc_fixings"] = build_fixing_schedule(
            updates["acc_start_date"], updates["acc_end_date"], updates["acc_freq"]
        )
        _queue_updates_and_rerun(updates, "Applied spreadsheet inputs.")

if do_parse:
    parsed = parse_accumulator_syntax(st.session_state.acc_syntax_input)
    if parsed is None:
        st.error("Could not parse syntax.")
    else:
        updates = {}
        for field in ["contract", "token", "freq", "start_date", "end_date",
                      "strike_mode", "strike_val", "ko_mode", "ko_val", "gearing"]:
            updates[f"acc_{field}"] = parsed[field]
        updates["acc_size"] = float(parsed["size"] or 0.0)
        updates["acc_size_mode"] = "TOKEN"
        updates["acc_extra_modifiers"] = " ".join(parsed["modifiers"])
        updates["acc_fixings"] = build_fixing_schedule(parsed["start_date"], parsed["end_date"], parsed["freq"])
        _queue_updates_and_rerun(updates, "Parsed syntax into fields.")

if do_pull_spot:
    spot_val = fetch_deribit_spot(st.session_state.acc_token)
    if spot_val is None:
        st.warning("Could not fetch spot from Deribit.")
    else:
        _queue_updates_and_rerun({"acc_spot": float(spot_val)}, f"Spot updated to {spot_val:,.2f}")

if do_regen_schedule:
    st.session_state.acc_fixings = build_fixing_schedule(
        st.session_state.acc_start_date, st.session_state.acc_end_date, st.session_state.acc_freq
    )

if do_apply_schedule:
    new_fixings: list[date] = []
    for v in edited["fixing_date"].tolist():
        if pd.isna(v):
            continue
        new_fixings.append(pd.to_datetime(v).date())
    st.session_state.acc_fixings = sorted(set(new_fixings))

shift_days = 1 if st.session_state.acc_freq == "D" else 7
if do_shift_fwd:
    st.session_state.acc_fixings = [d + timedelta(days=shift_days) for d in st.session_state.acc_fixings]

if do_run_solver:
    solver_df = solve_candidates(
        st.session_state.acc_contract, st.session_state.acc_spot, st.session_state.acc_fixings,
        solver_target_premium, total_size_units, iv_points,
        st.session_state.acc_fallback_iv_pct / 100.0,
        st.session_state.acc_solver_strike_min_pct, st.session_state.acc_solver_strike_max_pct,
        st.session_state.acc_solver_ko_min_pct, st.session_state.acc_solver_ko_max_pct,
        st.session_state.acc_solver_step_pct, g_grid,
    )
    st.session_state.acc_solver_results = solver_df

solver_results = st.session_state.get("acc_solver_results", pd.DataFrame())
if isinstance(solver_results, pd.DataFrame) and not solver_results.empty:
    st.dataframe(solver_results, use_container_width=True, hide_index=True)

if do_apply_best and isinstance(solver_results, pd.DataFrame) and not solver_results.empty:
    top = solver_results.iloc[0]
    _queue_updates_and_rerun({
        "acc_strike_mode": "pct", "acc_ko_mode": "pct",
        "acc_strike_val": float(top["strike_pct"]),
        "acc_ko_val": float(top["ko_pct"]),
        "acc_gearing": float(top["gearing"]),
    }, "Applied best suggestion.")

if do_set_base:
    st.session_state.acc_base_setup = {
        "strike_mode": st.session_state.acc_strike_mode,
        "strike_val": float(st.session_state.acc_strike_val),
        "ko_mode": st.session_state.acc_ko_mode,
        "ko_val": float(st.session_state.acc_ko_val),
        "gearing": float(st.session_state.acc_gearing),
        "size": float(total_size_units),
    }
    st.success("Base setup captured.")

# Compute outputs
mods = serialize_modifiers()
if st.session_state.acc_default_charge_token and st.session_state.acc_default_charge_token not in mods:
    mods.append(st.session_state.acc_default_charge_token)
mod_errors = modifiers_validation(mods)
strike_abs = level_to_abs(st.session_state.acc_strike_mode, st.session_state.acc_strike_val, st.session_state.acc_spot)
ko_abs = level_to_abs(st.session_state.acc_ko_mode, st.session_state.acc_ko_val, st.session_state.acc_spot)

estimated_premium, by_fixing = estimate_accumulator_premium(
    st.session_state.acc_contract, st.session_state.acc_spot, strike_abs, ko_abs,
    st.session_state.acc_gearing, total_size_units, st.session_state.acc_fixings,
    iv_points, st.session_state.acc_fallback_iv_pct / 100.0,
)

syntax = build_current_syntax(mods)
payload = {
    "contract": st.session_state.acc_contract,
    "token": st.session_state.acc_token,
    "size": st.session_state.acc_size,
    "size_mode": st.session_state.acc_size_mode,
    "total_size_units": total_size_units,
    "per_fixing_size_units": per_fixing_size_units,
    "frequency": st.session_state.acc_freq,
    "start_date": st.session_state.acc_start_date.isoformat(),
    "end_date": st.session_state.acc_end_date.isoformat(),
    "spot_ref": st.session_state.acc_spot,
    "strike": {"mode": st.session_state.acc_strike_mode, "value": st.session_state.acc_strike_val, "absolute": strike_abs},
    "ko": {"mode": st.session_state.acc_ko_mode, "value": st.session_state.acc_ko_val, "absolute": ko_abs},
    "gearing": st.session_state.acc_gearing,
    "weekly_usd_notional": st.session_state.acc_weekly_usd_notional,
    "total_usd_notional": total_usd_notional,
    "fixings": [d.isoformat() for d in st.session_state.acc_fixings],
    "modifiers": mods,
    "syntax": syntax,
    "pricing": {
        "method": "discrete european fixing heuristic",
        "estimated_premium_usd": estimated_premium,
        "estimated_charge_usd": charge_usd,
        "estimated_net_premium_usd": estimated_premium + charge_usd,
        "fallback_iv_pct": st.session_state.acc_fallback_iv_pct,
    },
}

# Variety generation
if do_make_varieties:
    base = st.session_state.acc_base_setup
    if not base:
        st.warning("Set the current deal as base first.")
    elif base["strike_mode"] != "pct" or base["ko_mode"] != "pct":
        st.warning("Variety generator requires strike and KO in pct mode.")
    else:
        rows = []
        labels = ["Variety 1", "Variety 2", "Variety 3"]
        strike_offsets = [st.session_state.acc_var_strike_step_pct, 0.0, -st.session_state.acc_var_strike_step_pct]
        ko_offsets = [st.session_state.acc_var_ko_step_pct, 0.0, -st.session_state.acc_var_ko_step_pct]
        gear_offsets = [st.session_state.acc_var_gearing_step, 0.0, -st.session_state.acc_var_gearing_step]
        for i in range(3):
            strike_pct = base["strike_val"] + strike_offsets[i]
            ko_pct = base["ko_val"] + ko_offsets[i]
            gearing_i = max(0.1, base["gearing"] + gear_offsets[i])
            strike_abs_i = st.session_state.acc_spot * (1.0 + strike_pct / 100.0)
            ko_abs_i = st.session_state.acc_spot * (1.0 + ko_pct / 100.0)
            prem_i, _ = estimate_accumulator_premium(
                st.session_state.acc_contract, st.session_state.acc_spot,
                strike_abs_i, ko_abs_i, gearing_i, total_size_units,
                st.session_state.acc_fixings, iv_points, st.session_state.acc_fallback_iv_pct / 100.0,
            )
            rows.append({
                "label": labels[i], "strike_pct": strike_pct, "ko_pct": ko_pct,
                "gearing": gearing_i, "estimated_premium_usd": prem_i,
            })
        st.session_state.acc_varieties_df = pd.DataFrame(rows)

# Output section
st.subheader("Outputs")
if mod_errors:
    for err in mod_errors:
        st.warning(err)

o1, o2 = st.columns(2)
with o1:
    st.text_area("Pricer syntax", value=syntax, height=90)
with o2:
    st.metric("Estimated premium (USD)", f"{estimated_premium:,.2f}")
    st.metric("Estimated charge (USD)", f"{charge_usd:,.2f}")
    st.metric("Estimated net premium (USD)", f"{(estimated_premium + charge_usd):,.2f}")
    st.metric("Strike absolute", f"{strike_abs:,.2f}")
    st.metric("KO absolute", f"{ko_abs:,.2f}")

# Charts
st.subheader("Interpretation Charts")
fix_df = pd.DataFrame(by_fixing)
ch1, ch2 = st.columns(2)
with ch1:
    if not fix_df.empty:
        fix_df["fixing_date"] = pd.to_datetime(fix_df["fixing_date"])
        fig_fix = go.Figure()
        fig_fix.add_trace(go.Bar(x=fix_df["fixing_date"], y=fix_df["expected_usd"], name="Premium by fixing"))
        fig_fix.update_layout(**PLOTLY_LAYOUT, title="Per-fixing Premium", xaxis_title="Date", yaxis_title="USD", height=320)
        st.plotly_chart(fig_fix, use_container_width=True)

with ch2:
    if not fix_df.empty:
        fig_iv = go.Figure()
        fig_iv.add_trace(go.Scatter(x=fix_df["dte"], y=fix_df["iv"].apply(lambda x: x * 100), mode="lines+markers", name="IV"))
        fig_iv.update_layout(**PLOTLY_LAYOUT, title="IV by Fixing", xaxis_title="DTE", yaxis_title="IV %", height=320)
        st.plotly_chart(fig_iv, use_container_width=True)

sens_strike_df, sens_ko_df = compute_sensitivity_curves(
    st.session_state.acc_contract, st.session_state.acc_spot,
    st.session_state.acc_strike_mode, st.session_state.acc_strike_val,
    st.session_state.acc_ko_mode, st.session_state.acc_ko_val,
    st.session_state.acc_gearing, total_size_units, st.session_state.acc_fixings,
    iv_points, st.session_state.acc_fallback_iv_pct / 100.0,
)

ch3, ch4 = st.columns(2)
with ch3:
    if not sens_strike_df.empty:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=sens_strike_df["offset_pct_pts"], y=sens_strike_df["premium_usd"], mode="lines+markers"))
        fig_s.update_layout(**PLOTLY_LAYOUT, title="Premium vs Strike", xaxis_title="Strike offset (pct)", yaxis_title="USD", height=320)
        st.plotly_chart(fig_s, use_container_width=True)
with ch4:
    if not sens_ko_df.empty:
        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(x=sens_ko_df["offset_pct_pts"], y=sens_ko_df["premium_usd"], mode="lines+markers"))
        fig_k.update_layout(**PLOTLY_LAYOUT, title="Premium vs KO", xaxis_title="KO offset (pct)", yaxis_title="USD", height=320)
        st.plotly_chart(fig_k, use_container_width=True)

# JSON download
st.download_button(
    "Download JSON", data=json.dumps(payload, indent=2),
    file_name="accumulator_payload.json", mime="application/json",
)

# Varieties display
varieties_df = st.session_state.get("acc_varieties_df", pd.DataFrame())
if isinstance(varieties_df, pd.DataFrame) and not varieties_df.empty:
    st.subheader("Generated Varieties")
    st.dataframe(varieties_df, use_container_width=True, hide_index=True)

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=varieties_df["label"], y=varieties_df["estimated_premium_usd"],
        marker_color=["#4e79a7", "#59a14f", "#e15759"],
    ))
    fig_comp.update_layout(**PLOTLY_LAYOUT, title="Varieties Premium Comparison", height=320)
    st.plotly_chart(fig_comp, use_container_width=True)
