"""
Vol Surface Upload — Screenshot upload and parse tool.

Upload a screenshot of a vol-surface table, extract rows via LLM vision,
and run forward-vol / carry analyses.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import base64
import io
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from lib.deribit import get_instruments
from lib.telegram import send_photo, is_configured, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from lib.constants import PLOTLY_LAYOUT

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Vol Surface Upload",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Vol Surface Upload")
st.caption("Upload a screenshot of a surface table. Extract rows and run forward-vol analyses.")


# ---------------------------------------------------------------------------
# Constants and data structures
# ---------------------------------------------------------------------------

SURFACE_COLUMNS = [
    "tenor", "stat_atm_vol", "stat_yield",
    "targets_atm_vol", "targets_yield", "targets_25rr", "targets_25bf", "targets_rho", "targets_vov",
    "params_atm_vol", "params_yield", "params_25rr", "params_25bf", "params_rho", "params_vov",
    "calib_atm_vol", "calib_yield", "calib_25rr", "calib_25bf", "calib_rho", "calib_vov",
]

DERIBIT_CURRENCY_MAP = {
    "BTC": "BTC",
    "ETH": "ETH",
    "SOL": "SOL_USDC",
    "HYPE": "HYPE_USDC",
}


@dataclass
class SurfaceRow:
    tenor: str
    stat_atm_vol: Optional[float] = None
    stat_yield: Optional[float] = None
    targets_atm_vol: Optional[float] = None
    targets_yield: Optional[float] = None
    targets_25rr: Optional[float] = None
    targets_25bf: Optional[float] = None
    targets_rho: Optional[float] = None
    targets_vov: Optional[float] = None
    params_atm_vol: Optional[float] = None
    params_yield: Optional[float] = None
    params_25rr: Optional[float] = None
    params_25bf: Optional[float] = None
    params_rho: Optional[float] = None
    params_vov: Optional[float] = None
    calib_atm_vol: Optional[float] = None
    calib_yield: Optional[float] = None
    calib_25rr: Optional[float] = None
    calib_25bf: Optional[float] = None
    calib_rho: Optional[float] = None
    calib_vov: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_surface_df() -> pd.DataFrame:
    default_tenors = [
        "15AUG26", "22AUG26", "29AUG26", "26SEP26",
        "31OCT26", "26DEC26", "27MAR27", "25JUN27",
    ]
    return pd.DataFrame([{"tenor": t} for t in default_tenors], columns=SURFACE_COLUMNS)


def _normalize_tenor_label(ts: pd.Timestamp) -> str:
    return ts.strftime("%d%b%y").upper()


def _parse_deribit_expiry_label(label: str) -> Optional[pd.Timestamp]:
    text = str(label).strip().upper()
    for fmt in ("%d%b%y", "%d%b%Y"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue
    return None


@st.cache_data(show_spinner=False, ttl=300, max_entries=64)
def fetch_deribit_expiry_tenors(base_asset: str) -> List[str]:
    """Get list of live expiry tenor labels from Deribit."""
    base = str(base_asset or "").strip().upper()
    currency = DERIBIT_CURRENCY_MAP.get(base)
    if not currency:
        return []
    instruments = get_instruments(currency, "option")
    if not instruments:
        return []
    expiries: set[pd.Timestamp] = set()
    for r in instruments:
        name = str(r.get("instrument_name", ""))
        parts = name.split("-")
        if len(parts) < 2:
            continue
        exp = _parse_deribit_expiry_label(parts[1])
        if exp is not None:
            expiries.add(exp.normalize())
    if not expiries:
        return []
    return [_normalize_tenor_label(ts) for ts in sorted(expiries)]


def _sync_manual_df_to_tenors(df: pd.DataFrame, tenors: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame([{"tenor": t} for t in tenors], columns=SURFACE_COLUMNS)
    existing = df.copy()
    if "tenor" not in existing.columns:
        existing["tenor"] = ""
    existing["tenor"] = existing["tenor"].astype(str).str.upper().str.strip()
    rows: List[Dict[str, Any]] = []
    for t in tenors:
        match = existing[existing["tenor"] == t]
        if not match.empty:
            row = match.iloc[0].to_dict()
            row["tenor"] = t
            rows.append(row)
        else:
            rows.append({"tenor": t})
    synced = pd.DataFrame(rows)
    for c in SURFACE_COLUMNS:
        if c not in synced.columns:
            synced[c] = np.nan
    return synced[SURFACE_COLUMNS]


def _parse_pct_or_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    s = str(value).strip()
    if not s or s in {"-", "—", "–", "n/a", "N/A", "None"}:
        return None
    s = s.replace(",", "")
    try:
        if s.endswith("%"):
            return float(s[:-1])
        return float(s)
    except ValueError:
        return None


def _extract_key(payload: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in payload:
            return payload[k]
    return None


def _normalize_rows(raw_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[SurfaceRow] = []
    for r in raw_rows:
        tenor_val = _extract_key(r, "tenor", "maturity", "expiry", "date")
        if tenor_val is None:
            continue
        rows.append(SurfaceRow(
            tenor=str(tenor_val).strip().upper(),
            stat_atm_vol=_parse_pct_or_float(_extract_key(r, "stat_atm_vol", "statModel_atm_vol")),
            stat_yield=_parse_pct_or_float(_extract_key(r, "stat_yield", "statModel_yield")),
            targets_atm_vol=_parse_pct_or_float(_extract_key(r, "targets_atm_vol", "target_atm_vol")),
            targets_yield=_parse_pct_or_float(_extract_key(r, "targets_yield", "target_yield")),
            targets_25rr=_parse_pct_or_float(_extract_key(r, "targets_25rr", "target_25rr")),
            targets_25bf=_parse_pct_or_float(_extract_key(r, "targets_25bf", "target_25bf")),
            targets_rho=_parse_pct_or_float(_extract_key(r, "targets_rho", "target_rho")),
            targets_vov=_parse_pct_or_float(_extract_key(r, "targets_vov", "target_vov")),
            params_atm_vol=_parse_pct_or_float(_extract_key(r, "params_atm_vol", "model_params_atm_vol")),
            params_yield=_parse_pct_or_float(_extract_key(r, "params_yield", "model_params_yield")),
            params_25rr=_parse_pct_or_float(_extract_key(r, "params_25rr", "model_params_25rr")),
            params_25bf=_parse_pct_or_float(_extract_key(r, "params_25bf", "model_params_25bf")),
            params_rho=_parse_pct_or_float(_extract_key(r, "params_rho", "model_params_rho")),
            params_vov=_parse_pct_or_float(_extract_key(r, "params_vov", "model_params_vov")),
            calib_atm_vol=_parse_pct_or_float(_extract_key(r, "calib_atm_vol", "calibration_atm_vol")),
            calib_yield=_parse_pct_or_float(_extract_key(r, "calib_yield", "calibration_yield")),
            calib_25rr=_parse_pct_or_float(_extract_key(r, "calib_25rr", "calibration_25rr")),
            calib_25bf=_parse_pct_or_float(_extract_key(r, "calib_25bf", "calibration_25bf")),
            calib_rho=_parse_pct_or_float(_extract_key(r, "calib_rho", "calibration_rho")),
            calib_vov=_parse_pct_or_float(_extract_key(r, "calib_vov", "calibration_vov")),
        ))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([r.__dict__ for r in rows])


def _sortable_tenor_key(s: str) -> tuple[int, str]:
    s = str(s).strip().upper()
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    if len(s) >= 7 and s[0:2].isdigit():
        mmm = s[2:5]
        yy = s[5:7]
        if mmm in month_map and yy.isdigit():
            return (0, f"20{yy}{month_map[mmm]:02d}{int(s[0:2]):02d}")
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.notna(dt):
            return (0, dt.strftime("%Y%m%d"))
    except Exception:
        pass
    return (1, s)


def _parse_tenor_to_date(tenor: str) -> Optional[pd.Timestamp]:
    s = str(tenor).strip().upper()
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    try:
        if len(s) >= 7 and s[0:2].isdigit() and s[2:5] in month_map and s[5:7].isdigit():
            day = int(s[0:2])
            month = month_map[s[2:5]]
            year = 2000 + int(s[5:7])
            return pd.Timestamp(year=year, month=month, day=day)
    except Exception:
        pass
    parsed = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return parsed if pd.notna(parsed) else None


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

def _get_provider_key(provider: str) -> Optional[str]:
    provider = provider.lower().strip()
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return key
        try:
            return st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai", {}).get("api_key")
        except Exception:
            return None
    else:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            return key
        try:
            return st.secrets.get("ANTHROPIC_API_KEY") or st.secrets.get("anthropic", {}).get("api_key")
        except Exception:
            return None


def _load_json_from_model_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


def _extract_rows_openai(image_bytes: bytes, model: str, api_key: str) -> Dict[str, Any]:
    import openai as _openai

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = (
        "You are parsing a volatility surface table from an image.\n"
        "Return STRICT JSON only with shape:\n"
        '{"pair":"BTC/USD","rows":[{"tenor":"28MAY26","stat_atm_vol":"77.01%","stat_yield":"0.00%",'
        '"targets_atm_vol":"110","targets_yield":"","targets_25rr":"5","targets_25bf":"2.5",'
        '"targets_rho":"12.18%","targets_vov":"1969.3",'
        '"params_atm_vol":"111.18%","params_yield":"0.00%","params_25rr":"5.00%","params_25bf":"2.35%",'
        '"params_rho":"-","params_vov":"-",'
        '"calib_atm_vol":"","calib_yield":"","calib_25rr":"","calib_25bf":"","calib_rho":"","calib_vov":""}]}\n'
        "Rules: include pair from header, keep % signs, empty string for missing values, include all rows."
    )
    client = _openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model, temperature=0,
        messages=[
            {"role": "system", "content": "You extract financial table data from screenshots into strict JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ],
    )
    content = response.choices[0].message.content if response.choices else ""
    payload = _load_json_from_model_text(content)
    if not isinstance(payload, dict):
        return {"pair": None, "rows": []}
    rows = payload.get("rows", [])
    return {"pair": payload.get("pair"), "rows": rows if isinstance(rows, list) else []}


def _extract_rows_anthropic(image_bytes: bytes, model: str, api_key: str) -> Dict[str, Any]:
    import anthropic as _anth

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = (
        "Parse the table in this image and return STRICT JSON only with keys 'pair' and 'rows'. "
        "Each row: tenor, stat_atm_vol, stat_yield, targets_atm_vol, targets_yield, "
        "targets_25rr, targets_25bf, targets_rho, targets_vov, params_atm_vol, params_yield, "
        "params_25rr, params_25bf, params_rho, params_vov, calib_atm_vol, calib_yield, calib_25rr, "
        "calib_25bf, calib_rho, calib_vov. Use empty string for missing values."
    )
    client = _anth.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model, max_tokens=4096, temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
        ]}],
    )
    text = ""
    for part in msg.content:
        if getattr(part, "type", "") == "text":
            text = part.text
            break
    payload = _load_json_from_model_text(text)
    if not isinstance(payload, dict):
        return {"pair": None, "rows": []}
    rows = payload.get("rows", [])
    return {"pair": payload.get("pair"), "rows": rows if isinstance(rows, list) else []}


@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def extract_surface_rows(image_bytes: bytes, provider: str, model: str, api_key: str) -> Dict[str, Any]:
    if provider == "anthropic":
        return _extract_rows_anthropic(image_bytes, model, api_key)
    return _extract_rows_openai(image_bytes, model, api_key)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _safe_vol_to_decimal(v: Any) -> float:
    if v is None:
        return np.nan
    try:
        vf = float(v)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(vf):
        return np.nan
    return vf / 100.0 if abs(vf) > 3 else vf


def _build_term_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    prefix = source.strip().lower()
    out = df[["tenor"]].copy()
    out["atm"] = pd.to_numeric(df.get(f"{prefix}_atm_vol"), errors="coerce")
    out["rr"] = pd.to_numeric(df.get(f"{prefix}_25rr"), errors="coerce")
    out["bf"] = pd.to_numeric(df.get(f"{prefix}_25bf"), errors="coerce")
    out["call25"] = out["atm"] + out["bf"] + 0.5 * out["rr"]
    out["put25"] = out["atm"] + out["bf"] - 0.5 * out["rr"]
    out["expiry"] = out["tenor"].apply(_parse_tenor_to_date)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

    def _to_naive(ts):
        if not isinstance(ts, pd.Timestamp) or pd.isna(ts):
            return None
        t = ts.tz_localize(None) if ts.tzinfo is not None else ts
        return t.normalize()

    out["dte"] = out["expiry"].apply(
        lambda d: int((_to_naive(d) - today).days) if _to_naive(d) is not None else np.nan
    )
    if out["dte"].isna().all():
        out["dte"] = np.arange(1, len(out) + 1) * 30
    else:
        out["dte"] = out["dte"].interpolate().bfill().ffill()
    out = out.sort_values("dte").reset_index(drop=True)
    return out[out["dte"] > 0].reset_index(drop=True)


def _build_marked_term_df(df: pd.DataFrame) -> pd.DataFrame:
    params_df = _build_term_df(df, source="params")
    calib_df = _build_term_df(df, source="calib")
    if params_df.empty and calib_df.empty:
        return pd.DataFrame()
    if params_df.empty:
        return calib_df.copy()
    if calib_df.empty:
        return params_df.copy()
    merged = params_df.copy()
    for c in ["atm", "rr", "bf", "call25", "put25"]:
        merged[c] = calib_df[c].combine_first(params_df[c])
    return merged


def _forward_vols_from_term(term_df: pd.DataFrame, vol_col: str) -> List[float]:
    fwd: List[float] = []
    for i in range(len(term_df)):
        v_now = term_df.loc[i, vol_col]
        t_now = float(term_df.loc[i, "dte"]) / 365.0
        if pd.isna(v_now) or t_now <= 0:
            fwd.append(np.nan)
            continue
        v_now_dec = _safe_vol_to_decimal(v_now)
        if i == 0:
            fwd.append(v_now_dec * 100.0)
            continue
        v_prev = term_df.loc[i - 1, vol_col]
        t_prev = float(term_df.loc[i - 1, "dte"]) / 365.0
        if pd.isna(v_prev) or t_now <= t_prev:
            fwd.append(v_now_dec * 100.0)
            continue
        v_prev_dec = _safe_vol_to_decimal(v_prev)
        var_fwd = (v_now_dec ** 2 * t_now - v_prev_dec ** 2 * t_prev) / (t_now - t_prev)
        fwd.append(np.sqrt(max(0.0, var_fwd)) * 100.0)
    return fwd


def _build_combined_forward_vols_chart(term_df: pd.DataFrame, quant_df=None) -> Optional[go.Figure]:
    specs = [("atm", "ATM", "#4e79a7"), ("call25", "25d Call", "#59a14f"), ("put25", "25d Put", "#e15759")]
    if len(term_df) < 2:
        return None
    x_cat = term_df["tenor"].astype(str).tolist()
    fig = go.Figure()
    has_series = False
    for col, label, color in specs:
        if col not in term_df or term_df[col].notna().sum() < 2:
            continue
        spot = pd.to_numeric(term_df[col], errors="coerce")
        fwd = _forward_vols_from_term(term_df, col)
        fig.add_trace(go.Scatter(x=x_cat, y=spot, name=f"{label} Vol", line=dict(color=color, width=2.2, shape="spline"), mode="lines+markers"))
        fig.add_trace(go.Scatter(x=x_cat, y=fwd, name=f"{label} Fwd", line=dict(color=color, width=1.8, dash="dash", shape="spline"), mode="lines+markers", opacity=0.75))
        has_series = True
    if not has_series:
        return None
    if isinstance(quant_df, pd.DataFrame) and not quant_df.empty and "atm" in quant_df.columns:
        fig.add_trace(go.Scatter(
            x=quant_df["tenor"].astype(str).tolist(),
            y=pd.to_numeric(quant_df["atm"], errors="coerce"),
            name="Quant ATM (Stat)", line=dict(color="#8c564b", width=2, dash="dot"), mode="lines+markers",
        ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Combined Forward Vols", xaxis_title="Expiry", yaxis_title="IV", height=500,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0))
    return fig


def _build_term_structure_chart(marked_df, metric_col, metric_label, quant_df=None, color="#4e79a7"):
    if metric_col not in marked_df.columns or marked_df[metric_col].notna().sum() < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=marked_df["tenor"].astype(str).tolist(),
        y=pd.to_numeric(marked_df[metric_col], errors="coerce"),
        name=f"Marked {metric_label}", line=dict(color=color, width=2.4, shape="spline"), mode="lines+markers",
    ))
    if isinstance(quant_df, pd.DataFrame) and not quant_df.empty and metric_col in quant_df.columns:
        fig.add_trace(go.Scatter(
            x=quant_df["tenor"].astype(str).tolist(),
            y=pd.to_numeric(quant_df[metric_col], errors="coerce"),
            name=f"Quant {metric_label}", line=dict(color="#8c564b", width=2, dash="dot"), mode="lines+markers",
        ))
    fig.update_layout(**PLOTLY_LAYOUT, title=f"{metric_label} Term Structure", xaxis_title="Expiry",
                      yaxis_title=metric_label, height=360,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0))
    return fig


def _build_forward_vol_matrix(term_df, vol_col, vol_label):
    sub = term_df[["tenor", "dte", vol_col]].dropna().copy()
    if len(sub) < 2:
        return None
    sub = sub.sort_values("dte").reset_index(drop=True)
    dtes = sub["dte"].astype(float).tolist()
    labels = sub["tenor"].astype(str).tolist()
    iv_dec = [_safe_vol_to_decimal(v) for v in sub[vol_col].tolist()]
    n = len(sub)
    z = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            t_i = dtes[i] / 365.0
            t_j = dtes[j] / 365.0
            v_i, v_j = iv_dec[i], iv_dec[j]
            if pd.isna(v_i) or not np.isfinite(v_i):
                continue
            if i == j:
                z[i, j] = v_i * 100.0
                continue
            if pd.isna(v_j) or not np.isfinite(v_j) or t_j <= t_i:
                continue
            fwd_var = ((v_j ** 2) * t_j - (v_i ** 2) * t_i) / (t_j - t_i)
            if fwd_var >= 0:
                z[i, j] = np.sqrt(fwd_var) * 100.0
    z_min, z_max = np.nanmin(z), np.nanmax(z)
    if np.isnan(z_min) or np.isnan(z_max) or z_max <= z_min:
        z_min, z_max = 0, 100
    text_mat = [[f"{z[i, j]:.1f}" if np.isfinite(z[i, j]) else "" for j in range(n)] for i in range(n)]
    fig = go.Figure(data=go.Heatmap(
        x=labels, y=labels, z=z, text=text_mat, texttemplate="%{text}", textfont=dict(size=11),
        showscale=True, colorscale=[[0, "#e15759"], [0.5, "#FFF9C4"], [1, "#59a14f"]],
        zmin=z_min, zmax=z_max, hoverongaps=False,
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=f"{vol_label} Forward Vol Matrix (%)",
                      xaxis_title="Expiry (to)", yaxis_title="Expiry (from)",
                      xaxis=dict(type="category", tickangle=-45, side="bottom"),
                      yaxis=dict(type="category", autorange="reversed"),
                      height=400 + max(0, (n - 6) * 28))
    return fig


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

ADVANCED_MODEL_PRIORITY = [
    ("anthropic", "claude-sonnet-4-20250514"),
    ("openai", "gpt-4.1"),
]


def _resolve_advanced_model():
    for provider, model in ADVANCED_MODEL_PRIORITY:
        api_key = _get_provider_key(provider)
        if api_key:
            return provider, model, api_key, None
    return None, None, None, "No API key detected. Configure ANTHROPIC_API_KEY or OPENAI_API_KEY."


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram_screenshot(photo_bytes: bytes, caption: str = "") -> tuple[bool, str]:
    if not photo_bytes:
        return False, "No image."
    if not is_configured():
        return False, "Telegram not configured."
    try:
        send_photo(photo_bytes, caption=caption)
        return True, "Sent to Telegram."
    except Exception as exc:
        return False, f"Send failed: {exc}"


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

# Upload section
st.markdown("#### 1) Upload or paste image")
uploaded = st.file_uploader("Upload surface screenshot", type=["png", "jpg", "jpeg", "webp"])

# Clipboard paste support
try:
    from streamlit_paste_button import paste_image_button
    paste_result = paste_image_button(label="Paste screenshot (Ctrl/Cmd+V)", key="vol_surface_paste")
    if paste_result is not None and getattr(paste_result, "image_data", None) is not None:
        buf = io.BytesIO()
        paste_result.image_data.save(buf, format="PNG")
        st.session_state["vol_surface_clipboard_bytes"] = buf.getvalue()
except ImportError:
    st.caption("Install streamlit-paste-button for clipboard paste support.")

# Resolve image source
image_sources: Dict[str, bytes] = {}
if uploaded is not None:
    image_sources["Uploaded file"] = uploaded.getvalue()
clipboard_bytes = st.session_state.get("vol_surface_clipboard_bytes")
if isinstance(clipboard_bytes, (bytes, bytearray)) and len(clipboard_bytes) > 0:
    image_sources["Pasted screenshot"] = bytes(clipboard_bytes)

selected_image_bytes: Optional[bytes] = None
if image_sources:
    source_label = (
        st.radio("Image source", list(image_sources.keys()), horizontal=True)
        if len(image_sources) > 1 else next(iter(image_sources.keys()))
    )
    selected_image_bytes = image_sources[source_label]

# Sidebar
with st.sidebar:
    st.markdown("### Extraction Settings")
    resolved_provider, resolved_model, resolved_api_key, resolved_err = _resolve_advanced_model()
    st.caption(f"Provider: `{resolved_provider or 'unavailable'}`")
    st.caption(f"Model: `{resolved_model or 'unavailable'}`")
    run_extract = st.button("Extract + Build Deck", type="primary", use_container_width=True)
    if is_configured():
        st.markdown("### Telegram")
        telegram_caption = st.text_input("Caption", value="Vol Surface screenshot")
        send_to_telegram = st.button("Send to Telegram", use_container_width=True)
    else:
        send_to_telegram = False

if selected_image_bytes is not None:
    st.image(selected_image_bytes, caption="Selected image", use_container_width=True)

# Manual table input
with st.expander("No-key mode: manual table input", expanded=False):
    st.caption("Use when API keys are unavailable.")
    m1, m2 = st.columns(2)
    with m1:
        manual_base = st.text_input("Base", value=st.session_state.get("vol_surface_manual_base", "BTC"))
    with m2:
        manual_quote = st.text_input("Quote", value=st.session_state.get("vol_surface_manual_quote", "USD"))
    st.session_state["vol_surface_manual_base"] = manual_base
    st.session_state["vol_surface_manual_quote"] = manual_quote

    if "vol_surface_manual_df" not in st.session_state:
        st.session_state["vol_surface_manual_df"] = _empty_surface_df()

    use_deribit_expiries = st.checkbox("Use live Deribit expiries", value=True)
    if use_deribit_expiries:
        live_tenors = fetch_deribit_expiry_tenors(manual_base)
        if live_tenors:
            current_df = st.session_state["vol_surface_manual_df"]
            current_tenors = (
                current_df["tenor"].astype(str).str.upper().str.strip().tolist()
                if isinstance(current_df, pd.DataFrame) and "tenor" in current_df.columns else []
            )
            if current_tenors != live_tenors:
                st.session_state["vol_surface_manual_df"] = _sync_manual_df_to_tenors(current_df, live_tenors)
            st.caption(f"Loaded {len(live_tenors)} expiries for `{manual_base.upper()}`.")

    edited_manual_df = st.data_editor(
        st.session_state["vol_surface_manual_df"],
        use_container_width=True, num_rows="dynamic", key="vol_surface_manual_editor",
    )
    if st.button("Save manual table", use_container_width=True):
        st.session_state["vol_surface_manual_df"] = edited_manual_df.copy()
        st.session_state["vol_surface_analysis_df"] = _normalize_rows(edited_manual_df.to_dict(orient="records"))
        st.session_state["vol_surface_analysis_pair"] = f"{manual_base.upper()}/{manual_quote.upper()}"
        st.success("Manual table saved.")

# Telegram send
if send_to_telegram and selected_image_bytes:
    ok, msg = send_telegram_screenshot(selected_image_bytes, caption=telegram_caption.strip())
    if ok:
        st.success(msg)
    else:
        st.error(msg)

# Extract action
if run_extract:
    if selected_image_bytes is None:
        st.warning("Upload or paste an image first.")
        st.stop()
    if not resolved_provider or not resolved_api_key:
        st.warning(resolved_err or "No API key. Use manual table input.")
        st.stop()
    with st.spinner("Extracting table..."):
        try:
            extract_payload = extract_surface_rows(
                selected_image_bytes, resolved_provider, resolved_model, resolved_api_key
            )
            raw_rows = extract_payload.get("rows", [])
            parsed_pair = extract_payload.get("pair")
            parsed_df = _normalize_rows(raw_rows)
        except Exception as exc:
            st.error(f"Extraction failed: {exc}")
            st.stop()
    if parsed_df.empty:
        st.error("No rows parsed. Try a clearer screenshot.")
        st.stop()
    parsed_df = parsed_df.sort_values(by="tenor", key=lambda s: s.map(_sortable_tenor_key)).reset_index(drop=True)
    st.session_state["vol_surface_analysis_df"] = parsed_df
    st.session_state["vol_surface_analysis_pair"] = str(parsed_pair).strip() if parsed_pair else None
    st.success(f"Parsed {len(parsed_df)} rows.")

# ---------------------------------------------------------------------------
# Analysis display
# ---------------------------------------------------------------------------

analysis_df = st.session_state.get("vol_surface_analysis_df")
if isinstance(analysis_df, pd.DataFrame) and not analysis_df.empty:
    st.markdown("#### 2) Extracted table")
    pair_label = st.session_state.get("vol_surface_analysis_pair")
    if pair_label:
        st.caption(f"Pair: `{pair_label}`")
    st.dataframe(analysis_df, use_container_width=True, hide_index=True)

    st.markdown("#### 3) Forward-vol analyses")
    st.caption("Vols: calibration where available, else model params. Stat model = quant baseline.")

    marked_df = _build_marked_term_df(analysis_df)
    quant_df = _build_term_df(analysis_df, source="stat")

    if marked_df.empty or marked_df["atm"].notna().sum() < 2:
        st.warning("Not enough data for forward-vol analyses.")
        st.stop()

    # Summary metrics
    last = marked_df.iloc[-1]
    first = marked_df.iloc[0]
    slope = float(last["atm"]) - float(first["atm"]) if pd.notna(first["atm"]) and pd.notna(last["atm"]) else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", str(len(marked_df)))
    c2.metric("ATM Slope", f"{slope:+.2f}" if slope is not None else "-")
    c3.metric("Avg 25RR", f"{marked_df['rr'].dropna().mean():.2f}" if marked_df["rr"].notna().any() else "-")
    c4.metric("Avg 25BF", f"{marked_df['bf'].dropna().mean():.2f}" if marked_df["bf"].notna().any() else "-")

    # Forward vols chart
    fig_fwd = _build_combined_forward_vols_chart(marked_df, quant_df=quant_df)
    if fig_fwd:
        st.plotly_chart(fig_fwd, use_container_width=True)

    # Term structure
    t1, t2, t3 = st.columns(3)
    with t1:
        fig = _build_term_structure_chart(marked_df, "atm", "ATM Vol", quant_df, "#4e79a7")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with t2:
        fig = _build_term_structure_chart(marked_df, "rr", "25RR", quant_df, "#59a14f")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with t3:
        fig = _build_term_structure_chart(marked_df, "bf", "25BF", quant_df, "#e15759")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Forward vol matrix
    fig_matrix = _build_forward_vol_matrix(marked_df, "atm", "ATM")
    if fig_matrix:
        st.plotly_chart(fig_matrix, use_container_width=True)

    m1, m2 = st.columns(2)
    with m1:
        fig = _build_forward_vol_matrix(marked_df, "call25", "25d Call")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with m2:
        fig = _build_forward_vol_matrix(marked_df, "put25", "25d Put")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Download
    st.download_button(
        "Download parsed table as CSV",
        data=analysis_df.to_csv(index=False).encode("utf-8"),
        file_name="vol_surface_parsed.csv",
        mime="text/csv",
    )
else:
    st.info("Upload an image and click Extract, or use manual table input.")
