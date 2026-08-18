"""
MCM Analytics — volatility commands (Deribit-only ports of the FalconX bot).

Every command returns ``(figure_or_list, dataframe, text)`` so the page can
render them uniformly.  Any of the three may be None.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import sqrt

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib import history, surface

# Import EVERYTHING from fx_style (Colors, TEMPLATE, Styles, Timezone helpers, etc.)
from lib.fx_style import *

_finite = surface.finite


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _pct(v) -> float:
    """Coerce an IV to vol points (percent)."""
    if not _finite(v):
        return float("nan")
    v = float(v)
    return v * 100.0 if abs(v) < 5 else v


def _dec(v) -> float:
    """Coerce an IV to decimal."""
    if not _finite(v):
        return float("nan")
    v = float(v)
    return v / 100.0 if abs(v) >= 5 else v


def _row_to_map(row: pd.Series) -> dict[float, float]:
    return {float(c): float(row[c]) for c in row.index if _finite(row.get(c))}


def _fmt_ago(delta: timedelta) -> str:
    """Human label for an elapsed time, e.g. '6d Ago', '18h Ago'."""
    hours = delta.total_seconds() / 3600.0
    if hours < 1:
        return "<1h Ago"
    if hours < 36:
        return f"{round(hours)}h Ago"
    days = hours / 24.0
    if days < 45:
        return f"{round(days)}d Ago"
    return f"{round(days / 30.0)}mo Ago"


def _lookback_points(frame: pd.DataFrame, now: datetime,
                     requests: list[tuple[str, timedelta]],
                     min_gap_hours: float = 6.0):
    """
    Resolve fixed lookback requests (e.g. 24h/1w/1m ago) against ``frame``,
    relabeling honestly when the history doesn't reach back that far and
    dropping requests that would otherwise re-plot the same underlying row
    under a second, wrong label.

    ``requests`` must be ordered smallest delta first. Returns a list of
    ``(display_label, style_key, ts, row)`` — ``display_label`` is the
    original nice label when the nearest row is reasonably close to the
    requested horizon, otherwise the actual elapsed time (e.g. "3d Ago"
    standing in for "1 Week Ago"). ``style_key`` is always the original
    nominal label, so callers can still key a fixed color/dash per bucket
    regardless of how the point ended up labeled. ``ts`` is the row's actual
    timestamp, for callers that need to pair it with a second series (e.g.
    matching a put frame to the same real moment as a resolved call frame).
    """
    out: list[tuple[str, str, pd.Timestamp, pd.Series]] = []
    seen_ts: list[pd.Timestamp] = []
    if frame is None or frame.empty:
        return out
    for label, delta in requests:
        row, ts = history.surface_row_near(frame, now - delta)
        if row is None or ts is None:
            continue
        if any(abs((ts - s).total_seconds()) < min_gap_hours * 3600 for s in seen_ts):
            continue  # same underlying snapshot as an already-accepted, closer request
        gap = abs((pd.Timestamp(ts) - pd.Timestamp(now - delta)).total_seconds())
        tolerance = delta.total_seconds() * 0.25 + 6 * 3600
        use_label = label if gap <= tolerance else _fmt_ago(now - ts)
        out.append((use_label, label, pd.Timestamp(ts), row))
        seen_ts.append(pd.Timestamp(ts))
    return out


# ===========================================================================
# 1. Vol term structure
# ===========================================================================

def cmd_vol_term_structure(asset: str, **kwargs):
    """ATM IV across listed expiries, with 24h / 1w / 1m lagged snapshots."""
    now = _now()
    live = surface.iv_by_dte_for_delta(asset, "delta50")
    live = {float(k): float(v) for k, v in live.items() if _finite(v)}
    if len(live) < 2:
        return None, None, "No vol surface data available."

    dtes = surface.listed_expiries(asset)
    if not dtes:
        dtes = sorted(int(d) for d in live)

    frame, estimated, src = history.surface_history(asset, "delta50", days=35)

    fig = go.Figure()
    x_cat = [(now.date() + timedelta(days=int(d))).strftime("%d-%b") for d in dtes]

    series: list[tuple[str, str, dict[float, float]]] = [("Current", "Current", live)]
    for label, style_key, _ts, row in _lookback_points(frame, now, [
        ("24h Ago", timedelta(hours=24)),
        ("1 Week Ago", timedelta(days=7)),
        ("1 Month Ago", timedelta(days=30)),
    ]):
        series.append((label, style_key, _row_to_map(row)))

    for label, style_key, vals in series:
        style = TERM_STRUCTURE_SERIES_STYLE.get(style_key, TERM_STRUCTURE_SERIES_STYLE["Current"])
        y = [_pct(surface.interp_at_dte(vals, float(d))) for d in dtes]
        is_current = label == "Current"
        name = label if (is_current or not estimated) else f"{label} (est.)"
        fig.add_trace(go.Scatter(
            x=x_cat, y=y, name=name,
            mode="lines+markers+text" if is_current else "lines+markers",
            text=surface.tidy_labels(x_cat) if is_current else [""] * len(x_cat),
            textposition="top center", textfont=dict(size=12),
            line=dict(color=style["color"], width=style["width"],
                      dash=style["dash"], shape="spline"),
            marker=dict(symbol=style["marker_symbol"], size=8 if is_current else 7),
        ))

    fig.update_layout(
        title=f"{asset} Vol Term Structure",
        xaxis_title="Expiry (Deribit)", yaxis_title="Implied Volatility (IV)",
        height=450, template=TEMPLATE, xaxis=dict(type="category"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    note = (f"Lagged snapshots reconstructed from {src} — Deribit serves no "
            "option-IV history." if (estimated and len(series) > 1) else None)
    finalize(fig, note=note, legend_rows=1)
    return fig, None, None


# ===========================================================================
# 2. Skew term structure
# ===========================================================================

def cmd_skew_term_structure(asset: str, **kwargs):
    """25d risk reversal (call IV - put IV) across listed expiries."""
    now = _now()
    vols = surface.option_vols_by_dte(asset)
    call = {float(k): float(v) for k, v in vols.get("call25", {}).items() if _finite(v)}
    put = {float(k): float(v) for k, v in vols.get("put25", {}).items() if _finite(v)}
    if len(call) < 2 or len(put) < 2:
        return None, None, "No skew data available."

    dtes = surface.listed_expiries(asset) or sorted(int(d) for d in call)
    x_cat = [(now.date() + timedelta(days=int(d))).strftime("%d-%b") for d in dtes]

    fc, estimated, src = history.surface_history(asset, "deltaCall25", days=35)
    fp, _, _ = history.surface_history(asset, "deltaPut25", days=35)

    fig = go.Figure()
    entries: list[tuple[str, str, dict[float, float], dict[float, float]]] = [
        ("Current", "Current", call, put)]
    if fc is not None and fp is not None and not fc.empty and not fp.empty:
        for label, style_key, ts, rc in _lookback_points(fc, now, [
            ("24h Ago", timedelta(hours=24)),
            ("1 Week Ago", timedelta(days=7)),
            ("1 Month Ago", timedelta(days=30)),
        ]):
            # Pair the put frame to the SAME actual moment as the resolved
            # call row, not a fresh (and possibly differently-aged) lookup.
            rp, _ = history.surface_row_near(fp, ts)
            if rp is not None:
                entries.append((label, style_key, _row_to_map(rc), _row_to_map(rp)))

    for label, style_key, cmap, pmap in entries:
        style = TERM_STRUCTURE_SERIES_STYLE.get(style_key, TERM_STRUCTURE_SERIES_STYLE["Current"])
        y = []
        for d in dtes:
            c = surface.interp_at_dte(cmap, float(d))
            p = surface.interp_at_dte(pmap, float(d))
            y.append((_dec(c) - _dec(p)) * 100.0 if (_finite(c) and _finite(p))
                     else float("nan"))
        is_current = label == "Current"
        name = label if (is_current or not estimated) else f"{label} (est.)"
        fig.add_trace(go.Scatter(
            x=x_cat, y=y, name=name,
            mode="lines+markers+text" if is_current else "lines+markers",
            text=surface.tidy_labels(x_cat) if is_current else [""] * len(x_cat),
            textposition="top center", textfont=dict(size=12),
            line=dict(color=style["color"], width=style["width"],
                      dash=style["dash"], shape="spline"),
            marker=dict(symbol=style["marker_symbol"], size=8 if is_current else 7),
        ))

    fig.update_layout(
        title=f"{asset} Skew Term Structure (25Δ Call - Put)",
        xaxis_title="Expiry (Deribit)", yaxis_title="25 Delta Skew (Call - Put)",
        height=450, template=TEMPLATE, xaxis=dict(type="category"),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="rgba(0,0,0,0.3)"),
    )
    finalize(fig, legend_rows=1)
    return fig, None, None


# ===========================================================================
# 3. Vol surface (smile) figure — shared by vol_run and vol_smile
# ===========================================================================

def build_vol_surface_figure(asset: str, dte: int) -> go.Figure | None:
    """Smile for one expiry: current + lagged snapshots + high/low cloud."""
    now = _now()
    cur = surface.current_smile(asset, int(dte))
    if len(cur) < 3:
        return None

    fig = go.Figure()
    snaps: dict[str, dict[int, float]] = {"Current": cur}

    # Historical smiles, re-levelled per delta bucket.
    hist_maps: dict[str, pd.DataFrame] = {}
    estimated = True
    src = "none"
    for key in surface.DELTA_KEYS:
        f, est, s = history.surface_history(asset, key, days=35)
        if f is not None and not f.empty:
            hist_maps[key] = f
            estimated, src = est, s

    cloud: dict[int, tuple[float, float]] = {}
    cloud_days = 0
    snap_style_key: dict[str, str] = {}
    if hist_maps:
        ref_frame = hist_maps["delta50"] if "delta50" in hist_maps else next(iter(hist_maps.values()))
        seen_ts: list[pd.Timestamp] = []
        for label, delta in (("Yesterday", timedelta(hours=24)),
                             ("Week Ago", timedelta(days=7)),
                             ("1 Month Ago", timedelta(days=30))):
            _, ref_ts = history.surface_row_near(ref_frame, now - delta)
            if ref_ts is None:
                continue
            if any(abs((ref_ts - s).total_seconds()) < 6 * 3600 for s in seen_ts):
                continue  # same underlying snapshot as an already-accepted, closer label
            gap = abs((pd.Timestamp(ref_ts) - pd.Timestamp(now - delta)).total_seconds())
            tolerance = delta.total_seconds() * 0.25 + 6 * 3600
            disp_label = label if gap <= tolerance else _fmt_ago(now - ref_ts)

            pts: dict[int, float] = {}
            for key, frame in hist_maps.items():
                row, _ts = history.surface_row_near(frame, now - delta)
                if row is None:
                    continue
                v = surface.interp_at_dte(_row_to_map(row), float(dte))
                if _finite(v) and v > 0:
                    pts[surface.DELTA_X_MAP[key]] = _pct(v)
            if len(pts) >= 3:
                snaps[disp_label] = pts
                snap_style_key[disp_label] = label
                seen_ts.append(pd.Timestamp(ref_ts))
        # High/low band across the whole window.
        for key, frame in hist_maps.items():
            x = surface.DELTA_X_MAP[key]
            vals = []
            for _, row in frame.iterrows():
                v = surface.interp_at_dte(_row_to_map(row), float(dte))
                if _finite(v) and v > 0:
                    vals.append(_pct(v))
            if len(vals) >= 2:
                cloud[x] = (float(np.min(vals)), float(np.max(vals)))
            any_frame = frame
        try:
            cloud_days = int((any_frame.index[-1] - any_frame.index[0]).days)
        except Exception:
            cloud_days = 35

    if len(cloud) >= 3:
        cxs = sorted(cloud)
        c_lo = [cloud[x][0] for x in cxs]
        c_hi = [cloud[x][1] for x in cxs]
        fig.add_trace(go.Scatter(x=cxs, y=c_lo, mode="lines",
                                 line=dict(width=0, shape="spline"),
                                 hoverinfo="skip", showlegend=False, name="_cloud_lo"))
        fig.add_trace(go.Scatter(x=cxs, y=c_hi, mode="lines",
                                 line=dict(width=0, shape="spline"),
                                 fill="tonexty", fillcolor="rgba(46,108,181,0.13)",
                                 hoverinfo="skip", showlegend=True,
                                 name=f"High–Low ({cloud_days}d)"))

    for label, pts in snaps.items():
        if not pts or len(pts) < 3:
            continue
        style_key = snap_style_key.get(label, label)
        style = SMILE_SNAPSHOT_STYLE.get(style_key, SMILE_SNAPSHOT_STYLE["Current"])
        xs = sorted(pts)
        name = label if label == "Current" or not estimated else f"{label} (est.)"
        fig.add_trace(go.Scatter(
            x=xs, y=[pts[x] for x in xs], name=name,
            line=dict(color=style["color"], dash=style["dash"],
                      width=style["width"], shape="spline"),
            mode="lines+markers",
        ))

    # Key-delta labels on the current line.
    keymap = [(75, "25dP", "top center"), (50, "ATM", "bottom center"),
              (25, "25dC", "top center")]
    kx, ky, kt, kpos = [], [], [], []
    for x, lbl, pos in keymap:
        if x in cur:
            kx.append(x); ky.append(cur[x]); kt.append(f"{lbl} {cur[x]:.1f}"); kpos.append(pos)
    if kx:
        fig.add_trace(go.Scatter(
            x=kx, y=ky, mode="markers+text", text=kt, textposition=kpos,
            textfont=dict(size=11, color=NAVY),
            marker=dict(size=9, color=NAVY, line=dict(color="white", width=1)),
            showlegend=False, hoverinfo="skip", cliponaxis=False))

    # Stats box: ATM, 25d RR, 10d BF, skew direction, ATM range.
    atm = cur.get(50)
    c25, p25 = cur.get(25), cur.get(75)
    c10, p10 = cur.get(10), cur.get(90)
    lines = []
    if _finite(atm):
        lines.append(f"ATM IV: <b>{atm:.1f}%</b>")
    rr = (c25 - p25) if (_finite(c25) and _finite(p25)) else None
    if rr is not None:
        delta_txt = ""
        style_key_to_label = {v: k for k, v in snap_style_key.items()}
        for style_key, tag in (("Week Ago", "1w"), ("Yesterday", "1d"), ("1 Month Ago", "1m")):
            prior = snaps.get(style_key_to_label.get(style_key, style_key))
            if prior and 25 in prior and 75 in prior:
                delta_txt = f" ({tag} {rr - (prior[25] - prior[75]):+.1f})"
                break
        lines.append(f"25d RR: <b>{rr:+.1f}</b>{delta_txt}")
    if _finite(c10) and _finite(p10) and _finite(atm):
        lines.append(f"10d BF: <b>{((c10 + p10) / 2.0 - atm):+.1f}</b>")
    if rr is not None:
        direction = "put-skewed" if rr < -0.05 else ("call-skewed" if rr > 0.05 else "symmetric")
        lines.append(f"Skew: {direction}")
    if 50 in cloud:
        lines.append(f"ATM range: <b>{cloud[50][1] - cloud[50][0]:.1f}</b> ({cloud_days}d)")
    if lines:
        fig.add_annotation(
            x=0.985, y=0.98, xref="paper", yref="paper", xanchor="right",
            yanchor="top", text="<br>".join(lines), showarrow=False, align="left",
            font=dict(size=12, color="#13314F"),
            bgcolor="rgba(255,255,255,0.86)", bordercolor="#2E6CB5",
            borderwidth=1, borderpad=7)

    expiry_str = (now.date() + timedelta(days=int(dte))).strftime("%d-%b-%Y")
    fig.update_layout(
        title=f"{asset} {expiry_str} Vol Surface",
        xaxis_title="Call Delta", yaxis_title="Implied Volatility (IV)",
        xaxis=dict(autorange="reversed"), height=450, template=TEMPLATE,
    )
    finalize(fig, legend_rows=2)
    return fig


def cmd_vol_smile(asset: str, dte: int | None = None, **kwargs):
    """Volatility smile for the listed expiry nearest the requested tenor."""
    target = int(dte) if dte is not None else 30
    resolved = surface.resolve_dte(asset, target)
    fig = build_vol_surface_figure(asset, resolved)
    if fig is None:
        return None, None, "No vol smile data."
    return fig, None, None


# ===========================================================================
# 4. Vol run
# ===========================================================================

def cmd_vol_run(asset: str, **kwargs):
    """The headline table: ATM, moves, RV, wings and forward IV per expiry."""
    now = _now()
    cfg = surface.acfg(asset)
    vols = surface.option_vols_by_dte(asset)
    atm_map = {float(k): float(v) for k, v in vols.get("atm", {}).items() if _finite(v)}
    if len(atm_map) < 1:
        return None, None, "No vol data available."

    call_map = {float(k): float(v) for k, v in vols.get("call25", {}).items() if _finite(v)}
    put_map = {float(k): float(v) for k, v in vols.get("put25", {}).items() if _finite(v)}
    dtes = surface.listed_expiries(asset) or sorted(int(d) for d in atm_map)

    # Historical ATM for the 3h / session-open change columns.
    frame, estimated, src = history.surface_history(asset, "delta50", days=3)
    row_3h = history.surface_row_at(frame, now - timedelta(hours=3)) if frame is not None else None
    open_ts = now.replace(hour=0, minute=0, second=0, microsecond=0)
    row_open = history.surface_row_at(frame, open_ts) if frame is not None else None
    map_3h = _row_to_map(row_3h) if row_3h is not None else {}
    map_open = _row_to_map(row_open) if row_open is not None else {}

    # Parkinson RV by tenor from Deribit perp candles.
    daily = history.perp_ohlc(asset, days=400, resolution="1D")
    var_hl = None
    if daily is not None and not daily.empty and "high" in daily and "low" in daily:
        ratio = daily["high"] / daily["low"].replace(0, np.nan)
        var_hl = np.log(ratio) ** 2 / (4 * np.log(2))

    def _rv_at_dte(days: int) -> float:
        if var_hl is None or days <= 0:
            return float("nan")
        d = min(int(days), len(var_hl))
        if d < 2:
            return float("nan")
        last = var_hl.rolling(d, min_periods=d).mean().iloc[-1]
        if not _finite(last) or last <= 0:
            return float("nan")
        return float(np.sqrt(365 * last) * 100)

    rows = []
    today = now.date()
    for i, dte in enumerate(dtes):
        iv_dec = surface.interp_at_dte(atm_map, float(dte))
        if not _finite(iv_dec) or iv_dec <= 0:
            continue
        iv_pct = iv_dec * 100.0
        daily_move = iv_dec * sqrt(1.0 / 365.0) * 100
        be_move = iv_dec * sqrt(max(dte, 0) / 365.0) * 100

        v3 = surface.interp_at_dte(map_3h, float(dte)) if map_3h else float("nan")
        vo = surface.interp_at_dte(map_open, float(dte)) if map_open else float("nan")
        atm_3h_chg = (iv_dec - _dec(v3)) * 100 if _finite(v3) else float("nan")
        atm_open_chg = (iv_dec - _dec(vo)) * 100 if _finite(vo) else float("nan")

        rv_pct = _rv_at_dte(int(dte))
        iv_minus_rv = iv_pct - rv_pct if _finite(rv_pct) else float("nan")

        c25 = surface.interp_at_dte(call_map, float(dte)) if call_map else float("nan")
        p25 = surface.interp_at_dte(put_map, float(dte)) if put_map else float("nan")

        # Forward IV from this expiry to the next listed one.
        fwd_iv = float("nan")
        if 0 < i < len(dtes) - 1:
            dte_next = dtes[i + 1]
            v_long = surface.interp_at_dte(atm_map, float(dte_next))
            f = surface.forward_vol(iv_dec, dte / 365.0, v_long, dte_next / 365.0)
            fwd_iv = f * 100.0 if _finite(f) else float("nan")

        rows.append({
            "Token": asset,
            "Expiry": (today + timedelta(days=int(dte))).strftime("%d-%b-%Y"),
            "DTE": int(dte),
            "ATM σ%": round(iv_pct, 1),
            "Daily Move %": round(daily_move, 2),
            "BE Move %": round(be_move, 2),
            "ATM 3h": round(atm_3h_chg, 1) if _finite(atm_3h_chg) else np.nan,
            "ATM Open": round(atm_open_chg, 1) if _finite(atm_open_chg) else np.nan,
            "RV%": round(rv_pct, 1) if _finite(rv_pct) else np.nan,
            "IV−RV": round(iv_minus_rv, 1) if _finite(iv_minus_rv) else np.nan,
            "25d σ Call": round(_pct(c25), 1) if _finite(c25) else np.nan,
            "25d σ Put": round(_pct(p25), 1) if _finite(p25) else np.nan,
            "Fwd IV": round(fwd_iv, 1) if _finite(fwd_iv) else np.nan,
        })

    if not rows:
        return None, None, "No vol data available."
    df = pd.DataFrame(rows)

    # Header banner.
    spot = surface.deribit.get_index_price(cfg["index"])
    perp = surface.perp_mark(asset)
    dvol = surface.dvol_now(asset)
    rv = history.latest_rv(asset)
    lead = []
    if spot:
        lead.append(f"**{asset} SPOT** ${spot:,.{cfg['price_dp']}f}")
    if dvol is not None:
        lead.append(f"**DVOL** {dvol:.1f}")
    if spot and perp:
        lead.append(f"**PERP** {((perp - spot) / spot * 100):+.2f}%")
    rv_bits = [f"{lbl} {rv[k]:.1f}" for k, lbl in
               (("24hr", "24h"), ("3d", "3d"), ("7d", "7d"), ("30d", "30d"), ("90d", "90d"))
               if k in rv and _finite(rv[k])]
    groups = ["   |   ".join(lead)] if lead else []
    if rv_bits:
        groups.append("**RV**  " + "  ·  ".join(rv_bits))
    header = "      ".join(groups)

    figs = []
    for dte in sorted({int(d) for d in df["DTE"].tolist()}):
        f = build_vol_surface_figure(asset, dte)
        if f is not None:
            figs.append(f)

    return (figs or None), df, header


# ===========================================================================
# 5. Forward vols
# ===========================================================================

_SERIES_SPECS = [
    ("delta50", "ATM", NAVY),
    ("deltaCall25", "25d Call", GREEN),
    ("deltaPut25", "25d Put", RED),
]


def cmd_forward_vols(asset: str, **kwargs):
    """Combined ATM / 25d call / 25d put spot and forward vol term structure."""
    all_series = []
    for key, label, color in _SERIES_SPECS:
        s = surface.build_forward_vol_term_series(asset, key, label)
        if s:
            all_series.append((s, color))
    if not all_series:
        return None, None, "No vol data available for forward vols."

    preferred = next((s for s, _ in all_series if s["vol_label"] == "ATM"),
                     all_series[0][0])
    x_cat = preferred["x_cat"]

    fig = go.Figure()
    for s, color in all_series:
        is_pref = s is preferred
        fig.add_trace(go.Scatter(
            x=s["x_cat"], y=s["spot_vols"], name=f"{s['vol_label']} Vol",
            mode="lines+markers+text" if is_pref else "lines+markers",
            text=s["expiry_labels"] if is_pref else [""] * len(s["x_cat"]),
            textposition="top center", textfont=dict(size=12),
            line=dict(color=color, width=2.2, shape="spline")))
        fig.add_trace(go.Scatter(
            x=s["x_cat"], y=s["forward_vols"], name=f"{s['vol_label']} Fwd",
            mode="lines+markers", opacity=0.75,
            line=dict(color=color, width=1.8, dash="dash", shape="spline")))

    fig.update_layout(
        title=f"{asset} Combined Forward Vols (ATM / 25d Call / 25d Put)",
        xaxis_title="Expiry (Deribit)", yaxis_title="Implied Volatility (IV)",
        height=500, template=TEMPLATE,
        xaxis=dict(type="category", categoryorder="array", categoryarray=x_cat))
    finalize(fig, legend_rows=2)
    return fig, None, None


# ===========================================================================
# 6. Vega carry waterfall (steepness)
# ===========================================================================

def _coming_friday(ref_date=None):
    if ref_date is None:
        ref_date = _now().date()
    return ref_date + timedelta(days=(4 - ref_date.weekday()) % 7)


def _carry_waterfall(asset: str, delta_key: str, vol_label: str,
                     start_from_coming_friday: bool = True, **kwargs):
    s = surface.build_forward_vol_term_series(asset, delta_key, vol_label)
    if not s:
        return None, None, f"No {vol_label} steepness data available."
    dtes, x_cat, spot = s["dtes"], s["x_cat"], s["spot_vols"]
    expiry_dates = s["expiry_dates"]
    n = min(len(dtes), len(x_cat), len(spot))
    if n < 2:
        return None, None, f"Insufficient {vol_label} tenor points for vega carry."

    cutoff = _coming_friday() if start_from_coming_friday else None
    labels, carry, slopes, weights = [], [], [], []
    for i in range(n - 1):
        d0, d1 = int(dtes[i]), int(dtes[i + 1])
        if d0 <= 3 or d1 <= 3:
            continue
        if cutoff is not None and expiry_dates[i] < cutoff:
            continue
        gap = d1 - d0
        if gap <= 0:
            continue
        slope = (float(spot[i + 1]) - float(spot[i])) / float(gap)   # vol pts/day
        mid_dte = max(1.0, 0.5 * (d0 + d1))
        w30 = float(np.sqrt(30.0 / mid_dte))     # vega ~ sqrt(T); weight to 30D
        labels.append(f"{x_cat[i]}→{x_cat[i + 1]}")
        slopes.append(slope)
        weights.append(w30)
        carry.append(100000.0 * slope * w30)

    if not carry:
        return None, None, f"No {vol_label} tenor segments after excluding <=3DTE."

    total = float(np.nansum(carry))
    fig = go.Figure(go.Waterfall(
        x=labels + ["Total"], y=carry + [total],
        measure=["relative"] * len(carry) + ["total"],
        connector={"line": {"color": "rgba(120,120,120,0.45)", "width": 1}},
        increasing={"marker": {"color": GREEN}},
        decreasing={"marker": {"color": RED}},
        totals={"marker": {"color": NAVY}},
        text=[f"${v:,.0f}" for v in carry] + [f"${total:,.0f}"],
        textposition="outside",
        customdata=np.column_stack([slopes + [np.nan], weights + [np.nan],
                                    carry + [total]]),
        hovertemplate=("<b>%{x}</b><br>" + f"{vol_label} slope: " +
                       "%{customdata[0]:+.3f} vol pts/day<br>"
                       "30D vega weight: %{customdata[1]:.2f}x<br>"
                       "Expected carry: $%{customdata[2]:,.0f} / day<extra></extra>"),
    ))
    note = ("Assumes static curve shape; excludes <=3DTE; carry normalized to "
            "30D vega equivalence.")
    if cutoff is not None:
        note += f" Starts at coming-Friday expiry ({cutoff:%d-%b})."
    fig.update_layout(
        title=f"{asset} {vol_label} Vega Carry Waterfall ($100k Vega, 30D-Weighted, Daily)",
        xaxis_title="Adjacent tenor pair (chronological)",
        yaxis_title="Expected carry ($ / day)",
        height=500, template=TEMPLATE,
        xaxis=dict(type="category", tickangle=-30),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="rgba(0,0,0,0.35)"),
        margin=dict(b=120))
    finalize(fig, note=note, legend_rows=0)
    return fig, None, None


def cmd_forward_vol_steepness(asset, **kw):
    return _carry_waterfall(asset, "delta50", "ATM", **kw)


def cmd_forward_vol_steepness_25d_call(asset, **kw):
    return _carry_waterfall(asset, "deltaCall25", "25d Call", **kw)


def cmd_forward_vol_steepness_25d_put(asset, **kw):
    return _carry_waterfall(asset, "deltaPut25", "25d Put", **kw)


def cmd_forward_vol_steepness_multidelta(asset: str, **kwargs):
    """Carry per 30d, grouped by tenor pair, for ATM / 25d call / 25d put."""
    rows = []
    for key, label, _ in _SERIES_SPECS:
        s = surface.build_forward_vol_term_series(asset, key, label)
        if not s:
            continue
        dtes, x_cat, spot = s["dtes"], s["x_cat"], s["spot_vols"]
        for i in range(min(len(dtes), len(spot)) - 1):
            d0, d1 = int(dtes[i]), int(dtes[i + 1])
            if d0 <= 3 or d1 <= 3:
                continue
            gap = d1 - d0
            if gap <= 0:
                continue
            slope = (float(spot[i + 1]) - float(spot[i])) / float(gap)
            rows.append({"Type": label, "Pair": f"{x_cat[i]}→{x_cat[i + 1]}",
                         "From DTE": d0, "To DTE": d1,
                         "Carry (vol pts / 30d)": slope * 30.0})
    if not rows:
        return None, None, "No multi-delta steepness data available."

    df = pd.DataFrame(rows)
    pair_order = (df[["Pair", "From DTE", "To DTE"]].drop_duplicates()
                  .sort_values(["From DTE", "To DTE"])["Pair"].tolist())
    colors = {"ATM": NAVY, "25d Call": GREEN, "25d Put": RED}
    fig = go.Figure()
    for label in ("ATM", "25d Call", "25d Put"):
        sub = df[df["Type"] == label]
        if sub.empty:
            continue
        ymap = dict(zip(sub["Pair"], sub["Carry (vol pts / 30d)"]))
        fig.add_trace(go.Bar(
            x=pair_order, y=[ymap.get(p, np.nan) for p in pair_order],
            name=label, marker_color=colors[label], opacity=0.9,
            hovertemplate="<b>%{x}</b><br>" + f"{label} carry: " +
                          "%{y:+.2f} vol pts / 30d<extra></extra>"))
    fig.update_layout(
        title=f"{asset} Forward Vol Steepness (Carry / 30d)",
        xaxis_title="Adjacent tenor pair", yaxis_title="Carry (vol pts / 30d)",
        barmode="group", height=460, template=TEMPLATE,
        xaxis=dict(type="category", tickangle=-35),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="rgba(0,0,0,0.35)"),
        margin=dict(b=110))
    finalize(fig, legend_rows=1)
    return fig, None, None


# ===========================================================================
# 7. Forward vol matrix
# ===========================================================================

def _forward_vol_matrix(asset: str, delta_key: str, vol_label: str):
    iv = surface.iv_by_dte_for_delta(asset, delta_key)
    iv = {int(k): float(v) for k, v in iv.items() if _finite(v) and int(k) >= 0}
    if not iv:
        return None, None, "No vol data for forward vol matrix."
    dtes = sorted(iv)
    if len(dtes) < 2:
        return None, None, "Need at least two expiries for forward vol matrix."

    today = _now().date()
    labels = [(today + timedelta(days=d)).strftime("%d%b%y") for d in dtes]
    n = len(dtes)
    Z = np.full((n, n), np.nan)
    for i in range(n):
        v_i = iv[dtes[i]]
        if not _finite(v_i):
            continue
        T_i = dtes[i] / 365.0
        for j in range(i, n):
            if i == j:
                Z[i, j] = v_i * 100.0
                continue
            v_j, T_j = iv[dtes[j]], dtes[j] / 365.0
            if not _finite(v_j) or T_j <= T_i:
                continue
            fwd_var = (v_j ** 2 * T_j - v_i ** 2 * T_i) / (T_j - T_i)
            if fwd_var >= 0:
                Z[i, j] = float(np.sqrt(fwd_var) * 100.0)

    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        z_min, z_max = 0, 100

    fig = go.Figure(go.Heatmap(
        x=labels, y=labels, z=Z,
        text=[[f"{Z[i, j]:.1f}" if np.isfinite(Z[i, j]) else "" for j in range(n)]
              for i in range(n)],
        texttemplate="%{text}", textfont=dict(size=11), showscale=True,
        colorscale=[[0, RED], [0.5, "#FFF9C4"], [1, GREEN]],
        zmin=z_min, zmax=z_max, hoverongaps=False))
    fig.update_layout(
        title=f"{asset} {vol_label} Forward Vol Matrix (%)",
        xaxis_title="Expiry (to)", yaxis_title="Expiry (from)",
        xaxis=dict(type="category", tickangle=-45, side="bottom"),
        yaxis=dict(type="category", autorange="reversed"),
        height=400 + max(0, (n - 6) * 28), template=TEMPLATE,
        margin=dict(b=80, l=100))
    finalize(fig, legend_rows=0, keep_margin=True)
    return fig, None, None


def cmd_forward_vol_matrix(asset, **kw):
    return _forward_vol_matrix(asset, "delta50", "ATM")


# ===========================================================================
# 8. ATM IV box plot
# ===========================================================================

_BOX_TENORS = [("1W", 7), ("2W", 14), ("1M", 30), ("2M", 60), ("3M", 90),
               ("6M", 180), ("9M", 270), ("1Y", 365)]


def _tenor_snap_tol(target: float) -> float:
    return max(7.0, 0.08 * float(target))


def _percentile_color(p: float) -> str:
    if not _finite(p):
        return RED
    if p >= 75.0:
        return RED
    if p <= 25.0:
        return GREEN
    return AMBER


def cmd_atm_iv_box_plot(asset: str, lookback_days: int = 90, **kwargs):
    """ATM IV distribution by tenor over the lookback, with today's curve."""
    frame, estimated, src = history.surface_history(asset, "delta50",
                                                    days=int(lookback_days))
    if frame is None or frame.empty or len(frame) < 8:
        return None, None, "No ATM IV history available for box plot."

    cols = sorted(float(c) for c in frame.columns)
    dte_min, dte_max = cols[0], cols[-1]
    tenors = [(l, d) for l, d in _BOX_TENORS
              if dte_min - _tenor_snap_tol(d) <= d <= dte_max + _tenor_snap_tol(d)]
    if not tenors:
        return None, None, "No standard tenors fall within the available DTE range."

    dist: dict[str, list[float]] = {l: [] for l, _ in tenors}
    for _, row in frame.iterrows():
        vals = _row_to_map(row)
        if not vals:
            continue
        for lbl, d in tenors:
            v = surface.interp_at_dte(vals, float(d), flat_outside=False)
            if _finite(v) and v > 0:
                dist[lbl].append(_pct(v))

    live = surface.iv_by_dte_for_delta(asset, "delta50")
    live = {float(k): float(v) for k, v in live.items() if _finite(v)}
    current = {}
    for lbl, d in tenors:
        v = surface.interp_at_dte(live, float(d)) if live else float("nan")
        if _finite(v) and v > 0:
            current[lbl] = _pct(v)

    items = [(l, d) for l, d in tenors if len(dist[l]) >= 2 and l in current]
    if not items:
        return None, None, "No ATM IV distribution data to plot."

    labels = [l for l, _ in items]
    q1s, meds, q3s, lows, highs, means, stds, iqrs, pcts, curs = ([] for _ in range(10))
    for lbl, _ in items:
        arr = np.asarray(dist[lbl], dtype=float)
        cur = current[lbl]
        q1, med, q3 = (float(np.percentile(arr, p)) for p in (25, 50, 75))
        q1s.append(q1); meds.append(med); q3s.append(q3)
        lows.append(float(np.min(arr))); highs.append(float(np.max(arr)))
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)
        iqrs.append(q3 - q1)
        pcts.append(float(100.0 * np.mean(arr <= cur)))
        curs.append(cur)

    y_lo, y_hi = min(lows + curs), max(highs + curs)
    span = max(1.0, y_hi - y_lo)
    yrange = [y_lo - 0.13 * span, y_hi + 0.13 * span]

    fig = go.Figure()
    for i in range(len(labels)):
        if i % 2:
            fig.add_shape(type="rect", xref="x", yref="paper",
                          x0=i - 0.5, x1=i + 0.5, y0=0, y1=1,
                          fillcolor="rgba(30,77,122,0.035)",
                          line=dict(width=0), layer="below")

    hover = ("<b>%{x}</b><br>max: %{upperfence:.1f}<br>q3: %{q3:.1f}<br>"
             "median: %{median:.1f}<br>mean: %{customdata[3]:.1f}<br>"
             "q1: %{q1:.1f}<br>min: %{lowerfence:.1f}<br>"
             "σ (stdev): %{customdata[0]:.2f}<br>IQR: %{customdata[1]:.1f}<br>"
             "current percentile: %{customdata[2]:.0f}%<extra></extra>")
    fig.add_trace(go.Box(
        x=labels, q1=q1s, median=meds, q3=q3s, lowerfence=lows,
        upperfence=highs, mean=means, sd=stds, boxmean="sd",
        name=f"{lookback_days}D range", marker_color=BLUE,
        line=dict(color=NAVY, width=1.5), fillcolor="rgba(55,144,199,0.24)",
        whiskerwidth=0.5, width=0.55,
        customdata=np.column_stack([stds, iqrs, pcts, means]),
        hovertemplate=hover))

    fig.add_trace(go.Scatter(x=labels, y=highs, mode="text",
                            text=[f"{h:.0f}" for h in highs],
                            textposition="top center",
                            textfont=dict(size=9, color="rgba(90,90,90,0.8)"),
                            showlegend=False, hoverinfo="skip", cliponaxis=False))
    fig.add_trace(go.Scatter(x=labels, y=lows, mode="text",
                            text=[f"{l:.0f}" for l in lows],
                            textposition="bottom center",
                            textfont=dict(size=9, color="rgba(90,90,90,0.8)"),
                            showlegend=False, hoverinfo="skip", cliponaxis=False))

    stem_x, stem_y = [], []
    for i, lbl in enumerate(labels):
        stem_x += [lbl, lbl, None]
        stem_y += [meds[i], curs[i], None]
    fig.add_trace(go.Scatter(x=stem_x, y=stem_y, mode="lines",
                            line=dict(color="rgba(120,120,120,0.5)", width=1.3),
                            showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=labels, y=curs, mode="lines",
                            line=dict(color="rgba(198,40,40,0.16)", width=9,
                                      shape="spline"),
                            showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=labels, y=curs, name="Current ATM IV", mode="lines+markers+text",
        line=dict(color=RED, width=3, shape="spline"),
        marker=dict(color=[_percentile_color(p) for p in pcts], size=10,
                    symbol="diamond", line=dict(color="white", width=1.4)),
        text=[f"{c:.1f}" for c in curs], textposition="middle right",
        textfont=dict(size=10, color=RED),
        hovertemplate="<b>%{x}</b><br>Current ATM IV: %{y:.1f}<extra></extra>"))

    fig.update_layout(
        title=dict(text=f"{asset} ATM IV Box Plot by Tenor ({lookback_days}D)",
                   x=0.5, xanchor="center", y=0.975, yanchor="top",
                   font=dict(size=21, color=NAVY,
                             family="Arial Black, Arial, sans-serif")),
        yaxis_title="ATM Implied Vol (vol pts)", height=560, template=TEMPLATE,
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(type="category", categoryorder="array", categoryarray=labels,
                   title=None, showgrid=False,
                   tickfont=dict(size=13, color=NAVY), ticklen=4),
        yaxis=dict(zeroline=False, range=yrange, showgrid=True,
                   gridcolor="rgba(30,77,122,0.09)", tickfont=dict(size=11),
                   ticksuffix="  "),
        boxgap=0.5, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.015, xanchor="center",
                    x=0.5, font=dict(size=11), bgcolor="rgba(255,255,255,0)"),
        margin=dict(t=175, b=150, l=66, r=46))

    fig.add_annotation(x=0.5, y=1.26, xref="paper", yref="paper", showarrow=False,
                       text=f"<span style='color:#7a869a'>· {lookback_days}-day "
                            f"distribution vs current ·</span>", font=dict(size=12.5))
    fig.add_annotation(
        x=0.5, y=1.17, xref="paper", yref="paper", showarrow=False, font=dict(size=11),
        text=("<b>How to read:</b>   <span style='color:#1E4D7A'>▭</span> box = IQR "
              "(P25–P75)   <span style='color:#1E4D7A'>│</span> whiskers = min–max   "
              "<span style='color:#1E4D7A'>─</span> median   "
              "<span style='color:#C62828'>◆</span> current IV"))
    if estimated:
        fig.add_annotation(x=0.5, y=-0.28, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=10, color=GREY),
                           text=f"Distribution reconstructed from {src}; "
                                "Deribit publishes no historical option IV.")
    for i, lbl in enumerate(labels):
        fig.add_annotation(x=lbl, xref="x", y=-0.045, yref="paper", yanchor="top",
                           showarrow=False, font=dict(size=9, color="#555"),
                           text=f"σ {stds[i]:.1f}<br>IQR {iqrs[i]:.1f}<br>"
                                f"<b>P{pcts[i]:.0f}</b>")
    finalize(fig, legend_rows=1, keep_margin=True)
    return fig, None, None


# ===========================================================================
# 9. Vol / skew time series and intraday variants
# ===========================================================================

def _time_series_frames(asset: str, dte: int, days: int, keys: list[str]):
    out: dict[str, pd.Series] = {}
    estimated, src = True, "none"
    for key in keys:
        s, est, sr = history.iv_series_at_dte(asset, dte, key, days=days)
        if s is not None and len(s) >= 2:
            out[key] = s
            estimated, src = est, sr
    return out, estimated, src


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series] | None:
    """Align two series on a shared index, tolerating small timestamp drift."""
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return None
    idx = a.index.intersection(b.index)
    if len(idx) >= 2:
        return a.reindex(idx), b.reindex(idx)
    try:
        b2 = b.reindex(a.index, method="nearest",
                       tolerance=pd.Timedelta("2h")).dropna()
    except Exception:
        return None
    if len(b2) < 2:
        return None
    return a.reindex(b2.index), b2


def _est_note(estimated: bool, src: str) -> str | None:
    """Text for the reconstruction caveat, or None when the data is recorded."""
    if estimated and src != "none":
        return (f"Reconstructed from {src} — Deribit serves no historical "
                "option IV.")
    return None


def _vol_series_chart(asset: str, dte: int, days: int, title: str,
                      intraday: bool = False):
    keys = ["delta50", "deltaCall25", "deltaPut25"]
    if not intraday:
        keys += ["deltaCall10", "deltaPut10"]
    frames, estimated, src = _time_series_frames(asset, dte, days, keys)
    if "delta50" not in frames:
        return None, None, "No vol time series data."

    spec = [("delta50", "ATM Vol", NAVY, "solid"),
            ("deltaCall25", "25Δ Call Vol", BLUE, "dot"),
            ("deltaPut25", "25Δ Put Vol", "#888888", "dot"),
            ("deltaCall10", "10Δ Call Vol", GREEN, "dash"),
            ("deltaPut10", "10Δ Put Vol", ORANGE, "dash")]

    fig = go.Figure()
    for key, name, color, dash in spec:
        s = frames.get(key)
        if s is None:
            continue
        fig.add_trace(go.Scatter(
            x=to_local(s.index), y=s.values, name=name,
            mode="lines+markers" if intraday else "lines",
            line=dict(color=color, dash=dash, shape="spline")))

    if intraday:
        px = history.perp_ohlc(asset, days=2, resolution="15")
        if px is not None and not px.empty:
            px = px[px.index >= frames["delta50"].index[0]]
            if not px.empty:
                fig.add_trace(go.Scatter(
                    x=to_local(px.index), y=px["close"].values, name="Perp",
                    line=dict(color=ORANGE, width=1.5, dash="dot", shape="spline"),
                    mode="lines", yaxis="y2", connectgaps=False))
                fig.update_layout(yaxis2=dict(overlaying="y", side="right",
                                              title="Perp", showgrid=False,
                                              tickformat=",.0f"))

    fig.update_layout(
        title=title, xaxis_title=XAXIS_TIME,
        yaxis_title="Implied Volatility (IV)" if intraday else "Implied Volatility (IV %)",
        height=450, template=TEMPLATE,
        xaxis=dict(type="date", tickformat=TIME_TICKFORMAT, tickangle=-90))
    if not intraday:
        fig.update_layout(yaxis=dict(ticksuffix="%"))
    n_series = len(fig.data)
    finalize(fig, note=_est_note(estimated, src),
             legend_rows=2 if n_series > 3 else 1)
    return fig, None, None


def cmd_vol_time_series(asset: str, dte: int | None = None, **kwargs):
    resolved = surface.resolve_dte(asset, int(dte) if dte is not None else 30)
    expiry = (_now().date() + timedelta(days=resolved)).strftime("%d-%b-%Y")
    return _vol_series_chart(asset, resolved, 30,
                             f"{asset} Vol Time Series ({expiry})")


def cmd_intraday_vol(asset: str, dte: int | None = None, **kwargs):
    resolved = surface.resolve_dte(asset, int(dte) if dte is not None else 30)
    expiry = (_now().date() + timedelta(days=resolved)).strftime("%d-%b-%Y")
    return _vol_series_chart(asset, resolved, 1, f"{asset} {expiry} Vol",
                             intraday=True)


def _skew_series_chart(asset: str, dte: int, days: int, title: str,
                       intraday: bool = False):
    keys = ["deltaCall25", "deltaPut25"] + ([] if intraday else
                                            ["deltaCall10", "deltaPut10"])
    frames, estimated, src = _time_series_frames(asset, dte, days, keys)
    if "deltaCall25" not in frames or "deltaPut25" not in frames:
        return None, None, "No skew time series data."

    pair = _align(frames["deltaCall25"], frames["deltaPut25"])
    if pair is None:
        return None, None, "No skew time series data."
    s25 = (pair[0] - pair[1]).dropna()
    if len(s25) < 2:
        return None, None, "No skew time series data."

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=to_local(s25.index), y=s25.values, name="25Δ Skew (Call − Put)",
        mode="lines+markers" if intraday else "lines",
        line=dict(color=NAVY, shape="spline")))

    if not intraday and "deltaCall10" in frames and "deltaPut10" in frames:
        pair10 = _align(frames["deltaCall10"], frames["deltaPut10"])
        if pair10 is not None:
            s10 = (pair10[0] - pair10[1]).dropna()
            fig.add_trace(go.Scatter(
                x=to_local(s10.index), y=s10.values, name="10Δ Skew (Call − Put)",
                line=dict(color=BLUE, dash="dot", shape="spline"), yaxis="y2"))
            spread_pair = _align(s10, s25)
            if spread_pair is not None:
                spread = (spread_pair[0] - spread_pair[1]).dropna()
                fig.add_trace(go.Scatter(
                    x=to_local(spread.index), y=spread.values,
                    name="Spread (10Δ − 25Δ)",
                    line=dict(color=GREEN, width=1.5, dash="dash", shape="spline")))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right",
                                          title="10Δ Skew (pp)", showgrid=False))

    if intraday:
        px = history.perp_ohlc(asset, days=2, resolution="15")
        if px is not None and not px.empty:
            px = px[px.index >= s25.index[0]]
            if not px.empty:
                fig.add_trace(go.Scatter(
                    x=to_local(px.index), y=px["close"].values, name="Perp",
                    line=dict(color=ORANGE, width=1.5, dash="dot", shape="spline"),
                    mode="lines", yaxis="y2", connectgaps=False))
                fig.update_layout(yaxis2=dict(overlaying="y", side="right",
                                              title="Perp", showgrid=False,
                                              tickformat=",.0f"))

    fig.update_layout(
        title=title, xaxis_title=XAXIS_TIME, yaxis_title="Skew (pp)",
        height=450, template=TEMPLATE,
        xaxis=dict(type="date", tickformat=TIME_TICKFORMAT, tickangle=-90),
        yaxis=dict(title="25Δ Skew (pp)", side="left"))
    n_series = len(fig.data)
    finalize(fig, note=_est_note(estimated, src),
             legend_rows=2 if n_series > 3 else 1)
    return fig, None, None


def cmd_skew_time_series(asset: str, dte: int | None = None, **kwargs):
    resolved = surface.resolve_dte(asset, int(dte) if dte is not None else 30)
    expiry = (_now().date() + timedelta(days=resolved)).strftime("%d-%b-%Y")
    return _skew_series_chart(asset, resolved, 30,
                              f"{asset} Skew Time Series ({expiry})")


def cmd_intraday_skew(asset: str, dte: int | None = None, **kwargs):
    resolved = surface.resolve_dte(asset, int(dte) if dte is not None else 30)
    expiry = (_now().date() + timedelta(days=resolved)).strftime("%d-%b-%Y")
    return _skew_series_chart(asset, resolved, 1,
                              f"{asset} {expiry} Intraday Skew", intraday=True)