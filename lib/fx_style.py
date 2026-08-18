"""
MCM Analytics — shared chart styling and table rendering.

Mirrors the FalconX bot's visual language so charts blasted to Telegram look
identical: white plot background, navy/blue/red/green palette, banner-topped
table images, and the branded watermark.
"""

from __future__ import annotations

import io
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Core palette (verbatim from the FalconX page).
NAVY = "#1E4D7A"
BLUE = "#3790C7"
RED = "#C62828"
GREEN = "#2E7D32"
AMBER = "#EF8A00"
ORANGE = "#E65100"
PURPLE = "#7B1FA2"
GREY = "#8a8a8a"
HEADER_NAVY = "#13314F"
RULE_BLUE = "#2E6CB5"

TEMPLATE = "plotly_white"

# All timestamps are displayed in this zone.
DISPLAY_TZ = "Asia/Singapore"
DISPLAY_TZ_LABEL = "Singapore"
XAXIS_EST = f"Date ({DISPLAY_TZ_LABEL})"      # kept under the old name for callers
XAXIS_TIME = XAXIS_EST
TIME_TICKFORMAT = "%d-%b %H:%M"                # 24h, matching SGT convention
EXPIRY_LABEL_FONT_SIZE = 12

# Snapshot styling for term-structure charts.
TERM_STRUCTURE_SERIES_STYLE = {
    "Current":     {"color": "#1F3A93", "dash": "solid",  "marker_symbol": "circle",      "width": 2.4},
    "24h Ago":     {"color": "#C2185B", "dash": "dash",   "marker_symbol": "diamond",     "width": 1.8},
    "1 Week Ago":  {"color": "#00897B", "dash": "dot",    "marker_symbol": "square",      "width": 1.8},
    "1 Month Ago": {"color": "#F57C00", "dash": "dashdot", "marker_symbol": "triangle-up", "width": 1.8},
}

# Smile snapshot styling.
SMILE_SNAPSHOT_STYLE = [
    ("Current", NAVY, "solid"),
    ("Yesterday", GREEN, "dot"),
    ("Week Ago", PURPLE, "dot"),
    ("1 Month Ago", ORANGE, "dot"),
]

SIGNED_COLS = {"ATM 3h", "ATM Open", "IV−RV", "4hr Chg", "24hr Chg"}
_COL_WIDTHS = {"Token": 0.60, "Expiry": 1.12, "DTE": 0.50,
               "Daily Move %": 1.28, "BE Move %": 1.12}


def to_local(index) -> pd.Index:
    """Convert a UTC datetime index to the display zone, tz-naive for plotting."""
    try:
        idx = pd.DatetimeIndex(index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        return idx.tz_convert(DISPLAY_TZ).tz_localize(None)
    except Exception:
        return index


def local_now() -> "pd.Timestamp":
    """Current wall-clock time in the display zone."""
    return pd.Timestamp.now(tz="UTC").tz_convert(DISPLAY_TZ)


def to_local_ts(ts):
    """Convert a single timestamp to the display zone (tz-naive)."""
    try:
        t = pd.Timestamp(ts)
        if t.tz is None:
            t = t.tz_localize("UTC")
        return t.tz_convert(DISPLAY_TZ).tz_localize(None)
    except Exception:
        return pd.Timestamp(ts)


# Legacy alias — the codebase used US/Eastern before the move to Singapore.
to_est = to_local


def base_layout(**overrides) -> dict:
    layout = dict(template=TEMPLATE, height=450)
    layout.update(overrides)
    return layout


def add_watermark(fig: go.Figure, text: str = "MCM Analytics") -> None:
    """
    Faint branding mark, pinned to the top-right of the header block.

    It used to sit below the plot, where rotated date ticks pushed it into the
    axis labels; the top-right corner is always free because legends are
    left-anchored.
    """
    try:
        fig.add_annotation(
            x=1.0, y=1.0, xref="paper", yref="paper",
            xanchor="right", yanchor="bottom", text=text, showarrow=False,
            font=dict(size=10, color="rgba(138,138,138,0.65)"),
        )
    except Exception:
        pass


def _title_text(fig) -> str:
    """Read a figure's current title text across plotly/stub layout shapes."""
    try:
        layout = fig.layout
        title = layout.get("title") if isinstance(layout, dict) else getattr(layout, "title", None)
        if title is None:
            return ""
        if isinstance(title, str):
            return title
        if isinstance(title, dict):
            return title.get("text") or ""
        return getattr(title, "text", "") or ""
    except Exception:
        return ""


def finalize(fig, note: str | None = None, legend_rows: int = 1,
             extra_top: int = 0, keep_margin: bool = False):
    """
    Lay out the header block so the title, note and legend never collide.

    The note used to be a free-floating paper annotation just above the plot,
    which is exactly where a horizontal legend grows when it wraps to a second
    row — so on the five-series charts the two drew on top of each other.  It
    is now a second line of the title, and the top margin is sized from the
    number of legend rows the chart can actually produce.
    """
    if fig is None:
        return fig
    try:
        if note:
            base = _title_text(fig)
            marker = "<span style='font-size:11px"
            if marker not in base:
                fig.update_layout(title=dict(
                    text=f"{base}<br><span style='font-size:11px;color:#8a8a8a'>{note}</span>",
                    x=0.0, xanchor="left", y=0.985, yanchor="top"))
        rows = max(1, int(legend_rows))
        if not keep_margin:
            top = 52 + 24 * rows + (22 if note else 0) + int(extra_top)
            fig.update_layout(margin=dict(t=top))
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0,
            font=dict(size=11)))
        add_watermark(fig)
    except Exception:
        pass
    return fig


def add_table_banner(fig: go.Figure, header_text: str, plot_h: float) -> None:
    """
    Header banner above a table image.

    ``header_text`` is split on runs of 6+ spaces: the first group becomes the
    large navy lead line, the remainder a smaller grey sub-line.
    """
    if not header_text:
        return
    import re

    def _fmt(s: str) -> str:
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return s.strip()

    groups = [g for g in re.split(r"\s{6,}", header_text) if g.strip()]
    if not groups:
        return
    lead = _fmt(groups[0])
    sub = _fmt("   ".join(groups[1:])) if len(groups) > 1 else ""

    def _y(px_above_top: float) -> float:
        return 1.0 + (px_above_top / max(plot_h, 1.0))

    fig.add_shape(type="rect", xref="paper", yref="paper",
                  x0=0, x1=1, y0=_y(18), y1=_y(96),
                  fillcolor="rgba(19,49,79,0.05)", line=dict(width=0), layer="below")
    fig.add_shape(type="line", xref="paper", yref="paper",
                  x0=0, x1=1, y0=_y(18), y1=_y(18),
                  line=dict(color=RULE_BLUE, width=2))
    fig.add_annotation(x=0.5, y=_y(66), xref="paper", yref="paper",
                       xanchor="center", yanchor="middle", text=lead,
                       showarrow=False, font=dict(size=18, color=HEADER_NAVY))
    if sub:
        fig.add_annotation(x=0.5, y=_y(36), xref="paper", yref="paper",
                           xanchor="center", yanchor="middle", text=sub,
                           showarrow=False, font=dict(size=13, color="#5A6675"))


def fig_to_png(fig: go.Figure, width: int = 1200, height: int = 800) -> bytes | None:
    """Render a figure to PNG bytes via kaleido.  None when unavailable."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        return None


def dataframe_to_table_image(df: pd.DataFrame, header_text: str = "",
                             width: int = 1180,
                             max_height: int = 1400) -> bytes | None:
    """
    Render a DataFrame as a banner-topped table PNG in the FalconX house style.

    Zebra striping, navy header, en-dash for blanks, red/green tint on the
    signed columns.
    """
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    n_rows = len(df)

    header_h, row_h = 42, 31
    top_margin = 108 if header_text else 46
    bottom = 46
    height = min(max_height, top_margin + header_h + row_h * n_rows + bottom)

    def _cell_text(col, v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "–"
        if isinstance(v, str) and not v.strip():
            return "–"
        return str(v)

    values, fills, fonts = [], [], []
    for col in cols:
        col_vals, col_fill, col_font = [], [], []
        for i, v in enumerate(df[col].tolist()):
            txt = _cell_text(col, v)
            zebra = "#FFFFFF" if i % 2 == 0 else "#F4F7FB"
            colour = "#1B1B1B"
            if txt == "–":
                colour = "#9AA5B1"
            elif col in SIGNED_COLS:
                try:
                    num = float(str(v).replace("%", "").replace("$", "").replace(",", ""))
                    if num > 0:
                        colour, zebra = "#0E7A38", "rgba(14,122,56,0.10)"
                    elif num < 0:
                        colour, zebra = "#C0392B", "rgba(192,57,43,0.10)"
                except (TypeError, ValueError):
                    pass
            col_vals.append(txt)
            col_fill.append(zebra)
            col_font.append(colour)
        values.append(col_vals)
        fills.append(col_fill)
        fonts.append(col_font)

    widths = [_COL_WIDTHS.get(c, 0.88) for c in cols]

    fig = go.Figure(data=[go.Table(
        columnwidth=widths,
        header=dict(values=[f"<b>{c}</b>" for c in cols],
                    fill_color=HEADER_NAVY,
                    font=dict(color="white", size=13),
                    align="center", height=header_h,
                    line=dict(color="#E7ECF3", width=1)),
        cells=dict(values=values,
                   fill_color=fills,
                   font=dict(color=fonts, size=12),
                   align="center", height=row_h,
                   line=dict(color="#E7ECF3", width=1)),
    )])
    fig.update_layout(
        width=width, height=height,
        margin=dict(l=10, r=10, t=top_margin, b=bottom),
        paper_bgcolor="white", template=TEMPLATE,
    )
    if header_text:
        add_table_banner(fig, header_text, plot_h=height - top_margin - bottom)
    add_watermark(fig)
    return fig_to_png(fig, width=width, height=height)


def style_dataframe(df: pd.DataFrame):
    """Streamlit-facing styling: zebra rows, signed colouring, bold Total row."""
    if df is None or df.empty:
        return df

    def _row_style(row):
        base = "background-color: rgba(0,0,0,0.04);" if row.name % 2 else ""
        if str(row.iloc[0]).strip().lower() == "total":
            base += "font-weight: 700;"
        return [base] * len(row)

    def _signed(v):
        try:
            num = float(str(v).replace("%", "").replace("$", "").replace(",", "").replace("+", ""))
        except (TypeError, ValueError):
            return ""
        if num > 0:
            return "color: #0a0;"
        if num < 0:
            return "color: #c00;"
        return ""

    styler = df.style.apply(_row_style, axis=1)
    for col in df.columns:
        if col in SIGNED_COLS or col in ("Delta", "Vega", "Gamma (1%)",
                                         "Net Puts", "Net Calls"):
            styler = styler.map(_signed, subset=[col])
    return styler


# ---------------------------------------------------------------------------
# Light / dark theming
# ---------------------------------------------------------------------------

_DARK = {
    "template": "plotly_dark",
    "paper": "rgba(0,0,0,0)",
    "plot": "rgba(0,0,0,0)",
    "font": "#E6EAF1",
    "grid": "rgba(230,234,241,0.12)",
    "box": "rgba(20,26,38,0.92)",
    "box_font": "#E6EAF1",
}
_LIGHT = {
    "template": "plotly_white",
    "paper": "white",
    "plot": "white",
    "font": "#333333",
    "grid": "rgba(30,77,122,0.09)",
    "box": "rgba(255,255,255,0.86)",
    "box_font": HEADER_NAVY,
}


def active_theme(default: str = "dark") -> str:
    """
    The theme to draw in: the user's explicit choice, else Streamlit's own.

    Falls back to ``default`` when neither is available (e.g. Telegram export).
    """
    try:
        import streamlit as st
        choice = st.session_state.get("mcm_chart_theme", "Auto")
        if choice in ("Light", "Dark"):
            return choice.lower()
        detected = getattr(getattr(st, "context", None), "theme", None)
        kind = getattr(detected, "type", None)
        if kind in ("light", "dark"):
            return kind
    except Exception:
        pass
    return default


def apply_theme(fig, mode: str | None = None):
    """
    Re-skin a finished figure for light or dark.

    Charts are authored light (FalconX house style, which is what reads well in
    Telegram); this repaints them for on-screen dark mode without touching the
    data or the series colours, which stay legible on both backgrounds.
    """
    if fig is None:
        return fig
    palette = _DARK if (mode or active_theme()) == "dark" else _LIGHT
    try:
        fig.update_layout(
            template=palette["template"],
            paper_bgcolor=palette["paper"],
            plot_bgcolor=palette["plot"],
            font=dict(color=palette["font"]),
        )
        layout = getattr(fig, "layout", {})
        for axis in ("xaxis", "yaxis", "xaxis2", "yaxis2"):
            try:
                fig.update_layout(**{axis: dict(gridcolor=palette["grid"])})
            except Exception:
                continue
        # Repaint the stat/info boxes, which are authored on white.
        anns = getattr(layout, "annotations", None) or getattr(fig, "annotations", [])
        for ann in anns:
            try:
                bg = ann["bgcolor"] if isinstance(ann, dict) else getattr(ann, "bgcolor", None)
            except Exception:
                bg = None
            if not bg:
                continue
            try:
                if isinstance(ann, dict):
                    ann["bgcolor"] = palette["box"]
                    if ann.get("font"):
                        ann["font"]["color"] = palette["box_font"]
                else:
                    ann.bgcolor = palette["box"]
                    if getattr(ann, "font", None) is not None:
                        ann.font.color = palette["box_font"]
            except Exception:
                continue
    except Exception:
        pass
    return fig
