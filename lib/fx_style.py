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
XAXIS_EST = "Date (US Eastern)"
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


def to_est(index) -> pd.Index:
    """Convert a UTC datetime index to US/Eastern, tz-naive for plotting."""
    try:
        idx = pd.DatetimeIndex(index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        return idx.tz_convert("US/Eastern").tz_localize(None)
    except Exception:
        return index


def base_layout(**overrides) -> dict:
    layout = dict(template=TEMPLATE, height=450)
    layout.update(overrides)
    return layout


def add_watermark(fig: go.Figure, text: str = "MCM Analytics") -> None:
    """Faint corner watermark, matching the FalconX branding slot."""
    try:
        fig.add_annotation(
            x=0.995, y=-0.16, xref="paper", yref="paper",
            xanchor="right", yanchor="bottom", text=text, showarrow=False,
            font=dict(size=10, color="rgba(19,49,79,0.35)"),
        )
    except Exception:
        pass


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
