"""
Plotly styling, color palettes, timezone utilities, and table image generation for MCM Analytics.
"""

import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Color Palette & Constants
# ---------------------------------------------------------------------------

NAVY = "#13314F"
BLUE = "#3790C7"
GREEN = "#0E7A38"
RED = "#C0392B"
GREY = "#888888"
ORANGE = "#E65100"
AMBER = "#EF8A00"
PURPLE = "#7B1FA2"

TEMPLATE = "plotly_white"
TIME_TICKFORMAT = "%d-%b %H:%M"
XAXIS_TIME = "Time (SGT)"
DISPLAY_TZ_LABEL = "SGT"

# Table image styling
_TBL_HEADER_FILL = "#13314F"
_TBL_HEADER_FONT = "#FFFFFF"
_TBL_ROW_EVEN = "#FFFFFF"
_TBL_ROW_ODD = "#F4F7FB"
_TBL_GRID = "#E7ECF3"
_TBL_TEXT = "#1F2A37"
_TBL_TEXT_MUTED = "#AEB7C2"
_TBL_POS = "#0E7A38"
_TBL_NEG = "#C0392B"
_TBL_POS_BG = "rgba(14,122,56,0.10)"
_TBL_NEG_BG = "rgba(192,57,43,0.10)"
_TBL_ACCENT = "#2E6CB5"
_TBL_BANNER_BG = "rgba(19,49,79,0.05)"
_TBL_BANNER_SUB = "#5A6675"
_TBL_FONT_FAMILY = "DejaVu Sans, Arial, Helvetica, sans-serif"
_TBL_BLANK = "–"


# ---------------------------------------------------------------------------
# Timezone Helpers
# ---------------------------------------------------------------------------

def local_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def to_local(dt):
    """Convert datetime object to display format or timezone."""
    if dt is None:
        return None
    return dt


def to_local_ts(idx):
    """Convert DatetimeIndex or Series timestamps for chart display."""
    if idx is None or (hasattr(idx, "__len__") and len(idx) == 0):
        return idx
    try:
        if hasattr(idx, "tz"):
            if idx.tz is None:
                return idx.tz_localize("UTC")
            return idx.tz_convert("UTC")
        return idx
    except Exception:
        return idx


# ---------------------------------------------------------------------------
# Chart Theming & Watermarking
# ---------------------------------------------------------------------------

def add_watermark(fig: go.Figure) -> None:
    """Optional watermark hook for charts (clean/no-op)."""
    return


def apply_theme(fig: go.Figure, theme: str = "auto") -> go.Figure:
    """Apply standard clean styling and template to a Plotly figure."""
    if fig is None:
        return None
    
    fig.update_layout(
        template=TEMPLATE,
        font=dict(family=_TBL_FONT_FAMILY, color=_TBL_TEXT),
        paper_bgcolor="white" if theme.lower() == "light" else None,
        plot_bgcolor="white" if theme.lower() == "light" else None,
    )
    return fig


def finalize(fig: go.Figure, height: int = 450, title: str = None) -> go.Figure:
    """Apply final sizing and standard layout adjustments to a figure."""
    if fig is None:
        return None
    
    layout_update = dict(
        height=height,
        template=TEMPLATE,
    )
    if title:
        layout_update["title"] = title
        
    fig.update_layout(**layout_update)
    return fig


def fig_to_png(fig: go.Figure, width: int = 1200, height: int = 800) -> bytes:
    """Convert a Plotly figure into PNG image bytes."""
    if fig is None:
        return None
    try:
        return fig.to_image(format="png", width=width, height=height)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Table Image Generation
# ---------------------------------------------------------------------------

def _add_table_banner(fig: go.Figure, header_text: str, plot_h: float) -> None:
    """Renders the summary banner at the top of table images."""
    if not header_text or plot_h <= 0:
        return
    parts = [p for p in re.split(r"\s{6,}", header_text.strip()) if p]
    if not parts:
        return
    lead = parts[0]
    sub = "     ".join(parts[1:]) if len(parts) > 1 else ""

    def _fmt(s: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)

    def _y(px: float) -> float:
        return 1.0 + px / plot_h

    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, x1=1,
                  y0=_y(4), y1=_y(96), fillcolor=_TBL_BANNER_BG, line=dict(width=0), layer="below")
    fig.add_shape(type="line", xref="paper", yref="paper", x0=0, x1=1,
                  y0=_y(6), y1=_y(6), line=dict(color=_TBL_ACCENT, width=2))
    fig.add_annotation(text=_fmt(lead), xref="paper", yref="paper", x=0.012, y=_y(66),
                       xanchor="left", yanchor="middle", showarrow=False, align="left",
                       font=dict(size=18, color=_TBL_HEADER_FILL, family=_TBL_FONT_FAMILY))
    if sub:
        fig.add_annotation(text=_fmt(sub), xref="paper", yref="paper", x=0.012, y=_y(30),
                           xanchor="left", yanchor="middle", showarrow=False, align="left",
                           font=dict(size=13, color=_TBL_BANNER_SUB, family=_TBL_FONT_FAMILY))


def dataframe_to_table_image(df: pd.DataFrame, header_text: str = None, width: int = 1180, max_height: int = 1400) -> bytes:
    """Renders a pandas DataFrame as a styled table image for Telegram."""
    if df is None or df.empty:
        return None
    try:
        disp = df.copy()
        raw = df.copy()
        cols = list(disp.columns)
        nrows, ncols = len(df), len(cols)

        # 1. Format Values
        for col in cols:
            if col in ("Basis % (1Y APR)", "Basis %", "Basis Low %", "Basis High %"):
                disp[col] = disp[col].apply(lambda v: f"{v:+.3f}%" if pd.notna(v) and np.isfinite(v) else _TBL_BLANK)
            elif col in ("ATM 3h", "ATM Open", "IV−RV", "4hr Chg", "24hr Chg"):
                disp[col] = disp[col].apply(lambda v: f"{v:+.1f}" if pd.notna(v) and np.isfinite(v) else _TBL_BLANK)
            elif col in ("ATM σ%", "RV%", "25d σ Call", "25d σ Put", "IV", "Fwd IV"):
                disp[col] = disp[col].apply(lambda v: f"{v:.1f}" if pd.notna(v) and np.isfinite(v) else _TBL_BLANK)
            elif col == "Mid Price":
                disp[col] = disp[col].apply(lambda v: f"${float(v):,.2f}" if pd.notna(v) and np.isfinite(v) else _TBL_BLANK)
            elif col == "Basis $":
                disp[col] = disp[col].apply(lambda v: f"${float(v):,.0f}" if pd.notna(v) and np.isfinite(v) else _TBL_BLANK)
            elif col in ("Delta", "Vega", "Gamma (1%)"):
                disp[col] = disp[col].apply(lambda v: f"${float(v):,.0f}" if pd.notna(v) and np.isfinite(v) else _TBL_BLANK)
            elif col in ("Net Puts", "Net Calls", "Gross Notional"):
                disp[col] = disp[col].apply(lambda v: f"{int(v):,}" if pd.notna(v) and np.isfinite(v) else _TBL_BLANK)
            else:
                disp[col] = disp[col].apply(lambda v: _TBL_BLANK if pd.isna(v) else str(v))

        # 2. Conditional Color Matrix
        color_cols = {"ATM 3h", "ATM Open", "IV−RV", "4hr Chg", "24hr Chg", "Basis % (1Y APR)", "Basis %", "Delta", "Vega", "Gamma (1%)"}
        row_colors = [_TBL_ROW_EVEN if r % 2 == 0 else _TBL_ROW_ODD for r in range(nrows)]
        fill_color = [row_colors[:] for _ in range(ncols)]
        font_color = []

        for ci, c in enumerate(cols):
            if c in color_cols:
                col_colors, col_fills = [], []
                for ri, val in enumerate(raw[c]):
                    if pd.notna(val) and np.isfinite(val) and float(val) > 0:
                        col_colors.append(_TBL_POS)
                        col_fills.append(_TBL_POS_BG)
                    elif pd.notna(val) and np.isfinite(val) and float(val) < 0:
                        col_colors.append(_TBL_NEG)
                        col_fills.append(_TBL_NEG_BG)
                    else:
                        col_colors.append(_TBL_TEXT)
                        col_fills.append(row_colors[ri])
                font_color.append(col_colors)
                fill_color[ci] = col_fills
            else:
                font_color.append([_TBL_TEXT_MUTED if str(v) == _TBL_BLANK else _TBL_TEXT for v in disp[c]])

        header_h = 42
        row_h = 31
        has_banner = bool(header_text)
        top_margin = 108 if has_banner else 28
        bottom_margin = 46
        height = min(max_height, header_h + row_h * nrows + top_margin + bottom_margin)

        # 3. Build Plotly Table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[f"<b>{c}</b>" for c in disp.columns],
                fill_color=_TBL_HEADER_FILL,
                font=dict(color=_TBL_HEADER_FONT, size=13, family=_TBL_FONT_FAMILY),
                align=["left" if c in ("Token", "Expiry") else "right" for c in cols],
                height=header_h,
                line=dict(color=_TBL_HEADER_FILL, width=1),
            ),
            cells=dict(
                values=[disp[c].tolist() for c in cols],
                fill_color=fill_color,
                font=dict(size=12, color=font_color, family=_TBL_FONT_FAMILY),
                align=["left" if c in ("Token", "Expiry") else "right" for c in cols],
                height=row_h,
                line=dict(color=_TBL_GRID, width=1),
            ),
        )])

        fig.update_layout(
            margin=dict(l=22, r=22, t=top_margin, b=bottom_margin),
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=height,
            width=width,
            font=dict(family=_TBL_FONT_FAMILY, color=_TBL_TEXT),
        )

        if has_banner:
            _add_table_banner(fig, header_text, plot_h=height - top_margin - bottom_margin)

        return fig.to_image(format="png", width=width, height=height)
    except Exception as e:
        print(f"Error generating table image: {e}")
        return None