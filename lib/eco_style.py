"""Plotly styling modelled on the Ecoinometrics look (cream card, serif headline with one
highlighted word, red = positive / teal = negative, dashed grid, source footer).
Branding in the footer is ours ("MCM Analytics"), not theirs."""
from __future__ import annotations
import textwrap
import plotly.graph_objects as go

BG, INK, MUTED = "#F2DFCE", "#0B2A35", "#5F6B6D"
ORANGE, RED, TEAL, YELLOW = "#FF6B0A", "#F51E0A", "#4A9AB5", "#E8B30C"
SERIF = "Georgia, 'Times New Roman', serif"
# Fed-funds style colour ramp: teal -> yellow -> orange -> red
RAMP = [[0, TEAL], [0.35, "#C9C96A"], [0.6, YELLOW], [0.8, "#F08A0A"], [1, RED]]


def style(fig: go.Figure, title: str, highlight: str | None = None, subtitle: str = "",
          notes: str = "", height: int = 640, yfmt: str | None = None) -> go.Figure:
    """Apply the look. `highlight` is a word/phrase in `title` rendered in orange."""
    t = title.replace(highlight, f"<span style='color:{ORANGE}'>{highlight}</span>", 1) if highlight else title
    sub = "<br>".join(textwrap.wrap(subtitle, 95)) if subtitle else ""
    top = 55 + 25 * (sub.count("<br>") + 1 if sub else 0)
    fig.update_layout(
        template=None, paper_bgcolor=BG, plot_bgcolor=BG, height=height,
        font=dict(family=SERIF, color=MUTED, size=14),
        title=dict(text=f"<b>{t}</b>", xref="paper", x=0, xanchor="left", y=0.99, yanchor="top",
                   font=dict(family=SERIF, size=28, color=INK)),
        margin=dict(l=75, r=30, t=top + 25, b=135),
        showlegend=False, hoverlabel=dict(font_family=SERIF),
        xaxis=dict(showgrid=False, linecolor=MUTED, zeroline=False, tickfont=dict(size=13)),
        yaxis=dict(gridcolor=MUTED, griddash="dash", gridwidth=1, zeroline=True,
                   zerolinecolor=MUTED, zerolinewidth=1.5, tickfont=dict(size=13)),
    )
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    if sub:
        fig.add_annotation(text=sub, xref="paper", yref="paper", x=0, y=1.0, xanchor="left",
                           yanchor="bottom", yshift=12, showarrow=False, align="left",
                           font=dict(family=SERIF, size=17, color=INK))
    fig.add_annotation(text="<br>".join(textwrap.wrap(notes, 90)), xref="paper", yref="paper",
                       x=0.0, y=-0.2, xanchor="left", yanchor="top", showarrow=False,
                       align="left", font=dict(size=11, color=MUTED))
    fig.add_annotation(text="MCM Analytics", xref="paper", yref="paper", x=1.0, y=-0.2,
                       xanchor="right", yanchor="top", showarrow=False,
                       font=dict(family=SERIF, size=20, color=MUTED))
    return fig


def note(fig: go.Figure, x, y, text: str, ax: int = 0, ay: int = -45, color: str = INK, **kw):
    """Annotation with an arrow, in the same serif look."""
    fig.add_annotation(x=x, y=y, text=text, showarrow=True, arrowhead=2, arrowwidth=1,
                       arrowcolor=color, ax=ax, ay=ay, font=dict(family=SERIF, size=13, color=color), **kw)
