"""Plotly styling modelled on the Ecoinometrics look (cream card, serif headline with one
highlighted word, red = positive / teal = negative, dashed grid, source footer).
Branding in the footer is ours ("MCM Analytics"), not theirs."""
from __future__ import annotations
import math
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


# One swatch per year on the hack scatter, oldest first. Cooler years read as
# earlier; the last swatches are the page's orange and red. Each one stays
# readable on the cream card and on the white background fig_to_png forces
# for a Telegram export (pale yellow does not).
HACK_YEAR_COLOURS = (
    "#1B4F72",  # navy
    "#4A9AB5",  # teal
    "#1F7A4D",  # green
    "#6C3A8A",  # purple — olive sat too close to the gold year beside it
    "#C48A12",  # gold
    "#E07A3D",  # clay
    "#FF6B0A",  # orange
    "#F51E0A",  # red
    "#7A1230",  # burgundy
)


def hack_scatter(res, horizon: int) -> go.Figure:
    """Hack-size scatter coloured by year.

    ``res`` is the frame from ``ecoinometrics.hack_event_returns``. Filled
    circles finished the window. Open circles are still inside it: their y is
    ``partial`` (return to the latest close), the hover says ``(incomplete)``,
    and they are left out of the average. The y-axis zero line, drawn by
    ``style``, is the return-sign cue. Point labels and the year legend use
    ``INK`` so a light-theme PNG export keeps them dark.
    """
    horizon = int(horizon)
    done = res[res["complete"].astype(bool)]
    n_done, n_open = len(done), len(res) - len(done)
    if n_done:
        m = float(done["ret"].mean())
        subtitle = (
            f"Across {n_done} exchange hacks with a full {horizon}-day window "
            f"the average Bitcoin return was {m:.1%}."
        )
    else:
        m = None
        subtitle = f"No hack in this list has a full {horizon}-day window yet."
    if n_open:
        subtitle += (
            f" {n_open} more {'is' if n_open == 1 else 'are'} still inside the window "
            "(open circles, excluded from the average)."
        )
    subtitle += " Colour is the year. Size tells us little about what happens next."
    # Two wrapped lines is what the footer slot under style() actually shows.
    notes = (
        f"Notes: UTC event-date close to +{horizon} days. Deribit bars close at 08:00 UTC "
        "the next morning, so a late-evening hack starts after the theft. "
        "Open circles are unfinished and excluded from the average."
    )

    years = sorted({int(y) for y in res["date"].dt.year})
    colour = {y: HACK_YEAR_COLOURS[i % len(HACK_YEAR_COLOURS)] for i, y in enumerate(years)}
    xmax = float(res["loss_usd_m"].max())
    fig = go.Figure()
    for year in years:
        g = res[res["date"].dt.year == year].sort_values(
            ["complete", "loss_usd_m"], ascending=[False, True])
        positions, symbols, sizes, widths, custom = [], [], [], [], []
        for _, row in g.iterrows():
            v, r = float(row["loss_usd_m"]), float(row["partial"])
            # The largest hack is the rightmost point on the log axis; its
            # label stays to the left of the dot so it isn't clipped.
            if v == xmax:
                positions.append("middle left")
            elif v > 400:
                positions.append("top left")
            elif r < 0:
                positions.append("bottom right")
            else:
                positions.append("top right")
            open_ = not bool(row["complete"])
            symbols.append("circle-open" if open_ else "circle")
            sizes.append(16 if open_ else 12)
            widths.append(2.2 if open_ else 0.4)
            if open_:
                custom.append(f"<br>(incomplete, {int(row['elapsed'])}d of {horizon}d)")
            else:
                custom.append("")
        fig.add_trace(go.Scatter(
            x=g["loss_usd_m"], y=g["partial"].astype(float) * 100,
            mode="markers+text", name=str(year),
            text=g["exchange"].astype(str) + " (" + g["date"].dt.year.astype(int).astype(str) + ")",
            textposition=positions,
            textfont=dict(family="Arial", size=12, color=INK),
            marker=dict(size=sizes, symbol=symbols, color=colour[year],
                        line=dict(width=widths, color=colour[year])),
            customdata=custom,
            hovertemplate="%{text}<br>%{y:.1f}% | $%{x:,.1f}M%{customdata}<extra></extra>",
        ))

    style(fig, "Bitcoin Returns After Major Crypto Hacks", "Bitcoin", subtitle, notes)
    lo = min(float(res["loss_usd_m"].min()) * 0.65, 4.0)
    hi = float(res["loss_usd_m"].max()) * 2.2
    fig.update_xaxes(
        type="log",
        title_text="Reported hack size (USD millions, log scale)",
        tickvals=[5, 10, 30, 100, 300, 1000, 3000, 10000],
        ticktext=["$5M", "$10M", "$30M", "$100M", "$300M", "$1,000M", "$3,000M", "$10,000M"],
        range=[math.log10(lo), math.log10(hi)],
    )
    fig.update_yaxes(ticksuffix="%", title_text=f"Bitcoin return after {horizon} days")
    if m is not None and m == m:  # not NaN
        sizes = sorted(float(v) for v in res["loss_usd_m"])
        note_x = math.sqrt(sizes[-1] * sizes[-2]) if len(sizes) >= 2 else sizes[-1]
        # Log axis `range` is in log10 units, and an annotation's x is read in
        # that same space (a raw dollar value lands off the paper).
        note(fig, math.log10(note_x), m * 100,
             f"On average, returns<br>are {m:.1%}", ax=-24, ay=-46,
             xref="x", yref="y")
    mgn = fig.layout.margin
    fig.update_layout(
        showlegend=True,
        legend=dict(
            title=dict(text="Year", font=dict(family=SERIF, size=13, color=INK)),
            orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.01,
            font=dict(family=SERIF, size=12, color=INK),
            bgcolor="rgba(0,0,0,0)", itemsizing="constant",
        ),
        margin=dict(l=mgn.l, r=120, t=mgn.t, b=mgn.b),
    )
    return fig
