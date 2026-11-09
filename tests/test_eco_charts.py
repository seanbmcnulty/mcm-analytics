"""Chart-structure checks for the two new Ecoinometrics views.

Locks the light export palette (dark axis text, white card) and the rule
that a missing ratio breaks the line instead of bridging the gap.
"""
import numpy as np
import pandas as pd

from lib import ecoinometrics as eco
from lib import eco_charts as charts


def _font_color(font):
    if font is None:
        return None
    return getattr(font, "color", None) or (font.get("color") if isinstance(font, dict) else None)


def _assert_dark(color):
    c = str(color or "").replace(" ", "").lower()
    assert c not in ("", "white", "#fff", "#ffffff", "#fafafa"), color
    assert not c.startswith("rgba(255,255,255"), color


def _assert_light_card(fig):
    assert fig.layout.paper_bgcolor == "white"
    assert fig.layout.plot_bgcolor == "white"
    _assert_dark(fig.layout.font.color)
    _assert_dark(_font_color(fig.layout.xaxis.title.font))
    _assert_dark(_font_color(fig.layout.xaxis.tickfont))
    _assert_dark(_font_color(fig.layout.yaxis.title.font))
    _assert_dark(_font_color(fig.layout.yaxis.tickfont))
    assert fig.layout.xaxis.title.text == "Date (SGT)"
    title = fig.layout.title.text or ""
    # The grey note span is added by finalize. The note body must not contain
    # a raw "<" (that is what broke Telegram HTML captions on other pages).
    note = title.split("font-size:11px", 1)[-1]
    assert "<=" not in note and "&" not in charts._RATIO_NOTE and "&" not in charts._FLOW_NOTE
    assert "<" not in charts._RATIO_NOTE and "<" not in charts._FLOW_NOTE


def test_ratio_figure_colours_by_close_and_breaks_gaps():
    dates = pd.date_range("2024-01-01", periods=6)
    df = pd.DataFrame({
        "date": dates,
        "close": [100.0, 110.0, np.nan, 130.0, 140.0, 150.0],
        "sma": [100.0, 100.0, np.nan, 100.0, 100.0, 100.0],
        "ratio": [1.0, 1.1, np.nan, np.nan, 1.4, 1.5],
    })
    # Leading null ratio is trimmed; the interior null must remain a break.
    df.loc[0, "ratio"] = np.nan
    fig = charts.figure_price_sma_ratio(df)
    _assert_light_card(fig)
    assert "Close / 200-day SMA" in (fig.layout.yaxis.title.text or "")
    assert "Missing days are left blank" in (fig.layout.title.text or "")
    marker = next(tr for tr in fig.data if tr.name == "Close / 200-day SMA")
    plotted = [pd.Timestamp(x).normalize() for x in marker.x]
    assert plotted[0] == pd.Timestamp("2024-01-02")
    assert pd.Timestamp("2024-01-03") not in plotted
    assert marker.marker.colorbar.title.text == "BTC close (USD)"
    _assert_dark(_font_color(marker.marker.colorbar.title.font))
    for tr in fig.data:
        if tr.mode != "lines":
            continue
        prev = None
        for x in tr.x:
            if x is None:
                prev = None
                continue
            if prev is not None:
                delta = pd.Timestamp(x).normalize() - pd.Timestamp(prev).normalize()
                assert delta == pd.Timedelta(days=1), (prev, x)
            prev = x

    # Adjacent sessions stay connected when the colour bin changes.
    # Only a missing close opens a gap.
    cont = pd.DataFrame({
        "date": pd.date_range("2024-02-01", periods=5),
        "close": [10.0, 20.0, 80.0, 90.0, 100.0],
        "sma": [50.0] * 5,
        "ratio": [0.2, 0.4, 1.6, 1.8, 2.0],
    })
    linked = charts.figure_price_sma_ratio(cont)
    pairs = set()
    for tr in linked.data:
        if tr.mode != "lines":
            continue
        xs = [pd.Timestamp(v).normalize() for v in tr.x if v is not None]
        pairs.update(zip(xs, xs[1:]))
    days = list(pd.date_range("2024-02-01", periods=5))
    for a, b in zip(days, days[1:]):
        assert (a, b) in pairs, (a, b, pairs)


def test_flow_figure_uses_regime_colours_and_light_axes():
    flows = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=63),
        "flow": [10.0] * 63,
    })
    labelled = eco.etf_flow_regime(flows)
    fig = charts.figure_etf_flow_regime(labelled)
    _assert_light_card(fig)
    assert "Net flow (BTC)" in (fig.layout.yaxis.title.text or "")
    title = fig.layout.title.text or ""
    # finalize wraps the note, so the two words may be split by a <br>.
    assert "independent approximation" in charts._FLOW_NOTE
    assert "Ecoinometrics model" in title
    assert "Farside" in title
    names = [tr.name for tr in fig.data]
    assert "Strong inflow" in names
    strong = next(tr for tr in fig.data if tr.name == "Strong inflow")
    assert list(strong.y)[-1] == 10.0
    assert strong.marker.color == charts.REGIME_COLOURS["strong_inflow"]

    short = eco.etf_flow_regime(flows.iloc[:10])
    warmup = charts.figure_etf_flow_regime(short)
    assert "Unclassified" in [tr.name for tr in warmup.data]
    assert any(len(tr.y) == 10 for tr in warmup.data if tr.name == "Unclassified")
    empty = charts.figure_etf_flow_regime(labelled.iloc[0:0])
    assert empty.layout.paper_bgcolor == "white"
