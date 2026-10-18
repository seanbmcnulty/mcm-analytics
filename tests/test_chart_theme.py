"""apply_chart_theme: switching themes must not leak colours (no network)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from lib import fx_style  # noqa: E402


def _fig():
    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_trace(go.Scatter(y=[1, 2]))
    f.add_trace(go.Scatter(y=[3, 1]), secondary_y=True)
    f.add_annotation(text="box", bgcolor="rgba(255,255,255,0.8)", showarrow=False)
    return f


def test_light_then_auto_clears_colours():
    f = fx_style.apply_chart_theme(_fig(), "light")
    assert f.layout.paper_bgcolor == "white"
    assert f.layout.yaxis2.tickfont.color == fx_style._TBL_TEXT
    fx_style.apply_chart_theme(f, "auto")
    assert f.layout.paper_bgcolor is None and f.layout.font.color is None
    for ax in (f.layout.xaxis, f.layout.yaxis, f.layout.yaxis2):
        assert ax.color is None and ax.tickfont.color is None and ax.title.font.color is None


def test_dark_uses_light_text_and_keeps_boxed_annotation_readable():
    f = fx_style.apply_chart_theme(_fig(), "dark")
    assert f.layout.font.color == fx_style._DARK_TEXT
    assert f.layout.yaxis2.tickfont.color == fx_style._DARK_TEXT
    assert f.layout.paper_bgcolor == fx_style._DARK_BG
    assert f.layout.annotations[0].font.color == fx_style._TBL_TEXT


def test_streamlit_theme_arg():
    assert fx_style.streamlit_chart_theme("auto") == "streamlit"
    assert fx_style.streamlit_chart_theme("light") is None
    assert fx_style.streamlit_chart_theme("dark") is None
