"""Regression tests for Telegram HTML-safe captions."""
from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.telegram_caption import caption_from_title


def test_caption_from_title() -> None:
    # <=3DTE in a Plotly note previously broke Telegram HTML parse_mode.
    raw = (
        "BTC Forward Vol Steepness<br>"
        "<span style='font-size:11px'>excludes <=3DTE; weighted to 30D</span>"
    )
    out = caption_from_title(raw, "fallback")
    assert "<=" not in out, out
    assert "&lt;=" in out, out
    assert "<span" not in out.lower(), out
    assert "</span>" not in out.lower(), out
    assert "<br" not in out.lower(), out
    assert "\n" in out
    assert out == html.escape(
        "BTC Forward Vol Steepness\nexcludes <=3DTE; weighted to 30D"
    )

    # br variants + span stripping
    assert "A\nB" in caption_from_title("A<br/>B", "fb")
    assert "A\nB" in caption_from_title("A<br />B", "fb")

    # truncation
    long = "x" * 900
    truncated = caption_from_title(long, "fb")
    assert truncated.endswith("…")
    assert len(truncated) <= 701  # 700 + ellipsis (ellipsis may be multi-byte char)

    # None / empty fallback
    assert caption_from_title(None, "FALLBACK") == "FALLBACK"
    assert caption_from_title("", "FALLBACK") == "FALLBACK"
    assert caption_from_title("   ", "FALLBACK") == "FALLBACK"

    # literal & < > escape
    esc = caption_from_title("A & B < C > D", "fb")
    assert "&amp;" in esc and "&lt;" in esc and "&gt;" in esc


if __name__ == "__main__":
    test_caption_from_title()
    print("test_caption_from_title: OK")
