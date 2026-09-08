"""Telegram HTML-safe caption helpers shared by MCM Bot / Block Trades."""

from __future__ import annotations

import html
import re

_SPAN_TAG_RE = re.compile(r"</?span[^>]*>", re.IGNORECASE)
_CAPTION_SAFE_LEN = 700  # under Telegram's 1024-char caption cap after escaping


def caption_from_title(title_text: str | None, fallback: str) -> str:
    """Turn a Plotly figure's title.text into a safe Telegram HTML-mode caption.

    Plotly titles can carry Plotly markup (<br>, <span style=...>) while the
    note text itself may contain literal characters that aren't markup (e.g.
    "excludes <=3DTE"). Sent under parse_mode=HTML, that "<=" plus a later
    </span> reads as one malformed tag and Telegram rejects the message with
    400 can't parse entities. Convert <br> to newlines, strip <span> tags,
    html.escape() the rest, and truncate generously under the caption cap.
    """
    if not title_text:
        return fallback
    text = (str(title_text).replace("<br>", "\n")
                           .replace("<br/>", "\n")
                           .replace("<br />", "\n"))
    text = _SPAN_TAG_RE.sub("", text).strip()
    if len(text) > _CAPTION_SAFE_LEN:
        text = text[:_CAPTION_SAFE_LEN].rstrip() + "…"
    text = html.escape(text)
    return text or fallback
