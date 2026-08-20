"""
MCM Analytics — Telegram integration for report blasting.

Reads bot_token and chat_id from secrets.toml (placeholder until configured).
Provides rate-limited posting with retry logic.
"""

import time
import io
import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

def _load_credentials() -> tuple[str, str]:
    """Load Telegram credentials from Streamlit secrets or config files."""
    import os
    import toml
    from pathlib import Path

    # 1. Streamlit secrets (native)
    try:
        token = st.secrets.get("telegram", {}).get("bot_token", "")
        chat_id = st.secrets.get("telegram", {}).get("chat_id", "")
        if token and chat_id and "YOUR_" not in token:
            return token, chat_id
    except Exception:
        pass

    # 2. Config file (~/.config/mcm-analytics/secrets.toml)
    config_path = Path.home() / ".config" / "mcm-analytics" / "secrets.toml"
    if config_path.exists():
        try:
            data = toml.load(config_path)
            tg = data.get("telegram", {})
            token = tg.get("bot_token", "")
            chat_id = tg.get("chat_id", "")
            if token and "YOUR_" not in token:
                return token, chat_id
        except Exception:
            pass

    # 3. Environment variables
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if token and chat_id:
        return token, chat_id

    return "", ""


def credentials() -> tuple[str, str]:
    """
    Resolve (token, chat_id) fresh on every call.

    These used to be resolved once at import time, which meant that on
    Streamlit Cloud — where the module is imported before the secrets store is
    necessarily readable — a miss was cached for the life of the process and
    the app claimed Telegram was unconfigured forever.
    """
    try:
        return _load_credentials()
    except Exception:
        return "", ""


def __getattr__(name):
    """Keep TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID working as live lookups."""
    if name == "TELEGRAM_BOT_TOKEN":
        return credentials()[0]
    if name == "TELEGRAM_CHAT_ID":
        return credentials()[1]
    raise AttributeError(name)

# Rate limiting — Telegram's Bot API caps outgoing messages to a single
# chat at roughly 1/second, and specifically caps *group* chats at 20/min
# (see https://core.telegram.org/bots/faq#my-bot-is-hitting-limits-how-do-i-avoid-this).
# The dashboard's "Send all" can queue 100+ photos to one chat_id, and if
# that chat is a group, 20/min is the binding constraint — 1.5s spacing
# (40/min) blew straight through it, which is why a few images went
# missing. 3.2s spacing keeps every send under the group cap with a
# buffer, at the cost of a slower "Send all" (worth it for completeness
# over speed here).
_last_send_time = 0.0
_MIN_SEND_INTERVAL = 3.2  # seconds between sends to the same chat
_MAX_429_RETRIES = 3      # extra retries if Telegram still says slow down


def is_configured() -> bool:
    """True when a usable bot token and chat id are available right now."""
    token, chat_id = credentials()
    return bool(token and chat_id and "YOUR_" not in token)


def config_status() -> str:
    """Human-readable reason Telegram is unavailable (for the UI)."""
    token, chat_id = credentials()
    if not token and not chat_id:
        return "No credentials found in secrets, config file or environment."
    if not token:
        return "chat_id found but bot_token is missing."
    if not chat_id:
        return "bot_token found but chat_id is missing."
    if "YOUR_" in token:
        return "bot_token is still the placeholder value."
    return "Configured."


def _post_with_backoff(request_fn):
    """Call `request_fn()` (a zero-arg thunk returning a requests.Response),
    honoring Telegram's own 429 "Too Many Requests" response instead of
    just giving up. On 429, Telegram tells you exactly how long to wait
    in `parameters.retry_after` — sleep that plus a small buffer and try
    again, up to `_MAX_429_RETRIES` times. `_throttle()` below prevents
    most 429s in the first place; this is the backstop for the ones that
    still get through (a burst right after a redeploy, another process
    posting to the same chat, etc.) — without it, a rate-limited send was
    just dropped, which is why charts were going missing.

    Returns the final Response, or None if the request itself raised
    (network error, timeout) after all attempts.
    """
    for attempt in range(_MAX_429_RETRIES + 1):
        try:
            resp = request_fn()
        except Exception:
            return None
        if resp.status_code != 429:
            return resp
        if attempt >= _MAX_429_RETRIES:
            return resp
        try:
            retry_after = float(resp.json().get("parameters", {}).get("retry_after", 5))
        except Exception:
            retry_after = 5.0
        time.sleep(retry_after + 0.5)
    return None


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Send a text message to the configured Telegram chat."""
    token, chat_id = credentials()
    if not (token and chat_id):
        return False
    _throttle()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = _post_with_backoff(lambda: requests.post(url, data={
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
    }, timeout=10))
    return resp is not None and resp.status_code == 200


def send_photo(image_bytes: bytes, caption: str = "") -> bool:
    """Send a photo (PNG bytes) to the configured Telegram chat."""
    token, chat_id = credentials()
    if not (token and chat_id):
        return False
    _throttle()
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption[:1024]
        data["parse_mode"] = "HTML"

    def _post():
        # Rebuilt fresh on every attempt — a BytesIO consumed by one POST
        # can't be replayed on a retry.
        files = {"photo": ("chart.png", io.BytesIO(image_bytes), "image/png")}
        return requests.post(url, files=files, data=data, timeout=30)

    resp = _post_with_backoff(_post)
    return resp is not None and resp.status_code == 200


def _throttle():
    """Enforce minimum interval between sends."""
    global _last_send_time
    now = time.time()
    elapsed = now - _last_send_time
    if elapsed < _MIN_SEND_INTERVAL:
        time.sleep(_MIN_SEND_INTERVAL - elapsed)
    _last_send_time = time.time()
