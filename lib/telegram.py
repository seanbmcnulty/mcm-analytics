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


# Module-level credentials (loaded once)
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID = _load_credentials()

# Rate limiting
_last_send_time = 0.0
_MIN_SEND_INTERVAL = 1.5  # seconds between sends


def is_configured() -> bool:
    """Check if Telegram credentials are configured."""
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
                and "YOUR_" not in TELEGRAM_BOT_TOKEN)


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Send a text message to the configured Telegram chat."""
    if not is_configured():
        return False
    _throttle()
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": parse_mode,
        }, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def send_photo(image_bytes: bytes, caption: str = "") -> bool:
    """Send a photo (PNG bytes) to the configured Telegram chat."""
    if not is_configured():
        return False
    _throttle()
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {"photo": ("chart.png", io.BytesIO(image_bytes), "image/png")}
        data = {"chat_id": TELEGRAM_CHAT_ID}
        if caption:
            data["caption"] = caption[:1024]
            data["parse_mode"] = "HTML"
        resp = requests.post(url, files=files, data=data, timeout=30)
        return resp.status_code == 200
    except Exception:
        return False


def _throttle():
    """Enforce minimum interval between sends."""
    global _last_send_time
    now = time.time()
    elapsed = now - _last_send_time
    if elapsed < _MIN_SEND_INTERVAL:
        time.sleep(_MIN_SEND_INTERVAL - elapsed)
    _last_send_time = time.time()
