"""
Site password gate (same idea as the Mayor's Squawk Box login).

Every page calls ``require_password()`` right after its ``sys.path`` line,
before any other Streamlit call, so nothing renders until the visitor has
entered the shared password.

Password source (first non-empty wins):
  1. ``st.secrets["auth"]["password"]``  (Streamlit Cloud / .streamlit/secrets.toml)
  2. env var ``MCM_APP_PASSWORD``

FAILS OPEN while neither is set: the app behaves exactly as before and shows
a small dismissible warning, so deploying this can never lock the owner out
or break the scheduled Home chart blasts before the secret is pasted in.

Once logged in, a fingerprint of the password sits in ``st.session_state``,
so page-to-page navigation and the Home auto-pipeline (``st.switch_page``)
don't ask again. Changing the password invalidates every session and every
"remember this device" cookie (both are HMACs keyed on the password).

"Remember this device": an ``<expiry>.<hmac>`` cookie (30 days) set from a
zero-height component and read back through ``st.context.cookies`` on the
next visit. Best-effort only: if cookies are blocked the visitor simply logs
in again.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time

import streamlit as st

SESSION_KEY = "_mcm_auth"            # holds session_fingerprint() once logged in
COOKIE_NAME = "mcm_auth"
REMEMBER_DAYS = 30
_PENDING_COOKIE = "_mcm_auth_set_cookie"
_DISMISS_KEY = "_mcm_auth_warn_dismissed"
_FAILS_KEY = "_mcm_auth_fails"
_LOGGED_OUT = "_mcm_auth_logged_out"
_MAX_PW_CHARS = 256
CONTACT = "@seanbmcnulty"


# ---------------------------------------------------------------------------
# pure helpers (unit-tested without a Streamlit runtime)
# ---------------------------------------------------------------------------
def configured_password() -> str:
    """The site password, or '' when none is configured (gate fails open)."""
    pw = ""
    try:
        sect = st.secrets.get("auth")
        if sect is not None:
            pw = str(sect.get("password", "") or "")
    except Exception:            # no secrets.toml at all, or malformed
        pw = ""
    pw = pw.strip()
    if not pw:
        pw = (os.environ.get("MCM_APP_PASSWORD") or "").strip()
    return pw


def _mac(pw: str, msg: str) -> str:
    return hmac.new(pw.encode(), msg.encode(), hashlib.sha256).hexdigest()


def session_fingerprint(pw: str) -> str:
    return _mac(pw, "mcm-session-v1")


def check_password(entered: str, pw: str) -> bool:
    if not pw or not entered or len(entered) > _MAX_PW_CHARS:
        return False
    return hmac.compare_digest(entered.encode(), pw.encode())


def make_remember_token(pw: str, now: float | None = None, days: int = REMEMBER_DAYS) -> str:
    exp = int((now if now is not None else time.time()) + days * 86400)
    return f"{exp}.{_mac(pw, f'mcm-remember-v1:{exp}')}"


def verify_remember_token(token: str | None, pw: str, now: float | None = None) -> bool:
    if not token or not pw or "." not in token:
        return False
    exp_s, _, sig = token.partition(".")
    try:
        exp = int(exp_s)
    except ValueError:
        return False
    if exp < (now if now is not None else time.time()):
        return False
    return hmac.compare_digest(sig, _mac(pw, f"mcm-remember-v1:{exp}"))


# ---------------------------------------------------------------------------
# Streamlit side
# ---------------------------------------------------------------------------
def _cookie_js(value: str, max_age: int) -> str:
    # value is [0-9a-f.] only; no escaping needed.
    return ("<script>try{document.cookie='" + COOKIE_NAME + "=" + value +
            f"; Max-Age={max_age}; Path=/; SameSite=Lax" +
            "'+(location.protocol==='https:'?'; Secure':'');}catch(e){}</script>")


def _emit_cookie(value: str, max_age: int) -> None:
    html = _cookie_js(value, max_age)
    try:
        if hasattr(st, "iframe"):                      # Streamlit >= 1.52
            with st.container(height=1, border=False):
                st.iframe(html, height=1)
        else:
            import streamlit.components.v1 as components
            components.html(html, height=0)
    except Exception:
        pass


def _cookie_value() -> str | None:
    try:
        return st.context.cookies.get(COOKIE_NAME)
    except Exception:
        return None


def is_authenticated(pw: str | None = None) -> bool:
    pw = configured_password() if pw is None else pw
    if not pw:
        return True
    tok = st.session_state.get(SESSION_KEY)
    return isinstance(tok, str) and hmac.compare_digest(tok, session_fingerprint(pw))


def logout() -> None:
    st.session_state.pop(SESSION_KEY, None)
    st.session_state.pop(_PENDING_COOKIE, None)
    st.session_state["_mcm_auth_clear_cookie"] = True
    st.session_state[_LOGGED_OUT] = True   # ignore this connection's cookie snapshot


_HIDE_CHROME_CSS = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarNav"],
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"] { display: none !important; }
.mcm-login-title { text-align:center; font-family: ui-monospace, Menlo, Consolas, monospace;
  font-weight:700; letter-spacing:.18em; text-transform:uppercase; font-size:1.05rem; margin:0; }
.mcm-login-sub { text-align:center; opacity:.65; font-size:.85rem; margin:.25rem 0 0 0; }
.mcm-login-foot { text-align:center; opacity:.55; font-size:.75rem; margin-top:.75rem; }
</style>
"""


def _login_screen(pw: str) -> None:
    try:
        st.set_page_config(page_title="MCM Analytics — Sign in", page_icon="🔒",
                           layout="centered", initial_sidebar_state="collapsed")
    except Exception:
        pass
    st.markdown(_HIDE_CHROME_CSS, unsafe_allow_html=True)
    if st.session_state.pop("_mcm_auth_clear_cookie", False):
        _emit_cookie("", 0)
    st.markdown("<div style='height:12vh'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        with st.container(border=True):
            st.markdown("<p class='mcm-login-title'>📊 MCM Analytics</p>"
                        "<p class='mcm-login-sub'>Private terminal. Enter the access password.</p>",
                        unsafe_allow_html=True)
            with st.form("mcm_login_form", border=False):
                entered = st.text_input("Password", type="password", key="mcm_login_pw",
                                        placeholder="Password", label_visibility="collapsed")
                remember = st.checkbox(f"Remember this device for {REMEMBER_DAYS} days",
                                       value=True, key="mcm_login_remember")
                submitted = st.form_submit_button("Enter", type="primary", width="stretch")
            if submitted:
                fails = int(st.session_state.get(_FAILS_KEY, 0))
                if fails >= 5:                       # slow down guessing in this session
                    time.sleep(min(2 * (fails - 4), 10))
                if check_password(entered or "", pw):
                    st.session_state[SESSION_KEY] = session_fingerprint(pw)
                    st.session_state.pop(_FAILS_KEY, None)
                    st.session_state.pop(_LOGGED_OUT, None)
                    st.session_state.pop("mcm_login_pw", None)
                    if remember:
                        st.session_state[_PENDING_COOKIE] = make_remember_token(pw)
                    st.rerun()
                else:
                    st.session_state[_FAILS_KEY] = fails + 1
                    st.error("Incorrect password.")
            st.markdown(f"<p class='mcm-login-foot'>Need access? Ask {CONTACT}.</p>",
                        unsafe_allow_html=True)
    st.stop()


def _fail_open_warning() -> None:
    if st.session_state.get(_DISMISS_KEY):
        return
    c1, c2 = st.columns([12, 1])
    with c1:
        st.warning("🔓 Password protection isn't configured yet. Add `[auth] password` "
                   "to the app's Streamlit secrets to turn on the login screen.")
    with c2:
        st.button("Dismiss", key="_mcm_auth_dismiss",
                  on_click=lambda: st.session_state.__setitem__(_DISMISS_KEY, True))


def require_password() -> None:
    """Gate the current page. Call before any other Streamlit command."""
    pw = configured_password()
    if not pw:
        _fail_open_warning()
        return
    if is_authenticated(pw):
        tok = st.session_state.pop(_PENDING_COOKIE, None)
        if tok:
            _emit_cookie(tok, REMEMBER_DAYS * 86400)
        logout_button()
        return
    if not st.session_state.get(_LOGGED_OUT) and verify_remember_token(_cookie_value(), pw):
        st.session_state[SESSION_KEY] = session_fingerprint(pw)
        logout_button()
        return
    _login_screen(pw)


def logout_button() -> None:
    """Small sidebar 'Log out' control (only when the gate is enforcing)."""
    if st.sidebar.button("🔒 Log out", key="_mcm_logout"):
        logout()
        st.rerun()
