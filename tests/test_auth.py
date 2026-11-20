"""Site password gate (lib/auth.py): pure helpers + Streamlit AppTest flows.

    PYTHONPATH=. python3 tests/test_auth.py
"""
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import auth  # noqa: E402

APP = ROOT / "tests" / "fixtures" / "auth_app" / "app.py"
PW = "unit-test-pass-123"


@contextmanager
def _no_env_pw():
    saved = os.environ.pop("MCM_APP_PASSWORD", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["MCM_APP_PASSWORD"] = saved


def _text(at):
    return " ".join([m.value for m in at.markdown] + [t.value for t in at.title])


def _new(secret=True):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=30)
    if secret:
        at.secrets["auth"] = {"password": PW}
    return at


def test_helpers():
    assert auth.check_password(PW, PW)
    assert not auth.check_password("nope", PW)
    assert not auth.check_password("", PW) and not auth.check_password(PW, "")
    assert not auth.check_password("x" * 300, "x" * 300)
    tok = auth.make_remember_token(PW, now=1_000_000)
    assert re.fullmatch(r"\d+\.[0-9a-f]{64}", tok)
    assert auth.verify_remember_token(tok, PW, now=1_000_100)
    assert not auth.verify_remember_token(tok, PW, now=1_000_000 + 31 * 86400)   # expired
    assert not auth.verify_remember_token(tok, "other", now=1_000_100)            # rotated pw
    exp, sig = tok.split(".")
    assert not auth.verify_remember_token(f"{int(exp) + 999}.{sig}", PW, now=1_000_100)  # tampered
    assert not auth.verify_remember_token("garbage", PW)
    assert not auth.verify_remember_token(None, PW)
    assert auth.session_fingerprint(PW) != auth.session_fingerprint("other")
    assert PW not in tok and PW not in auth.session_fingerprint(PW)


def test_env_fallback():
    saved = os.environ.get("MCM_APP_PASSWORD")
    os.environ["MCM_APP_PASSWORD"] = "  from-env  "
    try:
        assert auth.configured_password() == "from-env"
    finally:
        if saved is None:
            os.environ.pop("MCM_APP_PASSWORD", None)
        else:
            os.environ["MCM_APP_PASSWORD"] = saved


def test_no_secret_fails_open():
    with _no_env_pw():
        at = _new(secret=False)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert "SECRET HOME CONTENT" in _text(at)
        assert any("isn't configured" in w.value for w in at.warning)
        assert not at.text_input                          # no login form
        at.button(key="_mcm_auth_dismiss").click().run()
        assert not any("isn't configured" in w.value for w in at.warning)
        assert "SECRET HOME CONTENT" in _text(at)


def test_wrong_password_blocked():
    with _no_env_pw():
        at = _new()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert "SECRET" not in _text(at)
        assert "MCM Analytics" in _text(at)
        at.text_input(key="mcm_login_pw").input("wrong")
        at.button[0].click().run()
        assert any("Incorrect password" in e.value for e in at.error)
        assert "SECRET" not in _text(at)
        assert auth.SESSION_KEY not in at.session_state


def test_right_password_and_page_switch():
    with _no_env_pw():
        at = _new()
        at.run()
        at.text_input(key="mcm_login_pw").input(PW)
        at.button[0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert "SECRET HOME CONTENT" in _text(at), _text(at)
        assert not at.text_input
        at.switch_page("pages/other.py").run()
        assert "SECRET OTHER PAGE" in _text(at), _text(at)
        assert not at.text_input                           # not asked again
        at.switch_page("app.py").run()
        assert "SECRET HOME CONTENT" in _text(at)
        # log out -> gate again
        at.sidebar.button(key="_mcm_logout").click().run()
        assert "SECRET" not in _text(at) and at.text_input


def test_env_var_password_enforces():
    saved = os.environ.get("MCM_APP_PASSWORD")
    os.environ["MCM_APP_PASSWORD"] = PW
    try:
        at = _new(secret=False)
        at.run()
        assert "SECRET" not in _text(at)
        at.text_input(key="mcm_login_pw").input(PW)
        at.button[0].click().run()
        assert "SECRET HOME CONTENT" in _text(at)
    finally:
        if saved is None:
            os.environ.pop("MCM_APP_PASSWORD", None)
        else:
            os.environ["MCM_APP_PASSWORD"] = saved


def test_every_page_is_gated_first():
    files = [ROOT / "Home.py"] + sorted((ROOT / "pages").glob("*.py"))
    assert len(files) >= 16
    for f in files:
        src = f.read_text(encoding="utf-8")
        gate = src.find("\nrequire_password()")
        assert gate > 0, f"{f.name}: no require_password() call"
        body = src[gate:]
        head = src[:gate]
        # nothing Streamlit-ish may run before the gate
        assert not re.search(r"^\s*(st\.|import streamlit|from streamlit)", head, re.M), f.name
        assert not re.search(r"^[a-z_]+\.render_page\(", head, re.M), f.name
        assert not re.search(r"^(?!_fresh\.reload_if_changed\(_auth\))[a-z_]+\.reload_if_changed\(",
                             head, re.M), f.name
        assert "from lib.auth import require_password" in head, f.name
        # the gate module itself is reloaded after a Cloud redeploy (lib/freshness.py)
        assert "_fresh.reload_if_changed(_auth)" in head, f.name


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
