"""Streamlit AppTest smoke run for the Home chart blast (needs network: the
pages fetch live Deribit / Cboe / FRED data). Telegram is mocked: every
send is recorded, nothing leaves the box.

    PYTHONPATH=. python3 tests/apptest_blast.py [page ...]

Checks the Home selector (widgets, defaults, group buttons, the button
label) and runs each page's hook as a one-step plan with a small selection,
asserting it sends only what was selected, then finishes with the summary.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["MCM_BLAST_SELECTION_PATH"] = str(Path(tempfile.mkdtemp()) / "sel.json")

from streamlit.testing.v1 import AppTest  # noqa: E402

from lib import blast  # noqa: E402
import lib.telegram as tg  # noqa: E402

SENT: list = []


def _photo(png, caption="", parse_mode="HTML"):
    assert png, "empty PNG"
    SENT.append(("photo", caption))
    return True


def _msg(text, parse_mode="HTML"):
    SENT.append(("msg", text.splitlines()[0] if text else ""))
    return True


tg.send_photo = _photo
tg.send_message = _msg
tg.is_configured = lambda: True


def _fake_cboe(symbol="IBIT", timeout=25.0):
    """Synthetic Cboe chain (the box's IP gets a 403 from Cboe)."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    opts = []
    for days in (20, 55, 120):
        ymd = (now + timedelta(days=days)).strftime("%y%m%d")
        for k in range(30, 71, 2):
            for cp in "CP":
                iv = max(0.40 + (0.002 * (50 - k) if cp == "P" else 0.0005 * (k - 50)), 0.1)
                opts.append(dict(option=f"IBIT{ymd}{cp}{k * 1000:08d}", bid=1.0, ask=1.2, iv=iv,
                                 open_interest=100.0 + k, volume=10.0 * (k % 5)))
    return {"timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "data": {"current_price": 50.0, "iv30": 40.1, "prev_day_close": 49.0,
                     "price_change_percent": 1.0, "volume": 1, "options": opts}}


if os.environ.get("MCM_APPTEST_REAL_CBOE") != "1":
    import lib.ibit as _ibit
    _ibit.fetch_raw = _fake_cboe


def _errors(at):
    return [e.value for e in at.exception] + [e.value for e in at.error]


def check_home():
    at = AppTest.from_file(str(ROOT / "Home.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    cbs = {cb.key: cb for cb in at.checkbox if cb.key and cb.key.startswith("blast_")}
    assert len(cbs) == len(blast.ITEMS) + len(blast.ALL_ASSETS), len(cbs)
    on = {k.split("::", 1)[1] for k, cb in cbs.items() if k.startswith("blast_cb::") and cb.value}
    assert on == blast.default_items(), on ^ blast.default_items()
    assets = [k.split("::", 1)[1] for k, cb in cbs.items() if k.startswith("blast_asset::") and cb.value]
    assert assets == ["BTC", "ETH"], assets
    go = at.button(key="blast_go")
    assert go.label == blast.BUTTON_LABEL and not go.disabled
    print("  label:", go.label)
    print("  caption:", [c.value for c in at.caption if c.value.startswith("Selected:")][0])
    # group buttons
    at.button(key="blast_gnone_mcm").click().run()
    assert not any(at.checkbox(key=f"blast_cb::{it.key}").value for it in blast.ITEMS if it.page == "mcm")
    at.button(key="blast_gall_eco").click().run()
    assert all(at.checkbox(key=f"blast_cb::{it.key}").value for it in blast.ITEMS if it.page == "eco")
    at.button(key="blast_none").click().run()
    assert at.button(key="blast_go").disabled                       # nothing selected
    at.button(key="blast_reset").click().run()
    on = {it.key for it in blast.ITEMS if at.checkbox(key=f"blast_cb::{it.key}").value}
    assert on == blast.default_items()
    # save as default -> a new session starts from it
    at.checkbox(key="blast_cb::eco.fed_tone").check().run()
    at.checkbox(key="blast_asset::SOL").check().run()
    at.button(key="blast_save").click().run()
    at2 = AppTest.from_file(str(ROOT / "Home.py"), default_timeout=180)
    at2.run()
    assert at2.checkbox(key="blast_cb::eco.fed_tone").value and at2.checkbox(key="blast_asset::SOL").value
    blast.forget_saved()
    # a finished blast's summary shows on Home
    at2.session_state[blast.SUMMARY] = {"log": [{"step": "mcm", "title": "MCM Bot (BTC)", "sent": 2,
                                                 "failed": [], "note": None}],
                                        "skipped": [], "finished": "2026-11-20T01:00:00+00:00"}
    at2.run()
    assert any("Last chart blast" in s.value for s in at2.success), [s.value for s in at2.success]
    assert not at2.exception
    print("ok  Home selector")


def check_chain():
    """Press the Home button with a small selection and follow the hand-offs
    (switch_page / rerun) through three pages to the summary."""
    at = AppTest.from_file(str(ROOT / "Home.py"), default_timeout=600)
    at.run()
    at.button(key="blast_none").click().run()
    for k in ("fng.summary", "spot_vol.summary", "tbrv.overview_text"):
        at.checkbox(key=f"blast_cb::{k}").check().run()
    at.checkbox(key="blast_asset::ETH").uncheck().run()
    SENT.clear()
    at.button(key="blast_go").click().run()
    hops = 0
    while at.session_state[blast.FLAG] is not None and hops < 8:
        hops += 1
        at.run()                      # AppTest stops at each switch_page; continue
    assert not at.exception, [e.value for e in at.exception]
    kinds = [c.split("</b>")[0] for _, c in SENT]
    assert kinds == ["📊 <b>Time Based Realized Vol — Overview", "📈 <b>Spot Vol Correlation – BTC",
                     "<b>BTC Fear &amp; Greed Signal"], SENT
    summary = at.session_state[blast.SUMMARY]
    assert [e["step"] for e in summary["log"]] == ["tbrv:BTC", "spot_vol", "fng:BTC"], summary
    print(f"ok  chain (hops={hops}) log={[(e['step'], e['sent'], e['failed']) for e in summary['log']]}")


def check_default_chain():
    """Press the Home button with the untouched default selection (slow: the
    full twice-daily blast, every image rendered, sends mocked)."""
    at = AppTest.from_file(str(ROOT / "Home.py"), default_timeout=1500)
    at.run()
    SENT.clear()
    t0 = time.time()
    at.button(key="blast_go").click().run()
    hops = 0
    while at.session_state[blast.FLAG] is not None and hops < 8:
        hops += 1
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    summary = at.session_state[blast.SUMMARY]
    steps = [e["step"] for e in summary["log"]]
    assert steps == ["mcm", "bt_deribit", "tbrv:BTC", "tbrv:ETH", "spot_vol"], steps
    print(f"ok  default chain {time.time() - t0:.0f}s, {len(SENT)} Telegram calls: "
          f"{[(e['title'], e['sent'], len(e['failed'])) for e in summary['log']]}")
    for e in summary["log"]:
        if e["failed"]:
            print("    failed:", e["title"], e["failed"][:6])


# page id -> (items, assets, expected predicate on SENT)
CASES = {
    "mcm": ({"mcm.vol_run", "mcm.moonphase"}, ["BTC"],
            lambda s: len(s) == 2 and "/vol_run" in s[0][1] and "moon" in s[1][1].lower()),
    "bt_deribit": ({"bt.venue.deribit", "bt.chart.scatter", "bt.chart.put_call"}, ["BTC"],
                   lambda s: len(s) == 2),
    "bt_okx": ({"bt.venue.okx", "bt.chart.scatter"}, ["BTC", "ETH"], lambda s: len(s) <= 2),
    # opt-in "won vs market" chart: sent only when Paradex read-only keys are configured (0 sends otherwise)
    "bt_paradex": ({"bt.venue.paradex", "bt.chart.won"}, ["BTC"],
                   lambda s: len(s) <= 1 and all("Won vs Market" in c for _, c in s)),
    "tbrv": ({"tbrv.overview_text", "tbrv.cross_heatmap"}, ["BTC"],
             lambda s: [k for k, _ in s] == ["msg", "photo"]),
    "regime": ({"regime.summary", "regime.gauge"}, ["BTC"], lambda s: [k for k, _ in s] == ["msg", "photo"]),
    "spot_vol": ({"spot_vol.summary", "spot_vol.cvol_scatter", "spot_vol.dvol_spread"}, ["BTC"],
                 lambda s: [k for k, _ in s] == ["msg", "photo", "photo"]),
    "macro": ({"macro.summary", "macro.dvol_crush"}, ["ETH"], lambda s: [k for k, _ in s] == ["msg", "photo"]),
    "fng": ({"fng.summary"}, ["HYPE"], lambda s: s == [("msg", "<b>HYPE Fear &amp; Greed Signal</b>")]),
    "ibit": ({"ibit.max_pain"}, ["BTC"], lambda s: len(s) == 1 and "Max pain" in s[0][1]),
    "eco": ({"eco.fed_tone", "eco.drawdowns"}, ["BTC"],
            lambda s: [c for _, c in s] == ["Ecoinometrics — Fed tone", "Ecoinometrics — Drawdowns"]),
    # per-asset page: one step per asset; rich/cheap sends its chart + quintiles
    "vixdvol": ({"vixdvol.rich_cheap", "vixdvol.ranking"}, ["BTC", "ETH"],
                lambda s: [c for _, c in s][:3] == ["TradFi vol vs BTC DVOL — Rich/cheap vs TradFi",
                                                     "TradFi vol vs BTC DVOL — Rich/cheap quintiles",
                                                     "TradFi vol vs BTC DVOL — Predictor ranking"]
                and len(s) == 6 and "ETH" in s[-1][1]),
}


def check_page(pid):
    items, assets, ok = CASES[pid]
    steps, _ = blast.build_plan(items, assets)
    assert steps and all(st_.split(":")[0] == pid for st_ in steps), steps
    at = AppTest.from_file(str(ROOT / blast.PAGES[pid].path), default_timeout=600)
    from datetime import datetime, timezone
    at.session_state[blast.PLAN] = {"steps": steps, "items": sorted(items), "assets": assets,
                                    "skipped": [], "log": [], "started": ""}
    at.session_state[blast.FLAG] = steps[0]
    at.session_state[blast.STARTED] = datetime.now(timezone.utc)
    SENT.clear()
    t0 = time.time()
    at.run()
    errs = _errors(at)
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state[blast.FLAG] is None, at.session_state[blast.FLAG]
    summary = at.session_state[blast.SUMMARY]
    assert ok(list(SENT)), (SENT, summary, errs[:3], [w.value[:300] for w in at.warning][:3])
    banner = [s.value for s in list(at.success) + list(at.warning) if "Chart blast finished" in s.value]
    assert banner, "no summary banner"
    print(f"ok  {pid:10s} {time.time() - t0:5.1f}s sent={SENT} log={summary['log']}"
          + (f" page-errors={errs}" if errs else ""))


if __name__ == "__main__":
    which = sys.argv[1:] or ["home", "chain"] + list(CASES)
    failures = []
    for w in which:
        try:
            {"home": check_home, "chain": check_chain, "default_chain": check_default_chain}.get(w, lambda: check_page(w))()
        except Exception as exc:  # noqa: BLE001
            import traceback
            failures.append(w)
            print(f"ERR {w}: {exc}")
            traceback.print_exc()
    print("FAILED: " + ", ".join(failures) if failures else "all AppTest checks passed")
    sys.exit(1 if failures else 0)
