"""Home chart blast (lib/blast.py): registry, plans, the step state machine,
saved selection, and that every page carries its hook. No Telegram sends."""
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import blast  # noqa: E402

LEGACY_STEPS = ["mcm", "bt_deribit", "tbrv:BTC", "tbrv:ETH", "spot_vol"]


def _src(path):
    return (ROOT / path).read_text(encoding="utf-8")


# ---------------------------------------------------------------- registry
def test_keys_unique_and_pages_known():
    keys = [it.key for it in blast.ITEMS]
    assert len(keys) == len(set(keys))
    for it in blast.ITEMS:
        assert it.page in blast.PAGES or it.page == "bt", it


def test_mcm_items_cover_every_command():
    from lib import commands as cmdreg
    mcm = {it.short for it in blast.ITEMS if it.page == "mcm"}
    assert mcm == set(cmdreg.COMMAND_NAMES) | {"vol_surfaces"}


def test_block_trade_chart_keys_match_pages():
    from lib import block_flow
    reg = tuple(it.short for it in blast.ITEMS if it.page == "bt" and it.short not in block_flow.OWN_CHART_KEYS)
    assert reg == block_flow.CHART_KEYS
    own = tuple(it.short for it in blast.ITEMS if it.page == "bt" and it.short in block_flow.OWN_CHART_KEYS)
    assert own == block_flow.OWN_CHART_KEYS
    assert not any(it.default for it in blast.ITEMS if it.key == "bt.chart.won")   # private data: opt-in
    m = re.search(r"BLAST_CHART_KEYS = \((.*?)\)", _src("pages/02_Block_Trades_-_Deribit.py"), re.S)
    assert tuple(re.findall(r'"(\w+)"', m.group(1))) == reg


def test_page_specific_keys_exist_in_pages():
    for page, path in (("tbrv", "pages/06_Time_Based_Realized_Vol.py"),
                       ("regime", "pages/07_Regime_Identifier.py"),
                       ("spot_vol", "pages/08_Spot_Vol_Correlation.py"),
                       ("macro", "pages/10_Macro_Event_Impact.py")):
        src = _src(path)
        for it in blast.ITEMS:
            if it.page == page and it.short != "cross_asset":
                assert f'"{it.short}"' in src, (path, it.short)
    assert '"cross_asset"' in _src("pages/07_Regime_Identifier.py")
    eco = _src("pages/13_Ecoinometrics_Charts.py")
    for it in blast.ITEMS:
        if it.page == "eco":
            assert f'"{it.short}"' in eco, it.short


def test_ibit_sections_match_page():
    src = _src("pages/12_IBIT.py")
    titles = set(re.findall(r'sec\(\s*"([^"]+)"', src))
    ib = {it.short for it in blast.ITEMS if it.page == "ibit"}
    assert set(blast.IBIT_SECTIONS) == ib
    for k, secs in blast.IBIT_SECTIONS.items():
        for t in secs:
            assert t in titles, t
    # every section that draws a chart is selectable
    charted = set()
    cur = None
    for line in src.splitlines():
        m = re.search(r'sec\(\s*"([^"]+)"', line)
        if m:
            cur = m.group(1)
        if re.search(r"\bshow\(", line) and not line.lstrip().startswith("def ") and cur:
            charted.add(cur)
    assert charted <= set(blast.IBIT_ITEM_BY_SECTION), charted - set(blast.IBIT_ITEM_BY_SECTION)


def test_every_page_has_a_hook():
    for pid, meta in blast.PAGES.items():
        if pid.startswith("bt_") and pid != "bt_deribit":
            src = _src(meta.path)
            assert "block_flow.render_page" in src
            assert re.search(r"freshness\.reload_if_changed\((\w+, )*blast, block_flow\)", src), pid
            continue
        src = _src(meta.path)
        assert f'blast.active_step("{pid}")' in src, pid
        assert "blast.finish(" in src, pid
        assert "reload_if_changed(blast" in src, pid
    bf = _src("lib/block_flow.py")
    assert 'blast.active_step(page_id)' in bf and "blast.finish(" in bf


def test_home_uses_selector_and_label():
    src = _src("Home.py")
    assert "blast.BUTTON_LABEL" in src
    assert "Refresh BTC/ETH & send to Telegram" in blast.BUTTON_LABEL


# ---------------------------------------------------------------- plans
def test_default_plan_is_the_old_pipeline():
    steps, skipped = blast.build_plan(blast.default_items(), blast.DEFAULT_ASSETS)
    assert steps == LEGACY_STEPS and skipped == []
    # default selection == every MCM command, all Deribit charts, TBRV, Spot Vol
    pages = {blast.ITEM_BY_KEY[k].page for k in blast.default_items()}
    assert pages == {"mcm", "bt", "bt_deribit", "tbrv", "spot_vol"}
    assert blast.items_for_page(blast.default_items(), "bt_okx") == set()


def test_plan_orders_and_skips():
    items = {"eco.fed_tone", "regime.gauge", "mcm.vol_run", "ibit.gex", "ibit.smile",
             "bt.venue.okx", "bt.chart.scatter", "fng.summary"}
    steps, skipped = blast.build_plan(items, ["SOL", "HYPE"])
    assert steps == ["mcm", "bt_okx", "fng:SOL", "fng:HYPE", "ibit:0", "ibit:3", "eco"]
    assert len(skipped) == 1 and skipped[0].startswith("Regime Identifier")
    steps, _ = blast.build_plan(items, ["ETH", "BTC"])
    assert steps[:4] == ["mcm", "bt_okx", "regime", "fng:BTC"]
    # a venue without any chart type is nothing to send
    assert blast.build_plan({"bt.venue.okx"}, ["BTC"])[0] == []
    assert blast.build_plan(set(), ["BTC"])[0] == []


# ---------------------------------------------------------------- state machine
class _Switch(BaseException):
    def __init__(self, target):
        self.target = target


class _FakeSt:
    def __init__(self):
        self.session_state = {}
        self.shown = []

    def switch_page(self, path):
        raise _Switch(path)

    def rerun(self):
        raise _Switch("rerun")

    def stop(self):
        raise _Switch("stop")

    def success(self, body):
        self.shown.append(("success", body))

    def warning(self, body):
        self.shown.append(("warning", body))

    def info(self, body):
        self.shown.append(("info", body))


def _with_fake(fn):
    import lib.telegram as tg
    real_st, real_cfg = blast.st, tg.is_configured
    fake = _FakeSt()
    blast.st = fake
    tg.is_configured = lambda: True
    try:
        fn(fake, tg)
    finally:
        blast.st = real_st
        tg.is_configured = real_cfg


def _run(fake, page):
    step = blast.active_step(page)
    assert step is not None, (page, fake.session_state.get(blast.FLAG))
    try:
        blast.finish(step, 3, [] if page != "spot_vol" else ["x chart"])
    except _Switch as sw:
        return sw.target
    return None


def test_state_machine_runs_default_plan():
    def body(fake, tg):
        first = blast.start(blast.default_items(), blast.DEFAULT_ASSETS)
        assert first == "pages/01_MCM_Bot.py"
        assert blast.active_step("spot_vol") is None          # not this page's turn
        assert _run(fake, "mcm") == "pages/02_Block_Trades_-_Deribit.py"
        assert _run(fake, "bt_deribit") == "pages/06_Time_Based_Realized_Vol.py"
        step = blast.active_step("tbrv")
        assert step.asset == "BTC" and blast.assets_for(step) == ["BTC"]
        assert _run(fake, "tbrv") == "rerun"                   # same page, next asset
        assert blast.active_step("tbrv").asset == "ETH"
        assert _run(fake, "tbrv") == "pages/08_Spot_Vol_Correlation.py"
        step = blast.active_step("spot_vol")
        assert blast.assets_for(step) == ["BTC", "ETH"] and "summary" in blast.selected(step)
        assert _run(fake, "spot_vol") is None                  # last step: summary, no switch
        assert fake.session_state[blast.FLAG] is None
        assert blast.PLAN not in fake.session_state
        summary = fake.session_state[blast.SUMMARY]
        sent, failed, lines = blast.summary_lines(summary)
        assert sent == 15 and failed == 1 and len(lines) == 5
        assert fake.shown[-1][0] == "warning" and "x chart" in fake.shown[-1][1]
    _with_fake(body)


def test_legacy_flag_and_unconfigured_telegram():
    def body(fake, tg):
        fake.session_state[blast.FLAG] = "tbrv_eth"            # a run from before the deploy
        step = blast.active_step("tbrv")
        assert step is not None and step.asset == "ETH" and step.id == "tbrv:ETH"
        assert _run(fake, "tbrv") == "pages/08_Spot_Vol_Correlation.py"
        fake.session_state[blast.FLAG] = "nonsense"            # unknown flag never sticks
        assert blast.current_step() is None and fake.session_state[blast.FLAG] is None
        blast.start({"regime.gauge"}, ["BTC"])
        tg.is_configured = lambda: False
        assert blast.active_step("regime") is None and not blast.running()
    _with_fake(body)


def test_abort_and_stop_move_on():
    def body(fake, tg):
        blast.start({"regime.gauge", "macro.summary"}, ["BTC"])
        try:
            blast.stop("regime", "fetch failed")
        except _Switch as sw:
            assert sw.target == "pages/10_Macro_Event_Impact.py"
        assert fake.session_state[blast.PLAN]["log"][0]["failed"] == ["fetch failed"]
        blast.abort("regime", "ignored: not this page's turn")
        assert blast.current_step().page == "macro"
    _with_fake(body)


def test_step_timeout_is_per_step():
    def body(fake, tg):
        blast.start(blast.default_items(), blast.DEFAULT_ASSETS)
        t0 = fake.session_state[blast.STARTED]
        _run(fake, "mcm")
        assert fake.session_state[blast.STARTED] >= t0
    _with_fake(body)


# ---------------------------------------------------------------- saved selection
def test_save_and_load_selection():
    with tempfile.TemporaryDirectory() as d:
        old = os.environ.get("MCM_BLAST_SELECTION_PATH")
        os.environ["MCM_BLAST_SELECTION_PATH"] = str(Path(d) / "sel.json")
        try:
            assert blast.load_saved() is None
            assert blast.save({"eco.fed_tone", "mcm.vol_run", "gone.key"}, ["ETH", "SOL", "XRP"])
            items, assets = blast.load_saved()
            assert items == {"eco.fed_tone", "mcm.vol_run"} and assets == ["ETH", "SOL"]
            blast.forget_saved()
            assert blast.load_saved() is None
        finally:
            if old is None:
                os.environ.pop("MCM_BLAST_SELECTION_PATH", None)
            else:
                os.environ["MCM_BLAST_SELECTION_PATH"] = old


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
