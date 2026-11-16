"""Manual check (needs network): run the Ecoinometrics page with lib modules
from an older commit already in sys.modules, the way Streamlit Cloud does
after a git pull. Usage: python tests/stale_module_sim.py <dir with old lib/*.py>
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import lib  # noqa: E402

OLD = Path(sys.argv[1])
for name in ("fx_style", "ecoinometrics", "eco_style", "eco_charts", "eco_refresh"):
    real = ROOT / "lib" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"lib.{name}", real)  # reload() reads the real file
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"lib.{name}"] = mod
    setattr(lib, name, mod)
    exec(compile((OLD / "lib" / f"{name}.py").read_text(), str(real), "exec"), mod.__dict__)
assert not hasattr(sys.modules["lib.eco_charts"], "figure_nasdaq_sma"), "old modules not loaded"

from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file(str(ROOT / "pages" / "13_Ecoinometrics_Charts.py"), default_timeout=300)
at.run()
print("exceptions:", [str(e.value)[:300] for e in at.exception])
print("errors:", [str(e.value)[:200] for e in at.error])
print("charts:", len(at.get("plotly_chart")))
print("reloaded:", sys.modules["lib.eco_charts"].__dict__.get("figure_nasdaq_sma") is not None)
sys.exit(1 if at.exception or at.error else 0)
