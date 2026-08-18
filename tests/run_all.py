"""Exercise every bot command against the synthetic feed."""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import fake_deribit
from lib import deribit

for name in dir(fake_deribit):
    if name.startswith("__"):
        continue
    if hasattr(deribit, name) and callable(getattr(fake_deribit, name, None)):
        setattr(deribit, name, getattr(fake_deribit, name))

from lib import commands as cmdreg   # noqa: E402

ASSETS = sys.argv[1:] or ["BTC", "SOL"]
failures = []
for asset in ASSETS:
    for cn in cmdreg.COMMAND_NAMES:
        dte = 30 if cn in cmdreg.EXPIRY_COMMANDS else None
        try:
            fig, df, text = cmdreg.run_command(asset, "/" + cn, expiry_target_days=dte)
        except Exception:
            failures.append((asset, cn, "RAISED\n" + traceback.format_exc()))
            continue
        nfigs = len(fig) if isinstance(fig, list) else (1 if fig is not None else 0)
        nrows = 0 if df is None else len(df)
        is_err = nfigs == 0 and nrows == 0 and bool(text)
        status = "ERR " if is_err else "ok  "
        detail = (text or "")[:90] if is_err else f"figs={nfigs} rows={nrows}"
        print(f"{status}{asset:5s} /{cn:34s} {detail}")
        if is_err:
            failures.append((asset, cn, text))

print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} command(s) returned no output:")
    for a, c, t in failures:
        print(f"  - {a} /{c}: {str(t)[:200]}")
    sys.exit(1)
print("All commands produced output.")
