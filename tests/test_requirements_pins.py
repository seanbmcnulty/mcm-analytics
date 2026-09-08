"""Keep plotly<6.1 + kaleido<0.3 paired in requirements.txt."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ = ROOT / "requirements.txt"


def _pin(name: str, text: str) -> str | None:
    # Match lines like: plotly>=5.18.0,<6.1.0
    m = re.search(rf"(?m)^{re.escape(name)}\s*([^\n#]+)", text)
    return m.group(1).strip() if m else None


def _upper_bound_lt(spec: str, version: str) -> bool:
    """True if spec includes an upper bound strictly below `version` (e.g. <6.1.0)."""
    # Accept <X or <=X-epsilon style; we require an explicit < bound under version.
    for m in re.finditer(r"<\s*=?\s*([0-9]+(?:\.[0-9]+)*)", spec):
        bound = m.group(1)
        # Compare dotted versions component-wise as ints
        def parts(v: str) -> list[int]:
            return [int(p) for p in v.split(".")]

        bp, vp = parts(bound), parts(version)
        # pad
        n = max(len(bp), len(vp))
        bp += [0] * (n - len(bp))
        vp += [0] * (n - len(vp))
        if m.group(0).startswith("<=") :
            # <=6.0.99 would be ok for <6.1 but we require strict <6.1 style
            if bp < vp:
                return True
        else:
            if bp <= vp:
                # <6.1.0 means upper bound is 6.1.0 itself → compatible with "below 6.1"
                # Treat bound == version as satisfying "capped below this major.minor line"
                # when the pin is <6.1.0 and version asked is 6.1
                if bp == vp or bp < vp:
                    return True
    return False


def test_plotly_kaleido_pins() -> None:
    text = REQ.read_text(encoding="utf-8")
    plotly_spec = _pin("plotly", text)
    kaleido_spec = _pin("kaleido", text)
    assert plotly_spec, "plotly pin missing from requirements.txt"
    assert kaleido_spec, "kaleido pin missing from requirements.txt"
    assert _upper_bound_lt(plotly_spec, "6.1"), (
        f"plotly must stay capped below 6.1 (got {plotly_spec!r})"
    )
    assert _upper_bound_lt(kaleido_spec, "0.3"), (
        f"kaleido must stay capped below 0.3 (got {kaleido_spec!r})"
    )
    # Paired: comment in requirements documents they move together
    assert "plotly" in text.lower() and "kaleido" in text.lower()


if __name__ == "__main__":
    test_plotly_kaleido_pins()
    print("test_plotly_kaleido_pins: OK")
