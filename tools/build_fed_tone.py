"""Build data/fed_tone.csv: dictionary hawk/dove score per FOMC press conference.

Usage: python tools/build_fed_tone.py [first_year]

Shares ``lib.eco_refresh.build_fed_tone`` with the Ecoinometrics page. An
existing CSV is merged: a date that was not rescored keeps its previous row,
and a run that fetches nothing leaves the file untouched.

Not Ecoinometrics' proprietary index — a transparent lexicon proxy
(``lib/ecoinometrics.py``). Needs ``requests`` and ``pypdf``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eco_refresh import build_fed_tone


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # Historical default for a bare CLI invocation. The page and the GitHub
    # Action pass 2019 so they cover the bundled file. Merging means a 2022
    # start still keeps 2019–2021 rows already on disk.
    first = int(args[0]) if args else 2022
    result = build_fed_tone(first, log=print)
    print(result.message)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
