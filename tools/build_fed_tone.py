"""Build data/fed_tone.csv: dictionary hawk/dove score per FOMC press conference.
Usage: python tools/build_fed_tone.py [first_year]   (needs: requests, pypdf)
Not Ecoinometrics' proprietary index — a transparent lexicon proxy (lib/ecoinometrics.py)."""
import io, re, sys
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.ecoinometrics import chair_text, tone_score, DATA_DIR
from pypdf import PdfReader

UA = {"User-Agent": "Mozilla/5.0 (compatible; mcm-analytics)"}
BASE = "https://www.federalreserve.gov"
first = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
urls = {}
for y in range(first, pd.Timestamp.utcnow().year + 1):
    page = "/monetarypolicy/fomccalendars.htm" if y >= pd.Timestamp.utcnow().year - 4 else f"/monetarypolicy/fomchistorical{y}.htm"
    try:
        html = requests.get(BASE + page, headers=UA, timeout=30).text
    except Exception as e:
        print("skip", y, e); continue
    for m in re.finditer(r"presconf(\d{8})\.(?:htm|pdf)", html):
        urls[m.group(1)] = f"{BASE}/mediacenter/files/FOMCpresconf{m.group(1)}.pdf"
rows = []
for ds, u in sorted(urls.items()):
    if int(ds[:4]) < first: continue
    try:
        r = requests.get(u, headers=UA, timeout=60); r.raise_for_status()
        txt = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(r.content)).pages)
        rows.append({"date": pd.to_datetime(ds), **tone_score(chair_text(txt))})
        print(ds, rows[-1]["score"])
    except Exception as e:
        print("fail", ds, e)
if not rows:
    sys.exit("no transcripts fetched; existing data/fed_tone.csv left untouched")
out = pd.DataFrame(rows).sort_values("date")
out.to_csv(DATA_DIR / "fed_tone.csv", index=False)
print("wrote", len(out))
