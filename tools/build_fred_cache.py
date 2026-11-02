"""Cache FRED series used by pages/13 into data/fred_<ID>.csv (Streamlit Cloud often
times out reaching fred.stlouisfed.org; lib/ecoinometrics.fred_series falls back to these).
Usage: python tools/build_fred_cache.py   (needs: requests, pandas)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.ecoinometrics import fred_series, DATA_DIR

SERIES = ["DFF", "NASDAQCOM", "CPIAUCSL", "DGS10"]
bad = 0
for sid in SERIES:
    try:
        df = fred_series(sid, live_only=True)
        df.to_csv(DATA_DIR / f"fred_{sid}.csv", index=False)
        print(sid, len(df), df["date"].iloc[-1].date())
    except Exception as e:
        bad += 1
        print("fail", sid, e)
sys.exit(1 if bad == len(SERIES) else 0)
