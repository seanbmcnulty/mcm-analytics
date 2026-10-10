"""
Block Trades — Derive.

Derive (derive.xyz) RFQ / block option flow for BTC, ETH, SOL and HYPE, built on
the public Derive API (no auth).  Same page layout as 02_Block_Trades_-_Deribit.py;
all logic lives in lib/derive.py (fetch + normalise) and lib/block_flow.py
(analytics, charts, Telegram, page renderer).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib import block_flow, derive

block_flow.render_page(derive.VENUE)
