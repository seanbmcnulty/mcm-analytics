"""
Block Trades — Binance.

Binance Options public block-trade tape (eapi blockTrades, no auth) for BTC, ETH, SOL, BNB and other
listed options.  Same page layout as 02_Block_Trades_-_Deribit.py; all logic lives in
lib/binance.py (fetch + normalise) and lib/block_flow.py (analytics, charts, Telegram, page renderer).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib import binance, block_flow

block_flow.render_page(binance.VENUE)
