"""
Block Trades — Bybit.

Bybit option trades flagged isBlockTrade (public, no auth) for BTC, ETH, SOL and other listed coins.  Same page layout as 02_Block_Trades_-_Deribit.py; all logic lives in
lib/bybit.py (fetch + normalise) and lib/block_flow.py (analytics, charts,
Telegram, page renderer).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib import block_flow, bybit

block_flow.render_page(bybit.VENUE)
