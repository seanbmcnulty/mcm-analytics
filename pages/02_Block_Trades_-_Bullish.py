"""
Block Trades — Bullish.

Bullish BTC option tape (public, no auth); blocks are trades at/above a minimum size.  Same page layout as 02_Block_Trades_-_Deribit.py; all logic lives in
lib/bullish.py (fetch + normalise) and lib/block_flow.py (analytics, charts,
Telegram, page renderer).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib import block_flow, bullish

block_flow.render_page(bullish.VENUE)
