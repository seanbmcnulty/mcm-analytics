"""
Block Trades — Paradex.

Paradex block-trade tape (trade_type=BLOCK_TRADE, public, no auth) for BTC, ETH,
SOL and HYPE options.  Same page layout as 02_Block_Trades_-_Deribit.py; all
logic lives in lib/paradex.py (fetch + normalise) and lib/block_flow.py
(analytics, charts, Telegram, page renderer).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib import block_flow, paradex

block_flow.render_page(paradex.VENUE)
