"""
Block Trades — OKX.

OKX RFQ block-trade tape (public, no auth) for BTC, ETH, SOL and XAU options.  Same page layout as 02_Block_Trades_-_Deribit.py; all logic lives in
lib/okx.py (fetch + normalise) and lib/block_flow.py (analytics, charts,
Telegram, page renderer).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib import block_flow, okx

block_flow.render_page(okx.VENUE)
