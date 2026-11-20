"""
Block Trades — OKX.

OKX RFQ block-trade tape (public, no auth) for BTC, ETH, SOL and XAU options.  Same page layout as 02_Block_Trades_-_Deribit.py; all logic lives in
lib/okx.py (fetch + normalise) and lib/block_flow.py (analytics, charts,
Telegram, page renderer).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib import auth as _auth, freshness as _fresh  # noqa: E402
_fresh.reload_if_changed(_auth)  # pick up a changed gate after a Cloud redeploy
from lib.auth import require_password  # noqa: E402
require_password()  # site password gate: before anything renders

from lib import blast, block_flow, freshness, own_trades, okx

# Pick up a changed lib.block_flow / lib.blast after a Cloud redeploy (old code
# stays in memory otherwise, and the Home chart blast hook would be missing).
freshness.reload_if_changed(own_trades, okx, blast, block_flow)

block_flow.render_page(okx.VENUE)
