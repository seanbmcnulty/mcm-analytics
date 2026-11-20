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
from lib.auth import require_password  # noqa: E402
require_password()  # site password gate: before anything renders

from lib import blast, block_flow, freshness, derive

# Pick up a changed lib.block_flow / lib.blast after a Cloud redeploy (old code
# stays in memory otherwise, and the Home chart blast hook would be missing).
freshness.reload_if_changed(blast, block_flow)

block_flow.render_page(derive.VENUE)
