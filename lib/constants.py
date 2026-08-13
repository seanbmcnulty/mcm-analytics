"""
MCM Analytics — shared constants and asset configuration.
"""

# Asset configuration: all Deribit-listed option currencies
ASSET_CONFIG = {
    "BTC": {
        "style": "inverse",
        "deribit_ccy": "BTC",
        "deribit_prefix": "BTC",
        "index": "btc_usd",
        "perp": "BTC-PERPETUAL",
        "has_dvol": True,
        "min_block": 12.5,
        "contract_size": 1.0,
        "price_dp": 0,
    },
    "ETH": {
        "style": "inverse",
        "deribit_ccy": "ETH",
        "deribit_prefix": "ETH",
        "index": "eth_usd",
        "perp": "ETH-PERPETUAL",
        "has_dvol": True,
        "min_block": 125.0,
        "contract_size": 1.0,
        "price_dp": 0,
    },
    "SOL": {
        "style": "linear",
        "deribit_ccy": "USDC",
        "deribit_prefix": "SOL_USDC",
        "index": "sol_usdc",
        "perp": "SOL_USDC-PERPETUAL",
        "has_dvol": False,
        "min_block": 5000.0,
        "contract_size": 10,
        "price_dp": 2,
    },
    "HYPE": {
        "style": "linear",
        "deribit_ccy": "USDC",
        "deribit_prefix": "HYPE_USDC",
        "index": "hype_usdc",
        "perp": "HYPE_USDC-PERPETUAL",
        "has_dvol": False,
        "min_block": 5000.0,
        "contract_size": 10,
        "price_dp": 2,
    },
}

ASSETS = list(ASSET_CONFIG.keys())

# Color scheme for assets
ASSET_COLORS = {
    "BTC": "#f7931a",
    "ETH": "#627eea",
    "SOL": "#9945ff",
    "HYPE": "#00d1a0",
}

# Plotly dark theme template
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#fafafa"),
    margin=dict(l=40, r=20, t=40, b=30),
)
