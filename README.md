# MCM Analytics

Crypto derivatives analytics powered by **Deribit public API data**. No authentication required — all data comes from Deribit's public endpoints.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run Home.py
```

The app will open at `http://localhost:8020`.

## Pages

| # | Page | Description |
|---|------|-------------|
| 1 | **MCM Bot** | Full markets bot: vol surface, term structure, skew, trade flow, RV |
| 2 | **Block Trades** | Deribit block trade analysis with Greeks and Telegram reporting |
| 6 | **Realized Vol** | Multi-estimator RV matrix (BTC/ETH/SOL/HYPE) |
| 7 | **Regime Identifier** | Vol regime classification (GARCH + implied vol) |
| 8 | **Spot-Vol Correlation** | DVOL vs spot price analysis (BTC/ETH) |
| 10 | **Macro Event Impact** | CPI/FOMC/NFP surprise z-scores + price reactions |

## Asset Universe

| Asset | Deribit Currency | DVOL Available | Notes |
|-------|-----------------|----------------|-------|
| BTC | BTC | ✅ | Inverse (coin-margined) |
| ETH | ETH | ✅ | Inverse (coin-margined) |
| SOL | USDC | ❌ | Linear (USDC-margined) |
| HYPE | USDC | ❌ | Linear (USDC-margined) |

## Telegram Integration

To enable Telegram blast (chart/report sharing):

1. Create a Telegram bot via [@BotFather](https://t.me/BotFather)
2. Get your chat ID (message [@userinfobot](https://t.me/userinfobot))
3. Configure credentials:

**Option A: Streamlit secrets** (recommended for Streamlit Cloud)

Create `.streamlit/secrets.toml`:
```toml
[telegram]
bot_token = "YOUR_BOT_TOKEN"
chat_id = "YOUR_CHAT_ID"
```

**Option B: Config file**

Create `~/.config/mcm-analytics/secrets.toml`:
```toml
[telegram]
bot_token = "YOUR_BOT_TOKEN"
chat_id = "YOUR_CHAT_ID"
```

**Option C: Environment variables**
```bash
export TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"
export TELEGRAM_CHAT_ID="YOUR_CHAT_ID"
```

## Limitations

1. **No historical vol surface on first run** — DVOL history available for BTC/ETH. Over time you can store snapshots to build historical data for z-scores and percentiles.
2. **Asset universe: BTC, ETH, SOL, HYPE only** — These are Deribit's currently listed option currencies.
3. **No DVOL for SOL/HYPE** — Those assets get live ATM IV from the option chain but no DVOL history.
4. **Funding rates: Deribit only** — No cross-exchange comparison.
5. **RV asset list: 4 assets** — Only Deribit-listed currencies with tradingview OHLC data.

## Data Sources

All data comes from Deribit's **public** API endpoints:
- Index prices (`/public/get_index_price`)
- Option chains (`/public/get_book_summary_by_currency`)
- DVOL index (`/public/get_volatility_index_data`)
- Trade history (`/public/get_last_trades_by_currency_and_time`)
- Funding rates (`/public/get_funding_rate_history`)
- OHLC/Tradingview (`/public/get_tradingview_chart_data`)
- Instruments (`/public/get_instruments`)

No API key or authentication is needed.

## Requirements

- Python 3.10+
- ~150MB disk space for dependencies
- Internet connection (fetches live data from Deribit)

## Streamlit Cloud Deployment

This app is ready to deploy on [Streamlit Community Cloud](https://share.streamlit.io):

1. Push this repo to GitHub
2. Go to share.streamlit.io → "New app"
3. Select the repo, branch `main`, main file `Home.py`
4. Add Telegram secrets in the app settings (Advanced → Secrets)
5. Deploy!

## Project Structure

```
mcm-analytics/
├── Home.py                        # Landing page
├── requirements.txt               # Python dependencies
├── .streamlit/config.toml         # Streamlit config (port, theme)
├── secrets.toml.example           # Template for secrets
├── lib/
│   ├── deribit.py                 # Deribit public API client
│   ├── telegram.py                # Telegram integration
│   ├── vol_math.py                # Vol math (RV estimators, BS)
│   ├── instruments.py             # Instrument name parsing
│   └── constants.py               # Shared constants
├── pages/
│   ├── 01_MCM_Bot.py
│   ├── 02_Block_Trades.py
│   ├── 06_Realized_Vol.py
│   ├── 07_Regime_Identifier.py
│   ├── 08_Spot_Vol_Correlation.py
│   └── 10_Macro_Event_Impact.py
└── data/
    └── macro_events_calendar.csv  # Bundled macro event calendar
```

## License

Private — for internal use only.
