"""
Block Trades — Deribit public block trade analysis.

Replicates the Deribit Block Trades page:
- Scatter Plot overlaid with Perpetual Price (yaxis2) and DVOL Index (yaxis3)
- Strike vs Expiry Bubble Maps (Log scale)
- Net Flow Heatmaps & Gross Volume Heatmaps
- Cumulative Volume Breakdown (Buys/Sells/Calls/Puts)
- Expiry & Structure Statistics Tables with Dollar Greeks & Theo Edge
- Multi-Asset Tabs + ALL Overview Grid
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import html
import concurrent.futures
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from scipy.stats import norm
import streamlit as st

from lib import cache as cache_lib
from lib import fx_style
from lib import telegram
from lib.constants import TTL_SHORT, TTL_MEDIUM

# ---------------------------------------------------------------------------
# Page Config & Styles
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Block Trades — Deribit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1E4D7A 0%, #3790C7 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: #fff;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .asset-header {
        background-color: #1E4D7A;
        padding: 0.5rem;
        border-radius: 0.5rem;
        color: #fff;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>📊 BLOCK TRADES - DERIBIT</h1></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants & Layout Settings
# ---------------------------------------------------------------------------
SGT = timezone(timedelta(hours=8))
BACKGROUND_COLOR = '#FFFFFF'
TEXT_COLOR = '#000000'
SECONDARY_COLOR = '#3790C7'
MARKER_SIZE_CAP = 80

DEFAULT_MIN_SIZES = {
    'BTC': 12.5,
    'ETH': 125.0,
    'SOL_USDC': 125.0,
    'XRP_USDC': 1000.0,
    'TRX_USDC': 100.0,
    'AVAX_USDC': 10.0,
    'HYPE_USDC': 10.0,
}

ASSETS = list(DEFAULT_MIN_SIZES.keys())

# Deribit groups every USDC-settled option under a single "USDC" currency
# feed (SOL/XRP/TRX/AVAX/HYPE options are filtered out of it client-side by
# instrument prefix) - so there are really only 3 distinct trades feeds to
# fetch, not 7. Fetching by currency once and reusing it across assets cuts
# out 4 redundant identical API calls per page load.
CURRENCY_MAP = {
    'BTC': 'BTC', 'ETH': 'ETH', 'SOL_USDC': 'USDC', 'XRP_USDC': 'USDC',
    'TRX_USDC': 'USDC', 'AVAX_USDC': 'USDC', 'HYPE_USDC': 'USDC',
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BACKGROUND_COLOR,
    plot_bgcolor=BACKGROUND_COLOR,
    font=dict(color=TEXT_COLOR),
    margin=dict(l=40, r=40, t=40, b=40)
)

# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
def current_anchor_sgt(now_sg):
    seven_am = now_sg.replace(hour=7, minute=0, second=0, microsecond=0)
    seven_pm = now_sg.replace(hour=19, minute=0, second=0, microsecond=0)
    if now_sg >= seven_pm:
        return seven_pm
    elif now_sg >= seven_am:
        return seven_am
    else:
        return (now_sg - timedelta(days=1)).replace(hour=19, minute=0, second=0, microsecond=0)

with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    
    lookback_mode = st.radio("Time Window", ["Last 24 Hours", "Last 12 Hours", "Custom Start Time"], index=1)
    now_sg = datetime.now(SGT)
    
    if lookback_mode == "Last 24 Hours":
        start_datetime_sgt = now_sg - timedelta(hours=24)
    elif lookback_mode == "Last 12 Hours":
        start_datetime_sgt = now_sg - timedelta(hours=12)
    else:
        default_anchor = current_anchor_sgt(now_sg)
        start_date = st.date_input("Start Date", value=default_anchor.date())
        start_time = st.time_input("Start Time (SGT)", value=default_anchor.time())
        start_datetime_sgt = datetime.combine(start_date, start_time).replace(tzinfo=SGT)

    start_datetime_utc = start_datetime_sgt.astimezone(timezone.utc)

    st.subheader("Minimum Block Sizes")
    min_sizes = {}
    for asset, default_val in DEFAULT_MIN_SIZES.items():
        clean_name = asset.replace('_USDC', '')
        min_sizes[asset] = st.number_input(
            f"{clean_name} Min Size",
            min_value=0.1,
            value=float(default_val),
            step=1.0 if default_val >= 10 else 0.1
        )

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    if auto_refresh:
        time.sleep(60)
        st.rerun()
    cache_lib.render_refresh_button()

    st.divider()
    st.subheader("Telegram")
    if telegram.is_configured():
        st.success("Telegram ready")
    else:
        st.warning("Telegram not configured")
        st.caption(telegram.config_status())

# ---------------------------------------------------------------------------
# API Fetching Helpers
# ---------------------------------------------------------------------------
def _get_json_with_retry(url, params=None, timeout=10, max_retries=2, pace=0.15):
    """GET + .json() with a couple of retries and real 429 backoff.

    Previously every call here was try/except-swallowed with zero retries, so
    a single transient timeout or rate-limit response silently turned into an
    empty DataFrame downstream (e.g. the perp line vanishing from a chart)
    with no sign of why. This keeps the same "never raise, just return None
    on failure" contract callers already rely on, but gives a request an
    actual chance to succeed first.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                wait = 1.0
                try:
                    wait = float(r.headers.get('Retry-After', wait))
                except (TypeError, ValueError):
                    pass
                time.sleep(min(max(wait, 0.5), 3.0))
                continue
            r.raise_for_status()
            if pace:
                time.sleep(pace)  # Pace API limit
            return r.json()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(0.3 * (attempt + 1))
    if last_exc is not None:
        print(f"Deribit request failed after retries ({url}): {last_exc}")
    return None

@st.cache_data(ttl=TTL_SHORT, show_spinner=False)
def get_current_spot(asset: str) -> float:
    index_map = {
        'BTC': 'btc_usd', 'ETH': 'eth_usd', 'SOL_USDC': 'sol_usdc',
        'XRP_USDC': 'xrp_usdc', 'TRX_USDC': 'trx_usdc',
        'AVAX_USDC': 'avax_usdc', 'HYPE_USDC': 'hype_usdc'
    }
    index_name = index_map.get(asset, f"{asset.lower()}_usd")
    url = 'https://deribit.com/api/v2/public/get_index_price'
    data = _get_json_with_retry(url, params={'index_name': index_name}, timeout=5)
    if not data:
        return 0.0
    try:
        return float(data['result']['index_price'])
    except (KeyError, TypeError, ValueError):
        return 0.0

@st.cache_data(ttl=TTL_MEDIUM, show_spinner=False)
def fetch_trades_by_currency(currency: str, start_dt_utc: datetime) -> pd.DataFrame:
    """Fetch+parse the raw trades feed for one Deribit currency (BTC/ETH/USDC).

    Deribit exposes all USDC-settled option flow (SOL, XRP, TRX, AVAX, HYPE)
    under a single "USDC" currency - the individual assets are filtered out
    of this shared feed by instrument prefix in fetch_public_trades() below.
    Fetching here once per currency (cached) instead of once per asset avoids
    5 redundant, identical Deribit calls per page load.

    Deribit caps each response at 1000 trades and sets `has_more=True` when
    there are more in range. The old code made a single call and silently
    dropped everything past the first 1000 - for a busy 12/24h BTC or ETH
    window (or the combined USDC feed across 5 assets), that's enough
    trading to blow past 1000 and quietly lose a large chunk of the window.
    This walks forward page by page (advancing start_timestamp to just past
    the last trade's timestamp each time, since sorting=asc) until Deribit
    reports no more pages or a safety cap is hit.
    """
    start_ms = int(start_dt_utc.timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    url = "https://www.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
    all_trades = []
    cursor_ms = start_ms
    max_pages = 20  # 20 x 1000 = 20k trades/currency/load - comfortably above anything realistic
    for _page in range(max_pages):
        params = {
            'currency': currency, 'kind': 'option',
            'start_timestamp': cursor_ms, 'end_timestamp': end_ms,
            'count': 1000, 'sorting': 'asc',
        }
        data = _get_json_with_retry(url, params=params, timeout=10)
        if not data or 'result' not in data:
            break
        result = data['result']
        page_trades = result.get('trades', [])
        if not page_trades:
            break
        all_trades.extend(page_trades)
        if not result.get('has_more'):
            break
        last_ts = page_trades[-1].get('timestamp')
        if last_ts is None:
            break
        next_cursor = int(last_ts) + 1  # +1ms so the boundary trade isn't fetched twice
        if next_cursor <= cursor_ms:
            break  # cursor didn't move forward - stop rather than loop forever
        cursor_ms = next_cursor

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)

    # The +1ms page boundary above is a "should never overlap" guard, not a
    # guarantee - if two trades ever land in the same millisecond, drop the
    # duplicate by trade_id rather than double-counting it.
    if 'trade_id' in df.columns:
        df = df.drop_duplicates(subset='trade_id', keep='last')

    # Parse details
    if "timestamp" in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(SGT)

    if "amount" in df.columns:
        df['abs_amount'] = df['amount'].abs()
        df['amount'] = np.where(df['direction'] == 'buy', df['abs_amount'], -df['abs_amount'])

    # Parse strikes and expiries
    if "instrument_name" in df.columns:
        df['strike'] = pd.to_numeric(df['instrument_name'].str.extract(r'-(\d+(?:\.\d+)*)-')[0], errors='coerce')
        df['expiry'] = pd.to_datetime(df['instrument_name'].str.extract(r'^[^-]*-([0-9]{1,2}[A-Z]{3}[0-9]{2})-')[0], format='%d%b%y', errors='coerce')
        df['option_type'] = df['instrument_name'].str.extract(r'-(C|P)$')[0]

    if 'mark_price' not in df.columns and 'price' in df.columns:
        df['mark_price'] = df['price']

    return df.dropna(subset=['strike', 'expiry']).reset_index(drop=True)

def fetch_public_trades(asset: str, start_dt_utc: datetime) -> pd.DataFrame:
    """Per-asset view over the shared per-currency trades feed."""
    currency = CURRENCY_MAP.get(asset, asset)
    df = fetch_trades_by_currency(currency, start_dt_utc)
    if df.empty:
        return df

    # Filter by instrument prefix for USDC settled (SOL, XRP, AVAX, HYPE, TRX)
    if "_USDC" in asset and "instrument_name" in df.columns:
        prefix = f"{asset}-"
        df = df[df['instrument_name'].str.startswith(prefix)].copy()

    return df.reset_index(drop=True) if not df.empty else df

@st.cache_data(ttl=TTL_MEDIUM, show_spinner=False)
def fetch_historical_spot(asset: str, start_dt_utc: datetime) -> pd.DataFrame:
    perp_map = {
        'BTC': 'BTC-PERPETUAL', 'ETH': 'ETH-PERPETUAL', 'SOL_USDC': 'SOL_USDC-PERPETUAL',
        'XRP_USDC': 'XRP_USDC-PERPETUAL', 'TRX_USDC': 'TRX_USDC-PERPETUAL',
        'AVAX_USDC': 'AVAX_USDC-PERPETUAL', 'HYPE_USDC': 'HYPE_USDC-PERPETUAL'
    }
    instrument = perp_map.get(asset, f"{asset}-PERPETUAL")
    start_ms = int(start_dt_utc.timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    params = {
        'instrument_name': instrument,
        'start_timestamp': start_ms,
        'end_timestamp': end_ms,
        'resolution': '5'  # 5-minute candles to safely avoid the 1000 limit
    }
    data = _get_json_with_retry(url, params=params, timeout=10)
    if not data:
        return pd.DataFrame()
    chart = data.get('result', {})
    if chart.get('status') == 'ok' and chart.get('ticks') and chart.get('close'):
        # NB: chart['ticks'] is a plain list, so pd.to_datetime(...) here
        # returns a DatetimeIndex, not a Series - it has no `.dt` accessor
        # (that only exists on Series). The old `.dt.tz_convert(...)` call
        # raised AttributeError on every single invocation; wrapped in the
        # try/except this used to live in, that made this function return an
        # empty DataFrame 100% of the time, which is why the perp price line
        # never actually appeared on the "Block Trades Over Time" chart.
        ts = pd.to_datetime(chart['ticks'], unit='ms', utc=True).tz_convert(SGT)
        return pd.DataFrame({'timestamp': ts, 'close': chart['close']})
    print(f"Tradingview API did not return OK for {instrument}: {chart}")
    return pd.DataFrame()

@st.cache_data(ttl=TTL_MEDIUM, show_spinner=False)
def fetch_dvol(asset: str, start_dt_utc: datetime) -> pd.Series:
    if asset not in ['BTC', 'ETH']:
        return pd.Series(dtype=float)

    start_ms = int(start_dt_utc.timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    params = {
        'currency': asset,
        'start_timestamp': start_ms,
        'end_timestamp': end_ms,
        'resolution': '60'
    }
    data = _get_json_with_retry(url, params=params, timeout=10)
    if data:
        result = data.get('result', {})
        rows = result.get('data', [])
        if rows:
            cols = ['timestamp', 'open', 'high', 'low', 'close']
            df = pd.DataFrame(rows, columns=cols)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(SGT)
            df.set_index('timestamp', inplace=True)
            return df['close'].sort_index()
    return pd.Series(dtype=float)

# ---------------------------------------------------------------------------
# Greeks Math
# ---------------------------------------------------------------------------
def compute_vectorized_greeks(df: pd.DataFrame, spot: float, asset: str) -> pd.DataFrame:
    if df.empty or spot <= 0:
        return df

    df = df.copy()
    now_utc = datetime.now(timezone.utc)
    
    # Calculate time to expiry in years
    expiries_utc = df['expiry'].dt.tz_localize('UTC') if df['expiry'].dt.tz is None else df['expiry'].dt.tz_convert('UTC')
    df['tte'] = ((expiries_utc - now_utc).dt.total_seconds() / (365.25 * 86400)).clip(lower=0.001)

    # IV estimate
    default_ivs = {'BTC': 0.60, 'ETH': 0.65, 'SOL_USDC': 0.75, 'XRP_USDC': 0.80}
    sigma = default_ivs.get(asset, 0.70)
    
    S = spot
    K = df['strike'].values
    T = df['tte'].values
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (0.5 * sigma**2) * T) / (sigma * sqrt_T)
    
    # Greeks
    df['delta'] = np.where(df['option_type'] == 'C', norm.cdf(d1), norm.cdf(d1) - 1)
    df['gamma'] = norm.pdf(d1) / (S * sigma * sqrt_T)
    df['vega'] = S * norm.pdf(d1) * sqrt_T * 0.01

    # Dollar Greeks
    amt = df['amount'].values
    df['dollar_delta'] = df['delta'] * amt * S
    df['dollar_gamma_1pct'] = df['gamma'] * amt * (S ** 2) * 0.01
    df['dollar_vega'] = df['vega'] * amt
    df['gross_notional'] = df['abs_amount'] * df['strike']

    # Theo Edge (Spread cross) in USD
    theo_asset_units = (df['mark_price'] - df['price']) * df['amount']
    if asset in ['BTC', 'ETH']:
        df['edge'] = (theo_asset_units * spot).abs()
    else:
        df['edge'] = theo_asset_units.abs()

    return df

# ---------------------------------------------------------------------------
# Statistics Tables
# ---------------------------------------------------------------------------
def create_expiry_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    
    df['net_puts'] = np.where(df['option_type'] == 'P', df['amount'], 0)
    df['net_calls'] = np.where(df['option_type'] == 'C', df['amount'], 0)

    grouped = df.groupby('expiry').agg({
        'dollar_delta': 'sum',
        'dollar_gamma_1pct': 'sum',
        'dollar_vega': 'sum',
        'net_puts': 'sum',
        'net_calls': 'sum',
        'gross_notional': 'sum',
        'edge': 'sum'
    }).reset_index()

    grouped['Expiry'] = grouped['expiry'].dt.strftime('%d-%b-%Y')
    grouped = grouped.sort_values('expiry').rename(columns={
        'dollar_delta': 'Delta',
        'dollar_gamma_1pct': 'Gamma (1%)',
        'dollar_vega': 'Vega',
        'net_puts': 'Net Puts',
        'net_calls': 'Net Calls',
        'gross_notional': 'Gross Notional',
        'edge': 'Edge'
    })

    total_row = pd.DataFrame([{
        'Expiry': 'Total',
        'Delta': grouped['Delta'].sum(),
        'Gamma (1%)': grouped['Gamma (1%)'].sum(),
        'Vega': grouped['Vega'].sum(),
        'Net Puts': grouped['Net Puts'].sum(),
        'Net Calls': grouped['Net Calls'].sum(),
        'Gross Notional': grouped['Gross Notional'].sum(),
        'Edge': grouped['Edge'].sum()
    }])

    return pd.concat([grouped[['Expiry', 'Delta', 'Vega', 'Gamma (1%)', 'Net Puts', 'Net Calls', 'Gross Notional', 'Edge']], total_row], ignore_index=True)

def style_statistics_table(df: pd.DataFrame):
    if df.empty: return df
    df_disp = df.copy()

    for col in ['Delta', 'Vega', 'Gamma (1%)', 'Edge']:
        if col in df_disp.columns:
            df_disp[col] = df_disp[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "$0")

    for col in ['Net Puts', 'Net Calls', 'Gross Notional']:
        if col in df_disp.columns:
            df_disp[col] = df_disp[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")

    def color_val(val):
        if isinstance(val, str) and ('$' in val or ',' in val):
            try:
                num = float(val.replace('$', '').replace(',', ''))
                return 'color: red' if num < 0 else 'color: green' if num > 0 else ''
            except Exception:
                return ''
        return ''

    # Handle pandas >= 2.1.0 .map() deprecation safely
    styler = df_disp.style
    subset_cols = ['Delta', 'Vega', 'Gamma (1%)', 'Net Puts', 'Net Calls']
    subset_cols = [c for c in subset_cols if c in df_disp.columns]
    
    if hasattr(styler, 'map'):
        return styler.map(color_val, subset=subset_cols)
    else:
        return styler.applymap(color_val, subset=subset_cols)

# ---------------------------------------------------------------------------
# Plotting Functions (Deribit Style)
# ---------------------------------------------------------------------------
def plot_scatter_with_spot_and_dvol(data, hist_spot, dvol_series, current_spot, asset, marker_size):
    fig = go.Figure()
    if data.empty:
        fig.update_layout(title=f'{asset} Block Trades Over Time: No Data')
        fx_style.add_watermark(fig)
        fig = fx_style.apply_theme(fig)
        return fig

    df = data.copy()
    df['minute'] = df['timestamp'].dt.floor('min').dt.tz_localize(None)

    grouped = df.groupby(['minute', 'instrument_name', 'strike', 'expiry', 'option_type']).agg({
        'amount': 'sum',
        'price': 'mean',
        'mark_price': 'mean'
    }).reset_index()

    for direction in [1, -1]:
        for opt in ['C', 'P']:
            trades = grouped[(np.sign(grouped['amount']) == direction) & (grouped['option_type'] == opt)]
            if trades.empty: continue

            marker_symbol = 'square' if opt == 'P' else 'circle'
            color = 'green' if direction == 1 else 'red'

            hover_text = []
            for row in trades.itertuples():
                pnl = (row.mark_price - row.price) * row.amount * (current_spot if asset in ['BTC', 'ETH'] else 1.0)
                ts_str = row.minute.strftime('%Y-%m-%d %H:%M:%S') + ' SGT'
                exp_str = row.expiry.strftime('%d%b%y').upper() if pd.notna(row.expiry) else '—'
                hover_text.append(
                    f"Timestamp: {ts_str}<br>"
                    f"Strike: {row.strike:,.0f}<br>"
                    f"Expiry: {exp_str}<br>"
                    f"Amount: {row.amount:.1f}<br>"
                    f"Trade Price: {row.price:.4f}<br>"
                    f"Mark Price: {row.mark_price:.4f}<br>"
                    f"Current PnL: ${pnl:,.0f}"
                )

            sizes = (trades['amount'].abs() * marker_size).clip(upper=MARKER_SIZE_CAP)

            fig.add_trace(go.Scatter(
                x=trades['minute'], y=trades['strike'], mode='markers',
                marker=dict(size=sizes, color=color, symbol=marker_symbol, opacity=0.75, line=dict(width=1, color='white')),
                text=hover_text, hoverinfo='text',
                name=f'{"Buy" if direction == 1 else "Sell"} {opt}'
            ))

    # Historical Spot Price (yaxis2)
    if not hist_spot.empty:
        hist_ts = hist_spot['timestamp'].dt.tz_localize(None)
        fig.add_trace(go.Scatter(
            x=hist_ts, y=hist_spot['close'], mode='lines',
            name=f'{asset} Perpetual', line=dict(color=SECONDARY_COLOR, width=1.5),
            yaxis='y2'
        ))

    # DVOL Volatility Index (yaxis3)
    if not dvol_series.empty:
        dvol_ts = dvol_series.index.tz_localize(None)
        fig.add_trace(go.Scatter(
            x=dvol_ts, y=dvol_series.values, mode='lines',
            name='DVOL', line=dict(color='orange', width=1.5),
            yaxis='y3'
        ))

    layout = dict(
        title=f'{asset} Block Trades Over Time (Perp & DVOL)',
        xaxis_title='Timestamp (SGT)',
        yaxis_title='Option Strike',
        yaxis=dict(type='linear', autorange=True),
        yaxis2=dict(title=f'{asset} Perpetual', overlaying='y', side='right', showgrid=False, autorange=True),
        paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font=dict(color=TEXT_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=600,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    if not dvol_series.empty:
        layout['yaxis3'] = dict(title='DVOL', overlaying='y', side='right', position=0.95, showgrid=False, autorange=True)

    fig.update_layout(**layout)
    fx_style.add_watermark(fig)
    fig = fx_style.apply_theme(fig)
    return fig

def plot_strike_vs_expiry(data, asset, size_multiplier=1.0):
    fig = go.Figure()
    if data.empty:
        fig.update_layout(title=f'{asset} Aggregated Trades by Expiry and Strike: No Data')
        return fig

    df = data.copy()
    df['expiry_str'] = df['expiry'].dt.strftime('%d%b%y').str.upper()
    df['premium_usd'] = df['abs_amount'] * df['price'] * df['index_price']

    agg = df.groupby(['strike', 'expiry', 'expiry_str', 'option_type']).agg(
        amount=('amount', 'sum'),
        abs_amount=('abs_amount', 'sum'),
        premium_usd=('premium_usd', 'sum')
    ).reset_index()

    agg['direction'] = np.where(agg['amount'] >= 0, 'Buy', 'Sell')
    agg['scaled_size'] = agg['premium_usd'] * size_multiplier

    max_val = float(agg['scaled_size'].max()) if not agg.empty else 1.0
    sizeref = 2.0 * max_val / (40 ** 2)

    expiry_order = agg[['expiry_str', 'expiry']].drop_duplicates().sort_values('expiry')
    category_order = expiry_order['expiry_str'].tolist()

    for direction, color in (('Buy', 'green'), ('Sell', 'red')):
        for opt, symbol in (('C', 'circle'), ('P', 'square')):
            sel = agg[(agg['direction'] == direction) & (agg['option_type'] == opt)]
            if sel.empty: continue

            hover_text = []
            for r in sel.itertuples():
                hover_text.append(
                    f"Expiry: {r.expiry_str}<br>"
                    f"Strike: {r.strike:,.0f}<br>"
                    f"Type: {opt}<br>"
                    f"Direction: {direction}<br>"
                    f"Premium: ${r.premium_usd:,.0f}<br>"
                    f"Net Amount: {r.amount:.1f}"
                )

            fig.add_trace(go.Scatter(
                x=sel['expiry_str'], y=sel['strike'], mode='markers',
                marker=dict(
                    size=sel['scaled_size'], sizemode='area', sizeref=max(sizeref, 1e-12), sizemin=4,
                    color=color, symbol=symbol, opacity=0.75, line=dict(width=1, color='white')
                ),
                text=hover_text, hoverinfo='text',
                name=f'{direction} {opt}'
            ))

    min_strike = max(agg['strike'].min(), 1)
    max_strike = max(agg['strike'].max(), min_strike + 1)
    ticks = np.geomspace(min_strike, max_strike, 10)
    ticks = np.unique(np.round(ticks, 0))

    fig.update_layout(
        title=f'{asset} Aggregated Trades by Expiry and Strike',
        xaxis_title='Expiry Date', yaxis_title='Strike Price (Log Scale)',
        paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font=dict(color=TEXT_COLOR),
        xaxis=dict(type='category', categoryorder='array', categoryarray=category_order),
        yaxis=dict(type='log', tickvals=ticks, ticktext=[f'{t:,.0f}' for t in ticks],
                   ticks='outside', showline=True, linewidth=1, linecolor='black', mirror=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=500,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fx_style.add_watermark(fig)
    fig = fx_style.apply_theme(fig)
    return fig

def _square_heatmap_axes(pivot):
    """xaxis/yaxis kwargs that make every cell in a strike x expiry heatmap
    a uniform square, regardless of the actual numeric gaps between strikes
    or calendar gaps between expiries.

    By default the strike axis is numeric/continuous, so Plotly spaces
    columns by actual price distance (a 5000-wide strike gap renders visibly
    narrower than a 20000-wide one), and the expiry axis - built from
    '%Y-%m-%d' strings - gets auto-detected as a date axis and spaces rows
    by actual calendar distance (a weekly expiry next to a quarterly one
    would render with very different row heights). Forcing both to
    `type='category'` makes every column/row exactly one unit wide/tall
    regardless of the real value gap, and `scaleanchor`+`scaleratio=1` then
    forces those units to be equal in pixels too, so cells end up square
    instead of just uniformly-spaced rectangles.
    """
    return (
        dict(title='Strike', type='category',
             categoryorder='array', categoryarray=list(pivot.columns)),
        dict(title='Expiry', type='category',
             categoryorder='array', categoryarray=list(pivot.index.strftime('%Y-%m-%d')),
             scaleanchor='x', scaleratio=1),
    )

def plot_net_heatmap(data, asset):
    fig = go.Figure()
    if data.empty:
        fig.update_layout(title=f'{asset} Option Net Strike vs Expiry Heatmap: No Data')
        return fig

    heatmap_data = data.groupby(['strike', 'expiry']).agg({'amount': 'sum'}).reset_index()
    pivot = heatmap_data.pivot(index='expiry', columns='strike', values='amount').fillna(0)
    pivot = pivot.reindex(sorted(pivot.index), method='nearest')

    max_abs_val = np.max(np.abs(pivot.values)) or 1
    colorscale = [[0, 'red'], [0.5, 'white'], [1, 'green']]

    fig.add_trace(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index.strftime('%Y-%m-%d'),
        colorscale=colorscale, zmin=-max_abs_val, zmax=max_abs_val,
        colorbar=dict(title='Net Contracts', bgcolor=BACKGROUND_COLOR, bordercolor=TEXT_COLOR, tickfont=dict(color=TEXT_COLOR)),
        hovertemplate='Strike: %{x}<br>Expiry: %{y}<br>Net Flow: %{z}<extra></extra>'
    ))
    xaxis, yaxis = _square_heatmap_axes(pivot)
    fig.update_layout(
        title=f'{asset} Option Strike vs Expiry Net Flow Heatmap',
        xaxis=xaxis, yaxis=yaxis,
        paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font=dict(color=TEXT_COLOR),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fx_style.add_watermark(fig)
    fig = fx_style.apply_theme(fig)
    return fig

def plot_gross_volume_heatmap(data, asset):
    fig = go.Figure()
    if data.empty:
        fig.update_layout(title=f'{asset} Volume Heatmap: No Data')
        return fig

    heatmap_data = data.groupby(['strike', 'expiry']).agg({'abs_amount': 'sum'}).reset_index()
    pivot = heatmap_data.pivot(index='expiry', columns='strike', values='abs_amount').fillna(0)
    pivot = pivot.reindex(sorted(pivot.index), method='nearest')

    max_val = np.max(pivot.values) or 1
    volume_colorscale = [[0.0, 'white'], [0.5, '#66b2ff'], [1.0, '#0000ff']]

    fig.add_trace(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index.strftime('%Y-%m-%d'),
        colorscale=volume_colorscale, zmin=0, zmax=max_val,
        colorbar=dict(title='Gross Contracts', bgcolor=BACKGROUND_COLOR, bordercolor=TEXT_COLOR, tickfont=dict(color=TEXT_COLOR)),
        hovertemplate='Strike: %{x}<br>Expiry: %{y}<br>Gross Volume: %{z}<extra></extra>'
    ))
    xaxis, yaxis = _square_heatmap_axes(pivot)
    fig.update_layout(
        title=f'{asset} Gross Volume Heatmap',
        xaxis=xaxis, yaxis=yaxis,
        paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font=dict(color=TEXT_COLOR),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fx_style.add_watermark(fig)
    fig = fx_style.apply_theme(fig)
    return fig

def plot_cumulative_flow(data, asset):
    fig = go.Figure()
    if data.empty:
        fig.update_layout(title=f'{asset} Cumulative Flow: No Data')
        return fig

    df = data.copy()
    df['minute'] = df['timestamp'].dt.floor('min')

    vol = df.groupby('minute')['abs_amount'].sum().cumsum().reset_index()
    buys = df[df['direction'] == 'buy'].groupby('minute')['abs_amount'].sum().cumsum().reset_index()
    sells = df[df['direction'] == 'sell'].groupby('minute')['abs_amount'].sum().cumsum().reset_index()
    calls = df[df['option_type'] == 'C'].groupby('minute')['abs_amount'].sum().cumsum().reset_index()
    puts = df[df['option_type'] == 'P'].groupby('minute')['abs_amount'].sum().cumsum().reset_index()

    fig.add_trace(go.Scatter(x=vol['minute'], y=vol['abs_amount'], mode='lines', line=dict(color=SECONDARY_COLOR, width=2), name='Total Volume'))
    fig.add_trace(go.Scatter(x=buys['minute'], y=buys['abs_amount'], mode='lines', line=dict(color='green'), name='Cumulative Buys'))
    fig.add_trace(go.Scatter(x=sells['minute'], y=sells['abs_amount'], mode='lines', line=dict(color='red'), name='Cumulative Sells'))
    fig.add_trace(go.Scatter(x=calls['minute'], y=calls['abs_amount'], mode='lines', line=dict(color='blue'), name='Cumulative Calls'))
    fig.add_trace(go.Scatter(x=puts['minute'], y=puts['abs_amount'], mode='lines', line=dict(color='purple'), name='Cumulative Puts'))

    fig.update_layout(
        title=f'{asset} Cumulative Volume Over Time',
        xaxis_title='Date (SGT)', yaxis_title='Contracts',
        paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font=dict(color=TEXT_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fx_style.add_watermark(fig)
    fig = fx_style.apply_theme(fig)
    return fig

# ---------------------------------------------------------------------------
# Telegram Sending (same rendering path as pages/01_MCM_Bot.py's
# send_result_to_telegram, adapted for this page's plain chart set - no
# tables here, so it's just "render each figure to PNG, sanitize its title
# into a caption, send")
# ---------------------------------------------------------------------------
def _caption_from_title(title_text, fallback: str) -> str:
    """Telegram-safe caption from a figure's title. Our titles here are
    plain f-strings with no Plotly <span>/<br> markup, but html.escape()
    is cheap insurance against parse_mode=HTML choking on a literal '<' or
    '&' (e.g. a future title containing a raw comparison operator) - see
    pages/01_MCM_Bot.py's _caption_from_title for the incident this pattern
    was built to avoid."""
    if not title_text:
        return fallback
    text = html.escape(str(title_text).replace("<br>", "\n").strip())
    return (text[:1024] if len(text) > 1024 else text) or fallback

def _render_png_with_reason(fig):
    """Render one themed figure to PNG bytes via kaleido - same call as
    fx_style.fig_to_png, but also returns the underlying exception text on
    failure. fx_style.fig_to_png only prints that text to stdout (visible
    in Streamlit Cloud's Manage app -> Logs), so a render failure here
    would otherwise show only a generic "failed to render" with no way to
    tell why (missing Chromium, a timeout, an unsupported layout property)
    without leaving the app. Surfacing it directly in the failure reason
    saves that round trip."""
    try:
        return fig.to_image(format="png", width=1200, height=800), None
    except Exception as e:
        return None, str(e)

def _send_chart_to_telegram(fig, caption_base: str):
    """Render one chart (light theme, matching 01_MCM_Bot.py's Telegram
    rendering) and send it. Retries the render once, same as
    01_MCM_Bot.py's send_result_to_telegram, in case of a transient
    kaleido hiccup. Returns (ok, failure_reason_or_None); the reason
    marks whether the failure was at render (kaleido) or send (Telegram)
    stage, since those point at different root causes."""
    if fig is None:
        return False, f"{caption_base}: no chart to send"
    title_obj = getattr(getattr(fig, "layout", None), "title", None)
    title_text = getattr(title_obj, "text", None) if title_obj else None
    caption = _caption_from_title(title_text, caption_base)

    themed = fx_style.apply_theme(fig, "light")
    png, err = _render_png_with_reason(themed)
    if png is None:
        png, err = _render_png_with_reason(themed)
    if png is None:
        detail = f" ({err})" if err else ""
        return False, f"{caption_base}: failed to **render**{detail}"
    if telegram.send_photo(png, caption):
        return True, None
    return False, f"{caption_base}: failed to **send**"

def _send_many_to_telegram(fig_label_pairs: list, label: str) -> None:
    """Send a batch of (clean_asset_name, fig) pairs to Telegram with a
    progress bar and a success/failure summary - shared by the toolbar's
    per-asset, BTC+ETH, and All buttons so there's one code path instead
    of three copies (mirrors 01_MCM_Bot.py's _send_to_telegram)."""
    if not fig_label_pairs:
        st.warning(f"Nothing to send for {label}.")
        return
    total_sent = total_failed = 0
    all_reasons: list = []
    progress = st.progress(0.0) if len(fig_label_pairs) > 1 else None
    with st.spinner(f"Sending {label} to Telegram..."):
        for i, (clean_name, fig) in enumerate(fig_label_pairs, start=1):
            ok, reason = _send_chart_to_telegram(fig, f"{clean_name} block trades")
            if ok:
                total_sent += 1
            else:
                total_failed += 1
                if reason:
                    all_reasons.append(reason)
            if progress:
                progress.progress(i / len(fig_label_pairs))
    if progress:
        progress.empty()
    if total_sent:
        st.success(f"Sent {total_sent} chart(s) to Telegram."
                   + (f" ({total_failed} failed — see detail below.)" if total_failed else ""))
    else:
        st.error("Failed to send charts to Telegram.")
    if all_reasons:
        n_render = sum(1 for r in all_reasons if "render" in r)
        n_send = sum(1 for r in all_reasons if "send" in r)
        with st.expander(f"⚠️ {len(all_reasons)} chart(s) had trouble — "
                         f"{n_render} failed to render, {n_send} failed to send. Click for detail."):
            st.caption(
                "**Render** failures happen before Telegram is ever contacted "
                "(the chart image itself couldn't be built — usually kaleido; "
                "the text in parentheses is the underlying error). **Send** "
                "failures mean the image rendered fine but Telegram rejected "
                "or didn't receive it.")
            for r in all_reasons:
                st.markdown(f"- {r}")

# ---------------------------------------------------------------------------
# Data Pipeline & Tab Rendering
# ---------------------------------------------------------------------------
asset_data_dict = {}
hist_spot_dict = {}
dvol_dict = {}
spot_dict = {}

with st.spinner("Fetching Block Trades, Spot, and DVOL data..."):
    # Previously this looped over the 7 assets sequentially, issuing 4 blocking
    # HTTP calls per asset (spot, trades, perp candles, DVOL) one after another
    # - ~26 serial round-trips (5 of the "trades" calls were also fetching the
    # exact same USDC currency feed redundantly, see fetch_public_trades).
    # That serial waterfall is what made the page slow to load, and long
    # enough that a rate-limited or timed-out call late in the sequence
    # (e.g. a later asset's perp price) would silently come back empty with
    # no retry - which is what could make the perp line look "missing" on a
    # chart even though the plotting code for it is correct.
    # Fetching every asset/currency concurrently cuts wall-clock time down to
    # roughly the slowest single call instead of the sum of all of them, and
    # the retry/backoff in _get_json_with_retry makes each call more likely
    # to actually succeed.
    unique_currencies = sorted(set(CURRENCY_MAP.get(a, a) for a in ASSETS))

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        spot_futures = {executor.submit(get_current_spot, a): a for a in ASSETS}
        trades_futures = {
            executor.submit(fetch_trades_by_currency, c, start_datetime_utc): c
            for c in unique_currencies
        }
        hist_futures = {
            executor.submit(fetch_historical_spot, a, start_datetime_utc): a
            for a in ASSETS
        }
        dvol_futures = {
            executor.submit(fetch_dvol, a, start_datetime_utc): a for a in ASSETS
        }

        for fut in concurrent.futures.as_completed(spot_futures):
            spot_dict[spot_futures[fut]] = fut.result()

        currency_trades = {}
        for fut in concurrent.futures.as_completed(trades_futures):
            currency_trades[trades_futures[fut]] = fut.result()

        for fut in concurrent.futures.as_completed(hist_futures):
            hist_spot_dict[hist_futures[fut]] = fut.result()

        for fut in concurrent.futures.as_completed(dvol_futures):
            dvol_dict[dvol_futures[fut]] = fut.result()

    for asset in ASSETS:
        spot = spot_dict[asset]
        raw_trades = currency_trades.get(CURRENCY_MAP.get(asset, asset), pd.DataFrame())
        if not raw_trades.empty and "_USDC" in asset and "instrument_name" in raw_trades.columns:
            raw_trades = raw_trades[raw_trades['instrument_name'].str.startswith(f"{asset}-")].copy()

        if not raw_trades.empty:
            filtered = raw_trades[raw_trades['abs_amount'] >= min_sizes[asset]].copy()
            filtered['index_price'] = spot
            filtered = compute_vectorized_greeks(filtered, spot, asset)
            asset_data_dict[asset] = filtered
        else:
            asset_data_dict[asset] = pd.DataFrame()

# Summary Top Metrics
m_cols = st.columns(len(ASSETS))
for idx, asset in enumerate(ASSETS):
    clean = asset.replace('_USDC', '')
    cnt = len(asset_data_dict[asset])
    vol = asset_data_dict[asset]['abs_amount'].sum() if cnt > 0 else 0
    m_cols[idx].metric(f"{clean} Blocks", f"{cnt}", f"{vol:,.0f} ctrs")

# ---------------------------------------------------------------------------
# Send to Telegram — one button per asset, a BTC+ETH combo, plus "All",
# matching pages/01_MCM_Bot.py's toolbar (send just one asset, a pair, or
# everything) instead of a single all-or-nothing send. Clicks are captured
# here, but the actual sending happens after the tab loop below, once
# asset_figs_dict is fully populated with every asset's charts.
# ---------------------------------------------------------------------------
_tg_configured = telegram.is_configured()
st.caption("Send charts to Telegram (as images):" if _tg_configured
           else f"Telegram not configured: {telegram.config_status()}")
_tg_cols = st.columns(len(ASSETS) + 2)
send_asset_clicked = {}
for _a, _col in zip(ASSETS, _tg_cols[:len(ASSETS)]):
    _clean = _a.replace('_USDC', '')
    with _col:
        send_asset_clicked[_a] = st.button(
            _clean, key=f"tg_send_{_a}", width="stretch",
            disabled=not _tg_configured,
            help=f"Send {_clean}'s 5 charts to Telegram.")
_BTC_ETH = [a for a in ASSETS if a in ("BTC", "ETH")]
with _tg_cols[len(ASSETS)]:
    send_btc_eth_clicked = st.button(
        "BTC+ETH", key="tg_send_btc_eth", width="stretch",
        disabled=not _tg_configured or not _BTC_ETH,
        help="Send BTC and ETH's charts to Telegram.")
with _tg_cols[len(ASSETS) + 1]:
    send_all_clicked = st.button(
        "📤 All", key="tg_send_all", width="stretch",
        disabled=not _tg_configured,
        help="Send every asset's charts to Telegram (35 charts total).")

# Tabs
tab_names = [f"📈 {a.replace('_USDC', '')}" for a in ASSETS] + ["📊 ALL (2x2 Grid)", "📋 Block Trade Statistics"]
tabs = st.tabs(tab_names)

def get_asset_multipliers(asset):
    if asset == 'BTC': return 0.15, 1.0
    if asset == 'ETH': return 0.5/15, 1.0/15
    if 'SOL' in asset: return 0.5/150, 1.0/150
    if 'XRP' in asset: return 0.5/1500, 1.0/1500
    if 'TRX' in asset: return 0.5/1500, 1.0/1500
    if 'AVAX' in asset: return 0.5/150, 1.0/150
    if 'HYPE' in asset: return 0.5/150, 1.0/150
    return 0.5/150, 1.0/150

# ---------------------------------------------------------------------------
# Render Individual Asset Tabs
# ---------------------------------------------------------------------------
asset_figs_dict = {}  # asset -> list of figs, built once here and reused by
                       # both st.plotly_chart (on-screen) and the Telegram
                       # send buttons below, instead of re-computing them.

for idx, asset in enumerate(ASSETS):
    clean_name = asset.replace('_USDC', '')
    with tabs[idx]:
        st.markdown(f'<div class="asset-header">{clean_name} Option Block Flow Analysis</div>', unsafe_allow_html=True)

        df_asset = asset_data_dict[asset]
        hist_spot = hist_spot_dict[asset]
        dvol_s = dvol_dict[asset]
        spot_px = spot_dict[asset]

        m_size, s_mult = get_asset_multipliers(asset)

        fig_scatter = plot_scatter_with_spot_and_dvol(df_asset, hist_spot, dvol_s, spot_px, clean_name, m_size)
        fig_strike_expiry = plot_strike_vs_expiry(df_asset, clean_name, s_mult)
        fig_net_heatmap = plot_net_heatmap(df_asset, clean_name)
        fig_gross_volume = plot_gross_volume_heatmap(df_asset, clean_name)
        fig_cumulative = plot_cumulative_flow(df_asset, clean_name)
        asset_figs_dict[asset] = [
            fig_scatter, fig_strike_expiry, fig_net_heatmap, fig_gross_volume, fig_cumulative,
        ]

        # 1. Scatter with Spot & DVOL
        st.plotly_chart(fig_scatter, width="stretch", key=f"tab_scatter_{asset}")

        # 2. Strike vs Expiry
        st.plotly_chart(fig_strike_expiry, width="stretch", key=f"tab_strike_expiry_{asset}")

        # 3. Heatmaps Side-by-Side (equal-width columns, so both boxes match)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_net_heatmap, width="stretch", key=f"tab_net_heatmap_{asset}")
        with col2:
            st.plotly_chart(fig_gross_volume, width="stretch", key=f"tab_gross_volume_{asset}")

        # 4. Cumulative Flow (moved out of the heatmap row so both heatmaps
        # above are equal-sized boxes instead of one being paired with this
        # chart at half width and the other sitting full-width alone)
        st.plotly_chart(fig_cumulative, width="stretch", key=f"tab_cumulative_{asset}")

# Handle the toolbar's Telegram buttons now that every asset's figures have
# been built (asset_figs_dict is fully populated by the tab loop above).
def _figs_for(assets: list) -> list:
    return [(a.replace('_USDC', ''), fig) for a in assets for fig in asset_figs_dict[a]]

for _a in ASSETS:
    if send_asset_clicked.get(_a):
        _send_many_to_telegram(_figs_for([_a]), _a.replace('_USDC', ''))

if send_btc_eth_clicked and _BTC_ETH:
    _send_many_to_telegram(_figs_for(_BTC_ETH), "BTC+ETH")

if send_all_clicked:
    _send_many_to_telegram(_figs_for(ASSETS), "all assets")

# ---------------------------------------------------------------------------
# Render ALL (2x2 Grid) Tab
# ---------------------------------------------------------------------------
with tabs[len(ASSETS)]:
    st.markdown('<div class="asset-header">📊 ALL Block Trades Overview</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_scatter_with_spot_and_dvol(
                asset_data_dict['BTC'], hist_spot_dict['BTC'], dvol_dict['BTC'], spot_dict['BTC'], 'BTC', 0.15
            ), 
            width="stretch",
            key="grid_scatter_btc"
        )
    with c2:
        st.plotly_chart(
            plot_scatter_with_spot_and_dvol(
                asset_data_dict['ETH'], hist_spot_dict['ETH'], dvol_dict['ETH'], spot_dict['ETH'], 'ETH', 0.5/15
            ), 
            width="stretch",
            key="grid_scatter_eth"
        )
        
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            plot_scatter_with_spot_and_dvol(
                asset_data_dict['SOL_USDC'], hist_spot_dict['SOL_USDC'], pd.Series(dtype=float), spot_dict['SOL_USDC'], 'SOL', 0.5/150
            ), 
            width="stretch",
            key="grid_scatter_sol"
        )
    with c4:
        st.plotly_chart(
            plot_scatter_with_spot_and_dvol(
                asset_data_dict['XRP_USDC'], hist_spot_dict['XRP_USDC'], pd.Series(dtype=float), spot_dict['XRP_USDC'], 'XRP', 0.5/1500
            ), 
            width="stretch",
            key="grid_scatter_xrp"
        )

# ---------------------------------------------------------------------------
# Render Statistics Tab
# ---------------------------------------------------------------------------
with tabs[len(ASSETS) + 1]:
    st.markdown('<div class="asset-header">📋 Block Trade Statistics (Greeks & Expiry Breakdown)</div>', unsafe_allow_html=True)
    
    stat_sub_tabs = st.tabs([a.replace('_USDC', '') for a in ASSETS])
    for s_idx, asset in enumerate(ASSETS):
        clean = asset.replace('_USDC', '')
        with stat_sub_tabs[s_idx]:
            df_asset = asset_data_dict[asset]
            if not df_asset.empty:
                exp_table = create_expiry_table(df_asset)
                st.subheader(f"{clean} Greeks & Volume by Expiry")
                # Keys are not required for st.dataframe
                st.dataframe(style_statistics_table(exp_table), width="stretch")
            else:
                st.info(f"No block trades found for {clean} in the selected time range.")