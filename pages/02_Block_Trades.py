"""
Block Trades — Deribit public block trade analysis.

Replicates the Deribit Block Trades page:
- Scatter Plot overlaid with Historical Spot Price (yaxis2) and DVOL Index (yaxis3)
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
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from scipy.stats import norm
import streamlit as st

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
# Constants
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

# ---------------------------------------------------------------------------
# API Fetching Helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def get_current_spot(asset: str) -> float:
    index_map = {
        'BTC': 'btc_usd', 'ETH': 'eth_usd', 'SOL_USDC': 'sol_usdc',
        'XRP_USDC': 'xrp_usdc', 'TRX_USDC': 'trx_usdc',
        'AVAX_USDC': 'avax_usdc', 'HYPE_USDC': 'hype_usdc'
    }
    index_name = index_map.get(asset, f"{asset.lower()}_usd")
    url = f'https://deribit.com/api/v2/public/get_index_price?index_name={index_name}'
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return float(r.json()['result']['index_price'])
    except Exception:
        return 0.0

@st.cache_data(ttl=60, show_spinner=False)
def fetch_public_trades(asset: str, start_dt_utc: datetime) -> pd.DataFrame:
    currency = "USDC" if "_USDC" in asset else asset
    start_ms = int(start_dt_utc.timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    url = (
        f"https://www.deribit.com/api/v2/public/get_last_trades_by_currency_and_time?"
        f"currency={currency}&kind=option&start_timestamp={start_ms}&end_timestamp={end_ms}&count=1000&sorting=asc"
    )
    
    all_trades = []
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if 'result' in data and 'trades' in data['result']:
            all_trades = data['result']['trades']
    except Exception:
        return pd.DataFrame()

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    
    # Filter by instrument prefix for USDC settled
    if "_USDC" in asset:
        prefix = f"{asset}-"
        df = df[df['instrument_name'].str.startswith(prefix)].copy()

    if df.empty:
        return pd.DataFrame()

    # Parse details
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(SGT)
    df['abs_amount'] = df['amount'].abs()
    df['amount'] = np.where(df['direction'] == 'buy', df['abs_amount'], -df['abs_amount'])
    
    # Parse strikes and expiries
    df['strike'] = pd.to_numeric(df['instrument_name'].str.extract(r'-(\d+(?:\.\d+)*)-')[0], errors='coerce')
    df['expiry'] = pd.to_datetime(df['instrument_name'].str.extract(r'^[^-]*-([0-9]{1,2}[A-Z]{3}[0-9]{2})-')[0], format='%d%b%y', errors='coerce')
    df['option_type'] = df['instrument_name'].str.extract(r'-(C|P)$')[0]
    
    if 'mark_price' not in df.columns and 'price' in df.columns:
        df['mark_price'] = df['price']
    
    return df.dropna(subset=['strike', 'expiry']).reset_index(drop=True)

@st.cache_data(ttl=60, show_spinner=False)
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
        'resolution': '5'  # 5-minute candles for smooth line
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        chart = r.json().get('result', {})
        if chart.get('status') == 'ok' and chart.get('ticks') and chart.get('close'):
            ts = pd.to_datetime(chart['ticks'], unit='ms', utc=True).dt.tz_convert(SGT)
            return pd.DataFrame({'timestamp': ts, 'close': chart['close']})
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
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
    try:
        r = requests.get(url, params=params, timeout=10)
        result = r.json().get('result', {})
        data = result.get('data', [])
        if data:
            cols = ['timestamp', 'open', 'high', 'low', 'close']
            df = pd.DataFrame(data, columns=cols)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(SGT)
            df.set_index('timestamp', inplace=True)
            return df['close'].sort_index()
    except Exception:
        pass
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

    return df_disp.style.applymap(color_val, subset=['Delta', 'Vega', 'Gamma (1%)', 'Net Puts', 'Net Calls'])

# ---------------------------------------------------------------------------
# Plotting Functions (Deribit Style)
# ---------------------------------------------------------------------------
def plot_scatter_with_spot_and_dvol(data, hist_spot, dvol_series, current_spot, asset, marker_size):
    fig = go.Figure()
    if data.empty:
        fig.update_layout(title=f'{asset} Block Trades Over Time: No Data')
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
            name='Historical Spot Price', line=dict(color=SECONDARY_COLOR, width=1.5),
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
        title=f'{asset} Block Trades Over Time (Spot & DVOL)',
        xaxis_title='Timestamp (SGT)',
        yaxis_title='Option Strike',
        yaxis=dict(type='linear', autorange=True),
        yaxis2=dict(title='Historical Spot Price', overlaying='y', side='right', showgrid=False, autorange=True),
        paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font=dict(color=TEXT_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=600
    )
    if not dvol_series.empty:
        layout['yaxis3'] = dict(title='DVOL', overlaying='y', side='right', position=0.95, showgrid=False, autorange=True)

    fig.update_layout(**layout)
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
        height=500
    )
    return fig

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
    fig.update_layout(
        title=f'{asset} Option Strike vs Expiry Net Flow Heatmap',
        xaxis_title='Strike', yaxis_title='Expiry',
        paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font=dict(color=TEXT_COLOR),
        height=400
    )
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
    fig.update_layout(
        title=f'{asset} Gross Volume Heatmap',
        xaxis_title='Strike', yaxis_title='Expiry',
        paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font=dict(color=TEXT_COLOR),
        height=400
    )
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
        height=400
    )
    return fig

# ---------------------------------------------------------------------------
# Data Pipeline & Tab Rendering
# ---------------------------------------------------------------------------
asset_data_dict = {}
hist_spot_dict = {}
dvol_dict = {}
spot_dict = {}

with st.spinner("Fetching Block Trades, Spot, and DVOL data..."):
    for asset in ASSETS:
        spot = get_current_spot(asset)
        spot_dict[asset] = spot
        
        raw_trades = fetch_public_trades(asset, start_datetime_utc)
        if not raw_trades.empty:
            filtered = raw_trades[raw_trades['abs_amount'] >= min_sizes[asset]].copy()
            filtered['index_price'] = spot
            filtered = compute_vectorized_greeks(filtered, spot, asset)
            asset_data_dict[asset] = filtered
        else:
            asset_data_dict[asset] = pd.DataFrame()
            
        hist_spot_dict[asset] = fetch_historical_spot(asset, start_datetime_utc)
        dvol_dict[asset] = fetch_dvol(asset, start_datetime_utc)

# Summary Top Metrics
m_cols = st.columns(len(ASSETS))
for idx, asset in enumerate(ASSETS):
    clean = asset.replace('_USDC', '')
    cnt = len(asset_data_dict[asset])
    vol = asset_data_dict[asset]['abs_amount'].sum() if cnt > 0 else 0
    m_cols[idx].metric(f"{clean} Blocks", f"{cnt}", f"{vol:,.0f} ctrs")

# Tabs
tab_names = [f"📈 {a.replace('_USDC', '')}" for a in ASSETS] + ["📊 ALL (2x2 Grid)", "📋 Block Trade Statistics"]
tabs = st.tabs(tab_names)

def get_asset_multipliers(asset):
    if asset == 'BTC': return 0.15, 1.0
    if asset == 'ETH': return 0.5/15, 1.0/15
    if 'SOL' in asset: return 0.5/150, 1.0/150
    if 'XRP' in asset: return 0.5/1500, 1.0/1500
    return 0.5/150, 1.0/150

# Individual Asset Tabs
for idx, asset in enumerate(ASSETS):
    clean_name = asset.replace('_USDC', '')
    with tabs[idx]:
        st.markdown(f'<div class="asset-header">{clean_name} Option Block Flow Analysis</div>', unsafe_allow_html=True)
        
        df_asset = asset_data_dict[asset]
        hist_spot = hist_spot_dict[asset]
        dvol_s = dvol_dict[asset]
        spot_px = spot_dict[asset]
        
        m_size, s_mult = get_asset_multipliers(asset)
        
        # 1. Scatter with Spot & DVOL
        st.plotly_chart(
            plot_scatter_with_spot_and_dvol(df_asset, hist_spot, dvol_s, spot_px, clean_name, m_size),
            use_container_width=True
        )
        
        # 2. Strike vs Expiry
        st.plotly_chart(
            plot_strike_vs_expiry(df_asset, clean_name, s_mult),
            use_container_width=True
        )
        
        # 3. Heatmaps Side-by-Side
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_net_heatmap(df_asset, clean_name), use_container_width=True)
        with col2:
            st.plotly_chart(plot_cumulative_flow(df_asset, clean_name), use_container_width=True)
            
        # 4. Gross Volume Heatmap
        st.plotly_chart(plot_gross_volume_heatmap(df_asset, clean_name), use_container_width=True)

# ALL (2x2 Grid) Tab
with tabs[len(ASSETS)]:
    st.markdown('<div class="asset-header">📊 ALL Block Trades Overview</div>', unsafe_allow_html=True)
    
    grid_assets = ['BTC', 'ETH', 'SOL_USDC', 'XRP_USDC']
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_scatter_with_spot_and_dvol(asset_data_dict['BTC'], hist_spot_dict['BTC'], dvol_dict['BTC'], spot_dict['BTC'], 'BTC', 0.15), use_container_width=True)
    with c2:
        st.plotly_chart(plot_scatter_with_spot_and_dvol(asset_data_dict['ETH'], hist_spot_dict['ETH'], dvol_dict['ETH'], spot_dict['ETH'], 'ETH', 0.5/15), use_container_width=True)
        
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_scatter_with_spot_and_dvol(asset_data_dict['SOL_USDC'], hist_spot_dict['SOL_USDC'], pd.Series(dtype=float), spot_dict['SOL_USDC'], 'SOL', 0.5/150), use_container_width=True)
    with c4:
        st.plotly_chart(plot_scatter_with_spot_and_dvol(asset_data_dict['XRP_USDC'], hist_spot_dict['XRP_USDC'], pd.Series(dtype=float), spot_dict['XRP_USDC'], 'XRP', 0.5/1500), use_container_width=True)

# Statistics Tab
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
                st.dataframe(style_statistics_table(exp_table), use_container_width=True)
            else:
                st.info(f"No block trades found for {clean} in the selected time range.")