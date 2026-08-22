"""
Fear & Greed Signal — contrarian delta-lean backtest.

Answers one question: at what Crypto Fear & Greed Index levels has leaning
delta long (fear) or short (greed) historically paid off, and how confident
should you be in following that signal versus ignoring it?

Fear & Greed data: alternative.me's Crypto Fear & Greed Index (daily, back
to 2018-02-01) — the one external, non-Deribit data source in this app; see
lib/fng.py's module docstring. Price data: Deribit perpetual OHLC (matches
what you'd actually trade delta lean on), via lib/deribit.py.

This is a statistical backtest of a directional signal, not a trading
system: no funding, fees, slippage or leverage costs are modeled, and
forward-return windows overlap (a 7d window starting today shares 6 days
with tomorrow's), so treat the significance stats as indicative, not
textbook-independent-sample rigorous. Said again in the footer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import streamlit as st

from lib.deribit import get_tradingview_ohlc
from lib.constants import ASSET_CONFIG, ASSETS, ASSET_COLORS, PLOTLY_LAYOUT
from lib.telegram import send_message, is_configured
from lib import fx_style
from lib import fng

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fear & Greed Signal",
    page_icon="😱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("😱 Fear & Greed Signal")
st.caption(
    "Contrarian delta-lean backtest — at what Fear & Greed levels has leaning "
    "long (fear) or short (greed) historically paid off?"
)

BUCKET_ORDER = fng.BUCKET_ORDER
BUCKET_SIGNAL = fng.BUCKET_SIGNAL
BUCKET_COLORS = fng.BUCKET_COLORS

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Parameters")

    selected_asset = st.selectbox("Asset", options=ASSETS, index=0)
    cfg = ASSET_CONFIG[selected_asset]
    st.caption(
        "Fear & Greed is one market-wide series (not per-asset) — this "
        f"backtests it against {selected_asset}'s own perpetual price."
    )

    LOOKBACK_MAP = {
        "1y": 365,
        "2y": 730,
        "3y": 1095,
        "Max (since 2018)": 3200,
    }
    lookback_label = st.selectbox(
        "Lookback", options=list(LOOKBACK_MAP.keys()), index=3,
        help="Capped by whichever is shorter: F&G history (from 2018-02-01) "
             "or this asset's Deribit perp history.",
    )
    lookback_days = LOOKBACK_MAP[lookback_label]

    st.divider()
    st.markdown("**Bucket thresholds**")
    extreme_fear_edge = st.slider(
        "Extreme Fear below", min_value=5, max_value=40, value=25,
        help="alternative.me's own default is 25",
    )
    extreme_greed_edge = st.slider(
        "Extreme Greed above", min_value=60, max_value=95, value=75,
        help="alternative.me's own default is 75",
    )
    neutral_half_width = st.slider(
        "Neutral band half-width (around 50)", min_value=0, max_value=20, value=5,
        help="Fear ends / Greed starts at 50 ± this. alternative.me's own default is 5.",
    )
    fear_hi = 50 - neutral_half_width
    greed_lo = 50 + neutral_half_width
    thresholds_valid = extreme_fear_edge < fear_hi and greed_lo < extreme_greed_edge
    thresholds = {
        "extreme_fear": extreme_fear_edge,
        "fear": fear_hi,
        "greed": greed_lo,
        "extreme_greed": extreme_greed_edge,
    }

    st.divider()
    horizons = st.multiselect(
        "Forward-return horizons (days)",
        options=[1, 3, 7, 14, 21, 30, 60],
        default=[1, 3, 7, 14, 30],
    )
    headline_horizon = st.selectbox(
        "Headline horizon (drives the signal panel)",
        options=horizons if horizons else [7],
        index=min(2, max(0, len(horizons) - 1)) if horizons else 0,
    )

    max_lean_pct = st.slider(
        "Max delta lean at an extreme reading (%)", min_value=5, max_value=100,
        value=25, step=5,
        help="Extreme Fear/Greed get this full lean; Fear/Greed get half; "
             "Neutral gets none. Purely a display scaling — the backtest "
             "below uses the ±1 / ±0.5 / 0 signal directly.",
    )

if not thresholds_valid:
    st.error(
        "Threshold ordering is invalid: Extreme Fear edge must be below the "
        "Fear/Neutral split, and the Greed/Neutral split must be below the "
        "Extreme Greed edge. Adjust the sliders in the sidebar."
    )
    st.stop()

if not horizons:
    st.warning("Select at least one forward-return horizon from the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Fetching Fear & Greed Index history...", ttl=3600)
def fetch_fng() -> pd.DataFrame | None:
    return fng.get_fng_history(limit=0)


@st.cache_data(show_spinner="Fetching perpetual OHLC...", ttl=300)
def fetch_price(asset: str, days: int) -> pd.DataFrame | None:
    instrument = ASSET_CONFIG[asset]["perp"]
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000
    df = get_tradingview_ohlc(instrument, "1D", start_ms, end_ms)
    if df is None or df.empty:
        return None
    return df.sort_values("timestamp").reset_index(drop=True)


fng_df = fetch_fng()
price_df = fetch_price(selected_asset, lookback_days)

if fng_df is None or fng_df.empty:
    st.error(
        "Could not fetch the Fear & Greed Index from alternative.me right now "
        "(network issue or the API is down). Nothing else on this page depends "
        "on Deribit, so this is the only failure mode — try again shortly."
    )
    st.stop()

if price_df is None or price_df.empty:
    st.error(f"Could not fetch Deribit perpetual OHLC for {selected_asset}.")
    st.stop()


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def merge_price_fng(price_df: pd.DataFrame, fng_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join daily perp close against the daily F&G reading for the
    same UTC calendar date."""
    p = price_df.copy()
    p["date"] = pd.to_datetime(p["timestamp"]).dt.normalize()
    f = fng_df.copy()
    f["date"] = pd.to_datetime(f["timestamp"]).dt.normalize()
    merged = pd.merge(
        p[["date", "close"]], f[["date", "value", "classification"]],
        on="date", how="inner",
    )
    return merged.sort_values("date").reset_index(drop=True)


def add_bucket(merged: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    out = merged.copy()
    out["bucket"] = out["value"].apply(lambda v: fng.classify(v, thresholds))
    out["signal"] = out["bucket"].map(BUCKET_SIGNAL)
    return out


def add_forward_returns(merged: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = merged.copy()
    for h in horizons:
        out[f"fwd_{h}"] = out["close"].shift(-h) / out["close"] - 1
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion — better-behaved
    than a normal-approximation CI for small n or extreme win rates."""
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    margin = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return ((center - margin) / denom, (center + margin) / denom)


def bucket_win_stats(merged: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Per (bucket, horizon): sample size, mean/median forward return, and
    — for the buckets that carry a directional signal (everything but
    Neutral) — the win rate (probability the contrarian call was right),
    its Wilson 95% CI, and a two-sided binomial-test p-value vs. a coin
    flip."""
    rows = []
    for bucket in BUCKET_ORDER:
        sig = BUCKET_SIGNAL[bucket]
        sub = merged[merged["bucket"] == bucket]
        for h in horizons:
            rets = sub[f"fwd_{h}"].dropna()
            n = len(rets)
            row = {"bucket": bucket, "horizon": h, "signal": sig, "n": n,
                   "mean_ret": np.nan, "median_ret": np.nan,
                   "win_rate": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                   "p_value": np.nan, "t_stat": np.nan}
            if n > 0:
                row["mean_ret"] = float(rets.mean())
                row["median_ret"] = float(rets.median())
                if n > 1 and rets.std(ddof=1) > 0:
                    t_stat, _ = stats.ttest_1samp(rets, 0)
                    row["t_stat"] = float(t_stat)
                if sig != 0:
                    correct = (rets > 0) if sig > 0 else (rets < 0)
                    k = int(correct.sum())
                    row["win_rate"] = k / n
                    row["ci_lo"], row["ci_hi"] = wilson_ci(k, n)
                    try:
                        row["p_value"] = float(stats.binomtest(k, n, 0.5).pvalue)
                    except Exception:
                        row["p_value"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def decile_win_stats(merged: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Threshold-free view: bucket the raw 0-100 value into deciles and
    check win rate assuming 'long below 50, short above 50' — lets you see
    where the edge actually concentrates before picking bucket thresholds."""
    edges = list(range(0, 101, 10))
    labels = [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)]
    d = merged.copy()
    d["decile"] = pd.cut(d["value"], bins=edges, labels=labels, include_lowest=True)
    rows = []
    for i, lab in enumerate(labels):
        mid = (edges[i] + edges[i + 1]) / 2
        direction = 1 if mid < 50 else (-1 if mid > 50 else 0)
        rets = d.loc[d["decile"] == lab, f"fwd_{horizon}"].dropna()
        n = len(rets)
        row = {"decile": lab, "mid": mid, "n": n, "direction": direction,
               "mean_ret": float(rets.mean()) if n else np.nan,
               "win_rate": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
        if n > 0 and direction != 0:
            correct = (rets > 0) if direction > 0 else (rets < 0)
            k = int(correct.sum())
            row["win_rate"] = k / n
            row["ci_lo"], row["ci_hi"] = wilson_ci(k, n)
        rows.append(row)
    return pd.DataFrame(rows)


def regression_stats(merged: pd.DataFrame, horizon: int) -> dict | None:
    sub = merged[["value", f"fwd_{horizon}"]].dropna()
    if len(sub) < 10:
        return None
    res = stats.linregress(sub["value"], sub[f"fwd_{horizon}"])
    return {
        "slope": res.slope, "intercept": res.intercept, "r": res.rvalue,
        "r2": res.rvalue ** 2, "p_value": res.pvalue, "n": len(sub),
    }


def compute_equity_curve(merged: pd.DataFrame) -> pd.DataFrame:
    """Daily-rebalanced: hold position = bucket signal (±1 extreme, ±0.5
    fear/greed, 0 neutral) determined at day t's close, realized over that
    day's 1-day forward return (fwd_1 = close(t+1)/close(t) - 1). These
    windows are contiguous, not overlapping, so compounding day-by-day is
    valid even though the longer-horizon stats above overlap."""
    d = merged[["date", "signal", "fwd_1"]].dropna(subset=["fwd_1"]).copy()
    d["strat_ret"] = d["signal"] * d["fwd_1"]
    d["strategy_equity"] = (1 + d["strat_ret"]).cumprod()
    d["buyhold_equity"] = (1 + d["fwd_1"]).cumprod()
    return d


merged = merge_price_fng(price_df, fng_df)
if merged.empty:
    st.error("No overlapping dates between the Fear & Greed history and this asset's price history.")
    st.stop()

merged = add_bucket(merged, thresholds)
merged = add_forward_returns(merged, horizons)

bucket_stats_df = bucket_win_stats(merged, horizons)
decile_df = decile_win_stats(merged, headline_horizon)
reg = regression_stats(merged, headline_horizon)
equity_df = compute_equity_curve(merged)

current_value = float(fng_df["value"].iloc[-1])
current_classification_api = fng_df["classification"].iloc[-1]
current_bucket = fng.classify(current_value, thresholds)
current_signal = BUCKET_SIGNAL[current_bucket]
current_lean_pct = current_signal * max_lean_pct

headline_row = bucket_stats_df[
    (bucket_stats_df["bucket"] == current_bucket) & (bucket_stats_df["horizon"] == headline_horizon)
]
headline = headline_row.iloc[0].to_dict() if not headline_row.empty else None

# ---------------------------------------------------------------------------
# Current signal panel
# ---------------------------------------------------------------------------

st.subheader("Current Signal")

sig_cols = st.columns([1, 1, 1, 2])

with sig_cols[0]:
    color = BUCKET_COLORS.get(current_bucket, "#888")
    st.markdown(
        f'<div style="background-color:{color}; padding:1rem; border-radius:0.5rem; '
        f'text-align:center; font-weight:bold; font-size:1.3rem; color:white;">'
        f'{current_value:.0f} — {current_bucket}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"alternative.me label: {current_classification_api}")

with sig_cols[1]:
    lean_dir = "Long" if current_lean_pct > 0 else ("Short" if current_lean_pct < 0 else "Flat")
    st.metric("Suggested delta lean", f"{lean_dir} {abs(current_lean_pct):.0f}%" if current_lean_pct != 0 else "Flat")
    st.caption(f"{selected_asset} · scaled to ±{max_lean_pct}% max")

with sig_cols[2]:
    if headline and headline["n"] > 0 and not np.isnan(headline["win_rate"]):
        st.metric(
            f"Historical win rate ({headline_horizon}d)",
            f"{headline['win_rate'] * 100:.0f}%",
            help="Share of past instances in this bucket where the contrarian "
                 "call (long if fear, short if greed) was the right direction.",
        )
        st.caption(
            f"95% CI [{headline['ci_lo'] * 100:.0f}%, {headline['ci_hi'] * 100:.0f}%] "
            f"· n={headline['n']} · p={headline['p_value']:.3f}"
        )
    elif current_bucket == "Neutral":
        st.metric(f"Historical win rate ({headline_horizon}d)", "—")
        st.caption("Neutral carries no directional call.")
    else:
        st.metric(f"Historical win rate ({headline_horizon}d)", "—")
        st.caption("No historical observations in this bucket at this horizon yet.")

with sig_cols[3]:
    if headline and not np.isnan(headline.get("p_value", np.nan)):
        sig_word = "statistically significant" if headline["p_value"] < 0.05 else "not statistically significant"
        edge_word = "above" if (not np.isnan(headline["win_rate"]) and headline["win_rate"] > 0.5) else "at or below"
        st.markdown(
            f"Win rate is **{edge_word} 50%** and **{sig_word}** at the 5% level "
            f"(p={headline['p_value']:.3f}, n={headline['n']}). "
            + ("Worth weighting the signal." if headline["p_value"] < 0.05 else
               "Treat the signal as noisy at this bucket/horizon — small n or a "
               "genuinely weak edge, the stats alone can't tell you which.")
        )
    else:
        st.markdown("Not enough same-bucket history yet to judge significance — treat with caution.")

st.divider()

# ---------------------------------------------------------------------------
# Price + Fear & Greed chart
# ---------------------------------------------------------------------------

st.subheader(f"{selected_asset} Price vs Fear & Greed Index")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=merged["date"], y=merged["close"], name=f"{selected_asset} Perp",
    line=dict(color=ASSET_COLORS.get(selected_asset, "#fff"), width=1.6),
    yaxis="y1",
))
fig1.add_trace(go.Scatter(
    x=merged["date"], y=merged["value"], name="Fear & Greed", yaxis="y2",
    line=dict(color="#bdbdbd", width=1.2),
))
fig1.add_hrect(y0=0, y1=thresholds["extreme_fear"], yref="y2",
               fillcolor=BUCKET_COLORS["Extreme Fear"], opacity=0.08, line_width=0)
fig1.add_hrect(y0=thresholds["extreme_greed"], y1=100, yref="y2",
               fillcolor=BUCKET_COLORS["Extreme Greed"], opacity=0.08, line_width=0)

fig1.update_layout(
    **PLOTLY_LAYOUT,
    height=420,
    yaxis=dict(title=f"{selected_asset} Price (USD)"),
    yaxis2=dict(title="Fear & Greed (0-100)", overlaying="y", side="right", range=[0, 100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    hovermode="x unified",
)
fx_style.add_watermark(fig1)
st.plotly_chart(fx_style.apply_theme(fig1), width="stretch")
st.caption(
    f"Shaded bands mark Extreme Fear (<{thresholds['extreme_fear']}) and "
    f"Extreme Greed (>{thresholds['extreme_greed']})."
)

# ---------------------------------------------------------------------------
# Bucket win-rate table + chart
# ---------------------------------------------------------------------------

st.subheader(f"Win Rate by Bucket — {headline_horizon}d forward return")

headline_bucket_df = bucket_stats_df[bucket_stats_df["horizon"] == headline_horizon].set_index("bucket").loc[BUCKET_ORDER].reset_index()

fig2 = go.Figure()
plot_df = headline_bucket_df[headline_bucket_df["signal"] != 0]
fig2.add_trace(go.Bar(
    x=plot_df["bucket"], y=plot_df["win_rate"] * 100,
    marker_color=[BUCKET_COLORS[b] for b in plot_df["bucket"]],
    error_y=dict(
        type="data", symmetric=False,
        array=(plot_df["ci_hi"] - plot_df["win_rate"]) * 100,
        arrayminus=(plot_df["win_rate"] - plot_df["ci_lo"]) * 100,
    ),
    text=[f"n={int(n)}" for n in plot_df["n"]],
    textposition="outside",
    name="Win rate",
))
fig2.add_hline(y=50, line_dash="dash", line_color="#888", annotation_text="Coin flip (50%)")
fig2.update_layout(
    **PLOTLY_LAYOUT, height=380,
    yaxis_title="Win rate (%)", yaxis_range=[0, 105],
)
fx_style.add_watermark(fig2)
st.plotly_chart(fx_style.apply_theme(fig2), width="stretch")

display_bucket_df = headline_bucket_df.copy()
display_bucket_df["win_rate"] = display_bucket_df["win_rate"].apply(lambda v: f"{v * 100:.0f}%" if pd.notna(v) else "—")
display_bucket_df["mean_ret"] = display_bucket_df["mean_ret"].apply(lambda v: f"{v * 100:+.2f}%" if pd.notna(v) else "—")
display_bucket_df["median_ret"] = display_bucket_df["median_ret"].apply(lambda v: f"{v * 100:+.2f}%" if pd.notna(v) else "—")
display_bucket_df["p_value"] = display_bucket_df["p_value"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
display_bucket_df["ci"] = headline_bucket_df.apply(
    lambda r: f"[{r['ci_lo'] * 100:.0f}%, {r['ci_hi'] * 100:.0f}%]" if pd.notna(r["ci_lo"]) else "—", axis=1)
st.dataframe(
    display_bucket_df[["bucket", "n", "mean_ret", "median_ret", "win_rate", "ci", "p_value"]].rename(columns={
        "bucket": "Bucket", "n": "N", "mean_ret": "Mean fwd ret", "median_ret": "Median fwd ret",
        "win_rate": "Win rate", "ci": "95% CI", "p_value": "p-value",
    }),
    width="stretch", hide_index=True,
)

with st.expander("All horizons (every bucket × horizon combination)"):
    all_disp = bucket_stats_df.copy()
    all_disp["win_rate"] = all_disp["win_rate"].apply(lambda v: f"{v * 100:.0f}%" if pd.notna(v) else "—")
    all_disp["mean_ret"] = all_disp["mean_ret"].apply(lambda v: f"{v * 100:+.2f}%" if pd.notna(v) else "—")
    all_disp["p_value"] = all_disp["p_value"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    st.dataframe(
        all_disp[["bucket", "horizon", "n", "mean_ret", "win_rate", "p_value"]].rename(columns={
            "bucket": "Bucket", "horizon": "Horizon (d)", "n": "N", "mean_ret": "Mean fwd ret",
            "win_rate": "Win rate", "p_value": "p-value",
        }),
        width="stretch", hide_index=True,
    )

# ---------------------------------------------------------------------------
# Decile view (threshold-free)
# ---------------------------------------------------------------------------

st.subheader(f"Win Rate by F&G Decile (threshold-free) — {headline_horizon}d")
st.caption("Long below 50, short above 50 — shows where the edge concentrates before you pick bucket cutoffs.")

fig3 = go.Figure()
dplot = decile_df[decile_df["direction"] != 0]
fig3.add_trace(go.Bar(
    x=dplot["decile"], y=dplot["win_rate"] * 100,
    marker_color=["#C0392B" if m < 50 else "#1E8449" for m in dplot["mid"]],
    error_y=dict(
        type="data", symmetric=False,
        array=(dplot["ci_hi"] - dplot["win_rate"]) * 100,
        arrayminus=(dplot["win_rate"] - dplot["ci_lo"]) * 100,
    ),
    text=[f"n={int(n)}" for n in dplot["n"]],
    textposition="outside",
))
fig3.add_hline(y=50, line_dash="dash", line_color="#888")
fig3.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title="F&G value range", yaxis_title="Win rate (%)", yaxis_range=[0, 105])
fx_style.add_watermark(fig3)
st.plotly_chart(fx_style.apply_theme(fig3), width="stretch")

# ---------------------------------------------------------------------------
# Scatter + regression
# ---------------------------------------------------------------------------

st.subheader(f"F&G Value vs {headline_horizon}d Forward Return")

scatter_cols = st.columns([2, 1])

with scatter_cols[0]:
    fig4 = go.Figure()
    for bucket in BUCKET_ORDER:
        sub = merged[merged["bucket"] == bucket]
        fig4.add_trace(go.Scatter(
            x=sub["value"], y=sub[f"fwd_{headline_horizon}"] * 100,
            mode="markers", name=bucket,
            marker=dict(size=5, color=BUCKET_COLORS[bucket], opacity=0.55),
        ))
    if reg:
        xline = np.linspace(0, 100, 50)
        yline = (reg["slope"] * xline + reg["intercept"]) * 100
        fig4.add_trace(go.Scatter(
            x=xline, y=yline, mode="lines", name="Regression",
            line=dict(color="#f1c40f", dash="dash", width=2),
        ))
    fig4.update_layout(
        **PLOTLY_LAYOUT, height=380,
        xaxis_title="Fear & Greed value", yaxis_title=f"{headline_horizon}d forward return (%)",
    )
    fx_style.add_watermark(fig4)
    st.plotly_chart(fx_style.apply_theme(fig4), width="stretch")

with scatter_cols[1]:
    st.markdown("**Regression diagnostics**")
    if reg:
        st.markdown(f"- Slope: `{reg['slope'] * 100:.4f}%` return per F&G point")
        st.markdown(f"- Correlation (r): `{reg['r']:.3f}`")
        st.markdown(f"- R²: `{reg['r2']:.4f}`")
        st.markdown(f"- p-value: `{reg['p_value']:.4f}`")
        st.markdown(f"- n: `{reg['n']}`")
        if reg["slope"] < 0 and reg["p_value"] < 0.05:
            st.success("Negative slope, significant — consistent with the contrarian hypothesis (higher F&G → lower forward returns).")
        elif reg["slope"] < 0:
            st.info("Slope points the contrarian way but isn't significant at 5% — could easily be noise.")
        else:
            st.warning("Slope doesn't point the contrarian way over this window/horizon.")
        st.caption("R² this low is normal for a single sentiment input predicting short-term returns — read it as a weak tilt, not a forecast.")
    else:
        st.info("Not enough data for a regression.")

# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

st.subheader("Signal Backtest: Cumulative Return")
st.caption(
    "Daily-rebalanced position = bucket signal (±1 Extreme, ±0.5 Fear/Greed, 0 Neutral) "
    "applied to next-day return. No funding, fees, slippage, or leverage costs modeled — "
    "this isolates the directional signal only, not a tradable strategy."
)

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=equity_df["date"], y=equity_df["strategy_equity"],
                           name="F&G contrarian signal", line=dict(color="#26a69a", width=2)))
fig5.add_trace(go.Scatter(x=equity_df["date"], y=equity_df["buyhold_equity"],
                           name="Buy & hold", line=dict(color=ASSET_COLORS.get(selected_asset, "#fff"), width=1.5, dash="dot")))
fig5.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_title="Growth of $1", hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
fx_style.add_watermark(fig5)
st.plotly_chart(fx_style.apply_theme(fig5), width="stretch")

eq_cols = st.columns(3)
if len(equity_df) > 1:
    strat_total = equity_df["strategy_equity"].iloc[-1] - 1
    bh_total = equity_df["buyhold_equity"].iloc[-1] - 1
    eq_cols[0].metric("Signal total return", f"{strat_total * 100:+.1f}%")
    eq_cols[1].metric("Buy & hold total return", f"{bh_total * 100:+.1f}%")
    eq_cols[2].metric("Days in backtest", f"{len(equity_df)}")

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Telegram Report")

if not is_configured():
    st.info("Telegram not configured. Set credentials in secrets.toml or environment.")
else:
    if st.button("Send Signal Summary to Telegram", type="primary"):
        lines = [f"<b>{selected_asset} Fear &amp; Greed Signal</b>"]
        lines.append(f"Generated: {fx_style.local_now():%Y-%m-%d %H:%M} {fx_style.DISPLAY_TZ_LABEL}")
        lines.append("")
        lines.append(f"F&amp;G: {current_value:.0f} ({current_bucket})")
        lean_txt = f"{'Long' if current_lean_pct > 0 else 'Short' if current_lean_pct < 0 else 'Flat'} {abs(current_lean_pct):.0f}%"
        lines.append(f"Suggested lean: {lean_txt}")
        if headline and headline["n"] > 0 and not np.isnan(headline["win_rate"]):
            lines.append(
                f"Historical win rate ({headline_horizon}d): {headline['win_rate'] * 100:.0f}% "
                f"(n={headline['n']}, p={headline['p_value']:.3f})"
            )
        lines.append("")
        lines.append("Not a trading signal on its own — funding/fees/slippage not modeled.")
        msg = "\n".join(lines)
        if send_message(msg):
            st.success("Signal summary sent to Telegram")
        else:
            st.error("Failed to send to Telegram")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.caption(
    f"Data: alternative.me Crypto Fear & Greed Index (daily) + Deribit perpetual OHLC "
    f"({cfg['perp']}) | {len(merged)} overlapping days | "
    f"Forward-return windows overlap — significance stats are indicative, not "
    f"independent-sample rigorous | Informational only, not financial advice | "
    f"Last refresh: {fx_style.local_now():%H:%M:%S} {fx_style.DISPLAY_TZ_LABEL}"
)
