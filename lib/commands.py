"""
MCM Analytics — bot command registry and dispatcher.

Mirrors the FalconX bot's command surface, backed entirely by Deribit data.
"""

from __future__ import annotations

from lib import cmd_market, cmd_vol

# Commands that take an expiry / DTE choice.
EXPIRY_COMMANDS = {
    "vol_time_series", "skew_time_series", "intraday_basis",
    "intraday_vol", "intraday_skew", "vol_smile",
}

DTE_PRESETS = {"1 week": 7, "1 month": 30, "3 month": 90}

COMMANDS = [
    "/vol_run",
    "/vol_term_structure",
    "/skew_term_structure",
    "/butterfly_term_structure",
    "/basis_run",
    "/forward_vols",
    "/forward_vol_steepness",
    "/forward_vol_steepness_25d_call",
    "/forward_vol_steepness_25d_put",
    "/forward_vol_steepness_multidelta",
    "/atm_iv_box_plot",
    "/forward_vol_matrix",
    "/intraday_basis",
    "/intraday_vol",
    "/intraday_skew",
    "/vol_time_series",
    "/skew_time_series",
    "/vol_smile",
    "/plot_funding_rates",
    "/rv_plot",
    "/block_trades_summary",
    "/moonphase",
]

COMMAND_NAMES = tuple(c.lstrip("/") for c in COMMANDS)

COMMAND_HELP = {
    "basis_run": "Shows basis of perp and dated futures on Deribit.",
    "vol_run": "Displays forward vol as of current timestamp.",
    "vol_term_structure": "Shows implied volatility across different listed expiries.",
    "rv_plot": "Displays realized volatility time series.",
    "block_trades_summary": "Displays table of significant market trades by volume as of current timestamp.",
    "skew_term_structure": "Shows option skew across different listed expiries.",
    "butterfly_term_structure": "Shows 10Δ butterfly (wings vs ATM) across different listed expiries.",
    "plot_funding_rates": "Displays historical Deribit perpetual funding rates.",
    "vol_smile": "Shows volatility smile for options.",
    "vol_time_series": "Displays volatility time series.",
    "skew_time_series": "Displays skew time series.",
    "forward_vols": "Displays combined ATM/25d call/25d put forward volatility term structure.",
    "forward_vol_steepness": "ATM-only $100k vega daily carry waterfall, weighted to 30D vega equivalence.",
    "forward_vol_steepness_25d_call": "25d call $100k vega daily carry waterfall, weighted to 30D vega equivalence.",
    "forward_vol_steepness_25d_put": "25d put $100k vega daily carry waterfall, weighted to 30D vega equivalence.",
    "forward_vol_steepness_multidelta": "ATM/25d call/25d put carry(pts/30d) grouped bar chart by tenor pair.",
    "atm_iv_box_plot": "ATM IV 90D range box plot by tenor with the current term structure overlaid.",
    "forward_vol_matrix": "Forward vol matrix heatmap between all expiries.",
    "intraday_basis": "Shows today's basis movements.",
    "intraday_vol": "Shows today's volatility movements.",
    "intraday_skew": "Shows today's skew movements.",
    "moonphase": "Displays current lunar phase 🌙",
}

HANDLERS = {
    "vol_run": cmd_vol.cmd_vol_run,
    "vol_term_structure": cmd_vol.cmd_vol_term_structure,
    "skew_term_structure": cmd_vol.cmd_skew_term_structure,
    "butterfly_term_structure": cmd_vol.cmd_butterfly_term_structure,
    "basis_run": cmd_market.cmd_basis_run,
    "forward_vols": cmd_vol.cmd_forward_vols,
    "forward_vol_steepness": cmd_vol.cmd_forward_vol_steepness,
    "forward_vol_steepness_25d_call": cmd_vol.cmd_forward_vol_steepness_25d_call,
    "forward_vol_steepness_25d_put": cmd_vol.cmd_forward_vol_steepness_25d_put,
    "forward_vol_steepness_multidelta": cmd_vol.cmd_forward_vol_steepness_multidelta,
    "atm_iv_box_plot": cmd_vol.cmd_atm_iv_box_plot,
    "forward_vol_matrix": cmd_vol.cmd_forward_vol_matrix,
    "intraday_basis": cmd_market.cmd_intraday_basis,
    "intraday_vol": cmd_vol.cmd_intraday_vol,
    "intraday_skew": cmd_vol.cmd_intraday_skew,
    "vol_time_series": cmd_vol.cmd_vol_time_series,
    "skew_time_series": cmd_vol.cmd_skew_time_series,
    "vol_smile": cmd_vol.cmd_vol_smile,
    "plot_funding_rates": cmd_market.cmd_plot_funding_rates,
    "rv_plot": cmd_market.cmd_rv_plot,
    "block_trades_summary": cmd_market.cmd_block_trades_summary,
    "moonphase": cmd_market.cmd_moonphase,
}


def short_error(exc: BaseException) -> str:
    msg = str(exc).strip() or exc.__class__.__name__
    return msg if len(msg) <= 180 else msg[:177] + "..."


def run_command(asset: str, command: str, expiry_target_days: int | None = None):
    """
    Execute one command.

    Returns ``(figure_or_list, dataframe, text)``.  Errors are returned as the
    text element rather than raised, so a dashboard sweep never dies on one
    bad tile.
    """
    name = command.strip().lstrip("/").split()[0]
    handler = HANDLERS.get(name)
    if handler is None:
        return None, None, f"Unknown command: /{name}"
    kwargs = {}
    if name in EXPIRY_COMMANDS and expiry_target_days is not None:
        kwargs["dte"] = int(expiry_target_days)
    try:
        return handler(asset, **kwargs)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI as text
        return None, None, f"/{name} failed: {short_error(exc)}"
