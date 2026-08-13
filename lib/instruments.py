"""
MCM Analytics — Deribit instrument name parsing and helpers.
"""

import re
from datetime import datetime, timezone
from dataclasses import dataclass


@dataclass
class ParsedInstrument:
    """Parsed components of a Deribit instrument name."""
    currency: str       # BTC, ETH, SOL_USDC, HYPE_USDC
    expiry_str: str     # e.g. "27JUN26"
    strike: float       # e.g. 120000.0
    kind: str           # "C" or "P"
    raw: str            # original instrument name

    @property
    def expiry_date(self) -> datetime:
        """Parse expiry string to datetime (08:00 UTC on expiry day)."""
        return parse_expiry_date(self.expiry_str)

    @property
    def base_currency(self) -> str:
        """Get the base currency (BTC, ETH, SOL, HYPE)."""
        if "_USDC" in self.currency:
            return self.currency.replace("_USDC", "")
        return self.currency


# Month code mappings
MONTH_CODES = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Regex for Deribit option instrument names
# Examples: BTC-27JUN26-120000-C, SOL_USDC-27JUN26-6d4-P
_INSTRUMENT_RE = re.compile(
    r"^([A-Z_]+)-(\d{1,2}[A-Z]{3}\d{2})-([0-9d]+)-([CP])$"
)


def parse_instrument(name: str) -> ParsedInstrument | None:
    """
    Parse a Deribit option instrument name into its components.
    Returns None if the name doesn't match the expected format.
    """
    m = _INSTRUMENT_RE.match(name)
    if not m:
        return None

    currency = m.group(1)
    expiry_str = m.group(2)
    strike_str = m.group(3)
    kind = m.group(4)

    # Parse strike: "120000" or "6d4" (= 6.4 for linear)
    strike = parse_strike(strike_str)

    return ParsedInstrument(
        currency=currency,
        expiry_str=expiry_str,
        strike=strike,
        kind=kind,
        raw=name,
    )


def parse_strike(s: str) -> float:
    """Parse a Deribit strike string. 'd' means decimal point (e.g. '6d4' = 6.4)."""
    return float(s.replace("d", "."))


def parse_expiry_date(expiry_str: str) -> datetime:
    """
    Parse Deribit expiry string (e.g. '27JUN26') to datetime.
    Deribit options expire at 08:00 UTC on the expiry day.
    """
    # Extract day, month, year
    day = int(expiry_str[:len(expiry_str) - 5])
    month_str = expiry_str[-5:-2]
    year = 2000 + int(expiry_str[-2:])
    month = MONTH_CODES.get(month_str, 1)
    return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)


def dte_from_expiry(expiry_ts_ms: int) -> float:
    """Compute days-to-expiry from a millisecond timestamp."""
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    return max((expiry_ts_ms - now_ms) / (1000 * 86400), 0.0)


def format_strike(strike: float, dp: int = 0) -> str:
    """Format a strike price for display (e.g. 120000 → '120k')."""
    if dp == 0 and strike >= 1000:
        if strike % 1000 == 0:
            return f"{int(strike // 1000)}k"
        return f"{strike:,.0f}"
    if dp > 0:
        return f"{strike:.{dp}f}"
    return f"{strike:,.0f}"
