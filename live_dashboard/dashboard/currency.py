"""Dynamic currency formatting for the Spartus Live Dashboard.

Provides a single source of truth for the account currency symbol.
Call ``set_currency("GBP")`` once at startup (from main.py after MT5
connects), then use ``sym()`` or ``fmt()`` everywhere in the UI.

XAUUSD *prices* are always in USD regardless of account currency.
Only balance, equity, margin, and P/L values use the account currency.
"""

from typing import Dict

# ISO 4217 code -> display symbol
CURRENCY_SYMBOLS: Dict[str, str] = {
    "USD": "$",
    "GBP": "\u00a3",    # £
    "EUR": "\u20ac",     # €
    "JPY": "\u00a5",     # ¥
    "AUD": "A$",
    "CAD": "C$",
    "CHF": "Fr",
    "NZD": "NZ$",
    "SGD": "S$",
    "HKD": "HK$",
    "ZAR": "R",
    "PLN": "z\u0142",   # zł
    "CZK": "K\u010d",   # Kč
    "HUF": "Ft",
    "SEK": "kr",
    "NOK": "kr",
    "DKK": "kr",
    "MXN": "Mex$",
    "BRL": "R$",
    "INR": "\u20b9",     # ₹
    "CNY": "\u00a5",     # ¥
    "KRW": "\u20a9",     # ₩
    "THB": "\u0e3f",     # ฿
    "RUB": "\u20bd",     # ₽
    "TRY": "\u20ba",     # ₺
}

_code: str = "USD"
_sym: str = "$"


def set_currency(code: str) -> None:
    """Set the account currency (called once after MT5 connects)."""
    global _code, _sym
    _code = code.upper()
    _sym = CURRENCY_SYMBOLS.get(_code, _code)


def sym() -> str:
    """Return the currency symbol (e.g. '$', '£', '€')."""
    return _sym


def code() -> str:
    """Return the 3-letter ISO currency code (e.g. 'USD', 'GBP')."""
    return _code


def fmt(value: float, decimals: int = 2) -> str:
    """Format a monetary value with the account currency symbol.

    Example: fmt(1234.5) -> '£1,234.50' (if account is GBP)
    """
    return f"{_sym}{value:,.{decimals}f}"


def fmt_signed(value: float, decimals: int = 2) -> str:
    """Format with explicit +/- sign prefix.

    Example: fmt_signed(42.0)  -> '+£42.00'
             fmt_signed(-7.3)  -> '-£7.30'
    """
    if value >= 0:
        return f"+{_sym}{value:,.{decimals}f}"
    return f"-{_sym}{abs(value):,.{decimals}f}"
