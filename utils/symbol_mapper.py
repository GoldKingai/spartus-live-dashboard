"""Symbol name mapping between canonical names and broker-specific variants.

Each broker may use different naming for the same instrument (e.g. US500
vs SPX500 vs USA500IDXUSD).  This module provides a lookup mechanism so
the MT5Bridge can resolve whichever symbol name the connected broker
actually exposes.

Usage:
    from utils.symbol_mapper import resolve_symbol, BROKER_ALTERNATIVES

    mt5_name = resolve_symbol("US500", available_symbols)
"""

from typing import Dict, List, Optional, Set

# ------------------------------------------------------------------
# Default 1:1 mapping (canonical -> expected MT5 name)
# ------------------------------------------------------------------
SYMBOL_MAP_DEFAULT: Dict[str, str] = {
    "XAUUSD": "XAUUSD",
    "EURUSD": "EURUSD",
    "XAGUSD": "XAGUSD",
    "USDJPY": "USDJPY",
    "US500": "US500",
    "USOIL": "USOIL",
}

# ------------------------------------------------------------------
# Alternative names brokers commonly use for each instrument
# ------------------------------------------------------------------
BROKER_ALTERNATIVES: Dict[str, List[str]] = {
    "US500": ["SPX500", "USA500", "USA500IDXUSD", "SP500m", "US500.cash"],
    "USOIL": ["WTI", "CL-OIL", "USOUSD", "LIGHTCMDUSD", "XTIUSD", "OIL.WTI"],
    "XAUUSD": ["GOLD", "XAUUSDm"],
    "EURUSD": ["EURUSDm"],
    "XAGUSD": ["SILVER", "XAGUSDm"],
    "USDJPY": ["USDJPYm"],
}


def resolve_symbol(
    canonical: str,
    available_symbols: Set[str],
    symbol_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Resolve a canonical symbol name to the broker's actual name.

    Resolution order:
        1. Check the explicit *symbol_map* override (config.symbol_map).
        2. Check the canonical name itself.
        3. Walk through BROKER_ALTERNATIVES looking for a match.

    Args:
        canonical:         Standardised name (e.g. "XAUUSD", "US500").
        available_symbols: Set of symbol names the MT5 terminal reports.
        symbol_map:        Optional config-level overrides.

    Returns:
        The broker symbol name if found, otherwise ``None``.
    """
    # 1. Explicit config override
    if symbol_map:
        mapped = symbol_map.get(canonical)
        if mapped and mapped in available_symbols:
            return mapped

    # 2. Try the canonical name directly
    if canonical in available_symbols:
        return canonical

    # 3. Walk alternatives
    alternatives = BROKER_ALTERNATIVES.get(canonical, [])
    for alt in alternatives:
        if alt in available_symbols:
            return alt

    return None


def build_resolved_map(
    available_symbols: Set[str],
    symbol_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build a full canonical -> broker-name mapping for all known symbols.

    Args:
        available_symbols: Set of symbol names the MT5 terminal reports.
        symbol_map:        Optional config-level overrides.

    Returns:
        Dict mapping canonical names to resolved broker names.
        Symbols that could not be resolved are omitted.
    """
    resolved: Dict[str, str] = {}
    for canonical in SYMBOL_MAP_DEFAULT:
        broker_name = resolve_symbol(canonical, available_symbols, symbol_map)
        if broker_name is not None:
            resolved[canonical] = broker_name
    return resolved
