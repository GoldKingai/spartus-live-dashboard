"""Symbol name mapping between canonical names and broker-specific variants.

Each broker may use different naming for the same instrument (e.g. US500
vs SPX500 vs USA500IDXUSD).  Brokers may also add suffixes for account
types (e.g. XAUUSD+ for Raw ECN, XAUUSDm for micro, XAUUSDr for raw).

This module provides a lookup mechanism so the MT5Bridge can resolve
whichever symbol name the connected broker actually exposes.

Resolution order:
    1. Explicit config override (symbol_map in YAML)
    2. Canonical name (e.g. "XAUUSD")
    3. Known alternative names (e.g. "GOLD", "SILVER")
    4. Automatic suffix detection (e.g. "XAUUSD+", "XAUUSDr", "XAUUSD.raw")
       — tries all alternatives with suffixes too

Usage:
    from utils.symbol_mapper import resolve_symbol, BROKER_ALTERNATIVES

    mt5_name = resolve_symbol("US500", available_symbols)
"""

import logging
from typing import Dict, List, Optional, Set

log = logging.getLogger(__name__)

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
    "US500": [
        "SP500", "SPX500", "USA500", "USA500IDXUSD", "SP500m", "US500.cash",
        "US500Cash", "SPX", "SPX500USD", "S&P500", "SandP500",
        "US500USD", "US_500", ".US500", "US500_Cash",
    ],
    "USOIL": [
        "WTI", "CL-OIL", "USOUSD", "LIGHTCMDUSD", "XTIUSD", "OIL.WTI",
        "CRUDEOIL", "UKOIL", "WTIUSD", "OILUSD", "USOILUSD",
    ],
    "XAUUSD": ["GOLD", "GOLDUSD"],
    "EURUSD": ["EURUSDx"],
    "XAGUSD": ["SILVER", "SILVERUSD", "XAGUSDX"],
    "USDJPY": ["USDJPYx"],
}

# ------------------------------------------------------------------
# Fuzzy-match keywords used as a last resort when all else fails.
# Maps canonical names to search terms for scanning available symbols.
# ------------------------------------------------------------------
_FUZZY_KEYWORDS: Dict[str, List[str]] = {
    "US500": ["500", "SPX", "S&P"],
    "USOIL": ["OIL", "WTI", "XTI", "CRUDE", "CL-"],
}

# ------------------------------------------------------------------
# Common broker suffixes for different account types
# The resolver tries each canonical/alternative name with these appended.
# ------------------------------------------------------------------
_BROKER_SUFFIXES: List[str] = [
    "+",       # Vantage Raw ECN (XAUUSD+)
    "m",       # Micro accounts (XAUUSDm)
    "r",       # Raw accounts (XAUUSDr)
    ".raw",    # Raw accounts (XAUUSD.raw)
    ".pro",    # Pro accounts (XAUUSD.pro)
    ".ecn",    # ECN accounts (XAUUSD.ecn)
    ".",       # Some brokers use trailing dot
    "_",       # Some brokers use underscore
    ".sml",    # Small/mini accounts
    "c",       # Cent accounts
    ".std",    # Standard accounts
    "#",       # CFD variants (XAUUSD#)
    "-",       # Dash suffix
]


def _try_with_suffixes(
    base_name: str,
    available_symbols: Set[str],
) -> Optional[str]:
    """Try a base symbol name with all known broker suffixes.

    Args:
        base_name:         Symbol name without suffix (e.g. "XAUUSD")
        available_symbols: Set of symbol names the MT5 terminal reports.

    Returns:
        The matching suffixed name if found, otherwise None.
    """
    for suffix in _BROKER_SUFFIXES:
        candidate = base_name + suffix
        if candidate in available_symbols:
            return candidate
    return None


def resolve_symbol(
    canonical: str,
    available_symbols: Set[str],
    symbol_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Resolve a canonical symbol name to the broker's actual name.

    Resolution order:
        1. Check the explicit *symbol_map* override (config.symbol_map).
        2. Check the canonical name itself.
        3. Try the canonical name with broker suffixes (+, m, r, .raw, etc.)
        4. Walk through BROKER_ALTERNATIVES looking for exact matches.
        5. Try each alternative with broker suffixes.

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

    # 3. Try canonical name with suffixes
    suffixed = _try_with_suffixes(canonical, available_symbols)
    if suffixed:
        log.info("Symbol %s resolved via suffix: %s", canonical, suffixed)
        return suffixed

    # 4. Walk alternatives (exact match)
    alternatives = BROKER_ALTERNATIVES.get(canonical, [])
    for alt in alternatives:
        if alt in available_symbols:
            return alt

    # 5. Try each alternative with suffixes
    for alt in alternatives:
        suffixed = _try_with_suffixes(alt, available_symbols)
        if suffixed:
            log.info("Symbol %s resolved via alternative+suffix: %s", canonical, suffixed)
            return suffixed

    # 6. Fuzzy keyword search (last resort)
    keywords = _FUZZY_KEYWORDS.get(canonical)
    if keywords:
        candidates = []
        for sym in available_symbols:
            sym_upper = sym.upper()
            if any(kw.upper() in sym_upper for kw in keywords):
                candidates.append(sym)
        if len(candidates) == 1:
            log.info("Symbol %s resolved via fuzzy match: %s", canonical, candidates[0])
            return candidates[0]
        elif candidates:
            log.info(
                "Symbol %s: multiple fuzzy matches found: %s — add the correct one to config.symbol_map",
                canonical, ", ".join(sorted(candidates)[:10]),
            )

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
