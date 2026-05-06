"""
Default screening universe: liquid Binance USDT pairs + Yahoo Finance (indexes, FX, majors).

Yahoo symbols use yfinance (unofficial, rate limits apply). Not for HFT.
"""

from __future__ import annotations

from typing import Final, FrozenSet, Tuple

# Top ~30 liquid Binance USDT spot pairs (US-style "BASE/QUOTE" for app; API maps to BASEUSDT).
BINANCE_USDT: Final[Tuple[str, ...]] = (
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "LINK/USDT",
    "AVAX/USDT",
    "TRX/USDT",
    "DOT/USDT",
    "POL/USDT",
    "SHIB/USDT",
    "LTC/USDT",
    "BCH/USDT",
    "UNI/USDT",
    "ATOM/USDT",
    "NEAR/USDT",
    "APT/USDT",
    "FIL/USDT",
    "ARB/USDT",
    "OP/USDT",
    "INJ/USDT",
    "TON/USDT",
    "ETC/USDT",
    "XLM/USDT",
    "VET/USDT",
    "ICP/USDT",
    "ALGO/USDT",
    "SUI/USDT",
)

# Major US / global equity indexes (^ = index in Yahoo).
YAHOO_INDEXES: Final[Tuple[str, ...]] = (
    "^GSPC",  # S&P 500
    "^DJI",  # Dow
    "^IXIC",  # Nasdaq Composite
    "^RUT",  # Russell 2000
    "^FTSE",  # FTSE 100 (may vary by region)
    "^N225",  # Nikkei 225
    "^HSI",  # Hang Seng
    "^GDAXI",  # DAX
)

# Large-cap US names (Yahoo tickers).
YAHOO_STOCKS: Final[Tuple[str, ...]] = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
    "V",
    "JNJ",
)

# FX vs USD (=X suffix on Yahoo).
YAHOO_FX: Final[Tuple[str, ...]] = (
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "NZDUSD=X",
)

BINANCE_SYMBOL_SET: Final[FrozenSet[str]] = frozenset(BINANCE_USDT)


def feed_for_symbol(symbol: str) -> str:
    """Return 'binance' or 'yahoo' for OHLCV routing."""
    sym = symbol.strip()
    su = sym.upper()
    if "/" in sym and su.endswith("/USDT"):
        return "binance"
    if sym in BINANCE_SYMBOL_SET:
        return "binance"
    return "yahoo"


def default_symbol_universe() -> Tuple[str, ...]:
    return BINANCE_USDT + YAHOO_INDEXES + YAHOO_STOCKS + YAHOO_FX
