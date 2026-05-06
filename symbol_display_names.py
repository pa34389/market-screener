"""Human-readable names for tickers in the default universe (dashboard \"Name\" column)."""

from __future__ import annotations

from typing import Dict

# Keys must match symbols used in scans (see universe.py).
_TICKER_NAMES: Dict[str, str] = {
    # Binance USDT spot
    "BTC/USDT": "Bitcoin",
    "ETH/USDT": "Ethereum",
    "SOL/USDT": "Solana",
    "BNB/USDT": "BNB",
    "XRP/USDT": "XRP",
    "DOGE/USDT": "Dogecoin",
    "ADA/USDT": "Cardano",
    "LINK/USDT": "Chainlink",
    "AVAX/USDT": "Avalanche",
    "TRX/USDT": "TRON",
    "DOT/USDT": "Polkadot",
    "POL/USDT": "Polygon (POL)",
    "SHIB/USDT": "Shiba Inu",
    "LTC/USDT": "Litecoin",
    "BCH/USDT": "Bitcoin Cash",
    "UNI/USDT": "Uniswap",
    "ATOM/USDT": "Cosmos",
    "NEAR/USDT": "NEAR Protocol",
    "APT/USDT": "Aptos",
    "FIL/USDT": "Filecoin",
    "ARB/USDT": "Arbitrum",
    "OP/USDT": "Optimism",
    "INJ/USDT": "Injective",
    "TON/USDT": "Toncoin",
    "ETC/USDT": "Ethereum Classic",
    "XLM/USDT": "Stellar",
    "VET/USDT": "VeChain",
    "ICP/USDT": "Internet Computer",
    "ALGO/USDT": "Algorand",
    "SUI/USDT": "Sui",
    # Indexes (Yahoo)
    "^GSPC": "S&P 500 Index",
    "^DJI": "Dow Jones Industrial Average",
    "^IXIC": "Nasdaq Composite",
    "^RUT": "Russell 2000",
    "^FTSE": "FTSE 100",
    "^N225": "Nikkei 225",
    "^HSI": "Hang Seng Index",
    "^GDAXI": "DAX",
    # US equities (Yahoo)
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc. (Class A)",
    "AMZN": "Amazon.com Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "JNJ": "Johnson & Johnson",
    # FX (Yahoo)
    "EURUSD=X": "Euro / US Dollar",
    "GBPUSD=X": "British Pound / US Dollar",
    "USDJPY=X": "US Dollar / Japanese Yen",
    "AUDUSD=X": "Australian Dollar / US Dollar",
    "USDCAD=X": "US Dollar / Canadian Dollar",
    "USDCHF=X": "US Dollar / Swiss Franc",
    "NZDUSD=X": "New Zealand Dollar / US Dollar",
}


def ticker_display_name(symbol: str) -> str:
    """Return a short full name for UI; unknown symbols return the trimmed ticker."""
    key = (symbol or "").strip()
    if not key:
        return "—"
    return _TICKER_NAMES.get(key, key)
