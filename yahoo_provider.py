"""Yahoo Finance OHLCV via yfinance — same column contract as Binance provider."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

TIMEFRAME_TO_YF: Dict[str, str] = {
    "1m": "1m",
    "3m": "2m",  # yfinance has no 3m; approximate
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "2h": "60m",  # coarse fallback
    "4h": "4h",
    "6h": "4h",
    "8h": "4h",
    "12h": "1d",
    "1d": "1d",
    "3d": "1d",
    "1w": "1wk",
}


def _period_for(interval: str, limit: int) -> str:
    if interval in ("1m", "2m", "5m"):
        return "7d"
    if interval in ("15m", "30m", "60m", "1h"):
        return "60d"
    if interval == "4h":
        return "730d"
    if interval == "1d":
        return "10y"
    if interval == "1wk":
        return "max"
    return "730d"


class YahooOHLCVProvider:
    def __init__(self) -> None:
        try:
            import yfinance as yf  # noqa: F401

            self._yf = yf
        except ImportError as exc:
            raise RuntimeError("Install yfinance for stocks/indexes/FX: pip install yfinance") from exc

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        yf_interval = TIMEFRAME_TO_YF.get(timeframe, "1h")
        period = _period_for(yf_interval, limit)
        ticker = self._yf.Ticker(symbol)
        raw = ticker.history(period=period, interval=yf_interval, auto_adjust=False)
        if raw is None or raw.empty:
            raise ValueError(f"Yahoo Finance returned no rows for {symbol!r} ({yf_interval})")
        df = raw.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df.reset_index()
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "datetime"})
        elif "Date" in df.columns:
            df = df.rename(columns={"Date": "datetime"})
        else:
            df = df.rename(columns={df.columns[0]: "datetime"})
        if df["datetime"].dt.tz is None:
            df["datetime"] = df["datetime"].dt.tz_localize("UTC")
        else:
            df["datetime"] = df["datetime"].dt.tz_convert("UTC")
        df = df[["datetime", "open", "high", "low", "close", "volume"]]
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.tail(max(50, min(limit, len(df)))).reset_index(drop=True)
        return df
