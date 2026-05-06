"""
Meridian setup scout — HTF structure, POI, LTF sweeps, DAG scoring (MVP).

Uses Binance public REST klines by default. No ccxt required.

Install:
    python3 -m venv .venv && source .venv/bin/activate  # recommended on PEP 668 systems
    pip install -r requirements.txt

Run once:
    python setup_scout.py

Run tests:
    python setup_scout.py --test

Run synthetic/offline:
    python setup_scout.py --provider synthetic

Dashboard (HTML + SQLite activity log, Binance by default):
    pip install -r requirements.txt
    python webapp.py
    Open http://127.0.0.1:5050/ — optional: MARKET_SCREENER_POLL_SECONDS=0 to disable hourly background scans
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import unittest
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Tuple, cast

from state_machine import (
    DagState,
    compute_dag_state,
    compute_score_components,
    grade_from_score,
    legacy_status_from_dag,
    missing_conditions_for_state,
    should_alert_dag,
)
from symbol_display_names import ticker_display_name
from universe import default_symbol_universe, feed_for_symbol

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv() -> bool:
        return False


try:
    import requests
except ImportError:
    requests = None

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

Direction = Literal["bullish", "bearish", "neutral"]
SetupStatus = Literal["invalid", "watch", "poi_entered", "sweep_seen", "confirmed"]
POIType = Literal["bullish_order_block", "bearish_order_block", "bullish_fvg", "bearish_fvg"]
DataProviderName = Literal["binance", "synthetic", "routed"]


@dataclass(frozen=True)
class Config:
    """`data_provider=routed`: Binance for */USDT crypto, Yahoo for indexes/equities/FX."""

    data_provider: DataProviderName = "routed"
    symbols: Tuple[str, ...] = field(default_factory=default_symbol_universe)
    htf: str = "4h"
    ltf: str = "30m"
    limit: int = 300
    htf_swing_left_right: int = 5
    ltf_swing_left_right: int = 3
    displacement_atr_mult: float = 1.2
    poi_lookback_after_msb: int = 20
    min_rr: float = 2.0
    near_poi_atr_mult: float = 0.25
    max_candles_since_msb: int = 80
    poll_seconds: int = 300
    use_ai_summary: bool = False
    send_telegram: bool = False
    binance_base_url: str = "https://api.binance.com"
    request_timeout: int = 15


@dataclass(frozen=True)
class Zone:
    zone_type: POIType
    low: float
    high: float
    created_at: str
    source_timeframe: str

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high


@dataclass(frozen=True)
class StructureState:
    bias: Direction
    msb_found: bool
    msb_price: Optional[float]
    range_low: Optional[float]
    range_high: Optional[float]
    midpoint: Optional[float]
    external_liquidity_target: Optional[float]
    last_swing_high: Optional[float]
    last_swing_low: Optional[float]


@dataclass(frozen=True)
class LTFState:
    liquidity_sweep: bool
    sweep_direction: Optional[Direction]
    swept_level: Optional[float]
    ltf_msb: bool
    ltf_msb_direction: Optional[Direction]
    invalidation: Optional[float]
    engineered_liquidity: bool = False
    breaker_retest_seen: bool = False


@dataclass
class SetupSignal:
    symbol: str
    htf: str
    ltf: str
    status: SetupStatus
    score: int
    bias: Direction
    current_price: float
    range_low: Optional[float]
    range_high: Optional[float]
    midpoint: Optional[float]
    range_location: str
    poi: Optional[Zone]
    ltf_state: LTFState
    target: Optional[float]
    invalidation: Optional[float]
    rr: Optional[float]
    alert: str
    dag_state: DagState = "NO_SETUP"
    classification: str = "Ignore"
    missing_conditions: Tuple[str, ...] = ()
    data_feed: str = "binance"
    structured: Dict[str, Any] = field(default_factory=dict)


class OHLCVProvider(Protocol):
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        ...


class BinanceOHLCVProvider:
    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        timeout: int = 15,
        session: Optional[Any] = None,
    ) -> None:
        if session is None and requests is None:
            raise RuntimeError("requests is not installed. Install it with: pip install requests")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        params = {
            "symbol": to_binance_symbol(symbol),
            "interval": timeframe,
            "limit": clamp_limit(limit, 1, 1000),
        }
        response = self.session.get(f"{self.base_url}/api/v3/klines", params=params, timeout=self.timeout)
        status_code = getattr(response, "status_code", 200)
        data = response.json()
        if status_code >= 400:
            raise RuntimeError(f"Binance API error {status_code}: {data}")
        if isinstance(data, dict) and "code" in data:
            raise RuntimeError(f"Binance API error {data.get('code')}: {data.get('msg', data)}")
        return normalise_binance_klines(data)


class SyntheticOHLCVProvider:
    def __init__(self, seed: int = 7) -> None:
        self.seed = seed

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        rng_seed = self.seed + sum(ord(c) for c in symbol + timeframe)
        rng = np.random.default_rng(rng_seed)
        n = max(limit, 140)
        base = 100.0 if symbol.upper().startswith("BTC") else 50.0
        drift = np.concatenate(
            [
                np.linspace(0, 18, n // 3),
                np.linspace(18, 6, n // 3),
                np.linspace(6, 24, n - 2 * (n // 3)),
            ]
        )
        noise = rng.normal(0, 0.8, size=n).cumsum() * 0.12
        close = base + drift + noise
        open_ = np.roll(close, 1)
        open_[0] = close[0] - 0.2
        spread = rng.uniform(0.4, 1.4, size=n)
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        volume = rng.uniform(100, 1000, size=n)
        if timeframe.lower() in {"4h", "1d", "12h"} and n > 90:
            close[-24:-14] = close[-25] + np.linspace(1, 14, 10)
            open_[-24:-14] = close[-24:-14] - 0.6
            high[-24:-14] = close[-24:-14] + 1.5
            low[-24:-14] = close[-24:-14] - 1.0
            close[-13:] = close[-14] - np.linspace(1, 8, 13)
            open_[-13:] = close[-13:] + 0.4
            high[-13:] = close[-13:] + 1.2
            low[-13:] = close[-13:] - 1.2
        if timeframe.lower() in {"30m", "15m", "1h"} and n > 90:
            local_low = float(np.min(low[-45:-10]))
            low[-4] = local_low - 1.0
            close[-4] = local_low + 0.4
            open_[-4] = local_low + 0.9
            prior_high = float(np.max(high[-35:-5]))
            high[-1] = prior_high + 1.0
            close[-1] = high[-1] - 0.2
            open_[-1] = close[-1] - 0.6
        dates = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq=timeframe_to_pandas_freq(timeframe))
        df = pd.DataFrame({"datetime": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
        return df.tail(limit).reset_index(drop=True)


def clamp_limit(value: int, low: int, high: int) -> int:
    return max(low, min(int(value), high))


def to_binance_symbol(symbol: str) -> str:
    cleaned = symbol.upper().replace("/", "").replace("-", "").replace("_", "").strip()
    if not cleaned:
        raise ValueError("Symbol cannot be empty")
    return cleaned


def timeframe_to_pandas_freq(timeframe: str) -> str:
    mapping = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1D",
        "3d": "3D",
        "1w": "1W",
    }
    return mapping.get(timeframe, "30min")


def normalise_binance_klines(raw: Sequence[Sequence[Any]]) -> pd.DataFrame:
    if not isinstance(raw, Sequence) or len(raw) == 0:
        raise ValueError("Binance kline response was empty")
    rows = []
    for item in raw:
        if len(item) < 6:
            raise ValueError(f"Invalid Binance kline row: {item}")
        rows.append([item[0], item[1], item[2], item[3], item[4], item[5]])
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    return df.reset_index(drop=True)


class RoutedOHLCVProvider:
    """Binance for crypto USDT; Yahoo Finance for indexes, equities, FX."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._synthetic = SyntheticOHLCVProvider()
        self._binance = BinanceOHLCVProvider(base_url=cfg.binance_base_url, timeout=cfg.request_timeout)
        self._yahoo: Optional[Any] = None

    def _yahoo_provider(self) -> Any:
        if self._yahoo is None:
            from yahoo_provider import YahooOHLCVProvider

            self._yahoo = YahooOHLCVProvider()
        return self._yahoo

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        if self.cfg.data_provider == "synthetic":
            return self._synthetic.fetch_ohlcv(symbol, timeframe, limit)
        if feed_for_symbol(symbol) == "yahoo":
            return self._yahoo_provider().fetch_ohlcv(symbol, timeframe, limit)
        return self._binance.fetch_ohlcv(symbol, timeframe, limit)


def make_provider(cfg: Config) -> OHLCVProvider:
    if cfg.data_provider == "synthetic":
        return SyntheticOHLCVProvider()
    if cfg.data_provider == "binance":
        return BinanceOHLCVProvider(base_url=cfg.binance_base_url, timeout=cfg.request_timeout)
    if cfg.data_provider == "routed":
        return RoutedOHLCVProvider(cfg)
    raise ValueError(f"Unknown data provider: {cfg.data_provider!r}")


def fetch_daily_close_series(provider: OHLCVProvider, symbol: str, days: int = 7) -> List[float]:
    """Last ``days`` daily closes, oldest first (left to right on a sparkline)."""
    if days < 2:
        days = 2
    limit = max(days + 2, 10)
    df = provider.fetch_ohlcv(symbol, "1d", limit)
    validate_ohlcv(df)
    closes = [float(x) for x in df["close"].tolist()]
    if len(closes) >= days:
        return closes[-days:]
    return closes


def validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"OHLCV data missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("OHLCV data is empty")


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    validate_ohlcv(df)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_swings(df: pd.DataFrame, left_right: int = 3) -> pd.DataFrame:
    validate_ohlcv(df)
    if left_right < 1:
        raise ValueError("left_right must be >= 1")
    out = df.copy().reset_index(drop=True)
    window = left_right * 2 + 1
    out["swing_high"] = False
    out["swing_low"] = False
    if len(out) < window:
        return out
    rolling_high = out["high"].rolling(window=window, center=True).max()
    rolling_low = out["low"].rolling(window=window, center=True).min()
    out.loc[out["high"] == rolling_high, "swing_high"] = True
    out.loc[out["low"] == rolling_low, "swing_low"] = True
    out.loc[out.index[:left_right], ["swing_high", "swing_low"]] = False
    out.loc[out.index[-left_right:], ["swing_high", "swing_low"]] = False
    return out


def add_swings_confirmed(df: pd.DataFrame, left_right: int = 3) -> pd.DataFrame:
    """Swing only confirmed after `left_right` bars on each side (no lookahead on the last bars)."""
    validate_ohlcv(df)
    if left_right < 1:
        raise ValueError("left_right must be >= 1")
    out = df.copy().reset_index(drop=True)
    n = left_right
    out["swing_high"] = False
    out["swing_low"] = False
    if len(out) < n * 2 + 1:
        return out
    for i in range(n, len(out) - n):
        window_h = out["high"].iloc[i - n : i + n + 1]
        if float(out["high"].iloc[i]) == float(window_h.max()):
            out.loc[i, "swing_high"] = True
        window_l = out["low"].iloc[i - n : i + n + 1]
        if float(out["low"].iloc[i]) == float(window_l.min()):
            out.loc[i, "swing_low"] = True
    return out


def get_recent_swings(df: pd.DataFrame, count: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return df[df["swing_high"]].tail(count), df[df["swing_low"]].tail(count)


def detect_htf_structure(df: pd.DataFrame, cfg: Config) -> StructureState:
    work = add_swings_confirmed(df, cfg.htf_swing_left_right)
    swing_highs, swing_lows = get_recent_swings(work, count=12)
    if swing_highs.empty or swing_lows.empty:
        return StructureState("neutral", False, None, None, None, None, None, None, None)
    last_swing_high = float(swing_highs.iloc[-1]["high"])
    last_swing_low = float(swing_lows.iloc[-1]["low"])
    events: List[Tuple[int, Direction, float]] = []
    for idx, row in swing_highs.iterrows():
        breaks = work.loc[int(idx) + 1 :][work.loc[int(idx) + 1 :, "close"] > float(row["high"])]
        if not breaks.empty:
            events.append((int(breaks.index[0]), "bullish", float(row["high"])))
    for idx, row in swing_lows.iterrows():
        breaks = work.loc[int(idx) + 1 :][work.loc[int(idx) + 1 :, "close"] < float(row["low"])]
        if not breaks.empty:
            events.append((int(breaks.index[0]), "bearish", float(row["low"])))
    recent_floor = max(0, len(work) - cfg.max_candles_since_msb)
    events = [event for event in events if event[0] >= recent_floor]
    if not events:
        return StructureState("neutral", False, None, None, None, None, None, last_swing_high, last_swing_low)
    break_idx, bias, msb_price = max(events, key=lambda item: item[0])
    if bias == "bullish":
        prior_lows = work.loc[:break_idx][work.loc[:break_idx, "swing_low"]]
        if prior_lows.empty:
            return StructureState("neutral", False, None, None, None, None, None, last_swing_high, last_swing_low)
        range_low = float(prior_lows.iloc[-1]["low"])
        range_high = float(work.loc[break_idx:, "high"].max())
        midpoint = (range_low + range_high) / 2
        return StructureState(
            "bullish",
            True,
            msb_price,
            range_low,
            range_high,
            midpoint,
            range_high,
            last_swing_high,
            last_swing_low,
        )
    prior_highs = work.loc[:break_idx][work.loc[:break_idx, "swing_high"]]
    if prior_highs.empty:
        return StructureState("neutral", False, None, None, None, None, None, last_swing_high, last_swing_low)
    range_high = float(prior_highs.iloc[-1]["high"])
    range_low = float(work.loc[break_idx:, "low"].min())
    midpoint = (range_low + range_high) / 2
    return StructureState(
        "bearish",
        True,
        msb_price,
        range_low,
        range_high,
        midpoint,
        range_low,
        last_swing_high,
        last_swing_low,
    )


def range_location(price: float, structure: StructureState) -> str:
    if structure.range_low is None or structure.range_high is None or structure.midpoint is None:
        return "unknown"
    return "premium" if price >= structure.midpoint else "discount"


def price_in_valid_half(price: float, structure: StructureState) -> bool:
    location = range_location(price, structure)
    return (structure.bias == "bullish" and location == "discount") or (structure.bias == "bearish" and location == "premium")


def detect_fvgs(df: pd.DataFrame, timeframe: str) -> List[Zone]:
    validate_ohlcv(df)
    zones: List[Zone] = []
    for i in range(2, len(df)):
        c1 = df.iloc[i - 2]
        c3 = df.iloc[i]
        if float(c1["high"]) < float(c3["low"]):
            zones.append(Zone("bullish_fvg", float(c1["high"]), float(c3["low"]), str(c3["datetime"]), timeframe))
        if float(c1["low"]) > float(c3["high"]):
            zones.append(Zone("bearish_fvg", float(c3["high"]), float(c1["low"]), str(c3["datetime"]), timeframe))
    return zones


def detect_order_block(df: pd.DataFrame, structure: StructureState, cfg: Config, timeframe: str) -> Optional[Zone]:
    if structure.bias == "neutral" or not structure.msb_found:
        return None
    validate_ohlcv(df)
    work = df.copy().reset_index(drop=True)
    work["atr"] = atr(work)
    work["body"] = (work["close"] - work["open"]).abs()
    recent = work.tail(cfg.poi_lookback_after_msb + 30)
    if structure.bias == "bullish":
        displacement = recent[
            (recent["close"] > recent["open"]) & (recent["body"] > recent["atr"] * cfg.displacement_atr_mult)
        ]
        if displacement.empty:
            return None
        prior = work.loc[: int(displacement.index[-1])].tail(cfg.poi_lookback_after_msb)
        candles = prior[prior["close"] < prior["open"]]
        if candles.empty:
            return None
        candle = candles.iloc[-1]
        return Zone("bullish_order_block", float(candle["low"]), float(candle["high"]), str(candle["datetime"]), timeframe)
    displacement = recent[
        (recent["close"] < recent["open"]) & (recent["body"] > recent["atr"] * cfg.displacement_atr_mult)
    ]
    if displacement.empty:
        return None
    prior = work.loc[: int(displacement.index[-1])].tail(cfg.poi_lookback_after_msb)
    candles = prior[prior["close"] > prior["open"]]
    if candles.empty:
        return None
    candle = candles.iloc[-1]
    return Zone("bearish_order_block", float(candle["low"]), float(candle["high"]), str(candle["datetime"]), timeframe)


def select_valid_poi(df: pd.DataFrame, structure: StructureState, cfg: Config, timeframe: str) -> Optional[Zone]:
    validate_ohlcv(df)
    price = float(df["close"].iloc[-1])
    candidates: List[Zone] = []
    order_block = detect_order_block(df, structure, cfg, timeframe)
    if order_block:
        candidates.append(order_block)
    fvgs = detect_fvgs(df, timeframe)
    if structure.bias == "bullish":
        candidates.extend([zone for zone in fvgs if zone.zone_type == "bullish_fvg"][-5:])
    elif structure.bias == "bearish":
        candidates.extend([zone for zone in fvgs if zone.zone_type == "bearish_fvg"][-5:])
    if not candidates or structure.midpoint is None:
        return None
    valid = []
    for zone in candidates:
        zone_mid = (zone.low + zone.high) / 2
        if structure.bias == "bullish" and zone_mid <= structure.midpoint:
            valid.append(zone)
        if structure.bias == "bearish" and zone_mid >= structure.midpoint:
            valid.append(zone)
    if not valid:
        return None
    return min(valid, key=lambda zone: 0.0 if zone.contains(price) else min(abs(price - zone.low), abs(price - zone.high)))


def _engineered_liquidity(
    work: pd.DataFrame,
    bias: Direction,
    poi: Zone,
    lookback: int = 25,
) -> bool:
    tail = work.tail(lookback)
    if bias == "bullish":
        lows = tail[tail["swing_low"]]
        for _, r in lows.iterrows():
            lv = float(r["low"])
            if poi.low <= lv <= poi.high:
                return True
        return False
    highs = tail[tail["swing_high"]]
    for _, r in highs.iterrows():
        hv = float(r["high"])
        if poi.low <= hv <= poi.high:
            return True
    return False


def detect_ltf_state(
    df: pd.DataFrame,
    bias: Direction,
    cfg: Config,
    poi: Optional[Zone],
    scan_ltf: bool,
) -> LTFState:
    if bias == "neutral" or not scan_ltf:
        return LTFState(False, None, None, False, None, None, False, False)
    work = add_swings_confirmed(df, cfg.ltf_swing_left_right)
    engineered = bool(poi and _engineered_liquidity(work, bias, poi))
    swing_highs, swing_lows = get_recent_swings(work.iloc[:-1], count=5)
    if swing_highs.empty or swing_lows.empty:
        return LTFState(False, None, None, False, None, None, engineered, False)
    recent = work.tail(8)
    last_close = float(work["close"].iloc[-1])
    prior_high = float(swing_highs.iloc[-1]["high"])
    prior_low = float(swing_lows.iloc[-1]["low"])
    if bias == "bullish":
        sweep_rows = recent[(recent["low"] < prior_low) & (recent["close"] > prior_low)]
        liquidity_sweep = not sweep_rows.empty
        invalidation = float(sweep_rows["low"].min()) if liquidity_sweep else None
        ltf_msb = liquidity_sweep and last_close > prior_high
        return LTFState(
            liquidity_sweep,
            "bullish" if liquidity_sweep else None,
            prior_low if liquidity_sweep else None,
            ltf_msb,
            "bullish" if ltf_msb else None,
            invalidation,
            engineered,
            False,
        )
    sweep_rows = recent[(recent["high"] > prior_high) & (recent["close"] < prior_high)]
    liquidity_sweep = not sweep_rows.empty
    invalidation = float(sweep_rows["high"].max()) if liquidity_sweep else None
    ltf_msb = liquidity_sweep and last_close < prior_low
    return LTFState(
        liquidity_sweep,
        "bearish" if liquidity_sweep else None,
        prior_high if liquidity_sweep else None,
        ltf_msb,
        "bearish" if ltf_msb else None,
        invalidation,
        engineered,
        False,
    )


def detect_breaker_zone_and_retest(ltf_df: pd.DataFrame, bias: Direction, ltf_msb: bool) -> Tuple[Optional[float], Optional[float], bool]:
    """Simplified breaker: opposing candle before last bar; retest if price later overlaps that zone."""
    if not ltf_msb or bias == "neutral":
        return None, None, False
    work = ltf_df.reset_index(drop=True)
    n = len(work)
    if n < 6:
        return None, None, False
    last_i = n - 1
    if bias == "bullish":
        b_lo: Optional[float] = None
        b_hi: Optional[float] = None
        chosen_j: Optional[int] = None
        for j in range(last_i - 1, max(1, last_i - 28), -1):
            row = work.iloc[j]
            if float(row["close"]) < float(row["open"]):
                b_lo, b_hi = float(row["low"]), float(row["high"])
                chosen_j = j
                break
        if b_lo is None or chosen_j is None:
            return None, None, False
        tail = work.iloc[chosen_j + 1 :]
        retest = any(
            float(r["low"]) <= b_hi and float(r["high"]) >= b_lo for _, r in tail.iterrows()
        )
        return b_lo, b_hi, retest
    b_lo, b_hi = None, None
    chosen_j = None
    for j in range(last_i - 1, max(1, last_i - 28), -1):
        row = work.iloc[j]
        if float(row["close"]) > float(row["open"]):
            b_lo, b_hi = float(row["low"]), float(row["high"])
            chosen_j = j
            break
    if b_lo is None or chosen_j is None:
        return None, None, False
    tail = work.iloc[chosen_j + 1 :]
    retest = any(float(r["low"]) <= b_hi and float(r["high"]) >= b_lo for _, r in tail.iterrows())
    return b_lo, b_hi, retest


def resolve_external_liquidity(htf_df: pd.DataFrame, structure: StructureState, cfg: Config) -> Optional[float]:
    """Prefer nearest unswept HTF swing beyond price; fallback to dealing-range extreme."""
    if not structure.msb_found or structure.bias == "neutral":
        return structure.external_liquidity_target
    work = add_swings_confirmed(htf_df, cfg.htf_swing_left_right)
    price = float(htf_df["close"].iloc[-1])
    if structure.bias == "bullish":
        highs = work.loc[work["swing_high"], "high"]
        above = [float(h) for h in highs if float(h) > price]
        if above:
            return min(above)
        return structure.range_high
    lows = work.loc[work["swing_low"], "low"]
    below = [float(lo) for lo in lows if float(lo) < price]
    if below:
        return max(below)
    return structure.range_low


def price_invalidated(bias: Direction, price: float, invalidation: Optional[float]) -> bool:
    if invalidation is None or bias == "neutral":
        return False
    if bias == "bullish":
        return price < invalidation
    return price > invalidation


def poi_atr_distance(price: float, poi: Zone) -> float:
    if poi.contains(price):
        return 0.0
    return min(abs(price - poi.low), abs(price - poi.high))


def compute_rr(
    bias: Direction,
    price: float,
    target: Optional[float],
    invalidation: Optional[float],
) -> Optional[float]:
    if bias == "neutral" or target is None or invalidation is None:
        return None
    if bias == "bullish":
        risk = price - invalidation
        reward = target - price
    else:
        risk = invalidation - price
        reward = price - target
    if risk <= 0 or reward <= 0:
        return None
    return reward / risk


def pick_invalidation(
    bias: Direction,
    poi: Optional[Zone],
    ltf: LTFState,
) -> Optional[float]:
    if bias == "bullish":
        candidates = [ltf.invalidation, poi.low if poi else None]
        return min((c for c in candidates if c is not None), default=None)
    candidates = [ltf.invalidation, poi.high if poi else None]
    return max((c for c in candidates if c is not None), default=None)


def build_alert(
    symbol: str,
    status: SetupStatus,
    bias: Direction,
    dag: DagState,
    rr: Optional[float],
) -> str:
    parts = [symbol, str(dag), status, bias]
    if rr is not None:
        parts.append(f"RR~{rr:.2f}")
    return " | ".join(str(p) for p in parts)


def evaluate_symbol(cfg: Config, symbol: str, provider: OHLCVProvider) -> SetupSignal:
    feed = feed_for_symbol(symbol)
    htf_df = provider.fetch_ohlcv(symbol, cfg.htf, cfg.limit)
    ltf_df = provider.fetch_ohlcv(symbol, cfg.ltf, cfg.limit)
    structure = detect_htf_structure(htf_df, cfg)
    price = float(ltf_df["close"].iloc[-1])
    loc = range_location(price, structure)
    poi = select_valid_poi(htf_df, structure, cfg, cfg.htf)
    atr_series = atr(ltf_df)
    atr_last = float(atr_series.iloc[-1]) if len(atr_series) > 0 and pd.notna(atr_series.iloc[-1]) else 0.0
    inside_poi = bool(poi and poi.contains(price))
    near_poi = bool(
        poi
        and not inside_poi
        and atr_last > 0
        and poi_atr_distance(price, poi) <= cfg.near_poi_atr_mult * atr_last
    )
    scan_ltf = bool(poi and (inside_poi or near_poi))
    ltf_base = detect_ltf_state(ltf_df, structure.bias, cfg, poi, scan_ltf)
    _, _, breaker_seen = detect_breaker_zone_and_retest(ltf_df, structure.bias, ltf_base.ltf_msb)
    ltf = LTFState(
        ltf_base.liquidity_sweep,
        ltf_base.sweep_direction,
        ltf_base.swept_level,
        ltf_base.ltf_msb,
        ltf_base.ltf_msb_direction,
        ltf_base.invalidation,
        ltf_base.engineered_liquidity,
        breaker_seen,
    )
    target = resolve_external_liquidity(htf_df, structure, cfg)
    invalidation = pick_invalidation(structure.bias, poi, ltf)
    rr = compute_rr(structure.bias, price, target, invalidation)
    invalidated = bool(
        invalidation is not None
        and price_invalidated(structure.bias, price, invalidation)
        and (ltf.liquidity_sweep or ltf.ltf_msb)
    )
    htf_msb = structure.msb_found
    bias_neutral = structure.bias == "neutral"
    valid_range = structure.range_low is not None and structure.range_high is not None
    price_valid_half = price_in_valid_half(price, structure)
    poi_found = poi is not None
    rr_ok = rr is not None and rr >= cfg.min_rr
    ob = detect_order_block(htf_df, structure, cfg, cfg.htf)
    displacement_hint = ob is not None or len(detect_fvgs(htf_df, cfg.htf)) > 0
    dag = compute_dag_state(
        invalidated=invalidated,
        htf_msb=htf_msb,
        bias_neutral=bias_neutral,
        valid_range=valid_range,
        price_valid_half=price_valid_half,
        poi_found=poi_found,
        inside_poi=inside_poi,
        near_poi=near_poi,
        liquidity_sweep=ltf.liquidity_sweep,
        ltf_msb=ltf.ltf_msb,
        rr=rr,
        min_rr=cfg.min_rr,
        breaker_retest_seen=ltf.breaker_retest_seen,
    )
    missing = missing_conditions_for_state(
        dag,
        htf_msb=htf_msb,
        valid_range=valid_range,
        price_valid_half=price_valid_half,
        poi_found=poi_found,
        inside_or_near_poi=inside_poi or near_poi,
        liquidity_sweep=ltf.liquidity_sweep,
        ltf_msb=ltf.ltf_msb,
        rr=rr,
        min_rr=cfg.min_rr,
        breaker_retest_seen=ltf.breaker_retest_seen,
    )
    score = compute_score_components(
        msb_found=htf_msb,
        bias_neutral=bias_neutral,
        valid_range=valid_range,
        price_valid_half=price_valid_half,
        poi_found=poi_found,
        inside_or_near_poi=inside_poi or near_poi,
        liquidity_sweep=ltf.liquidity_sweep,
        ltf_msb=ltf.ltf_msb,
        rr_ok=rr_ok,
        engineered=ltf.engineered_liquidity,
        displacement_hint=displacement_hint,
    )
    classification = grade_from_score(score)
    status = cast(SetupStatus, legacy_status_from_dag(dag))
    alert = build_alert(symbol, status, structure.bias, dag, rr)
    direction = "long" if structure.bias == "bullish" else ("short" if structure.bias == "bearish" else "neutral")
    structured: Dict[str, Any] = {
        "symbol": symbol,
        "direction": direction,
        "htf": cfg.htf,
        "ltf": cfg.ltf,
        "state": dag,
        "score": score,
        "classification": classification,
        "htf_bias": structure.bias,
        "msb_level": structure.msb_price,
        "range_high": structure.range_high,
        "range_low": structure.range_low,
        "midpoint": structure.midpoint,
        "current_price": price,
        "premium_discount": loc,
        "poi": asdict(poi) if poi else None,
        "ltf_confirmation": {
            "liquidity_built": ltf.engineered_liquidity,
            "sweep_seen": ltf.liquidity_sweep,
            "swept_level": ltf.swept_level,
            "ltf_msb_seen": ltf.ltf_msb,
            "breaker_retest_seen": ltf.breaker_retest_seen,
        },
        "risk": {
            "entry_reference": None,
            "invalidation": invalidation,
            "target": target,
            "rr": rr,
        },
        "missing_conditions": missing,
    }
    return SetupSignal(
        symbol=symbol,
        htf=cfg.htf,
        ltf=cfg.ltf,
        status=status,
        score=score,
        bias=structure.bias,
        current_price=price,
        range_low=structure.range_low,
        range_high=structure.range_high,
        midpoint=structure.midpoint,
        range_location=loc,
        poi=poi,
        ltf_state=ltf,
        target=target,
        invalidation=invalidation,
        rr=rr,
        alert=alert,
        dag_state=dag,
        classification=classification,
        missing_conditions=tuple(missing),
        data_feed=feed,
        structured=structured,
    )


def signal_to_dict(sig: SetupSignal) -> Dict[str, Any]:
    d = asdict(sig)
    d.pop("structured", None)
    if sig.poi is not None:
        d["poi"] = asdict(sig.poi)
    d["ltf_state"] = asdict(sig.ltf_state)
    d["setup"] = dict(sig.structured)
    d["ticker_display_name"] = ticker_display_name(sig.symbol)
    return d


def maybe_openai_summary(signals: List[SetupSignal]) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY not set; skip AI summary")
        return
    try:
        from llm_explain import explain_setup
    except ImportError:
        log.warning("llm_explain not available; skip AI summary")
        return
    for s in signals:
        if not should_alert_dag(s.dag_state):
            continue
        text = explain_setup(s.structured)
        if text:
            log.info("AI scout %s: %s", s.symbol, text)


def maybe_telegram(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing; skip Telegram")
        return
    if requests is None:
        log.warning("requests not installed; skip Telegram")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat, "text": text[:3500]}, timeout=15)
        if r.status_code >= 400:
            log.error("Telegram HTTP %s: %s", r.status_code, r.text[:500])
    except Exception:
        log.exception("Telegram send failed")


def run_scan(cfg: Config, provider: OHLCVProvider) -> List[SetupSignal]:
    signals: List[SetupSignal] = []
    for symbol in cfg.symbols:
        try:
            signals.append(evaluate_symbol(cfg, symbol, provider))
        except Exception:
            log.exception("evaluate_symbol failed for %s", symbol)
    return signals


def print_signals(signals: List[SetupSignal]) -> None:
    for s in signals:
        log.info("%s", json.dumps(signal_to_dict(s), indent=2, default=str))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Meridian setup scout — structure + OTE heuristics")
    p.add_argument("--test", action="store_true", help="Run unit tests")
    p.add_argument("--provider", choices=("binance", "synthetic", "routed"), default=None)
    p.add_argument("--once", action="store_true", help="Single scan then exit (default if not looping)")
    p.add_argument("--loop", action="store_true", help="Repeat every poll_seconds")
    p.add_argument("--poll-seconds", type=int, default=None)
    p.add_argument("--symbols", type=str, default=None, help="Comma-separated, e.g. BTC/USDT,SOL/USDT")
    p.add_argument("--ai", action="store_true", help="OpenAI summary after scan (needs OPENAI_API_KEY)")
    p.add_argument("--telegram", action="store_true", help="Send compact alerts (needs bot token + chat id)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.test:
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    cfg_kw: Dict[str, Any] = {}
    if args.provider:
        cfg_kw["data_provider"] = args.provider
    if args.poll_seconds is not None:
        cfg_kw["poll_seconds"] = args.poll_seconds
    if args.symbols:
        cfg_kw["symbols"] = tuple(s.strip() for s in args.symbols.split(",") if s.strip())
    elif not os.environ.get("MARKET_SCREENER_SYMBOLS"):
        cfg_kw["symbols"] = ("BTC/USDT", "ETH/USDT")
    cfg_kw["use_ai_summary"] = bool(args.ai)
    cfg_kw["send_telegram"] = bool(args.telegram)
    cfg = Config(**{**asdict(Config()), **cfg_kw})

    provider = make_provider(cfg)
    do_loop = bool(args.loop)

    while True:
        signals = run_scan(cfg, provider)
        print_signals(signals)
        if cfg.use_ai_summary:
            maybe_openai_summary(signals)
        if cfg.send_telegram:
            for s in signals:
                if should_alert_dag(s.dag_state):
                    maybe_telegram(s.alert)
        if not do_loop:
            break
        log.info("Sleeping %s s", cfg.poll_seconds)
        time.sleep(cfg.poll_seconds)
    return 0


class TestHelpers(unittest.TestCase):
    def test_to_binance_symbol(self) -> None:
        self.assertEqual(to_binance_symbol("btc/usdt"), "BTCUSDT")

    def test_clamp_limit(self) -> None:
        self.assertEqual(clamp_limit(5000, 1, 1000), 1000)

    def test_normalise_binance_klines(self) -> None:
        raw = [[0, "1", "2", "0.5", "1.5", "10"]]
        df = normalise_binance_klines(raw)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(float(df["close"].iloc[0]), 1.5)

    def test_synthetic_provider(self) -> None:
        p = SyntheticOHLCVProvider()
        df = p.fetch_ohlcv("BTC/USDT", "30m", 50)
        validate_ohlcv(df)
        self.assertEqual(len(df), 50)

    def test_detect_htf_structure_runs(self) -> None:
        p = SyntheticOHLCVProvider()
        df = p.fetch_ohlcv("ETH/USDT", "4h", 200)
        st = detect_htf_structure(df, Config())
        self.assertIn(st.bias, ("bullish", "bearish", "neutral"))

    def test_fetch_daily_close_series(self) -> None:
        p = SyntheticOHLCVProvider()
        ser = fetch_daily_close_series(p, "BTC/USDT", 7)
        self.assertEqual(len(ser), 7)
        self.assertTrue(all(isinstance(x, float) for x in ser))


if __name__ == "__main__":
    raise SystemExit(main())
