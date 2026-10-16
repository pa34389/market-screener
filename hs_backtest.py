"""
Head & shoulders "pivot block" backtester + parameter optimiser (BTC, 5m by default).

Bearish rules (bullish is the exact mirror):
    1. Left shoulder (LS): a swing-high pivot. Its candle is the *pivot block*;
       zone = [LS candle low, LS candle high], mid = 50% of that candle.
    2. Head: a higher high after the LS (highest high since the LS).
    3. Neckline: the lowest low between the LS and the head (the head's low).
    4. MSB: first candle that *closes* below the neckline after the head.
    5. Wait for the right shoulder to come back into the pivot block:
         - entry "mid":   resting sell limit at the zone mid
         - entry "close": price trades into the zone, then a candle closes back
                          below the zone low -> short at that close
       Setup cancels if price trades above the stop before entry, or after
       ``expiry`` bars.
    6. Stop: head high + ``stop_buffer_atr`` x ATR(14) at the MSB candle.
    7. Target: fixed R multiple (3..7R tested).

Fills are conservative: a candle that touches both stop and target counts as a
loss, and a mid fill whose candle also reaches the stop is a loss.

Usage:
    python hs_backtest.py --download --start 2022-01-01      # Binance BTCUSDT 5m, cached in data/
    python hs_backtest.py --csv my_tradingview_export.csv     # or any OHLC CSV
    python hs_backtest.py --synthetic                         # offline smoke run (fake data)
    python -m unittest test_hs_backtest
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"

TARGETS_R: Tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 7.0)

# Grid searched by default (see optimise()).
GRID: Dict[str, Tuple] = {
    "pivot_len": (3, 5, 8, 12),
    "entry": ("mid", "close"),
    "stop_buffer_atr": (0.0, 0.1, 0.25, 0.5),
    "expiry": (24, 48, 96, 192),
    "min_risk_pct": (0.0, 0.15, 0.3),
    "direction": ("both", "short", "long"),
    "target_r": TARGETS_R,
}


@dataclass(frozen=True)
class Fees:
    """Per-side fee rates (fraction of notional). Defaults ~ Bybit USDT perps, regular tier."""

    maker: float = 0.0002
    taker: float = 0.00055
    stop_slippage: float = 0.0  # extra fraction lost on stop fills


@dataclass(frozen=True)
class Setup:
    direction: str  # "short" | "long"
    ls_idx: int
    head_idx: int
    msb_idx: int
    zone_low: float  # in the (possibly mirrored) price space
    zone_high: float
    head: float
    neckline: float
    atr: float


# --------------------------------------------------------------------------- data


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLC from CSV (TradingView export, Binance dump, or this module's cache)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in ("datetime", "time", "timestamp", "date", "open_time") if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"No time column in {path}; expected one of datetime/time/timestamp/date")
    t = df[tcol]
    if np.issubdtype(t.dtype, np.number):
        unit = "ms" if t.iloc[0] > 1e11 else "s"
        dt = pd.to_datetime(t, unit=unit, utc=True)
    else:
        dt = pd.to_datetime(t, utc=True)
    out = pd.DataFrame({"datetime": dt})
    for c in ("open", "high", "low", "close"):
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out["volume"] = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else 0.0
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)


def download_binance(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start: str = "2022-01-01",
    end: Optional[str] = None,
    base_url: str = "https://api.binance.com",
) -> pd.DataFrame:
    """Paginated Binance spot klines, cached to data/<symbol>_<interval>.csv.gz and topped up on rerun."""
    import requests

    from setup_scout import normalise_binance_klines

    cache = DATA_DIR / f"{symbol}_{interval}.csv.gz"
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else int(time.time() * 1000)
    frames: List[pd.DataFrame] = []
    if cache.exists():
        cached = load_csv(cache)
        frames.append(cached)
        if len(cached) and cached["datetime"].iloc[0] <= pd.Timestamp(start, tz="UTC"):
            start_ms = int(cached["datetime"].iloc[-1].timestamp() * 1000) + 1
    session = requests.Session()
    cursor = start_ms
    while cursor < end_ms:
        r = session.get(
            f"{base_url}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": 1000},
            timeout=20,
        )
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        frames.append(normalise_binance_klines(rows))
        cursor = int(rows[-1][0]) + 1
        log.info("downloaded %s .. %s", frames[-1]["datetime"].iloc[0], frames[-1]["datetime"].iloc[-1])
        time.sleep(0.05)
    df = pd.concat(frames).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    return df[(df["datetime"] >= lo) & (df["datetime"] <= hi)].reset_index(drop=True)


def synthetic_ohlcv(n: int = 60_000, seed: int = 1) -> pd.DataFrame:
    """Random-walk candles for smoke runs only — results on this mean nothing."""
    rng = np.random.default_rng(seed)
    ret = rng.standard_t(4, n) * 0.0012
    close = 30_000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(rng.normal(0, 0.0008, (2, n))) * close
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    dt = pd.date_range("2022-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"datetime": dt, "open": open_, "high": high, "low": low, "close": close, "volume": 0.0})


# --------------------------------------------------------------------------- detection


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()


def pivot_highs(high: np.ndarray, n: int) -> np.ndarray:
    """Index of bars whose high is strictly above the n bars before and >= the n bars after."""
    s = pd.Series(high)
    left = s.shift(1).rolling(n).max()
    right = s[::-1].shift(1).rolling(n).max()[::-1]
    mask = (s > left) & (s >= right)
    return np.flatnonzero(mask.to_numpy())


def detect_setups(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    pivot_len: int,
    direction: str = "short",
    max_pattern_bars: int = 300,
) -> List[Setup]:
    """Bearish H&S setups in the given price space (pass mirrored arrays for longs).

    Only information available at the MSB close is used: the LS pivot is confirmed
    (head is after the pivot's right-hand bars by construction), the head is the
    highest high since the LS, and the MSB is the first close below the neckline.
    """
    n = len(high)
    by_msb: Dict[Tuple[int, int], Setup] = {}
    for i in pivot_highs(high, pivot_len):
        head_idx = -1
        head = high[i]
        neckline = np.inf
        run_min = np.inf  # min low over (i, k)
        for k in range(i + 1, min(n, i + max_pattern_bars)):
            if high[k] > head:
                head, head_idx, neckline = high[k], k, run_min
            elif head_idx > 0 and close[k] < neckline:
                if neckline < low[i]:
                    # Keep the LS closest to the head (most recent lower pivot high).
                    key = (head_idx, k)
                    prev = by_msb.get(key)
                    if prev is None or prev.ls_idx < i:
                        by_msb[key] = Setup(direction, i, head_idx, k, low[i], high[i], head, neckline, atr[k])
                break
            run_min = min(run_min, low[k])
    return sorted(by_msb.values(), key=lambda s: s.msb_idx)


# --------------------------------------------------------------------------- simulation


def _first_true(mask: np.ndarray) -> int:
    """Index of first True or -1."""
    if mask.size == 0:
        return -1
    j = int(np.argmax(mask))
    return j if mask[j] else -1


def find_entry(
    s: Setup, high: np.ndarray, close: np.ndarray, stop: float, mode: str, max_expiry: int
) -> Tuple[int, float, bool]:
    """Returns (entry_idx, entry_price, stopped_on_fill_bar); entry_idx -1 if no entry."""
    a = s.msb_idx + 1
    b = min(len(high), a + max_expiry)
    h = high[a:b]
    if mode == "mid":
        mid = (s.zone_low + s.zone_high) / 2
        j = _first_true(h >= mid)
        if j < 0:
            return -1, 0.0, False
        return a + j, mid, bool(h[j] >= stop)
    if mode == "close":
        touched = np.maximum.accumulate(h >= s.zone_low)
        cand = touched & (close[a:b] < s.zone_low)
        j = _first_true(cand)
        c = _first_true(h >= stop)
        if j < 0 or (0 <= c <= j):
            return -1, 0.0, False
        return a + j, float(close[a + j]), False
    raise ValueError(f"unknown entry mode {mode!r}")


def resolve_exits(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, start: int, entry: float, stop: float,
    targets: Sequence[float],
) -> List[Tuple[int, float, str]]:
    """Short-side exits from bar ``start`` onward. One (exit_idx, exit_price, reason) per target price."""
    n = len(high)
    out: List[Optional[Tuple[int, float, str]]] = [None] * len(targets)
    pos, chunk = start, 512
    while pos < n and any(o is None for o in out):
        end = min(n, pos + chunk)
        hs, ls = high[pos:end], low[pos:end]
        js = _first_true(hs >= stop)
        lim = js if js >= 0 else len(hs)
        for t_i, tgt in enumerate(targets):
            if out[t_i] is not None:
                continue
            jt = _first_true(ls[:lim] <= tgt)  # same-bar stop+target -> stop wins
            if jt >= 0:
                out[t_i] = (pos + jt, tgt, "target")
            elif js >= 0:
                out[t_i] = (pos + js, stop, "stop")
        pos, chunk = end, chunk * 2
    last = n - 1
    return [o if o is not None else (last, float(close[last]), "open") for o in out]


def simulate(
    df: pd.DataFrame,
    pivot_lens: Iterable[int] = GRID["pivot_len"],
    entries: Iterable[str] = GRID["entry"],
    stop_buffers: Iterable[float] = GRID["stop_buffer_atr"],
    max_expiry: int = max(GRID["expiry"]),
    targets_r: Sequence[float] = TARGETS_R,
    fees: Fees = Fees(),
    max_pattern_bars: int = 300,
) -> pd.DataFrame:
    """Every candidate trade for every (pivot_len, entry, stop_buffer, target) combo, longs and shorts.

    Expiry, min-risk, direction and one-position-at-a-time are applied later in
    ``evaluate`` so the expensive part runs once.
    """
    h0, l0, c0 = (df[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
    atr = atr_wilder(h0, l0, c0)
    # Longs are shorts on the price-negated series.
    spaces = {"short": (h0, l0, c0, 1.0), "long": (-l0, -h0, -c0, -1.0)}
    rows = []
    for pl in pivot_lens:
        for direction, (h, l, c, sign) in spaces.items():
            for s in detect_setups(h, l, c, atr, pl, direction, max_pattern_bars):
                for mode in entries:
                    for buf in stop_buffers:
                        stop = s.head + buf * s.atr
                        e_idx, e_px, stopped = find_entry(s, h, c, stop, mode, max_expiry)
                        if e_idx < 0:
                            continue
                        risk = stop - e_px
                        if risk <= 0:
                            continue
                        tgts = [e_px - r * risk for r in targets_r]
                        if stopped:
                            exits = [(e_idx, stop, "stop")] * len(tgts)
                        else:
                            exits = resolve_exits(h, l, c, e_idx + 1, e_px, stop, tgts)
                        entry_abs = abs(e_px)
                        entry_fee = (fees.maker if mode == "mid" else fees.taker) * entry_abs
                        for r_mult, (x_idx, x_px, reason) in zip(targets_r, exits):
                            gross = (e_px - x_px) / risk
                            x_abs = abs(x_px)
                            if reason == "target":
                                exit_fee = fees.maker * x_abs
                            else:
                                exit_fee = (fees.taker + fees.stop_slippage) * x_abs
                            rows.append(
                                (pl, mode, buf, r_mult, direction, s.msb_idx, e_idx - s.msb_idx, e_idx, x_idx,
                                 sign * e_px, sign * stop, sign * x_px, risk / entry_abs * 100, reason, gross,
                                 gross - (entry_fee + exit_fee) / risk, s.ls_idx, s.head_idx)
                            )
    cols = ["pivot_len", "entry", "stop_buffer_atr", "target_r", "direction", "msb_idx", "bars_to_entry",
            "entry_idx", "exit_idx", "entry_px", "stop_px", "exit_px", "risk_pct", "reason", "r_gross", "r_net",
            "ls_idx", "head_idx"]
    trades = pd.DataFrame(rows, columns=cols)
    if len(trades):
        trades["entry_time"] = df["datetime"].to_numpy()[trades["entry_idx"].to_numpy()]
        trades["exit_time"] = df["datetime"].to_numpy()[trades["exit_idx"].to_numpy()]
    return trades


# --------------------------------------------------------------------------- evaluation


def _non_overlapping(entry_idx: np.ndarray, exit_idx: np.ndarray) -> np.ndarray:
    """Greedy one-position-at-a-time filter (rows already sorted by entry)."""
    keep = np.zeros(len(entry_idx), dtype=bool)
    busy_until = -1
    for k in range(len(entry_idx)):
        if entry_idx[k] > busy_until:
            keep[k] = True
            busy_until = exit_idx[k]
    return keep


def stats(r: np.ndarray) -> Dict[str, float]:
    if len(r) == 0:
        return {"trades": 0, "win_rate": np.nan, "exp_r": np.nan, "total_r": 0.0, "pf": np.nan, "max_dd_r": 0.0}
    eq = np.cumsum(r)
    dd = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:] - eq
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "exp_r": float(r.mean()),
        "total_r": float(r.sum()),
        "pf": float(wins / losses) if losses > 0 else np.inf,
        "max_dd_r": float(dd.max()),
    }


def select_trades(trades: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    t = trades
    t = t[(t.pivot_len == cfg["pivot_len"]) & (t.entry == cfg["entry"])
          & (t.stop_buffer_atr == cfg["stop_buffer_atr"]) & (t.target_r == cfg["target_r"])
          & (t.bars_to_entry <= cfg["expiry"]) & (t.risk_pct >= cfg["min_risk_pct"])]
    if cfg["direction"] != "both":
        t = t[t.direction == cfg["direction"]]
    t = t.sort_values(["entry_idx", "msb_idx"])
    return t[_non_overlapping(t.entry_idx.to_numpy(), t.exit_idx.to_numpy())]


def evaluate(trades: pd.DataFrame, split_idx: int, grid: Dict[str, Tuple] = GRID) -> pd.DataFrame:
    """Stats for every grid config, in-sample (entry < split_idx) and out-of-sample."""
    keys = list(grid)
    out = []
    groups = dict(tuple(trades.groupby(["pivot_len", "entry", "stop_buffer_atr", "target_r"])))
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        g = groups.get((cfg["pivot_len"], cfg["entry"], cfg["stop_buffer_atr"], cfg["target_r"]))
        if g is None:
            continue
        sel = select_trades(g, cfg)
        is_mask = sel.entry_idx.to_numpy() < split_idx
        row = dict(cfg)
        for tag, part in (("is", sel[is_mask]), ("oos", sel[~is_mask])):
            for k, v in stats(part.r_net.to_numpy()).items():
                row[f"{tag}_{k}"] = v
        row["all_gross_exp_r"] = float(sel.r_gross.mean()) if len(sel) else np.nan
        for k, v in stats(sel.r_net.to_numpy()).items():
            row[f"all_{k}"] = v
        out.append(row)
    return pd.DataFrame(out)


def report(results: pd.DataFrame, min_trades: int, top: int = 15) -> str:
    lines = []
    ok = results[results.is_trades >= min_trades]
    if ok.empty:
        return f"No config reached {min_trades} in-sample trades."
    best = ok.sort_values("is_total_r", ascending=False).head(top)
    cols = ["pivot_len", "entry", "stop_buffer_atr", "expiry", "min_risk_pct", "direction", "target_r",
            "is_trades", "is_win_rate", "is_exp_r", "is_total_r", "is_max_dd_r",
            "oos_trades", "oos_win_rate", "oos_exp_r", "oos_total_r", "oos_max_dd_r"]
    lines.append(f"Top {top} configs by in-sample total R (net of fees, >= {min_trades} IS trades):")
    lines.append(best[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    lines.append("")
    lines.append("Parameter sensitivity — median full-sample expectancy (R/trade, net) across all other settings:")
    for p in ("target_r", "entry", "stop_buffer_atr", "pivot_len", "expiry", "min_risk_pct", "direction"):
        m = results.groupby(p)[["all_exp_r", "all_trades"]].median()
        lines.append(f"  {p}: " + ", ".join(f"{k}={v.all_exp_r:+.3f} (n~{v.all_trades:.0f})" for k, v in m.iterrows()))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="H&S pivot-block backtest + optimiser")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--download", action="store_true", help="Binance spot klines (cached under data/)")
    src.add_argument("--csv", type=str, help="OHLC CSV (e.g. TradingView export)")
    src.add_argument("--synthetic", action="store_true", help="Random-walk smoke test (meaningless results)")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="5m")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--is-frac", type=float, default=0.7, help="In-sample fraction (rest is out-of-sample)")
    p.add_argument("--min-trades", type=int, default=50)
    p.add_argument("--maker", type=float, default=Fees.maker)
    p.add_argument("--taker", type=float, default=Fees.taker)
    p.add_argument("--slippage", type=float, default=Fees.stop_slippage)
    p.add_argument("--out", default=str(DATA_DIR / "hs_results.csv"))
    args = p.parse_args(argv)

    if args.download:
        df = download_binance(args.symbol, args.interval, args.start, args.end)
    elif args.csv:
        df = load_csv(args.csv)
    else:
        df = synthetic_ohlcv()
    log.info("bars: %d  (%s -> %s)", len(df), df.datetime.iloc[0], df.datetime.iloc[-1])

    fees = Fees(args.maker, args.taker, args.slippage)
    t0 = time.time()
    trades = simulate(df, fees=fees)
    log.info("candidate trades: %d in %.1fs", len(trades), time.time() - t0)
    split = int(len(df) * args.is_frac)
    log.info("in-sample to %s, out-of-sample after", df.datetime.iloc[split])
    results = evaluate(trades, split)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    trades.to_csv(Path(args.out).with_name("hs_trades.csv.gz"), index=False)
    print(report(results, args.min_trades))
    print(f"\nFull grid -> {args.out}; every candidate trade -> {Path(args.out).with_name('hs_trades.csv.gz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
