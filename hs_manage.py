"""
Trade-management study: do exit rules change the H&S system's results?

Same setups as hs_patterns.py (one fixed configuration; every setup is one trade). Each trade is
walked bar by bar under several exit rules. Conservative fills throughout:
    - within a bar the adverse move is assumed to happen first (stop checked before targets);
    - a stop moved because of this bar's favourable move only applies from the next bar;
    - stop / break-even / trailing / time exits pay taker, limit targets pay maker.

Rules:
    fixed4R      stop or 4R target (the baseline)
    be1R_4R      move stop to entry once +1R is reached, target 4R
    half2R_4R    take 50% at 2R and move stop to entry, rest at 4R
    trail2R      no target; once +2R is reached trail the stop 2R behind the best price
    time48_4R    4R target, but exit at the close after 48 bars if neither stop nor target hit

Usage:
    python hs_manage.py                 # 5m
    python hs_manage.py --timeframe 1h
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hs_backtest import atr_wilder, cache_path, detect_setups, find_entry, load_csv, resample_ohlcv

RULES = ("fixed4R", "be1R_4R", "half2R_4R", "trail2R", "time48_4R")


def manage(h: np.ndarray, l: np.ndarray, c: np.ndarray, start: int, entry: float, stop: float,
           rule: str, horizon: int) -> List[tuple]:
    """Short-space walk from bar ``start``. Returns [(fraction, exit_price, kind)] with kind in
    {"target", "stop"}; "stop" covers every taker exit (stop, break-even, trail, time, open)."""
    risk = stop - entry
    cur_stop, best = stop, entry
    legs: List[tuple] = []
    left = 1.0
    end = min(len(h), start + horizon)
    for k in range(start, end):
        if h[k] >= cur_stop:  # adverse first
            legs.append((left, cur_stop, "stop"))
            return legs
        if rule == "half2R_4R" and left == 1.0 and l[k] <= entry - 2 * risk:
            legs.append((0.5, entry - 2 * risk, "target"))
            left = 0.5
        tgt = entry - 4 * risk
        if rule != "trail2R" and l[k] <= tgt:
            legs.append((left, tgt, "target"))
            return legs
        best = min(best, l[k])
        # Stop changes take effect from the next bar.
        if rule == "be1R_4R" and best <= entry - risk:
            cur_stop = min(cur_stop, entry)
        if rule == "half2R_4R" and left < 1.0:
            cur_stop = min(cur_stop, entry)
        if rule == "trail2R" and best <= entry - 2 * risk:
            cur_stop = min(cur_stop, best + 2 * risk)
        if rule == "time48_4R" and k - start + 1 >= 48:
            legs.append((left, c[k], "stop"))
            return legs
    legs.append((left, c[end - 1], "stop"))
    return legs


def run(df: pd.DataFrame, pivot_len: int, entry_mode: str, buf: float, expiry: int, horizon: int,
        maker: float, taker: float) -> pd.DataFrame:
    o0, h0, l0, c0 = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    atr = atr_wilder(h0, l0, c0)
    entry_fee = maker if entry_mode == "mid" else taker
    rows = []
    for direction, (o, h, l, c) in {"short": (o0, h0, l0, c0), "long": (-o0, -l0, -h0, -c0)}.items():
        for s in detect_setups(o, h, l, c, atr, pivot_len, direction):
            stop = s.head + buf * s.atr
            e_idx, e_px, stopped = find_entry(s, h, c, stop, entry_mode, expiry)
            if e_idx < 0 or stop - e_px <= 0:
                continue
            risk = stop - e_px
            row: Dict = {"entry_idx": e_idx, "direction": direction, "risk_pct": risk / abs(e_px) * 100}
            for rule in RULES:
                legs = [(1.0, stop, "stop")] if stopped else manage(h, l, c, e_idx + 1, e_px, stop, rule, horizon)
                gross = sum(f * (e_px - px) for f, px, _ in legs) / risk
                fees = (entry_fee * abs(e_px) + sum(f * abs(px) * (maker if kind == "target" else taker)
                                                    for f, px, kind in legs)) / risk
                row[f"{rule}_gross"] = gross
                row[f"{rule}_net"] = gross - fees
            rows.append(row)
    return pd.DataFrame(rows).sort_values("entry_idx").reset_index(drop=True)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Exit-rule study for the H&S system")
    p.add_argument("--exchange", choices=("bybit", "binance"), default="bybit")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--timeframe", default=None)
    p.add_argument("--is-frac", type=float, default=0.7)
    p.add_argument("--pivot-len", type=int, default=5)
    p.add_argument("--entry", choices=("close", "mid"), default="close")
    p.add_argument("--stop-buffer-atr", type=float, default=0.1)
    p.add_argument("--expiry", type=int, default=96)
    p.add_argument("--horizon-hours", type=int, default=168)
    p.add_argument("--maker", type=float, default=0.0002)
    p.add_argument("--taker", type=float, default=0.00055)
    args = p.parse_args(argv)

    df = load_csv(cache_path("BTCUSDT", "5m", args.exchange))
    df = df[df.datetime >= pd.Timestamp(args.start, tz="UTC")].reset_index(drop=True)
    bph = 12
    if args.timeframe:
        df = resample_ohlcv(df, args.timeframe)
        bph = int(pd.Timedelta("1h") / pd.Timedelta(args.timeframe))
    split = int(len(df) * args.is_frac)
    t = run(df, args.pivot_len, args.entry, args.stop_buffer_atr, args.expiry, args.horizon_hours * bph,
            args.maker, args.taker)
    is_mask = t.entry_idx < split
    print(f"Trades: {len(t)} (IS {is_mask.sum()}, OOS {(~is_mask).sum()}); "
          f"fees maker {args.maker:.4%} taker {args.taker:.4%}; mean R per trade (±1 s.e.)")
    out = []
    for rule in RULES:
        r = {"rule": rule}
        for tag, m in (("IS", is_mask), ("OOS", ~is_mask)):
            for kind in ("gross", "net"):
                x = t.loc[m, f"{rule}_{kind}"]
                r[f"{tag}_{kind}"] = f"{x.mean():+.3f}±{x.std(ddof=1) / np.sqrt(len(x)):.3f}"
            r[f"{tag}_win%"] = f"{(t.loc[m, f'{rule}_gross'] > 0).mean():.0%}"
        out.append(r)
    print(pd.DataFrame(out).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
