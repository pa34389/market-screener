"""
Winners vs losers: which setup features predict how far an H&S trade runs before its stop?

For one fixed configuration (default: pivot_len 5, close entry, stop = head + 0.1 ATR, both
directions) every detected setup is one observation. The outcome is target-independent: the
maximum favourable excursion in R before the stop is hit (MFE). "Hit kR" = MFE >= k.

Every feature is measured on the bar *before* the entry bar (or earlier), so nothing uses
information that was unavailable when the trade was placed.

Features are bucketed into quintiles using in-sample (first 70% of bars) cut points only, then
the same cut points are applied out-of-sample. A pattern is only worth anything if it shows up
in-sample AND repeats out-of-sample.

Usage:
    python hs_patterns.py                       # Bybit cache, 2022-01-01 onward
    python hs_patterns.py --entry mid --pivot-len 8
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
import pandas as pd

from hs_backtest import atr_wilder, cache_path, detect_setups, find_entry, load_csv, resample_ohlcv

FEATURES = {
    "risk_pct": "stop distance, % of price",
    "risk_atr": "stop distance, x ATR",
    "atr_pct": "volatility: ATR, % of price",
    "pattern_atr": "head above zone top, x ATR",
    "zone_atr": "pivot block height, x ATR",
    "break_atr": "MSB depth: zone low - neckline, x ATR",
    "msb_body_atr": "MSB candle body, x ATR",
    "ls_to_head": "bars LS -> head",
    "head_to_msb": "bars head -> MSB",
    "msb_to_entry": "bars MSB -> entry",
    "trend_1h": "with-trend distance from 1h EMA200, % (+ = trade with trend)",
    "trend_1d": "with-trend distance from daily EMA50, % (+ = trade with trend)",
    "move_24h": "with-trend 24h price change, % (+ = trade with trend)",
}


def mfe_before_stop(h: np.ndarray, l: np.ndarray, start: int, entry: float, stop: float, horizon: int) -> float:
    """Max favourable excursion (short space, in price) from bar ``start`` until the stop is hit.
    The stop bar's own low is excluded (same-bar stop + target counts as a loss)."""
    end = min(len(h), start + horizon)
    hs = h[start:end]
    hit = np.flatnonzero(hs >= stop)
    stop_at = hit[0] if len(hit) else len(hs)
    if stop_at == 0:
        return 0.0
    return max(0.0, entry - float(l[start:start + stop_at].min()))


def build(df: pd.DataFrame, pivot_len: int, entry_mode: str, buf: float, expiry: int, horizon: int,
          bars_per_hour: int = 12) -> pd.DataFrame:
    o0, h0, l0, c0 = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    atr = atr_wilder(h0, l0, c0)
    ema_1h = pd.Series(c0).ewm(span=200 * bars_per_hour, adjust=False).mean().to_numpy()
    ema_1d = pd.Series(c0).ewm(span=50 * 24 * bars_per_hour, adjust=False).mean().to_numpy()
    day = 24 * bars_per_hour
    ny = df.datetime.dt.tz_convert("America/New_York")
    rows = []
    for direction, (o, h, l, c, e1, e2) in {
        "short": (o0, h0, l0, c0, ema_1h, ema_1d),
        "long": (-o0, -l0, -h0, -c0, -ema_1h, -ema_1d),
    }.items():
        for s in detect_setups(o, h, l, c, atr, pivot_len, direction):
            stop = s.head + buf * s.atr
            e_idx, e_px, stopped = find_entry(s, h, c, stop, entry_mode, expiry)
            if e_idx < 1:
                continue
            risk = stop - e_px
            if risk <= 0:
                continue
            p = e_idx - 1  # features from the bar before entry
            px = abs(c[p])
            mfe = 0.0 if stopped else mfe_before_stop(h, l, e_idx + 1, e_px, stop, horizon)
            rows.append({
                "direction": direction,
                "entry_idx": e_idx,
                "entry_time": df.datetime.iloc[e_idx],
                "mfe_r": mfe / risk,
                "risk_pct": risk / abs(e_px) * 100,
                "risk_atr": risk / atr[p],
                "atr_pct": atr[p] / px * 100,
                "pattern_atr": (s.head - s.zone_high) / s.atr,
                "zone_atr": (s.zone_high - s.zone_low) / s.atr,
                "break_atr": (s.zone_low - s.neckline) / s.atr,
                "msb_body_atr": (o[s.msb_idx] - c[s.msb_idx]) / s.atr,
                "ls_to_head": s.head_idx - s.ls_idx,
                "head_to_msb": s.msb_idx - s.head_idx,
                "msb_to_entry": e_idx - s.msb_idx,
                # In short space "with trend" = price below the EMA; the mirrored long space flips it for free.
                "trend_1h": (e1[p] - c[p]) / px * 100,
                "trend_1d": (e2[p] - c[p]) / px * 100,
                "move_24h": (c[p - day] - c[p]) / px * 100 if p >= day else np.nan,
                "ny_hour": ny.iloc[e_idx].hour,
                "weekday": ny.iloc[e_idx].day_name()[:3],
            })
    return pd.DataFrame(rows).sort_values("entry_idx").reset_index(drop=True)


def bucket_table(t: pd.DataFrame, col: str, is_mask: np.ndarray, targets=(2, 3, 4, 5), q: int = 5) -> pd.DataFrame:
    """Quintiles from IS cut points; hit rates and 4R gross expectancy IS vs OOS per bucket."""
    edges = np.unique(np.nanquantile(t.loc[is_mask, col], np.linspace(0, 1, q + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    b = pd.cut(t[col], edges, include_lowest=True)
    out = []
    for tag, m in (("IS", is_mask), ("OOS", ~is_mask)):
        g = t[m].groupby(b[m], observed=True)
        part = pd.DataFrame({"n": g.size()})
        for k in targets:
            part[f"hit{k}R"] = g.mfe_r.apply(lambda s, k=k: (s >= k).mean())
        part["exp4R"] = g.mfe_r.apply(lambda s: np.where(s >= 4, 4.0, -1.0).mean())
        part.columns = pd.MultiIndex.from_product([[tag], part.columns])
        out.append(part)
    return pd.concat(out, axis=1)


def spearman(t: pd.DataFrame, col: str, mask: np.ndarray) -> float:
    x = t.loc[mask, [col, "mfe_r"]].dropna()
    return float(x[col].rank().corr(x.mfe_r.rank())) if len(x) > 10 else np.nan


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Winners vs losers feature analysis for the H&S system")
    p.add_argument("--exchange", choices=("bybit", "binance"), default="bybit")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--is-frac", type=float, default=0.7)
    p.add_argument("--pivot-len", type=int, default=5)
    p.add_argument("--entry", choices=("close", "mid"), default="close")
    p.add_argument("--stop-buffer-atr", type=float, default=0.1)
    p.add_argument("--expiry", type=int, default=96)
    p.add_argument("--horizon-hours", type=int, default=168, help="max time to follow a trade (default 1 week)")
    p.add_argument("--timeframe", default=None, help='resample, e.g. "15min" or "1h" (default: 5m as cached)')
    args = p.parse_args(argv)
    pd.set_option("display.width", 220)

    df = load_csv(cache_path("BTCUSDT", "5m", args.exchange))
    df = df[df.datetime >= pd.Timestamp(args.start, tz="UTC")].reset_index(drop=True)
    bph = 12
    if args.timeframe:
        df = resample_ohlcv(df, args.timeframe)
        bph = int(pd.Timedelta("1h") / pd.Timedelta(args.timeframe))
    split = int(len(df) * args.is_frac)
    t = build(df, args.pivot_len, args.entry, args.stop_buffer_atr, args.expiry, args.horizon_hours * bph, bph)
    is_mask = (t.entry_idx < split).to_numpy()
    print(f"Setups: {len(t)} (IS {is_mask.sum()}, OOS {(~is_mask).sum()}), split at {df.datetime.iloc[split]}")
    for tag, m in (("IS", is_mask), ("OOS", ~is_mask)):
        r = t.mfe_r[m]
        print(f"  {tag}: hit 1R {np.mean(r >= 1):.2f}  2R {np.mean(r >= 2):.2f}  3R {np.mean(r >= 3):.2f}  "
              f"4R {np.mean(r >= 4):.2f}  5R {np.mean(r >= 5):.2f}  |  break-even hit rate at kR = 1/(k+1)")
    fee = (0.00055 + 0.0002) / (t.risk_pct / 100)
    print(f"  median perp fee drag: {fee.median():.2f}R per trade (median stop {t.risk_pct.median():.2f}% of price)\n")

    print("Rank correlation of each feature with run length (MFE in R), IS vs OOS (same sign = repeatable):")
    corr = pd.DataFrame({f: {"IS": spearman(t, f, is_mask), "OOS": spearman(t, f, ~is_mask)} for f in FEATURES}).T
    corr["same_sign"] = np.sign(corr.IS) == np.sign(corr.OOS)
    corr["meaning"] = [FEATURES[f] for f in corr.index]
    print(corr.sort_values("IS", key=abs, ascending=False).to_string(float_format=lambda x: f"{x:+.3f}"))
    print()

    for f in FEATURES:
        print(f"== {f}: {FEATURES[f]}")
        print(bucket_table(t, f, is_mask).to_string(float_format=lambda x: f"{x:.2f}"))
        print()
    for col in ("direction", "weekday", "ny_hour"):
        g = t.assign(win4=t.mfe_r >= 4, is_=np.where(is_mask, "IS", "OOS")).groupby([col, "is_"]).win4.agg(["mean", "size"])
        print(f"== hit 4R by {col}")
        print(g.unstack("is_").to_string(float_format=lambda x: f"{x:.2f}"))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
