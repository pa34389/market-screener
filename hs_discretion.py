"""
Can discretionary chart-reading rules give the H&S system an edge?

Each setup (pivot_len 5, close entry, stop = head + 0.1 ATR, both directions by default) is described
the way a discretionary trader reads a chart: pattern geometry and symmetry, displacement on the MSB,
rejection wicks, volume, liquidity sweeps, compression, round numbers, trend, stretch and session.
Every feature uses only candles closed at or before the bar before entry.

Rules are *found* on in-sample data only (Bybit, first 70% of 2022 -> now) and *judged* twice on data
they never saw: out-of-sample (Bybit, last 30%) and a holdout (Binance 2018-2021). Each subset is
also compared with the opposite trade at the same moment and stop distance ("flip"): a rule with
directional skill beats its flip; a rule that only finds good moments to trade does not.

Usage:
    python hs_discretion.py
    python hs_discretion.py --timeframe 15min
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hs_backtest import atr_wilder, cache_path, load_csv, resample_ohlcv
from hs_patterns import build, outcome

TARGET_R = 4.0

# Chart-reading features: name -> what a discretionary trader would call it.
FEATURES: Dict[str, str] = {
    # geometry
    "head_over_ls_atr": "head prominence over the left shoulder high (ATR)",
    "rs_depth": "how deep the right shoulder got into the zone (0 = zone low, 1 = zone top)",
    "rs_vs_ls_atr": "right shoulder high vs left shoulder high (ATR, - = lower RS)",
    "time_sym": "log(time head->entry / time LS->head): shoulder timing symmetry",
    "displacement_atr": "how far price pushed beyond the neckline before the right shoulder (ATR)",
    "pattern_atr": "head above zone top (ATR)",
    "zone_atr": "pivot block height (ATR)",
    "break_atr": "neckline distance below the zone (ATR)",
    "risk_atr": "stop distance (ATR)",
    # displacement / MSB quality
    "msb_body_atr": "MSB candle body (ATR)",
    "msb_close_loc": "MSB close location in its range (1 = closed on its low)",
    "msb_fvg": "fair value gap left on the head -> MSB leg",
    "head_to_msb": "bars head -> MSB (speed of the break)",
    "msb_to_entry": "bars MSB -> entry (speed of the right shoulder)",
    # rejection
    "head_wick": "upper-wick share of the head candle (rejection)",
    "rs_wick": "upper-wick share of the right-shoulder high candle",
    # volume
    "vol_head_vs_ls": "log volume at head vs left shoulder (- = exhaustion)",
    "vol_msb": "MSB volume vs prior 4h average",
    "vol_rs_vs_head": "log average volume on the right shoulder vs the head leg",
    # liquidity
    "head_swept_high": "head took out the highest high of the 8h before the left shoulder",
    "round_dist_atr": "head distance to the nearest $1,000 level (ATR)",
    # regime
    "atr_ratio": "ATR now vs its 24h average (expansion > 1)",
    "range_24h_atr": "24h range (ATR)",
    "atr_pct": "volatility: ATR as % of price",
    "trend_1h": "with-trend distance from 1h EMA200 (%)",
    "trend_1d": "with-trend distance from daily EMA50 (%)",
    "stretch_z": "stretch vs 20h mean (std devs, + = stretched the way the trade fades)",
    "vwap_atr": "stretch vs UTC-day VWAP (ATR)",
    "run_into_head": "% run into the head over the prior 24h",
    "swept_pd": "head swept the previous day's high/low",
    "ny_x_lon": "NY entry after sweeping and reclaiming the London range",
    "lon_x_asia": "London entry after sweeping and reclaiming the Asia range",
    "hour_sin": "time of day (sin)",
    "hour_cos": "time of day (cos)",
    "is_long": "long (1) or short (0)",
}


def add_features(t: pd.DataFrame, df: pd.DataFrame, bars_per_hour: int) -> pd.DataFrame:
    """Chart-reading features for each setup row produced by hs_patterns.build."""
    o0, h0, l0, c0, v = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
    atr = atr_wilder(h0, l0, c0)
    atr_day = pd.Series(atr).rolling(24 * bars_per_hour).mean().to_numpy()
    hh_day = pd.Series(h0).rolling(24 * bars_per_hour).max().to_numpy()
    ll_day = pd.Series(l0).rolling(24 * bars_per_hour).min().to_numpy()
    vol_avg = pd.Series(v).rolling(4 * bars_per_hour).mean().shift(1).to_numpy()
    spaces = {"short": (o0, h0, l0, c0), "long": (-o0, -l0, -h0, -c0)}
    look = 8 * bars_per_hour
    out = []
    for r in t.itertuples(index=False):
        o, h, l, c = spaces[r.direction]
        ls, hd, msb, e = r.ls_idx, r.head_idx, r.msb_idx, r.entry_idx
        p = e - 1
        a = r.atr_msb
        zl, zh = r.zone_low_s, r.zone_high_s
        rs_slice = slice(msb + 1, e + 1)
        rs_high = h[rs_slice].max() if e > msb else np.nan
        rs_bar = msb + 1 + int(np.argmax(h[rs_slice])) if e > msb else e
        disp_low = l[msb:e].min() if e > msb else l[msb]

        def upper_wick(k: int) -> float:
            rng = h[k] - l[k]
            return (h[k] - max(o[k], c[k])) / rng if rng > 0 else 0.0

        def vsum(k0: int, k1: int) -> float:
            return float(v[max(0, k0):k1 + 1].sum())

        fvg = any(h[k + 1] < l[k - 1] for k in range(hd + 1, msb + 1) if k + 1 <= msb)
        msb_rng = h[msb] - l[msb]
        prior = h[max(0, ls - look):ls]
        price = abs(c[p])
        out.append({
            "head_over_ls_atr": (r.head_s - h[ls]) / a,
            "rs_depth": (rs_high - zl) / (zh - zl) if zh > zl else np.nan,
            "rs_vs_ls_atr": (rs_high - h[ls]) / a,
            "time_sym": np.log(max(1, e - hd) / max(1, hd - ls)),
            "displacement_atr": (r.neckline_s - disp_low) / a,
            "msb_close_loc": (h[msb] - c[msb]) / msb_rng if msb_rng > 0 else 0.5,
            "msb_fvg": float(fvg),
            "head_wick": upper_wick(hd),
            "rs_wick": upper_wick(rs_bar),
            "vol_head_vs_ls": np.log((vsum(hd - 1, hd + 1) + 1e-9) / (vsum(ls - 1, ls + 1) + 1e-9)),
            "vol_msb": v[msb] / vol_avg[msb] if vol_avg[msb] > 0 else np.nan,
            "vol_rs_vs_head": np.log((v[msb + 1:e + 1].mean() + 1e-9) / (v[ls:hd + 1].mean() + 1e-9)) if e > msb else np.nan,
            "head_swept_high": float(len(prior) > 0 and r.head_s > prior.max()),
            "round_dist_atr": abs(abs(r.head_s) - round(abs(r.head_s) / 1000) * 1000) / a,
            "atr_ratio": atr[p] / atr_day[p] if atr_day[p] > 0 else np.nan,
            "range_24h_atr": (hh_day[p] - ll_day[p]) / atr[p],
            "hour_sin": np.sin(2 * np.pi * r.ny_hour / 24),
            "hour_cos": np.cos(2 * np.pi * r.ny_hour / 24),
            "is_long": float(r.direction == "long"),
        })
    f = pd.DataFrame(out, index=t.index)
    t = pd.concat([t, f], axis=1)
    for col in ("swept_pd", "ny_x_lon", "lon_x_asia"):
        t[col] = t[col].astype(float)
    return t


def load_set(exchange: str, start: str, end: Optional[str], timeframe: Optional[str]) -> Tuple[pd.DataFrame, int]:
    df = load_csv(cache_path("BTCUSDT", "5m", exchange))
    df = df[df.datetime >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.datetime < pd.Timestamp(end, tz="UTC")]
    df = df.reset_index(drop=True)
    bph = 12
    if timeframe:
        df = resample_ohlcv(df, timeframe)
        bph = int(pd.Timedelta("1h") / pd.Timedelta(timeframe))
    t = build(df, 5, "close", 0.1, 96, 168 * bph, bph)
    return add_features(t, df, bph), len(df)


def spearman(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & y.notna()
    return float(x[m].rank().corr(y[m].rank())) if m.sum() > 30 else np.nan


def fmt(o: dict) -> str:
    if not o.get("n"):
        return "n=0"
    return (f"n={o['n']:4d} hit {o['hit']:.2f} gross {o['gross']:+.2f}±{o['se']:.2f} "
            f"flip {o['flip']:+.2f} net {o['net']:+.2f}")


def main(argv: Optional[List[str]] = None) -> int:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor, export_text

    p = argparse.ArgumentParser(description="Discretionary-rule search with out-of-sample and holdout checks")
    p.add_argument("--timeframe", default=None)
    p.add_argument("--is-frac", type=float, default=0.7)
    args = p.parse_args(argv)
    pd.set_option("display.width", 250)

    a, n_a = load_set("bybit", "2022-01-01", None, args.timeframe)
    split = int(n_a * args.is_frac)
    sets = {"IS": a[a.entry_idx < split], "OOS": a[a.entry_idx >= split]}
    sets["HOLD"], _ = load_set("binance", "2018-01-01", "2022-01-01", args.timeframe)
    for k, v in sets.items():
        v["g"] = np.where(v.mfe_r >= TARGET_R, TARGET_R, -1.0)
    feats = list(FEATURES)
    print("Sets: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()) + f"  (target {TARGET_R:.0f}R, gross R/trade)")
    print("Baseline: " + " | ".join(f"{k} {fmt(outcome(v, TARGET_R))}" for k, v in sets.items()) + "\n")

    # 1. Feature screen: does each feature relate to run length the same way in all three sets?
    rows = []
    for f in feats:
        rho = {k: spearman(v[f], v.mfe_r) for k, v in sets.items()}
        cut = np.nanquantile(sets["IS"][f], [0.2, 0.8])
        top = {k: v.g[v[f] >= cut[1]].mean() - v.g[v[f] <= cut[0]].mean() for k, v in sets.items()}
        rows.append({"feature": f, **{f"rho_{k}": x for k, x in rho.items()},
                     **{f"top-bot_{k}": x for k, x in top.items()},
                     "consistent": bool(np.all(np.sign(list(rho.values())) == np.sign(rho["IS"]))
                                        and np.all(np.sign(list(top.values())) == np.sign(top["IS"])))})
    screen = pd.DataFrame(rows).sort_values("rho_IS", key=abs, ascending=False)
    print("1) Feature screen: rank corr with run length, and top-20% minus bottom-20% gross R (cut points from IS)")
    print(screen.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print()

    # 2. Failure anatomy: fast failures (never reached +0.5R) vs big runners (>= 6R).
    print("2) Failure anatomy: median feature, fast failures vs runners (same direction in all sets = real)")
    an = []
    for f in feats:
        d = {}
        for k, v in sets.items():
            ff, big = v[v.mfe_r < 0.5][f], v[v.mfe_r >= 6][f]
            d[k] = (big.mean() - ff.mean()) / (v[f].std() + 1e-12)
        an.append({"feature": f, **{f"runner-fail_{k}": x for k, x in d.items()},
                   "consistent": bool(np.all(np.sign(list(d.values())) == np.sign(d["IS"])))})
    an = pd.DataFrame(an).sort_values("runner-fail_IS", key=abs, ascending=False)
    print(an.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("   share of fast failures: " + ", ".join(f"{k} {np.mean(v.mfe_r < 0.5):.0%}" for k, v in sets.items()) + "\n")

    # 3. Human-readable rules: a shallow tree fitted on IS only.
    X = {k: v[feats].astype(float) for k, v in sets.items()}
    med = X["IS"].median()
    X = {k: x.fillna(med) for k, x in X.items()}
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(150, len(X["IS"]) // 25), random_state=0)
    tree.fit(X["IS"], sets["IS"].g)
    print("3) Rules found on IS (value = mean gross R at 4R in that branch):")
    print(export_text(tree, feature_names=feats, decimals=2))
    leaves = {k: tree.apply(x) for k, x in X.items()}
    lr = []
    for leaf in np.unique(leaves["IS"]):
        lr.append({"leaf": leaf, **{k: fmt(outcome(sets[k][leaves[k] == leaf], TARGET_R)) for k in sets}})
    print(pd.DataFrame(lr).to_string(index=False))
    good = [lf for lf in np.unique(leaves["IS"]) if sets["IS"].g[leaves["IS"] == lf].mean() > 0.15]
    print("   'take' = IS branches above +0.15R: " + " | ".join(
        f"{k} {fmt(outcome(sets[k][np.isin(leaves[k], good)], TARGET_R))}" for k in sets) + "\n")

    # 4. A scoring model (gradient boosting, heavily regularised) fitted on IS only.
    gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=200, min_samples_leaf=200,
                                        l2_regularization=1.0, random_state=0)
    gbm.fit(X["IS"], sets["IS"].g)
    score = {k: gbm.predict(x) for k, x in X.items()}
    print("4) Scoring model: take only the best-scored setups (thresholds from IS score quantiles)")
    for qt in (0.5, 0.7, 0.9):
        thr = np.quantile(score["IS"], qt)
        print(f"   top {1 - qt:.0%}: " + " | ".join(f"{k} {fmt(outcome(sets[k][score[k] >= thr], TARGET_R))}" for k in sets))
    for k in sets:
        print(f"   rank corr(score, run length) {k}: {spearman(pd.Series(score[k]), sets[k].mfe_r.reset_index(drop=True)):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
