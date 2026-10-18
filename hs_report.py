"""
Out-of-sample report + charts for hs_backtest.py results.

Reads the grid (hs_results*.csv) and the candidate trades (hs_trades*.csv.gz)
written by hs_backtest.py, then:
    - ranks configs on out-of-sample (OOS) expectancy, but only those that are
      also positive in-sample and sit on a plateau (neighbouring parameter values
      also positive OOS);
    - prints marginal tables for target_r / entry / direction;
    - shows gross vs net, per-year breakdowns and the IS-picked champion's OOS;
    - saves an equity curve (R) for the top 3 and a target_r x expiry heatmap.

Usage:
    python hs_report.py                                    # default (Bybit perp) fee run
    python hs_report.py --tag _spot                        # reads hs_results_spot.csv / hs_trades_spot.csv.gz
    python hs_report.py --compare _spot                    # adds a net-of-spot-fees column for the top configs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hs_backtest import DATA_DIR, GRID, cache_path, load_csv, select_trades, stats

ORDINAL = ("pivot_len", "stop_buffer_atr", "expiry", "min_risk_pct", "target_r")
CATEGORICAL = ("entry", "direction")
KEYS = list(GRID)

# Reference palette (dataviz skill): categorical slots 1-3, diverging blue<->red, light surface.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE, INK, INK_2, MUTED, GRID_LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"


def load(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    res = pd.read_csv(DATA_DIR / f"hs_results{tag}.csv")
    trades = pd.read_csv(DATA_DIR / f"hs_trades{tag}.csv.gz", parse_dates=["entry_time", "exit_time"])
    return res, trades


def add_plateau(res: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    """Median OOS expectancy of one-step neighbours (same entry/direction, one ordinal param moved one notch)."""
    res = res.copy()
    idx = {k: {v: i for i, v in enumerate(GRID[k])} for k in ORDINAL}
    lookup = res.set_index(KEYS)["oos_exp_r"].to_dict()
    lookup_n = res.set_index(KEYS)["oos_trades"].to_dict()
    plateau, share = [], []
    for row in res[KEYS].itertuples(index=False):
        cfg = dict(zip(KEYS, row))
        vals = []
        for k in ORDINAL:
            i = idx[k][cfg[k]]
            for j in (i - 1, i + 1):
                if 0 <= j < len(GRID[k]):
                    nb = tuple(GRID[k][j] if kk == k else cfg[kk] for kk in KEYS)
                    if lookup_n.get(nb, 0) >= min_trades:
                        vals.append(lookup[nb])
        plateau.append(float(np.median(vals)) if vals else np.nan)
        share.append(float(np.mean(np.array(vals) > 0)) if vals else np.nan)
    res["nb_oos_exp_r"] = plateau
    res["nb_pos_share"] = share
    return res


def config_trades(trades: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    g = trades[(trades.pivot_len == cfg["pivot_len"]) & (trades.entry == cfg["entry"])
               & (trades.stop_buffer_atr == cfg["stop_buffer_atr"]) & (trades.target_r == cfg["target_r"])]
    return select_trades(g, cfg)


def fmt_cfg(cfg: Dict) -> str:
    return (f"{cfg['direction']} {cfg['entry']} piv{cfg['pivot_len']} buf{cfg['stop_buffer_atr']} "
            f"exp{cfg['expiry']} minrisk{cfg['min_risk_pct']} {cfg['target_r']:.0f}R")


def marginal(ok: pd.DataFrame, by: str) -> pd.DataFrame:
    g = ok.groupby(by)
    return pd.DataFrame({
        "configs": g.size(),
        "is_exp_med": g.is_exp_r.median(),
        "oos_exp_med": g.oos_exp_r.median(),
        "oos_pos_share": g.oos_exp_r.apply(lambda s: (s > 0).mean()),
        "oos_trades_med": g.oos_trades.median(),
    })


def per_year(sel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, part in sel.groupby(sel.entry_time.dt.year):
        s = stats(part.r_net.to_numpy())
        rows.append({"year": y, **{k: s[k] for k in ("trades", "win_rate", "exp_r", "total_r", "pf", "max_dd_r")},
                     "gross_exp_r": part.r_gross.mean()})
    return pd.DataFrame(rows).set_index("year")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID_LINE, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_equity(picks: List[Dict], trades: pd.DataFrame, split_time: pd.Timestamp, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
    _style(ax)
    for color, cfg in zip(SERIES, picks):
        sel = config_trades(trades, cfg).sort_values("exit_time")
        eq = sel.r_net.cumsum().to_numpy()
        t = sel.exit_time.to_numpy()
        ax.step(np.concatenate([[t[0]], t]), np.concatenate([[0.0], eq]), where="post", color=color, linewidth=2,
                label=f"{fmt_cfg(cfg)}  (n={len(sel)}, {eq[-1]:+.0f}R)")
    ax.axvline(split_time, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(split_time, ax.get_ylim()[1], "  out-of-sample →", color=INK_2, fontsize=9, va="top")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylabel("Cumulative R (net of fees)", color=INK_2)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def plot_heatmap(res: pd.DataFrame, base: Dict, path: Path, title: str, min_trades: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    fixed = {k: base[k] for k in KEYS if k not in ("target_r", "expiry")}
    sub = res.loc[np.logical_and.reduce([res[k] == v for k, v in fixed.items()])]
    panels = (("is", "In-sample"), ("oos", "Out-of-sample"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    vmax = max(0.05, np.nanmax(np.abs(sub[["is_exp_r", "oos_exp_r"]].to_numpy())))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    for ax, (tag, name) in zip(axes, panels):
        piv = sub.pivot(index="target_r", columns="expiry", values=f"{tag}_exp_r").reindex(
            index=GRID["target_r"], columns=GRID["expiry"])
        n = sub.pivot(index="target_r", columns="expiry", values=f"{tag}_trades").reindex(
            index=GRID["target_r"], columns=GRID["expiry"])
        ax.imshow(piv.to_numpy(), cmap=cmap, norm=norm, aspect="auto", origin="lower")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v, k = piv.iat[i, j], n.iat[i, j]
                weak = "" if k >= min_trades else "*"
                ax.text(j, i, f"{v:+.2f}{weak}\nn={k:.0f}", ha="center", va="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(GRID["expiry"])), [f"{e}" for e in GRID["expiry"]])
        ax.set_yticks(range(len(GRID["target_r"])), [f"{t:.0f}R" for t in GRID["target_r"]])
        ax.set_xlabel("Expiry (bars after MSB)", color=INK_2)
        ax.set_title(f"{name}: expectancy R/trade (net)", color=INK, fontsize=10.5, loc="left")
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Target", color=INK_2)
    fig.suptitle(title, color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.text(0.01, 0.01, f"* fewer than {min_trades} trades: weak evidence", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="OOS report + charts for hs_backtest results")
    p.add_argument("--tag", default="", help="suffix of hs_results<tag>.csv / hs_trades<tag>.csv.gz")
    p.add_argument("--compare", default=None, help="second tag (e.g. _spot) to show the top configs under")
    p.add_argument("--min-trades", type=int, default=50)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--exchange", choices=("bybit", "binance"), default="bybit", help="must match the hs_backtest run")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="5m")
    p.add_argument("--start", default="2022-01-01", help="must match the hs_backtest run")
    p.add_argument("--is-frac", type=float, default=0.7, help="must match the hs_backtest run")
    args = p.parse_args(argv)
    pd.set_option("display.width", 220)
    ff = lambda x: f"{x:.3f}"  # noqa: E731

    res, trades = load(args.tag)
    res = add_plateau(res, args.min_trades)
    ok = res[(res.is_trades >= args.min_trades) & (res.oos_trades >= args.min_trades)].copy()
    # Same split as hs_backtest.main: first is_frac of the bars it was run on.
    bars = load_csv(cache_path(args.symbol, args.interval, args.exchange))
    bars = bars[bars.datetime >= pd.Timestamp(args.start, tz="UTC")].reset_index(drop=True)
    split_idx = int(len(bars) * args.is_frac)
    split_time = bars.datetime.iloc[split_idx]

    print(f"Configs: {len(res)}; with >= {args.min_trades} trades both IS and OOS: {len(ok)}")
    print(f"OOS positive (net): {(ok.oos_exp_r > 0).mean():.1%} of those; IS positive: {(ok.is_exp_r > 0).mean():.1%}")
    both = ok[(ok.is_exp_r > 0) & (ok.oos_exp_r > 0)]
    print(f"Positive in BOTH IS and OOS: {len(both)} ({len(both) / max(1, len(ok)):.1%})")
    print(f"Correlation IS vs OOS expectancy across configs: {ok.is_exp_r.corr(ok.oos_exp_r):+.2f}\n")

    champ = ok.sort_values("is_total_r", ascending=False).iloc[0]
    print("Honest check - champion picked on IN-SAMPLE total R, then its OOS:")
    print(f"  {fmt_cfg(champ)}: IS {champ.is_trades:.0f} tr {champ.is_exp_r:+.3f}R/tr  ->  "
          f"OOS {champ.oos_trades:.0f} tr {champ.oos_exp_r:+.3f}R/tr, PF {champ.oos_pf:.2f}\n")

    robust = both[both.nb_pos_share >= 0.75].copy()
    robust["score"] = robust[["is_exp_r", "oos_exp_r", "nb_oos_exp_r"]].min(axis=1)
    robust = robust.sort_values("score", ascending=False)
    cols = KEYS + ["is_trades", "is_exp_r", "oos_trades", "oos_win_rate", "oos_exp_r", "oos_pf", "oos_max_dd_r",
                   "oos_total_r", "nb_oos_exp_r", "nb_pos_share"]
    print(f"Top {args.top} plateau configs (positive IS & OOS, >=75% of neighbours positive OOS), "
          "ranked by min(IS exp, OOS exp, neighbour-median OOS exp):")
    print(robust[cols].head(args.top).to_string(index=False, float_format=ff) if len(robust) else "  none")
    print()

    for by in ("target_r", "entry", "direction", "pivot_len", "stop_buffer_atr", "expiry", "min_risk_pct"):
        print(f"By {by} (configs with >= {args.min_trades} IS & OOS trades):")
        print(marginal(ok, by).to_string(float_format=ff))
        print()

    picks = [dict(r[KEYS]) for _, r in robust.head(3).iterrows()]
    pick_label = "Top 3 plateau settings"
    if not picks:
        picks = [dict(r[KEYS]) for _, r in ok.sort_values("oos_exp_r", ascending=False).head(3).iterrows()]
        pick_label = "No plateau found; top 3 by OOS expectancy (picked on OOS, likely luck)"
        print(f"{pick_label}\n")
    cmp_trades = load(args.compare)[1] if args.compare else None
    for cfg in picks:
        sel = config_trades(trades, cfg)
        oos = sel[sel.entry_idx >= split_idx]
        print(f"== {fmt_cfg(cfg)}")
        print(f"   OOS gross {oos.r_gross.mean():+.3f}R/tr vs net {oos.r_net.mean():+.3f}R/tr  "
              f"(median risk {sel.risk_pct.median():.2f}% of price)")
        if cmp_trades is not None:
            sel2 = config_trades(cmp_trades, cfg)
            oos2 = sel2[sel2.entry_idx >= split_idx]
            s2 = stats(oos2.r_net.to_numpy())
            print(f"   OOS net of {args.compare} fees: {s2['exp_r']:+.3f}R/tr, total {s2['total_r']:+.1f}R, "
                  f"PF {s2['pf']:.2f}, maxDD {s2['max_dd_r']:.1f}R")
        print(per_year(sel).to_string(float_format=ff))
        print()

    if not args.no_plots:
        eq = DATA_DIR / f"hs_equity_top3{args.tag}.png"
        hm = DATA_DIR / f"hs_heatmap_target_expiry{args.tag}.png"
        fee = "spot 0.1%/0.1%" if args.tag.endswith("_spot") else "perp 0.02%/0.055%"
        plot_equity(picks, trades, split_time, eq, f"{pick_label}, {args.exchange.title()} BTCUSDT spot 5m, net of {fee} fees")
        plot_heatmap(res, picks[0], hm, f"target_r × expiry, other params fixed at: {fmt_cfg(picks[0])} "
                                         f"({fee} fees)", args.min_trades)
        print(f"Saved {eq}\nSaved {hm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
