"""Unit tests for the H&S pivot-block backtester (hand-built candles, no market data)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hs_backtest import Fees, GRID, evaluate, load_csv, pivot_highs, resolve_exits, simulate, synthetic_ohlcv, trades_path

# (open, high, low, close). Left shoulder idx 3 (zone 102-105, mid 103.5), neckline 98 (idx 5),
# head 108 (idx 7), MSB close 97.5 (idx 9), right shoulder tags mid at idx 11, closes back below
# the zone low at idx 12.
BEAR_HS = [
    (100, 101, 99, 100),
    (100, 102, 99.5, 101.5),
    (101.5, 103, 101, 102.5),
    (102.5, 105, 102, 104),
    (104, 104.5, 101, 101.5),
    (101.5, 102, 98, 99),
    (99, 103, 98.5, 102.5),
    (102.5, 108, 102, 107),
    (107, 107.5, 103, 103.5),
    (103.5, 104, 97, 97.5),
    (97.5, 100, 96, 99),
    (99, 103.6, 98.5, 103),
    (103, 103.2, 101, 101.5),
]


def _frame(bars):
    a = np.asarray(bars, dtype=float)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=len(a), freq="5min", tz="UTC"),
            "open": a[:, 0],
            "high": a[:, 1],
            "low": a[:, 2],
            "close": a[:, 3],
            "volume": 0.0,
        }
    )


def _falling(start, n, step=2.0):
    out, o = [], start
    for _ in range(n):
        out.append((o, o + 0.5, o - step, o - step * 0.75))
        o -= step * 0.75
    return out


def _mirror(bars, axis=200.0):
    return [(axis - o, axis - l, axis - h, axis - c) for o, h, l, c in bars]


def _run(bars, **kw):
    return simulate(
        _frame(bars), pivot_lens=(2,), stop_buffers=(0.0,), targets_r=(3.0,), fees=Fees(0, 0), **kw
    )


class TestPivots(unittest.TestCase):
    def test_left_shoulder_is_pivot(self) -> None:
        self.assertIn(3, pivot_highs(np.array([b[1] for b in BEAR_HS]), 2))


class TestShort(unittest.TestCase):
    def setUp(self) -> None:
        self.t = _run(BEAR_HS + _falling(101, 15)).set_index("entry")

    def test_mid_entry(self) -> None:
        r = self.t.loc["mid"]
        self.assertEqual((r.direction, r.msb_idx, r.entry_idx), ("short", 9, 11))
        self.assertAlmostEqual(r.entry_px, 103.5)
        self.assertAlmostEqual(r.stop_px, 108.0)
        self.assertEqual(r.reason, "target")
        self.assertAlmostEqual(r.r_gross, 3.0)

    def test_close_entry(self) -> None:
        r = self.t.loc["close"]
        self.assertEqual(r.entry_idx, 12)
        self.assertAlmostEqual(r.entry_px, 101.5)
        self.assertAlmostEqual(r.exit_px, 101.5 - 3 * 6.5)

    def test_stop_and_target_same_bar_is_loss(self) -> None:
        bars = BEAR_HS + [(101.5, 109, 70, 75)]
        r = _run(bars).set_index("entry").loc["close"]
        self.assertEqual(r.reason, "stop")
        self.assertAlmostEqual(r.r_gross, -1.0)

    def test_cancel_if_head_taken_before_close_entry(self) -> None:
        bars = BEAR_HS[:12] + [(103, 109, 101, 101.5)] + _falling(101, 15)
        t = _run(bars)
        self.assertNotIn("close", set(t.entry))

    def test_expiry_filters(self) -> None:
        t = _run(BEAR_HS + _falling(101, 15))
        grid = dict(GRID, pivot_len=(2,), stop_buffer_atr=(0.0,), target_r=(3.0,), expiry=(1, 3),
                    min_risk_pct=(0.0,), direction=("short",))
        res = evaluate(t, split_idx=10_000, grid=grid).set_index(["entry", "expiry"])
        self.assertEqual(res.loc[("mid", 1)].is_trades, 0)  # mid fill is 2 bars after the MSB
        self.assertEqual(res.loc[("mid", 3)].is_trades, 1)


class TestLong(unittest.TestCase):
    def test_inverse_hs_mirrors_short(self) -> None:
        t = _run(_mirror(BEAR_HS + _falling(101, 15))).set_index("entry")
        r = t.loc["mid"]
        self.assertEqual(r.direction, "long")
        self.assertAlmostEqual(r.entry_px, 200 - 103.5)
        self.assertAlmostEqual(r.stop_px, 200 - 108.0)
        self.assertAlmostEqual(r.r_gross, 3.0)


class TestExits(unittest.TestCase):
    def test_open_trade_marks_to_last_close(self) -> None:
        h = np.array([100.0, 100.5])
        l = np.array([99.0, 99.5])
        c = np.array([99.5, 100.0])
        self.assertEqual(resolve_exits(h, l, c, 0, 100.0, 102.0, [94.0]), [(1, 100.0, "open")])

    def test_fees_reduce_r(self) -> None:
        t = simulate(_frame(BEAR_HS + _falling(101, 15)), pivot_lens=(2,), stop_buffers=(0.0,), targets_r=(3.0,),
                     fees=Fees(maker=0.001, taker=0.001))
        r = t.set_index("entry").loc["mid"]
        self.assertAlmostEqual(r.r_net, 3.0 - (0.001 * 103.5 + 0.001 * 90.0) / 4.5)


class TestOutputs(unittest.TestCase):
    def test_trades_path_follows_results_tag(self) -> None:
        self.assertEqual(trades_path("data/hs_results.csv").name, "hs_trades.csv.gz")
        self.assertEqual(trades_path("data/hs_results_spot.csv").name, "hs_trades_spot.csv.gz")


    def test_load_csv_round_trips_string_and_epoch_times(self) -> None:
        df = _frame(BEAR_HS)
        with tempfile.TemporaryDirectory() as d:
            iso = Path(d) / "iso.csv.gz"
            df.to_csv(iso, index=False)  # datetime written as text, like the download cache
            epoch = Path(d) / "tv.csv"
            df.assign(time=[int(t.timestamp()) for t in df.datetime]).drop(columns="datetime").to_csv(epoch, index=False)
            for path in (iso, epoch):
                got = load_csv(path)
                self.assertEqual(list(got.datetime), list(df.datetime))
                self.assertEqual(list(got.close), list(df.close))


class TestSmoke(unittest.TestCase):
    def test_synthetic_runs(self) -> None:
        t = simulate(synthetic_ohlcv(5_000), pivot_lens=(5,), stop_buffers=(0.1,))
        self.assertGreater(len(t), 0)
        self.assertTrue(set(t.reason) <= {"target", "stop", "open"})


if __name__ == "__main__":
    unittest.main()
