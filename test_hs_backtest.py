"""Unit tests for the H&S pivot-block backtester (hand-built candles + one saved Bybit reference trade)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import hs_backtest
from hs_backtest import (Fees, GRID, atr_wilder, detect_setups, download_bybit, evaluate, find_entry, killzone_of,
                         load_csv, pivot_block, resample_ohlcv, pivot_highs, resolve_exits, simulate, synthetic_ohlcv, trades_path)

REFERENCE_CSV = Path(__file__).resolve().parent / "testdata" / "bybit_btcusdt_5m_2026-10-15.csv"

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

    def test_swing_needs_only_one_lower_bar_after(self) -> None:
        # idx 3 is higher than the 3 bars before; the next bar is lower, then the head leg takes it out.
        highs = np.array([1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0])
        self.assertIn(3, pivot_highs(highs, 3))
        self.assertNotIn(4, pivot_highs(highs, 3))

    def test_pivot_block_is_last_bullish_candle(self) -> None:
        o = np.array([10.0, 10.0, 11.0, 12.5])
        c = np.array([9.0, 11.0, 12.6, 12.0])  # idx 3 makes the high but closes red
        self.assertEqual(pivot_block(o, c, 3, 3), 2)
        self.assertEqual(pivot_block(o, c, 2, 3), 2)  # bullish high candle is its own block
        self.assertEqual(pivot_block(np.array([2.0, 2.0]), np.array([1.0, 1.0]), 1, 3), 1)  # no bullish: fallback


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

    def test_close_entry_ignores_zone_edge_poke(self) -> None:
        # After the MSB a bar pokes the zone low (102) without reaching the mid (103.5) and closes below:
        # not an entry. The real right shoulder (reaches 103.6) then closes below the zone low.
        poke = (99, 102.5, 98.5, 99)
        t = _run(BEAR_HS[:11] + [poke] + BEAR_HS[11:] + _falling(101, 15)).set_index("entry")
        self.assertEqual(t.loc["close"].entry_idx, 13)
        self.assertAlmostEqual(t.loc["close"].entry_px, 101.5)

    def test_cancel_if_head_taken_before_close_entry(self) -> None:
        bars = BEAR_HS[:12] + [(103, 109, 101, 101.5)] + _falling(101, 15)
        t = _run(bars)
        self.assertNotIn("close", set(t.entry))

    def test_expiry_filters(self) -> None:
        t = _run(BEAR_HS + _falling(101, 15))
        grid = dict(GRID, pivot_len=(2,), stop_buffer_atr=(0.0,), target_r=(3.0,), expiry=(1, 3),
                    min_risk_pct=(0.0,), direction=("short",), killzone=("any",))
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


class TestKillzones(unittest.TestCase):
    def test_labels_follow_new_york_time_with_dst(self) -> None:
        t = pd.DatetimeIndex(["2024-01-15 07:00", "2024-07-15 06:00", "2024-07-15 05:55", "2024-07-15 17:35",
                              "2024-07-15 20:00", "2024-01-16 01:00", "2024-07-15 14:05"], tz="UTC")
        # EST 02:00 london | EDT 02:00 london | EDT 01:55 none | EDT 13:35 ny_pm | EDT 16:00 none (end exclusive)
        # | EST 20:00 asia | EDT 10:05 london_close
        self.assertEqual(list(killzone_of(t)), ["london", "london", "", "ny_pm", "", "asia", "london_close"])

    def test_filter_skips_entries_outside_the_zone(self) -> None:
        # BEAR_HS entries (2024-01-01 00:55 / 01:00 UTC = 19:55 / 20:00 EST): mid is outside Asia, close is inside.
        t = _run(BEAR_HS + _falling(101, 15))
        grid = dict(GRID, pivot_len=(2,), stop_buffer_atr=(0.0,), target_r=(3.0,), expiry=(192,),
                    min_risk_pct=(0.0,), direction=("short",), killzone=("any", "asia", "london"))
        res = evaluate(t, split_idx=10_000, grid=grid).set_index(["entry", "killzone"])
        self.assertEqual(res.loc[("mid", "any")].is_trades, 1)
        self.assertEqual(res.loc[("mid", "asia")].is_trades, 0)
        self.assertEqual(res.loc[("close", "asia")].is_trades, 1)
        self.assertEqual(res.loc[("close", "london")].is_trades, 0)


class TestReferenceTrade(unittest.TestCase):
    """Pav's reference short, Bybit BTCUSDT spot 5m, 15 Oct 2026 (UTC+10), as drawn on his TradingView chart:
    pivot block 02:40 (zone ~81,850-81,975, mid ~81,915), head 03:05, MSB line ~81,840, short 81,827.4,
    stop 82,097.3, target 80,735.9 (4.04R) hit ~12:20."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_csv(REFERENCE_CSV)
        cls.local = cls.df.datetime.dt.tz_convert("Etc/GMT-10").dt.strftime("%H:%M").to_numpy()
        cls.o, cls.h, cls.l, cls.c = (cls.df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        cls.atr = atr_wilder(cls.h, cls.l, cls.c)

    def at(self, hhmm: str) -> int:
        return int(np.flatnonzero(self.local == hhmm)[0])

    def setup_for_head(self, pivot_len: int, n_bars: int = None):
        sl = slice(0, n_bars)
        found = [s for s in detect_setups(self.o[sl], self.h[sl], self.l[sl], self.c[sl], self.atr[sl], pivot_len)
                 if s.head_idx == self.at("03:05")]
        self.assertEqual(len(found), 1, "exactly one setup per head")
        return found[0]

    def test_pattern_matches_chart_for_every_pivot_len(self) -> None:
        for pl in GRID["pivot_len"]:
            with self.subTest(pivot_len=pl):
                s = self.setup_for_head(pl)
                self.assertEqual(self.local[s.ls_idx], "02:45")  # swing-high candle
                self.assertEqual(self.local[s.pb_idx], "02:40")  # candle that closes to create it
                self.assertEqual((s.zone_low, s.zone_high), (81860.1, 81973.0))
                self.assertEqual(s.head, 82088.6)
                self.assertEqual(s.neckline, 81840.0)
                self.assertEqual(self.local[s.msb_idx], "03:30")

    def test_entries(self) -> None:
        s = self.setup_for_head(3)
        e_idx, e_px, stopped = find_entry(s, self.h, self.c, 82097.3, "close", 192)
        self.assertEqual((self.local[e_idx], e_px, stopped), ("06:35", 81816.1, False))  # not the 03:35 / 05:25 pokes
        e_idx, e_px, stopped = find_entry(s, self.h, self.c, 82097.3, "mid", 192)
        self.assertEqual((self.local[e_idx], stopped), ("06:05", False))
        self.assertAlmostEqual(e_px, (81860.1 + 81973.0) / 2)

    def test_trade_wins_like_the_chart(self) -> None:
        e_idx = self.at("06:35")
        x_idx, x_px, reason = resolve_exits(self.h, self.l, self.c, e_idx + 1, 81827.4, 82097.3, [80735.9])[0]
        self.assertEqual((reason, self.local[x_idx]), ("target", "12:20"))
        t = simulate(self.df, pivot_lens=(3,), stop_buffers=(0.0,), targets_r=(4.0,), fees=Fees(0, 0))
        t = t[(t.direction == "short") & (t.head_idx == self.at("03:05"))].set_index("entry")
        self.assertEqual(set(t.reason), {"target"})
        self.assertAlmostEqual(t.loc["close"].r_gross, 4.0)

    def test_reference_entry_is_outside_every_killzone(self) -> None:
        # 06:35 AEST = 16:35 New York (EDT): after the NY PM killzone closes.
        e_time = self.df.datetime.iloc[self.at("06:35")]
        self.assertEqual(killzone_of(pd.DatetimeIndex([e_time]))[0], "")

    def test_discretionary_features_on_reference_trade(self) -> None:
        from hs_discretion import add_features
        from hs_patterns import build

        t = build(self.df, 5, "close", 0.1, 96, 2016, 12)
        t = add_features(t, self.df, 12)
        r = t[(t.direction == "short") & (t.entry_idx == self.at("06:35"))].iloc[0]
        # Right shoulder peaked at 81,983.9 (06:20), above the 81,860.1-81,973.0 zone top.
        self.assertAlmostEqual(r.rs_depth, (81983.9 - 81860.1) / (81973.0 - 81860.1), places=6)
        self.assertAlmostEqual(r.rs_vs_ls_atr, (81983.9 - 81989.7) / r.atr_msb, places=6)
        # MSB candle 03:30: high 81,952.7, low 81,794.1, close 81,825.2.
        self.assertAlmostEqual(r.msb_close_loc, (81952.7 - 81825.2) / (81952.7 - 81794.1), places=6)
        self.assertEqual(r.lon_x_asia + r.ny_x_lon, 0.0)  # 16:35 New York: neither session sweep

    def test_no_lookahead(self) -> None:
        full = self.setup_for_head(3)
        cut = self.setup_for_head(3, n_bars=full.msb_idx + 1)  # data ends at the MSB close
        self.assertEqual(full, cut)
        e_idx, e_px, _ = find_entry(full, self.h, self.c, 82097.3, "close", 192)
        e2 = find_entry(full, self.h[: e_idx + 1], self.c[: e_idx + 1], 82097.3, "close", 192)
        self.assertEqual((e_idx, e_px), e2[:2])


class _FakeBybit:
    """Serves ``bars`` like Bybit v5 kline: the newest ``limit`` bars in [start, end], newest first."""

    def __init__(self, bars):
        self.bars, self.calls = bars, 0

    def get(self, url, params, timeout):
        self.calls += 1
        rows = [b for b in self.bars if params["start"] <= b[0] <= params["end"]][-params["limit"]:]
        if not rows:  # inside a history gap Bybit ignores start and returns the bars before it
            rows = [b for b in self.bars if b[0] <= params["end"]][-params["limit"]:]
        body = {"retCode": 0, "result": {"list": [[str(x) for x in b] for b in reversed(rows)]}}
        return mock.Mock(raise_for_status=lambda: None, json=lambda: body)


class TestBybitDownload(unittest.TestCase):
    def test_paginates_ascending_and_caches(self) -> None:
        t0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
        bars = [(t0 + k * 300_000, 100 + k, 101 + k, 99 + k, 100.5 + k, 1.0) for k in range(2500)]
        end = pd.Timestamp(bars[-1][0], unit="ms", tz="UTC").isoformat()
        fake = _FakeBybit(bars)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(hs_backtest, "DATA_DIR", Path(d)):
            df = download_bybit(start="2024-01-01", end=end, session=fake)
            self.assertEqual(len(df), 2500)
            self.assertTrue(df.datetime.is_monotonic_increasing)
            self.assertEqual(list(df.open[:2]), [100.0, 101.0])
            self.assertTrue((Path(d) / "BTCUSDT_5m_bybit.csv.gz").exists())
            calls = fake.calls
            self.assertEqual(len(download_bybit(start="2024-01-01", end=end, session=fake)), 2500)
            self.assertLessEqual(fake.calls - calls, 1)  # rerun only tops up from the cache

    def test_skips_gap_in_history(self) -> None:
        t0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
        bars = [(t0 + k * 300_000, 1, 2, 0.5, 1.5, 1.0) for k in list(range(1500)) + list(range(4000, 4500))]
        end = pd.Timestamp(bars[-1][0], unit="ms", tz="UTC").isoformat()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(hs_backtest, "DATA_DIR", Path(d)):
            df = download_bybit(start="2024-01-01", end=end, session=_FakeBybit(bars))
        self.assertEqual(len(df), 2000)
        self.assertTrue(df.datetime.is_unique and df.datetime.is_monotonic_increasing)

    def test_retries_transient_errors(self) -> None:
        t0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
        fake = _FakeBybit([(t0 + k * 300_000, 1, 2, 0.5, 1.5, 1.0) for k in range(10)])
        real_get, failed = fake.get, []

        def flaky(url, params, timeout):
            if not failed:
                failed.append(1)
                return mock.Mock(raise_for_status=lambda: None, json=lambda: {"retCode": 10016, "retMsg": "internal error"})
            return real_get(url, params, timeout)

        fake.get = flaky
        with tempfile.TemporaryDirectory() as d, mock.patch.object(hs_backtest, "DATA_DIR", Path(d)):
            df = download_bybit(start="2024-01-01", end="2024-01-01 00:45", session=fake, retry_wait=0)
        self.assertEqual(len(df), 10)


class TestResample(unittest.TestCase):
    def test_15min_bars_and_gaps(self) -> None:
        df = _frame(BEAR_HS[:6])  # 00:00 .. 00:25 -> two 15m bars
        df = pd.concat([df, _frame(BEAR_HS[6:9]).assign(
            datetime=pd.date_range("2024-01-01 01:00", periods=3, freq="5min", tz="UTC"))], ignore_index=True)
        r = resample_ohlcv(df, "15min")
        self.assertEqual([t.strftime("%H:%M") for t in r.datetime], ["00:00", "00:15", "01:00"])  # 00:30/00:45 gap dropped
        self.assertEqual(tuple(r.iloc[0][["open", "high", "low", "close"]]), (100, 103, 99, 102.5))  # bars 00:00-00:10
        self.assertEqual(tuple(r.iloc[2][["open", "high", "low", "close"]]), (99, 108, 98.5, 103.5))


class TestPatternAnalysis(unittest.TestCase):
    def test_session_ranges_exist_only_once_complete(self) -> None:
        from hs_patterns import session_context

        # Hourly bars from 2024-01-15 18:00 to 2024-01-16 10:00 New York (EST = UTC-5); high = 100 + hour index.
        t = pd.date_range("2024-01-15 23:00", "2024-01-16 15:00", freq="1h", tz="UTC")
        df = pd.DataFrame({"datetime": t, "open": 100.0, "high": 100.0 + np.arange(len(t)),
                           "low": 99.0 - np.arange(len(t)), "close": 100.0, "volume": 1.0})
        ctx = session_context(df, bars_per_hour=1)
        ny = t.tz_convert("America/New_York")
        at = lambda hh: int(np.flatnonzero((ny.day == 16) & (ny.hour == hh))[0])  # noqa: E731
        # Asia range for the 16th = 15th 20:00-23:00 bars (indices 2..5): high 105, low 94.
        self.assertEqual((ctx["asia_hi"][at(1)], ctx["asia_lo"][at(1)]), (105.0, 94.0))
        # London range (02:00-04:00 = indices 8..10) is unknown before 05:00, then high 110.
        self.assertTrue(np.isnan(ctx["lon_hi"][at(4)]))
        self.assertEqual(ctx["lon_hi"][at(5)], 110.0)

    def test_mfe_stops_counting_at_the_stop_bar(self) -> None:
        from hs_patterns import mfe_before_stop

        h = np.array([100.0, 100.5, 99.0, 103.0, 99.0])
        l = np.array([99.0, 97.0, 95.0, 90.0, 80.0])  # bar 3 hits the stop (103) and also trades to 90
        self.assertAlmostEqual(mfe_before_stop(h, l, 1, 100.0, 102.0, 100), 5.0)  # only bars 1-2 count
        self.assertAlmostEqual(mfe_before_stop(h, l, 3, 100.0, 102.0, 100), 0.0)  # stopped on the first bar
        self.assertAlmostEqual(mfe_before_stop(h, l, 4, 100.0, 102.0, 100), 20.0)  # never stopped


class TestTradeManagement(unittest.TestCase):
    # Short from 100, stop 101 (1R = 1). Each bar is (high, low).
    @staticmethod
    def _walk(bars, rule):
        from hs_manage import manage

        h = np.array([b[0] for b in bars], dtype=float)
        l = np.array([b[1] for b in bars], dtype=float)
        c = (h + l) / 2
        return manage(h, l, c, 0, 100.0, 101.0, rule, 1000)

    def test_fixed_target_and_same_bar_stop_first(self) -> None:
        self.assertEqual(self._walk([(100.5, 97), (99, 95.5)], "fixed4R"), [(1.0, 96.0, "target")])
        self.assertEqual(self._walk([(101.2, 90)], "fixed4R"), [(1.0, 101.0, "stop")])

    def test_break_even_applies_from_next_bar(self) -> None:
        self.assertEqual(self._walk([(100.5, 98.8), (100.2, 99.5)], "be1R_4R"), [(1.0, 100.0, "stop")])
        # The bar that reaches +1R and then trades back to entry does not stop out at break-even.
        self.assertEqual(self._walk([(100.5, 98.8), (99.5, 95.9)], "be1R_4R"), [(1.0, 96.0, "target")])

    def test_partial_then_break_even(self) -> None:
        legs = self._walk([(100.5, 97.9), (100.1, 99)], "half2R_4R")
        self.assertEqual(legs, [(0.5, 98.0, "target"), (0.5, 100.0, "stop")])

    def test_trailing_stop(self) -> None:
        # Best 97.5 (+2.5R) -> trail at 99.5; next bar's high 99.6 exits there.
        self.assertEqual(self._walk([(100.5, 97.5), (99.6, 98)], "trail2R"), [(1.0, 99.5, "stop")])


class TestSmoke(unittest.TestCase):
    def test_synthetic_runs(self) -> None:
        t = simulate(synthetic_ohlcv(5_000), pivot_lens=(5,), stop_buffers=(0.1,))
        self.assertGreater(len(t), 0)
        self.assertTrue(set(t.reason) <= {"target", "stop", "open"})


if __name__ == "__main__":
    unittest.main()
