"""Unit tests for DAG states (no market data)."""

from __future__ import annotations

import unittest

from state_machine import compute_dag_state


class TestDag(unittest.TestCase):
    def test_no_setup_without_msb(self) -> None:
        s = compute_dag_state(
            invalidated=False,
            htf_msb=False,
            bias_neutral=False,
            valid_range=True,
            price_valid_half=True,
            poi_found=True,
            inside_poi=True,
            near_poi=False,
            liquidity_sweep=False,
            ltf_msb=False,
            rr=3.0,
            min_rr=2.0,
            breaker_retest_seen=True,
        )
        self.assertEqual(s, "NO_SETUP")

    def test_invalid_rr_after_msb(self) -> None:
        s = compute_dag_state(
            invalidated=False,
            htf_msb=True,
            bias_neutral=False,
            valid_range=True,
            price_valid_half=True,
            poi_found=True,
            inside_poi=True,
            near_poi=False,
            liquidity_sweep=True,
            ltf_msb=True,
            rr=1.2,
            min_rr=2.0,
            breaker_retest_seen=True,
        )
        self.assertEqual(s, "INVALID_RR")


if __name__ == "__main__":
    unittest.main()
