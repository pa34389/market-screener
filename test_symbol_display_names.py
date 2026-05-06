"""Display names for dashboard ticker column."""

from __future__ import annotations

import unittest

from symbol_display_names import ticker_display_name
from universe import default_symbol_universe


class TestTickerDisplayNames(unittest.TestCase):
    def test_known_crypto(self) -> None:
        self.assertEqual(ticker_display_name("BTC/USDT"), "Bitcoin")

    def test_unknown_returns_symbol(self) -> None:
        self.assertEqual(ticker_display_name("ZZZ/USDT"), "ZZZ/USDT")

    def test_universe_coverage(self) -> None:
        for sym in default_symbol_universe():
            name = ticker_display_name(sym)
            self.assertTrue(name, sym)
            if sym in ("BTC/USDT", "AAPL", "^GSPC", "EURUSD=X"):
                self.assertNotEqual(name, sym, f"{sym} should map to a long name")


if __name__ == "__main__":
    unittest.main()
