"""Integration tests for SQLite store and Flask webapp (synthetic provider, no Binance)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import db_store
from webapp import create_app


class TestDbStore(unittest.TestCase):
    def setUp(self) -> None:
        self._fd, self.path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(self._fd)

    def tearDown(self) -> None:
        try:
            os.remove(self.path)
        except OSError:
            pass

    def test_init_and_activity(self) -> None:
        conn = db_store.connect(Path(self.path))
        try:
            db_store.init_schema(conn)
            db_store.log_activity(conn, "test", "hello", {"x": 1})
            conn.commit()
            rows = db_store.list_activity(conn, limit=5)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["kind"], "test")
        finally:
            conn.close()


class TestWebappScan(unittest.TestCase):
    def setUp(self) -> None:
        self._fd, self.path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(self._fd)
        self.prev_db = os.environ.get("MARKET_SCREENER_DB")
        self.prev_prov = os.environ.get("MARKET_SCREENER_PROVIDER")
        self.prev_sym = os.environ.get("MARKET_SCREENER_SYMBOLS")
        os.environ["MARKET_SCREENER_DB"] = self.path
        os.environ["MARKET_SCREENER_PROVIDER"] = "synthetic"
        os.environ["MARKET_SCREENER_SYMBOLS"] = "BTC/USDT,ETH/USDT"

    def tearDown(self) -> None:
        if self.prev_db is None:
            os.environ.pop("MARKET_SCREENER_DB", None)
        else:
            os.environ["MARKET_SCREENER_DB"] = self.prev_db
        if self.prev_prov is None:
            os.environ.pop("MARKET_SCREENER_PROVIDER", None)
        else:
            os.environ["MARKET_SCREENER_PROVIDER"] = self.prev_prov
        if self.prev_sym is None:
            os.environ.pop("MARKET_SCREENER_SYMBOLS", None)
        else:
            os.environ["MARKET_SCREENER_SYMBOLS"] = self.prev_sym
        try:
            os.remove(self.path)
        except OSError:
            pass

    def test_post_scan_and_latest(self) -> None:
        app = create_app()
        client = app.test_client()
        r = client.post("/api/scan", json={})
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        body = r.get_json()
        self.assertIn("scan_id", body)
        self.assertEqual(len(body.get("signals", [])), 2)
        latest = client.get("/api/latest").get_json()
        self.assertIsNotNone(latest.get("scan"))
        self.assertEqual(latest["scan"]["id"], body["scan_id"])
        self.assertEqual(len(latest["scan"]["signals"]), 2)
        sig0 = latest["scan"]["signals"][0]
        pl = sig0.get("payload") or sig0
        self.assertIn("ticker_display_name", pl)
        self.assertTrue(str(pl["ticker_display_name"]).strip())
        act = client.get("/api/activity").get_json()
        self.assertGreaterEqual(len(act.get("activity", [])), 2)

    def test_sparklines_get(self) -> None:
        app = create_app()
        client = app.test_client()
        r = client.get("/api/sparklines?symbols=BTC/USDT,ETH/USDT")
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        body = r.get_json()
        self.assertIn("series", body)
        btc = body["series"].get("BTC/USDT") or []
        self.assertEqual(len(btc), 7, body)

    def test_sparklines_post_bulk(self) -> None:
        app = create_app()
        client = app.test_client()
        syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        r = client.post("/api/sparklines", json={"symbols": syms})
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        body = r.get_json()
        self.assertEqual(len(body["series"]["SOL/USDT"]), 7)

    def test_cron_scan_without_secret_returns_503(self) -> None:
        app = create_app()
        client = app.test_client()
        prev = os.environ.pop("CRON_SECRET", None)
        try:
            r = client.get("/api/cron/scan")
            self.assertEqual(r.status_code, 503)
        finally:
            if prev is not None:
                os.environ["CRON_SECRET"] = prev

    def test_cron_scan_wrong_bearer_returns_401(self) -> None:
        prev = os.environ.get("CRON_SECRET")
        os.environ["CRON_SECRET"] = "expected-secret"
        try:
            app = create_app()
            client = app.test_client()
            r = client.get("/api/cron/scan", headers={"Authorization": "Bearer wrong"})
            self.assertEqual(r.status_code, 401)
        finally:
            if prev is None:
                os.environ.pop("CRON_SECRET", None)
            else:
                os.environ["CRON_SECRET"] = prev


if __name__ == "__main__":
    unittest.main()
