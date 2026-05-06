"""
Local dashboard + SQLite log for market-screener.

Uses Binance public REST (no API key) via setup_scout.

Run:
    export MARKET_SCREENER_DB=./data/screener.sqlite3   # optional
    python webapp.py

Open http://127.0.0.1:5050/

Optional:
    MARKET_SCREENER_MAX_SYMBOLS=15   # cap a large default universe while testing
    MARKET_SCREENER_PROVIDER=routed  # default: Binance USDT + Yahoo for other tickers
    MARKET_SCREENER_POLL_SECONDS=0   # disable background scans (default is 3600: hourly)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory

import db_store
from setup_scout import (
    Config,
    SetupSignal,
    evaluate_symbol,
    fetch_daily_close_series,
    make_provider,
    signal_to_dict,
)

log = logging.getLogger(__name__)

_scan_lock = threading.Lock()
_SPARKLINE_MAX_SYMBOLS = 96

def _static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def _assets_dir() -> Path:
    return _static_dir() / "assets"


def _read_poll_seconds() -> int:
    """Background scan interval. Default 3600 (1 hour). Set MARKET_SCREENER_POLL_SECONDS=0 to disable.
    Values below 3600 are rounded up to one hour (no sub-hourly server polling)."""
    raw = (os.environ.get("MARKET_SCREENER_POLL_SECONDS", "3600") or "3600").strip()
    if raw == "0":
        return 0
    try:
        v = int(raw)
    except ValueError:
        return 3600
    if v <= 0:
        return 0
    return max(3600, v)


def _cfg_from_env() -> Config:
    base = Config()
    prov_raw = os.environ.get("MARKET_SCREENER_PROVIDER", "routed").lower()
    if prov_raw == "synthetic":
        cfg = Config(**{**asdict(base), "data_provider": "synthetic"})
    elif prov_raw == "binance":
        cfg = Config(**{**asdict(base), "data_provider": "binance"})
    else:
        cfg = Config(**{**asdict(base), "data_provider": "routed"})
    raw = os.environ.get("MARKET_SCREENER_SYMBOLS", "")
    if raw.strip():
        syms = tuple(s.strip() for s in raw.split(",") if s.strip())
        cfg = Config(**{**asdict(cfg), "symbols": syms})
    return cfg


def _parse_symbols_body(data: Any) -> Optional[Tuple[str, ...]]:
    if not data or not isinstance(data, dict):
        return None
    s = data.get("symbols")
    if s is None:
        return None
    if isinstance(s, str):
        out = tuple(x.strip() for x in s.split(",") if x.strip())
        return out or None
    if isinstance(s, list):
        out = tuple(str(x).strip() for x in s if str(x).strip())
        return out or None
    return None


def trim_cfg_symbols(cfg: Config) -> Config:
    lim_raw = os.environ.get("MARKET_SCREENER_MAX_SYMBOLS", "").strip()
    if lim_raw.isdigit() and int(lim_raw) > 0:
        cap = int(lim_raw)
        return Config(**{**asdict(cfg), "symbols": tuple(cfg.symbols[:cap])})
    return cfg


def run_scan_detailed(cfg: Config, provider: Any) -> Tuple[List[SetupSignal], List[Dict[str, str]]]:
    signals: List[SetupSignal] = []
    errors: List[Dict[str, str]] = []
    for symbol in cfg.symbols:
        try:
            signals.append(evaluate_symbol(cfg, symbol, provider))
        except Exception as exc:  # noqa: BLE001 — surface per-symbol Binance failures
            log.exception("evaluate_symbol failed for %s", symbol)
            errors.append({"symbol": symbol, "error": str(exc)})
    return signals, errors


def execute_scan(conn: Any, cfg: Config) -> Dict[str, Any]:
    cfg_run = trim_cfg_symbols(cfg)
    provider = make_provider(cfg_run)
    t0 = time.time()
    scan_id = db_store.insert_scan(
        conn,
        provider=cfg_run.data_provider,
        symbols=list(cfg_run.symbols),
        htf=cfg_run.htf,
        ltf=cfg_run.ltf,
    )
    db_store.log_activity(
        conn,
        "scan",
        f"Scan #{scan_id} started ({len(cfg_run.symbols)} symbols, {cfg_run.htf}/{cfg_run.ltf})",
        {"scan_id": scan_id, "symbols": list(cfg_run.symbols)},
    )
    signals, errors = run_scan_detailed(cfg_run, provider)
    for sig in signals:
        db_store.insert_signal(conn, scan_id, sig.symbol, signal_to_dict(sig))
    for err in errors:
        db_store.log_activity(
            conn,
            "error",
            f"{err['symbol']}: {err['error'][:200]}",
            {"scan_id": scan_id, "symbol": err["symbol"]},
        )
    duration_ms = int((time.time() - t0) * 1000)
    summary = json.dumps(errors[:20]) if errors else None
    db_store.finalize_scan(
        conn,
        scan_id,
        signal_count=len(signals),
        error_count=len(errors),
        duration_ms=duration_ms,
        error_summary=summary,
    )
    db_store.log_activity(
        conn,
        "scan",
        f"Scan #{scan_id} finished: {len(signals)} ok, {len(errors)} errors, {duration_ms} ms",
        {"scan_id": scan_id},
    )
    conn.commit()
    return {
        "scan_id": scan_id,
        "signals": [signal_to_dict(s) for s in signals],
        "errors": errors,
        "duration_ms": duration_ms,
    }


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/assets")
    db_path = db_store.default_db_path()

    @app.before_request
    def _ensure_db() -> None:
        # lazy init file + schema on first request
        pass

    def get_conn() -> Any:
        conn = db_store.connect(db_path)
        db_store.init_schema(conn)
        return conn

    @app.get("/")
    def index() -> Any:
        return send_from_directory(_static_dir(), "index.html")

    @app.get("/favicon.ico")
    def favicon() -> Any:
        # Default tab request path; small PNG (not full logo-mark) for fast loads.
        return send_from_directory(_assets_dir(), "favicon.png", mimetype="image/png")

    @app.get("/health")
    def health() -> Any:
        return jsonify({"ok": True, "db": str(db_path)})

    @app.get("/api/latest")
    def api_latest() -> Any:
        conn = get_conn()
        try:
            row = db_store.latest_scan_with_signals(conn)
            return jsonify({"scan": row})
        finally:
            conn.close()

    @app.get("/api/scans")
    def api_scans() -> Any:
        limit = int(request.args.get("limit", 40))
        conn = get_conn()
        try:
            return jsonify({"scans": db_store.list_scans(conn, limit=limit)})
        finally:
            conn.close()

    @app.get("/api/scans/<int:scan_id>")
    def api_scan_detail(scan_id: int) -> Any:
        conn = get_conn()
        try:
            scan = db_store.get_scan(conn, scan_id)
            if not scan:
                return jsonify({"error": "not found"}), 404
            scan["signals"] = db_store.list_signals_for_scan(conn, scan_id)
            return jsonify({"scan": scan})
        finally:
            conn.close()

    @app.get("/api/activity")
    def api_activity() -> Any:
        limit = int(request.args.get("limit", 80))
        conn = get_conn()
        try:
            return jsonify({"activity": db_store.list_activity(conn, limit=limit)})
        finally:
            conn.close()

    @app.get("/api/stats")
    def api_stats() -> Any:
        conn = get_conn()
        try:
            return jsonify(db_store.aggregate_stats(conn))
        finally:
            conn.close()

    def _sparkline_series(symbols: List[str]) -> Dict[str, List[float]]:
        cfg = trim_cfg_symbols(_cfg_from_env())
        provider = make_provider(cfg)
        series: Dict[str, List[float]] = {}
        for sym in symbols:
            try:
                series[sym] = fetch_daily_close_series(provider, sym, 7)
            except Exception as exc:  # noqa: BLE001
                log.warning("sparkline fetch failed for %s: %s", sym, exc)
                series[sym] = []
        return series

    @app.get("/api/sparklines")
    def api_sparklines() -> Any:
        """Daily closes for the last 7 sessions per symbol (GET: comma query; small lists only)."""
        raw = (request.args.get("symbols") or "").strip()
        if not raw:
            return jsonify({"series": {}})
        syms = [s.strip() for s in raw.split(",") if s.strip()][:_SPARKLINE_MAX_SYMBOLS]
        return jsonify({"series": _sparkline_series(syms)})

    @app.post("/api/sparklines")
    def api_sparklines_post() -> Any:
        """Daily closes for large symbol lists (POST JSON avoids URL length limits)."""
        body = request.get_json(silent=True) or {}
        raw = body.get("symbols")
        if not isinstance(raw, list):
            return jsonify({"series": {}})
        syms = [str(s).strip() for s in raw if str(s).strip()][:_SPARKLINE_MAX_SYMBOLS]
        return jsonify({"series": _sparkline_series(syms)})

    @app.post("/api/scan")
    def api_scan() -> Any:
        cfg = _cfg_from_env()
        body = request.get_json(silent=True) or {}
        syms = _parse_symbols_body(body)
        if syms:
            cfg = Config(**{**asdict(cfg), "symbols": syms})
        if body.get("htf"):
            cfg = Config(**{**asdict(cfg), "htf": str(body["htf"])})
        if body.get("ltf"):
            cfg = Config(**{**asdict(cfg), "ltf": str(body["ltf"])})

        with _scan_lock:
            conn = get_conn()
            try:
                result = execute_scan(conn, cfg)
                return jsonify(result)
            except Exception as exc:  # noqa: BLE001
                log.exception("execute_scan failed")
                db_store.log_activity(conn, "error", f"Scan failed: {exc}", {})
                conn.commit()
                return jsonify({"error": str(exc)}), 500
            finally:
                conn.close()

    @app.route("/api/cron/scan", methods=["GET", "POST"])
    def cron_scan() -> Any:
        """Vercel Cron: set CRON_SECRET in project env; Vercel sends ``Authorization: Bearer <CRON_SECRET>``."""
        secret = os.environ.get("CRON_SECRET")
        if not secret:
            return jsonify({"error": "CRON_SECRET not configured"}), 503
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {secret}":
            return jsonify({"error": "unauthorized"}), 401
        cfg = trim_cfg_symbols(_cfg_from_env())
        with _scan_lock:
            conn = get_conn()
            try:
                result = execute_scan(conn, cfg)
                return jsonify(
                    {
                        "ok": True,
                        "scan_id": result.get("scan_id"),
                        "signal_count": len(result.get("signals", [])),
                        "error_count": len(result.get("errors", [])),
                        "duration_ms": result.get("duration_ms"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("cron scan failed")
                try:
                    db_store.log_activity(conn, "error", f"Cron scan failed: {exc}", {})
                    conn.commit()
                except Exception:
                    log.exception("cron scan error log failed")
                return jsonify({"error": str(exc)}), 500
            finally:
                conn.close()

    return app


def _poll_loop(app: Flask, interval: int) -> None:
    interval = max(3600, int(interval))
    while True:
        with app.app_context():
            with _scan_lock:
                conn = db_store.connect(db_store.default_db_path())
                try:
                    db_store.init_schema(conn)
                    execute_scan(conn, _cfg_from_env())
                except Exception:
                    log.exception("background poll scan failed")
                finally:
                    conn.close()
        time.sleep(interval)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    app = create_app()
    port = int(os.environ.get("PORT", "5050"))
    poll = _read_poll_seconds()
    if poll > 0:
        t = threading.Thread(target=_poll_loop, args=(app, poll), daemon=True)
        t.start()
        log.info("Background scan every %s s (~%.1f h)", poll, poll / 3600)
    log.info("Dashboard http://127.0.0.1:%s/ (SQLite %s)", port, db_store.default_db_path())
    app.run(host="127.0.0.1", port=port, debug=os.environ.get("FLASK_DEBUG") == "1", use_reloader=False)


if __name__ == "__main__":
    main()
