"""SQLite persistence for scan runs, per-symbol signals, and activity log."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "screener.sqlite3"


def default_db_path() -> Path:
    import os

    raw = os.environ.get("MARKET_SCREENER_DB")
    return Path(raw).expanduser() if raw else DEFAULT_DB_PATH


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    p = path or default_db_path()
    ensure_parent(p)
    conn = sqlite3.connect(str(p), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            provider TEXT NOT NULL,
            symbols_json TEXT NOT NULL,
            htf TEXT NOT NULL,
            ltf TEXT NOT NULL,
            signal_count INTEGER NOT NULL DEFAULT 0,
            error_count INTEGER NOT NULL DEFAULT 0,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            error_summary TEXT
        );
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
            symbol TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            FOREIGN KEY (scan_id) REFERENCES scans(id)
        );
        CREATE INDEX IF NOT EXISTS idx_signals_scan ON signals(scan_id);
        CREATE TABLE IF NOT EXISTS activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            kind TEXT NOT NULL,
            message TEXT NOT NULL,
            detail_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_activity_id ON activity(id);
        """
    )
    conn.commit()


@contextmanager
def session(path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    conn = connect(path)
    try:
        init_schema(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_activity(
    conn: sqlite3.Connection,
    kind: str,
    message: str,
    detail: Optional[Dict[str, Any]] = None,
) -> None:
    conn.execute(
        "INSERT INTO activity (created_at, kind, message, detail_json) VALUES (?,?,?,?)",
        (time.time(), kind, message, json.dumps(detail) if detail else None),
    )


def insert_scan(
    conn: sqlite3.Connection,
    *,
    provider: str,
    symbols: List[str],
    htf: str,
    ltf: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO scans (created_at, provider, symbols_json, htf, ltf, signal_count, error_count, duration_ms)
        VALUES (?,?,?,?,?,0,0,0)
        """,
        (time.time(), provider, json.dumps(symbols), htf, ltf),
    )
    return int(cur.lastrowid)


def finalize_scan(
    conn: sqlite3.Connection,
    scan_id: int,
    *,
    signal_count: int,
    error_count: int,
    duration_ms: int,
    error_summary: Optional[str],
) -> None:
    conn.execute(
        """
        UPDATE scans SET signal_count=?, error_count=?, duration_ms=?, error_summary=?
        WHERE id=?
        """,
        (signal_count, error_count, duration_ms, error_summary, scan_id),
    )


def insert_signal(conn: sqlite3.Connection, scan_id: int, symbol: str, payload: Dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO signals (scan_id, symbol, payload_json) VALUES (?,?,?)",
        (scan_id, symbol, json.dumps(payload, default=str)),
    )


def list_scans(conn: sqlite3.Connection, limit: int = 40) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM scans ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_scan_row(r) for r in rows]


def get_scan(conn: sqlite3.Connection, scan_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
    return _scan_row(row) if row else None


def list_signals_for_scan(conn: sqlite3.Connection, scan_id: int) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, symbol, payload_json FROM signals WHERE scan_id=? ORDER BY symbol ASC",
        (scan_id,),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "symbol": r["symbol"],
                "payload": json.loads(r["payload_json"]),
            }
        )
    return out


def list_activity(conn: sqlite3.Connection, limit: int = 80) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, created_at, kind, message, detail_json FROM activity ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "kind": r["kind"],
                "message": r["message"],
                "detail": json.loads(r["detail_json"]) if r["detail_json"] else None,
            }
        )
    return out


def _scan_row(r: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": r["id"],
        "created_at": r["created_at"],
        "provider": r["provider"],
        "symbols": json.loads(r["symbols_json"]),
        "htf": r["htf"],
        "ltf": r["ltf"],
        "signal_count": r["signal_count"],
        "error_count": r["error_count"],
        "duration_ms": r["duration_ms"],
        "error_summary": r["error_summary"],
    }


def aggregate_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    total = int(conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0])
    row = conn.execute(
        "SELECT id, created_at, signal_count, error_count, duration_ms FROM scans ORDER BY id DESC LIMIT 1"
    ).fetchone()
    last = None
    if row:
        last = {
            "id": row["id"],
            "created_at": row["created_at"],
            "signal_count": row["signal_count"],
            "error_count": row["error_count"],
            "duration_ms": row["duration_ms"],
        }
    return {"total_scans": total, "last_scan": last}


def latest_scan_with_signals(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return None
    scan = _scan_row(row)
    scan["signals"] = list_signals_for_scan(conn, int(scan["id"]))
    return scan
