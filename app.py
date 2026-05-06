"""
Vercel entrypoint: Flask instance must be named ``app`` (see Vercel Python / Flask docs).

Local dev: prefer ``python webapp.py`` (unchanged). ``python app.py`` also works for parity checks.
"""

from __future__ import annotations

import os

# Vercel serverless: ephemeral filesystem; disable in-process poll (use Cron instead).
if os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"):
    os.environ.setdefault("MARKET_SCREENER_DB", "/tmp/screener.sqlite3")
    os.environ.setdefault("MARKET_SCREENER_POLL_SECONDS", "0")
    # Cap universe unless you override in project env (long scans hit function max duration).
    os.environ.setdefault("MARKET_SCREENER_MAX_SYMBOLS", "25")

from webapp import create_app

app = create_app()
