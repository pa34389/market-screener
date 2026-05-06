# Deploy Meridian Screener on Vercel

## What you get

- One **Flask serverless** function (`app.py` → `app`) for UI + API.
- SQLite on **`/tmp`** by default on Vercel (data resets when the function scales to zero or cycles). Override with **`MARKET_SCREENER_DB`** only if you attach a compatible persistent store (advanced).
- **Hourly scans** via **Vercel Cron** calling `GET/POST /api/cron/scan` (not the in-process thread from `webapp.py`).

## One-time setup

1. Install CLI: `npm i -g vercel` (or use `npx vercel`).
2. From this directory (`market-screener/`):

   ```bash
   vercel link
   vercel env add CRON_SECRET
   ```

   Generate a long random string for `CRON_SECRET`. Vercel Cron sends `Authorization: Bearer <CRON_SECRET>` automatically when this env is set.

3. Optional project env (dashboard or `vercel env add`):

   | Variable | Purpose |
   |----------|---------|
   | `MARKET_SCREENER_PROVIDER` | `routed` (default), `synthetic`, or `binance` |
   | `MARKET_SCREENER_MAX_SYMBOLS` | Cap symbols (default **25** on Vercel in `app.py` if unset) |
   | `MARKET_SCREENER_SYMBOLS` | Comma list to override universe |
   | `MARKET_SCREENER_DB` | Override SQLite path (default `/tmp/screener.sqlite3` on Vercel) |

4. Deploy:

   ```bash
   vercel --prod
   ```

5. After first deploy, open the production URL, then **Run scan** once (or wait for the top-of-hour cron). Cron returns **503** until `CRON_SECRET` is set.

6. In Vercel **Project → Settings → Functions**, raise **Max duration** (e.g. **300s** or your plan maximum) so full scans can finish.

## Limits you should know

- **Function max duration** defaults low on some plans; raise it in the dashboard for long scans. Keep **`MARKET_SCREENER_MAX_SYMBOLS`** reasonable.
- **Bundle size** must stay under Vercel’s Python function limit (large `.venv` must not be uploaded; `.vercelignore` excludes it).
- **Background `MARKET_SCREENER_POLL_SECONDS` thread** does not run on Vercel; use **Cron** only.

## Local vs Vercel

| | Local `python webapp.py` | Vercel |
|---|--------------------------|--------|
| Entry | `webapp.py` → `main()` | `app.py` → `app` |
| SQLite | `./data/screener.sqlite3` | `/tmp/screener.sqlite3` |
| Hourly scan | Env `MARKET_SCREENER_POLL_SECONDS` (default 3600) | Cron `/api/cron/scan` |
