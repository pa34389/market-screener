# Meridian Screener: go-live plan (modular)

This document is the **plan**, a **self-review**, and **executable modules**. It does not assume access to your hosting accounts.

---

## 1. Goals

| Goal | Success criteria |
|------|------------------|
| Public usefulness | Anonymous or free users can see a **bounded** useful view (delayed snapshot or small universe). |
| Sustainability | **Rate limits** and caps on server-side scans so Binance/Yahoo traffic cannot bankrupt you. |
| Trust | Disclaimer visible; no order execution; data delays documented. |
| Commercial option | Paid tier = speed, universe, alerts, retention (see prior freemium notes). |

---

## 2. Plan (modules, in order)

### Module A — Freeze the artifact

- [ ] Tag a release in git (e.g. `screener-v0.1.0`).
- [ ] Document runtime: Python version, `pip install -r requirements.txt`, `python webapp.py`, env vars (`MARKET_SCREENER_*`).
- [ ] Confirm `python test_webapp.py` and `python test_symbol_display_names.py` pass on CI or locally.

### Module B — Hosting choice

Pick **one**:

| Option | Pros | Cons |
|--------|------|------|
| **B.1 VPS** (Fly.io, Railway, Render, small VM) | Full control, SQLite file volume, long-lived process | You operate restarts, disk, backups |
| **B.2 PaaS container** | Simple deploy | Ephemeral disk unless you attach volume or move DB to managed Postgres |
| **B.3 Serverless** | Scale to zero | Flask + SQLite + long scans are a poor fit unless you redesign jobs/queue |

**Recommendation:** **B.1** or **B.2 with a persistent volume** for `MARKET_SCREENER_DB`, or migrate to Postgres later.

### Module C — Configuration (production)

- [ ] Set `MARKET_SCREENER_PROVIDER` (e.g. `routed` or `synthetic` for demos).
- [ ] Set `MARKET_SCREENER_MAX_SYMBOLS` to a safe cap for the free/public tier.
- [ ] Set `MARKET_SCREENER_DB` to a path on persistent storage.
- [ ] Background scans default **hourly** (`MARKET_SCREENER_POLL_SECONDS`, default `3600`). Set to `0` to disable. Sub-hourly values are not supported server-side.
- [ ] **Secrets:** only add what you use (e.g. `OPENAI_API_KEY`, Telegram) via host secret manager, never commit.

### Module D — Process model

- [ ] Run **one** web worker behind a reverse proxy (Caddy/nginx) or platform ingress.
- [ ] Use a **single scan lock** (already in app) awareness: scale-out requires a queue or sticky sessions; start with **one replica**.

### Module E — Edge and TLS

- [ ] TLS termination at proxy or platform.
- [ ] Set `FLASK_DEBUG=0` in production.
- [ ] Restrict admin endpoints if you add any later; today surface is mostly read + `POST /api/scan` (protect with auth or token if exposed).

### Module F — Observability

- [ ] Platform logs + log drain (optional).
- [ ] Uptime check on `GET /health`.
- [ ] Alert on 5xx or process crash.

### Module G — Legal and product

- [ ] Keep footer disclaimer accurate for jurisdiction and “not investment advice.”
- [ ] Document Yahoo/Binance unofficial/delayed nature on a `/about` or README link if you go broad public.

### Module H — Freemium (when you add billing)

- [ ] Auth (even magic link) before on-demand scan.
- [ ] Quotas per user/IP; free = delayed shared snapshot; paid = higher limits.
- [ ] Stripe (or similar) in a **separate** small service if you outgrow a single Flask app.

---

## 3. Self-review (critique of the plan)

| Risk | Mitigation |
|------|------------|
| **Upstream rate limits** (Binance/Yahoo) | Hard cap symbols; backoff; cache OHLCV where possible; avoid per-user unbounded “scan now.” |
| **SQLite on ephemeral disk** | Persistent volume or external DB; nightly backup of DB file. |
| **`POST /api/scan` abuse** | Auth, CAPTCHA, or secret cron token; per-IP rate limit at proxy. |
| **yfinance flakiness** | Accept occasional empty rows; log; synthetic provider for demos. |
| **Compliance** | No execution, clear non-advice copy; review affiliate links if added. |

**Verdict:** The plan is **shippable for a v0** on a single VM or one container + volume. It is **not** complete for high-traffic multi-tenant SaaS without Module H and stronger abuse controls.

---

## 4. “Set it live” — what this repo can and cannot do

**In-repo (done or doable without your keys):** code, tests, Dockerfile optional, env template, this plan, Name column + API field `ticker_display_name`.

**Cannot be executed from Cursor without your resources:**

- DNS for your domain  
- Hosting account (Fly/Railway/Render/AWS/Vercel **for Python long-running**, etc.)  
- TLS certificates (usually automatic on platforms)  
- Production env injection  
- Legal review for your country  

**Suggested first live step (you run):**

1. Create app on host, attach persistent disk path for SQLite.  
2. Set env vars from Module C.  
3. Start: `python webapp.py` or `gunicorn` wrapper if you add it.  
4. Hit `https://your-host/health` then open `/`.  
5. Run one scan; confirm signals and **Name** column populate.

---

## 5. Optional next engineering tasks (not blocking v0)

- Dockerfile + `gunicorn` + `WEB_CONCURRENCY=1`.  
- `POST /api/scan` guard: `X-Scan-Token` header checked against env.  
- Migrate SQLite → Postgres for multi-instance.  
- Job queue for scans (Redis + RQ / Celery).
