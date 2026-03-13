# Deploy OCTTS

## Option 1: Local One-Click Start

Copy the environment template, fill in your secrets, then run:

```bash
./start.sh
```

Or use:

```bash
make dev
```

This will:

1. create `.env` if it does not exist
2. create `.venv`
3. install dependencies
4. start the FastAPI app on port `8000`

## Option 2: Docker Compose

1. Copy the environment file:

```bash
cp .env.example .env
```

2. Fill in at least:

- `TUSHARE_TOKEN`
- `LLM_API_KEY`
- `WECOM_WEBHOOK_URL`
- `OCTTS_STOCK_POOL`
- `OCTTS_AUTOMATION_ENABLED`

3. Start everything:

```bash
docker compose up -d --build
```

4. Open:

- `http://127.0.0.1:8000/dashboard`
- `http://127.0.0.1:8000/stocks/600000.SH`

## Production Notes

- `docker-compose.yml` starts `octts` and `redis`.
- Redis is used as the default memory backend in containers.
- Mounts are configured so local memory artifacts can persist across restarts.
- Historical records are stored per symbol under `memory/history/`.
- Put OCTTS behind Nginx or Caddy if you need HTTPS and domain access.
- Keep `.env` outside version control.

## Suggested Server Workflow

```bash
git clone <your-repo-url>
cd OCTTS
cp .env.example .env
# edit .env
docker compose up -d --build
```

## Built-In Scheduler

If `OCTTS_AUTOMATION_ENABLED=true`, the service will automatically run the configured stock pool on weekdays only (`Monday` through `Friday`) at:

- `OCTTS_AUTOMATION_MORNING_TIME`
- `OCTTS_AUTOMATION_AFTERNOON_TIME`
- `OCTTS_AUTOMATION_REVIEW_TIME`

This is enough for a simple four-symbol always-on deployment.

## OpenClaw Integration After Deploy

Once OCTTS is reachable from your OpenClaw runtime, point OpenClaw cron/webhooks at:

```bash
POST http://<host>:8000/analyze
```

Use these phases:

- `morning`
- `afternoon`
- `review`
