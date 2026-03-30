# OCTTS

OpenClaw Trend Tracking System: a memory-enhanced stock analysis assistant for
daily A-share monitoring, trend validation, and replay summaries.

## Features

- Scheduled analysis windows for morning, close, and nightly review.
- Historical memory that carries prior conclusions into the next run.
- Historical hit tracking that verifies prior suggestions against new prices.
- Per-symbol local history files for easier replay and archival.
- Market data integration through `Tushare Pro`.
- Structured reasoning and report generation through an OpenAI-compatible LLM gateway such as UCloud ModelVerse.
- Enterprise WeCom delivery for mobile notifications.
- OpenClaw-friendly HTTP entrypoint for cron or webhook orchestration.
- Optional built-in scheduler that can run morning, afternoon, and review analysis automatically.
- Built-in overview dashboard, single-stock detail page, and OpenClaw status entry.

## Project Layout

```text
src/octts/
  api.py
  config.py
  clients/
  prompts/
  schemas/
  services/
tests/
ops/openclaw/
```

## Environment Variables

Create a `.env` file from `.env.example` or export these variables:

```bash
cp .env.example .env

TINYSHARE_TOKEN=your-token
LLM_API_KEY=your-key
LLM_MODEL=deepseek-ai/DeepSeek-V3
LLM_BASE_URL=https://api.modelverse.cn/v1
WECOM_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=...
REDIS_URL=redis://localhost:6379/0
OCTTS_STOCK_POOL=300433.SZ,603920.SH
OCTTS_MEMORY_BACKEND=file
OCTTS_HISTORY_DIR_PATH=memory/history
OCTTS_HISTORY_LIMIT_PER_SYMBOL=30
OCTTS_DEFAULT_LOOKBACK_DAYS=20
OCTTS_MINUTE_FREQ=30MIN
OCTTS_LLM_MAX_TOKENS=3000
OCTTS_LLM_JSON_MODE=true
OCTTS_LLM_RETRY_ATTEMPTS=3
OCTTS_AUTOMATION_ENABLED=true
OCTTS_AUTOMATION_TIMEZONE=Asia/Shanghai
OCTTS_AUTOMATION_PHASES=review
OCTTS_AUTOMATION_MORNING_TIME=09:35
OCTTS_AUTOMATION_AFTERNOON_TIME=14:35
OCTTS_AUTOMATION_REVIEW_TIME=20:30
OCTTS_AUTOMATION_NOTIFY=true

# 选股系统配置
OCTTS_SCREENING_ENABLED=true
OCTTS_SCREENING_TIME=15:35
OCTTS_SCREENING_STRATEGIES=oversold_bounce,volume_breakout,golden_cross
OCTTS_SCREENING_NOTIFY=true

# 定时邮件配置
OCTTS_EMAIL_ENABLED=true
OCTTS_EMAIL_SEND_TIME=20:45
```

`DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL` / `DEEPSEEK_BASE_URL` are still accepted as legacy aliases, but new deployments should prefer `LLM_*`.
`OCTTS_HISTORY_FILE_PATH` is still accepted as a legacy alias and will be migrated to per-symbol files under the derived directory.
For local startup, `OCTTS_MEMORY_BACKEND=file` avoids requiring Redis. Docker deployment still overrides this to `redis`.
`OCTTS_LLM_MAX_TOKENS` controls the initial completion cap. If a response appears truncated, OCTTS now retries with a larger token budget automatically. The default has been raised to `3000` for richer multi-field analysis.
`OCTTS_LLM_JSON_MODE` tries to request JSON-mode output when the provider supports it, and `OCTTS_LLM_RETRY_ATTEMPTS` controls automatic repair retries after the initial failed attempt. A value of `3` means the system will retry up to three times before surfacing the error.

## One-Click Local Start

Fastest option:

```bash
./start.sh
```

Alternative:

```bash
make dev
```

Both commands will:

- create `.env` from `.env.example` if needed
- create `.venv`
- install dependencies
- start the app on port `8000`

## Manual Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
uvicorn octts.api:app --host 0.0.0.0 --port 8000 --reload
```

## One-Click Docker Deploy

```bash
cp .env.example .env
# edit .env
docker compose up -d --build
```

See `DEPLOY.md` for the full deployment guide.

## Manual Analysis

Run one manual analysis request:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"phase":"review","stock_pool":["600000.SH"]}'
```

Open the dashboard:

```bash
open http://127.0.0.1:8000/dashboard
```

Open a single-stock detail page:

```bash
open http://127.0.0.1:8000/stocks/600000.SH
```

Run one review-only backtest request:

```bash
curl -X POST http://127.0.0.1:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{"start_date":"20250101","end_date":"20250331","stock_pool":["600000.SH"]}'
```

## How Validation Works

Each generated report now includes a structured `decision` object:

- `signal`
- `entry_zone`
- `stop_loss`
- `take_profit`
- `holding_horizon`
- `invalidation_condition`

On the next run, OCTTS compares the new `high / low / close` against the previous
decision and marks it as one of:

- `watching_entry`: price has not yet entered the suggested zone
- `entered`: price entered the suggested zone
- `take_profit_hit`: price touched the first target
- `stop_loss_hit`: price touched the stop loss
- `expired`: suggestion outlived its intended horizon
- `no_signal`: suggestion was an `avoid`

## Automation

When `OCTTS_AUTOMATION_ENABLED=true`, the API process will automatically run the configured stock pool on weekdays only (`Monday` through `Friday`). The active slots are controlled by `OCTTS_AUTOMATION_PHASES`, which now defaults to `review` for a single daily LLM pass:

- `morning` at `OCTTS_AUTOMATION_MORNING_TIME`
- `afternoon` at `OCTTS_AUTOMATION_AFTERNOON_TIME`
- `review` at `OCTTS_AUTOMATION_REVIEW_TIME`

Example values:

- `review`
- `morning,review`
- `morning,afternoon,review`

If you already use OpenClaw or system cron, you can disable the built-in scheduler and keep external orchestration.

Time config responsibilities:

- `OCTTS_SCREENING_TIME`: weekday run time for automatic intelligent screening only.
- `OCTTS_AUTOMATION_REVIEW_TIME`: weekday run time for the regular review analysis phase.
- `OCTTS_EMAIL_SEND_TIME`: weekday run time for scheduled report email delivery.

Automatic intelligent screening continues to use `OCTTS_SCREENING_TIME`; it does not reuse `OCTTS_AUTOMATION_REVIEW_TIME`.

## OpenClaw

See `ops/openclaw/README.md` for suggested cron and webhook integration.

## Frontend Pages

- `/dashboard`: stock pool overview, signal cards, validation summary, and OpenClaw panel
- `/stocks/{ts_code}`: single-stock detail page with history replay and decision context
- `/openclaw/status`: reserved JSON endpoint for OpenClaw orchestration status

The dashboard also supports:

- manual single-symbol analysis
- one-click analysis for the default stock pool
- clearing all local analysis data
- clearing the current symbol from its detail page

Repeated analysis for the same `ts_code + trade_date + phase` now overwrites that slot instead of appending duplicate records.

## Local History Layout

- `memory/latest_memory.json`: latest compact memory keyed by symbol
- `memory/history/<ts_code>.json`: full analysis history for each symbol

## Common Commands

```bash
make bootstrap   # create venv and install dependencies
make dev         # local dev server with reload
make run         # local server without reload
make test        # run tests
make docker-up   # build and start docker services
make docker-down # stop docker services
```
