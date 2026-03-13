# OpenClaw Integration

This project is designed so that `OpenClaw` handles scheduling and orchestration,
while the Python service handles market data, memory, analysis, and delivery.

## Recommended Flow

1. Run the OCTTS API somewhere OpenClaw can reach, for example `http://127.0.0.1:8000`.
2. Configure OpenClaw cron jobs for `09:35`, `14:35`, and `20:30`.
3. Each cron job should trigger an HTTP call to `/analyze` with the target phase.
4. OCTTS reads the last memory for each stock, fetches current market data,
   validates older suggestions against the new price range, generates a new report,
   stores compact memory, archives the decision, and pushes to WeCom.

For quick setup, OCTTS itself can now be started with:

```bash
./start.sh
```

Or deployed with:

```bash
docker compose up -d --build
```

## Suggested Phase Mapping

- `09:35` -> `morning`
- `14:35` -> `afternoon`
- `20:30` -> `review`

## Example Request Payloads

Morning:

```json
{
  "phase": "morning",
  "notify": true
}
```

Review:

```json
{
  "phase": "review",
  "notify": true,
  "stock_pool": ["600000.SH", "000001.SZ"]
}
```

## OpenClaw Webhook Strategy

If you already use OpenClaw webhooks, let OpenClaw trigger a shell script or HTTP
request that POSTs to OCTTS:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"phase":"review","notify":true}'
```

## OpenClaw Cron Job Notes

OpenClaw supports native cron jobs and agent/webhook automation. Keep scheduling
logic in OpenClaw and avoid building another scheduler into OCTTS. That keeps the
project focused on analysis quality and memory continuity.

## Dashboard

After the first successful run, open:

```bash
http://127.0.0.1:8000/dashboard
```

The dashboard shows:

- latest trend judgement
- structured trading decision
- validation status for the current tracked suggestion
- historical timeline of prior suggestions for each symbol
- click-through navigation from overview cards to `/stocks/{ts_code}`
- reserved OpenClaw status entry for future gateway health/job integration

## Operational Advice

- Use Redis in production so memory survives restarts and scales cleanly.
- Keep `WECOM_WEBHOOK_URL` and `LLM_API_KEY` only in environment variables.
- If Tushare minute or moneyflow data is unavailable, OCTTS will fall back to
  partial analysis from the data it can fetch.
- Start with WeCom markdown notifications; add richer cards only after the core
  analysis loop is stable.
