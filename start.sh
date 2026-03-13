#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PORT=8000

port_pids() {
  lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
}

wait_for_port_release() {
  local attempts="${1:-20}"
  local delay_seconds="${2:-0.5}"
  local remaining="$attempts"

  while [[ "$remaining" -gt 0 ]]; do
    if [[ -z "$(port_pids)" ]]; then
      return 0
    fi
    sleep "$delay_seconds"
    remaining=$((remaining - 1))
  done

  return 1
}

ensure_port_available() {
  local pids
  pids="$(port_pids)"
  if [[ -z "$pids" ]]; then
    return
  fi

  echo "Port $PORT is already in use."
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    echo "  PID $pid: $(ps -p "$pid" -o command= 2>/dev/null || echo "unknown process")"
  done <<< "$pids"

  if [[ ! -t 0 ]]; then
    echo "Cannot prompt in non-interactive mode. Stop the existing process or free port $PORT first."
    exit 1
  fi

  read -r -p "Kill the process(es) above and continue? [y/N] " reply
  if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo "Startup cancelled."
    exit 1
  fi

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    kill -TERM "$pid" 2>/dev/null || true
  done <<< "$pids"

  if wait_for_port_release 20 0.5; then
    return
  fi

  echo "Port $PORT is still busy, forcing shutdown..."
  pids="$(port_pids)"
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    kill -KILL "$pid" 2>/dev/null || true
  done <<< "$pids"

  if ! wait_for_port_release 10 0.5; then
    echo "Failed to free port $PORT."
    exit 1
  fi
}

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

if [[ ! -f .venv/.octts_bootstrapped ]]; then
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/pip install -e '.[dev]'
  touch .venv/.octts_bootstrapped
fi

echo
echo "OCTTS starting on http://127.0.0.1:$PORT"
echo "Dashboard: http://127.0.0.1:$PORT/dashboard"
echo "Stock detail example: http://127.0.0.1:$PORT/stocks/600000.SH"
echo

ensure_port_available

exec .venv/bin/uvicorn octts.api:app --host 0.0.0.0 --port "$PORT" --reload
