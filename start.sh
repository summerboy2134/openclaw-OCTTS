#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PORT=8000

port_pids() {
  lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
}

cleanup_port_state() {
  local pids
  pids="$(lsof -tiTCP:"$PORT" 2>/dev/null || true)"
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    kill -TERM "$pid" 2>/dev/null || true
  done <<< "$pids"

  sleep 1

  pids="$(lsof -tiTCP:"$PORT" 2>/dev/null || true)"
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    kill -KILL "$pid" 2>/dev/null || true
  done <<< "$pids"
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

  echo "Port $PORT is already in use. Stopping existing listener..."
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    echo "  PID $pid: $(ps -p "$pid" -o command= 2>/dev/null || echo "unknown process")"
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

deps_fingerprint() {
  shasum pyproject.toml | awk '{print $1}'
}

ensure_dependencies() {
  local marker_file=".venv/.octts_bootstrapped"
  local fingerprint_file=".venv/.octts_deps_fingerprint"
  local current_fingerprint=""
  local installed_fingerprint=""
  local needs_install=0

  current_fingerprint="$(deps_fingerprint)"
  if [[ -f "$fingerprint_file" ]]; then
    installed_fingerprint="$(<"$fingerprint_file")"
  fi

  if [[ ! -f "$marker_file" ]]; then
    needs_install=1
  elif [[ "$current_fingerprint" != "$installed_fingerprint" ]]; then
    needs_install=1
  elif ! .venv/bin/python -c "import fastapi, sqlalchemy" >/dev/null 2>&1; then
    needs_install=1
  fi

  if [[ "$needs_install" -eq 1 ]]; then
    echo "Installing or refreshing Python dependencies..."
    .venv/bin/python -m pip install --upgrade pip >/dev/null
    .venv/bin/pip install -e '.[dev]'
    printf '%s\n' "$current_fingerprint" > "$fingerprint_file"
    touch "$marker_file"
  fi
}

ensure_dependencies

echo
echo "OCTTS starting on http://127.0.0.1:$PORT"
echo "Dashboard: http://127.0.0.1:$PORT/dashboard"
echo "Stock detail example: http://127.0.0.1:$PORT/stocks/600000.SH"
echo

ensure_port_available
cleanup_port_state

exec .venv/bin/uvicorn octts.api:app --host 0.0.0.0 --port "$PORT"
