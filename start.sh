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

detect_conda_env() {
  if [[ -n "${OCTTS_CONDA_ENV:-}" ]]; then
    echo "${OCTTS_CONDA_ENV}"
    return 0
  fi
  if [[ "${CONDA_DEFAULT_ENV:-}" == "ai-test" ]]; then
    echo "ai-test"
    return 0
  fi
  if [[ -n "${CONDA_PREFIX:-}" && "${CONDA_PREFIX:-}" == *"/envs/ai-test" ]]; then
    echo "ai-test"
    return 0
  fi
  echo ""
}

deps_fingerprint() {
  shasum pyproject.toml | awk '{print $1}'
}

ensure_dependencies_venv() {
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
  fi

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

ensure_dependencies_conda() {
  local conda_env="$1"
  local marker_file=".conda/.octts_bootstrapped_${conda_env}"
  local fingerprint_file=".conda/.octts_deps_fingerprint_${conda_env}"
  local current_fingerprint=""
  local installed_fingerprint=""
  local needs_install=0

  mkdir -p .conda

  current_fingerprint="$(deps_fingerprint)"
  if [[ -f "$fingerprint_file" ]]; then
    installed_fingerprint="$(<"$fingerprint_file")"
  fi

  if [[ ! -f "$marker_file" ]]; then
    needs_install=1
  elif [[ "$current_fingerprint" != "$installed_fingerprint" ]]; then
    needs_install=1
  elif ! conda run -n "$conda_env" python -c "import fastapi, sqlalchemy" >/dev/null 2>&1; then
    needs_install=1
  fi

  if [[ "$needs_install" -eq 1 ]]; then
    echo "Installing or refreshing Python dependencies in conda env: $conda_env"
    conda run -n "$conda_env" python -m pip install --upgrade pip >/dev/null
    conda run -n "$conda_env" python -m pip install -e '.[dev]'
    printf '%s\n' "$current_fingerprint" > "$fingerprint_file"
    touch "$marker_file"
  fi
}

CONDA_ENV_NAME="$(detect_conda_env)"
if [[ -n "$CONDA_ENV_NAME" ]]; then
  echo "Using conda env: $CONDA_ENV_NAME"
  conda run -n "$CONDA_ENV_NAME" python -c "import sys; print('Python:', sys.executable); print(sys.version)"
  ensure_dependencies_conda "$CONDA_ENV_NAME"
else
  echo "Using venv: .venv"
  ensure_dependencies_venv
fi

echo
echo "OCTTS starting on http://127.0.0.1:$PORT"
echo "Dashboard: http://127.0.0.1:$PORT/dashboard"
echo "Stock detail example: http://127.0.0.1:$PORT/stocks/600000.SH"
echo

ensure_port_available
cleanup_port_state

echo "Press Ctrl+C to stop."

SERVER_PID=""
cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup_server INT TERM EXIT

if [[ -n "$CONDA_ENV_NAME" ]]; then
  # If the env is already active, prefer direct python for correct signal handling.
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    "${CONDA_PREFIX}/bin/python" -m uvicorn octts.api:app --host 0.0.0.0 --port "$PORT" &
  else
    conda run -n "$CONDA_ENV_NAME" python -m uvicorn octts.api:app --host 0.0.0.0 --port "$PORT" &
  fi
else
  .venv/bin/uvicorn octts.api:app --host 0.0.0.0 --port "$PORT" &
fi

SERVER_PID="$!"
wait "$SERVER_PID"
