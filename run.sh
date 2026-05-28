#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

wait_for_docker() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi

  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "Docker is not running. Opening Docker Desktop..."
    open -a Docker >/dev/null 2>&1 || true
  fi

  echo "Waiting for Docker daemon..."
  for _ in {1..90}; do
    if docker info >/dev/null 2>&1; then
      echo "Docker is ready."
      return 0
    fi
    sleep 2
  done

  echo "Docker daemon is still unavailable. Start Docker Desktop and rerun ./run.sh." >&2
  exit 1
}

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Fill one LLM API key for analysis features."
fi

mkdir -p logs data
wait_for_docker

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
echo "Using compose file: $COMPOSE_FILE"

docker compose -f "$COMPOSE_FILE" up -d --build

echo "Waiting for HTTP health endpoint..."
for _ in {1..90}; do
  if curl -fsS http://localhost:8000/api/health >/dev/null 2>&1; then
    if curl -fsS http://localhost:3000 >/dev/null 2>&1; then
      echo "TradingAgents-CN is running:"
      echo "  Frontend: http://localhost:3000"
      echo "  Backend:  http://localhost:8000"
      exit 0
    fi
  fi
  if curl -fsS http://localhost/health >/dev/null 2>&1 || curl -fsS http://localhost/api/health >/dev/null 2>&1; then
    echo "TradingAgents-CN is running: http://localhost"
    exit 0
  fi
  sleep 2
done

echo "Services started, but health check did not pass yet. Current status:"
docker compose -f "$COMPOSE_FILE" ps
exit 1
