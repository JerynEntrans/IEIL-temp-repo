#!/usr/bin/env bash
# Production Airflow entrypoint.
# Waits for the RDS postgres instance; does NOT wait for LocalStack.
set -euo pipefail

wait_for_port() {
  local host="$1"
  local port="$2"
  echo "Waiting for $host:$port..."
  python - <<PY
import socket, time, sys
host = "${host}"
port = int("${port}")
for _ in range(120):
    try:
        with socket.create_connection((host, port), timeout=2):
            sys.exit(0)
    except OSError:
        time.sleep(2)
print(f"ERROR: could not reach {host}:{port} after 240 s", flush=True)
sys.exit(1)
PY
}

wait_for_port "${DB_HOST}" "${DB_PORT:-5432}"

case "${1:-}" in
  init)
    airflow db migrate
    airflow users create \
      --username admin \
      --firstname Admin \
      --lastname User \
      --role Admin \
      --email jeryn@thunai.com \
      --password "${AIRFLOW_ADMIN_PASSWORD:-admin}" || true
    ;;
  webserver)
    exec airflow webserver
    ;;
  scheduler)
    exec airflow scheduler
    ;;
  *)
    exec "$@"
    ;;
esac
