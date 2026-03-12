#!/usr/bin/env bash
set -euo pipefail

wait_for_port() {
  local host="$1"
  local port="$2"
  python - <<PY
import socket, time
host = "${host}"
port = int("${port}")
for _ in range(120):
    try:
        with socket.create_connection((host, port), timeout=2):
            raise SystemExit(0)
    except OSError:
        time.sleep(2)
raise SystemExit(1)
PY
}

wait_for_port postgres 5432
wait_for_port localstack 4566

case "${1:-}" in
  init)
    airflow db migrate
    airflow users create \
      --username admin \
      --firstname Admin \
      --lastname User \
      --role Admin \
      --email admin@example.com \
      --password admin || true
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
