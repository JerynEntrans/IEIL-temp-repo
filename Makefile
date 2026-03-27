SHELL := /bin/bash

COMPOSE ?= docker compose
LOCALSTACK_SERVICE ?= localstack
BACKUP_WAIT_SECONDS ?= 2

.PHONY: help build build-no-cache up start stop safe-stop down restart ps logs logs-localstack \
	health wait-localstack s3-buckets s3-flush clean clean-volumes prune shell-localstack shell-postgres \
	train-forecast-local train-goalseek-local

help:
	@echo "IEIL local stack operations"
	@echo ""
	@echo "Core lifecycle:"
	@echo "  make build            Build images"
	@echo "  make up               Build and start all services in background"
	@echo "  make start            Start stopped services"
	@echo "  make stop             Stop all services"
	@echo "  make safe-stop        Flush LocalStack S3 backup, then stop all services"
	@echo "  make down             Stop and remove containers and networks"
	@echo "  make restart          Safe stop + start"
	@echo ""
	@echo "Observability:"
	@echo "  make ps               Show service status"
	@echo "  make logs             Follow all logs"
	@echo "  make logs-localstack  Follow LocalStack logs"
	@echo "  make health           Check docker compose, LocalStack and Postgres"
	@echo ""
	@echo "LocalStack S3:"
	@echo "  make s3-buckets       List S3 buckets in LocalStack"
	@echo "  make s3-flush         Force one immediate S3 backup snapshot"
	@echo ""
	@echo "Model training:"
	@echo "  make train-forecast-local [DEVICE_ID=desalter]  Train + activate forecast model"
	@echo "  make train-goalseek-local [DEVICE_ID=desalter]  Train + activate goal-seek model"
	@echo ""
	@echo "Shell access:"
	@echo "  make shell-localstack Open shell in LocalStack container"
	@echo "  make shell-postgres   Open shell in Postgres container"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Safe cleanup (down + remove orphans)"
	@echo "  make clean-volumes CONFIRM=1  Remove compose volumes too (destructive)"
	@echo "  make prune CONFIRM=1          Prune unused docker resources (destructive)"

build:
	$(COMPOSE) build

build-no-cache:
	$(COMPOSE) build --no-cache

up:
	$(COMPOSE) up -d --build

start:
	$(COMPOSE) start

stop:
	$(COMPOSE) stop

safe-stop: s3-flush
	@sleep $(BACKUP_WAIT_SECONDS)
	$(COMPOSE) stop

down: s3-flush
	@sleep $(BACKUP_WAIT_SECONDS)
	$(COMPOSE) down --remove-orphans

restart: safe-stop start

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f

logs-localstack:
	$(COMPOSE) logs -f $(LOCALSTACK_SERVICE)

wait-localstack:
	@echo "Waiting for LocalStack to be healthy..."
	@until [ "$$($(COMPOSE) ps --status running --services | grep -x $(LOCALSTACK_SERVICE))" = "$(LOCALSTACK_SERVICE)" ]; do \
		sleep 1; \
	done
	@echo "LocalStack container is running."

health:
	@echo "== compose status =="
	@$(COMPOSE) ps
	@echo ""
	@echo "== LocalStack health endpoint =="
	@curl -sS http://localhost:4566/_localstack/health || true
	@echo ""
	@echo "== Postgres readiness check =="
	@$(COMPOSE) exec -T postgres pg_isready -U $${APP_DB_USER:-ieil} -d $${APP_DB_NAME:-ieil} || true

s3-buckets:
	aws --endpoint-url=http://localhost:4566 s3 ls

s3-flush:
	@echo "Flushing LocalStack S3 backup..."
	@$(COMPOSE) exec -T $(LOCALSTACK_SERVICE) /etc/localstack/init/shutdown.d/00-save-s3.sh || true

clean:
	$(COMPOSE) down --remove-orphans

clean-volumes:
	@if [ "$(CONFIRM)" != "1" ]; then \
		echo "Refusing destructive action. Re-run with: make clean-volumes CONFIRM=1"; \
		exit 1; \
	fi
	$(COMPOSE) down -v --remove-orphans

prune:
	@if [ "$(CONFIRM)" != "1" ]; then \
		echo "Refusing destructive action. Re-run with: make prune CONFIRM=1"; \
		exit 1; \
	fi
	docker system prune -f

shell-localstack:
	$(COMPOSE) exec $(LOCALSTACK_SERVICE) bash

shell-postgres:
	$(COMPOSE) exec postgres bash

train-forecast-local:
	$(COMPOSE) exec -T airflow-scheduler python /workspace/scripts/train_local.py --model-type DESALTER_FORECAST --device-id $${DEVICE_ID:-desalter} --activate

train-goalseek-local:
	$(COMPOSE) exec -T airflow-scheduler python /workspace/scripts/train_local.py --model-type DESALTER_GOAL_SEEK --device-id $${DEVICE_ID:-desalter} --activate
