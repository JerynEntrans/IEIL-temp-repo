SHELL := /bin/bash

COMPOSE ?= docker compose
LOCALSTACK_SERVICE ?= localstack
BACKUP_WAIT_SECONDS ?= 2
DEPLOY_ENV ?= .env.deploy
EC2_USER ?= ubuntu

.PHONY: help build build-no-cache up start stop safe-stop down restart ps logs logs-localstack \
	health wait-localstack s3-buckets s3-flush clean clean-volumes prune shell-localstack shell-postgres \
	ssh/bastion ssh/airflow \
	train-forecast-local train-goalseek-local \
	deploy-check migrate-db-prod migrate-db-prod-tunnel \
	deploy-lambda-ingestion deploy-lambda-validation deploy-lambda-forecast \
	deploy-lambda-goalseek deploy-lambda-report deploy-lambdas \
	ecr-login airflow-image-build airflow-image-push deploy-airflow

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
	@echo "  make ssh/bastion      SSH into bastion host"
	@echo "  make ssh/airflow      SSH into airflow EC2 host"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Safe cleanup (down + remove orphans)"
	@echo "  make clean-volumes CONFIRM=1  Remove compose volumes too (destructive)"
	@echo "  make prune CONFIRM=1          Prune unused docker resources (destructive)"
	@echo ""
	@echo "── AWS DEPLOYMENT (requires $(DEPLOY_ENV)) ──────────────────────────────────"
	@echo "  make deploy-lambdas               Deploy all lambdas (except training)"
	@echo "  make deploy-lambda-ingestion      Deploy ingestion lambda only"
	@echo "  make deploy-lambda-validation     Deploy validation lambda only"
	@echo "  make deploy-lambda-forecast       Deploy forecast lambda only"
	@echo "  make deploy-lambda-goalseek       Deploy goal-seek lambda only"
	@echo "  make deploy-lambda-report         Deploy report lambda only"
	@echo "  make migrate-db-prod              Run Flyway migrations against RDS"
	@echo "  make migrate-db-prod-tunnel       Run migrations via EC2 SSH tunnel (private RDS)"
	@echo "  make ecr-login                    Authenticate Docker with ECR"
	@echo "  make airflow-image-build          Build production Airflow image"
	@echo "  make airflow-image-push           Build + push image to ECR"
	@echo "  make deploy-airflow               Push image + restart Airflow on EC2"

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

ssh/bastion: deploy-check
	@set -a && source $(DEPLOY_ENV) && set +a && \
	  BASTION_HOST=$${BASTION_HOST_IP:-$${BASTIAN_HOST_IP:-}} && \
	  BASTION_USER=$${BASTION_USER:-$${EC2_USER:-ubuntu}} && \
	  BASTION_KEY=$${BASTION_SSH_KEY:-$$AIRFLOW_EC2_INSTANCE_SSH_KEY} && \
	  if [ -z "$$BASTION_HOST" ]; then \
	    echo "ERROR: Set BASTION_HOST_IP (or BASTIAN_HOST_IP) in $(DEPLOY_ENV)"; \
	    exit 1; \
	  fi && \
	  ssh -i "$$BASTION_KEY" -o StrictHostKeyChecking=no "$$BASTION_USER@$$BASTION_HOST"

ssh/airflow: deploy-check
	@set -a && source $(DEPLOY_ENV) && set +a && \
	  ssh -i "$$AIRFLOW_EC2_INSTANCE_SSH_KEY" -o StrictHostKeyChecking=no "$(EC2_USER)@$$AIRFLOW_EC2_INSTANCE_PUBLIC_IP"

train-forecast-local:
	$(COMPOSE) exec -T airflow-scheduler python /workspace/scripts/train_local.py --model-type DESALTER_FORECAST --device-id $${DEVICE_ID:-desalter} --activate

train-goalseek-local:
	$(COMPOSE) exec -T airflow-scheduler python /workspace/scripts/train_local.py --model-type DESALTER_GOAL_SEEK --device-id $${DEVICE_ID:-desalter} --activate

# ── AWS DEPLOYMENT ──────────────────────────────────────────────────────────────
# All deploy targets source $(DEPLOY_ENV) so credentials/config are in one place.
# Usage: make deploy-lambdas   or   make deploy-airflow

deploy-check:
	@test -f $(DEPLOY_ENV) || { echo "ERROR: $(DEPLOY_ENV) not found. Copy and fill it in."; exit 1; }
	@test -r "$$(grep -E '^AIRFLOW_EC2_INSTANCE_SSH_KEY=' $(DEPLOY_ENV) | cut -d= -f2)" 2>/dev/null || \
		echo "WARN: SSH key path in AIRFLOW_EC2_INSTANCE_SSH_KEY may not exist yet."

# Run Flyway DB migrations against the real RDS instance.
migrate-db-prod: deploy-check
	@set -a && source $(DEPLOY_ENV) && set +a && \
	  docker run --rm \
	    -v "$(PWD)/db_migrations/flyway/sql:/flyway/sql:ro" \
	    flyway/flyway:10 \
	      -url="jdbc:postgresql://$$DB_HOST:$${DB_PORT:-5432}/$$DB_NAME" \
	      -user="$$DB_USER" \
	      -password="$$DB_PASSWORD" \
	      -connectRetries=30 \
	      -locations=filesystem:/flyway/sql \
	      migrate

# Run Flyway from bastion against private RDS:
# 1) copy SQL migrations to bastion
# 2) ensure Airflow metadata DB exists
# 3) execute flyway on bastion against RDS
# 4) clean up remote temp folder
# Example: make migrate-db-prod-tunnel
migrate-db-prod-tunnel: deploy-check
	@set -euo pipefail; \
	set -a; source "$(DEPLOY_ENV)"; set +a; \
	SSH_USER="$${EC2_USER:-ubuntu}"; \
	BASTION_HOST="$${BASTION_HOST_IP:-$${BASTIAN_HOST_IP:-}}"; \
	BASTION_USER="$${BASTION_USER:-$$SSH_USER}"; \
	BASTION_KEY="$${BASTION_SSH_KEY:-$${AIRFLOW_EC2_INSTANCE_SSH_KEY:-}}"; \
	DB_HOST_CLEAN="$${DB_HOST//\"/}"; \
	REMOTE_BASE="$${BASTION_FLYWAY_DIR:-/tmp/ieil-flyway}"; \
	REMOTE_DIR="$$REMOTE_BASE-$$(date +%s)"; \
	if [ -z "$$BASTION_HOST" ]; then \
	  echo "ERROR: Set BASTION_HOST_IP (or BASTIAN_HOST_IP) in $(DEPLOY_ENV)"; \
	  exit 1; \
	fi; \
	echo "Copying Flyway SQL to $$BASTION_USER@$$BASTION_HOST:$$REMOTE_DIR ..."; \
	ssh -o BatchMode=yes -o StrictHostKeyChecking=no -i "$$BASTION_KEY" "$$BASTION_USER@$$BASTION_HOST" "mkdir -p '$$REMOTE_DIR'"; \
	scp -o BatchMode=yes -o StrictHostKeyChecking=no -i "$$BASTION_KEY" -r "$(PWD)/db_migrations/flyway/sql" "$$BASTION_USER@$$BASTION_HOST:$$REMOTE_DIR/"; \
	echo "Ensuring Airflow database exists..."; \
	ssh -o BatchMode=yes -o StrictHostKeyChecking=no -i "$$BASTION_KEY" "$$BASTION_USER@$$BASTION_HOST" "\
	  command -v psql >/dev/null 2>&1 || { echo 'ERROR: psql not found on bastion PATH'; exit 127; }; \
	  export PGPASSWORD='$$DB_PASSWORD'; \
	  if ! psql -h $$DB_HOST_CLEAN -p $${DB_PORT:-5432} -U $$DB_USER -d postgres -tAc \"SELECT 1 FROM pg_roles WHERE rolname='$$AIRFLOW_DB_USER'\" | grep -q 1; then \
	    psql -h $$DB_HOST_CLEAN -p $${DB_PORT:-5432} -U $$DB_USER -d postgres -c \"CREATE ROLE \\\"$$AIRFLOW_DB_USER\\\" LOGIN PASSWORD '$$AIRFLOW_DB_PASSWORD';\"; \
	  else \
	    psql -h $$DB_HOST_CLEAN -p $${DB_PORT:-5432} -U $$DB_USER -d postgres -c \"ALTER ROLE \\\"$$AIRFLOW_DB_USER\\\" WITH LOGIN PASSWORD '$$AIRFLOW_DB_PASSWORD';\"; \
	  fi; \
	  if ! psql -h $$DB_HOST_CLEAN -p $${DB_PORT:-5432} -U $$DB_USER -d postgres -tAc \"SELECT 1 FROM pg_database WHERE datname='$$AIRFLOW_DB_NAME'\" | grep -q 1; then \
	    psql -h $$DB_HOST_CLEAN -p $${DB_PORT:-5432} -U $$DB_USER -d postgres -c \"CREATE DATABASE \\\"$$AIRFLOW_DB_NAME\\\" OWNER \\\"$$DB_USER\\\";\"; \
	  fi; \
	  psql -h $$DB_HOST_CLEAN -p $${DB_PORT:-5432} -U $$DB_USER -d postgres -c \"GRANT ALL PRIVILEGES ON DATABASE \\\"$$AIRFLOW_DB_NAME\\\" TO \\\"$$AIRFLOW_DB_USER\\\";\"; \
	  psql -h $$DB_HOST_CLEAN -p $${DB_PORT:-5432} -U $$DB_USER -d $$AIRFLOW_DB_NAME -c \"GRANT USAGE, CREATE ON SCHEMA public TO \\\"$$AIRFLOW_DB_USER\\\";\" \
	"; \
	echo "Running Flyway on bastion host..."; \
	ssh -o BatchMode=yes -o StrictHostKeyChecking=no -i "$$BASTION_KEY" "$$BASTION_USER@$$BASTION_HOST" "\
	  command -v flyway >/dev/null 2>&1 || { echo 'ERROR: flyway not found on bastion PATH'; exit 127; }; \
	  flyway \
	    -url=jdbc:postgresql://$$DB_HOST_CLEAN:$${DB_PORT:-5432}/$$DB_NAME \
	    -user=$$DB_USER \
	    -password=$$DB_PASSWORD \
	    -connectRetries=30 \
	    -locations=filesystem:$$REMOTE_DIR/sql \
	    migrate\
	"; \
	echo "Cleaning up remote migration folder..."; \
	ssh -o BatchMode=yes -o StrictHostKeyChecking=no -i "$$BASTION_KEY" "$$BASTION_USER@$$BASTION_HOST" "rm -rf '$$REMOTE_DIR'"; \
	echo "Migration from bastion complete."

# ── Lambda deploys ──────────────────────────────────────────────────────────────
deploy-lambda-ingestion: deploy-check
	@set -a && source $(DEPLOY_ENV) && set +a && \
	  LAMBDA_NAME=$$INGEST_LAMBDA_NAME \
	  bash scripts/deploy-lambda.sh ingestion_lambda common

deploy-lambda-validation: deploy-check
	@set -a && source $(DEPLOY_ENV) && set +a && \
	  LAMBDA_NAME=$$VALIDATION_LAMBDA_NAME \
	  bash scripts/deploy-lambda.sh validation_lambda common

deploy-lambda-forecast: deploy-check
	@set -a && source $(DEPLOY_ENV) && set +a && \
	  DEPS=$$([ "$$USE_ML_MODELS" = "true" ] && echo ml || echo common) && \
	  LAMBDA_NAME=$$FORECAST_LAMBDA_NAME \
	  bash scripts/deploy-lambda.sh forecast_lambda $$DEPS

deploy-lambda-goalseek: deploy-check
	@set -a && source $(DEPLOY_ENV) && set +a && \
	  DEPS=$$([ "$$USE_ML_MODELS" = "true" ] && echo ml || echo common) && \
	  LAMBDA_NAME=$$GOAL_SEEK_LAMBDA_NAME \
	  bash scripts/deploy-lambda.sh goal_seek_lambda $$DEPS

deploy-lambda-report: deploy-check
	@set -a && source $(DEPLOY_ENV) && set +a && \
	  LAMBDA_NAME=$$REPORT_LAMBDA_NAME \
	  bash scripts/deploy-lambda.sh report_lambda common

deploy-lambdas: deploy-lambda-ingestion deploy-lambda-validation \
                deploy-lambda-forecast deploy-lambda-goalseek deploy-lambda-report
	@echo "All lambdas deployed."

# ── Airflow EC2 deploy ──────────────────────────────────────────────────────────
ecr-login: deploy-check
	@set -a && source $(DEPLOY_ENV) && set +a && \
	  aws ecr get-login-password --region $$AWS_REGION | \
	  docker login --username AWS --password-stdin $$ECR_HOST

airflow-image-build: deploy-check
	@set -a && source $(DEPLOY_ENV) && set +a && \
	  docker build \
	    -f docker/Dockerfile.airflow.prod \
	    -t $$ECR_HOST/$$ECR_REPOSITORY:$$ECR_IMAGE_TAG \
	    -t $$AIRFLOW_DOCKER_IMAGE_NAME:latest \
	    .

airflow-image-push: ecr-login airflow-image-build
	@set -a && source $(DEPLOY_ENV) && set +a && \
	  docker push $$ECR_HOST/$$ECR_REPOSITORY:$$ECR_IMAGE_TAG
	@echo "Image pushed to ECR."

# Copies compose file + env to EC2, ECR-logins the EC2, pulls the new image,
# then restarts Airflow services.
deploy-airflow: airflow-image-push deploy-check
	@set -a && source $(DEPLOY_ENV) && set +a && \
	  SSH_USER="$${EC2_USER:-$(EC2_USER)}" && \
	  SSH_OPTS="-i $$AIRFLOW_EC2_INSTANCE_SSH_KEY -o StrictHostKeyChecking=no -o BatchMode=yes" && \
	  EC2="$$SSH_USER@$$AIRFLOW_EC2_INSTANCE_PUBLIC_IP" && \
	  echo "==> Copying files to EC2..." && \
	  scp $$SSH_OPTS $(DEPLOY_ENV)             $$EC2:/home/$$SSH_USER/.env.airflow && \
	  scp $$SSH_OPTS docker-compose.prod.yml   $$EC2:/home/$$SSH_USER/docker-compose.prod.yml && \
	  echo "==> Creating ECR login token locally..." && \
	  ECR_LOGIN_PASSWORD=$$(aws ecr get-login-password --region $$AWS_REGION) && \
	  [ -n "$$ECR_LOGIN_PASSWORD" ] || { echo "ERROR: failed to get ECR login password from local AWS creds"; exit 1; } && \
	  echo "==> Docker login on EC2..." && \
	  printf '%s' "$$ECR_LOGIN_PASSWORD" | ssh $$SSH_OPTS $$EC2 \
	    "command -v docker >/dev/null 2>&1 || { echo 'ERROR: docker not found on EC2 PATH'; exit 127; }; docker login --username AWS --password-stdin $$ECR_HOST" && \
	  echo "==> Pulling new image on EC2..." && \
	  ssh $$SSH_OPTS $$EC2 "docker pull $$ECR_HOST/$$ECR_REPOSITORY:$$ECR_IMAGE_TAG" && \
	  echo "==> Restarting Airflow services on EC2..." && \
	  ssh $$SSH_OPTS $$EC2 \
	    "cd /home/$$SSH_USER && \
	     ECR_HOST=$$ECR_HOST ECR_REPOSITORY=$$ECR_REPOSITORY ECR_IMAGE_TAG=$$ECR_IMAGE_TAG \
	     docker compose -f docker-compose.prod.yml --env-file .env.airflow up -d --force-recreate" && \
	  echo "==> Airflow deployed to $$AIRFLOW_EC2_INSTANCE_PUBLIC_IP."

