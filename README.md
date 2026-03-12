# IEIL – Temporary Development Repository

**Repository:** IEIL (Internal Engineering Integration Layer)  
**Status:** 🚧 Work in progress – temporary repo for initial development and end-to-end testing.

This repository contains the early-stage implementation of the IEIL data pipeline and supporting services.  
It is intended for **local experimentation, integration testing, and architecture validation** before moving to a permanent production repository.

> ⚠️ This repository will likely be made private or archived after initial validation.

---

# Project Overview

The goal of this repository is to prototype and validate the architecture for:

- Data ingestion workflows
- Processing pipelines
- Forecasting and analytical workflows
- Infrastructure orchestration

Key technologies currently used:

- **Python**
- **Apache Airflow**
- **PostgreSQL**
- **Docker / Docker Compose**
- **Flyway (database migrations)**
- **LocalStack (AWS emulation)**

---

# Repository Structure

Example structure (may change frequently during development):

```
.
├── dags/                  # Airflow DAG definitions
├── shared/                # Shared utilities and schemas
│   ├── schema/
│   └── utils/
├── requirements/          # Dependency files
│   ├── airflow.txt
│   ├── lambda_common.txt
│   ├── lambda_ml.txt
│   └── requirements_old.txt
├── docker/                # Docker related configs
├── flyway/                # Database migrations
├── scripts/               # Helper scripts
└── docker-compose.yml     # Local development stack
```

---

# Local Development Setup

## 1. Clone the repository

```bash
git clone https://github.com/JerynEntrans/IEIL-temp-repo.git
cd IEIL-temp-repo
```

---

## 2. Start the development environment

```bash
docker compose up --build
```

This will typically start:

- PostgreSQL
- Airflow
- Flyway migrations
- LocalStack (if enabled)

---

## 3. Access services

Typical local endpoints:

| Service | URL |
|------|------|
| Airflow UI | http://localhost:8080 |
| PostgreSQL | localhost:5432 |
| LocalStack | http://localhost:4566 |

Credentials and connection details may be configured in `.env`.

---

# Database Migrations

Database schema is managed using **Flyway**.

Migration files are located in:

```
flyway/sql
```

When the environment starts, Flyway automatically applies pending migrations.

---

# Development Notes

- This repository is **experimental** and may change frequently.
- Some modules are **placeholders or prototypes**.
- Not all workflows are production ready.
- Configuration and folder structure may evolve as the architecture stabilizes.

---

# Future Improvements

Planned enhancements include:

- S3-based ingestion pipelines
- Lambda-based ingestion services
- Improved DAG orchestration
- Production-ready deployment configuration
- Model registry and forecasting pipeline integration

---

# License

Internal / Private use.