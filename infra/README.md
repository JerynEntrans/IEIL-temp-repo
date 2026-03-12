# Infra prep for Pulumi or Terraform

This repo is very close to being IaC-friendly already. Before you move it to Pulumi or Terraform, make these repo-level conventions stick:

## 1. Keep local and cloud config driven by variables
Use the same variable names everywhere:
- `APP_DB_NAME`, `APP_DB_USER`, `APP_DB_PASSWORD`
- `AIRFLOW_DB_NAME`, `AIRFLOW_DB_USER`, `AIRFLOW_DB_PASSWORD`
- `AWS_DEFAULT_REGION`
- bucket names and Lambda names

That lets Docker Compose, Terraform, and Pulumi all drive the same app contract.

## 2. Separate concerns clearly
You currently have two logical databases:
- app database: business tables managed by Flyway
- Airflow metadata database: DAG runs, task instances, connections, logs metadata

Keep that separation in cloud too. It makes upgrades, backup policy, and troubleshooting much cleaner.

## 3. Make these paths/modules stable
Try to preserve these stable folders:
- `docker/`
- `db_migrations/flyway/sql/`
- `services/`
- `shared/`
- `airflow/`
- future `infra/terraform/` or `infra/pulumi/`

## 4. Prefer one source of truth for runtime settings
Create a small app config contract document and keep env names identical across:
- local compose
- Lambda config
- Airflow env
- IaC outputs

## 5. Cloud resources you will likely model first
- VPC/subnets/security groups
- RDS Postgres (or two databases inside one Postgres instance)
- S3 buckets for raw/reports
- Lambda functions + IAM roles
- EventBridge or Airflow hosting choice
- secrets storage
- networking from orchestration layer to DB/Lambdas

## 6. Recommended next refactors before IaC
- add `.env.local.example` and stop hardcoding DB values in compose
- keep Postgres bootstrap script under `docker/postgres/init/`
- move repeated Lambda names and bucket names to env vars everywhere
- document first-run and reset flow
- add healthchecks where startup order matters

## 7. Terraform/Pulumi-ready outputs to plan for
Whichever IaC tool you choose, expose outputs like:
- database host
- database port
- app db name
- airflow db name
- bucket names
- lambda names
- security group ids
- subnet ids

## 8. Biggest design choice later
Decide whether Airflow stays local/dev only, or becomes a managed cloud component.

Typical paths:
- **Terraform/Pulumi + MWAA** if you want managed Airflow on AWS
- **Terraform/Pulumi + ECS/EKS self-hosted Airflow** if you want more control
- **Terraform/Pulumi + EventBridge + Lambdas** if Airflow is only a local orchestration convenience
