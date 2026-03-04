from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param

from _lambda_utils import invoke_lambda

UTC = timezone.utc

INGESTION_LAMBDA = os.getenv("INGESTION_LAMBDA_NAME", "ingestion_lambda")
VALIDATION_LAMBDA = os.getenv("VALIDATION_LAMBDA_NAME", "validation_lambda")
FORECAST_LAMBDA = os.getenv("FORECAST_LAMBDA_NAME", "forecast_lambda")
GOAL_SEEK_LAMBDA = os.getenv("GOAL_SEEK_LAMBDA_NAME", "goal_seek_lambda")
REPORT_LAMBDA = os.getenv("REPORT_LAMBDA_NAME", "report_lambda")

ENV_RAW_BUCKET = os.getenv("RAW_S3_BUCKET", "")
ENV_RAW_PREFIX = os.getenv("RAW_S3_PREFIX", "raw/zoho")
ENV_REPORTS_BUCKET = os.getenv("REPORTS_S3_BUCKET", "") or ENV_RAW_BUCKET
ENV_REPORTS_PREFIX = os.getenv("REPORTS_S3_PREFIX", "reports")


def _resolve_param(params: dict, key: str, env_default: str | None = None):
    """Param overrides env. If param is empty string, treat it as not provided."""
    v = params.get(key)
    if isinstance(v, str) and v.strip() == "":
        v = None
    return v if v is not None else env_default


def build_event(**context):
    params = dict(context["params"])  # copy

    # Required (always from params)
    required_fields = ["plant_id", "device_id", "lookback_hours"]
    for field in required_fields:
        if not params.get(field):
            raise ValueError(f"{field} is required")

    # Optional override, else env default
    s3_bucket = _resolve_param(params, "s3_bucket", ENV_RAW_BUCKET)
    s3_prefix = _resolve_param(params, "s3_prefix", ENV_RAW_PREFIX)
    reports_s3_bucket = _resolve_param(params, "reports_s3_bucket", ENV_REPORTS_BUCKET)
    reports_s3_prefix = _resolve_param(params, "reports_s3_prefix", ENV_REPORTS_PREFIX)

    # Now validate resolved values
    if not s3_bucket:
        raise ValueError("s3_bucket is required (set RAW_S3_BUCKET env or provide params.s3_bucket)")
    if not s3_prefix:
        raise ValueError("s3_prefix is required (set RAW_S3_PREFIX env or provide params.s3_prefix)")
    if not reports_s3_bucket:
        raise ValueError(
            "reports_s3_bucket is required (set REPORTS_S3_BUCKET or RAW_S3_BUCKET env, or provide params.reports_s3_bucket)"
        )

    # Merge back resolved values so the Lambda always gets them
    params["s3_bucket"] = s3_bucket
    params["s3_prefix"] = s3_prefix
    params["reports_s3_bucket"] = reports_s3_bucket
    params["reports_s3_prefix"] = reports_s3_prefix

    return params


def run_ingestion(**context):
    event = build_event(**context)
    return invoke_lambda(INGESTION_LAMBDA, event)


def run_validation(**context):
    event = build_event(**context)
    ingest_out = context["ti"].xcom_pull(task_ids="ingestion") or {}
    return invoke_lambda(VALIDATION_LAMBDA, {**event, **ingest_out})


def run_forecast(**context):
    event = build_event(**context)
    val_out = context["ti"].xcom_pull(task_ids="validation") or {}
    return invoke_lambda(FORECAST_LAMBDA, {**event, **val_out})


def run_goal_seek(**context):
    event = build_event(**context)
    val_out = context["ti"].xcom_pull(task_ids="validation") or {}
    return invoke_lambda(GOAL_SEEK_LAMBDA, {**event, **val_out})


def run_report(**context):
    event = build_event(**context)
    forecast_out = context["ti"].xcom_pull(task_ids="forecast") or {}
    goal_out = context["ti"].xcom_pull(task_ids="goal_seek") or {}
    # run_id is already in event; this ensures report can find rows for that run_id
    return invoke_lambda(REPORT_LAMBDA, {**event, **forecast_out, **goal_out})


with DAG(
    dag_id="desalter_end_to_end",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["desalter", "pipeline", "lambda"],
    params={
        "plant_id": Param("", type="string", description="Plant Identifier (e.g., CDU1)"),
        "device_id": Param("", type="string", description="Device Identifier (e.g., desalter)"),
        "lookback_hours": Param(24, type="integer", minimum=1, description="Lookback window in hours"),

        "s3_bucket": Param(ENV_RAW_BUCKET, type="string", description="Raw S3 bucket (defaults from RAW_S3_BUCKET)"),
        "s3_prefix": Param(ENV_RAW_PREFIX, type="string", description="Raw S3 prefix (defaults from RAW_S3_PREFIX)"),

        "reports_s3_bucket": Param(ENV_REPORTS_BUCKET, type="string", description="Reports bucket (defaults from REPORTS_S3_BUCKET or RAW_S3_BUCKET)"),
        "reports_s3_prefix": Param(ENV_REPORTS_PREFIX, type="string", description="Reports prefix (defaults from REPORTS_S3_PREFIX)"),

        "metadata": Param(
            {"plant_name": "", "unit_name": "", "location_name": ""},
            type="object",
            description="Metadata information",
        ),
    },
) as dag:

    ingestion = PythonOperator(task_id="ingestion", python_callable=run_ingestion)
    validation = PythonOperator(task_id="validation", python_callable=run_validation)
    forecast = PythonOperator(task_id="forecast", python_callable=run_forecast)
    goal_seek = PythonOperator(task_id="goal_seek", python_callable=run_goal_seek)
    report = PythonOperator(task_id="report", python_callable=run_report)

    ingestion >> validation >> forecast >> goal_seek >> report
