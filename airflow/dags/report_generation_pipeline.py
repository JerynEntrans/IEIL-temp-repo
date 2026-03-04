from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param

from _lambda_utils import invoke_lambda

UTC = timezone.utc

REPORT_LAMBDA = os.getenv("REPORT_LAMBDA_NAME", "unset-REPORT_LAMBDA_NAME")

ENV_RAW_BUCKET = os.getenv("RAW_S3_BUCKET", "")
ENV_REPORTS_BUCKET = os.getenv("REPORTS_S3_BUCKET", "") or ENV_RAW_BUCKET
ENV_REPORTS_PREFIX = os.getenv("REPORTS_S3_PREFIX", "reports")


def _require_nonempty(params: dict, key: str) -> str:
    """Get non-empty string param, or raise ValueError."""
    v = params.get(key)
    if v is None:
        raise ValueError(f"{key} is required")
    if not isinstance(v, str):
        v = str(v)
    v = v.strip()
    if not v:
        raise ValueError(f"{key} is required (non-empty)")
    return v


def _resolve(params: dict, key: str, default: str) -> str:
    v = params.get(key)
    if isinstance(v, str) and v.strip() == "":
        v = None
    return v if v is not None else default


def build_event(**context) -> dict:
    p = dict(context["params"])

    p["run_id"] = _require_nonempty(p, "run_id")
    p["device_id"] = _require_nonempty(p, "device_id")

    p["reports_s3_bucket"] = _resolve(p, "reports_s3_bucket", ENV_REPORTS_BUCKET or ENV_RAW_BUCKET)
    p["reports_s3_prefix"] = _resolve(p, "reports_s3_prefix", ENV_REPORTS_PREFIX)

    if not p["reports_s3_bucket"]:
        raise ValueError("reports_s3_bucket is required (set REPORTS_S3_BUCKET or RAW_S3_BUCKET env, or provide param)")

    return p


def _run(**context):
    event = build_event(**context)
    return invoke_lambda(REPORT_LAMBDA, event)


with DAG(
    dag_id="report_generation_pipeline",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["desalter", "report", "lambda"],
    params={
        "run_id": Param("", type="string", description="Pipeline run_id to generate report for"),
        "device_id": Param("desalter", type="string", description="Device Identifier"),
        "report_type": Param("daily_summary", type="string", description="Report type name"),
        "reports_s3_bucket": Param(ENV_REPORTS_BUCKET or ENV_RAW_BUCKET, type="string", description="Reports bucket (env default, overrideable)"),
        "reports_s3_prefix": Param(ENV_REPORTS_PREFIX, type="string", description="Reports prefix (env default, overrideable)"),
        "data_start_ts": Param("", type="string", description="Optional ISO timestamp"),
        "data_end_ts": Param("", type="string", description="Optional ISO timestamp"),
    },
) as dag:
    report = PythonOperator(task_id="report", python_callable=_run)
