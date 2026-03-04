from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param

from _lambda_utils import invoke_lambda

UTC = timezone.utc

VALIDATION_LAMBDA = os.getenv("VALIDATION_LAMBDA_NAME", "unset-VALIDATION_LAMBDA_NAME")


def _require_nonempty(params: dict, key: str) -> str:
    v = params.get(key)
    if v is None:
        raise ValueError(f"{key} is required")
    if not isinstance(v, str):
        v = str(v)
    v = v.strip()
    if not v:
        raise ValueError(f"{key} is required (non-empty)")
    return v


def build_event(**context) -> dict:
    p = dict(context["params"])

    # required
    p["run_id"] = _require_nonempty(p, "run_id")
    p["device_id"] = _require_nonempty(p, "device_id")
    p["raw_s3_uri"] = _require_nonempty(p, "raw_s3_uri")

    # optional metadata object is already in params schema
    return p


def _run(**context):
    event = build_event(**context)
    return invoke_lambda(VALIDATION_LAMBDA, event)


with DAG(
    dag_id="validation_pipeline",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["desalter", "validation", "lambda"],
    params={
        "run_id": Param("", type="string", description="Pipeline run_id (from ingestion output)"),
        "device_id": Param("desalter", type="string", description="Device Identifier"),
        "raw_s3_uri": Param("", type="string", description="Raw S3 URI produced by ingestion (s3://bucket/key)"),
        "data_start_ts": Param("", type="string", description="Optional ISO timestamp"),
        "data_end_ts": Param("", type="string", description="Optional ISO timestamp"),
        "metadata": Param(
            {"plant_name": "", "unit_name": "", "location_name": ""},
            type="object",
            description="Optional metadata",
        ),
    },
) as dag:
    validation = PythonOperator(task_id="validation", python_callable=_run)
