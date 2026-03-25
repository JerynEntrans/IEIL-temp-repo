from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param

from _lambda_utils import invoke_lambda

UTC = timezone.utc
DEFAULT_FORECAST_HORIZONS = [30, 60, 120]

FORECAST_LAMBDA = os.getenv("FORECAST_LAMBDA_NAME", "unset-FORECAST_LAMBDA_NAME")


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

    p["run_id"] = _require_nonempty(p, "run_id")
    p["device_id"] = _require_nonempty(p, "device_id")

    # REQUIRED for run-id based forecast
    p["validated_run_id"] = _require_nonempty(p, "validated_run_id")

    # sanitize horizons_minutes: allow list or comma string
    mv = p.get("model_version")
    if isinstance(mv, str) and not mv.strip():
        p.pop("model_version", None)
    hm = p.get("horizons_minutes")
    if isinstance(hm, str):
        hm = [int(x.strip()) for x in hm.split(",") if x.strip()]
    if hm is None:
        hm = DEFAULT_FORECAST_HORIZONS
    hm = sorted(set(int(x) for x in hm))
    if any(x <= 0 for x in hm):
        raise ValueError(f"horizons_minutes must contain only positive integers. got={hm}")
    p["horizons_minutes"] = hm

    return p


def _run(**context):
    event = build_event(**context)
    return invoke_lambda(FORECAST_LAMBDA, event)


with DAG(
    dag_id="forecast_pipeline",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["desalter", "forecast", "lambda"],
    params={
        "run_id": Param("", type="string", description="Pipeline run_id (same run used to write forecast rows)"),
        "validated_run_id": Param("", type="string", description="Run id of validated data to read from"),
        "device_id": Param("desalter", type="string", description="Device Identifier"),
        "horizons_minutes": Param(DEFAULT_FORECAST_HORIZONS, type="array", description="List of horizons in minutes"),
        "model_version": Param("", type="string", description="Optional: pin model_version. If empty, use ACTIVE model from model_registry."),
        "data_start_ts": Param("", type="string", description="Optional ISO timestamp"),
        "data_end_ts": Param("", type="string", description="Optional ISO timestamp (base timestamp)"),
        "forecast_timestamp": Param("", type="string", description="Optional ISO timestamp override"),
    },
) as dag:
    forecast = PythonOperator(task_id="forecast", python_callable=_run)
