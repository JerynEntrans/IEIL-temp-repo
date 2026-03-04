from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param

from _lambda_utils import invoke_lambda

UTC = timezone.utc

GOAL_SEEK_LAMBDA = os.getenv("GOAL_SEEK_LAMBDA_NAME", "unset-GOAL_SEEK_LAMBDA_NAME")


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
    p["validated_run_id"] = _require_nonempty(p, "validated_run_id")
    mv = p.get("model_version")
    if isinstance(mv, str) and not mv.strip():
        p.pop("model_version", None)

    return p


def _run(**context):
    event = build_event(**context)
    return invoke_lambda(GOAL_SEEK_LAMBDA, event)


with DAG(
    dag_id="goal_seek_pipeline",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["desalter", "goal_seek", "lambda"],
    params={
        "run_id": Param("", type="string", description="Pipeline run_id (same run used to write goal seek row)"),
        "validated_run_id": Param("", type="string", description="Run id of validated data to read from"),
        "device_id": Param("desalter", type="string", description="Device Identifier"),
        "target_interface_level": Param(50.0, type="number", description="Target interface level"),
        "model_version": Param("", type="string", description="Optional: pin model_version. If empty, use ACTIVE model from model_registry."),
        "seed": Param(42, type="integer", description="Random seed for optimizer/search"),
        "trials": Param(200, type="integer", minimum=10, description="Number of candidate trials"),
        "data_start_ts": Param("", type="string", description="Optional ISO timestamp"),
        "data_end_ts": Param("", type="string", description="Optional ISO timestamp"),
        "run_timestamp": Param("", type="string", description="Optional ISO timestamp override"),
    },
) as dag:
    goal_seek = PythonOperator(task_id="goal_seek", python_callable=_run)
