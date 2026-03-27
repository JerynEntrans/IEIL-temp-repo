from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.models.param import Param
from airflow.utils.trigger_rule import TriggerRule

from _lambda_utils import invoke_lambda

UTC = timezone.utc

GOAL_SEEK_LAMBDA = os.getenv("GOAL_SEEK_LAMBDA_NAME", "unset-GOAL_SEEK_LAMBDA_NAME")
TRAINING_LAMBDA = os.getenv("TRAINING_LAMBDA_NAME", "training_lambda")


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


def _choose_train_path(**context):
    if bool(context["params"].get("train_goal_seek", False)):
        return "train_goal_seek_model"
    return "goal_seek"


def _run_goal_seek_training(**context):
    params = dict(context["params"])
    payload = {
        "action": "train_goal_seek",
        "run_id": params["run_id"],
        "validated_run_id": params.get("validated_run_id"),
        "device_id": params["device_id"],
        "manipulated_vars": params.get("train_manipulated_vars") or None,
        "targets": params.get("train_targets") or None,
        "n_estimators": params.get("train_n_estimators", 400),
        "learning_rate": params.get("train_learning_rate", 0.05),
        "max_depth": params.get("train_max_depth", 6),
        "train_split": params.get("train_split", 0.8),
        "data_start_ts": params.get("data_start_ts", ""),
        "data_end_ts": params.get("data_end_ts", ""),
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    return invoke_lambda(TRAINING_LAMBDA, payload)


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
        "train_goal_seek": Param(False, type="boolean", description="When true, train goal-seek model before running optimization"),
        "train_manipulated_vars": Param([], type="array", description="Optional manipulated vars for training lambda"),
        "train_targets": Param([], type="array", description="Optional target vars for training lambda"),
        "train_n_estimators": Param(400, type="integer", minimum=50, description="XGBoost n_estimators for goal-seek training"),
        "train_learning_rate": Param(0.05, type="number", description="XGBoost learning rate for goal-seek training"),
        "train_max_depth": Param(6, type="integer", minimum=1, description="XGBoost max depth for goal-seek training"),
        "train_split": Param(0.8, type="number", description="Train/test split ratio for goal-seek training"),
        "target_interface_level": Param(50.0, type="number", description="Target interface level"),
        "model_version": Param("", type="string", description="Optional: pin model_version. If empty, use ACTIVE model from model_registry."),
        "seed": Param(42, type="integer", description="Random seed for optimizer/search"),
        "trials": Param(200, type="integer", minimum=10, description="Number of candidate trials"),
        "data_start_ts": Param("", type="string", description="Optional ISO timestamp"),
        "data_end_ts": Param("", type="string", description="Optional ISO timestamp"),
        "run_timestamp": Param("", type="string", description="Optional ISO timestamp override"),
    },
) as dag:
    train_goal_seek_branch = BranchPythonOperator(
        task_id="train_goal_seek_branch",
        python_callable=_choose_train_path,
    )
    train_goal_seek_model = PythonOperator(
        task_id="train_goal_seek_model",
        python_callable=_run_goal_seek_training,
    )
    goal_seek = PythonOperator(
        task_id="goal_seek",
        python_callable=_run,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    train_goal_seek_branch >> train_goal_seek_model >> goal_seek
    train_goal_seek_branch >> goal_seek
