from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator

from _lambda_utils import invoke_lambda

UTC = timezone.utc

TRAINING_LAMBDA = os.getenv("TRAINING_LAMBDA_NAME", "training_lambda")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "")
SAGEMAKER_IMAGE_URI = os.getenv("SAGEMAKER_IMAGE_URI", "")
SAGEMAKER_INSTANCE_TYPE = os.getenv("SAGEMAKER_TRAINING_INSTANCE_TYPE", "ml.m5.xlarge")
SAGEMAKER_OUTPUT_S3 = os.getenv("SAGEMAKER_OUTPUT_S3_PATH", "")


def _require(params: dict, key: str) -> str:
    v = params.get(key)
    if not v or (isinstance(v, str) and not v.strip()):
        raise ValueError(f"{key} is required")
    return str(v).strip()


def build_sagemaker_training_config(**context) -> dict:
    """
    Construct the SageMaker training job config dict and push it to XCom.
    The SageMakerTrainingOperator reads this via the xcom_pull approach.
    """
    p = dict(context["params"])
    run_id = _require(p, "run_id")
    device_id = p.get("device_id", "desalter")
    model_type = _require(p, "model_type")
    image_uri = p.get("image_uri") or SAGEMAKER_IMAGE_URI
    role_arn = p.get("role_arn") or SAGEMAKER_ROLE_ARN
    instance_type = p.get("instance_type") or SAGEMAKER_INSTANCE_TYPE
    s3_output = p.get("s3_output_path") or SAGEMAKER_OUTPUT_S3

    if not image_uri:
        raise ValueError("image_uri (or SAGEMAKER_IMAGE_URI env) is required")
    if not role_arn:
        raise ValueError("role_arn (or SAGEMAKER_ROLE_ARN env) is required")
    if not s3_output:
        raise ValueError("s3_output_path (or SAGEMAKER_OUTPUT_S3_PATH env) is required")

    # Sanitise job name: SageMaker allows only alphanumeric + hyphens, max 63 chars
    job_name = f"{model_type.lower().replace('_', '-')}-{run_id[:20]}"[:63]

    import json

    hyperparameters = {
        "model-type": model_type,
        "n-estimators": str(p.get("n_estimators", 300)),
        "learning-rate": str(p.get("learning_rate", 0.05)),
        "max-depth": str(p.get("max_depth", 6)),
        "lookback": str(p.get("lookback", 10)),
    }
    if p.get("horizons_minutes"):
        hyperparameters["horizons-minutes"] = json.dumps(p["horizons_minutes"])
    if p.get("features"):
        hyperparameters["features"] = json.dumps(p["features"])
    if p.get("targets"):
        hyperparameters["targets"] = json.dumps(p["targets"])

    # DB credentials are injected via environment; pass through from Lambda env
    db_env = {
        "DB_HOST": os.getenv("DB_HOST", ""),
        "DB_PORT": os.getenv("DB_PORT", "5432"),
        "DB_NAME": os.getenv("DB_NAME", ""),
        "DB_USER": os.getenv("DB_USER", ""),
        "DB_PASSWORD": os.getenv("DB_PASSWORD", ""),
        "DB_SSLMODE": os.getenv("DB_SSLMODE", "disable"),
    }

    config = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        "RoleArn": role_arn,
        "OutputDataConfig": {"S3OutputPath": s3_output},
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 7200},
        "HyperParameters": hyperparameters,
        "Environment": db_env,
    }

    # Push both config and job name so register step can use them
    ti = context["ti"]
    ti.xcom_push(key="training_job_config", value=config)
    ti.xcom_push(key="training_job_name", value=job_name)
    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="device_id", value=device_id)
    ti.xcom_push(key="model_type", value=model_type)

    return config


def get_training_config_for_operator(**context):
    """Return the training config dict from XCom for the SageMakerTrainingOperator."""
    return context["ti"].xcom_pull(task_ids="build_config", key="training_job_config")


def run_register(**context):
    """Call the training Lambda to register the completed job in model_registry."""
    ti = context["ti"]
    p = dict(context["params"])

    training_job_name = ti.xcom_pull(task_ids="build_config", key="training_job_name")
    run_id = ti.xcom_pull(task_ids="build_config", key="run_id")
    device_id = ti.xcom_pull(task_ids="build_config", key="device_id")
    model_type = ti.xcom_pull(task_ids="build_config", key="model_type")

    payload = {
        "action": "register",
        "run_id": run_id,
        "device_id": device_id,
        "model_type": model_type,
        "training_job_name": training_job_name,
        "endpoint_name": p.get("endpoint_name") or os.getenv(
            f"SAGEMAKER_ENDPOINT_{model_type}",
            f"desalter-{model_type.lower().replace('_', '-')}",
        ),
        "image_uri": p.get("image_uri") or SAGEMAKER_IMAGE_URI,
        "role_arn": p.get("role_arn") or SAGEMAKER_ROLE_ARN,
        "instance_type": p.get("endpoint_instance_type") or os.getenv(
            "SAGEMAKER_ENDPOINT_INSTANCE_TYPE", "ml.m5.large"
        ),
        "model_version": p.get("model_version") or "",
        "data_start_ts": p.get("data_start_ts", ""),
        "data_end_ts": p.get("data_end_ts", ""),
    }

    return invoke_lambda(TRAINING_LAMBDA, payload)


with DAG(
    dag_id="training_pipeline",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["desalter", "training", "sagemaker"],
    params={
        "run_id": Param("", type="string", description="Unique run ID for this training run"),
        "device_id": Param("desalter", type="string", description="Device identifier"),
        "model_type": Param(
            "DESALTER_FORECAST",
            type="string",
            enum=["DESALTER_FORECAST", "DESALTER_GOAL_SEEK"],
            description="Which model to train via SageMaker and register in model_registry",
        ),
        "model_version": Param("", type="string", description="Optional semantic version (auto-generated if empty)"),
        "endpoint_name": Param("", type="string", description="SageMaker endpoint name (uses SAGEMAKER_ENDPOINT_<MODEL_TYPE> env if empty)"),
        "image_uri": Param("", type="string", description="ECR image URI for training container (uses SAGEMAKER_IMAGE_URI env if empty)"),
        "role_arn": Param("", type="string", description="SageMaker IAM role ARN (uses SAGEMAKER_ROLE_ARN env if empty)"),
        "instance_type": Param("", type="string", description="Training instance type (uses SAGEMAKER_TRAINING_INSTANCE_TYPE env if empty)"),
        "endpoint_instance_type": Param("ml.m5.large", type="string", description="Endpoint instance type"),
        "s3_output_path": Param("", type="string", description="S3 output path for model artifacts (uses SAGEMAKER_OUTPUT_S3_PATH env if empty)"),
        "n_estimators": Param(300, type="integer", minimum=50, description="XGBoost n_estimators"),
        "learning_rate": Param(0.05, type="number", description="XGBoost learning rate"),
        "max_depth": Param(6, type="integer", description="XGBoost max tree depth"),
        "lookback": Param(10, type="integer", description="Lookback window (FORECAST only)"),
        "horizons_minutes": Param([30, 60, 120], type="array", description="Forecast horizons in minutes (FORECAST only)"),
        "features": Param([], type="array", description="Feature column names (uses defaults if empty)"),
        "targets": Param([], type="array", description="Target column names (uses defaults if empty)"),
        "data_start_ts": Param("", type="string", description="Optional ISO timestamp"),
        "data_end_ts": Param("", type="string", description="Optional ISO timestamp"),
    },
) as dag:

    build_config = PythonOperator(
        task_id="build_config",
        python_callable=build_sagemaker_training_config,
    )

    sagemaker_train = SageMakerTrainingOperator(
        task_id="sagemaker_train",
        config="{{ ti.xcom_pull(task_ids='build_config', key='training_job_config') }}",
        wait_for_completion=True,
        aws_conn_id="aws_default",
    )

    register = PythonOperator(
        task_id="register",
        python_callable=run_register,
    )

    build_config >> sagemaker_train >> register
