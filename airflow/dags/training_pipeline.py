from __future__ import annotations

import os
import secrets
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
    run_id = f"trn-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"
    device_id = "desalter"
    model_type = _require(p, "model_type")
    image_uri = SAGEMAKER_IMAGE_URI
    role_arn = SAGEMAKER_ROLE_ARN
    instance_type = SAGEMAKER_INSTANCE_TYPE
    s3_output = SAGEMAKER_OUTPUT_S3

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
        "n-estimators": "300",
        "learning-rate": "0.05",
        "max-depth": "6",
        "lookback": "10",
    }
    if p.get("horizons_minutes"):
        hyperparameters["horizons-minutes"] = json.dumps(p["horizons_minutes"])

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
        "endpoint_name": os.getenv(
            f"SAGEMAKER_ENDPOINT_{model_type}",
            f"desalter-{model_type.lower().replace('_', '-')}",
        ),
        "image_uri": SAGEMAKER_IMAGE_URI,
        "role_arn": SAGEMAKER_ROLE_ARN,
        "instance_type": os.getenv("SAGEMAKER_ENDPOINT_INSTANCE_TYPE", "ml.m5.large"),
        "model_version": "",
        "data_start_ts": "",
        "data_end_ts": "",
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
        "model_type": Param(
            "DESALTER_FORECAST",
            type="string",
            enum=["DESALTER_FORECAST", "DESALTER_GOAL_SEEK"],
            description="Which model to train via SageMaker and register in model_registry",
        ),
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
