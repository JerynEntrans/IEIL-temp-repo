from __future__ import annotations

import json
import logging
import os
from typing import Optional

import boto3
import numpy as np

logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
_sm_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)


def invoke_sagemaker_endpoint(
    endpoint_name: str,
    X: np.ndarray,
    *,
    content_type: str = "text/csv",
    accept: str = "application/json",
) -> np.ndarray:
    """
    Call a SageMaker XGBoost endpoint with a 2D numpy array.

    Serialises rows as CSV (no header). Returns predictions as a numpy array
    shaped (n_samples, n_outputs) or (n_outputs,) for single-row input.
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)

    body = "\n".join(",".join(str(float(v)) for v in row) for row in X)
    logger.info(f"Invoking SageMaker endpoint '{endpoint_name}' with {X.shape[0]} row(s)")

    response = _sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Accept=accept,
        Body=body.encode("utf-8"),
    )
    raw = response["Body"].read().decode("utf-8")
    predictions = json.loads(raw)

    # SageMaker XGBoost built-in container returns {"predictions": [...]}
    if isinstance(predictions, dict) and "predictions" in predictions:
        predictions = predictions["predictions"]

    return np.array(predictions, dtype=float)


def get_sagemaker_client():
    return boto3.client("sagemaker", region_name=AWS_REGION)


def create_or_update_endpoint(
    *,
    sm_client,
    endpoint_name: str,
    model_name: str,
    execution_role_arn: str,
    model_s3_uri: str,
    image_uri: str,
    instance_type: str = "ml.m5.large",
):
    """
    Create (or update) a SageMaker real-time endpoint from a training job's model artifact.

    Steps:
      1. Create SageMaker Model object
      2. Create endpoint config
      3. Create or update the endpoint
    """
    config_name = f"{endpoint_name}-config"

    # 1. Create model
    logger.info(f"Creating SageMaker Model: {model_name}")
    try:
        sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": image_uri,
                "ModelDataUrl": model_s3_uri,
                "Environment": {},
            },
            ExecutionRoleArn=execution_role_arn,
        )
    except sm_client.exceptions.ClientError as e:
        if "already exists" in str(e).lower():
            logger.info(f"SageMaker Model '{model_name}' already exists, skipping creation")
        else:
            raise

    # 2. Create endpoint config
    logger.info(f"Creating endpoint config: {config_name}")
    try:
        sm_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": 1,
                }
            ],
        )
    except sm_client.exceptions.ClientError as e:
        if "already exists" in str(e).lower():
            logger.info(f"Endpoint config '{config_name}' already exists, skipping creation")
        else:
            raise

    # 3. Create or update endpoint
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        logger.info(f"Updating existing endpoint: {endpoint_name}")
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
    except sm_client.exceptions.ClientError:
        logger.info(f"Creating new endpoint: {endpoint_name}")
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )

    logger.info(f"Endpoint '{endpoint_name}' is being deployed (async – may take a few minutes)")
