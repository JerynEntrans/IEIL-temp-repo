from __future__ import annotations

import json
import os
import boto3

from shared.utils.logging import set_logging
logger = set_logging(__name__)

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
_lambda = boto3.client("lambda", region_name=AWS_REGION)


def invoke_lambda(function_name: str, payload: dict) -> dict:
    resp = _lambda.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload).encode("utf-8"),
    )
    payload_bytes = resp["Payload"].read()
    try:
        out = json.loads(payload_bytes.decode("utf-8")) if payload_bytes else {}
    except Exception:
        out = {"raw_payload": payload_bytes.decode("utf-8", errors="replace")}
    if resp.get("FunctionError"):
        raise RuntimeError(f"Lambda {function_name} error: {out}")
    logger.info(json.dumps(out, indent=2))
    return out
