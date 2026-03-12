import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime

import boto3
import xgboost as xgb

from shared.schema.db import ModelSpec

s3_client = boto3.client("s3")


@dataclass(frozen=True)
class S3WriteResult:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


class S3RawStore:
    def __init__(self, bucket: str, prefix: str):
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def put_json(self, *, device_id: str, run_id: str, window_end_utc: datetime, payload: dict) -> S3WriteResult:
        dt = window_end_utc.strftime("%Y-%m-%d")
        ts = window_end_utc.strftime("%Y%m%dT%H%M%SZ")
        key = (
            f"{self.prefix}/device_id={device_id}/dt={dt}/run_id={run_id}/"
            f"window_end={ts}/payload.json"
        )
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        return S3WriteResult(bucket=self.bucket, key=key)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3 uri: {uri}")
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3 uri: {uri}")
    return bucket, key


def get_json(bucket: str, key: str) -> dict:
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read()
    return json.loads(body.decode("utf-8"))


def list_keys(bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            keys.append(obj["Key"])
    return keys


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def download_s3_to_tmp(s3_uri: str, *, tmp_path: str) -> tuple[str, str]:
    bucket, key = parse_s3_uri(s3_uri)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    sha = sha256_bytes(body)
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(body)
    return tmp_path, sha


def load_booster_from_s3(spec: ModelSpec) -> xgb.Booster:
    cache_path = f"/tmp/xgb_model_{spec.id}.json"
    if os.path.exists(cache_path):
        booster = xgb.Booster()
        booster.load_model(cache_path)
        return booster

    bucket, key = parse_s3_uri(spec.s3_uri)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    got_sha = sha256_bytes(body)
    if spec.artifact_sha256 and got_sha != spec.artifact_sha256:
        raise ValueError(
            f"Model sha256 mismatch for {spec.s3_uri}: expected {spec.artifact_sha256}, got {got_sha}"
        )

    with open(cache_path, "wb") as f:
        f.write(body)

    booster = xgb.Booster()
    booster.load_model(cache_path)
    return booster
