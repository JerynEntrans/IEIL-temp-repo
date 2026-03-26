#!/usr/bin/env bash
# Deploy a single Lambda to real AWS using a zip package.
#
# Usage:
#   LAMBDA_NAME=dtp-ingestion bash scripts/deploy-lambda.sh ingestion_lambda common
#   LAMBDA_NAME=dtp-forecast   bash scripts/deploy-lambda.sh forecast_lambda  ml
#
# Required env vars (loaded from .env.deploy by Makefile):
#   LAMBDA_NAME              - AWS function name to update
#   AWS_REGION               - e.g. ap-south-1
#   DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SSLMODE
#   ZOHO_TOKEN_URL, ION_IOT_API_URL
#   ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN
#   RAW_S3_BUCKET, RAW_S3_PREFIX
#   REPORTS_S3_BUCKET, REPORTS_S3_PREFIX
#   MODEL_S3_BUCKET, MODEL_S3_PREFIX
#   USE_ML_MODELS, USE_SAGEMAKER_ENDPOINT

set -euo pipefail

SERVICE_NAME="${1:?Usage: deploy-lambda.sh <service_name> <deps_variant: common|ml>}"
DEPS_VARIANT="${2:-common}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${TMPDIR:-/tmp}/ieil-lambdas-deploy"
mkdir -p "$BUILD_ROOT"

export REGION="${AWS_REGION:-ap-south-1}"
FUNCTION_NAME="${LAMBDA_NAME:-$SERVICE_NAME}"

# ── Dependency cache ────────────────────────────────────────────────────────────
if [[ "$DEPS_VARIANT" == "ml" ]]; then
  REQ_FILE="$ROOT/requirements/lambda_ml.txt"
else
  REQ_FILE="$ROOT/requirements/lambda_common.txt"
fi

DEPS_CACHE="$BUILD_ROOT/deps_${DEPS_VARIANT}"
if [[ ! -f "$DEPS_CACHE/.ready" ]]; then
  echo "==> Installing $DEPS_VARIANT deps into $DEPS_CACHE..."
  rm -rf "$DEPS_CACHE"
  mkdir -p "$DEPS_CACHE"
  PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --no-cache-dir -q \
    --ignore-installed -r "$REQ_FILE" -t "$DEPS_CACHE"
  touch "$DEPS_CACHE/.ready"
else
  echo "==> Reusing cached $DEPS_VARIANT deps."
fi

# ── Package ─────────────────────────────────────────────────────────────────────
echo "==> Packaging $SERVICE_NAME..."
WORKDIR="$BUILD_ROOT/$SERVICE_NAME"
ZIP_PATH="$BUILD_ROOT/${SERVICE_NAME}.zip"

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cp -R "$DEPS_CACHE"/. "$WORKDIR/"
rm -f "$WORKDIR/.ready"
cp -R "$ROOT/shared"                     "$WORKDIR/shared"
cp -R "$ROOT/services/${SERVICE_NAME}/src" "$WORKDIR/src"

(cd "$WORKDIR" && rm -f "$ZIP_PATH" && zip -qr "$ZIP_PATH" .)

ZIP_SIZE=$(du -sh "$ZIP_PATH" | cut -f1)
ZIP_BYTES=$(stat -c%s "$ZIP_PATH")
echo "==> Zip: $ZIP_PATH ($ZIP_SIZE)"

# Lambda zip package limits:
# - direct upload API payload is ~70 MB
# - S3-based zip package must be <= 250 MB
DIRECT_UPLOAD_LIMIT=$((70 * 1024 * 1024))
S3_ZIP_LIMIT=$((250 * 1024 * 1024))
if (( ZIP_BYTES > S3_ZIP_LIMIT )); then
  echo "ERROR: Package is too large for Lambda zip deployment (${ZIP_SIZE})."
  echo "       Max supported zip size is 250 MB (via S3)."
  echo "       Reduce dependencies or migrate this function to container image deployment."
  exit 2
fi

# ── Deploy code ─────────────────────────────────────────────────────────────────
echo "==> Updating code for function: $FUNCTION_NAME ..."
if (( ZIP_BYTES <= DIRECT_UPLOAD_LIMIT )); then
  aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --zip-file "fileb://$ZIP_PATH" \
    --region "$REGION" \
    --output text --query 'FunctionArn'
else
  ARTIFACT_BUCKET="${LAMBDA_ARTIFACTS_BUCKET:-${RAW_S3_BUCKET:-}}"
  ARTIFACT_PREFIX="${LAMBDA_ARTIFACTS_PREFIX:-lambda-artifacts}"
  if [[ -z "$ARTIFACT_BUCKET" ]]; then
    echo "ERROR: Large package requires S3-based deployment but no artifact bucket is configured."
    echo "       Set LAMBDA_ARTIFACTS_BUCKET (or RAW_S3_BUCKET) and retry."
    exit 2
  fi

  ARTIFACT_KEY="$ARTIFACT_PREFIX/$FUNCTION_NAME/${SERVICE_NAME}-$(date +%Y%m%dT%H%M%S).zip"
  echo "==> Package exceeds direct upload limit; uploading artifact to s3://$ARTIFACT_BUCKET/$ARTIFACT_KEY"
  aws s3 cp "$ZIP_PATH" "s3://$ARTIFACT_BUCKET/$ARTIFACT_KEY" --region "$REGION"

  aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --s3-bucket "$ARTIFACT_BUCKET" \
    --s3-key "$ARTIFACT_KEY" \
    --region "$REGION" \
    --output text --query 'FunctionArn'
fi

echo "==> Waiting for code update to complete..."
aws lambda wait function-updated \
  --function-name "$FUNCTION_NAME" \
  --region "$REGION"

# ── Update configuration / env vars ─────────────────────────────────────────────
echo "==> Updating environment variables..."
ENV_JSON=$(python3 - <<PY
import json, os
v = {
  "DB_HOST":                 os.environ["DB_HOST"],
  "DB_PORT":                 os.environ.get("DB_PORT", "5432"),
  "DB_NAME":                 os.environ["DB_NAME"],
  "DB_USER":                 os.environ["DB_USER"],
  "DB_PASSWORD":             os.environ["DB_PASSWORD"],
  "DB_SSLMODE":              os.environ.get("DB_SSLMODE", "require"),
  "ZOHO_TOKEN_URL":          os.environ.get("ZOHO_TOKEN_URL", ""),
  "ION_IOT_API_URL":         os.environ.get("ION_IOT_API_URL", ""),
  "ZOHO_CLIENT_ID":          os.environ.get("ZOHO_CLIENT_ID", ""),
  "ZOHO_CLIENT_SECRET":      os.environ.get("ZOHO_CLIENT_SECRET", ""),
  "ZOHO_REFRESH_TOKEN":      os.environ.get("ZOHO_REFRESH_TOKEN", ""),
  "RAW_S3_BUCKET":           os.environ["RAW_S3_BUCKET"],
  "RAW_S3_PREFIX":           os.environ.get("RAW_S3_PREFIX", "raw/zoho"),
  "REPORTS_S3_BUCKET":       os.environ["REPORTS_S3_BUCKET"],
  "REPORTS_S3_PREFIX":       os.environ.get("REPORTS_S3_PREFIX", "reports"),
  "MODEL_S3_BUCKET":         os.environ.get("MODEL_S3_BUCKET", os.environ["RAW_S3_BUCKET"]),
  "MODEL_S3_PREFIX":         os.environ.get("MODEL_S3_PREFIX", "models"),
  "USE_ML_MODELS":           os.environ.get("USE_ML_MODELS", "false"),
  "USE_SAGEMAKER_ENDPOINT":  os.environ.get("USE_SAGEMAKER_ENDPOINT", "false"),
  "OFFLINE_JSON_TESTING":    "false",
}
print(json.dumps({"Variables": v}))
PY
)

aws lambda update-function-configuration \
  --function-name "$FUNCTION_NAME" \
  --region "$REGION" \
  --handler src.handler.handler \
  --timeout 900 \
  --memory-size 2048 \
  --environment "$ENV_JSON" \
  --output text --query 'FunctionArn'

echo "==> $FUNCTION_NAME deployed successfully."
