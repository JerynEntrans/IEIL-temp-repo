#!/usr/bin/env bash
# Deploy code+env for an existing Image-package Lambda.
# Usage: bash scripts/deploy-lambda-image.sh <function_name> <image_uri>

set -euo pipefail

FUNCTION_NAME="${1:?Usage: deploy-lambda-image.sh <function_name> <image_uri>}"
IMAGE_URI="${2:?Usage: deploy-lambda-image.sh <function_name> <image_uri>}"
REGION="${AWS_REGION:-ap-south-1}"

PACKAGE_TYPE=$(aws lambda get-function-configuration \
  --function-name "$FUNCTION_NAME" \
  --region "$REGION" \
  --query 'PackageType' --output text)

if [[ "$PACKAGE_TYPE" != "Image" ]]; then
  echo "ERROR: $FUNCTION_NAME is package type '$PACKAGE_TYPE', not 'Image'."
  echo "       Existing Zip functions cannot be switched in-place."
  echo "       Create a new Image-based Lambda function, then point env/function name to it."
  exit 2
fi

echo "==> Updating image for function: $FUNCTION_NAME"
aws lambda update-function-code \
  --function-name "$FUNCTION_NAME" \
  --image-uri "$IMAGE_URI" \
  --region "$REGION" \
  --output text --query 'FunctionArn'

echo "==> Waiting for image update to complete..."
aws lambda wait function-updated \
  --function-name "$FUNCTION_NAME" \
  --region "$REGION"

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
  --timeout 900 \
  --memory-size 2048 \
  --environment "$ENV_JSON" \
  --output text --query 'FunctionArn'

echo "==> $FUNCTION_NAME image deploy complete."
