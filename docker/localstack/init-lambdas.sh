#!/usr/bin/env bash
set -euo pipefail

export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=ap-south-1
export AWS_PAGER=""

ROOT=/repo
BUILD_ROOT=/tmp/ieil-lambdas
mkdir -p "$BUILD_ROOT"

aws_local() {
  aws --endpoint-url=http://localhost:4566 "$@"
}

aws_local s3 mb s3://ieil-raw || true
aws_local s3 mb s3://ieil-reports || true

install_common_deps() {
  local dest="$1"
  python -m pip install --no-cache-dir -q \
    boto3>=1.34.0 \
    requests>=2.31.0 \
    python-dateutil>=2.8.2 \
    python-dotenv>=1.0.0 \
    'psycopg[binary]>=3.1.18,<4.0.0' \
    numpy>=1.24.0,<2.0.0 \
    xgboost>=2.0.0,<3.0.0 \
    -t "$dest"
}

package_lambda() {
  local service_name="$1"
  local src_dir="$2"
  local zip_path="$BUILD_ROOT/${service_name}.zip"
  local workdir="$BUILD_ROOT/${service_name}"

  rm -rf "$workdir"
  mkdir -p "$workdir"
  install_common_deps "$workdir"
  cp -R "$ROOT/shared" "$workdir/shared"
  cp -R "$src_dir" "$workdir/src"

  (
    cd "$workdir"
    zip -qr "$zip_path" .
  )

  local env_json
  env_json=$(cat <<JSON
{"Variables":{
  "AWS_DEFAULT_REGION":"ap-south-1",
  "AWS_ACCESS_KEY_ID":"test",
  "AWS_SECRET_ACCESS_KEY":"test",
  "AWS_ENDPOINT_URL":"http://localstack:4566",
  "AWS_ENDPOINT_URL_S3":"http://localstack:4566",
  "DB_HOST":"postgres",
  "DB_PORT":"5432",
  "DB_NAME":"ieil",
  "DB_USER":"ieil",
  "DB_PASSWORD":"ieil",
  "DB_SSLMODE":"disable",
  "RAW_S3_BUCKET":"ieil-raw",
  "RAW_S3_PREFIX":"raw/zoho",
  "REPORTS_S3_BUCKET":"ieil-reports",
  "REPORTS_S3_PREFIX":"reports"
}}
JSON
)

  aws_local lambda delete-function --function-name "$service_name" >/dev/null 2>&1 || true
  aws_local lambda create-function \
    --function-name "$service_name" \
    --runtime python3.11 \
    --role arn:aws:iam::000000000000:role/lambda-role \
    --handler src.handler.handler \
    --timeout 900 \
    --memory-size 2048 \
    --zip-file "fileb://${zip_path}" \
    --environment "$env_json" >/dev/null
}

package_lambda ingestion_lambda "$ROOT/services/ingestion_lambda/src"
package_lambda validation_lambda "$ROOT/services/validation_lambda/src"
package_lambda forecast_lambda "$ROOT/services/forecast_lambda/src"
package_lambda goal_seek_lambda "$ROOT/services/goal_seek_lambda/src"
package_lambda report_lambda "$ROOT/services/report_lambda/src"

echo "Local IEIL lambdas created in LocalStack."
