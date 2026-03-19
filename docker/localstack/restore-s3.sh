#!/usr/bin/env bash
set -euo pipefail

export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-test}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-test}
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-ap-south-1}
export AWS_PAGER=""

BACKUP_ROOT=/var/lib/localstack/s3-backup
mkdir -p "$BACKUP_ROOT"

aws_local() {
  aws --endpoint-url=http://localhost:4566 "$@"
}

for bucket_dir in "$BACKUP_ROOT"/*; do
  if [[ ! -d "$bucket_dir" ]]; then
    continue
  fi

  bucket_name=$(basename "$bucket_dir")
  if [[ -z "$bucket_name" ]]; then
    continue
  fi

  aws_local s3 mb "s3://${bucket_name}" >/dev/null 2>&1 || true
  aws_local s3 sync "$bucket_dir/" "s3://${bucket_name}/" --exact-timestamps >/dev/null 2>&1 || true
done

echo "S3 restore hook completed."
