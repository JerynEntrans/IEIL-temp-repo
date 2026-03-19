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

bucket_names=$(aws_local s3api list-buckets --query 'Buckets[].Name' --output text 2>/dev/null || true)

for bucket_name in $bucket_names; do
  if [[ -z "$bucket_name" ]]; then
    continue
  fi

  bucket_dir="$BACKUP_ROOT/$bucket_name"
  mkdir -p "$bucket_dir"
  aws_local s3 sync "s3://${bucket_name}/" "$bucket_dir/" --delete >/dev/null 2>&1 || true
done

echo "S3 backup hook completed."
