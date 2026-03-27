#!/usr/bin/env bash
set -euo pipefail

# Build and push a custom SageMaker training image to ECR.
#
# Usage:
#   scripts/build_push_sagemaker_image.sh \
#     --region ap-south-1 \
#     --repo ieil-sagemaker-training \
#     --tag v1 \
#     --dockerfile docker/Dockerfile.sagemaker
#
# Optional:
#   --profile <aws_profile>
#   --context <build_context>          (default: repo root)
#   --local-image <local_name_prefix> (default: same as --repo)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AWS_REGION="ap-south-1"
REPO_NAME="ieil-sagemaker-training"
IMAGE_TAG="v1"
DOCKERFILE_PATH="docker/Dockerfile.sagemaker"
BUILD_CONTEXT=""
LOCAL_IMAGE_NAME=""
AWS_PROFILE=""

usage() {
  cat <<EOF
Build and push a SageMaker image to ECR.

Options:
  --region <region>            AWS region (default: ${AWS_REGION})
  --repo <name>                ECR repository name (default: ${REPO_NAME})
  --tag <tag>                  Image tag (default: ${IMAGE_TAG})
  --dockerfile <path>          Dockerfile path, relative to repo root
                               (default: ${DOCKERFILE_PATH})
  --context <path>             Build context, relative to repo root
                               (default: repo root)
  --local-image <name>         Local image name prefix (default: same as --repo)
  --profile <aws_profile>      AWS profile to use
  -h, --help                   Show this help

Example:
  scripts/build_push_sagemaker_image.sh --region ap-south-1 --repo ieil-sagemaker-training --tag v1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      AWS_REGION="$2"
      shift 2
      ;;
    --repo)
      REPO_NAME="$2"
      shift 2
      ;;
    --tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --dockerfile)
      DOCKERFILE_PATH="$2"
      shift 2
      ;;
    --context)
      BUILD_CONTEXT="$2"
      shift 2
      ;;
    --local-image)
      LOCAL_IMAGE_NAME="$2"
      shift 2
      ;;
    --profile)
      AWS_PROFILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$BUILD_CONTEXT" ]]; then
  BUILD_CONTEXT="$ROOT_DIR"
else
  BUILD_CONTEXT="$ROOT_DIR/$BUILD_CONTEXT"
fi

if [[ -z "$LOCAL_IMAGE_NAME" ]]; then
  LOCAL_IMAGE_NAME="$REPO_NAME"
fi

DOCKERFILE_ABS="$ROOT_DIR/$DOCKERFILE_PATH"
if [[ ! -f "$DOCKERFILE_ABS" ]]; then
  echo "Dockerfile not found: $DOCKERFILE_ABS" >&2
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI not found in PATH" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found in PATH" >&2
  exit 1
fi

AWS_ARGS=(--region "$AWS_REGION")
if [[ -n "$AWS_PROFILE" ]]; then
  AWS_ARGS+=(--profile "$AWS_PROFILE")
fi

echo "Resolving AWS account ID..."
ACCOUNT_ID="$(aws "${AWS_ARGS[@]}" sts get-caller-identity --query Account --output text)"

ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
LOCAL_IMAGE_REF="${LOCAL_IMAGE_NAME}:${IMAGE_TAG}"
REMOTE_IMAGE_REF="${ECR_REGISTRY}/${REPO_NAME}:${IMAGE_TAG}"

echo "Ensuring ECR repository exists: ${REPO_NAME}"
aws "${AWS_ARGS[@]}" ecr describe-repositories --repository-names "$REPO_NAME" >/dev/null 2>&1 \
  || aws "${AWS_ARGS[@]}" ecr create-repository --repository-name "$REPO_NAME" >/dev/null

echo "Logging into ECR registry: ${ECR_REGISTRY}"
aws "${AWS_ARGS[@]}" ecr get-login-password \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

echo "Building local image: ${LOCAL_IMAGE_REF}"
docker build -f "$DOCKERFILE_ABS" -t "$LOCAL_IMAGE_REF" "$BUILD_CONTEXT"

echo "Tagging image for ECR: ${REMOTE_IMAGE_REF}"
docker tag "$LOCAL_IMAGE_REF" "$REMOTE_IMAGE_REF"

echo "Pushing image: ${REMOTE_IMAGE_REF}"
docker push "$REMOTE_IMAGE_REF"

echo
echo "Done. Set this in your runtime config:"
echo "SAGEMAKER_IMAGE_URI=${REMOTE_IMAGE_REF}"
