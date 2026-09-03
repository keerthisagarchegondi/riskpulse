#!/usr/bin/env bash
set -euo pipefail

ENVIRONMENT="${1:-staging}"
IMAGE_TAG="${2:-${IMAGE_TAG:-latest}}"
DRY_RUN="${DRY_RUN:-false}"
DEPLOYMENT_BACKEND="${DEPLOYMENT_BACKEND:-local}"

case "${ENVIRONMENT}" in
  dev|staging|production) ;;
  *)
    echo "Unsupported environment: ${ENVIRONMENT}" >&2
    exit 2
    ;;
esac

AWS_REGION="${AWS_REGION:-us-east-1}"
ECS_CLUSTER="${ECS_CLUSTER:-}"
ECS_SERVICE_PREFIX="${ECS_SERVICE_PREFIX:-riskpulse-${ENVIRONMENT}}"
ECR_REGISTRY="${ECR_REGISTRY:-}"
SERVICES="${SERVICES:-api worker streamlit airflow}"
WAIT_FOR_STABILITY="${WAIT_FOR_STABILITY:-true}"

if [[ "${DEPLOYMENT_BACKEND}" == "local" ]]; then
  require_tool() {
    local tool="$1"
    if ! command -v "${tool}" >/dev/null 2>&1; then
      echo "Required tool not found: ${tool}" >&2
      exit 2
    fi
  }

  require_tool docker
  compose_file="docker-compose.yml"
  if [[ "${ENVIRONMENT}" == "production" ]]; then
    compose_file="docker-compose.prod.yml"
  fi

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY_RUN: would run docker compose -f ${compose_file} up -d --build"
    exit 0
  fi

  docker compose -f "${compose_file}" up -d --build
  docker compose -f "${compose_file}" ps
  echo "Local deployment completed for ${ENVIRONMENT} using ${compose_file}"
  exit 0
fi

if [[ -z "${ECS_CLUSTER}" ]]; then
  echo "ECS_CLUSTER is required" >&2
  exit 2
fi

if [[ -z "${ECR_REGISTRY}" ]]; then
  echo "ECR_REGISTRY is required" >&2
  exit 2
fi

require_tool() {
  local tool="$1"
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Required tool not found: ${tool}" >&2
    exit 2
  fi
}

require_tool aws
require_tool jq

register_task_definition() {
  local service="$1"
  local service_name="${ECS_SERVICE_PREFIX}-${service}"
  local image_uri="${ECR_REGISTRY}/riskpulse/${service}:${IMAGE_TAG}"

  echo "Preparing ${service_name} with image ${image_uri}"

  local current_task_definition
  current_task_definition="$(aws ecs describe-services \
    --region "${AWS_REGION}" \
    --cluster "${ECS_CLUSTER}" \
    --services "${service_name}" \
    --query 'services[0].taskDefinition' \
    --output text)"

  if [[ "${current_task_definition}" == "None" || -z "${current_task_definition}" ]]; then
    echo "Unable to resolve current task definition for ${service_name}" >&2
    exit 1
  fi

  local task_json
  task_json="$(aws ecs describe-task-definition \
    --region "${AWS_REGION}" \
    --task-definition "${current_task_definition}" \
    --query 'taskDefinition')"

  local next_task_json
  next_task_json="$(jq --arg service "${service}" --arg image "${image_uri}" '
    del(
      .taskDefinitionArn,
      .revision,
      .status,
      .requiresAttributes,
      .compatibilities,
      .registeredAt,
      .registeredBy
    )
    | .containerDefinitions |= (
        if length == 1 then
          .[0].image = $image | .
        else
          map(if .name == $service then .image = $image else . end)
        end
      )
  ' <<<"${task_json}")"

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY_RUN: would register task definition for ${service_name}"
    echo "DRY_RUN: would update ${service_name} to ${image_uri}"
    return 0
  fi

  local next_task_definition
  next_task_definition="$(aws ecs register-task-definition \
    --region "${AWS_REGION}" \
    --cli-input-json "${next_task_json}" \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)"

  aws ecs update-service \
    --region "${AWS_REGION}" \
    --cluster "${ECS_CLUSTER}" \
    --service "${service_name}" \
    --task-definition "${next_task_definition}" \
    --force-new-deployment >/dev/null

  echo "Updated ${service_name} to ${next_task_definition}"
}

for service in ${SERVICES}; do
  register_task_definition "${service}"
done

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "DRY_RUN: deployment plan completed for ${ENVIRONMENT}"
  exit 0
fi

if [[ "${WAIT_FOR_STABILITY}" == "true" ]]; then
  for service in ${SERVICES}; do
    service_name="${ECS_SERVICE_PREFIX}-${service}"
    echo "Waiting for ${service_name} to stabilize"
    aws ecs wait services-stable \
      --region "${AWS_REGION}" \
      --cluster "${ECS_CLUSTER}" \
      --services "${service_name}"
  done
fi

echo "Deployment completed for ${ENVIRONMENT} with image tag ${IMAGE_TAG}"
