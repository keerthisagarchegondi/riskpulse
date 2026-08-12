#!/usr/bin/env bash
set -euo pipefail

ENVIRONMENT="${1:-production}"
STATE_DIR="${2:-deployment-state}"
DRY_RUN="${DRY_RUN:-false}"

case "${ENVIRONMENT}" in
  staging|production) ;;
  *)
    echo "Unsupported environment: ${ENVIRONMENT}" >&2
    exit 2
    ;;
esac

AWS_REGION="${AWS_REGION:-us-east-1}"
ECS_CLUSTER="${ECS_CLUSTER:-}"
ECS_SERVICE_PREFIX="${ECS_SERVICE_PREFIX:-riskpulse-${ENVIRONMENT}}"
SERVICES="${SERVICES:-api worker streamlit airflow}"
WAIT_FOR_STABILITY="${WAIT_FOR_STABILITY:-true}"

if [[ -z "${ECS_CLUSTER}" ]]; then
  echo "ECS_CLUSTER is required" >&2
  exit 2
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "Required tool not found: aws" >&2
  exit 2
fi

rollback_service() {
  local service="$1"
  local service_name="${ECS_SERVICE_PREFIX}-${service}"
  local taskdef_file="${STATE_DIR}/${service}.taskdef"

  if [[ ! -s "${taskdef_file}" ]]; then
    echo "Missing rollback task definition state: ${taskdef_file}" >&2
    exit 1
  fi

  local previous_task_definition
  previous_task_definition="$(tr -d '\r\n' < "${taskdef_file}")"

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY_RUN: would roll back ${service_name} to ${previous_task_definition}"
    return 0
  fi

  aws ecs update-service \
    --region "${AWS_REGION}" \
    --cluster "${ECS_CLUSTER}" \
    --service "${service_name}" \
    --task-definition "${previous_task_definition}" \
    --force-new-deployment >/dev/null

  echo "Rollback started for ${service_name} to ${previous_task_definition}"
}

for service in ${SERVICES}; do
  rollback_service "${service}"
done

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "DRY_RUN: rollback plan completed for ${ENVIRONMENT}"
  exit 0
fi

if [[ "${WAIT_FOR_STABILITY}" == "true" ]]; then
  for service in ${SERVICES}; do
    service_name="${ECS_SERVICE_PREFIX}-${service}"
    echo "Waiting for rollback stability: ${service_name}"
    aws ecs wait services-stable \
      --region "${AWS_REGION}" \
      --cluster "${ECS_CLUSTER}" \
      --services "${service_name}"
  done
fi

echo "Rollback completed for ${ENVIRONMENT}"
