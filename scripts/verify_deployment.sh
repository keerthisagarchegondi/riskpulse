#!/usr/bin/env bash
set -euo pipefail

ENVIRONMENT="${1:-production}"
BASE_URL="${PRODUCTION_BASE_URL:-${RISKPULSE_BASE_URL:-}}"
STREAMLIT_URL="${PRODUCTION_STREAMLIT_URL:-${RISKPULSE_STREAMLIT_URL:-}}"
AIRFLOW_URL="${PRODUCTION_AIRFLOW_URL:-${RISKPULSE_AIRFLOW_URL:-}}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ECS_CLUSTER="${ECS_CLUSTER:-${PRODUCTION_ECS_CLUSTER:-}}"
ECS_SERVICE_PREFIX="${ECS_SERVICE_PREFIX:-${PRODUCTION_ECS_SERVICE_PREFIX:-riskpulse-prod}}"
SERVICES="${SERVICES:-api worker streamlit airflow}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_TERRAFORM="${RUN_TERRAFORM:-false}"
RUN_DOCKER="${RUN_DOCKER:-true}"
RUN_AWS_CHECKS="${RUN_AWS_CHECKS:-false}"
RUN_TESTS="${RUN_TESTS:-true}"
RUN_SMOKE="${RUN_SMOKE:-true}"
ALLOW_SYNTHETIC_PROD_WRITES="${ALLOW_SYNTHETIC_PROD_WRITES:-false}"
REPORT_DIR="${REPORT_DIR:-deployment-reports}"

case "${ENVIRONMENT}" in
  dev|staging|production) ;;
  *)
    echo "Unsupported environment: ${ENVIRONMENT}" >&2
    exit 2
    ;;
esac

mkdir -p "${REPORT_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

warn() {
  printf '[%s] WARNING: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

fail() {
  printf '[%s] ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -e "${path}" ]] || fail "Required file missing: ${path}"
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

log "RiskPulse deployment verification started for ${ENVIRONMENT}"

log "Checking repository artifacts"
require_file docker-compose.yml
require_file docker-compose.prod.yml
require_file infrastructure/docker/Dockerfile.api
require_file infrastructure/docker/Dockerfile.worker
require_file infrastructure/docker/Dockerfile.streamlit
require_file infrastructure/docker/Dockerfile.airflow
require_file infrastructure/terraform/modules/iam/main.tf
require_file infrastructure/terraform/modules/cloudwatch/main.tf
require_file infrastructure/aws/cloudwatch/dashboards/platform_dashboard.json
require_file infrastructure/aws/cloudwatch/alarms/high_fraud_rate.json
require_file infrastructure/aws/cloudwatch/alarms/pipeline_failure.json
require_file dashboards/streamlit/app.py
require_file dashboards/streamlit/pages/model_performance.py
require_file dashboards/streamlit/pages/alert_management.py
require_file airflow/dags/ingestion_dag.py
require_file airflow/dags/fraud_detection_dag.py
require_file airflow/dags/snowflake_load.py
require_file database/snowflake/schemas/raw_schema.sql
require_file database/snowflake/schemas/analytics_schema.sql
require_file database/snowflake/procedures/load_transactions.sql
require_file docs/production_readiness_checklist.txt

if find docs -maxdepth 2 -type f \( -name 'production_readiness_checklist.md' -o -name 'runbook.md' -o -name 'onboarding.md' -o -name 'deployment_guide.md' \) | grep -q .; then
  fail "Markdown production docs were created despite the no-.md constraint"
fi

if [[ "${RUN_DOCKER}" == "true" ]]; then
  if command_exists docker; then
    log "Validating Docker Compose development config"
    docker compose -f docker-compose.yml config --quiet
    log "Validating Docker Compose production config"
    docker compose -f docker-compose.prod.yml config --quiet
  else
    warn "Docker is not installed; skipping compose validation"
  fi
else
  warn "RUN_DOCKER=false; skipping Docker checks"
fi

if [[ "${RUN_TERRAFORM}" == "true" ]]; then
  command_exists terraform || fail "RUN_TERRAFORM=true but terraform is not installed"
  for module in infrastructure/terraform/modules/s3 infrastructure/terraform/modules/iam infrastructure/terraform/modules/cloudwatch; do
    log "Validating Terraform module ${module}"
    terraform -chdir="${module}" init -backend=false
    terraform -chdir="${module}" validate
  done
else
  warn "RUN_TERRAFORM=false; Terraform apply/validate skipped"
fi

if [[ "${RUN_TESTS}" == "true" ]]; then
  command_exists "${PYTHON_BIN}" || fail "PYTHON_BIN=${PYTHON_BIN} is required for production gate tests"
  log "Running production gate tests"
  "${PYTHON_BIN}" -m pytest tests/security -m security
  "${PYTHON_BIN}" -m pytest tests/data_quality -m data_quality
  "${PYTHON_BIN}" -m pytest tests/ml_validation -m ml_validation
  "${PYTHON_BIN}" -m pytest tests/performance/test_latency.py -m performance
else
  warn "RUN_TESTS=false; test gates skipped"
fi

if [[ "${RUN_AWS_CHECKS}" == "true" ]]; then
  command_exists aws || fail "RUN_AWS_CHECKS=true but aws CLI is not installed"
  [[ -n "${ECS_CLUSTER}" ]] || fail "ECS_CLUSTER or PRODUCTION_ECS_CLUSTER is required"

  log "Checking AWS caller identity"
  aws sts get-caller-identity --region "${AWS_REGION}" >/dev/null

  for service in ${SERVICES}; do
    service_name="${ECS_SERVICE_PREFIX}-${service}"
    log "Checking ECS service ${service_name}"
    aws ecs describe-services \
      --region "${AWS_REGION}" \
      --cluster "${ECS_CLUSTER}" \
      --services "${service_name}" \
      --query 'services[0].{status:status,running:runningCount,desired:desiredCount,taskDefinition:taskDefinition}' \
      --output json
  done

  log "Checking CloudWatch alarms"
  aws cloudwatch describe-alarms \
    --region "${AWS_REGION}" \
    --alarm-name-prefix "riskpulse" \
    --query 'MetricAlarms[].{name:AlarmName,state:StateValue,enabled:ActionsEnabled}' \
    --output json

  log "Checking CloudWatch log groups"
  aws logs describe-log-groups \
    --region "${AWS_REGION}" \
    --log-group-name-prefix "/riskpulse/prod" \
    --query 'logGroups[].logGroupName' \
    --output json
else
  warn "RUN_AWS_CHECKS=false; live AWS verification skipped"
fi

if [[ "${RUN_SMOKE}" == "true" ]]; then
  [[ -n "${BASE_URL}" ]] || fail "PRODUCTION_BASE_URL or RISKPULSE_BASE_URL is required for smoke tests"
  smoke_args=(scripts/smoke_test.py --base-url "${BASE_URL}" --retries 10 --retry-delay 15 --timeout 10 --report-file "${REPORT_DIR}/smoke-${ENVIRONMENT}.json")
  if [[ -n "${STREAMLIT_URL}" ]]; then
    smoke_args+=(--streamlit-url "${STREAMLIT_URL}")
  fi
  if [[ -n "${AIRFLOW_URL}" ]]; then
    smoke_args+=(--airflow-url "${AIRFLOW_URL}")
  fi
  if [[ "${ALLOW_SYNTHETIC_PROD_WRITES}" == "true" ]]; then
    smoke_args+=(--submit-test-transaction)
  fi
  log "Running smoke tests"
  "${PYTHON_BIN}" "${smoke_args[@]}"
else
  warn "RUN_SMOKE=false; endpoint smoke tests skipped"
fi

log "RiskPulse deployment verification completed for ${ENVIRONMENT}"
