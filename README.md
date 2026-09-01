# RiskPulse Setup Guide

RiskPulse is a fraud analytics and risk intelligence platform for real-time transaction ingestion, fraud scoring, alerting, dashboards, monitoring, and production deployment.

This README focuses on getting the repository set up locally, validating the services, and preparing the required external configuration for CI/CD and production.

## Prerequisites

Install these before starting:

- Python 3.11
- Git
- Docker Desktop with Docker Compose
- Make, optional on Windows but useful for common commands
- AWS CLI, required for deployment and CloudWatch/ECR/ECS checks
- Terraform, required for infrastructure provisioning
- Snowflake account access, required for analytical warehouse loading
- Power BI Desktop or Power BI Service access, required for executive dashboard work

## Repository Setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/keerthisagarchegondi/riskpulse.git
cd riskpulse
```

Create and activate a Python virtual environment.

Linux or macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project with development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

If `python` is not available on PATH, pass a specific interpreter to Make commands:

```bash
make test PYTHON="/absolute/path/to/python"
```

## Environment Configuration

Create a local environment file from the example:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Update `.env` before running real services. At minimum, configure:

- `RISKPULSE_ENV`
- `RISKPULSE_DB_HOST`
- `RISKPULSE_DB_PORT`
- `RISKPULSE_DB_NAME`
- `RISKPULSE_DB_USER`
- `RISKPULSE_DB_PASSWORD`
- `RISKPULSE_REDIS_HOST`
- `RISKPULSE_REDIS_PORT`
- `RISKPULSE_KAFKA_BOOTSTRAP_SERVERS`
- `RISKPULSE_JWT_SECRET`
- `RISKPULSE_API_KEY`
- `RISKPULSE_ENCRYPTION_KEY`
- `RISKPULSE_DASHBOARD_BASE_URL`
- `AWS_REGION`
- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD` or private key settings
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_WAREHOUSE`
- `SNOWFLAKE_ROLE`

Do not use placeholder secrets in production.

## Local Docker Setup

Start the local platform dependencies and services:

```bash
docker compose -f docker-compose.yml up -d
```

Check container status:

```bash
docker compose -f docker-compose.yml ps
```

View logs:

```bash
docker compose -f docker-compose.yml logs -f
```

Stop services:

```bash
docker compose -f docker-compose.yml down
```

The Makefile also provides shortcuts:

```bash
make docker-up
make docker-ps
make docker-logs
make docker-down
```

## Running Services Locally

Run the FastAPI application:

```bash
make run
```

Run the Kafka worker:

```bash
make run-worker
```

Run the Streamlit dashboard:

```bash
make run-streamlit
```

Default local endpoints:

- API: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`
- Streamlit: `http://127.0.0.1:8501`
- Airflow, when enabled: `http://127.0.0.1:8080`

## Database Setup

Start PostgreSQL through Docker, then run migrations:

```bash
make db-migrate
```

Seed development data when needed:

```bash
make db-seed
```

Some shell scripts use `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD`, so keep those aligned with the `RISKPULSE_DB_*` values.

## Quality Checks

Run formatting checks:

```bash
black --check src tests scripts dashboards ml airflow
isort --check-only src tests scripts dashboards ml airflow
flake8 src tests scripts dashboards ml airflow
```

Run type checks:

```bash
mypy src
```

Run unit tests with coverage:

```bash
pytest tests/unit --cov=src --cov-report=term-missing --cov-fail-under=90
```

Run security checks:

```bash
bandit -r src scripts dashboards ml airflow -c pyproject.toml -ll
safety check --full-report
```

Run all configured checks through Make:

```bash
make check-all
```

## Integration And Validation Tests

Integration tests require Docker services to be running:

```bash
make docker-up
pytest tests/integration -m integration
```

Performance tests:

```bash
pytest tests/performance -m performance
```

Security regression tests:

```bash
pytest tests/security -m security
```

Data quality checks:

```bash
pytest tests/data_quality -m data_quality
```

ML validation checks:

```bash
pytest tests/ml_validation -m ml_validation
```

## CI/CD Setup

The GitHub Actions workflows live in `.github/workflows`:

- `ci.yml`
- `cd-staging.yml`
- `cd-production.yml`

Configure these GitHub repository variables:

- `AWS_REGION`
- `ECR_REGISTRY`
- `STAGING_ECS_CLUSTER`
- `STAGING_ECS_SERVICE_PREFIX`
- `STAGING_BASE_URL`
- `PRODUCTION_ECS_CLUSTER`
- `PRODUCTION_ECS_SERVICE_PREFIX`
- `PRODUCTION_BASE_URL`
- `PRODUCTION_STREAMLIT_URL`
- `PRODUCTION_AIRFLOW_URL`

Configure these GitHub repository secrets:

- `STAGING_AWS_ROLE_ARN`
- `PRODUCTION_AWS_ROLE_ARN`

If notification webhooks are enabled, store webhook values as secrets rather than plain variables.

Production deployments should use a protected `production` environment with required approval.

## Production Setup Checklist

Before production deployment, verify:

- AWS IAM roles and least-privilege policies are applied
- AWS Secrets Manager contains database, JWT, API key, and encryption secrets
- ECR repositories exist for API, worker, Streamlit, and Airflow images
- ECS services or equivalent runtime targets exist
- CloudWatch log groups, dashboards, metrics, and alarms are configured
- PostgreSQL is provisioned, reachable, and migrated
- Redis is provisioned and reachable
- Kafka is provisioned and reachable
- Snowflake database, warehouse, role, and stages are configured
- S3 buckets and encryption policies exist
- DNS and TLS are configured for public endpoints
- Smoke tests pass against the production base URL
- No high or critical security findings remain

## Deployment Verification

Run local smoke tests:

```bash
python scripts/smoke_test.py --base-url http://127.0.0.1:8000
```

Run production verification:

```bash
RISKPULSE_BASE_URL=https://your-api-domain.example.com \
RISKPULSE_STREAMLIT_URL=https://your-dashboard-domain.example.com \
RISKPULSE_AIRFLOW_URL=https://your-airflow-domain.example.com \
RUN_AWS_CHECKS=true \
./scripts/verify_deployment.sh production
```

Run rollback if a deployment fails:

```bash
./scripts/rollback.sh production
```

## Common Troubleshooting

If editable installation fails because `README.md` is missing, make sure this file is committed and pushed.

If Docker services fail health checks, inspect logs with:

```bash
docker compose -f docker-compose.yml logs -f
```

If CI/CD deployment fails, confirm GitHub secrets, repository variables, AWS OIDC trust, ECR repositories, and ECS service names.

If integration tests fail locally, confirm PostgreSQL, Redis, and Kafka containers are healthy before rerunning tests.


RiskPulse - Fraud Analytics and Risk Intelligence Platform

Overview
RiskPulse is a production-oriented fraud analytics platform for real-time
transaction processing, fraud detection, alert management, operational
monitoring, and executive reporting. It combines streaming ingestion, data
quality validation, enrichment, rule-based fraud detection, anomaly detection,
machine-learning scoring, alert operations, dashboards, and cloud monitoring.

Core Capabilities
- Transaction ingestion through FastAPI and Kafka.
- Schema validation, quarantine, and business-rule enforcement.
- Data cleaning, normalization, feature engineering, and aggregation.
- Geo, device, merchant, and velocity enrichment.
- Fraud rules, anomaly detection, ML risk scoring, and score ensembles.
- Alert deduplication, suppression, throttling, routing, escalation, and notifications.
- PostgreSQL operational storage.
- S3 data lake storage.
- Snowflake analytics and Power BI executive dashboards.
- Streamlit operational dashboards.
- CloudWatch logging, custom metrics, dashboards, and alarms.
- IAM roles, least-privilege policies, secrets management, JWT/API key auth, and audit logging.
- Dockerized services and GitHub Actions CI/CD.
- Unit, integration, performance, security, data quality, and ML validation tests.

Architecture

Transactions
  -> FastAPI ingestion
  -> Kafka topic txn.raw.events
  -> Validation and quarantine
  -> Transformation and normalization
  -> Enrichment
  -> Fraud scoring
  -> Alert management
  -> PostgreSQL operational tables
  -> S3 data lake
  -> Snowflake analytics
  -> Streamlit and Power BI dashboards
  -> CloudWatch logs, metrics, dashboards, alarms

Tech Stack
- Language: Python 3.11+
- API: FastAPI
- Streaming: Apache Kafka and confluent-kafka
- Processing: pandas, numpy
- ML: scikit-learn, XGBoost, LightGBM, SHAP, Isolation Forest
- Operational database: PostgreSQL
- Cache/rate limits: Redis
- Warehouse: Snowflake
- Data lake: AWS S3
- Dashboards: Streamlit and Power BI
- Orchestration: Apache Airflow
- Monitoring: AWS CloudWatch
- Security: AWS IAM, AWS Secrets Manager, JWT, API keys, CORS, rate limits
- Containers: Docker and Docker Compose
- CI/CD: GitHub Actions, ECR, ECS deployment scripts

Repository Layout
- src: application source.
- airflow: DAGs and custom operators.
- dashboards: Streamlit and Power BI assets.
- database: PostgreSQL migrations, seeds, and Snowflake SQL.
- infrastructure: Dockerfiles, AWS policies, Terraform modules.
- ml: model training, notebooks, and model artifacts.
- config: environment and business configuration.
- scripts: deployment, rollback, OpenAPI, healthcheck, and data generation tools.
- tests: unit, integration, performance, security, data quality, and ML validation.
- docs: operational, security, deployment, testing, dashboard, and API documentation.

Quick Start

Prerequisites:
- Python 3.11+
- Docker Desktop
- Git
- Make, or equivalent shell commands

Setup:
git clone <repository-url>
cd riskpulse
python -m venv .venv

Windows PowerShell:
.venv\Scripts\Activate.ps1

Linux or macOS:
source .venv/bin/activate

Install:
python -m pip install --upgrade pip
pip install -e ".[dev]"

Configure:
copy .env.example .env

Start services:
docker compose -f docker-compose.yml up -d

Run API:
make run

Run dashboard:
make run-streamlit

Verify:
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready

Common Commands
- make install-dev
- make lint
- make format
- make test
- make test-unit
- make test-integration
- make test-coverage
- make test-performance
- make security-scan
- make run
- make run-worker
- make run-streamlit
- make docker-up
- make docker-down
- make docker-build
- make docker-build-prod
- make docker-test
- make smoke-test
- make verify-deployment
- make db-migrate
- make db-seed
- make generate-data

Testing Pyramid
- Unit tests: tests/unit
- Integration tests: tests/integration
- Performance tests: tests/performance
- Security tests: tests/security
- Data quality tests: tests/data_quality
- ML validation tests: tests/ml_validation

Minimum Gates
- Unit coverage target: 90 percent.
- API auth and injection tests must pass.
- Data freshness SLA: less than 15 minutes in validation fixtures.
- Volume anomaly checks: within 3 standard deviations.
- Model holdout quality, fairness, calibration, and latency gates must pass.
- Docker compose config must validate.
- Security scans must pass.

Operational Documentation
- docs/runbook.txt
- docs/onboarding.txt
- docs/deployment_guide.txt
- docs/security_architecture.txt
- docs/testing_strategy.txt
- docs/production_readiness_checklist.txt
- docs/powerbi_deployment.txt
- docs/user_guides/api_guide.txt
- docs/user_guides/streamlit_guide.txt
- docs/user_guides/powerbi_guide.txt

Deployment Summary
1. Pull request to develop runs CI.
2. Merge to develop deploys staging.
3. Validate staging with smoke, data quality, ML validation, and dashboard checks.
4. Pull request to main runs CI.
5. Production deployment requires protected environment approval.
6. Deployment captures rollback state.
7. Production smoke tests and five-minute monitoring run.
8. Rollback is automatic or manual through scripts/rollback.sh.

Security Summary
- API supports API keys and JWT bearer tokens.
- IAM roles are separated by api, worker, airflow, dashboard, and admin.
- Secrets are read from AWS Secrets Manager.
- Logging uses PII scrubbing and correlation IDs.
- SQL filters use allowlisted columns and parameterized values.
- Rate limiting is enforced per identity or client fallback.
- Security audit middleware records access events.

Contribution Summary
- Branch from develop.
- Keep changes scoped.
- Add tests for behavioral changes.
- Run relevant tests locally.
- Update docs when behavior, deployment, security, or operations change.
- Open a pull request and wait for CI and review.
- Never commit secrets or production data.

License
MIT.
