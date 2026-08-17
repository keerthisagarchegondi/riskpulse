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
