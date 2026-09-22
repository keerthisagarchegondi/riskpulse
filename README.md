# RiskPulse

RiskPulse is a local-first fraud analytics platform for transaction ingestion, fraud scoring, alert management, operational dashboards, and validation tests.

The repository now runs without AWS, Snowflake, CloudWatch, ECR, ECS, Terraform, SES, or SNS credentials. Those integrations remain optional for production-style deployments.

## Prerequisites

- Python 3.11
- Git
- Docker Desktop with Docker Compose
- Make, optional on Windows

## Setup

Clone and enter the repository:

```bash
git clone https://github.com/keerthisagarchegondi/riskpulse.git
cd riskpulse
```

Create a virtual environment.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

Create the local environment file:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

The defaults in `.env.example` are local-safe. They use:

- PostgreSQL on `localhost:15432`
- Redis on `localhost:16379`
- Kafka on `localhost:19092`
- local object storage under `.local_storage`
- local metrics under `.local_storage/metrics/metrics.jsonl`
- local notifications under `.local_storage/notifications`
- local Power BI data under `dashboards/powerbi/local_data`

## Run Locally

Start local infrastructure and services:

```bash
docker compose up -d
```

Or use Make:

```bash
make docker-up
```

Run database migrations:

```bash
make db-migrate
```

Seed development data when needed:

```bash
make db-seed
```

Run the API:

```bash
make run
```

Run the worker:

```bash
make run-worker
```

Run the Streamlit dashboard:

```bash
make run-streamlit
```

Default endpoints:

- API: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`
- Streamlit dashboard: `http://127.0.0.1:8501`
- Airflow, when enabled: `http://127.0.0.1:8080`
- PostgreSQL host port: `15432`
- Redis host port: `16379`
- Kafka host port: `19092`

Dashboard login defaults:

- Admin: `admin` / `riskpulse2024!`
- Analyst: `analyst` / `analyst2024!`

Change these with `DASHBOARD_ADMIN_USER`, `DASHBOARD_ADMIN_PASSWORD`, `DASHBOARD_ANALYST_USER`, and `DASHBOARD_ANALYST_PASSWORD`.

## Local Data Flow

The local setup uses Docker for PostgreSQL, Kafka, Redis, API, Streamlit, worker, and Airflow.

External cloud services are replaced by local backends by default:

- `RISKPULSE_STORAGE_BACKEND=local`
- `RISKPULSE_WAREHOUSE_BACKEND=local`
- `RISKPULSE_METRICS_BACKEND=local`
- `RISKPULSE_NOTIFICATION_BACKEND=local`
- `POWERBI_DATA_BACKEND=local`
- `DEPLOYMENT_BACKEND=local`
- `RUN_AWS_CHECKS=false`

If the database is unavailable, Streamlit can show local preview data so the frontend still opens. Once PostgreSQL is running and seeded, dashboards read local database/local storage data.

## Quality Checks

Format and lint:

```bash
black --check src tests scripts dashboards ml airflow
isort --check-only src tests scripts dashboards ml airflow
flake8 src tests scripts dashboards ml airflow
```

Type check:

```bash
mypy src
```

Unit tests:

```bash
pytest tests/unit
```

Integration tests require Docker services:

```bash
docker compose up -d postgres redis zookeeper kafka
pytest tests/integration -m integration
```

Security checks:

```bash
bandit -r src scripts dashboards ml airflow -c pyproject.toml -ll
safety check --full-report
```

Run the combined local checks:

```bash
make check-all
```

## Useful Commands

```bash
make install-dev
make format
make lint
make test
make test-unit
make test-integration
make test-coverage
make security-scan
make docker-up
make docker-ps
make docker-logs
make docker-down
make smoke-test
make generate-data
```

## CI/CD

GitHub Actions workflows live in `.github/workflows`.

For local-only CI validation, no AWS credentials are required. Docker Compose provides PostgreSQL, Redis, and Kafka.

Cloud deployment variables and secrets are only needed if you re-enable AWS deployment:

- `AWS_REGION`
- `ECR_REGISTRY`
- `STAGING_AWS_ROLE_ARN`
- `PRODUCTION_AWS_ROLE_ARN`
- staging and production service/base URL variables

Keep production secrets out of the repository. Use GitHub secrets or your deployment platform.

## Optional Cloud Integrations

Enable these only when you are ready for cloud deployment:

- AWS S3 object storage: set `RISKPULSE_STORAGE_BACKEND=s3`
- AWS CloudWatch: set `RISKPULSE_MONITORING__CLOUDWATCH__ENABLED=true`
- AWS Secrets Manager: set `RISKPULSE_SECURITY__SECRETS_MANAGER__ENABLED=true`
- Snowflake warehouse: set `RISKPULSE_WAREHOUSE_BACKEND=snowflake`
- Power BI refresh from Snowflake: configure Snowflake credentials and Power BI Service access
- SES/SNS notifications: switch notification backend and provide AWS credentials

## Troubleshooting

If Docker cannot bind ports, another local process is using the host port. The project defaults avoid common conflicts by using `15432`, `16379`, and `19092`.

If the dashboard says preview data is shown, start PostgreSQL, run migrations, and seed data:

```bash
docker compose up -d postgres
make db-migrate
make db-seed
```

If editable installation fails because `README.md` is missing, ensure this file is present in the repository root.

If Docker Desktop fails to start, fix Docker Desktop first; the app can run Python-only tests without Docker, but full local integration needs Docker.

## Repository Layout

- `src`: API, ingestion, storage, monitoring, fraud detection, and shared utilities
- `dashboards`: Streamlit and Power BI assets
- `database`: migrations, seeds, and warehouse SQL
- `airflow`: DAGs and orchestration code
- `ml`: model training and model artifacts
- `scripts`: smoke tests, deployment helpers, and data generation
- `tests`: unit, integration, performance, security, data quality, and ML validation tests
- `infrastructure`: Docker, Terraform, IAM, and CloudWatch assets

## Security Notes

- Do not commit `.env` or real credentials.
- Replace default dashboard and API secrets before any shared deployment.
- API authentication supports API keys and JWT bearer tokens.
- SQL filters use allowlisted columns and parameterized values.
- Logs scrub common PII and secrets before output.

## License

MIT
