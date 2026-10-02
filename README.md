# RiskPulse

RiskPulse is a local-first fraud analytics platform for transaction ingestion, fraud scoring, alert management, operational dashboards, and validation tests.

The repository now runs without AWS, Snowflake, CloudWatch, ECR, ECS, Terraform, SES, or SNS credentials. Those integrations remain optional for production-style deployments.

## Prerequisites

- Python 3.12 recommended (3.11+ supported)
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
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

Prepare the local environment (creates `.env` if absent, generates a model signing key if blank, and preserves an existing key):

```bash
python scripts/setup_local.py
```

Do not commit `.env`. Docker Compose reads its signing key and passes the same value to API, worker, dashboard, and Airflow containers.

The defaults in `.env.example` are local-safe. They use:

- PostgreSQL on `localhost:15432`
- Redis on `localhost:16379`
- Kafka on `localhost:19092`
- local object storage under `.local_storage`
- local metrics under `.local_storage/metrics/metrics.jsonl`
- local notifications under `.local_storage/notifications`
- local Power BI data under `dashboards/powerbi/local_data`

## Run Locally

Start the complete stack from the repository root. Docker Desktop must be running:

```bash
docker compose --env-file .env up -d --build
docker compose --env-file .env ps
```

This starts PostgreSQL, Kafka, Redis, API, worker, Streamlit, and Airflow. On a new PostgreSQL volume, Docker runs `database/migrations/*.sql` automatically. Wait until the services show healthy before continuing; Airflow can take longer on its first start. Do not also run `make run` or `make run-streamlit` on the same ports.

Seed the local database. In PowerShell:

```powershell
Get-Content database/seeds/seed_rules.sql -Raw | docker compose --env-file .env exec -T postgres psql -U riskpulse -d riskpulse -v ON_ERROR_STOP=1
Get-Content database/seeds/seed_test_data.sql -Raw | docker compose --env-file .env exec -T postgres psql -U riskpulse -d riskpulse -v ON_ERROR_STOP=1
```

On Linux or macOS:

```bash
docker compose --env-file .env exec -T postgres psql -U riskpulse -d riskpulse -v ON_ERROR_STOP=1 < database/seeds/seed_rules.sql
docker compose --env-file .env exec -T postgres psql -U riskpulse -d riskpulse -v ON_ERROR_STOP=1 < database/seeds/seed_test_data.sql
```

Verify the API, dashboard, Airflow, and a sample transaction:

```bash
python scripts/smoke_test.py --base-url http://127.0.0.1:8000 --streamlit-url http://127.0.0.1:8501 --airflow-url http://127.0.0.1:8080 --use-dev-key --submit-test-transaction
```

Check the local data count:

```bash
docker compose --env-file .env exec postgres psql -U riskpulse -d riskpulse -c "SELECT COUNT(*) FROM transactions;"
```

Stop the stack without deleting its data:

```bash
docker compose --env-file .env down
```

For host-side development instead of the full stack, start only the dependencies and run the Python services in separate terminals with `.env` loaded. Do not start their Docker counterparts at the same time:

```bash
docker compose --env-file .env up -d postgres redis zookeeper kafka
python -m dotenv run -- python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
python -m dotenv run -- python -m src.ingestion.kafka_consumer
python -m dotenv run -- python -m streamlit run dashboards/streamlit/app.py --server.address 127.0.0.1 --server.port 8501
```

Default endpoints:

- API: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`
- Streamlit dashboard: `http://127.0.0.1:8501`
- Airflow, when enabled: `http://127.0.0.1:8080`
- PostgreSQL host port: `15432`
- Redis host port: `16379`
- Kafka host port: `19092`

Airflow's `standalone` startup creates an admin account and reports its initial password in `docker compose --env-file .env logs airflow`.

Dashboard login defaults:

- Admin: `admin` / `riskpulse2024!`
- Analyst: `analyst` / `analyst2024!`

These defaults are for local development only; change them before exposing the dashboard using `DASHBOARD_ADMIN_USER`, `DASHBOARD_ADMIN_PASSWORD`, `DASHBOARD_ANALYST_USER`, and `DASHBOARD_ANALYST_PASSWORD`.

## Signed Models

`scripts/setup_local.py` generates a random 64-character `RISKPULSE_MODEL_SIGNING_KEY` in the ignored `.env` file. Keep it private and stable: the same key must be present when saving, registering, and loading a model. Changing or losing it makes previously signed artifacts unloadable. Staging and production refuse model loads without a key or valid signature. Local development still permits unsigned artifacts only when no key is configured.

Existing unsigned model files must not be signed blindly. To recreate the included anomaly detector from this repository's synthetic training source, run this after setup (it replaces the artifacts in `ml/models/isolation_forest`):

```bash
python -m dotenv run -- python ml/training/train_anomaly_detector.py --n-samples 10000
```

The training script saves signed `model.joblib`, `scaler.joblib`, and `metadata.joblib` files with matching `.sig` files. Use a trusted training pipeline for any other model. The local Compose stack runs one dashboard instance; deployments with multiple replicas need a shared ingress rate limit in addition to the in-process login limiter.

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

If the dashboard says preview data is shown, check PostgreSQL health and seed data using the commands above. Migrations run automatically only when the PostgreSQL volume is first created:

```bash
docker compose --env-file .env ps postgres
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
