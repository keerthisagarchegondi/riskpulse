# RiskPulse

RiskPulse is a local-first fraud detection platform. It accepts transaction events, validates and enriches them, sends them through a fraud-scoring pipeline, stores the results, and exposes monitoring views through FastAPI and Streamlit.

The repository runs locally without AWS, Snowflake, Power BI, or cloud credentials. PostgreSQL, Kafka, Redis, the API, the worker, the dashboard, and Airflow run through Docker Compose. Cloud integrations are optional adapters for a later deployment.

## The Complete Flow

A normal transaction moves through the system as follows:

1. A client submits a transaction to the FastAPI service.
2. The API validates the request, checks authentication, and applies rate limits.
3. The API publishes the accepted event to Kafka.
4. The worker consumes the Kafka event.
5. The processing pipeline validates the schema, applies business rules, cleans and normalizes values, engineers features, and enriches the record.
6. Fraud rules and model scoring calculate a risk result.
7. The result is written to the configured storage backends. Local PostgreSQL, Redis, local object storage, local warehouse files, local metrics, and local notifications are supported.
8. The Streamlit dashboard reads the available local data and displays operational, model, alert, and trend views.
9. Airflow can run scheduled ingestion and processing workflows.

In staging or production, the same flow can use AWS, Snowflake, CloudWatch, and external notification providers. Those integrations are disabled by default for local development.

## Main Repository Areas

- src/api: FastAPI application, routes, schemas, authentication, rate limiting, and security middleware.
- src/ingestion: Kafka producer and consumer code.
- src/validation, src/transformation, and src/enrichment: data-quality and feature-processing stages.
- src/fraud_detection: rules, scoring, model registry, model monitoring, and artifact security.
- src/storage: PostgreSQL, Redis, local storage, S3, and Snowflake adapters.
- src/monitoring: health checks, metrics, structured logging, and optional CloudWatch integration.
- dashboards/streamlit: dashboard application and pages.
- airflow/dags: scheduled orchestration workflows.
- database: PostgreSQL and Snowflake schemas, migrations, and procedures.
- infrastructure/docker: multi-stage Dockerfiles for the services.
- infrastructure/terraform: optional cloud infrastructure modules.
- scripts: health, smoke-test, deployment, rollback, and security verification scripts.
- tests: unit, integration, data-quality, model-validation, performance, and security tests.

## Requirements

For the complete local stack, install:

- Docker Desktop with Docker Compose v2.
- Git.
- Python 3.11 or newer for local tests and quality checks.
- At least 8 GB of free memory for the full Compose stack.

Docker is the application runtime. If Docker Desktop is stopped, the containers cannot start and the API will report dependency failures.

On Windows, open PowerShell and move into the repository:

~~~powershell
cd C:\Users\VSC\Desktop\projects\sivadharma\riskpulse
~~~

Confirm the tools are available:

~~~powershell
docker version
docker compose version
python --version
~~~

If Docker reports that the daemon is unavailable, start Docker Desktop and wait until its engine reports as running.

## First-Time Local Setup

Create the local environment file:

~~~powershell
Copy-Item .env.example .env
~~~

The local defaults use:

- PostgreSQL at localhost:5432.
- Redis at localhost:6379.
- Kafka at localhost:9092.
- FastAPI at http://localhost:8000.
- Streamlit at http://localhost:8501.
- Airflow at http://localhost:8080.
- Local storage under .local_storage.
- Local warehouse data under .local_storage/warehouse.
- Local metrics under .local_storage/metrics.
- Local notifications under .local_storage/notifications.

Keep the local defaults unless you need to change a port or backend. Do not put real credentials in .env, commit them, or paste them into source files.

If another application already owns ports 5432, 6379, 8000, 8501, or 8080, stop that application or change the host-side port in .env. Do not start a second PostgreSQL or Redis service on the same host port.

## Start the Local Platform

Build and start the development stack:

~~~powershell
docker compose up --build -d
~~~

Check the service state:

~~~powershell
docker compose ps
~~~

Watch the important logs:

~~~powershell
docker compose logs -f api worker streamlit
~~~

Wait until the containers show healthy. Then open:

- API documentation: http://localhost:8000/docs
- API OpenAPI JSON: http://localhost:8000/openapi.json
- API health: http://localhost:8000/health
- API liveness: http://localhost:8000/health/live
- API readiness: http://localhost:8000/health/ready
- Streamlit dashboard: http://localhost:8501
- Airflow: http://localhost:8080

The dashboard may show local preview data when PostgreSQL is unavailable. That helps with checking the UI, but it does not prove that the Kafka, worker, scoring, and persistence flow is healthy. Use the health endpoints and smoke test for that.

## Dashboard Login

The dashboard login values come from the local environment configuration. Use the values defined in .env.example or your own .env file. Development credentials are for local use only and must be replaced before any public deployment.

If the dashboard says that PostgreSQL is offline, check the container state first:

~~~powershell
docker compose ps postgres
docker compose logs postgres
~~~

After PostgreSQL becomes healthy, refresh the dashboard and call the API health endpoint again.

## API Authentication

The local API uses the configured development API key for protected requests. Set it in the shell rather than putting it in a URL:

~~~powershell
$env:RISKPULSE_API_KEY = "dev-api-key-riskpulse-2024"
~~~

The development key is intentionally not suitable for staging or production. Replace it with a long random secret before deploying outside a local machine. Production keys must come from the configured secret-management path and must never be committed.

## Submit a Test Transaction

The easiest interactive method is the Swagger UI:

1. Open http://localhost:8000/docs.
2. Use the authentication control if the endpoint requires it.
3. Select the transaction submission endpoint.
4. Provide a request body matching the displayed schema.
5. Submit the request.
6. Use the returned transaction ID to retrieve the transaction and inspect its status.

The standard-library smoke test performs non-destructive checks by default:

~~~powershell
$env:RISKPULSE_BASE_URL = "http://localhost:8000"
$env:RISKPULSE_API_KEY = "dev-api-key-riskpulse-2024"
python scripts/smoke_test.py --base-url $env:RISKPULSE_BASE_URL --api-key $env:RISKPULSE_API_KEY
~~~

A synthetic write is opt-in because it creates data:

~~~powershell
python scripts/smoke_test.py --base-url http://localhost:8000 --api-key $env:RISKPULSE_API_KEY --submit-test-transaction --verify-processed-transaction
~~~

Only use the write option against a local or explicitly approved test environment. A 202 Accepted response means only that the event was accepted for processing. The processing check waits for a final stored status and is the meaningful end-to-end check.

## Common Local Commands

Stop the stack:

~~~powershell
docker compose down
~~~

Stop containers and remove orphan containers while keeping named volumes:

~~~powershell
docker compose down --remove-orphans
~~~

Show all service logs:

~~~powershell
docker compose logs -f postgres redis kafka api worker streamlit airflow
~~~

Open a shell in the API container:

~~~powershell
docker compose exec api sh
~~~

Run a container health check:

~~~powershell
docker compose exec api python scripts/container_healthcheck.py api
~~~

Restart selected services:

~~~powershell
docker compose restart postgres redis kafka
~~~

Rebuild one service:

~~~powershell
docker compose build api
docker compose up -d api
~~~

## Run Tests Without Docker

Create and activate a virtual environment:

~~~powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
~~~

If PowerShell blocks activation, use the interpreter directly:

~~~powershell
.\\.venv\\Scripts\\python.exe -m pytest
~~~

Run the focused unit and security tests:

~~~powershell
.\\.venv\\Scripts\\python.exe -m pytest tests/unit tests/security -q
~~~

Run the full suite:

~~~powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
~~~

Run the code-quality checks:

~~~powershell
.\\.venv\\Scripts\\python.exe -m mypy src scripts/smoke_test.py scripts/verify_runtime_security.py
.\\.venv\\Scripts\\python.exe -m black --check src scripts tests
.\\.venv\\Scripts\\python.exe -m isort --check-only src scripts tests
.\\.venv\\Scripts\\python.exe -m flake8 src scripts tests
.\\.venv\\Scripts\\python.exe -m bandit -r src scripts -ll
~~~

Tests that require PostgreSQL, Kafka, Redis, or Docker need those services running. Tests that require cloud accounts should remain disabled for local-only development.

## Troubleshooting

### Port 5432 or 6379 is already allocated

Another PostgreSQL or Redis process already owns the host port. Stop that process or change the host-side port in .env. The container ports remain 5432 and 6379; only the host mapping changes.

### PostgreSQL is offline

Check the service and logs:

~~~powershell
docker compose ps postgres
docker compose logs postgres
docker compose restart postgres
~~~

The dashboard's preview mode is useful for UI work but is not live database data.

### Kafka or Redis is unhealthy

Inspect the dependency logs:

~~~powershell
docker compose logs kafka zookeeper redis
docker compose ps
~~~

Correct the underlying issue before restarting:

~~~powershell
docker compose restart zookeeper kafka redis
~~~

### Docker Desktop crashes or cannot start

Restart Docker Desktop and wait for the engine to become ready. If the error mentions a locked Docker socket or inference file, close other Docker Desktop processes and restart Windows if necessary. The application cannot work around a stopped Docker daemon.

### The API returns 503 in staging or production

This is intentional when a managed environment cannot reach Kafka or another required dependency. Check /health, worker logs, Kafka connectivity, and deployed secret configuration. Do not change the API to return 202 just to hide a missing pipeline dependency.

## Production and Staging Verification

The production workflow validates a deployment plan. It does not claim that a live environment is healthy without real endpoints, credentials, and reachable services.

Before running live verification, configure the operator environment with:

- PRODUCTION_BASE_URL: the real HTTPS API URL.
- PRODUCTION_STREAMLIT_URL: the real HTTPS dashboard URL.
- A read-scoped API key or bearer token for positive API authentication checks.
- Dashboard ingress Basic-auth credentials when that ingress is enabled.
- Real database, JWT, model-signing, and Airflow secrets through the deployment secret store.
- Docker or the actual deployment backend, depending on where the services run.

Run the verification from the deployment host:

~~~bash
bash scripts/verify_deployment.sh production
~~~

The production gate checks repository artifacts, Compose configuration, live HTTPS endpoints, authentication rejection and acceptance, dependency health, and the deployed database role. It rejects loopback and example URLs and does not print secret values.

To verify one approved synthetic transaction end to end, set ALLOW_SYNTHETIC_PROD_WRITES=true and use a test credential with write permission. This must be intentional because it creates data.

The external network boundary needs a separate probe. Verify that PostgreSQL, Redis, Kafka, and Airflow are not reachable from the public internet. In-container connectivity alone cannot prove firewall isolation.

## Security Rules

- Never commit .env, cloud credentials, passwords, API keys, JWT secrets, model-signing keys, or private certificates.
- Never trust a user ID sent by the frontend. Derive identity and permissions from authenticated server-side context.
- Keep authentication, rate limiting, input validation, parameterized queries, CORS restrictions, and security headers enabled.
- Use TLS for public API and dashboard traffic.
- Use a shared ingress rate limit when more than one dashboard or API replica is running.
- Set RISKPULSE_MODEL_SIGNING_KEY to a random value of at least 32 bytes before saving or loading managed model artifacts.
- Keep PostgreSQL, Kafka, Redis, and Airflow on private networks.
- Review the full Git history for secrets before publishing the repository.

## Optional Cloud Integrations

Local development does not need AWS or Snowflake. Enable cloud integrations only after the target resources and permissions exist:

- AWS Secrets Manager for managed credentials.
- CloudWatch for logs, metrics, dashboards, and alarms.
- S3 for object storage.
- SNS or SES for external notifications.
- ECR and ECS, or another container platform, for deployment.
- Snowflake for warehouse loading and Power BI refreshes.
- Terraform for cloud infrastructure provisioning.

Do not add cloud keys just to make the local stack start. Keep the local backends selected until cloud resources and permissions have been verified.

## Deployment Files

- docker-compose.yml is the development stack.
- docker-compose.prod.yml is the hardened Compose configuration with non-root containers, resource limits, health checks, read-only filesystems, and local-first backends.
- scripts/deploy.sh validates or starts deployment actions according to the selected backend.
- scripts/rollback.sh contains the rollback procedure for a configured deployment target.
- scripts/smoke_test.py performs standard-library smoke checks.
- scripts/verify_deployment.sh runs production readiness gates.
- scripts/verify_runtime_security.py checks managed secrets and PostgreSQL privileges from inside the API runtime.
- .github/workflows/ci.yml runs CI checks.
- .github/workflows/cd-staging.yml and .github/workflows/cd-production.yml validate the configured deployment path and require a configured target before a real deployment.

## Current Readiness Boundary

The repository is ready for local development and local testing when Docker Desktop is running and the Python tooling is installed.

It is not automatically production-certified because the code builds or the API returns 202. Production readiness requires a reachable HTTPS deployment, healthy PostgreSQL/Kafka/Redis services, valid managed secrets, least-privilege database permissions, ingress authentication and rate limiting, an approved end-to-end transaction test, and an external firewall check.

Until those conditions exist, keep deployment verification in dry-run mode.

## Safe Commit Workflow

Before committing:

~~~powershell
git diff --check
git status --short
.\\.venv\\Scripts\\python.exe -m pytest tests/unit tests/security -q
~~~

Review the diff for credentials, generated files, and unrelated changes. Commit locally with a short message that describes the change. Do not push until local checks and target deployment checks are complete.

