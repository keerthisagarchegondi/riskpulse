PYTHON ?= python
PIP := $(PYTHON) -m pip

.PHONY: install install-dev lint format test test-unit test-integration test-coverage run docker-up docker-down docker-build docker-test smoke-test verify-deployment clean help

# Default target
help: ## Show this help message
	@echo "RiskPulse - Fraud Analytics & Risk Intelligence Platform"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================

install: ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PYTHON) -m pre_commit install

install-airflow: ## Install with Airflow dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[airflow]"

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run all linters
	$(PYTHON) -m black --check src/ tests/
	$(PYTHON) -m isort --check-only src/ tests/
	$(PYTHON) -m flake8 src/ tests/
	$(PYTHON) -m mypy src/

format: ## Auto-format code
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m isort src/ tests/

security-scan: ## Run security checks
	$(PYTHON) -m bandit -r src/ -c pyproject.toml
	$(PYTHON) -m safety check

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit/ -v -m unit

test-integration: ## Run integration tests
	$(PYTHON) -m pytest tests/integration/ -v -m integration

test-coverage: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

test-performance: ## Run performance tests
	$(PYTHON) -m pytest tests/performance/ -v -m performance

# =============================================================================
# Application
# =============================================================================

run: ## Run the API server
	$(PYTHON) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

run-worker: ## Run the Kafka consumer worker
	$(PYTHON) -m src.ingestion.kafka_consumer

run-streamlit: ## Run the Streamlit dashboard
	$(PYTHON) -m streamlit run dashboards/streamlit/app.py --server.port 8501

# =============================================================================
# Docker
# =============================================================================

docker-up: ## Start all development services
	docker compose -f docker-compose.yml up -d

docker-down: ## Stop all development services
	docker compose -f docker-compose.yml down

docker-build: ## Build all Docker images
	docker compose -f docker-compose.yml build api worker streamlit airflow

docker-build-prod: ## Build production Docker images
	docker compose -f docker-compose.prod.yml build api worker streamlit airflow

docker-test: ## Validate compose files and run container health smoke checks
	docker compose -f docker-compose.yml config --quiet
	docker compose -f docker-compose.prod.yml config --quiet
	docker compose -f docker-compose.yml up -d --build
	docker compose -f docker-compose.yml ps
	docker compose -f docker-compose.yml down

smoke-test: ## Run API/dashboard deployment smoke tests
	$(PYTHON) scripts/smoke_test.py --base-url $${RISKPULSE_BASE_URL:-http://127.0.0.1:8000}

verify-deployment: ## Run production deployment verification gates
	./scripts/verify_deployment.sh production

docker-logs: ## View Docker service logs
	docker compose -f docker-compose.yml logs -f

docker-ps: ## Show running containers
	docker compose -f docker-compose.yml ps

# =============================================================================
# Database
# =============================================================================

db-migrate: ## Run database migrations
	@for file in database/migrations/*.sql; do \
		echo "Running $$file..."; \
		PGPASSWORD=$(POSTGRES_PASSWORD) psql -h localhost -U $(POSTGRES_USER) -d $(POSTGRES_DB) -f $$file; \
	done

db-seed: ## Seed database with test data
	@for file in database/seeds/*.sql; do \
		echo "Running $$file..."; \
		PGPASSWORD=$(POSTGRES_PASSWORD) psql -h localhost -U $(POSTGRES_USER) -d $(POSTGRES_DB) -f $$file; \
	done

db-reset: ## Reset database (drop and recreate)
	PGPASSWORD=$(POSTGRES_PASSWORD) psql -h localhost -U $(POSTGRES_USER) -d postgres -c "DROP DATABASE IF EXISTS $(POSTGRES_DB);"
	PGPASSWORD=$(POSTGRES_PASSWORD) psql -h localhost -U $(POSTGRES_USER) -d postgres -c "CREATE DATABASE $(POSTGRES_DB);"
	$(MAKE) db-migrate
	$(MAKE) db-seed

# =============================================================================
# Utilities
# =============================================================================

generate-data: ## Generate synthetic test data
	$(PYTHON) scripts/generate_test_data.py

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache htmlcov .coverage .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

check-all: lint test-coverage security-scan ## Run all checks (lint + test + security)
