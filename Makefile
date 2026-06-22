.PHONY: install install-dev lint format test test-unit test-integration test-coverage run docker-up docker-down docker-build clean help

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
	python -m pip install --upgrade pip
	pip install -e .

install-dev: ## Install development dependencies
	python -m pip install --upgrade pip
	pip install -e ".[dev]"
	pre-commit install

install-airflow: ## Install with Airflow dependencies
	python -m pip install --upgrade pip
	pip install -e ".[airflow]"

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run all linters
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

format: ## Auto-format code
	black src/ tests/
	isort src/ tests/

security-scan: ## Run security checks
	bandit -r src/ -c pyproject.toml
	safety check

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v -m unit

test-integration: ## Run integration tests
	pytest tests/integration/ -v -m integration

test-coverage: ## Run tests with coverage report
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

test-performance: ## Run performance tests
	pytest tests/performance/ -v -m performance

# =============================================================================
# Application
# =============================================================================

run: ## Run the API server
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

run-worker: ## Run the Kafka consumer worker
	python -m src.ingestion.kafka_consumer

run-streamlit: ## Run the Streamlit dashboard
	streamlit run dashboards/streamlit/app.py --server.port 8501

# =============================================================================
# Docker
# =============================================================================

docker-up: ## Start all development services
	docker compose -f docker-compose.dev.yml up -d

docker-down: ## Stop all development services
	docker compose -f docker-compose.dev.yml down

docker-build: ## Build all Docker images
	docker compose -f docker-compose.dev.yml build

docker-logs: ## View Docker service logs
	docker compose -f docker-compose.dev.yml logs -f

docker-ps: ## Show running containers
	docker compose -f docker-compose.dev.yml ps

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
	python scripts/generate_test_data.py

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache htmlcov .coverage .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

check-all: lint test-coverage security-scan ## Run all checks (lint + test + security)
