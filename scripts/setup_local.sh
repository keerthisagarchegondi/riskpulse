#!/usr/bin/env bash
# RiskPulse - Local Development Setup Script
set -euo pipefail

echo "=== RiskPulse Local Setup ==="

# Check prerequisites
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
    for candidate in python3 python py; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            PYTHON_BIN="${candidate}"
            break
        fi
    done
fi

[ -n "${PYTHON_BIN}" ] || { echo "Python 3.11+ required"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker required"; exit 1; }

# Create virtual environment
echo "Creating virtual environment..."
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Copy environment file
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

echo "Preparing local-only runtime defaults..."
python scripts/setup_local.py --skip-compose-validation

# Start Docker services
echo "Starting Docker services..."
docker compose -f docker-compose.dev.yml up -d

# Wait for services
echo "Waiting for services to be ready..."
sleep 10

echo ""
echo "=== Setup Complete ==="
echo "Activate venv: source .venv/bin/activate"
echo "Run API: make run"
echo "Run tests: make test"
echo "View services: make docker-ps"
