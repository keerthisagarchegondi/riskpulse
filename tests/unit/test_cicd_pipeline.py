"""Tests for GitHub Actions CI/CD and deployment scripts."""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOWS = Path(".github/workflows")


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _load_workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_ci_workflow_has_required_triggers_and_quality_gates() -> None:
    workflow = _load_workflow("ci.yml")
    content = _read(".github/workflows/ci.yml")

    assert workflow["name"] == "CI"
    assert "pull_request" in workflow[True]
    assert set(workflow[True]["pull_request"]["branches"]) == {"develop", "main"}
    assert "workflow_call" in workflow[True]
    assert "--cov-fail-under=${COVERAGE_MINIMUM}" in content
    assert "black --check" in content
    assert "isort --check-only" in content
    assert "flake8" in content
    assert "mypy src" in content
    assert "bandit -r src" in content
    assert "safety check" in content
    assert "docker compose -f docker-compose.yml build api worker streamlit airflow" in content
    assert "actions/upload-artifact@v6" in content


def test_staging_workflow_runs_ci_builds_and_validates_local_deploy() -> None:
    workflow = _load_workflow("cd-staging.yml")
    content = _read(".github/workflows/cd-staging.yml")

    assert workflow["name"] == "CD Staging"
    assert workflow[True]["push"]["branches"] == ["develop"]
    assert workflow["jobs"]["ci"]["uses"] == "./.github/workflows/ci.yml"
    assert "Build Local Images" in content
    assert "docker compose -f docker-compose.prod.yml build api worker streamlit airflow" in content
    assert "DEPLOYMENT_BACKEND: local" in content
    assert 'DRY_RUN: "true"' in content
    assert "bash scripts/deploy.sh staging" in content
    assert "docker compose --env-file .env.example -f docker-compose.yml config --quiet" in content
    assert "aws-actions/configure-aws-credentials" not in content
    assert "amazon-ecr-login" not in content
    assert "docker push" not in content


def test_production_workflow_has_manual_gate_monitoring_and_rollback() -> None:
    workflow = _load_workflow("cd-production.yml")
    content = _read(".github/workflows/cd-production.yml")

    assert workflow["name"] == "CD Production"
    assert "workflow_dispatch" in workflow[True]
    assert workflow[True]["push"]["branches"] == ["main"]
    assert workflow["jobs"]["deploy"]["environment"]["name"] == "production"
    assert "deploy-production" in content
    assert "DEPLOYMENT_BACKEND: local" in content
    assert "Validate local production deployment plan" in content
    assert "bash scripts/rollback.sh production deployment-state" in content
    assert (
        "docker compose --env-file .env.example -f docker-compose.prod.yml config --quiet"
        in content
    )
    assert "aws ecs" not in content


def test_deploy_and_rollback_scripts_default_to_local_compose() -> None:
    deploy = _read("scripts/deploy.sh")
    rollback = _read("scripts/rollback.sh")

    assert 'DEPLOYMENT_BACKEND="${DEPLOYMENT_BACKEND:-local}"' in deploy
    assert 'if [[ "${DEPLOYMENT_BACKEND}" == "local" ]]' in deploy
    assert 'docker compose -f "${compose_file}" up -d --build' in deploy
    assert "DRY_RUN" in deploy
    assert 'DEPLOYMENT_BACKEND="${DEPLOYMENT_BACKEND:-local}"' in rollback
    assert 'if [[ "${DEPLOYMENT_BACKEND}" == "local" ]]' in rollback
    assert 'docker compose -f "${compose_file}" up -d --force-recreate' in rollback


def test_deployment_guide_uses_txt_not_markdown() -> None:
    assert Path("docs/deployment_guide.txt").exists()
    assert not Path("docs/deployment_guide.md").exists()

    guide = _read("docs/deployment_guide.txt")
    assert "Branch Protection" in guide
    assert "90 percent" in guide
    assert "Rollback Flow" in guide
