"""Tests for Docker containerization artifacts."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(".")
DOCKER_DIR = ROOT / "infrastructure" / "docker"


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_all_required_dockerfiles_are_multistage_and_non_root() -> None:
    for name in ("api", "worker", "streamlit", "airflow"):
        content = _read(str(DOCKER_DIR / f"Dockerfile.{name}"))

        assert " AS builder" in content
        assert " AS runtime" in content
        assert "HEALTHCHECK" in content
        assert "USER root" not in content.split(" AS runtime", 1)[-1].split("CMD", 1)[-1]
        assert ("USER riskpulse" in content) or ("USER airflow" in content)


def test_dockerignore_excludes_large_and_sensitive_context() -> None:
    content = _read(".dockerignore")

    for pattern in (".git", ".env", "docs", "*.md", "*.ipynb", "node_modules"):
        assert pattern in content


def test_development_compose_runs_all_platform_services() -> None:
    compose = yaml.safe_load(_read("docker-compose.yml"))
    services = compose["services"]

    expected = {"zookeeper", "kafka", "postgres", "redis", "api", "worker", "streamlit", "airflow"}
    assert expected.issubset(services)

    for service_name in ("api", "worker", "streamlit", "airflow"):
        service = services[service_name]
        assert "build" in service
        assert "healthcheck" in service
        assert "depends_on" in service
        assert service["networks"]


def test_production_compose_has_resource_limits_restart_policies_and_awslogs() -> None:
    compose = yaml.safe_load(_read("docker-compose.prod.yml"))

    for service_name in ("api", "worker", "streamlit", "airflow"):
        service = compose["services"][service_name]

        assert service["restart"] == "unless-stopped"
        assert service["logging"]["driver"] == "awslogs"
        assert "healthcheck" in service
        assert "deploy" in service
        assert "limits" in service["deploy"]["resources"]
        assert "volumes" not in service
        assert "no-new-privileges:true" in service["security_opt"]

    assert compose["services"]["api"]["read_only"] is True
    assert compose["services"]["worker"]["read_only"] is True
    assert compose["services"]["streamlit"]["read_only"] is True


def test_makefile_has_docker_operation_targets() -> None:
    content = _read("Makefile")

    for target in ("docker-build:", "docker-build-prod:", "docker-up:", "docker-down:", "docker-test:"):
        assert target in content

    assert "docker-compose.yml config --quiet" in content
    assert "docker-compose.prod.yml config --quiet" in content
