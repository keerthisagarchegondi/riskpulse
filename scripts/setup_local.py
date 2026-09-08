"""Prepare RiskPulse for local-only development."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"
ENV_FILE = ROOT / ".env"

LOCAL_DEFAULTS = {
    "RISKPULSE_STORAGE_BACKEND": "local",
    "RISKPULSE_LOCAL_STORAGE_ROOT": ".local_storage",
    "RISKPULSE_WAREHOUSE_BACKEND": "local",
    "RISKPULSE_LOCAL_WAREHOUSE_ROOT": ".local_storage/warehouse",
    "RISKPULSE_METRICS_BACKEND": "local",
    "RISKPULSE_LOCAL_METRICS_PATH": ".local_storage/metrics/metrics.jsonl",
    "RISKPULSE_NOTIFICATION_BACKEND": "local",
    "RISKPULSE_LOCAL_NOTIFICATION_ROOT": ".local_storage/notifications",
    "RISKPULSE_MONITORING__CLOUDWATCH__ENABLED": "false",
    "RISKPULSE_SECURITY__SECRETS_MANAGER__ENABLED": "false",
    "POWERBI_DATA_BACKEND": "local",
    "POWERBI_LOCAL_DATA_DIR": "dashboards/powerbi/local_data",
    "RUN_AWS_CHECKS": "false",
    "RUN_TERRAFORM": "false",
    "DEPLOYMENT_BACKEND": "local",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-services", action="store_true", help="Start Docker Compose services")
    parser.add_argument(
        "--skip-compose-validation",
        action="store_true",
        help="Skip docker compose config validation",
    )
    args = parser.parse_args()

    ensure_env()
    env_values = upsert_env_values(LOCAL_DEFAULTS)
    ensure_local_dirs(env_values)
    write_powerbi_local_artifacts(env_values)

    if not args.skip_compose_validation:
        validate_compose()
    if args.start_services:
        run(["docker", "compose", "-f", "docker-compose.yml", "up", "-d"], check=True)

    print("Local setup ready.")
    print("Storage: .local_storage")
    print("Power BI CSV shells: dashboards/powerbi/local_data")
    return 0


def ensure_env() -> None:
    if ENV_FILE.exists():
        return
    if not ENV_EXAMPLE.exists():
        raise FileNotFoundError(".env.example is required to create .env")
    shutil.copyfile(ENV_EXAMPLE, ENV_FILE)


def upsert_env_values(defaults: dict[str, str]) -> dict[str, str]:
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    updated: list[str] = []
    resolved = defaults.copy()

    for line in lines:
        if not line or line.lstrip().startswith("#") or "=" not in line:
            updated.append(line)
            continue

        key, _ = line.split("=", 1)
        if key in defaults:
            seen.add(key)
            updated.append(f"{key}={defaults[key]}")
        else:
            updated.append(line)

    missing = [key for key in defaults if key not in seen]
    if missing:
        updated.extend(["", "# Local-only runtime defaults"])
        updated.extend(f"{key}={defaults[key]}" for key in missing)

    ENV_FILE.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return resolved


def ensure_local_dirs(env_values: dict[str, str]) -> None:
    paths = [
        env_values["RISKPULSE_LOCAL_STORAGE_ROOT"],
        env_values["RISKPULSE_LOCAL_WAREHOUSE_ROOT"],
        Path(env_values["RISKPULSE_LOCAL_METRICS_PATH"]).parent,
        env_values["RISKPULSE_LOCAL_NOTIFICATION_ROOT"],
        env_values["POWERBI_LOCAL_DATA_DIR"],
    ]
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        path.mkdir(parents=True, exist_ok=True)


def write_powerbi_local_artifacts(env_values: dict[str, str]) -> None:
    os.environ.update(env_values)
    sys.path.insert(0, str(ROOT))
    from dashboards.powerbi.data_connections.snowflake_connector import (
        LocalPowerBIConfig,
        PowerBILocalConnector,
    )

    data_dir = Path(env_values["POWERBI_LOCAL_DATA_DIR"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    connector = PowerBILocalConnector(LocalPowerBIConfig(data_dir))
    connector.ensure_dataset_files()
    connector.write_power_query_file(ROOT / "dashboards/powerbi/models/local_queries.pq")


def validate_compose() -> None:
    if shutil.which("docker") is None:
        print("Docker not found; skipped compose validation.")
        return
    run(["docker", "compose", "--env-file", ".env", "-f", "docker-compose.yml", "config", "--quiet"])
    run(
        [
            "docker",
            "compose",
            "--env-file",
            ".env",
            "-f",
            "docker-compose.prod.yml",
            "config",
            "--quiet",
        ]
    )


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, check=check)


if __name__ == "__main__":
    raise SystemExit(main())
