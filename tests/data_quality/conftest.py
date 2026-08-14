"""Shared fixtures for data quality checks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from scripts.generate_test_data import generate_data_quality_dataset


@pytest.fixture(scope="session")
def data_quality_dataset() -> dict[str, Any]:
    return generate_data_quality_dataset(
        n_transactions=360,
        fraud_rate=0.08,
        seed=39,
        generated_at=datetime.now(timezone.utc),
    )
