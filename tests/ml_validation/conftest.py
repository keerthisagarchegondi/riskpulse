"""Shared ML validation fixtures."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.generate_test_data import generate_ml_validation_dataset


@pytest.fixture(scope="session")
def ml_validation_dataset() -> dict[str, Any]:
    return generate_ml_validation_dataset(n_samples=800, seed=39)
