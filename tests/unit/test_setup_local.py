"""Local setup preserves generated model signing credentials."""

from __future__ import annotations

import re

import pytest

from scripts import setup_local


@pytest.mark.parametrize(
    "contents",
    ["RISKPULSE_ENV=dev\nRISKPULSE_MODEL_SIGNING_KEY=\n", "RISKPULSE_ENV=dev\n"],
)
def test_model_signing_key_is_generated_once(tmp_path, monkeypatch, contents: str) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(setup_local, "ENV_FILE", env_file)

    setup_local.ensure_model_signing_key()
    first = env_file.read_text(encoding="utf-8")
    assert re.search(r"^RISKPULSE_MODEL_SIGNING_KEY=[0-9a-f]{64}$", first, re.MULTILINE)

    setup_local.ensure_model_signing_key()
    assert env_file.read_text(encoding="utf-8") == first


def test_short_model_signing_key_is_not_silently_rotated(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("RISKPULSE_MODEL_SIGNING_KEY=short\n", encoding="utf-8")
    monkeypatch.setattr(setup_local, "ENV_FILE", env_file)

    with pytest.raises(ValueError, match="at least 32 bytes"):
        setup_local.ensure_model_signing_key()
    assert env_file.read_text(encoding="utf-8") == "RISKPULSE_MODEL_SIGNING_KEY=short\n"
