"""Signed model artifacts must be authenticated before joblib unpickling."""

from __future__ import annotations

import joblib
import pytest

from src.fraud_detection import artifact_security
from src.fraud_detection.model_registry import ModelRegistry, ModelServer
from src.fraud_detection.risk_scorer import RiskScorer


def test_tampered_artifact_never_reaches_joblib(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RISKPULSE_ENV", "prod")
    monkeypatch.setenv("RISKPULSE_MODEL_SIGNING_KEY", "a" * 64)
    model = tmp_path / "model.joblib"
    joblib.dump({"model": True}, model)
    artifact_security.sign_model_artifact(model)
    model.write_bytes(b"untrusted pickle")
    monkeypatch.setattr(
        artifact_security.joblib,
        "load",
        lambda *_args: pytest.fail("untrusted bytes reached joblib.load"),
    )

    with pytest.raises(ValueError, match="Invalid model artifact signature"):
        artifact_security.load_model_artifact(model)


def test_managed_model_loading_requires_key_and_signature(tmp_path, monkeypatch) -> None:
    model = tmp_path / "model.joblib"
    joblib.dump({"model": True}, model)
    monkeypatch.setenv("RISKPULSE_ENV", "prod")
    monkeypatch.delenv("RISKPULSE_MODEL_SIGNING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RISKPULSE_MODEL_SIGNING_KEY"):
        artifact_security.load_model_artifact(model)

    monkeypatch.setenv("RISKPULSE_MODEL_SIGNING_KEY", "b" * 64)
    with pytest.raises(ValueError, match="Unsigned model artifact"):
        artifact_security.load_model_artifact(model)

    artifact_security.sign_model_artifact(model)
    assert artifact_security.load_model_artifact(model) == {"model": True}


def test_scorer_and_registry_reject_modified_models(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RISKPULSE_ENV", "prod")
    monkeypatch.setenv("RISKPULSE_MODEL_SIGNING_KEY", "c" * 64)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = model_dir / "model.joblib"
    joblib.dump({"model": True}, model)
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_model("risk_scorer", "1.0.0", model_dir, "test")
    model.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="Invalid model artifact signature"):
        RiskScorer(enable_shap=False).load_model(model_dir)
    with pytest.raises(ValueError, match="Invalid model artifact signature"):
        ModelServer._default_model_loader(str(model_dir))
