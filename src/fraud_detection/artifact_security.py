"""Authenticate joblib artifacts before unpickling them."""

from __future__ import annotations

import hashlib
import hmac
import io
import os
from pathlib import Path
from typing import Any

import joblib


def _signing_key() -> bytes | None:
    value = os.environ.get("RISKPULSE_MODEL_SIGNING_KEY", "")
    if value:
        key = value.encode("utf-8")
        if len(key) < 32:
            raise ValueError("RISKPULSE_MODEL_SIGNING_KEY must be at least 32 bytes")
        return key
    if os.environ.get("RISKPULSE_ENV", "dev").lower() in {"prod", "production", "staging"}:
        raise RuntimeError("RISKPULSE_MODEL_SIGNING_KEY is required to load models")
    return None


def _signature(path: Path, data: bytes, key: bytes) -> str:
    return hmac.new(key, path.name.encode("utf-8") + b"\0" + data, hashlib.sha256).hexdigest()


def sign_model_artifact(path: Path) -> None:
    key = _signing_key()
    if key is not None:
        signature = _signature(path, path.read_bytes(), key)
        path.with_name(path.name + ".sig").write_text(signature, encoding="ascii")


def load_model_artifact(path: Path) -> Any:
    key = _signing_key()
    signature_path = path.with_name(path.name + ".sig")
    if key is None:
        if signature_path.exists():
            raise RuntimeError("RISKPULSE_MODEL_SIGNING_KEY is required for signed models")
        return joblib.load(path)

    data = path.read_bytes()
    try:
        expected = signature_path.read_text(encoding="ascii").strip()
    except FileNotFoundError as exc:
        raise ValueError(f"Unsigned model artifact: {path.name}") from exc
    if not hmac.compare_digest(expected, _signature(path, data, key)):
        raise ValueError(f"Invalid model artifact signature: {path.name}")
    return joblib.load(io.BytesIO(data))
