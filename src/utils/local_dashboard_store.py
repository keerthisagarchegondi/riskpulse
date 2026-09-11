"""Local JSONL store used by the dashboard in local development."""

from __future__ import annotations

import json
import os
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any


def local_dashboard_enabled() -> bool:
    return os.environ.get("RISKPULSE_STORAGE_BACKEND", "").strip().lower() == "local"


def _root() -> Path:
    path = Path(os.environ.get("RISKPULSE_LOCAL_DASHBOARD_ROOT", ".local_storage/dashboard"))
    return path if path.is_absolute() else Path.cwd() / path


def _path(name: str) -> Path:
    return _root() / f"{name}.jsonl"


def _json_default(value: Any) -> str | float:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _dump_model(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value)


def append_local_record(name: str, payload: dict[str, Any]) -> None:
    if not local_dashboard_enabled():
        return

    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default, sort_keys=True))
        handle.write("\n")


def read_local_records(name: str) -> list[dict[str, Any]]:
    path = _path(name)
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def record_local_transaction(
    transaction_id: str | uuid.UUID,
    transaction: Any,
    *,
    status: str = "accepted",
) -> None:
    data = _dump_model(transaction)
    now = datetime.now(timezone.utc).isoformat()
    data.update(
        {
            "transaction_id": str(transaction_id),
            "status": status,
            "created_at": data.get("created_at", now),
            "updated_at": now,
        }
    )
    append_local_record("transactions", data)


def record_local_score(transaction: dict[str, Any], score: Any) -> None:
    data = score.model_dump(mode="json") if hasattr(score, "model_dump") else _dump_model(score)
    now = datetime.now(timezone.utc).isoformat()
    final_score = float(data.get("final_score", 0.0))
    transaction_id = str(data["transaction_id"])
    append_local_record(
        "risk_scores",
        {
            "score_id": str(uuid.uuid4()),
            "transaction_id": transaction_id,
            "overall_score": final_score,
            "risk_score": final_score,
            "model_version": data.get("scoring_version", "local-rules"),
            "risk_classification": data.get("risk_classification", "low"),
            "alert_recommended": bool(data.get("alert_recommended", False)),
            "latency_ms": float(data.get("total_latency_ms", 0.0)),
            "scoring_timestamp": now,
        },
    )

    status = "flagged" if data.get("alert_recommended") else "approved"
    record_local_transaction(transaction_id, transaction, status=status)
