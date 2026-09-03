from __future__ import annotations

import json

import pytest

from src.alerting.notification_service import LocalEmailProvider, LocalSMSProvider


@pytest.mark.asyncio
async def test_local_email_provider_records_jsonl(tmp_path) -> None:
    output_path = tmp_path / "emails.jsonl"
    provider = LocalEmailProvider(output_path)

    result = await provider.send_email(
        to="analyst@example.com",
        subject="RiskPulse alert",
        body_html="<p>Alert</p>",
        body_text="Alert",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert result["status"] == "recorded"
    assert payload["channel"] == "email"
    assert payload["to"] == "analyst@example.com"


@pytest.mark.asyncio
async def test_local_sms_provider_records_jsonl(tmp_path) -> None:
    output_path = tmp_path / "sms.jsonl"
    provider = LocalSMSProvider(output_path)

    result = await provider.send_sms("+15555550100", "Critical alert")

    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert result["status"] == "recorded"
    assert payload["channel"] == "sms"
    assert payload["phone_number"] == "+15555550100"
