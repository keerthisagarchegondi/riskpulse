#!/usr/bin/env python3
# isort: skip_file
"""Back-test fraud detection rules against labelled historical data.

Usage::

    # Generate synthetic data and run backtest
    python -m scripts.backtest_rules

    # Point at a real JSON-lines file (each line: {"transaction": {...}, "is_fraud": bool})
    python -m scripts.backtest_rules --data path/to/labelled.jsonl

    # Restrict to specific rule categories
    python -m scripts.backtest_rules --categories amount velocity

    # Output metrics as JSON
    python -m scripts.backtest_rules --json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is importable when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.fraud_detection.rule_engine import FraudRuleEngine  # noqa: E402

# ── Synthetic Data Generator ─────────────────────────────────────────

_RNG = random.Random(42)

_COUNTRIES = ["US", "US", "US", "US", "CA", "GB", "RU", "NG", "BR"]
_CHANNELS = ["online", "pos", "atm", "mobile"]
_TXN_TYPES = ["purchase", "withdrawal", "transfer", "refund"]
_MCCS = ["5411", "5812", "5541", "7995", "5912", "4829", "5311", "5651"]


def _random_ts(base: datetime, offset_minutes: int = 0) -> str:
    dt = base + timedelta(minutes=offset_minutes)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _generate_legitimate(idx: int, base_ts: datetime) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    amount = round(_RNG.uniform(5, 500), 2)
    ts = _random_ts(base_ts, _RNG.randint(0, 1440))
    txn = {
        "external_transaction_id": f"TXN-BT-LEG-{idx:05d}",
        "account_id": f"ACC-{_RNG.randint(1, 50):05d}",
        "customer_id": f"CUST-{_RNG.randint(1, 50):05d}",
        "merchant_category_code": _RNG.choice(["5411", "5812", "5541", "5311", "5651"]),
        "transaction_amount": amount,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": _RNG.choice(["online", "pos", "mobile"]),
        "card_type": "credit",
        "device_id": f"device-{_RNG.randint(1, 20)}",
        "geo_latitude": 40.7128 + _RNG.uniform(-0.5, 0.5),
        "geo_longitude": -74.006 + _RNG.uniform(-0.5, 0.5),
        "geo_country": "US",
        "is_international": False,
        "transaction_timestamp": ts,
    }
    ctx: Dict[str, Any] = {
        "customer_avg_amount": round(_RNG.uniform(50, 300), 2),
        "customer_history_count": _RNG.randint(20, 200),
        "recent_transactions": [],
        "is_new_device": False,
        "device_age_days": _RNG.uniform(30, 365),
        "accounts_on_device": 1,
        "is_domestic_only": False,
        "days_since_last_transaction": _RNG.uniform(0, 5),
        "customer_mcc_distribution": {"5411": 0.4, "5812": 0.3, "5541": 0.2, "5311": 0.1},
        "customer_channels": {"online", "pos", "mobile"},
    }
    return txn, ctx


def _generate_fraudulent(idx: int, base_ts: datetime) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate a transaction that should trigger at least one rule."""
    pattern = _RNG.choice(
        [
            "high_amount",
            "structuring",
            "rapid",
            "impossible_travel",
            "new_device",
            "late_night",
            "high_risk_country",
            "card_testing",
            "dormant",
        ]
    )

    base_amount = round(_RNG.uniform(50, 200), 2)
    ts_offset = _RNG.randint(0, 1440)
    ts = _random_ts(base_ts, ts_offset)

    txn: Dict[str, Any] = {
        "external_transaction_id": f"TXN-BT-FRD-{idx:05d}",
        "account_id": f"ACC-{_RNG.randint(51, 60):05d}",
        "customer_id": f"CUST-{_RNG.randint(51, 60):05d}",
        "merchant_category_code": "5411",
        "transaction_amount": base_amount,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "card_type": "credit",
        "device_id": f"device-{_RNG.randint(1, 20)}",
        "geo_latitude": 40.7128,
        "geo_longitude": -74.006,
        "geo_country": "US",
        "is_international": False,
        "transaction_timestamp": ts,
    }
    ctx: Dict[str, Any] = {
        "customer_avg_amount": 100.0,
        "customer_history_count": 50,
        "recent_transactions": [],
        "is_new_device": False,
        "device_age_days": 30.0,
        "accounts_on_device": 1,
        "is_domestic_only": False,
        "days_since_last_transaction": 1.0,
        "customer_mcc_distribution": {"5411": 0.5, "5812": 0.3, "5541": 0.2},
        "customer_channels": {"online", "pos"},
    }

    if pattern == "high_amount":
        txn["transaction_amount"] = round(_RNG.uniform(2000, 10000), 2)
        ctx["customer_avg_amount"] = 100.0

    elif pattern == "structuring":
        txn["transaction_amount"] = round(_RNG.uniform(9001, 9999), 2)

    elif pattern == "rapid":
        recent = []
        for i in range(6):
            rt_ts = _random_ts(base_ts, ts_offset - _RNG.randint(1, 9))
            recent.append(
                {
                    "transaction_timestamp": rt_ts,
                    "transaction_amount": round(_RNG.uniform(10, 100), 2),
                }
            )
        ctx["recent_transactions"] = recent

    elif pattern == "impossible_travel":
        txn["geo_latitude"] = 51.5074  # London
        txn["geo_longitude"] = -0.1278
        txn["geo_country"] = "GB"
        txn["is_international"] = True
        last_ts = _random_ts(base_ts, ts_offset - 60)  # 1 hour ago
        ctx["last_transaction"] = {
            "geo_latitude": 40.7128,
            "geo_longitude": -74.006,
            "transaction_timestamp": last_ts,
        }

    elif pattern == "new_device":
        txn["transaction_amount"] = round(_RNG.uniform(1500, 5000), 2)
        ctx["is_new_device"] = True
        ctx["device_age_days"] = 0.0

    elif pattern == "late_night":
        late_ts = base_ts.replace(hour=_RNG.randint(0, 4), minute=_RNG.randint(0, 59))
        txn["transaction_timestamp"] = late_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        txn["transaction_amount"] = round(_RNG.uniform(2500, 8000), 2)

    elif pattern == "high_risk_country":
        txn["geo_country"] = _RNG.choice(["RU", "NG", "KP", "IR"])
        txn["is_international"] = True

    elif pattern == "card_testing":
        txn["transaction_amount"] = round(_RNG.uniform(500, 3000), 2)
        recent = []
        for i in range(4):
            rt_ts = _random_ts(base_ts, ts_offset - _RNG.randint(1, 25))
            recent.append(
                {
                    "transaction_timestamp": rt_ts,
                    "transaction_amount": round(_RNG.uniform(0.5, 4.0), 2),
                }
            )
        ctx["recent_transactions"] = recent

    elif pattern == "dormant":
        ctx["days_since_last_transaction"] = _RNG.uniform(91, 365)

    return txn, ctx


def generate_backtest_dataset(
    n_legitimate: int = 800,
    n_fraudulent: int = 200,
) -> Tuple[List[Dict[str, Any]], List[bool], List[Dict[str, Any]]]:
    base_ts = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    transactions: List[Dict[str, Any]] = []
    labels: List[bool] = []
    contexts: List[Dict[str, Any]] = []

    for i in range(n_legitimate):
        txn, ctx = _generate_legitimate(i, base_ts)
        transactions.append(txn)
        labels.append(False)
        contexts.append(ctx)

    for i in range(n_fraudulent):
        txn, ctx = _generate_fraudulent(i, base_ts)
        transactions.append(txn)
        labels.append(True)
        contexts.append(ctx)

    combined = list(zip(transactions, labels, contexts))
    _RNG.shuffle(combined)
    transactions, labels, contexts = zip(*combined)  # type: ignore[assignment]
    return list(transactions), list(labels), list(contexts)


# ── CLI ──────────────────────────────────────────────────────────────


def _load_jsonl(path: str) -> Tuple[List[Dict], List[bool], List[Optional[Dict]]]:
    transactions, labels, contexts = [], [], []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            transactions.append(obj["transaction"])
            labels.append(bool(obj["is_fraud"]))
            contexts.append(obj.get("context"))
    return transactions, labels, contexts


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest fraud detection rules")
    parser.add_argument("--data", type=str, help="Path to labelled JSONL file")
    parser.add_argument("--rules", type=str, help="Path to fraud_rules.yaml")
    parser.add_argument("--categories", nargs="*", help="Filter rule categories")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    parser.add_argument("--n-legit", type=int, default=800, help="Synthetic legitimate count")
    parser.add_argument("--n-fraud", type=int, default=200, help="Synthetic fraudulent count")
    args = parser.parse_args()

    engine = FraudRuleEngine(rules_path=args.rules) if args.rules else FraudRuleEngine()

    if args.data:
        transactions, labels, contexts = _load_jsonl(args.data)
        print(f"Loaded {len(transactions)} labelled transactions from {args.data}")
    else:
        transactions, labels, contexts = generate_backtest_dataset(args.n_legit, args.n_fraud)
        print(
            f"Generated {len(transactions)} synthetic transactions "
            f"({args.n_legit} legit, {args.n_fraud} fraud)"
        )

    start = time.perf_counter()
    metrics = engine.backtest(transactions, labels, contexts)
    elapsed = time.perf_counter() - start

    if args.categories:
        rule_ids_for_cats = {r.id for r in engine.rules if r.category in args.categories}
        metrics = {k: v for k, v in metrics.items() if k in rule_ids_for_cats}

    if args.json:
        output = {
            "elapsed_seconds": round(elapsed, 3),
            "total_transactions": len(transactions),
            "total_fraud": sum(labels),
            "rules": {k: v.to_dict() for k, v in metrics.items()},
        }
        print(json.dumps(output, indent=2))
        return

    # Table output
    print(f"\nBacktest completed in {elapsed:.2f}s\n")
    print(
        f"{'Rule ID':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} "
        f"{'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}"
    )
    print("-" * 86)

    for rule_id in sorted(metrics.keys()):
        m = metrics[rule_id]
        if m.total_evaluated == 0:
            continue
        print(
            f"{rule_id:<20} {m.precision:>10.4f} {m.recall:>10.4f} {m.f1_score:>10.4f} "
            f"{m.true_positives:>6} {m.false_positives:>6} "
            f"{m.false_negatives:>6} {m.true_negatives:>6}"
        )

    # Aggregate false positive rate
    total_fp = sum(m.false_positives for m in metrics.values())
    total_legit = sum(1 for lbl in labels if not lbl)
    fp_rate = total_fp / total_legit if total_legit else 0.0
    total_tp = sum(m.true_positives for m in metrics.values())
    total_fraud = sum(labels)
    detection_rate = total_tp / total_fraud if total_fraud else 0.0

    print(f"\nAggregate FP rate: {fp_rate:.2%}  |  Detection rate: {detection_rate:.2%}")
    print(f"Avg evaluation time: {engine.metrics.avg_evaluation_time_ms:.3f} ms")


if __name__ == "__main__":
    main()
