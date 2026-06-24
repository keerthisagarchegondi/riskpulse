"""Generate synthetic transaction data for testing, development, and load testing.

Supports multiple output modes:
- File: Save transactions as JSON for testing fixtures
- Kafka: Publish directly to Kafka for integration/load testing
- Stdout: Stream JSON lines for piping

Features realistic fraud patterns including:
- Velocity attacks (rapid transactions)
- Geographic anomalies (impossible travel)
- High-value transactions from suspicious merchants
- Card testing patterns (small amounts, many attempts)
- Account takeover patterns (device/IP changes)

Usage:
    # Generate to file
    python scripts/generate_test_data.py --count 1000 --output file

    # Publish to Kafka
    python scripts/generate_test_data.py --count 5000 --output kafka --rate 1000

    # Load test (1000+ events/second)
    python scripts/generate_test_data.py --count 10000 --output kafka --rate 2000 --load-test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================================
# Reference Data
# ============================================================================

MERCHANTS = [
    ("MERCH-001", "Amazon", "5411", "US"),
    ("MERCH-002", "Walmart", "5411", "US"),
    ("MERCH-003", "Shell Gas Station", "5541", "US"),
    ("MERCH-004", "Netflix", "4899", "US"),
    ("MERCH-005", "Uber Technologies", "4121", "US"),
    ("MERCH-006", "Starbucks Coffee", "5812", "US"),
    ("MERCH-007", "Apple Store", "5732", "US"),
    ("MERCH-008", "Target", "5311", "US"),
    ("MERCH-009", "Home Depot", "5211", "US"),
    ("MERCH-010", "Costco Wholesale", "5300", "US"),
    ("MERCH-011", "Delta Airlines", "3058", "US"),
    ("MERCH-012", "Marriott Hotels", "3501", "US"),
    # Suspicious/high-risk merchants
    ("MERCH-090", "Unknown Digital Store", "7995", "RU"),
    ("MERCH-091", "Wire Transfer Services", "6012", "NG"),
    ("MERCH-092", "Crypto Exchange Ltd", "6051", "KY"),
    ("MERCH-093", "Online Casino", "7995", "CW"),
    ("MERCH-094", "Gift Card Warehouse", "5947", "CN"),
]

LOCATIONS = [
    # US - Legitimate
    ("US", "New York", 40.7128, -74.0060),
    ("US", "Los Angeles", 34.0522, -118.2437),
    ("US", "Chicago", 41.8781, -87.6298),
    ("US", "Houston", 29.7604, -95.3698),
    ("US", "Phoenix", 33.4484, -112.0740),
    ("US", "San Francisco", 37.7749, -122.4194),
    ("US", "Miami", 25.7617, -80.1918),
    # International - Legitimate
    ("GB", "London", 51.5074, -0.1278),
    ("CA", "Toronto", 43.6532, -79.3832),
    ("DE", "Berlin", 52.5200, 13.4050),
    # High-risk locations
    ("RU", "Moscow", 55.7558, 37.6173),
    ("NG", "Lagos", 6.5244, 3.3792),
    ("CN", "Beijing", 39.9042, 116.4074),
    ("RO", "Bucharest", 44.4268, 26.1025),
]

CHANNELS = ["online", "pos", "atm", "mobile"]
CARD_TYPES = ["credit", "debit", "prepaid"]
TRANSACTION_TYPES = ["purchase", "withdrawal", "transfer", "refund"]
DEVICE_TYPES = ["mobile", "desktop", "tablet", "pos"]

# Amount distribution parameters for realistic patterns
AMOUNT_PROFILES = {
    "daily_small": (5, 50),         # Coffee, lunch
    "daily_medium": (20, 150),      # Groceries, gas
    "weekly_large": (100, 500),     # Shopping, dining
    "monthly_big": (500, 2000),     # Bills, electronics
    "rare_high": (2000, 10000),     # Travel, luxury
}


# ============================================================================
# Customer Profiles
# ============================================================================

class CustomerProfile:
    """Represents a synthetic customer with consistent behavior patterns."""

    def __init__(self, customer_id: str | None = None) -> None:
        self.customer_id = customer_id or f"CUST-{random.randint(10000, 99999)}"
        self.account_id = f"ACC-{self.customer_id.split('-')[1]}"
        self.home_location = random.choice(LOCATIONS[:7])  # US locations
        self.preferred_merchants = random.sample(MERCHANTS[:12], k=random.randint(3, 7))
        self.card_type = random.choice(CARD_TYPES)
        self.card_last_four = f"{random.randint(1000, 9999)}"
        self.device_id = f"device-{uuid.uuid4().hex[:8]}"
        self.device_type = random.choice(["mobile", "desktop"])
        self.avg_monthly_spend = random.uniform(1000, 8000)
        self.typical_hours = (random.randint(7, 10), random.randint(20, 23))


# ============================================================================
# Transaction Generator
# ============================================================================

def generate_transaction(
    customer: CustomerProfile | None = None,
    is_fraud: bool = False,
    fraud_pattern: str | None = None,
    base_timestamp: datetime | None = None,
) -> dict:
    """Generate a single synthetic transaction.

    Args:
        customer: Customer profile for consistent behavior.
        is_fraud: Whether to generate a fraudulent transaction.
        fraud_pattern: Specific fraud pattern to use.
        base_timestamp: Base time for the transaction.

    Returns:
        Transaction dictionary matching the Avro schema.
    """
    if customer is None:
        customer = CustomerProfile()

    base_time = base_timestamp or datetime.now(timezone.utc)

    if is_fraud:
        return _generate_fraud_transaction(customer, fraud_pattern, base_time)
    return _generate_legit_transaction(customer, base_time)


def _generate_legit_transaction(
    customer: CustomerProfile, base_time: datetime
) -> dict:
    """Generate a realistic legitimate transaction."""
    merchant = random.choice(customer.preferred_merchants)

    # Time within typical hours
    hour = random.randint(*customer.typical_hours)
    timestamp = base_time - timedelta(
        hours=random.randint(0, 48),
        minutes=random.randint(0, 59),
    )
    timestamp = timestamp.replace(hour=hour, minute=random.randint(0, 59))

    # Amount follows spending profile
    profile = random.choices(
        list(AMOUNT_PROFILES.keys()),
        weights=[0.3, 0.3, 0.2, 0.15, 0.05],
        k=1,
    )[0]
    low, high = AMOUNT_PROFILES[profile]
    amount = round(random.uniform(low, high), 2)

    # Location near home
    location = customer.home_location
    lat_jitter = random.uniform(-0.1, 0.1)
    lon_jitter = random.uniform(-0.1, 0.1)

    return {
        "external_transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
        "account_id": customer.account_id,
        "customer_id": customer.customer_id,
        "merchant_id": merchant[0],
        "merchant_name": merchant[1],
        "merchant_category_code": merchant[2],
        "transaction_amount": amount,
        "transaction_currency": "USD",
        "transaction_type": random.choices(
            TRANSACTION_TYPES, weights=[0.7, 0.1, 0.1, 0.1], k=1
        )[0],
        "channel": random.choices(
            CHANNELS, weights=[0.4, 0.3, 0.1, 0.2], k=1
        )[0],
        "card_type": customer.card_type,
        "card_last_four": customer.card_last_four,
        "ip_address": _generate_ip(domestic=True),
        "device_id": customer.device_id,
        "device_type": customer.device_type,
        "geo_latitude": location[2] + lat_jitter,
        "geo_longitude": location[3] + lon_jitter,
        "geo_country": location[0],
        "geo_city": location[1],
        "is_international": False,
        "transaction_timestamp": timestamp.isoformat(),
    }


def _generate_fraud_transaction(
    customer: CustomerProfile,
    pattern: str | None,
    base_time: datetime,
) -> dict:
    """Generate a fraudulent transaction with specific attack patterns."""
    patterns = [
        "high_value",
        "velocity_attack",
        "geo_anomaly",
        "card_testing",
        "account_takeover",
    ]
    fraud_type = pattern or random.choice(patterns)

    timestamp = base_time - timedelta(
        minutes=random.randint(0, 120),
    )

    if fraud_type == "high_value":
        return _fraud_high_value(customer, timestamp)
    elif fraud_type == "velocity_attack":
        return _fraud_velocity(customer, timestamp)
    elif fraud_type == "geo_anomaly":
        return _fraud_geo_anomaly(customer, timestamp)
    elif fraud_type == "card_testing":
        return _fraud_card_testing(customer, timestamp)
    else:  # account_takeover
        return _fraud_account_takeover(customer, timestamp)


def _fraud_high_value(customer: CustomerProfile, timestamp: datetime) -> dict:
    """High-value purchase from suspicious merchant at unusual time."""
    merchant = random.choice(MERCHANTS[12:])  # Suspicious merchants
    amount = round(random.uniform(3000, 9999), 2)

    return {
        "external_transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
        "account_id": customer.account_id,
        "customer_id": customer.customer_id,
        "merchant_id": merchant[0],
        "merchant_name": merchant[1],
        "merchant_category_code": merchant[2],
        "transaction_amount": amount,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "card_type": customer.card_type,
        "card_last_four": customer.card_last_four,
        "ip_address": _generate_ip(domestic=False),
        "device_id": f"device-{uuid.uuid4().hex[:8]}",  # New device
        "device_type": "desktop",
        "geo_latitude": random.choice(LOCATIONS[10:])[2],
        "geo_longitude": random.choice(LOCATIONS[10:])[3],
        "geo_country": random.choice(["RU", "NG", "CN", "RO"]),
        "geo_city": random.choice(["Moscow", "Lagos", "Beijing", "Bucharest"]),
        "is_international": True,
        "transaction_timestamp": timestamp.replace(
            hour=random.randint(1, 5)
        ).isoformat(),
    }


def _fraud_velocity(customer: CustomerProfile, timestamp: datetime) -> dict:
    """Rapid succession transaction (part of velocity attack)."""
    merchant = random.choice(MERCHANTS[:12])
    amount = round(random.uniform(100, 999), 2)

    return {
        "external_transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
        "account_id": customer.account_id,
        "customer_id": customer.customer_id,
        "merchant_id": merchant[0],
        "merchant_name": merchant[1],
        "merchant_category_code": merchant[2],
        "transaction_amount": amount,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "card_type": customer.card_type,
        "card_last_four": customer.card_last_four,
        "ip_address": _generate_ip(domestic=False),
        "device_id": customer.device_id,
        "device_type": customer.device_type,
        "geo_latitude": customer.home_location[2],
        "geo_longitude": customer.home_location[3],
        "geo_country": customer.home_location[0],
        "geo_city": customer.home_location[1],
        "is_international": False,
        "transaction_timestamp": timestamp.isoformat(),
    }


def _fraud_geo_anomaly(customer: CustomerProfile, timestamp: datetime) -> dict:
    """Transaction from impossible geographic location (impossible travel)."""
    # Far from home location
    far_location = random.choice(LOCATIONS[10:])

    return {
        "external_transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
        "account_id": customer.account_id,
        "customer_id": customer.customer_id,
        "merchant_id": random.choice(MERCHANTS[:12])[0],
        "merchant_name": random.choice(MERCHANTS[:12])[1],
        "merchant_category_code": random.choice(MERCHANTS[:12])[2],
        "transaction_amount": round(random.uniform(200, 2000), 2),
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "pos",
        "card_type": customer.card_type,
        "card_last_four": customer.card_last_four,
        "ip_address": _generate_ip(domestic=False),
        "device_id": customer.device_id,
        "device_type": "pos",
        "geo_latitude": far_location[2],
        "geo_longitude": far_location[3],
        "geo_country": far_location[0],
        "geo_city": far_location[1],
        "is_international": True,
        "transaction_timestamp": timestamp.isoformat(),
    }


def _fraud_card_testing(customer: CustomerProfile, timestamp: datetime) -> dict:
    """Small test transaction (card testing pattern)."""
    return {
        "external_transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
        "account_id": customer.account_id,
        "customer_id": customer.customer_id,
        "merchant_id": random.choice(MERCHANTS[12:])[0],
        "merchant_name": random.choice(MERCHANTS[12:])[1],
        "merchant_category_code": random.choice(MERCHANTS[12:])[2],
        "transaction_amount": round(random.uniform(0.50, 5.00), 2),
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "card_type": customer.card_type,
        "card_last_four": customer.card_last_four,
        "ip_address": _generate_ip(domestic=False),
        "device_id": f"device-{uuid.uuid4().hex[:8]}",
        "device_type": "desktop",
        "geo_latitude": random.choice(LOCATIONS[10:])[2],
        "geo_longitude": random.choice(LOCATIONS[10:])[3],
        "geo_country": random.choice(["RU", "NG", "RO"]),
        "geo_city": "Unknown",
        "is_international": True,
        "transaction_timestamp": timestamp.isoformat(),
    }


def _fraud_account_takeover(customer: CustomerProfile, timestamp: datetime) -> dict:
    """Legitimate-looking transaction from new device/IP (account takeover)."""
    merchant = random.choice(customer.preferred_merchants)

    return {
        "external_transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
        "account_id": customer.account_id,
        "customer_id": customer.customer_id,
        "merchant_id": merchant[0],
        "merchant_name": merchant[1],
        "merchant_category_code": merchant[2],
        "transaction_amount": round(random.uniform(500, 3000), 2),
        "transaction_currency": "USD",
        "transaction_type": "transfer",
        "channel": "online",
        "card_type": customer.card_type,
        "card_last_four": customer.card_last_four,
        "ip_address": _generate_ip(domestic=False),
        "device_id": f"device-{uuid.uuid4().hex[:8]}",  # New unknown device
        "device_type": random.choice(["desktop", "mobile"]),
        "geo_latitude": customer.home_location[2] + random.uniform(-2, 2),
        "geo_longitude": customer.home_location[3] + random.uniform(-2, 2),
        "geo_country": customer.home_location[0],
        "geo_city": customer.home_location[1],
        "is_international": False,
        "transaction_timestamp": timestamp.replace(
            hour=random.randint(0, 5)
        ).isoformat(),
    }


def _generate_ip(domestic: bool = True) -> str:
    """Generate a random IP address."""
    if domestic:
        # Common US IP ranges
        first_octet = random.choice([24, 50, 64, 68, 71, 72, 96, 98, 108, 174])
    else:
        # Foreign/suspicious ranges
        first_octet = random.choice([5, 31, 37, 46, 77, 85, 91, 185, 195, 212])
    return f"{first_octet}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


# ============================================================================
# Dataset Generation
# ============================================================================

def generate_dataset(
    n_transactions: int = 1000,
    fraud_rate: float = 0.02,
    n_customers: int | None = None,
) -> list[dict]:
    """Generate a dataset of synthetic transactions with realistic patterns.

    Creates a population of customers with consistent profiles and generates
    transactions following realistic spending patterns.

    Args:
        n_transactions: Total number of transactions to generate.
        fraud_rate: Proportion of fraudulent transactions (0.0-1.0).
        n_customers: Number of unique customers. Defaults to n_transactions // 5.

    Returns:
        List of transaction dictionaries.
    """
    n_customers = n_customers or max(n_transactions // 5, 10)
    customers = [CustomerProfile() for _ in range(n_customers)]

    n_fraud = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_fraud

    transactions = []

    # Generate legitimate transactions
    for _ in range(n_legit):
        customer = random.choice(customers)
        transactions.append(generate_transaction(customer=customer, is_fraud=False))

    # Generate fraudulent transactions with varied patterns
    fraud_patterns = ["high_value", "velocity_attack", "geo_anomaly", "card_testing", "account_takeover"]
    for i in range(n_fraud):
        customer = random.choice(customers)
        pattern = fraud_patterns[i % len(fraud_patterns)]
        transactions.append(
            generate_transaction(customer=customer, is_fraud=True, fraud_pattern=pattern)
        )

    random.shuffle(transactions)
    return transactions


def stream_transactions(
    rate_per_second: float = 100,
    fraud_rate: float = 0.02,
    n_customers: int = 200,
) -> Generator[dict, None, None]:
    """Generate an infinite stream of transactions at a specified rate.

    Yields transactions with realistic timing. Useful for continuous
    load testing and Kafka publishing.

    Args:
        rate_per_second: Target events per second.
        fraud_rate: Proportion of fraudulent transactions.
        n_customers: Customer pool size.

    Yields:
        Transaction dictionaries.
    """
    customers = [CustomerProfile() for _ in range(n_customers)]
    interval = 1.0 / rate_per_second

    while True:
        customer = random.choice(customers)
        is_fraud = random.random() < fraud_rate
        yield generate_transaction(customer=customer, is_fraud=is_fraud)
        time.sleep(interval)


# ============================================================================
# Kafka Publishing
# ============================================================================

def publish_to_kafka(
    transactions: list[dict],
    rate_limit: float | None = None,
    bootstrap_servers: str = "localhost:9092",
) -> dict:
    """Publish transactions to Kafka via the TransactionProducer.

    Args:
        transactions: List of transaction dictionaries.
        rate_limit: Max events per second (None for unlimited).
        bootstrap_servers: Kafka broker address.

    Returns:
        Summary dict with produced/failed counts and timing.
    """
    from src.ingestion.kafka_producer import TransactionProducer

    start_time = time.time()
    interval = 1.0 / rate_limit if rate_limit else 0

    with TransactionProducer(bootstrap_servers=bootstrap_servers) as producer:
        produced = 0
        failed = 0

        for i, txn in enumerate(transactions):
            try:
                producer.produce(txn)
                produced += 1
            except Exception as e:
                failed += 1
                if failed <= 5:  # Only log first few errors
                    print(f"  Error producing message {i}: {e}", file=sys.stderr)

            # Rate limiting
            if interval > 0 and (i + 1) % 10 == 0:
                time.sleep(interval * 10)

            # Progress reporting
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Progress: {i + 1}/{len(transactions)} "
                      f"({rate:.0f} events/sec)")

        # Flush remaining
        remaining = producer.flush(timeout=30.0)
        elapsed = time.time() - start_time

        metrics = producer.metrics.snapshot()

    return {
        "produced": produced,
        "failed": failed,
        "remaining_in_buffer": remaining,
        "elapsed_seconds": round(elapsed, 2),
        "events_per_second": round(produced / elapsed, 1) if elapsed > 0 else 0,
        "producer_metrics": metrics,
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic transaction data for RiskPulse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --count 1000 --output file
  %(prog)s --count 5000 --output kafka --rate 1000
  %(prog)s --count 10000 --output kafka --rate 2000 --load-test
  %(prog)s --count 100 --output stdout --fraud-rate 0.1
        """,
    )
    parser.add_argument(
        "--count", "-n", type=int, default=1000,
        help="Number of transactions to generate (default: 1000)",
    )
    parser.add_argument(
        "--output", "-o", choices=["file", "kafka", "stdout"], default="file",
        help="Output destination (default: file)",
    )
    parser.add_argument(
        "--fraud-rate", "-f", type=float, default=0.02,
        help="Fraud rate as decimal (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--customers", type=int, default=None,
        help="Number of unique customers (default: count/5)",
    )
    parser.add_argument(
        "--rate", "-r", type=float, default=None,
        help="Rate limit for Kafka publishing (events/second)",
    )
    parser.add_argument(
        "--bootstrap-servers", default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)",
    )
    parser.add_argument(
        "--output-file", default="tests/fixtures/generated_transactions.json",
        help="Output file path (for file mode)",
    )
    parser.add_argument(
        "--load-test", action="store_true",
        help="Run in load test mode with performance reporting",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"{'=' * 60}")
    print(f"RiskPulse Synthetic Transaction Generator")
    print(f"{'=' * 60}")
    print(f"  Count:      {args.count:,}")
    print(f"  Fraud rate: {args.fraud_rate:.1%}")
    print(f"  Output:     {args.output}")
    if args.rate:
        print(f"  Rate limit: {args.rate:,.0f} events/sec")
    print(f"{'=' * 60}")
    print()

    # Generate dataset
    print("Generating transactions...")
    gen_start = time.time()
    dataset = generate_dataset(
        n_transactions=args.count,
        fraud_rate=args.fraud_rate,
        n_customers=args.customers,
    )
    gen_elapsed = time.time() - gen_start
    print(f"  Generated {len(dataset):,} transactions in {gen_elapsed:.2f}s")

    # Count fraud indicators
    high_value = sum(1 for t in dataset if t["transaction_amount"] > 3000)
    international = sum(1 for t in dataset if t["is_international"])
    print(f"  High-value (>$3000): {high_value}")
    print(f"  International: {international}")
    print()

    # Output
    if args.output == "file":
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2, default=str)
        print(f"Saved to {output_path}")

    elif args.output == "kafka":
        print("Publishing to Kafka...")
        result = publish_to_kafka(
            transactions=dataset,
            rate_limit=args.rate,
            bootstrap_servers=args.bootstrap_servers,
        )
        print(f"\nKafka Publishing Results:")
        print(f"  Produced:    {result['produced']:,}")
        print(f"  Failed:      {result['failed']:,}")
        print(f"  Throughput:  {result['events_per_second']:,.1f} events/sec")
        print(f"  Duration:    {result['elapsed_seconds']:.2f}s")
        if result['remaining_in_buffer'] > 0:
            print(f"  WARNING: {result['remaining_in_buffer']} messages not delivered!")

        if args.load_test:
            metrics = result['producer_metrics']
            print(f"\nLoad Test Metrics:")
            print(f"  Avg latency:  {metrics['average_latency_ms']:.2f}ms")
            print(f"  Error rate:   {metrics['error_rate']:.4f}")
            print(f"  Bytes sent:   {metrics['bytes_produced']:,}")

            # Verify throughput target
            if result['events_per_second'] >= 1000:
                print(f"\n  ✓ PASS: Throughput target met "
                      f"({result['events_per_second']:.0f} >= 1000 events/sec)")
            else:
                print(f"\n  ✗ FAIL: Throughput below target "
                      f"({result['events_per_second']:.0f} < 1000 events/sec)")
                sys.exit(1)

    elif args.output == "stdout":
        for txn in dataset:
            print(json.dumps(txn, default=str))

    print(f"\nDone.")


if __name__ == "__main__":
    main()
