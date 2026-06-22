"""Generate synthetic transaction data for testing and development."""

import json
import random
import uuid
from datetime import datetime, timedelta, timezone


MERCHANTS = [
    ("MERCH-001", "Amazon", "5411"),
    ("MERCH-002", "Walmart", "5411"),
    ("MERCH-003", "Shell Gas", "5541"),
    ("MERCH-004", "Netflix", "4899"),
    ("MERCH-005", "Uber", "4121"),
    ("MERCH-006", "Starbucks", "5812"),
    ("MERCH-007", "Apple Store", "5732"),
    ("MERCH-008", "Unknown Merchant", "7995"),
    ("MERCH-009", "Wire Transfer Co", "6012"),
    ("MERCH-010", "Crypto Exchange", "6051"),
]

COUNTRIES = [
    ("US", "New York", 40.7128, -74.0060),
    ("US", "Los Angeles", 34.0522, -118.2437),
    ("US", "Chicago", 41.8781, -87.6298),
    ("GB", "London", 51.5074, -0.1278),
    ("RU", "Moscow", 55.7558, 37.6173),
    ("NG", "Lagos", 6.5244, 3.3792),
    ("CN", "Beijing", 39.9042, 116.4074),
]

CHANNELS = ["online", "pos", "atm", "mobile"]
CARD_TYPES = ["credit", "debit", "prepaid"]
TRANSACTION_TYPES = ["purchase", "withdrawal", "transfer", "refund"]


def generate_transaction(
    customer_id: str | None = None,
    is_fraud: bool = False,
) -> dict:
    """Generate a single synthetic transaction."""
    cust_id = customer_id or f"CUST-{random.randint(10000, 99999)}"
    account_id = f"ACC-{cust_id.split('-')[1]}"

    if is_fraud:
        # Fraud patterns
        amount = random.choice([
            random.uniform(5000, 9999),  # High amount
            9999.00,  # Just below reporting threshold
            random.uniform(1000, 3000),  # Moderate but unusual
        ])
        merchant = random.choice(MERCHANTS[7:])  # Suspicious merchants
        location = random.choice(COUNTRIES[4:])  # High-risk countries
        hour = random.randint(1, 5)  # Late night
    else:
        amount = random.uniform(5, 500)
        merchant = random.choice(MERCHANTS[:7])
        location = random.choice(COUNTRIES[:3])  # US locations
        hour = random.randint(8, 22)

    timestamp = datetime.now(timezone.utc) - timedelta(
        hours=random.randint(0, 72),
        minutes=random.randint(0, 59),
    )
    timestamp = timestamp.replace(hour=hour)

    return {
        "external_transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
        "account_id": account_id,
        "customer_id": cust_id,
        "merchant_id": merchant[0],
        "merchant_name": merchant[1],
        "merchant_category_code": merchant[2],
        "transaction_amount": round(amount, 2),
        "transaction_currency": "USD",
        "transaction_type": random.choice(TRANSACTION_TYPES),
        "channel": random.choice(CHANNELS),
        "card_type": random.choice(CARD_TYPES),
        "card_last_four": f"{random.randint(1000, 9999)}",
        "ip_address": f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}",
        "device_id": f"device-{uuid.uuid4().hex[:8]}",
        "device_type": random.choice(["mobile", "desktop", "tablet", "pos"]),
        "geo_latitude": location[2],
        "geo_longitude": location[3],
        "geo_country": location[0],
        "geo_city": location[1],
        "is_international": location[0] != "US",
        "transaction_timestamp": timestamp.isoformat(),
    }


def generate_dataset(
    n_transactions: int = 1000,
    fraud_rate: float = 0.02,
) -> list[dict]:
    """Generate a dataset of synthetic transactions."""
    transactions = []
    n_fraud = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_fraud

    # Generate legitimate transactions
    customers = [f"CUST-{random.randint(10000, 99999)}" for _ in range(n_legit // 5)]
    for _ in range(n_legit):
        cust = random.choice(customers)
        transactions.append(generate_transaction(customer_id=cust, is_fraud=False))

    # Generate fraudulent transactions
    for _ in range(n_fraud):
        transactions.append(generate_transaction(is_fraud=True))

    random.shuffle(transactions)
    return transactions


if __name__ == "__main__":
    print("Generating synthetic transaction data...")
    dataset = generate_dataset(n_transactions=1000, fraud_rate=0.02)

    output_path = "tests/fixtures/generated_transactions.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2, default=str)

    fraud_count = sum(1 for t in dataset if t["is_international"] and t["transaction_amount"] > 1000)
    print(f"Generated {len(dataset)} transactions ({fraud_count} potential fraud)")
    print(f"Saved to {output_path}")
