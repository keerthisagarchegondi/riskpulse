"""System-wide constants for RiskPulse platform."""

# Application
APP_NAME = "RiskPulse"
APP_VERSION = "0.1.0"

# Transaction Types
TRANSACTION_TYPES = ("purchase", "withdrawal", "transfer", "refund")

# Channels
CHANNELS = ("online", "pos", "atm", "mobile")

# Card Types
CARD_TYPES = ("credit", "debit", "prepaid")

# Transaction Status
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_DECLINED = "declined"
STATUS_FLAGGED = "flagged"

# Alert Severity
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"

# Alert Status
ALERT_OPEN = "open"
ALERT_INVESTIGATING = "investigating"
ALERT_RESOLVED = "resolved"
ALERT_FALSE_POSITIVE = "false_positive"

# Risk Tiers
RISK_TIER_LOW = "low"
RISK_TIER_STANDARD = "standard"
RISK_TIER_ELEVATED = "elevated"
RISK_TIER_HIGH = "high"

# Kafka Topics
TOPIC_RAW_EVENTS = "txn.raw.events"
TOPIC_VALIDATED = "txn.validated"
TOPIC_ENRICHED = "txn.enriched"
TOPIC_SCORED = "txn.scored"
TOPIC_FRAUD_ALERTS = "fraud.alerts"
TOPIC_DLQ = "fraud.dlq"
TOPIC_METRICS = "system.metrics"
TOPIC_AUDIT = "audit.events"

# S3 Paths
S3_RAW_PREFIX = "transactions"
S3_PROCESSED_PREFIX = "processed"
S3_MODELS_PREFIX = "models"
S3_ARCHIVE_PREFIX = "archive"

# Scoring Thresholds (defaults)
SCORE_THRESHOLD_LOW = 0.3
SCORE_THRESHOLD_MEDIUM = 0.5
SCORE_THRESHOLD_HIGH = 0.8
SCORE_THRESHOLD_CRITICAL = 0.95

# Rate Limiting
DEFAULT_RATE_LIMIT = 100  # requests per minute
DEFAULT_BURST_SIZE = 20

# Supported Currencies
SUPPORTED_CURRENCIES = ("USD", "EUR", "GBP", "CAD", "AUD", "JPY")

# High-Risk Countries (ISO 3166-1 alpha-2)
HIGH_RISK_COUNTRIES = ("NG", "RU", "UA", "VN", "PH", "ID", "BR", "MX")
