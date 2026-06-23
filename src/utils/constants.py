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

# High-Risk Countries (ISO 3166-1 alpha-3)
HIGH_RISK_COUNTRIES = ("NGA", "RUS", "UKR", "VNM", "PHL", "IDN", "BRA", "MEX")

# Merchant Category Codes (High-Risk)
HIGH_RISK_MCC = (
    "5967",  # Direct marketing — inbound teleservices
    "5966",  # Direct marketing — outbound teleservices
    "7995",  # Gambling/betting
    "5912",  # Drug stores/pharmacies
    "5962",  # Travel-related direct marketing
    "4829",  # Money transfer
    "6051",  # Non-financial institutions — foreign currency/crypto
    "6012",  # Financial institutions — merchandise and services
)

# Pipeline Configuration
BATCH_SIZE_DEFAULT = 100
MAX_BATCH_SIZE = 1000
POLL_TIMEOUT_MS = 1000
MAX_RETRIES = 3

# Velocity Thresholds (defaults)
VELOCITY_TXN_COUNT_1H = 10
VELOCITY_TXN_COUNT_24H = 50
VELOCITY_AMOUNT_24H = 10000.0
VELOCITY_UNIQUE_MERCHANTS_24H = 15
VELOCITY_UNIQUE_COUNTRIES_7D = 5

# Scoring Weights (defaults)
SCORING_WEIGHT_RULES = 0.3
SCORING_WEIGHT_ANOMALY = 0.3
SCORING_WEIGHT_ML = 0.4

# Time Windows (seconds)
DEDUP_WINDOW_SECONDS = 3600  # 1 hour
ALERT_DEDUP_WINDOW_SECONDS = 3600  # 1 hour
ESCALATION_TIMEOUT_SECONDS = 900  # 15 minutes

# Feature Engineering
IMPOSSIBLE_TRAVEL_SPEED_MPH = 500
RAPID_SUCCESSION_SECONDS = 60
AMOUNT_ZSCORE_THRESHOLD = 3.0

# Model Configuration
MODEL_PREDICTION_TIMEOUT_MS = 50
MODEL_CACHE_TTL_SECONDS = 60

# API Configuration
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
MAX_PAGE_SIZE = 200
DEFAULT_PAGE_SIZE = 50

# Health Check
HEALTH_CHECK_TIMEOUT_SECONDS = 5
DEPENDENCY_CHECK_INTERVAL_SECONDS = 30
