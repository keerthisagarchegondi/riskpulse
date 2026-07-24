variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "project_name" {
  description = "Project name used as prefix for bucket naming"
  type        = string
  default     = "riskpulse"
}

variable "aws_region" {
  description = "Primary AWS region for S3 buckets"
  type        = string
  default     = "us-east-1"
}

variable "dr_region" {
  description = "Disaster recovery region for cross-region replication"
  type        = string
  default     = "us-west-2"
}

variable "enable_replication" {
  description = "Enable cross-region replication for disaster recovery"
  type        = bool
  default     = false
}

variable "kms_key_arn" {
  description = "ARN of the KMS key for SSE-KMS encryption on processed/models buckets"
  type        = string
  default     = ""
}

variable "create_kms_key" {
  description = "Create a new KMS key for S3 encryption if kms_key_arn is not provided"
  type        = bool
  default     = true
}

variable "kms_key_deletion_window" {
  description = "Number of days before KMS key deletion (7-30)"
  type        = number
  default     = 30

  validation {
    condition     = var.kms_key_deletion_window >= 7 && var.kms_key_deletion_window <= 30
    error_message = "KMS key deletion window must be between 7 and 30 days."
  }
}

variable "sqs_queue_arn" {
  description = "ARN of the SQS queue for S3 event notifications"
  type        = string
  default     = ""
}

variable "enable_event_notifications" {
  description = "Enable S3 event notifications to SQS"
  type        = bool
  default     = true
}

variable "raw_glacier_transition_days" {
  description = "Days before raw data transitions to Glacier"
  type        = number
  default     = 90
}

variable "processed_glacier_transition_days" {
  description = "Days before processed data transitions to Glacier"
  type        = number
  default     = 180
}

variable "archive_deep_archive_transition_days" {
  description = "Days before archive data transitions to Deep Archive"
  type        = number
  default     = 365
}

variable "enable_versioning" {
  description = "Enable versioning on all S3 buckets"
  type        = bool
  default     = true
}

variable "force_destroy" {
  description = "Allow Terraform to destroy non-empty buckets (use only in dev)"
  type        = bool
  default     = false
}

variable "allowed_account_ids" {
  description = "List of AWS account IDs allowed to access the buckets"
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "noncurrent_version_expiration_days" {
  description = "Days before noncurrent object versions are permanently deleted"
  type        = number
  default     = 365
}

variable "abort_incomplete_multipart_days" {
  description = "Days after which incomplete multipart uploads are aborted"
  type        = number
  default     = 7
}
