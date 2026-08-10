variable "environment" {
  description = "Environment name."
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "project_name" {
  description = "Project name used for IAM resource names."
  type        = string
  default     = "riskpulse"
}

variable "aws_region" {
  description = "Primary AWS region."
  type        = string
  default     = "us-east-1"
}

variable "raw_bucket_arn" {
  description = "ARN of the raw transactions bucket."
  type        = string
}

variable "processed_bucket_arn" {
  description = "ARN of the processed data bucket."
  type        = string
}

variable "models_bucket_arn" {
  description = "ARN of the model artifact bucket."
  type        = string
}

variable "kms_key_arn" {
  description = "KMS key ARN used for platform encryption."
  type        = string
}

variable "notification_topic_arn" {
  description = "SNS topic ARN for fraud and ops notifications."
  type        = string
  default     = ""
}

variable "ses_identity_arn" {
  description = "SES identity ARN used by notification services."
  type        = string
  default     = ""
}

variable "database_rotation_lambda_arn" {
  description = "Lambda ARN that performs database secret rotation."
  type        = string
  default     = ""
}

variable "secret_arns" {
  description = "Secrets Manager ARNs used by the platform."
  type        = map(string)
  default     = {}
}

variable "trusted_service_principals" {
  description = "AWS service principals allowed to assume workload roles."
  type        = map(list(string))
  default = {
    api       = ["ecs-tasks.amazonaws.com"]
    worker    = ["ecs-tasks.amazonaws.com"]
    airflow   = ["ecs-tasks.amazonaws.com", "airflow.amazonaws.com"]
    dashboard = ["ecs-tasks.amazonaws.com"]
    admin     = ["ec2.amazonaws.com"]
  }
}

variable "admin_principal_arns" {
  description = "IAM principals allowed to assume the admin role."
  type        = list(string)
  default     = []
}

variable "snowflake_external_id" {
  description = "External ID required for Snowflake storage integration trust."
  type        = string
  default     = ""
  sensitive   = true
}

variable "tags" {
  description = "Tags applied to IAM resources."
  type        = map(string)
  default     = {}
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  namespace   = "RiskPulse/Platform"
  tags = merge(var.tags, {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Module      = "iam"
  })

  raw_objects       = "${var.raw_bucket_arn}/*"
  processed_objects = "${var.processed_bucket_arn}/*"
  models_objects    = "${var.models_bucket_arn}/*"
  all_secret_arns   = length(var.secret_arns) > 0 ? values(var.secret_arns) : ["arn:aws:secretsmanager:*:*:secret:${var.project_name}/${var.environment}/*"]
  notify_resources  = compact([var.notification_topic_arn, var.ses_identity_arn])
}

output "role_arns" {
  description = "IAM role ARNs for RiskPulse services."
  value = {
    api       = aws_iam_role.api.arn
    worker    = aws_iam_role.worker.arn
    airflow   = aws_iam_role.airflow.arn
    dashboard = aws_iam_role.dashboard.arn
    admin     = aws_iam_role.admin.arn
  }
}

output "policy_arns" {
  description = "IAM policy ARNs created by this module."
  value = {
    api       = aws_iam_policy.api.arn
    worker    = aws_iam_policy.worker.arn
    airflow   = aws_iam_policy.airflow.arn
    dashboard = aws_iam_policy.dashboard.arn
    admin     = aws_iam_policy.admin.arn
  }
}
