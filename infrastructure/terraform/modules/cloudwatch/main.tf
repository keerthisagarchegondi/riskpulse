variable "environment" {
  description = "Environment name."
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "project_name" {
  description = "Project name used for CloudWatch resource names."
  type        = string
  default     = "riskpulse"
}

variable "aws_region" {
  description = "AWS region for CloudWatch dashboard widgets."
  type        = string
  default     = "us-east-1"
}

variable "service_names" {
  description = "Services that ship logs to CloudWatch."
  type        = set(string)
  default     = ["api", "worker", "fraud-engine", "airflow"]
}

variable "log_retention_days" {
  description = "CloudWatch Logs retention period."
  type        = number
  default     = 30
}

variable "alarm_actions" {
  description = "SNS topic ARNs or other alarm action ARNs."
  type        = list(string)
  default     = []
}

variable "ok_actions" {
  description = "SNS topic ARNs or other OK action ARNs."
  type        = list(string)
  default     = []
}

variable "fraud_rate_spike_threshold" {
  description = "Fraud detection rate threshold representing a 2x baseline spike."
  type        = number
  default     = 2
}

variable "error_rate_threshold" {
  description = "Error rate percent threshold."
  type        = number
  default     = 1
}

variable "pipeline_p99_latency_threshold_ms" {
  description = "P99 pipeline latency alarm threshold in milliseconds."
  type        = number
  default     = 5000
}

variable "kafka_consumer_lag_threshold" {
  description = "Kafka consumer lag alarm threshold."
  type        = number
  default     = 10000
}

variable "tags" {
  description = "Tags applied to CloudWatch resources."
  type        = map(string)
  default     = {}
}

locals {
  namespace = "RiskPulse/Platform"
  tags = merge(var.tags, {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Module      = "cloudwatch"
  })
}

resource "aws_cloudwatch_log_group" "service" {
  for_each = var.service_names

  name              = "/${var.project_name}/${var.environment}/${each.value}"
  retention_in_days = var.log_retention_days
  tags              = local.tags
}

resource "aws_cloudwatch_dashboard" "platform" {
  dashboard_name = "${var.project_name}-${var.environment}-platform-health"
  dashboard_body = jsonencode({
    start          = "-PT3H"
    periodOverride = "auto"
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "Pipeline Health Overview"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          metrics = [
            [local.namespace, "TransactionsProcessedPerMinute", "Environment", var.environment],
            [".", "PipelineThroughputRecords", ".", "."],
            [".", "DependencyHealth", ".", ".", "Dependency", "kafka"],
            [".", "DependencyHealth", ".", ".", "Dependency", "postgresql"]
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Fraud Metrics"
          region = var.aws_region
          view   = "timeSeries"
          metrics = [
            [local.namespace, "FraudDetectionRate", "Environment", var.environment],
            [".", "AlertGenerationRate", ".", "."],
            [".", "FalsePositiveRate", ".", "."]
          ]
          yAxis = {
            left = {
              min = 0
            }
          }
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "Error Rate"
          region = var.aws_region
          view   = "timeSeries"
          metrics = [
            [local.namespace, "ErrorRate", "Environment", var.environment],
            [".", "ErrorCount", ".", "."]
          ]
          annotations = {
            horizontal = [
              {
                label = "1% error rate"
                value = var.error_rate_threshold
              }
            ]
          }
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "Latency and Consumer Lag"
          region = var.aws_region
          view   = "timeSeries"
          metrics = [
            [local.namespace, "PipelineLatency", "Environment", var.environment, "Percentile", "P99"],
            [".", "ModelPredictionLatency", ".", "."],
            [".", "KafkaConsumerLag", ".", "."]
          ]
          annotations = {
            horizontal = [
              {
                label = "P99 latency 5s"
                value = var.pipeline_p99_latency_threshold_ms
              },
              {
                label = "Lag 10K"
                value = var.kafka_consumer_lag_threshold
              }
            ]
          }
        }
      }
    ]
  })
}

resource "aws_cloudwatch_metric_alarm" "high_error_rate" {
  alarm_name          = "${var.project_name}-${var.environment}-high-error-rate"
  alarm_description   = "Service error rate is above ${var.error_rate_threshold}%."
  namespace           = local.namespace
  metric_name         = "ErrorRate"
  statistic           = "Average"
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.error_rate_threshold
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  period              = 300
  treat_missing_data  = "notBreaching"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.ok_actions
  tags                = local.tags
}

resource "aws_cloudwatch_metric_alarm" "pipeline_p99_latency" {
  alarm_name          = "${var.project_name}-${var.environment}-pipeline-p99-latency"
  alarm_description   = "Pipeline P99 latency is above ${var.pipeline_p99_latency_threshold_ms}ms."
  namespace           = local.namespace
  metric_name         = "PipelineLatency"
  statistic           = "Average"
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.pipeline_p99_latency_threshold_ms
  evaluation_periods  = 3
  datapoints_to_alarm = 2
  period              = 60
  treat_missing_data  = "missing"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.ok_actions
  tags                = local.tags

  dimensions = {
    Environment = var.environment
    Percentile  = "P99"
  }
}

resource "aws_cloudwatch_metric_alarm" "kafka_consumer_lag" {
  alarm_name          = "${var.project_name}-${var.environment}-kafka-consumer-lag"
  alarm_description   = "Kafka consumer lag is above ${var.kafka_consumer_lag_threshold} messages."
  namespace           = local.namespace
  metric_name         = "KafkaConsumerLag"
  statistic           = "Maximum"
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.kafka_consumer_lag_threshold
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  period              = 60
  treat_missing_data  = "notBreaching"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.ok_actions
  tags                = local.tags

  dimensions = {
    Environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "fraud_rate_spike" {
  alarm_name          = "${var.project_name}-${var.environment}-high-fraud-rate-spike"
  alarm_description   = "Fraud detection rate is more than 2x the configured production baseline."
  namespace           = local.namespace
  metric_name         = "FraudDetectionRate"
  statistic           = "Average"
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.fraud_rate_spike_threshold
  evaluation_periods  = 3
  datapoints_to_alarm = 2
  period              = 300
  treat_missing_data  = "notBreaching"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.ok_actions
  tags                = local.tags

  dimensions = {
    Environment = var.environment
    Severity    = "high"
  }
}

resource "aws_cloudwatch_metric_alarm" "critical_dependency_unhealthy" {
  alarm_name          = "${var.project_name}-${var.environment}-critical-dependency-unhealthy"
  alarm_description   = "A critical dependency health check is failing."
  namespace           = local.namespace
  metric_name         = "DependencyHealth"
  statistic           = "Minimum"
  comparison_operator = "LessThanThreshold"
  threshold           = 1
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  period              = 60
  treat_missing_data  = "breaching"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.ok_actions
  tags                = local.tags

  dimensions = {
    Environment = var.environment
  }
}

output "log_group_names" {
  description = "CloudWatch log groups created for RiskPulse services."
  value       = { for service, group in aws_cloudwatch_log_group.service : service => group.name }
}

output "dashboard_name" {
  description = "CloudWatch dashboard name."
  value       = aws_cloudwatch_dashboard.platform.dashboard_name
}

output "alarm_names" {
  description = "CloudWatch alarm names."
  value = [
    aws_cloudwatch_metric_alarm.high_error_rate.alarm_name,
    aws_cloudwatch_metric_alarm.pipeline_p99_latency.alarm_name,
    aws_cloudwatch_metric_alarm.kafka_consumer_lag.alarm_name,
    aws_cloudwatch_metric_alarm.fraud_rate_spike.alarm_name,
    aws_cloudwatch_metric_alarm.critical_dependency_unhealthy.alarm_name
  ]
}
