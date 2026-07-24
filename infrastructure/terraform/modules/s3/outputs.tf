output "bucket_ids" {
  description = "Map of data zone to S3 bucket IDs"
  value       = { for k, v in aws_s3_bucket.data_lake : k => v.id }
}

output "bucket_arns" {
  description = "Map of data zone to S3 bucket ARNs"
  value       = { for k, v in aws_s3_bucket.data_lake : k => v.arn }
}

output "bucket_names" {
  description = "Map of data zone to S3 bucket names"
  value       = { for k, v in local.buckets : k => v.name }
}

output "raw_bucket_name" {
  description = "Name of the raw data bucket"
  value       = aws_s3_bucket.data_lake["raw"].id
}

output "processed_bucket_name" {
  description = "Name of the processed data bucket"
  value       = aws_s3_bucket.data_lake["processed"].id
}

output "models_bucket_name" {
  description = "Name of the models bucket"
  value       = aws_s3_bucket.data_lake["models"].id
}

output "archive_bucket_name" {
  description = "Name of the archive bucket"
  value       = aws_s3_bucket.data_lake["archive"].id
}

output "kms_key_arn" {
  description = "ARN of the KMS key used for S3 encryption"
  value       = var.create_kms_key && var.kms_key_arn == "" ? aws_kms_key.s3_encryption[0].arn : var.kms_key_arn
}

output "kms_key_id" {
  description = "ID of the KMS key used for S3 encryption"
  value       = var.create_kms_key && var.kms_key_arn == "" ? aws_kms_key.s3_encryption[0].key_id : ""
}

output "replication_role_arn" {
  description = "ARN of the IAM role used for cross-region replication"
  value       = var.enable_replication ? aws_iam_role.replication[0].arn : ""
}

output "dr_bucket_names" {
  description = "Map of data zone to DR bucket names (empty if replication disabled)"
  value       = var.enable_replication ? { for k, v in aws_s3_bucket.dr : k => v.id } : {}
}
