locals {
  bucket_prefix = "${var.project_name}-${var.environment}"
  kms_key_arn   = var.create_kms_key && var.kms_key_arn == "" ? aws_kms_key.s3_encryption[0].arn : var.kms_key_arn

  common_tags = merge(var.tags, {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Module      = "s3-data-lake"
  })

  buckets = {
    raw = {
      name               = "${local.bucket_prefix}-raw"
      encryption_type    = "aws:s3"
      kms_key_arn        = null
      glacier_days       = var.raw_glacier_transition_days
      ia_days            = 30
      enable_replication = var.enable_replication
    }
    processed = {
      name               = "${local.bucket_prefix}-processed"
      encryption_type    = "aws:kms"
      kms_key_arn        = local.kms_key_arn
      glacier_days       = var.processed_glacier_transition_days
      ia_days            = 60
      enable_replication = var.enable_replication
    }
    models = {
      name               = "${local.bucket_prefix}-models"
      encryption_type    = "aws:kms"
      kms_key_arn        = local.kms_key_arn
      glacier_days       = null
      ia_days            = null
      enable_replication = var.enable_replication
    }
    archive = {
      name               = "${local.bucket_prefix}-archive"
      encryption_type    = "aws:kms"
      kms_key_arn        = local.kms_key_arn
      glacier_days       = null
      ia_days            = null
      enable_replication = false
    }
  }
}

# =============================================================================
# KMS Key for S3 Encryption
# =============================================================================

resource "aws_kms_key" "s3_encryption" {
  count = var.create_kms_key && var.kms_key_arn == "" ? 1 : 0

  description             = "${var.project_name} S3 data lake encryption key (${var.environment})"
  deletion_window_in_days = var.kms_key_deletion_window
  enable_key_rotation     = true
  multi_region            = var.enable_replication

  policy = jsonencode({
    Version = "2012-10-17"
    Id      = "${var.project_name}-s3-key-policy"
    Statement = [
      {
        Sid       = "EnableRootAccountAccess"
        Effect    = "Allow"
        Principal = { AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root" }
        Action    = "kms:*"
        Resource  = "*"
      },
      {
        Sid    = "AllowS3ServiceUsage"
        Effect = "Allow"
        Principal = { Service = "s3.amazonaws.com" }
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = merge(local.common_tags, { Name = "${var.project_name}-s3-encryption-key" })
}

resource "aws_kms_alias" "s3_encryption" {
  count = var.create_kms_key && var.kms_key_arn == "" ? 1 : 0

  name          = "alias/${var.project_name}-${var.environment}-s3"
  target_key_id = aws_kms_key.s3_encryption[0].key_id
}

data "aws_caller_identity" "current" {}

# =============================================================================
# S3 Buckets
# =============================================================================

resource "aws_s3_bucket" "data_lake" {
  for_each = local.buckets

  bucket        = each.value.name
  force_destroy = var.force_destroy

  tags = merge(local.common_tags, {
    Name     = each.value.name
    DataZone = each.key
  })
}

# =============================================================================
# Bucket Versioning
# =============================================================================

resource "aws_s3_bucket_versioning" "data_lake" {
  for_each = local.buckets

  bucket = aws_s3_bucket.data_lake[each.key].id

  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Suspended"
  }
}

# =============================================================================
# Server-Side Encryption
# =============================================================================

resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake" {
  for_each = local.buckets

  bucket = aws_s3_bucket.data_lake[each.key].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = each.value.encryption_type == "aws:kms" ? "aws:kms" : "AES256"
      kms_master_key_id = each.value.kms_key_arn
    }
    bucket_key_enabled = each.value.encryption_type == "aws:kms" ? true : false
  }
}

# =============================================================================
# Public Access Block (all buckets)
# =============================================================================

resource "aws_s3_bucket_public_access_block" "data_lake" {
  for_each = local.buckets

  bucket = aws_s3_bucket.data_lake[each.key].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# =============================================================================
# Lifecycle Rules — Raw Bucket
# =============================================================================

resource "aws_s3_bucket_lifecycle_configuration" "raw" {
  bucket = aws_s3_bucket.data_lake["raw"].id

  depends_on = [aws_s3_bucket_versioning.data_lake["raw"]]

  rule {
    id     = "raw-transition-to-ia"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
  }

  rule {
    id     = "raw-transition-to-glacier"
    status = "Enabled"

    transition {
      days          = var.raw_glacier_transition_days
      storage_class = "GLACIER"
    }
  }

  rule {
    id     = "raw-noncurrent-cleanup"
    status = "Enabled"

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_expiration_days
    }
  }

  rule {
    id     = "raw-abort-incomplete-multipart"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = var.abort_incomplete_multipart_days
    }
  }
}

# =============================================================================
# Lifecycle Rules — Processed Bucket
# =============================================================================

resource "aws_s3_bucket_lifecycle_configuration" "processed" {
  bucket = aws_s3_bucket.data_lake["processed"].id

  depends_on = [aws_s3_bucket_versioning.data_lake["processed"]]

  rule {
    id     = "processed-transition-to-ia"
    status = "Enabled"

    transition {
      days          = 60
      storage_class = "STANDARD_IA"
    }
  }

  rule {
    id     = "processed-transition-to-glacier"
    status = "Enabled"

    transition {
      days          = var.processed_glacier_transition_days
      storage_class = "GLACIER"
    }
  }

  rule {
    id     = "processed-noncurrent-cleanup"
    status = "Enabled"

    noncurrent_version_transition {
      noncurrent_days = 60
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = 120
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_expiration_days
    }
  }

  rule {
    id     = "processed-abort-incomplete-multipart"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = var.abort_incomplete_multipart_days
    }
  }
}

# =============================================================================
# Lifecycle Rules — Archive Bucket
# =============================================================================

resource "aws_s3_bucket_lifecycle_configuration" "archive" {
  bucket = aws_s3_bucket.data_lake["archive"].id

  depends_on = [aws_s3_bucket_versioning.data_lake["archive"]]

  rule {
    id     = "archive-transition-to-deep-archive"
    status = "Enabled"

    transition {
      days          = var.archive_deep_archive_transition_days
      storage_class = "DEEP_ARCHIVE"
    }
  }

  rule {
    id     = "archive-noncurrent-cleanup"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 730
    }
  }

  rule {
    id     = "archive-abort-incomplete-multipart"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = var.abort_incomplete_multipart_days
    }
  }
}

# =============================================================================
# Lifecycle Rules — Models Bucket (minimal — keep models accessible)
# =============================================================================

resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.data_lake["models"].id

  depends_on = [aws_s3_bucket_versioning.data_lake["models"]]

  rule {
    id     = "models-noncurrent-cleanup"
    status = "Enabled"

    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_expiration_days
    }
  }

  rule {
    id     = "models-abort-incomplete-multipart"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = var.abort_incomplete_multipart_days
    }
  }
}

# =============================================================================
# S3 Event Notifications (Raw bucket → SQS)
# =============================================================================

resource "aws_s3_bucket_notification" "raw_events" {
  count = var.enable_event_notifications && var.sqs_queue_arn != "" ? 1 : 0

  bucket = aws_s3_bucket.data_lake["raw"].id

  queue {
    id            = "new-raw-file-notification"
    queue_arn     = var.sqs_queue_arn
    events        = ["s3:ObjectCreated:*"]
    filter_suffix = ".parquet"
  }

  queue {
    id            = "new-raw-json-notification"
    queue_arn     = var.sqs_queue_arn
    events        = ["s3:ObjectCreated:*"]
    filter_suffix = ".json"
  }
}

resource "aws_s3_bucket_notification" "processed_events" {
  count = var.enable_event_notifications && var.sqs_queue_arn != "" ? 1 : 0

  bucket = aws_s3_bucket.data_lake["processed"].id

  queue {
    id            = "new-processed-file-notification"
    queue_arn     = var.sqs_queue_arn
    events        = ["s3:ObjectCreated:*"]
    filter_suffix = ".parquet"
  }
}

# =============================================================================
# Cross-Region Replication
# =============================================================================

resource "aws_iam_role" "replication" {
  count = var.enable_replication ? 1 : 0

  name = "${var.project_name}-${var.environment}-s3-replication-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "s3.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "replication" {
  count = var.enable_replication ? 1 : 0

  name = "${var.project_name}-${var.environment}-s3-replication-policy"
  role = aws_iam_role.replication[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetReplicationConfiguration",
          "s3:ListBucket"
        ]
        Resource = [for b in aws_s3_bucket.data_lake : b.arn]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging"
        ]
        Resource = [for b in aws_s3_bucket.data_lake : "${b.arn}/*"]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags"
        ]
        Resource = [
          "arn:aws:s3:::${local.bucket_prefix}-raw-dr/*",
          "arn:aws:s3:::${local.bucket_prefix}-processed-dr/*",
          "arn:aws:s3:::${local.bucket_prefix}-models-dr/*"
        ]
      }
    ]
  })
}

# DR Buckets in secondary region
resource "aws_s3_bucket" "dr" {
  for_each = var.enable_replication ? {
    raw       = "${local.bucket_prefix}-raw-dr"
    processed = "${local.bucket_prefix}-processed-dr"
    models    = "${local.bucket_prefix}-models-dr"
  } : {}

  provider      = aws.dr
  bucket        = each.value
  force_destroy = var.force_destroy

  tags = merge(local.common_tags, {
    Name     = each.value
    DataZone = each.key
    Purpose  = "disaster-recovery"
  })
}

resource "aws_s3_bucket_versioning" "dr" {
  for_each = var.enable_replication ? {
    raw       = "${local.bucket_prefix}-raw-dr"
    processed = "${local.bucket_prefix}-processed-dr"
    models    = "${local.bucket_prefix}-models-dr"
  } : {}

  provider = aws.dr
  bucket   = aws_s3_bucket.dr[each.key].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_replication_configuration" "raw" {
  count = var.enable_replication ? 1 : 0

  role   = aws_iam_role.replication[0].arn
  bucket = aws_s3_bucket.data_lake["raw"].id

  depends_on = [aws_s3_bucket_versioning.data_lake["raw"]]

  rule {
    id     = "raw-replication"
    status = "Enabled"

    destination {
      bucket        = aws_s3_bucket.dr["raw"].arn
      storage_class = "STANDARD_IA"
    }
  }
}

resource "aws_s3_bucket_replication_configuration" "processed" {
  count = var.enable_replication ? 1 : 0

  role   = aws_iam_role.replication[0].arn
  bucket = aws_s3_bucket.data_lake["processed"].id

  depends_on = [aws_s3_bucket_versioning.data_lake["processed"]]

  rule {
    id     = "processed-replication"
    status = "Enabled"

    destination {
      bucket        = aws_s3_bucket.dr["processed"].arn
      storage_class = "STANDARD_IA"
    }
  }
}

# =============================================================================
# Bucket Policies — Enforce encryption and restrict access
# =============================================================================

resource "aws_s3_bucket_policy" "enforce_encryption" {
  for_each = local.buckets

  bucket = aws_s3_bucket.data_lake[each.key].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyUnencryptedObjectUploads"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:PutObject"
        Resource  = "${aws_s3_bucket.data_lake[each.key].arn}/*"
        Condition = {
          StringNotEquals = {
            "s3:x-amz-server-side-encryption" = each.value.encryption_type == "aws:kms" ? "aws:kms" : "AES256"
          }
        }
      },
      {
        Sid       = "DenyInsecureTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.data_lake[each.key].arn,
          "${aws_s3_bucket.data_lake[each.key].arn}/*"
        ]
        Condition = {
          Bool = { "aws:SecureTransport" = "false" }
        }
      }
    ]
  })
}
