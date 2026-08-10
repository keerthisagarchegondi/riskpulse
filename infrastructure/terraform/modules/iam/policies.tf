data "aws_iam_policy_document" "cloudwatch_write" {
  statement {
    sid    = "WriteRiskPulseLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:DescribeLogStreams",
      "logs:PutLogEvents",
      "logs:PutRetentionPolicy"
    ]
    resources = ["arn:aws:logs:*:*:log-group:/${var.project_name}/${var.environment}/*"]
  }

  statement {
    sid       = "PublishRiskPulseMetrics"
    effect    = "Allow"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]

    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = [local.namespace]
    }
  }
}

data "aws_iam_policy_document" "secrets_read" {
  statement {
    sid    = "ReadConfiguredSecrets"
    effect = "Allow"
    actions = [
      "secretsmanager:DescribeSecret",
      "secretsmanager:GetSecretValue"
    ]
    resources = local.all_secret_arns
  }
}

data "aws_iam_policy_document" "secrets_rotate" {
  statement {
    sid    = "RotateDatabaseSecrets"
    effect = "Allow"
    actions = [
      "secretsmanager:CancelRotateSecret",
      "secretsmanager:DescribeSecret",
      "secretsmanager:GetRandomPassword",
      "secretsmanager:GetSecretValue",
      "secretsmanager:PutSecretValue",
      "secretsmanager:RotateSecret",
      "secretsmanager:UpdateSecretVersionStage"
    ]
    resources = local.all_secret_arns
  }
}

data "aws_iam_policy_document" "kms_data_key" {
  statement {
    sid    = "UsePlatformKey"
    effect = "Allow"
    actions = [
      "kms:Decrypt",
      "kms:DescribeKey",
      "kms:Encrypt",
      "kms:GenerateDataKey"
    ]
    resources = [var.kms_key_arn]
  }
}

data "aws_iam_policy_document" "api_permissions" {
  source_policy_documents = [
    data.aws_iam_policy_document.cloudwatch_write.json,
    data.aws_iam_policy_document.secrets_read.json,
    data.aws_iam_policy_document.kms_data_key.json
  ]

  dynamic "statement" {
    for_each = length(local.notify_resources) > 0 ? [1] : []

    content {
      sid    = "PublishNotifications"
      effect = "Allow"
      actions = [
        "sns:Publish",
        "ses:SendEmail",
        "ses:SendRawEmail"
      ]
      resources = local.notify_resources
    }
  }
}

data "aws_iam_policy_document" "worker_permissions" {
  source_policy_documents = [
    data.aws_iam_policy_document.cloudwatch_write.json,
    data.aws_iam_policy_document.secrets_read.json,
    data.aws_iam_policy_document.kms_data_key.json
  ]

  statement {
    sid    = "ReadRawTransactions"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket"
    ]
    resources = [
      var.raw_bucket_arn,
      local.raw_objects
    ]
  }

  statement {
    sid    = "WriteProcessedAndModels"
    effect = "Allow"
    actions = [
      "s3:AbortMultipartUpload",
      "s3:GetObject",
      "s3:ListMultipartUploadParts",
      "s3:PutObject"
    ]
    resources = [
      local.processed_objects,
      local.models_objects
    ]
  }
}

data "aws_iam_policy_document" "airflow_permissions" {
  source_policy_documents = [
    data.aws_iam_policy_document.cloudwatch_write.json,
    data.aws_iam_policy_document.secrets_read.json,
    data.aws_iam_policy_document.kms_data_key.json
  ]

  statement {
    sid    = "OrchestrateDataLake"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
      "s3:PutObject"
    ]
    resources = [
      var.raw_bucket_arn,
      var.processed_bucket_arn,
      local.raw_objects,
      local.processed_objects
    ]
  }
}

data "aws_iam_policy_document" "dashboard_permissions" {
  statement {
    sid    = "ReadAnalyticsData"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket"
    ]
    resources = [
      var.processed_bucket_arn,
      var.models_bucket_arn,
      local.processed_objects,
      local.models_objects
    ]
  }

  statement {
    sid    = "ReadCloudWatchDashboards"
    effect = "Allow"
    actions = [
      "cloudwatch:DescribeAlarms",
      "cloudwatch:GetDashboard",
      "cloudwatch:GetMetricData",
      "cloudwatch:ListDashboards",
      "logs:DescribeLogGroups",
      "logs:DescribeLogStreams",
      "logs:FilterLogEvents"
    ]
    resources = ["*"]
  }

  statement {
    sid       = "ReadDashboardSecrets"
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
    resources = local.all_secret_arns
  }
}

data "aws_iam_policy_document" "admin_permissions" {
  statement {
    sid       = "RiskPulseAdmin"
    effect    = "Allow"
    actions   = ["*"]
    resources = ["*"]

    condition {
      test     = "StringEqualsIfExists"
      variable = "aws:ResourceTag/Project"
      values   = [var.project_name]
    }
  }

  source_policy_documents = [
    data.aws_iam_policy_document.secrets_rotate.json
  ]
}

resource "aws_iam_policy" "api" {
  name        = "${local.name_prefix}-api-policy"
  description = "Least-privilege permissions for the RiskPulse API."
  policy      = data.aws_iam_policy_document.api_permissions.json
  tags        = local.tags
}

resource "aws_iam_policy" "worker" {
  name        = "${local.name_prefix}-worker-policy"
  description = "Least-privilege permissions for RiskPulse data processing."
  policy      = data.aws_iam_policy_document.worker_permissions.json
  tags        = local.tags
}

resource "aws_iam_policy" "airflow" {
  name        = "${local.name_prefix}-airflow-policy"
  description = "Least-privilege permissions for RiskPulse orchestration."
  policy      = data.aws_iam_policy_document.airflow_permissions.json
  tags        = local.tags
}

resource "aws_iam_policy" "dashboard" {
  name        = "${local.name_prefix}-dashboard-policy"
  description = "Read-only analytics permissions for RiskPulse dashboards."
  policy      = data.aws_iam_policy_document.dashboard_permissions.json
  tags        = local.tags
}

resource "aws_iam_policy" "admin" {
  name        = "${local.name_prefix}-admin-policy"
  description = "Ops admin permissions for tagged RiskPulse resources and secret rotation."
  policy      = data.aws_iam_policy_document.admin_permissions.json
  tags        = local.tags
}

resource "aws_iam_role_policy_attachment" "api" {
  role       = aws_iam_role.api.name
  policy_arn = aws_iam_policy.api.arn
}

resource "aws_iam_role_policy_attachment" "worker" {
  role       = aws_iam_role.worker.name
  policy_arn = aws_iam_policy.worker.arn
}

resource "aws_iam_role_policy_attachment" "airflow" {
  role       = aws_iam_role.airflow.name
  policy_arn = aws_iam_policy.airflow.arn
}

resource "aws_iam_role_policy_attachment" "dashboard" {
  role       = aws_iam_role.dashboard.name
  policy_arn = aws_iam_policy.dashboard.arn
}

resource "aws_iam_role_policy_attachment" "admin" {
  role       = aws_iam_role.admin.name
  policy_arn = aws_iam_policy.admin.arn
}

resource "aws_secretsmanager_secret_rotation" "database" {
  count = contains(keys(var.secret_arns), "database") && var.database_rotation_lambda_arn != "" ? 1 : 0

  secret_id           = var.secret_arns.database
  rotation_lambda_arn = var.database_rotation_lambda_arn

  rotation_rules {
    automatically_after_days = 30
  }
}
