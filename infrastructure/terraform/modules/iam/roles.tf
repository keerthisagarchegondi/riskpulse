data "aws_iam_policy_document" "api_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = var.trusted_service_principals.api
    }
  }
}

data "aws_iam_policy_document" "worker_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = var.trusted_service_principals.worker
    }
  }
}

data "aws_iam_policy_document" "airflow_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = var.trusted_service_principals.airflow
    }
  }
}

data "aws_iam_policy_document" "dashboard_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = var.trusted_service_principals.dashboard
    }
  }
}

data "aws_iam_policy_document" "admin_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = length(var.admin_principal_arns) > 0 ? "AWS" : "Service"
      identifiers = length(var.admin_principal_arns) > 0 ? var.admin_principal_arns : var.trusted_service_principals.admin
    }
  }
}

resource "aws_iam_role" "api" {
  name               = "${local.name_prefix}-api-role"
  assume_role_policy = data.aws_iam_policy_document.api_assume_role.json
  description        = "Least-privilege role for the RiskPulse API service."
  tags               = local.tags
}

resource "aws_iam_role" "worker" {
  name               = "${local.name_prefix}-worker-role"
  assume_role_policy = data.aws_iam_policy_document.worker_assume_role.json
  description        = "Least-privilege role for RiskPulse data processing workers."
  tags               = local.tags
}

resource "aws_iam_role" "airflow" {
  name               = "${local.name_prefix}-airflow-role"
  assume_role_policy = data.aws_iam_policy_document.airflow_assume_role.json
  description        = "Least-privilege role for RiskPulse orchestration tasks."
  tags               = local.tags
}

resource "aws_iam_role" "dashboard" {
  name               = "${local.name_prefix}-dashboard-role"
  assume_role_policy = data.aws_iam_policy_document.dashboard_assume_role.json
  description        = "Read-only analytics role for RiskPulse dashboards."
  tags               = local.tags
}

resource "aws_iam_role" "admin" {
  name               = "${local.name_prefix}-admin-role"
  assume_role_policy = data.aws_iam_policy_document.admin_assume_role.json
  description        = "Operations admin role for break-glass RiskPulse access."
  tags               = local.tags
}
