# Provider configuration for the S3 data lake module.
# The default provider is used for primary region resources.
# The "aws.dr" provider alias is required when enable_replication = true.

terraform {
  required_providers {
    aws = {
      source                = "hashicorp/aws"
      version               = ">= 5.0"
      configuration_aliases = [aws.dr]
    }
  }
}
