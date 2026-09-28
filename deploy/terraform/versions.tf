terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.7"
    }
  }
}

provider "aws" {
  region              = var.region
  profile             = var.profile
  allowed_account_ids = [var.account_id]
  default_tags {
    tags = { project = var.name }
  }
}

provider "aws" {
  alias               = "global"
  region              = "us-east-1"
  profile             = var.profile
  allowed_account_ids = [var.account_id]
  default_tags {
    tags = { project = var.name }
  }
}
