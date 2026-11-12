resource "aws_budgets_budget" "monthly" {
  count        = var.budget_email != "" ? 1 : 0
  name         = "${var.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }
}

locals {
  backend_services = [
    "AWS Lambda",
    "Amazon Simple Queue Service",
    "Amazon DynamoDB",
    "Amazon Elastic Container Service",
    "Amazon EC2 Container Registry (ECR)",
    "Amazon Simple Storage Service",
    "AmazonCloudWatch",
    "CloudWatch Events",
    "Amazon Simple Notification Service",
    "Amazon Cognito",
    "AWS Key Management Service",
  ]
}

resource "aws_ce_cost_allocation_tag" "project" {
  provider = aws.global
  tag_key  = "project"
  status   = "Active"
}

resource "aws_budgets_budget" "backend" {
  name         = "${var.name}-backend"
  budget_type  = "COST"
  limit_amount = tostring(var.killswitch_budget)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  depends_on   = [aws_ce_cost_allocation_tag.project, aws_sns_topic_policy.killswitch]

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:project${"$"}${var.name}"]
  }

  cost_filter {
    name   = "Service"
    values = local.backend_services
  }

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 100
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_sns_topic_arns = [aws_sns_topic.killswitch.arn]
  }

  dynamic "notification" {
    for_each = var.alert_email == "" ? [] : [var.alert_email]
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = 80
      threshold_type             = "PERCENTAGE"
      notification_type          = "FORECASTED"
      subscriber_email_addresses = [notification.value]
    }
  }
}
