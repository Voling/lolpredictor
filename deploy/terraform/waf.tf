locals {
  managed_rules = [
    { name = "ip-reputation", group = "AWSManagedRulesAmazonIpReputationList" },
    { name = "common", group = "AWSManagedRulesCommonRuleSet" },
    { name = "bad-inputs", group = "AWSManagedRulesKnownBadInputsRuleSet" },
  ]
}

resource "aws_wafv2_web_acl" "site" {
  provider = aws.global
  name     = "${var.name}-site"
  scope    = "CLOUDFRONT"

  default_action {
    allow {}
  }

  custom_response_body {
    key          = "too-many"
    content_type = "APPLICATION_JSON"
    content      = jsonencode({ detail = "Too many requests. Try again in a few minutes." })
  }

  dynamic "rule" {
    for_each = local.managed_rules
    content {
      name     = rule.value.name
      priority = rule.key

      override_action {
        none {}
      }

      statement {
        managed_rule_group_statement {
          vendor_name = "AWS"
          name        = rule.value.group
        }
      }

      visibility_config {
        cloudwatch_metrics_enabled = true
        metric_name                = "${var.name}-${rule.value.name}"
        sampled_requests_enabled   = true
      }
    }
  }

  rule {
    name     = "api-rate"
    priority = 10

    action {
      block {
        custom_response {
          response_code            = 429
          custom_response_body_key = "too-many"
        }
      }
    }

    statement {
      rate_based_statement {
        limit                 = var.api_rate_limit
        aggregate_key_type    = "IP"
        evaluation_window_sec = 300

        scope_down_statement {
          byte_match_statement {
            positional_constraint = "STARTS_WITH"
            search_string         = "/api/"

            field_to_match {
              uri_path {}
            }

            text_transformation {
              priority = 0
              type     = "NONE"
            }
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.name}-api-rate"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "site-rate"
    priority = 11

    action {
      block {
        custom_response {
          response_code            = 429
          custom_response_body_key = "too-many"
        }
      }
    }

    statement {
      rate_based_statement {
        limit                 = var.site_rate_limit
        aggregate_key_type    = "IP"
        evaluation_window_sec = 300
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.name}-site-rate"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.name}-site"
    sampled_requests_enabled   = true
  }
}
