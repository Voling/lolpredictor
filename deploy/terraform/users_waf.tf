locals {
  signup_targets = ["AWSCognitoIdentityProviderService.SignUp", "AWSCognitoIdentityProviderService.ResendConfirmationCode", "AWSCognitoIdentityProviderService.ForgotPassword"]
}

resource "aws_wafv2_web_acl" "users" {
  name  = "${var.name}-users"
  scope = "REGIONAL"

  default_action {
    allow {}
  }

  custom_response_body {
    key          = "too-many"
    content_type = "TEXT_PLAIN"
    content      = "Too many attempts from your network. Try again later."
  }

  rule {
    name     = "ip-reputation"
    priority = 0

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesAmazonIpReputationList"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.name}-users-ip-reputation"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "anonymous-signups"
    priority = 1

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesAnonymousIpList"

        scope_down_statement {
          or_statement {
            statement {
              byte_match_statement {
                positional_constraint = "STARTS_WITH"
                search_string         = "/signup"
                field_to_match {
                  uri_path {}
                }
                text_transformation {
                  priority = 0
                  type     = "LOWERCASE"
                }
              }
            }
            statement {
              byte_match_statement {
                positional_constraint = "EXACTLY"
                search_string         = local.signup_targets[0]
                field_to_match {
                  single_header {
                    name = "x-amz-target"
                  }
                }
                text_transformation {
                  priority = 0
                  type     = "NONE"
                }
              }
            }
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.name}-users-anonymous-signups"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "signup-rate"
    priority = 2

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
        limit                 = var.signup_rate_limit
        aggregate_key_type    = "IP"
        evaluation_window_sec = 600

        scope_down_statement {
          and_statement {
            statement {
              byte_match_statement {
                positional_constraint = "EXACTLY"
                search_string         = "POST"
                field_to_match {
                  method {}
                }
                text_transformation {
                  priority = 0
                  type     = "NONE"
                }
              }
            }
            statement {
              or_statement {
                statement {
                  byte_match_statement {
                    positional_constraint = "STARTS_WITH"
                    search_string         = "/signup"
                    field_to_match {
                      uri_path {}
                    }
                    text_transformation {
                      priority = 0
                      type     = "LOWERCASE"
                    }
                  }
                }
                statement {
                  byte_match_statement {
                    positional_constraint = "STARTS_WITH"
                    search_string         = "/forgotpassword"
                    field_to_match {
                      uri_path {}
                    }
                    text_transformation {
                      priority = 0
                      type     = "LOWERCASE"
                    }
                  }
                }
                dynamic "statement" {
                  for_each = local.signup_targets
                  content {
                    byte_match_statement {
                      positional_constraint = "EXACTLY"
                      search_string         = statement.value
                      field_to_match {
                        single_header {
                          name = "x-amz-target"
                        }
                      }
                      text_transformation {
                        priority = 0
                        type     = "NONE"
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.name}-users-signup-rate"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "post-rate"
    priority = 3

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
        limit                 = var.auth_rate_limit
        aggregate_key_type    = "IP"
        evaluation_window_sec = 600

        scope_down_statement {
          byte_match_statement {
            positional_constraint = "EXACTLY"
            search_string         = "POST"
            field_to_match {
              method {}
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
      metric_name                = "${var.name}-users-post-rate"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.name}-users"
    sampled_requests_enabled   = true
  }
}

resource "aws_wafv2_web_acl_association" "users" {
  resource_arn = aws_cognito_user_pool.users.arn
  web_acl_arn  = aws_wafv2_web_acl.users.arn
}
