resource "aws_cognito_user_pool" "users" {
  name                     = var.name
  username_attributes      = ["email"]
  auto_verified_attributes = ["email"]
  deletion_protection      = "ACTIVE"
  user_pool_tier           = "LITE"

  password_policy {
    minimum_length                   = 12
    require_lowercase                = true
    require_numbers                  = true
    require_symbols                  = false
    require_uppercase                = false
    temporary_password_validity_days = 3
  }

  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }

  admin_create_user_config {
    allow_admin_create_user_only = false
  }

  lambda_config {
    pre_sign_up = aws_lambda_function.signup_gate.arn
  }
}

resource "aws_cognito_user_pool_domain" "users" {
  domain                = "${var.name}-${substr(var.account_id, 0, 6)}"
  user_pool_id          = aws_cognito_user_pool.users.id
  managed_login_version = 1
}

resource "aws_cognito_user_pool_client" "site" {
  count                                = local.live ? 1 : 0
  name                                 = "${var.name}-site"
  user_pool_id                         = aws_cognito_user_pool.users.id
  generate_secret                      = false
  allowed_oauth_flows                  = ["code"]
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_scopes                 = ["openid", "email"]
  supported_identity_providers         = ["COGNITO"]
  callback_urls                        = [for host in local.site_hosts : "https://${host}/auth/"]
  logout_urls                          = [for host in local.site_hosts : "https://${host}/"]
  explicit_auth_flows                  = ["ALLOW_USER_PASSWORD_AUTH", "ALLOW_REFRESH_TOKEN_AUTH"]
  prevent_user_existence_errors        = "ENABLED"
  enable_token_revocation              = true
  access_token_validity                = 60
  id_token_validity                    = 60
  refresh_token_validity               = 30

  token_validity_units {
    access_token  = "minutes"
    id_token      = "minutes"
    refresh_token = "days"
  }
}
