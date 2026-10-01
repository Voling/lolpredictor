locals {
  live               = var.image_uri != ""
  riot_key_parameter = "/${var.name}/riot-api-key"
}

resource "aws_ecr_repository" "api" {
  name                 = "${var.name}-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the three newest images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 3 }
      action       = { type = "expire" }
    }]
  })
}

resource "aws_iam_role" "api" {
  name = "${var.name}-api"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "lambda.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "api" {
  name = "${var.name}-api"
  role = aws_iam_role.api.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "Logs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "${aws_cloudwatch_log_group.api.arn}:*"
      },
      {
        Sid      = "ContributedGamesWriteOnly"
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "${aws_s3_bucket.contributed.arn}/*"
      },
      {
        Sid      = "AccountsAndDailyLimits"
        Effect   = "Allow"
        Action   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem", "dynamodb:Query"]
        Resource = aws_dynamodb_table.accounts.arn
      },
      {
        Sid      = "RiotKey"
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = "arn:aws:ssm:${var.region}:${var.account_id}:parameter${local.riot_key_parameter}"
      },
      {
        Sid       = "RiotKeyDecrypt"
        Effect    = "Allow"
        Action    = ["kms:Decrypt"]
        Resource  = "*"
        Condition = { StringEquals = { "kms:ViaService" = "ssm.${var.region}.amazonaws.com" } }
      },
    ]
  })
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${var.name}-api"
  retention_in_days = var.log_days
}

resource "aws_lambda_function" "api" {
  count                          = local.live ? 1 : 0
  function_name                  = "${var.name}-api"
  role                           = aws_iam_role.api.arn
  package_type                   = "Image"
  image_uri                      = var.image_uri
  architectures                  = ["x86_64"]
  memory_size                    = var.memory_mb
  timeout                        = 60
  reserved_concurrent_executions = var.api_concurrency
  depends_on                     = [aws_iam_role_policy.api, aws_cloudwatch_log_group.api]

  lifecycle {
    ignore_changes = [image_uri]
  }

  environment {
    variables = {
      PUBLIC_API         = "1"
      COGNITO_POOL_ID    = aws_cognito_user_pool.users.id
      COGNITO_REGION     = var.region
      ACCOUNTS_TABLE     = aws_dynamodb_table.accounts.name
      RIOT_KEY_PARAMETER = local.riot_key_parameter
      DAILY_DUOS         = tostring(var.daily_duos)
      MAX_FRIENDS        = tostring(var.max_friends)
    }
  }
}

resource "aws_lambda_function_url" "api" {
  count              = local.live ? 1 : 0
  function_name      = aws_lambda_function.api[0].function_name
  authorization_type = "AWS_IAM"
}

resource "aws_lambda_permission" "edge_url" {
  count                  = local.live ? 1 : 0
  statement_id           = "cloudfront-url"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.api[0].function_name
  principal              = "cloudfront.amazonaws.com"
  source_arn             = aws_cloudfront_distribution.site[0].arn
  function_url_auth_type = "AWS_IAM"
}

resource "aws_lambda_permission" "edge_invoke" {
  count         = local.live ? 1 : 0
  statement_id  = "cloudfront-invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api[0].function_name
  principal     = "cloudfront.amazonaws.com"
  source_arn    = aws_cloudfront_distribution.site[0].arn
}
