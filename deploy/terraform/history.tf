locals {
  job_seconds    = 30
  job_visibility = 6 * local.job_seconds
  jobs_command   = ["python", "-m", "uvicorn", "synergy.api.jobs:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
  jobs_environment = {
    ACCOUNTS_TABLE             = aws_dynamodb_table.accounts.name
    RIOT_KEY_PARAMETER         = local.riot_key_parameter
    RIOT_LIMITS                = var.riot_limits
    EVALUATED_STORE            = "s3://${aws_s3_bucket.evaluated.bucket}"
    EVALUATED_CAP_GB           = tostring(var.evaluated_cap_gb)
    EVALUATE_QUEUE             = aws_sqs_queue.evaluate.url
    DOWNLOADS_QUEUE            = aws_sqs_queue.downloads.url
    HISTORY_QUEUE              = aws_sqs_queue.history.url
    AWS_LWA_ERROR_STATUS_CODES = "500-599"
  }
}

resource "aws_sqs_queue" "history_dead" {
  name                      = "${var.name}-history-dead"
  message_retention_seconds = 14 * 86400
}

resource "aws_sqs_queue" "history" {
  name                       = "${var.name}-history"
  visibility_timeout_seconds = local.job_visibility
  message_retention_seconds  = 86400
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.history_dead.arn
    maxReceiveCount     = var.job_receives
  })
}

resource "aws_sqs_queue" "downloads_dead" {
  name                      = "${var.name}-downloads-dead"
  message_retention_seconds = 14 * 86400
}

resource "aws_sqs_queue" "downloads" {
  name                       = "${var.name}-downloads"
  visibility_timeout_seconds = local.job_visibility
  message_retention_seconds  = 86400
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.downloads_dead.arn
    maxReceiveCount     = var.job_receives
  })
}

resource "aws_iam_role" "jobs" {
  name = "${var.name}-jobs"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "lambda.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "jobs" {
  name = "${var.name}-jobs"
  role = aws_iam_role.jobs.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "Logs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = ["${aws_cloudwatch_log_group.history.arn}:*", "${aws_cloudwatch_log_group.downloads.arn}:*", "${aws_cloudwatch_log_group.sweep.arn}:*"]
      },
      {
        Sid      = "ReadJobs"
        Effect   = "Allow"
        Action   = ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes", "sqs:ChangeMessageVisibility"]
        Resource = [aws_sqs_queue.history.arn, aws_sqs_queue.downloads.arn]
      },
      {
        Sid      = "QueueJobs"
        Effect   = "Allow"
        Action   = ["sqs:SendMessage"]
        Resource = [aws_sqs_queue.history.arn, aws_sqs_queue.downloads.arn, aws_sqs_queue.evaluate.arn]
      },
      {
        Sid      = "GameCache"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = "${aws_s3_bucket.evaluated.arn}/games/*"
      },
      {
        Sid      = "GameCacheMissesAre404"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.evaluated.arn
      },
      {
        Sid      = "JobsLimitsAndProgress"
        Effect   = "Allow"
        Action   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem", "dynamodb:Query"]
        Resource = aws_dynamodb_table.accounts.arn
      },
      {
        Sid      = "StoreSize"
        Effect   = "Allow"
        Action   = ["cloudwatch:GetMetricStatistics"]
        Resource = "*"
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

resource "aws_cloudwatch_log_group" "history" {
  name              = "/aws/lambda/${var.name}-history"
  retention_in_days = var.log_days
}

resource "aws_cloudwatch_log_group" "downloads" {
  name              = "/aws/lambda/${var.name}-downloads"
  retention_in_days = var.log_days
}

resource "aws_cloudwatch_log_group" "sweep" {
  name              = "/aws/lambda/${var.name}-sweep"
  retention_in_days = var.log_days
}

resource "aws_lambda_function" "history" {
  count         = local.live ? 1 : 0
  function_name = "${var.name}-history"
  role          = aws_iam_role.jobs.arn
  package_type  = "Image"
  image_uri     = var.image_uri
  architectures = ["x86_64"]
  memory_size   = 512
  timeout       = local.job_seconds
  depends_on    = [aws_iam_role_policy.jobs, aws_cloudwatch_log_group.history]

  image_config {
    command = local.jobs_command
  }

  lifecycle {
    ignore_changes = [image_uri]
  }

  environment {
    variables = merge(local.jobs_environment, { AWS_LWA_PASS_THROUGH_PATH = "/history" })
  }
}

resource "aws_lambda_function" "downloads" {
  count         = local.live ? 1 : 0
  function_name = "${var.name}-downloads"
  role          = aws_iam_role.jobs.arn
  package_type  = "Image"
  image_uri     = var.image_uri
  architectures = ["x86_64"]
  memory_size   = 512
  timeout       = local.job_seconds
  depends_on    = [aws_iam_role_policy.jobs, aws_cloudwatch_log_group.downloads]

  image_config {
    command = local.jobs_command
  }

  lifecycle {
    ignore_changes = [image_uri]
  }

  environment {
    variables = merge(local.jobs_environment, { AWS_LWA_PASS_THROUGH_PATH = "/downloads" })
  }
}

resource "aws_lambda_function" "sweep" {
  count         = local.live ? 1 : 0
  function_name = "${var.name}-sweep"
  role          = aws_iam_role.jobs.arn
  package_type  = "Image"
  image_uri     = var.image_uri
  architectures = ["x86_64"]
  memory_size   = 512
  timeout       = local.job_seconds
  depends_on    = [aws_iam_role_policy.jobs, aws_cloudwatch_log_group.sweep]

  image_config {
    command = local.jobs_command
  }

  lifecycle {
    ignore_changes = [image_uri]
  }

  environment {
    variables = merge(local.jobs_environment, { AWS_LWA_PASS_THROUGH_PATH = "/sweep" })
  }
}

resource "aws_cloudwatch_event_rule" "sweep" {
  name                = "${var.name}-sweep"
  description         = "Ask again for evaluations that went quiet, or fail them once their tries are spent"
  schedule_expression = "rate(5 minutes)"
}

resource "aws_cloudwatch_event_target" "sweep" {
  count = local.live ? 1 : 0
  rule  = aws_cloudwatch_event_rule.sweep.name
  arn   = aws_lambda_function.sweep[0].arn
}

resource "aws_lambda_permission" "sweep" {
  count         = local.live ? 1 : 0
  statement_id  = "sweep-schedule"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sweep[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.sweep.arn
}

resource "aws_lambda_event_source_mapping" "history" {
  count                   = local.live ? 1 : 0
  event_source_arn        = aws_sqs_queue.history.arn
  function_name           = aws_lambda_function.history[0].arn
  batch_size              = 1
  function_response_types = ["ReportBatchItemFailures"]

  scaling_config {
    maximum_concurrency = var.job_concurrency
  }
}

resource "aws_lambda_event_source_mapping" "downloads" {
  count                   = local.live ? 1 : 0
  event_source_arn        = aws_sqs_queue.downloads.arn
  function_name           = aws_lambda_function.downloads[0].arn
  batch_size              = 5
  function_response_types = ["ReportBatchItemFailures"]

  scaling_config {
    maximum_concurrency = var.job_concurrency
  }
}
