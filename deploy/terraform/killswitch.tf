data "archive_file" "killswitch" {
  type        = "zip"
  source_file = "${path.module}/../../synergy/killswitch.py"
  output_path = "${path.module}/.build/killswitch.zip"
}

resource "aws_sns_topic" "killswitch" {
  name = "${var.name}-killswitch"
}

resource "aws_sns_topic_policy" "killswitch" {
  arn = aws_sns_topic.killswitch.arn
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "BudgetsPublish"
      Effect    = "Allow"
      Principal = { Service = "budgets.amazonaws.com" }
      Action    = "sns:Publish"
      Resource  = aws_sns_topic.killswitch.arn
      Condition = {
        StringEquals = { "aws:SourceAccount" = var.account_id }
        ArnLike      = { "aws:SourceArn" = "arn:aws:budgets::${var.account_id}:*" }
      }
    }]
  })
}

resource "aws_cloudwatch_log_group" "killswitch" {
  name              = "/aws/lambda/${var.name}-killswitch"
  retention_in_days = var.log_days
}

resource "aws_iam_role" "killswitch" {
  name = "${var.name}-killswitch"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "lambda.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "killswitch" {
  count = local.live ? 1 : 0
  name  = "${var.name}-killswitch"
  role  = aws_iam_role.killswitch.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "Logs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "${aws_cloudwatch_log_group.killswitch.arn}:*"
      },
      {
        Sid      = "Concurrency"
        Effect   = "Allow"
        Action   = ["lambda:GetFunctionConcurrency", "lambda:PutFunctionConcurrency", "lambda:DeleteFunctionConcurrency"]
        Resource = local.function_arns
      },
      {
        Sid      = "Mappings"
        Effect   = "Allow"
        Action   = ["lambda:GetEventSourceMapping", "lambda:UpdateEventSourceMapping"]
        Resource = [aws_lambda_event_source_mapping.history[0].arn, aws_lambda_event_source_mapping.downloads[0].arn]
      },
      {
        Sid      = "EvaluatePipe"
        Effect   = "Allow"
        Action   = ["pipes:DescribePipe", "pipes:StartPipe", "pipes:StopPipe"]
        Resource = aws_pipes_pipe.evaluate.arn
      },
      {
        Sid      = "SweepRule"
        Effect   = "Allow"
        Action   = ["events:DescribeRule", "events:DisableRule", "events:EnableRule"]
        Resource = aws_cloudwatch_event_rule.sweep.arn
      },
      {
        Sid       = "ListJobTasks"
        Effect    = "Allow"
        Action    = ["ecs:ListTasks"]
        Resource  = "*"
        Condition = { ArnEquals = { "ecs:cluster" = aws_ecs_cluster.jobs.arn } }
      },
      {
        Sid      = "StopJobTasks"
        Effect   = "Allow"
        Action   = ["ecs:StopTask"]
        Resource = "arn:aws:ecs:${var.region}:${var.account_id}:task/${aws_ecs_cluster.jobs.name}/*"
      },
      {
        Sid      = "Summary"
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = aws_sns_topic.alerts.arn
      },
    ]
  })
}

resource "aws_lambda_function" "killswitch" {
  count            = local.live ? 1 : 0
  function_name    = "${var.name}-killswitch"
  role             = aws_iam_role.killswitch.arn
  runtime          = "python3.13"
  handler          = "killswitch.handler"
  filename         = data.archive_file.killswitch.output_path
  source_code_hash = data.archive_file.killswitch.output_base64sha256
  architectures    = ["arm64"]
  memory_size      = 128
  timeout          = 60
  depends_on       = [aws_iam_role_policy.killswitch, aws_cloudwatch_log_group.killswitch]

  environment {
    variables = {
      API_FUNCTION       = aws_lambda_function.api[0].function_name
      HISTORY_FUNCTION   = aws_lambda_function.history[0].function_name
      DOWNLOADS_FUNCTION = aws_lambda_function.downloads[0].function_name
      SWEEP_FUNCTION     = aws_lambda_function.sweep[0].function_name
      HISTORY_MAPPING    = aws_lambda_event_source_mapping.history[0].uuid
      DOWNLOADS_MAPPING  = aws_lambda_event_source_mapping.downloads[0].uuid
      EVALUATE_PIPE      = aws_pipes_pipe.evaluate.name
      SWEEP_RULE         = aws_cloudwatch_event_rule.sweep.name
      JOBS_CLUSTER       = aws_ecs_cluster.jobs.name
      ALERTS_TOPIC       = aws_sns_topic.alerts.arn
      API_CONCURRENCY    = tostring(var.api_concurrency)
      KILLSWITCH_BUDGET  = tostring(var.killswitch_budget)
    }
  }
}

resource "aws_lambda_permission" "killswitch" {
  count         = local.live ? 1 : 0
  statement_id  = "budget-topic"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.killswitch[0].function_name
  principal     = "sns.amazonaws.com"
  source_arn    = aws_sns_topic.killswitch.arn
}

resource "aws_sns_topic_subscription" "killswitch" {
  count      = local.live ? 1 : 0
  topic_arn  = aws_sns_topic.killswitch.arn
  protocol   = "lambda"
  endpoint   = aws_lambda_function.killswitch[0].arn
  depends_on = [aws_lambda_permission.killswitch]
}
