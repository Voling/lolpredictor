data "archive_file" "signup_gate" {
  type        = "zip"
  source_file = "${path.module}/../signup_gate/handler.py"
  output_path = "${path.module}/.build/signup_gate.zip"
}

resource "aws_cloudwatch_log_group" "signup_gate" {
  name              = "/aws/lambda/${var.name}-signup-gate"
  retention_in_days = var.log_days
}

resource "aws_iam_role" "signup_gate" {
  name = "${var.name}-signup-gate"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "lambda.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "signup_gate" {
  name = "${var.name}-signup-gate"
  role = aws_iam_role.signup_gate.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "Logs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "${aws_cloudwatch_log_group.signup_gate.arn}:*"
      },
      {
        Sid       = "SignupCounters"
        Effect    = "Allow"
        Action    = ["dynamodb:UpdateItem"]
        Resource  = aws_dynamodb_table.accounts.arn
        Condition = { "ForAllValues:StringEquals" = { "dynamodb:LeadingKeys" = ["signups"] } }
      },
    ]
  })
}

resource "aws_lambda_function" "signup_gate" {
  function_name    = "${var.name}-signup-gate"
  role             = aws_iam_role.signup_gate.arn
  runtime          = "python3.13"
  handler          = "handler.handler"
  filename         = data.archive_file.signup_gate.output_path
  source_code_hash = data.archive_file.signup_gate.output_base64sha256
  architectures    = ["arm64"]
  memory_size      = 128
  timeout          = 5
  depends_on       = [aws_iam_role_policy.signup_gate, aws_cloudwatch_log_group.signup_gate]

  environment {
    variables = {
      ACCOUNTS_TABLE = aws_dynamodb_table.accounts.name
      DAILY_SIGNUPS  = tostring(var.daily_signups)
      TOTAL_SIGNUPS  = tostring(var.total_signups)
    }
  }
}

resource "aws_lambda_permission" "signup_gate" {
  statement_id  = "cognito-pre-signup"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.signup_gate.function_name
  principal     = "cognito-idp.amazonaws.com"
  source_arn    = aws_cognito_user_pool.users.arn
}
