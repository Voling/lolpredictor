locals {
  live = var.image_uri != ""
}

resource "aws_ecr_repository" "api" {
  name                 = "${var.name}-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
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

resource "aws_iam_role_policy_attachment" "api_logs" {
  role       = aws_iam_role.api.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${var.name}-api"
  retention_in_days = var.log_days
}

resource "aws_lambda_function" "api" {
  count         = local.live ? 1 : 0
  function_name = "${var.name}-api"
  role          = aws_iam_role.api.arn
  package_type  = "Image"
  image_uri     = var.image_uri
  architectures = ["x86_64"]
  memory_size   = var.memory_mb
  timeout       = 60
  depends_on    = [aws_iam_role_policy_attachment.api_logs, aws_cloudwatch_log_group.api]
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
