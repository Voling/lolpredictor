output "site_url" {
  value = local.live ? "https://${local.site_host}" : ""
}

output "distribution_id" {
  value = local.live ? aws_cloudfront_distribution.site[0].id : ""
}

output "bucket" {
  value = aws_s3_bucket.site.bucket
}

output "repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "cognito_domain" {
  value = local.auth_origin
}

output "cognito_client_id" {
  value = local.live ? aws_cognito_user_pool_client.site[0].id : ""
}

output "contributed_store" {
  value = "s3://${aws_s3_bucket.contributed.bucket}"
}

output "riot_key_parameter" {
  value = local.riot_key_parameter
}

output "models_bucket" {
  value = aws_s3_bucket.models.bucket
}

output "github_backend_role" {
  value = aws_iam_role.github_backend.arn
}

output "github_frontend_role" {
  value = aws_iam_role.github_frontend.arn
}

output "function_name" {
  value = "${var.name}-api"
}
