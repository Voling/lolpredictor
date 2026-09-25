output "site_url" {
  value = local.live ? "https://${aws_cloudfront_distribution.site[0].domain_name}" : ""
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
