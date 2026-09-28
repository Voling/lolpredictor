locals {
  caching_optimized      = "658327ea-f89d-4fab-a63d-7e88639e58f6"
  caching_disabled       = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"
  all_viewer_except_host = "b689b0a8-53d0-40ab-baf2-68738e2966ac"
  api_domain             = local.live ? trimsuffix(trimprefix(aws_lambda_function_url.api[0].function_url, "https://"), "/") : ""
  custom_domain          = var.domain_name != ""
  site_host              = local.custom_domain ? var.domain_name : (local.live ? aws_cloudfront_distribution.site[0].domain_name : "")
  site_hosts             = compact([local.custom_domain ? var.domain_name : "", local.live ? aws_cloudfront_distribution.site[0].domain_name : ""])
  auth_origin            = "https://${aws_cognito_user_pool_domain.users.domain}.auth.${var.region}.amazoncognito.com"
  cognito_api            = "https://cognito-idp.${var.region}.amazonaws.com"
  api_methods            = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
}

resource "aws_s3_bucket" "site" {
  bucket_prefix = "${var.name}-site-"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "site" {
  bucket                  = aws_s3_bucket.site.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_cloudfront_origin_access_control" "site" {
  name                              = "${var.name}-site"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_origin_access_control" "api" {
  name                              = "${var.name}-api"
  origin_access_control_origin_type = "lambda"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_function" "pages" {
  name    = "${var.name}-pages"
  runtime = "cloudfront-js-2.0"
  publish = true
  code    = file("${path.module}/pages.js")
}

resource "aws_cloudfront_function" "api_gate" {
  name    = "${var.name}-api-gate"
  runtime = "cloudfront-js-2.0"
  publish = true
  code    = file("${path.module}/api_gate.js")
}

resource "aws_cloudfront_response_headers_policy" "site" {
  name = "${var.name}-security"

  security_headers_config {
    strict_transport_security {
      access_control_max_age_sec = 63072000
      include_subdomains         = true
      preload                    = false
      override                   = true
    }

    content_type_options {
      override = true
    }

    frame_options {
      frame_option = "DENY"
      override     = true
    }

    referrer_policy {
      referrer_policy = "strict-origin-when-cross-origin"
      override        = true
    }

    content_security_policy {
      content_security_policy = join("; ", [
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: https://ddragon.leagueoflegends.com",
        "font-src 'self' data:",
        "connect-src 'self' ${local.auth_origin} ${local.cognito_api}",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "object-src 'none'",
      ])
      override = true
    }
  }
}

data "aws_route53_zone" "site" {
  count = local.custom_domain ? 1 : 0
  name  = var.zone_name
}

resource "aws_acm_certificate" "site" {
  count             = local.custom_domain ? 1 : 0
  provider          = aws.global
  domain_name       = var.domain_name
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_route53_record" "validation" {
  for_each        = local.custom_domain ? { for option in aws_acm_certificate.site[0].domain_validation_options : option.domain_name => option } : {}
  zone_id         = data.aws_route53_zone.site[0].zone_id
  name            = each.value.resource_record_name
  type            = each.value.resource_record_type
  records         = [each.value.resource_record_value]
  ttl             = 300
  allow_overwrite = true
}

resource "aws_acm_certificate_validation" "site" {
  count                   = local.custom_domain ? 1 : 0
  provider                = aws.global
  certificate_arn         = aws_acm_certificate.site[0].arn
  validation_record_fqdns = [for record in aws_route53_record.validation : record.fqdn]
}

resource "aws_cloudfront_distribution" "site" {
  count               = local.live ? 1 : 0
  enabled             = true
  default_root_object = "index.html"
  price_class         = "PriceClass_100"
  http_version        = "http2and3"
  aliases             = local.custom_domain ? [var.domain_name] : []
  web_acl_id          = aws_wafv2_web_acl.site.arn

  origin {
    origin_id                = "site"
    domain_name              = aws_s3_bucket.site.bucket_regional_domain_name
    origin_access_control_id = aws_cloudfront_origin_access_control.site.id
  }

  origin {
    origin_id                = "api"
    domain_name              = local.api_domain
    origin_access_control_id = aws_cloudfront_origin_access_control.api.id
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
      origin_read_timeout    = 60
    }
  }

  default_cache_behavior {
    target_origin_id           = "site"
    viewer_protocol_policy     = "redirect-to-https"
    allowed_methods            = ["GET", "HEAD"]
    cached_methods             = ["GET", "HEAD"]
    cache_policy_id            = local.caching_optimized
    response_headers_policy_id = aws_cloudfront_response_headers_policy.site.id
    compress                   = true
    function_association {
      event_type   = "viewer-request"
      function_arn = aws_cloudfront_function.pages.arn
    }
  }

  ordered_cache_behavior {
    path_pattern               = "/api/*"
    target_origin_id           = "api"
    viewer_protocol_policy     = "redirect-to-https"
    allowed_methods            = local.api_methods
    cached_methods             = ["GET", "HEAD"]
    cache_policy_id            = local.caching_disabled
    origin_request_policy_id   = local.all_viewer_except_host
    response_headers_policy_id = aws_cloudfront_response_headers_policy.site.id
    compress                   = true
    function_association {
      event_type   = "viewer-request"
      function_arn = aws_cloudfront_function.api_gate.arn
    }
  }

  custom_error_response {
    error_code            = 403
    response_code         = 404
    response_page_path    = "/404.html"
    error_caching_min_ttl = 10
  }

  dynamic "custom_error_response" {
    for_each = [400, 404, 500, 502, 503, 504]
    content {
      error_code            = custom_error_response.value
      error_caching_min_ttl = 0
    }
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = !local.custom_domain
    acm_certificate_arn            = local.custom_domain ? aws_acm_certificate_validation.site[0].certificate_arn : null
    ssl_support_method             = local.custom_domain ? "sni-only" : null
    minimum_protocol_version       = local.custom_domain ? "TLSv1.2_2021" : null
  }
}

resource "aws_route53_record" "site" {
  for_each = local.custom_domain && local.live ? toset(["A", "AAAA"]) : toset([])
  zone_id  = data.aws_route53_zone.site[0].zone_id
  name     = var.domain_name
  type     = each.value

  alias {
    name                   = aws_cloudfront_distribution.site[0].domain_name
    zone_id                = aws_cloudfront_distribution.site[0].hosted_zone_id
    evaluate_target_health = false
  }
}

resource "aws_s3_bucket_policy" "site" {
  count  = local.live ? 1 : 0
  bucket = aws_s3_bucket.site.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "CloudFrontRead"
        Effect    = "Allow"
        Principal = { Service = "cloudfront.amazonaws.com" }
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.site.arn}/*"
        Condition = { StringEquals = { "AWS:SourceArn" = aws_cloudfront_distribution.site[0].arn } }
      },
      {
        Sid       = "TlsOnly"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource  = [aws_s3_bucket.site.arn, "${aws_s3_bucket.site.arn}/*"]
        Condition = { Bool = { "aws:SecureTransport" = "false" } }
      },
    ]
  })
}
