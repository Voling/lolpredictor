resource "aws_s3_bucket" "evaluated" {
  bucket_prefix = "${var.name}-evaluated-"
}

resource "aws_s3_bucket_public_access_block" "evaluated" {
  bucket                  = aws_s3_bucket.evaluated.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "evaluated" {
  bucket = aws_s3_bucket.evaluated.id

  rule {
    id     = "expire-evaluated"
    status = "Enabled"

    filter {}

    expiration {
      days = var.evaluated_days
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

resource "aws_s3_bucket_policy" "evaluated" {
  bucket     = aws_s3_bucket.evaluated.id
  depends_on = [aws_s3_bucket_public_access_block.evaluated]
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "TlsOnly"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.evaluated.arn, "${aws_s3_bucket.evaluated.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}
