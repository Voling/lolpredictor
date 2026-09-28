resource "aws_s3_bucket" "contributed" {
  bucket_prefix = "${var.name}-contributed-"
}

resource "aws_s3_bucket_public_access_block" "contributed" {
  bucket                  = aws_s3_bucket.contributed.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "contributed" {
  bucket = aws_s3_bucket.contributed.id

  rule {
    id     = "expire-unimported"
    status = "Enabled"

    filter {}

    expiration {
      days = var.contributed_days
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

resource "aws_s3_bucket_policy" "contributed" {
  bucket     = aws_s3_bucket.contributed.id
  depends_on = [aws_s3_bucket_public_access_block.contributed]
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "TlsOnly"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.contributed.arn, "${aws_s3_bucket.contributed.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}

resource "aws_dynamodb_table" "accounts" {
  name                        = "${var.name}-accounts"
  billing_mode                = "PAY_PER_REQUEST"
  hash_key                    = "pk"
  range_key                   = "sk"
  deletion_protection_enabled = true

  attribute {
    name = "pk"
    type = "S"
  }

  attribute {
    name = "sk"
    type = "S"
  }

  ttl {
    attribute_name = "expires"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = true
  }
}
