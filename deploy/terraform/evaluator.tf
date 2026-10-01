data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

resource "aws_sqs_queue" "evaluate_dead" {
  name                      = "${var.name}-evaluate-dead"
  message_retention_seconds = 14 * 86400
}

resource "aws_sqs_queue" "evaluate" {
  name                       = "${var.name}-evaluate"
  visibility_timeout_seconds = 120
  message_retention_seconds  = 86400
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.evaluate_dead.arn
    maxReceiveCount     = 3
  })
}

resource "aws_ecr_repository" "evaluator" {
  name                 = "${var.name}-evaluator"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "evaluator" {
  repository = aws_ecr_repository.evaluator.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the last five images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 5 }
      action       = { type = "expire" }
    }]
  })
}

resource "aws_ecs_cluster" "jobs" {
  name = var.name
}

resource "aws_ecs_cluster_capacity_providers" "jobs" {
  cluster_name       = aws_ecs_cluster.jobs.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 1
  }
}

resource "aws_cloudwatch_log_group" "evaluator" {
  name              = "/ecs/${var.name}-evaluator"
  retention_in_days = 14
}

resource "aws_security_group" "evaluator" {
  name        = "${var.name}-evaluator"
  description = "Outbound only, for Riot, S3 and DynamoDB"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_iam_role" "evaluator_execution" {
  name = "${var.name}-evaluator-execution"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy_attachment" "evaluator_execution" {
  role       = aws_iam_role.evaluator_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "evaluator" {
  name = "${var.name}-evaluator"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "evaluator" {
  name = "${var.name}-evaluator"
  role = aws_iam_role.evaluator.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "EvaluatedGames"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = "${aws_s3_bucket.evaluated.arn}/*"
      },
      {
        Sid      = "EvaluatedGamesList"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.evaluated.arn
      },
      {
        Sid      = "ServedModelRead"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.models.arn}/*"
      },
      {
        Sid      = "ServedModelList"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.models.arn
      },
      {
        Sid      = "Progress"
        Effect   = "Allow"
        Action   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:Query"]
        Resource = aws_dynamodb_table.accounts.arn
      },
      {
        Sid      = "RequeueOnInterruption"
        Effect   = "Allow"
        Action   = ["sqs:SendMessage"]
        Resource = aws_sqs_queue.evaluate.arn
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

resource "aws_ecs_task_definition" "evaluator" {
  family                   = "${var.name}-evaluator"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.evaluator_cpu
  memory                   = var.evaluator_memory
  execution_role_arn       = aws_iam_role.evaluator_execution.arn
  task_role_arn            = aws_iam_role.evaluator.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  ephemeral_storage {
    size_in_gib = 21
  }

  container_definitions = jsonencode([
    {
      name        = "postgres"
      image       = "public.ecr.aws/docker/library/postgres:17"
      essential   = false
      stopTimeout = 2
      environment = [
        { name = "POSTGRES_USER", value = "synergy" },
        { name = "POSTGRES_PASSWORD", value = "synergy" },
        { name = "POSTGRES_DB", value = "synergy" },
      ]
      healthCheck = {
        command     = ["CMD-SHELL", "pg_isready -U synergy"]
        interval    = 5
        timeout     = 3
        retries     = 10
        startPeriod = 10
      }
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.evaluator.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "postgres"
        }
      }
    },
    {
      name      = "evaluator"
      image     = "${aws_ecr_repository.evaluator.repository_url}:latest"
      essential = true
      dependsOn = [{ containerName = "postgres", condition = "HEALTHY" }]
      environment = [
        { name = "DATABASE_URL", value = "postgresql://synergy:synergy@127.0.0.1:5432/synergy" },
        { name = "MODEL_STORE", value = "s3://${aws_s3_bucket.models.bucket}" },
        { name = "EVALUATED_STORE", value = "s3://${aws_s3_bucket.evaluated.bucket}" },
        { name = "EVALUATED_CAP_GB", value = tostring(var.evaluated_cap_gb) },
        { name = "ACCOUNTS_TABLE", value = aws_dynamodb_table.accounts.name },
        { name = "RIOT_KEY_PARAMETER", value = local.riot_key_parameter },
        { name = "RIOT_BUDGET", value = tostring(var.riot_budget) },
        { name = "EVALUATE_QUEUE", value = aws_sqs_queue.evaluate.url },
        { name = "AWS_DEFAULT_REGION", value = var.region },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.evaluator.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "evaluator"
        }
      }
    },
  ])
}

resource "aws_iam_role" "evaluate_pipe" {
  name = "${var.name}-evaluate-pipe"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "pipes.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "evaluate_pipe" {
  name = "${var.name}-evaluate-pipe"
  role = aws_iam_role.evaluate_pipe.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"]
        Resource = aws_sqs_queue.evaluate.arn
      },
      {
        Effect   = "Allow"
        Action   = ["ecs:RunTask"]
        Resource = [aws_ecs_task_definition.evaluator.arn_without_revision, "${aws_ecs_task_definition.evaluator.arn_without_revision}:*"]
      },
      {
        Effect   = "Allow"
        Action   = ["iam:PassRole"]
        Resource = [aws_iam_role.evaluator.arn, aws_iam_role.evaluator_execution.arn]
      },
    ]
  })
}

resource "aws_cloudwatch_log_group" "evaluate_pipe" {
  name              = "/aws/pipes/${var.name}-evaluate"
  retention_in_days = 14
}

resource "aws_pipes_pipe" "evaluate" {
  name     = "${var.name}-evaluate"
  role_arn = aws_iam_role.evaluate_pipe.arn
  source   = aws_sqs_queue.evaluate.arn
  target   = aws_ecs_cluster.jobs.arn

  log_configuration {
    level                  = "ERROR"
    include_execution_data = ["ALL"]

    cloudwatch_logs_log_destination {
      log_group_arn = aws_cloudwatch_log_group.evaluate_pipe.arn
    }
  }

  source_parameters {
    sqs_queue_parameters {
      batch_size = 1
    }
  }

  target_parameters {
    ecs_task_parameters {
      task_definition_arn = aws_ecs_task_definition.evaluator.arn_without_revision
      task_count          = 1

      capacity_provider_strategy {
        capacity_provider = var.evaluator_spot ? "FARGATE_SPOT" : "FARGATE"
        weight            = 1
      }

      network_configuration {
        aws_vpc_configuration {
          subnets          = data.aws_subnets.default.ids
          security_groups  = [aws_security_group.evaluator.id]
          assign_public_ip = "ENABLED"
        }
      }

      overrides {
        container_override {
          name               = "evaluator"
          cpu                = var.evaluator_cpu - 256
          memory             = var.evaluator_memory - 1024
          memory_reservation = var.evaluator_memory - 2048
          environment {
            name  = "EVALUATE_PUUID"
            value = "$.body.puuid"
          }
        }
      }
    }
  }
}
