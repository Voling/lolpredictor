locals {
  dead_letter_queues = {
    history   = aws_sqs_queue.history_dead.name
    downloads = aws_sqs_queue.downloads_dead.name
    evaluate  = aws_sqs_queue.evaluate_dead.name
  }
  riot_key_logs = {
    api       = aws_cloudwatch_log_group.api.name
    history   = aws_cloudwatch_log_group.history.name
    downloads = aws_cloudwatch_log_group.downloads.name
    sweep     = aws_cloudwatch_log_group.sweep.name
    evaluator = aws_cloudwatch_log_group.evaluator.name
  }
}

resource "aws_sns_topic" "alerts" {
  name = "${var.name}-alerts"
}

resource "aws_sns_topic_subscription" "alerts" {
  count     = var.alert_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_metric_alarm" "dead_letters" {
  for_each            = local.dead_letter_queues
  alarm_name          = "${var.name}-${each.key}-dead-letters"
  alarm_description   = "A message reached ${each.value}. Fix the cause, then run python -m synergy dead --redrive"
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  dimensions          = { QueueName = each.value }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "evaluate_waiting" {
  alarm_name          = "${var.name}-evaluate-waiting"
  alarm_description   = "An evaluation has waited over 15 minutes for a task, the pipe may be off"
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateAgeOfOldestMessage"
  dimensions          = { QueueName = aws_sqs_queue.evaluate.name }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 900
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_log_metric_filter" "riot_key_rejected" {
  for_each       = local.riot_key_logs
  name           = "${var.name}-${each.key}-riot-key-rejected"
  log_group_name = each.value
  pattern        = "\"riot key rejected\""

  metric_transformation {
    name      = "RiotKeyRejected"
    namespace = var.name
    value     = "1"
  }
}

resource "aws_cloudwatch_metric_alarm" "riot_key_rejected" {
  alarm_name          = "${var.name}-riot-key-rejected"
  alarm_description   = "Riot answered 401 or 403, so every Riot call is on hold. Put a working key in SSM"
  namespace           = var.name
  metric_name         = "RiotKeyRejected"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  depends_on          = [aws_cloudwatch_log_metric_filter.riot_key_rejected]
}
