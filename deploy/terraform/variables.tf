variable "name" {
  type    = string
  default = "lolpredictor"
}

variable "account_id" {
  type        = string
  default     = "938224354854"
  description = "The only AWS account this configuration may change"
}

variable "profile" {
  type        = string
  default     = null
  description = "AWS CLI profile, null to use the default credentials"
}

variable "region" {
  type    = string
  default = "us-west-2"
}

variable "image_uri" {
  type        = string
  default     = ""
  description = "API image in the ECR repository by digest; empty only while the repository is first created"
}

variable "memory_mb" {
  type        = number
  default     = 4096
  description = "The served tables reach about 2.8 GB once loaded, and more memory also buys CPU for cold starts"
}

variable "api_concurrency" {
  type        = number
  default     = 10
  description = "Most copies of the API that may run at once, which caps the bill during a flood"
}

variable "daily_duos" {
  type    = number
  default = 20
}

variable "max_friends" {
  type    = number
  default = 10
}

variable "contributed_days" {
  type        = number
  default     = 30
  description = "Days a contributed game stays in S3 before it expires unimported"
}

variable "api_rate_limit" {
  type        = number
  default     = 300
  description = "API requests one IP may make in 5 minutes before the WAF blocks it"
}

variable "site_rate_limit" {
  type        = number
  default     = 2000
  description = "Requests of any kind one IP may make in 5 minutes before the WAF blocks it"
}

variable "domain_name" {
  type        = string
  default     = "lolpredictor.com"
  description = "Custom domain for the site, empty to serve on the CloudFront address"
}

variable "zone_name" {
  type        = string
  default     = "lolpredictor.com"
  description = "Route 53 hosted zone that holds domain_name"
}

variable "log_days" {
  type    = number
  default = 7
}

variable "budget_email" {
  type        = string
  default     = ""
  description = "Email for the monthly cost alert, empty for no alert"
}

variable "monthly_budget" {
  type    = number
  default = 25
}

variable "signup_rate_limit" {
  type        = number
  default     = 10
  description = "Sign ups, password resets and code resends one IP may start in 10 minutes, 10 is the lowest the WAF allows"
}

variable "auth_rate_limit" {
  type        = number
  default     = 30
  description = "Sign in and other form posts one IP may send to Cognito in 10 minutes"
}

variable "daily_signups" {
  type        = number
  default     = 10
  description = "New accounts allowed each UTC day across everyone"
}

variable "total_signups" {
  type        = number
  default     = 100
  description = "New accounts allowed in total until this is raised"
}

variable "evaluated_days" {
  type        = number
  default     = 14
  description = "Days a player's pulled games and readings stay in the evaluated store"
}

variable "evaluated_cap_gb" {
  type        = number
  default     = 100
  description = "Size of the evaluated store at which new pulls are refused"
}

variable "daily_evaluations" {
  type        = number
  default     = 20
  description = "New players pulled from Riot each UTC day across everyone"
}

variable "user_evaluations" {
  type        = number
  default     = 3
  description = "New players one account may ask for each UTC day"
}

variable "evaluator_cpu" {
  type        = number
  default     = 1024
  description = "Fargate CPU units for one evaluation task"
}

variable "evaluator_memory" {
  type        = number
  default     = 3072
  description = "Fargate memory in MB for one evaluation task"
}

variable "evaluator_spot" {
  type        = bool
  default     = true
  description = "Run evaluations on Fargate Spot; an interrupted task queues itself again"
}

variable "riot_budget" {
  type        = number
  default     = 60
  description = "Riot API calls a minute shared by the API and the evaluator"
}

variable "github_repository" {
  type    = string
  default = "Voling/lolpredictor"
}

variable "github_branch" {
  type    = string
  default = "main"
}
