variable "name" {
  type    = string
  default = "lolpredictor"
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
  default     = 3008
  description = "The served tables take about 2.5 GB once loaded"
}

variable "log_days" {
  type    = number
  default = 7
}
