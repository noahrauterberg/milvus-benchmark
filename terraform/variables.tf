variable "project_id" {
  description = "Default GCP Project ID"
  type        = string
}

variable "region" {
  description = "Default GCP region for resources - must align with 'zone'"
  type        = string
}

variable "zone" {
  description = "Default GCP zone for resources - must align with 'region'"
  type        = string
}

variable "load_generator_cidr" {
  description = "The IP address range that the load generator should use - defaults to 0.0.0.0/0"
  type        = string
  default     = "0.0.0.0/0"
}
