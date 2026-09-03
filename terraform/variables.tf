variable "do_token" {
  description = "DigitalOcean API token (pass via TF_VAR_do_token, never committed)"
  type        = string
  sensitive   = true
}

variable "project_id" {
  description = "Existing DigitalOcean project to attach the droplet to"
  type        = string
}

variable "region" {
  description = "DigitalOcean region slug"
  type        = string
  default     = "nyc3"
}

variable "droplet_size" {
  description = "DigitalOcean droplet size slug"
  type        = string
  default     = "s-1vcpu-2gb"
}
