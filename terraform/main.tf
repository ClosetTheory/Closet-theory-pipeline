terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  # Remote state on the existing DigitalOcean Spaces bucket "closet" (sgp1), under its own
  # prefix so it never collides with whatever else lives in that bucket. Credentials come
  # from AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env vars (a Spaces key scoped to this
  # bucket only) -- never hardcoded here.
  backend "s3" {
    bucket                      = "closet"
    key                         = "terraform-state/closet-theory-pipeline.tfstate"
    region                      = "us-east-1" # required by the s3 backend, ignored by Spaces
    endpoints                   = { s3 = "https://sgp1.digitaloceanspaces.com" }
    skip_credentials_validation = true
    skip_metadata_api_check     = true
    skip_region_validation      = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
    use_path_style              = false
  }
}

provider "digitalocean" {
  token = var.do_token
}

resource "digitalocean_ssh_key" "deploy" {
  name       = "closet-theory-ci-deploy"
  public_key = file("${path.module}/ci_deploy_key.pub")
}

resource "digitalocean_droplet" "app" {
  name     = "closet-theory-pipeline"
  image    = "docker-20-04" # DO Marketplace image: Ubuntu + Docker + Compose plugin preinstalled
  region   = var.region
  size     = var.droplet_size
  ssh_keys = [digitalocean_ssh_key.deploy.fingerprint]
}

resource "digitalocean_firewall" "app" {
  name = "closet-theory-pipeline"

  droplet_ids = [digitalocean_droplet.app.id]

  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  inbound_rule {
    protocol         = "tcp"
    port_range       = "80"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  inbound_rule {
    protocol         = "tcp"
    port_range       = "443"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "tcp"
    port_range             = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "udp"
    port_range             = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}

resource "digitalocean_project_resources" "app" {
  project = var.project_id
  resources = [
    digitalocean_droplet.app.urn,
  ]
}
