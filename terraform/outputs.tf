output "droplet_ip" {
  description = "Public IPv4 address of the app droplet"
  value       = digitalocean_droplet.app.ipv4_address
}
