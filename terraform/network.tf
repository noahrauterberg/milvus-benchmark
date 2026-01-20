resource "google_compute_firewall" "firewall" {
  name    = "allow-milvus"
  network = "default"
  allow {
    protocol = "tcp"
    ports    = ["2380", "9000", "19530"] # etcd, minio, milvus - in that order
  }
  source_ranges = [var.load_generator_cidr]
  target_tags   = ["milvus"]
}
