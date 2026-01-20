resource "google_compute_disk" "milvus_disk" {
  name = "milvus-disk"
  size = 50
}
resource "google_compute_instance" "milvus_vm" {
  name         = "milvus-instance"
  machine_type = "e2-standard-4"
  tags         = ["milvus"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
    }
  }

  attached_disk {
    source      = google_compute_disk.milvus_disk.id
    device_name = "milvus-data"
  }

  network_interface {
    network = "default"
    access_config {} # no config for auto-config
  }

  # Inline startup script to install and start milvus following the docs
  # Also creates a script to measure vm usage
  metadata_startup_script = <<-EOT
    #!/bin/bash
    set -euo pipefail

    apt-get update
    apt-get upgrade -y
    apt-get install -y docker.io docker-compose
    sudo systemctl enable docker
    sudo systemctl start docker

  touch monitoring.sh
    cat > monitoring.sh <<- EOF
      pid=$(pidof milvus)
      while true; do
        ts=$(date +%s)
        rss=$(grep VmRSS /proc/$pid/status | awk '{print $2}')
        cpu=$(ps -p $pid -o %cpu --no-headers)
        echo "$ts,$rss,$cpu" >> milvus_metrics.csv
        sleep 1
      done
    EOF

    sudo mkdir -p /opt/milvus/
    cd /opt/milvus

    sudo wget https://github.com/milvus-io/milvus/releases/download/v2.3.21/milvus-standalone-docker-compose.yml -O docker-compose.yml
    sudo docker-compose up -d
  EOT
}
