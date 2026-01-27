resource "google_compute_instance" "milvus_vm" {
  name                      = "milvus-instance"
  machine_type              = "n2-standard-8"
  tags                      = ["milvus"]
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 100
      type  = "pd-ssd"
    }
  }

  scratch_disk {
    interface = "NVME" # Recommended Hard Drive
  }

  network_interface {
    network = "default"
    access_config {} # no config for auto-config
  }

  # Inline startup script to install and start milvus following the docs,
  # using local SSD for data storage
  # Also creates a script to measure vm usage
  metadata_startup_script = <<-EOT
    #!/bin/bash
    set -euo pipefail

    apt-get update
    apt-get upgrade -y
    apt-get install -y docker.io docker-compose
    sudo systemctl enable docker
    sudo systemctl start docker

    # Format and mount local SSD
    SSD_DEVICE=$(find /dev/ | grep google-local-nvme-ssd)
    sudo mkfs.ext4 -F $SSD_DEVICE
    sudo mkdir -p /mnt/disks/milvus-data
    sudo mount $SSD_DEVICE /mnt/disks/milvus-data
    sudo chmod a+w /mnt/disks/milvus-data

    # Create monitoring script using docker stats for container metrics
    # Note that the memory usage does not match the unit used by ps in the load generator
    cat > monitoring.sh <<-'EOF'
      #!/bin/bash
      while true; do
        ts=$(date +%s)
        stats=$(docker stats --no-stream --format "{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}" milvus-standalone)
        cpu=$(echo "$stats" | cut -d',' -f1)
        mem_used=$(echo "$stats" | cut -d',' -f2 | cut -d'/' -f1 | xargs)
        mem_perc=$(echo "$stats" | cut -d',' -f3)
        echo "$ts,$cpu,$mem_used,$mem_perc" >> milvus-metrics.csv
        sleep 1
      done
    EOF
    chmod +x monitoring.sh
    echo "timestamp,cpu,mem_used,mem_perc" > milvus-metrics.csv

    # Setup Milvus with data on Local SSD
    sudo mkdir -p /mnt/disks/milvus-data/milvus
    cd /mnt/disks/milvus-data/milvus

    wget https://github.com/milvus-io/milvus/releases/download/v2.6.9/milvus-standalone-docker-compose.yml -O docker-compose.yml
    sudo docker-compose up -d
  EOT
}
