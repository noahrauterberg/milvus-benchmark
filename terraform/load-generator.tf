resource "google_compute_disk" "load_generator_disk" {
  name = "load-generator-disk"
  size = 50
  type = "hyperdisk-balanced"
}
resource "google_compute_instance" "load_generator_vm" {
  name         = "load-generator"
  machine_type = "n4-custom-20-40960" # this is necessary for recall calculations
  tags         = ["load-generator"]
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
      type = "hyperdisk-balanced"
    }
  }

  attached_disk {
    source      = google_compute_disk.load_generator_disk.id
    device_name = "load-generator-data"
  }

  network_interface {
    network = "default"
    access_config {} # no config for auto-config
  }

  # Inline startup script to install go and the milvus sdk
  metadata_startup_script = <<-EOT
    #!/bin/bash
    set -euo pipefail

    # Install dependencies
    apt-get update
    sudo apt-get upgrade -y
    sudo add-apt-repository -y ppa:longsleep/golang-backports
    sudo apt-get install -y golang-go unzip curl git

    mkdir -p /opt/benchmark/glove
    cd /opt/benchmark

    touch monitoring.sh
    cat > monitoring.sh <<-'EOF'
      #!/bin/bash
      pid=$(pidof benchmark)
      while true; do
        ts=$(date +%s)
        rss=$(grep VmRSS /proc/$pid/status | awk '{print $2}')
        cpu=$(ps -p $pid -o %cpu --no-headers)
        echo "$ts,$rss,$cpu" >> benchmark_metrics.csv
        sleep 1
      done
    EOF
    chmod +x monitoring.sh
    echo "timestamp,rss,cpu" > benchmark_metrics.csv

    # Since we can't set env variables across sessions, we write the ip to file
    echo "export MILVUS_IP=${google_compute_instance.milvus_vm.network_interface[0].network_ip}" > env.sh

    # Download GloVe
    GLOVE_DIR="/opt/benchmark/glove"
    mkdir -p $GLOVE_DIR

    # 200 dim takes really long, so download manually if needed
    for dim in 100; do
      GLOVE_URL="https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.$${dim}d.zip"
      GLOVE_ZIP_FILE="glove-$${dim}.zip"
      curl -L $GLOVE_URL -o $GLOVE_ZIP_FILE
      unzip $GLOVE_ZIP_FILE -d $GLOVE_DIR
      mv $GLOVE_DIR/*combined.txt $GLOVE_DIR/glove-$${dim}.txt
      rm $GLOVE_ZIP_FILE
    done

    # Clone and build the load generator
    git clone https://github.com/noahrauterberg/milvus-benchmark.git /opt/benchmark/repo
    cd /opt/benchmark/repo/load-generator
    go build -o /opt/benchmark/benchmark ./src
  EOT

  depends_on = [google_compute_instance.milvus_vm]
}

