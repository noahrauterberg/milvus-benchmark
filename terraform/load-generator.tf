resource "google_compute_disk" "load_generator_disk" {
  name = "load-generator-disk"
  size = 100
}
resource "google_compute_instance" "load_generator_vm" {
  name         = "load-generator"
  machine_type = "e2-standard-2"
  tags         = ["load-generator"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
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

    # Since we can't set env variables across sessions, we write the ip to file
    echo "export MILVUS_IP=${google_compute_instance.milvus_vm.network_interface[0].network_ip}" > env.sh

    # Download GloVe
    GLOVE_DIR="/opt/benchmark/glove"
    mkdir -p $GLOVE_DIR

    for dim in 50 100 200; do
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

