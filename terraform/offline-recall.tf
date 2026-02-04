resource "google_compute_instance" "offlince_recall_instance" {
  count                    = var.deploy_offline_recall_instance ? 1 : 0

  name                      = "offline-recall-instance"
  machine_type              = "n4-custom-24-49152" # high compute for recall computation

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
      type  = "hyperdisk-balanced"
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
}

