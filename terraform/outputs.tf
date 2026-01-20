output "milvus_internal_ip" {
  description = "Internal IP address of the Milvus instance"
  value       = google_compute_instance.milvus_vm.network_interface[0].network_ip
}

output "load_generator_internal_ip" {
  description = "Internal IP address of the load generator instance"
  value       = google_compute_instance.load_generator_vm.network_interface[0].network_ip
}
